"""
benchmark.py - Evaluation RAG vs baseline par LLM-as-a-judge.

Pour chaque question de `evaluation/questions.xlsx` (40 questions de droit
locatif, avec reponse de reference et grille de notation) et chaque modele,
trois modes sont compares :

  - baseline   : le modele repond seul, sans contexte ;
  - rag        : le modele repond a partir des top-k chunks ChromaDB ;
  - rag_recent : idem, en ecartant les sources anterieures a --min-year
                 (defaut 2014, loi ALUR) et les archives ;
  - agent      : (sur demande, --modes agent) le modele interroge lui-meme le
                 serveur MCP (rag.agent) : il peut decouper la question et
                 lancer plusieurs recherches avant de repondre.

Chaque reponse est notee par un juge LLM (temperature 0, sortie JSON) selon
la grille de la question, avec une penalite de -1 point par hallucination
detectee (loi, chiffre ou date inventes).

Les resultats sont ecrits au fil de l'eau dans evaluation/results/records.jsonl.
Chaque reponse est sauvegardee avant d'etre notee, et un modele (ou le juge)
dont le quota est epuise est mis de cote : relancer la commande reprend la ou
elle s'etait arretee (utile avec les quotas journaliers des offres gratuites).
Sorties :
  - records.jsonl : reponses, sources, latences, notes et justifications
  - summary.csv   : score (% du max), hallucinations, latence par modele et mode

Utilisation :

    python -m evaluation.benchmark
    python -m evaluation.benchmark --models gpt-oss-120b --limit 5
    python -m evaluation.benchmark --modes baseline rag_recent --judge gemma-4-31b
    python -m evaluation.benchmark --no-judge     # generer d'abord, noter plus tard
    python -m evaluation.benchmark --modes agent --models gpt-oss-120b --questions Q4 Q14

    # Recalculer summary.csv a partir des resultats publies (aucun appel API)
    python -m evaluation.benchmark --summary-only --judge claude-opus-5-5

    # Resume de l'evaluation de l'agent (13 questions ou le RAG classique perdait)
    python -m evaluation.benchmark --summary-only --judge claude-opus-5-5         --models gpt-oss-120b --modes baseline rag agent --summary-file summary_agent.csv         --questions Q4 Q5 Q6 Q9 Q14 Q17 Q21 Q22 Q25 Q26 Q34 Q38 Q40
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

from rag.ask import chat, generate_answer, retrieve
from rag.config import CHAT_MODELS, DEFAULT_EMBEDDING, DEFAULT_JUDGE_MODEL, EMBEDDING_MODELS

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVAL_DIR / "questions.xlsx"
RESULTS_DIR = EVAL_DIR / "results"
RECORDS_FILE = RESULTS_DIR / "records.jsonl"

MODES = ("baseline", "rag", "rag_recent", "agent")
DEFAULT_MODES = ["baseline", "rag", "rag_recent"]
DEFAULT_MODELS = ["gpt-oss-120b", "qwen3.8-27b", "gpt-oss-20b"]

SYSTEM_BASELINE = (
    "Tu es un assistant juridique spécialisé en droit immobilier français. "
    "Réponds de manière précise et concise. "
    "Si tu ne sais pas, dis-le clairement."
)

JUDGE_PROMPT = """\
Tu es un expert en droit immobilier français chargé d'évaluer un assistant IA.
Attribue une note PRÉCISE en suivant STRICTEMENT la grille de scoring fournie.

Question              : {question}
Réponse de Référence  : {reference}
Grille de Scoring     : {grid}
Note Maximale         : {max_score}

Réponse de l'Assistant IA à évaluer :
{answer}

Instructions :
1. Analyse la réponse par rapport à chaque critère de la grille.
2. Attribue les points pour chaque élément correctement mentionné.
3. Hallucination : si l'IA invente une loi, un chiffre ou une date inexistants,
   pénalise de -1 point (minimum 0).
4. Réponds UNIQUEMENT en JSON valide avec les clés :
   - "score_obtenu"           : entier ou décimal
   - "justification"          : string courte (max 150 mots)
   - "hallucination_detectee" : true/false
"""


def timed(fn, *args, **kwargs) -> tuple[str, float, str | None]:
    """Execute fn et renvoie (reponse, latence en s, erreur eventuelle)."""
    t0 = time.perf_counter()
    try:
        return fn(*args, **kwargs), round(time.perf_counter() - t0, 2), None
    except Exception as e:
        return "", round(time.perf_counter() - t0, 2), str(e)


def answer_baseline(question: str, model: str) -> str:
    return chat(model, [
        {"role": "system", "content": SYSTEM_BASELINE},
        {"role": "user", "content": question},
    ], temperature=0.1)


def parse_json(raw: str) -> dict:
    """Extrait l'objet JSON de la reponse du juge (tolere un texte autour)."""
    match = re.search(r"\{.*\}", raw, re.S)
    return json.loads(match.group(0) if match else raw)


def judge(row: pd.Series, answer: str, judge_model: str) -> dict:
    if not answer or len(answer.strip()) < 5:
        return {"score_obtenu": 0, "justification": "Réponse vide ou erreur API.",
                "hallucination_detectee": False}
    prompt = JUDGE_PROMPT.format(
        question=row["question"], reference=row["reponse_reference"],
        grid=row["elements_scoring"], max_score=row["max_score"], answer=answer,
    )
    raw = chat(judge_model, [
        {"role": "system", "content": "Tu es un juge expert. Réponds uniquement en JSON valide."},
        {"role": "user", "content": prompt},
    ], temperature=0, response_format={"type": "json_object"})
    return parse_json(raw)


def is_judged(record: dict, judge_model: str) -> bool:
    """Une reponse notee par un autre juge est re-notee, pour rester comparable."""
    return record["score"] is not None and record["judge"] == judge_model


def is_quota_error(error: str) -> bool:
    return "429" in error and "Request too large" not in error


def load_records() -> dict:
    if not RECORDS_FILE.exists():
        return {}
    records = {}
    for line in RECORDS_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            records[(r["id"], r["model"], r["mode"])] = r
    return records


def save_record(record: dict) -> None:
    with RECORDS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    summary = (
        df.groupby(["model", "mode"])
        .agg(questions=("id", "count"), score=("score", "sum"), max_score=("max_score", "sum"),
             hallucinations=("hallucination", "sum"),
             latency_mean_s=("latency_s", "mean"), words_mean=("words", "mean"))
        .reset_index()
    )
    summary["score_pct"] = (100 * summary["score"] / summary["max_score"]).round(1)
    summary["mode"] = pd.Categorical(summary["mode"], MODES)
    return summary.round(2).sort_values(["model", "mode"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(CHAT_MODELS),
                        help=f"Modeles a evaluer (defaut : {' '.join(DEFAULT_MODELS)})")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=MODES,
                        help=f"Modes a evaluer (defaut : {' '.join(DEFAULT_MODES)})")
    parser.add_argument("--judge", default=DEFAULT_JUDGE_MODEL,
                        help=f"Modele juge, parmi les modeles de rag/config.py (defaut : {DEFAULT_JUDGE_MODEL})")
    parser.add_argument("--embedding", default=DEFAULT_EMBEDDING, choices=list(EMBEDDING_MODELS))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-year", type=int, default=2014,
                        help="Annee minimale des sources en mode rag_recent (defaut : 2014)")
    parser.add_argument("--limit", type=int, default=None,
                        help="N'evaluer que les N premieres questions")
    parser.add_argument("--questions", nargs="+", default=None, metavar="ID",
                        help="N'evaluer que ces questions (ex : Q4 Q14)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Genere les reponses sans les noter (notation lors d'un prochain lancement)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Recalcule le resume a partir de records.jsonl, sans appel API")
    parser.add_argument("--summary-file", default="summary.csv",
                        help="Nom du fichier de resume dans evaluation/results/ (defaut : summary.csv)")
    args = parser.parse_args()
    if not args.summary_only and args.judge not in CHAT_MODELS:
        parser.error(f"--judge doit etre l'un de : {', '.join(CHAT_MODELS)}")

    questions = pd.read_excel(QUESTIONS_FILE)
    if args.questions:
        questions = questions[questions["id"].isin(args.questions)]
    if args.limit:
        questions = questions.head(args.limit)
    RESULTS_DIR.mkdir(exist_ok=True)
    done = load_records()
    # Modeles (ou juge) dont le quota est epuise : ignores jusqu'au prochain lancement.
    exhausted: set[str] = set()

    for _, row in ([] if args.summary_only else questions.iterrows()):
        pairs = [(m, mode) for m in args.models for mode in args.modes]
        to_answer = [(m, mode) for m, mode in pairs
                     if (row["id"], m, mode) not in done and m not in exhausted]
        to_judge = [(m, mode) for m, mode in pairs
                    if (row["id"], m, mode) in done or (m, mode) in to_answer]
        if not to_answer and (args.no_judge or all(
                is_judged(done[(row["id"], m, mode)], args.judge) for m, mode in to_judge)):
            continue
        print(f"\n[{row['id']}] {row['question'][:90]}")

        contexts = {
            "rag": lambda: retrieve(row["question"], args.top_k, args.embedding),
            "rag_recent": lambda: retrieve(row["question"], args.top_k, args.embedding, args.min_year),
        }
        modes_needed = {mode for _, mode in to_answer if mode in contexts}
        chunks = {mode: contexts[mode]() for mode in modes_needed}

        # 1. Generation : chaque reponse est sauvegardee avant d'etre notee.
        for model, mode in to_answer:
            if model in exhausted:
                continue
            agent_result: dict = {}
            if mode == "baseline":
                answer, latency, error = timed(answer_baseline, row["question"], model)
            elif mode == "agent":
                from rag.agent import run_agent  # import tardif : dependance MCP
                agent_result, latency, error = timed(run_agent, row["question"], model)
                answer = agent_result["answer"] if agent_result else ""
            else:
                answer, latency, error = timed(generate_answer, row["question"], chunks[mode], model)
            if error:
                print(f"  {model:<18} {mode:<11} erreur : {error[:100]}")
                if is_quota_error(error):
                    exhausted.add(model)
                    print(f"  -> quota atteint pour {model}, reprise au prochain lancement")
                continue
            record = {
                "id": row["id"], "model": model, "mode": mode,
                "answer": answer, "latency_s": latency, "words": len(answer.split()),
                "max_score": float(row["max_score"]),
                "sources": agent_result.get("sources") if mode == "agent" else [
                    {"title": c["metadata"].get("article_title", ""),
                     "url": c["metadata"].get("article_url", ""),
                     "year": c["metadata"].get("year", 0),
                     "similarity": c["similarity"]} for c in chunks.get(mode, [])],
                **({"tool_calls": agent_result.get("tool_calls", [])} if mode == "agent" else {}),
                "judge": None, "score": None, "hallucination": None, "justification": None,
            }
            save_record(record)
            done[(row["id"], model, mode)] = record

        # 2. Notation des reponses pas encore jugees.
        for model, mode in ([] if args.no_judge else pairs):
            record = done.get((row["id"], model, mode))
            if record is None or is_judged(record, args.judge) or args.judge in exhausted:
                continue
            verdict, _, error = timed(judge, row, record["answer"], args.judge)
            if error:
                print(f"  {model:<18} {mode:<11} erreur juge : {error[:100]}")
                if is_quota_error(error):
                    exhausted.add(args.judge)
                    print(f"  -> quota du juge atteint, notation reprise au prochain lancement")
                continue
            record.update({
                "judge": args.judge,
                "score": max(0.0, min(float(verdict.get("score_obtenu", 0)), record["max_score"])),
                "hallucination": bool(verdict.get("hallucination_detectee", False)),
                "justification": verdict.get("justification", ""),
            })
            save_record(record)
            print(f"  {model:<18} {mode:<11} {record['score']:g}/{record['max_score']:g}")

    records = [r for r in done.values()
               if r["model"] in args.models and r["mode"] in args.modes
               and r["id"] in set(questions["id"]) and is_judged(r, args.judge)]
    pending = sum(1 for r in done.values() if not is_judged(r, args.judge))
    if pending:
        print(f"\n{pending} reponse(s) en attente de notation.")
    if not records:
        print("\nAucun resultat.")
        return
    summary = summarize(records)
    summary.to_csv(RESULTS_DIR / args.summary_file, index=False, encoding="utf-8-sig")
    print("\n" + summary.to_string(index=False))
    print(f"\nResultats ecrits dans {RESULTS_DIR}")


if __name__ == "__main__":
    main()
