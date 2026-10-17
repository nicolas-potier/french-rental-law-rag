"""
ask.py - Pose une question au RAG : retrieval ChromaDB + generation LLM.

1. La question est vectorisee avec le meme modele d'embedding que les chunks.
2. Les top-k chunks les plus proches (similarite cosinus) sont recuperes.
3. Le LLM repond en s'appuyant uniquement sur ces extraits et cite les URL
   officielles (ANIL) dont proviennent les informations.

Option `--min-year` : ecarte les articles dates d'avant cette annee et les
pages d'archives (les articles non dates sont conserves). Le corpus ANIL
contient de nombreuses analyses juridiques anterieures a la loi ALUR (2014),
qui decrivent des regles depuis modifiees.

Utilisation :

    python -m rag.ask "Quel est le montant maximal du depot de garantie ?"
    python -m rag.ask "..." --min-year 2014
    python -m rag.ask "..." --model qwen3.8-27b --top-k 5 --full
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from rag.config import (
    CHAT_MODELS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING,
    EMBEDDING_MODELS,
    VECTOR_DB_DIR,
    get_client,
)
from rag.embeddings import embed_query

SYSTEM_PROMPT = (
    "Tu es un assistant juridique spécialisé en droit immobilier français. "
    "Réponds uniquement en te basant sur les extraits fournis. "
    "Si la réponse ne s'y trouve pas, dis-le clairement. "
    "Réponds directement à la question dès la première phrase. "
    "Avant de répondre, repère dans les extraits tous les éléments utiles : "
    "règle principale, conditions, exceptions, délais, montants, justificatifs, "
    "distinctions entre plusieurs cas et conséquences prévues. "
    "Inclue les éléments pertinents lorsqu'ils sont présents dans les extraits, "
    "sans développer inutilement. "
    "Cite explicitement la source utilisée pour chaque information importante, "
    "en indiquant l'URL fournie dans l'extrait. "
    "Si plusieurs extraits ou plusieurs URL soutiennent la réponse, cite-les séparément. "
    "Ne cite pas une URL qui ne soutient pas directement l'affirmation. "
    "Si aucune URL n'est disponible dans les extraits, indique que la source n'est pas précisée. "
    "Si les extraits ne permettent pas de répondre, écris clairement que les extraits "
    "fournis ne permettent pas de répondre à cette question. "
    "Sois précis, concis et factuel."
)


def open_collection(collection_name: str, db_dir: Path = VECTOR_DB_DIR):
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(db_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        return client.get_collection(name=collection_name)
    except Exception:
        raise SystemExit(
            f"ERREUR : collection '{collection_name}' introuvable dans {db_dir}. "
            "Lancez d'abord `python -m rag.chunk` puis `python -m rag.embed`."
        )


def recency_filter(min_year: int) -> dict:
    """Filtre ChromaDB : pas d'archives, et annee >= min_year (ou inconnue)."""
    return {"$and": [
        {"archived": False},
        {"$or": [{"year": {"$gte": min_year}}, {"year": 0}]},
    ]}


def retrieve(
    question: str,
    top_k: int = 3,
    embedding: str = DEFAULT_EMBEDDING,
    min_year: int | None = None,
    db_dir: Path = VECTOR_DB_DIR,
) -> list[dict]:
    """Retourne les top_k chunks : [{"text", "metadata", "similarity"}, ...]."""
    collection = open_collection(EMBEDDING_MODELS[embedding].collection, db_dir)
    results = collection.query(
        query_embeddings=[embed_query(question, embedding)],
        n_results=top_k,
        where=recency_filter(min_year) if min_year else None,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": doc, "metadata": meta, "similarity": round(1 - dist, 4)}
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def chat(model: str, messages: list[dict], temperature: float = 0.05, **kwargs) -> str:
    cfg = CHAT_MODELS[model]
    response = get_client(cfg.provider).chat.completions.create(
        model=cfg.model_id, messages=messages, temperature=temperature,
        **{**cfg.params, **kwargs},
    )
    content = response.choices[0].message.content or ""
    # Certains modeles de raisonnement renvoient leur reflexion entre balises.
    return re.sub(r"<(think|thought)>.*?</\1>", "", content, flags=re.S).strip()


def generate_answer(question: str, chunks: list[dict], model: str = DEFAULT_CHAT_MODEL) -> str:
    context = "\n\n---\n\n".join(
        f"[Extrait {i + 1}]\n{c['text']}" for i, c in enumerate(chunks)
    )
    return chat(model, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question : {question}\n\nExtraits :\n{context}"},
    ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("question", help="La question a poser au RAG")
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL, choices=list(CHAT_MODELS),
                        help=f"Modele de generation (defaut : {DEFAULT_CHAT_MODEL})")
    parser.add_argument("--embedding", default=DEFAULT_EMBEDDING, choices=list(EMBEDDING_MODELS),
                        help=f"Modele d'embedding / collection (defaut : {DEFAULT_EMBEDDING})")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Nombre de chunks recuperes (defaut : 3)")
    parser.add_argument("--min-year", type=int, default=None,
                        help="Ecarte les sources anterieures a cette annee et les archives")
    parser.add_argument("--db-dir", type=Path, default=VECTOR_DB_DIR,
                        help="Dossier ChromaDB (defaut : data/vector_db)")
    parser.add_argument("--full", action="store_true",
                        help="Affiche le texte complet des chunks recuperes")
    args = parser.parse_args()

    chunks = retrieve(args.question, args.top_k, args.embedding, args.min_year, args.db_dir)
    if not chunks:
        print("Aucun resultat trouve.")
        return

    answer = generate_answer(args.question, chunks, model=args.model)

    print("=" * 60)
    print(f"REPONSE ({args.model})")
    print("=" * 60)
    print(answer)
    print()
    print("=" * 60)
    print(f"SOURCES (top {len(chunks)})")
    print("=" * 60)
    for i, c in enumerate(chunks, start=1):
        meta = c["metadata"]
        print(f"\n#{i}  similarite ~ {c['similarity']:.3f}")
        print(f"  article : {meta.get('article_title', '?')} ({meta.get('year') or 'date inconnue'})")
        if meta.get("article_url"):
            print(f"  url     : {meta['article_url']}")
        if meta.get("section_title"):
            print(f"  section : {meta['section_title']}")
        if args.full:
            print(f"\n{c['text']}")


if __name__ == "__main__":
    main()
