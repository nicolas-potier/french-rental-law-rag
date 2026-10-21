"""
agent.py - Agent minimal : un LLM (Groq, tool calling OpenAI) pilote le serveur MCP.

Contrairement a `rag.ask` (une recherche, puis une generation), c'est ici le
modele qui decide quand chercher, combien de fois et avec quelle requete :
il decompose la question en sous-questions, appelle `search_rental_law` pour
chacune, `get_article` si un extrait est tronque, puis redige la reponse
sourcee. Les outils viennent du serveur `rag.mcp_server`, lance en
sous-processus (stdio) : son demarrage prend 10-20 s (chargement du modele
d'embedding), il faut donc le demarrer une seule fois par session.

API :

    run_agent_async(question, model, max_steps, session=None) -> dict
    run_agent(question, model, max_steps) -> dict       # enveloppe synchrone
    run_many_async(questions, model, max_steps) -> list[dict]
    AgentSession                                          # gestionnaire de contexte
    AgentWorker                                           # serveur partage, API synchrone

`AgentSession` demarre un serveur MCP et le reutilise pour plusieurs questions :

    async with AgentSession() as session:
        r1 = await run_agent_async("...", session=session)
        r2 = await run_agent_async("...", session=session)

Chaque resultat est un dict :

    {"answer": str,
     "tool_calls": [{"tool": str, "arguments": dict, "chars": int}, ...],
     "sources": [{"title": str, "url": str, "year": int (0 si inconnue), "similarity": float}, ...],
     "steps": int}   # nombre d'appels au LLM

`sources` liste les articles (URL uniques) renvoyes par `search_rental_law`,
dans l'ordre de premiere apparition.

Utilisation :

    python -m rag.agent "Quelle est la duree minimale d'un bail meuble ?"
    python -m rag.agent "..." --model gpt-oss-20b --max-steps 4 --trace
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import re
import sys
import threading
from concurrent.futures import Future
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import APIStatusError

from rag.ask import SYSTEM_PROMPT
from rag.config import CHAT_MODELS, DEFAULT_CHAT_MODEL, ROOT_DIR, get_client

AGENT_PROMPT = (
    "Tu disposes d'outils de recherche dans la documentation officielle de l'ANIL. "
    "Procède ainsi : décompose la question en sous-questions (ex. : le délai, puis la "
    "sanction en cas de retard) ; appelle search_rental_law une fois par sous-question, "
    "avec une requête courte et ciblée en français (jamais la question entière), sans "
    "utiliser min_year ; une question à plusieurs volets (délai ET conséquence, règle ET exception...) exige "
    "au moins deux appels distincts, émis dans le même tour, avec deux requêtes différentes ; "
    "appelle get_article (avec l'URL d'un extrait) lorsqu'un extrait est coupé ou manque "
    "de détails. Ne réponds jamais de mémoire : appuie-toi uniquement sur les résultats "
    "des outils et cite l'URL complète (https://...) de chaque information importante, écrite en clair "
    "dans le texte, sans renvoi numéroté ni format de citation spécial. Si les outils ne "
    "renvoient rien de pertinent, dis que la documentation ne permet pas de répondre.\n\n"
)
SYSTEM_MESSAGE = AGENT_PROMPT + SYSTEM_PROMPT

SHRUNK_TOOL_CHARS = 6_000
_SEPARATOR = "\n\n---\n\n"
_EXCERPT_RE = re.compile(r"\[(.*)\] \((\d{4}|[^)]*)\)\nURL: (.*)\nSimilarity: (-?[\d.]+)")


class AgentSession:
    """Serveur MCP demarre une fois, reutilisable pour plusieurs questions."""

    async def __aenter__(self) -> AgentSession:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "rag.mcp_server"],
            cwd=str(ROOT_DIR),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.mcp = await self._stack.enter_async_context(ClientSession(read, write))
        await self.mcp.initialize()
        listing = await self.mcp.list_tools()
        self.tools = [
            {"type": "function", "function": {
                "name": t.name, "description": t.description or "", "parameters": t.input_schema}}
            for t in listing.tools
        ]
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()


def parse_sources(tool_output: str) -> list[dict]:
    """Extrait {title, url, year, similarity} de chaque extrait renvoye par search_rental_law."""
    sources = []
    for block in tool_output.split(_SEPARATOR):
        m = _EXCERPT_RE.match(block)
        if m:
            title, year, url, similarity = m.groups()
            sources.append({"title": title, "url": url.strip(),
                            "year": int(year) if year.isdigit() else 0,
                            "similarity": float(similarity)})
    return sources


def _clean(content: str | None) -> str:
    return re.sub(r"<(think|thought)>.*?</\1>", "", content or "", flags=re.S).strip()


async def _complete(model: str, messages: list[dict], tools: list[dict], **kwargs):
    """Appel LLM synchrone execute dans un thread pour ne pas bloquer la boucle asyncio.

    Les offres gratuites limitent la taille d'une requete (tokens par minute) :
    si elle est refusee (413), les resultats d'outils les plus longs sont
    raccourcis a SHRUNK_TOOL_CHARS caracteres et l'appel est retente une fois.
    """
    cfg = CHAT_MODELS[model]

    def call():
        return get_client(cfg.provider).chat.completions.create(
            model=cfg.model_id, messages=messages, tools=tools, temperature=0.05,
            **{**cfg.params, **kwargs},
        )

    try:
        response = await asyncio.to_thread(call)
    except APIStatusError as exc:
        if exc.status_code != 413:
            raise
        for msg in messages:
            if msg["role"] == "tool" and len(msg["content"]) > SHRUNK_TOOL_CHARS:
                msg["content"] = msg["content"][:SHRUNK_TOOL_CHARS] + "\n\n[... resultat raccourci]"
        response = await asyncio.to_thread(call)
    return response.choices[0].message


async def _run(session: AgentSession, question: str, model: str, max_steps: int) -> dict:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": question},
    ]
    tool_calls: list[dict] = []
    sources: dict[str, dict] = {}
    steps = 0

    for round_ in range(max_steps + 1):
        # Apres max_steps tours d'outils, on force une reponse finale.
        final = round_ == max_steps
        msg = await _complete(model, messages, session.tools,
                              **({"tool_choice": "none"} if final else {}))
        steps += 1
        if not msg.tool_calls or final:
            return {"answer": _clean(msg.content), "tool_calls": tool_calls,
                    "sources": list(sources.values()), "steps": steps}

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in msg.tool_calls
            ],
        })
        for call in msg.tool_calls:
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments or "{}")
                result = await session.mcp.call_tool(name, arguments)
                text = "\n".join(getattr(c, "text", "") for c in result.content)
                if result.is_error:
                    text = f"Erreur de l'outil : {text}"
            except Exception as exc:  # le modele peut corriger son appel au tour suivant
                arguments, text = {}, f"Erreur de l'outil : {exc}"
            tool_calls.append({"tool": name, "arguments": arguments, "chars": len(text)})
            if name == "search_rental_law":
                for src in parse_sources(text):
                    sources.setdefault(src["url"], src)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})

    raise AssertionError("unreachable")


async def run_agent_async(question: str, model: str = DEFAULT_CHAT_MODEL,
                          max_steps: int = 6, session: AgentSession | None = None) -> dict:
    """Repond a une question. Sans `session`, demarre (puis arrete) son propre serveur MCP."""
    if session is not None:
        return await _run(session, question, model, max_steps)
    async with AgentSession() as own:
        return await _run(own, question, model, max_steps)


async def run_many_async(questions: list[str], model: str = DEFAULT_CHAT_MODEL,
                         max_steps: int = 6) -> list[dict]:
    """Repond a plusieurs questions (sequentiellement) avec un seul serveur MCP."""
    async with AgentSession() as session:
        return [await _run(session, q, model, max_steps) for q in questions]


def run_agent(question: str, model: str = DEFAULT_CHAT_MODEL, max_steps: int = 6) -> dict:
    """Version synchrone : demarre le serveur, repond a une question, l'arrete."""
    return asyncio.run(run_agent_async(question, model, max_steps))


class AgentWorker:
    """Serveur MCP partage entre plusieurs questions, utilisable depuis du code synchrone.

    La session MCP vit dans un thread dedie (sa propre boucle asyncio) ; `ask`
    lui transmet une question et attend la reponse. Utile pour un benchmark :
    le serveur ne demarre qu'une fois, et la latence mesuree est celle de l'agent.

        with AgentWorker() as worker:
            result = worker.ask("...", model="gpt-oss-120b")
    """

    def __init__(self, max_steps: int = 6) -> None:
        self.max_steps = max_steps
        self._requests: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(target=lambda: asyncio.run(self._serve()), daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._startup_error:
            raise self._startup_error

    async def _serve(self) -> None:
        try:
            async with AgentSession() as session:
                self._ready.set()
                while (item := await asyncio.to_thread(self._requests.get)) is not None:
                    question, model, future = item
                    try:
                        future.set_result(await _run(session, question, model, self.max_steps))
                    except Exception as exc:
                        future.set_exception(exc)
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()

    def ask(self, question: str, model: str = DEFAULT_CHAT_MODEL) -> dict:
        future: Future = Future()
        self._requests.put((question, model, future))
        return future.result()

    def close(self) -> None:
        self._requests.put(None)
        self._thread.join()

    def __enter__(self) -> AgentWorker:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("question", help="La question a poser a l'agent")
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL, choices=list(CHAT_MODELS),
                        help=f"Modele de generation (defaut : {DEFAULT_CHAT_MODEL})")
    parser.add_argument("--max-steps", type=int, default=6,
                        help="Nombre maximal de tours d'appels d'outils (defaut : 6)")
    parser.add_argument("--trace", action="store_true",
                        help="Affiche les appels d'outils faits par l'agent")
    args = parser.parse_args()

    result = run_agent(args.question, args.model, args.max_steps)

    print("=" * 60)
    print(f"REPONSE ({args.model}, {result['steps']} appel(s) LLM)")
    print("=" * 60)
    print(result["answer"])
    if args.trace:
        print()
        print("=" * 60)
        print(f"APPELS D'OUTILS ({len(result['tool_calls'])})")
        print("=" * 60)
        for i, call in enumerate(result["tool_calls"], start=1):
            print(f"#{i}  {call['tool']}({json.dumps(call['arguments'], ensure_ascii=False)})"
                  f"  -> {call['chars']} caracteres")
    print()
    print("=" * 60)
    print(f"SOURCES ({len(result['sources'])})")
    print("=" * 60)
    for i, src in enumerate(result["sources"], start=1):
        print(f"\n#{i}  similarite ~ {src['similarity']:.3f}")
        print(f"  article : {src['title']} ({src['year'] or 'date inconnue'})")
        print(f"  url     : {src['url']}")


if __name__ == "__main__":
    main()
