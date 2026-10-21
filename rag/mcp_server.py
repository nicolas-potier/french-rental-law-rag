"""
mcp_server.py - Serveur MCP exposant la recherche dans la documentation ANIL.

Expose le retrieval du projet a n'importe quel client MCP (Claude Desktop,
MCP Inspector, rag.agent...) : c'est le client qui decide quand chercher,
combien de fois, et avec quelle requete.

  - tool `search_rental_law` : recherche semantique (top-k extraits) ;
  - tool `get_article`       : texte complet d'un article a partir de son URL ;
  - prompt `grounded_answer` : consignes de reponse sourcee du projet.

Aucune cle API n'est necessaire : la recherche est entierement locale
(embeddings multilingual-e5 + ChromaDB). L'index doit avoir ete construit
au prealable (`python -m rag.chunk` puis `python -m rag.embed`).

Transport stdio : la sortie standard porte le protocole JSON-RPC, donc les
logs passent par la sortie d'erreur.

Utilisation :

    python -m rag.mcp_server            # lance le serveur (stdio)

    # MCP Inspector dans le navigateur (necessite uv ou Node.js ; le paquet
    # `rag` doit etre importable, d'ou PYTHONPATH) : voir docs/mcp.md
    PYTHONPATH=. mcp dev rag/mcp_server.py
"""

from __future__ import annotations

import logging
import re
import sys

from mcp.server.mcpserver import MCPServer

from rag.ask import SYSTEM_PROMPT, open_collection, retrieve
from rag.config import DEFAULT_EMBEDDING, EMBEDDING_MODELS
from rag.embeddings import embed_query

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("anil-mcp")

# Au-dela, un article est tronque pour ne pas saturer le contexte du modele.
MAX_ARTICLE_CHARS = 20_000
_HEADER_RE = re.compile(r"^Titre : .*?\n\n", re.S)

mcp = MCPServer(
    "anil-rental-law",
    instructions=(
        "Search tools over the official documentation of the ANIL (French national "
        "housing information agency) on residential rental law. Answer only from the "
        "returned excerpts and cite their URLs."
    ),
)


def _warm_up() -> None:
    """Charge le modele d'embedding et ouvre l'index une seule fois, au demarrage."""
    embed_query("préchauffage", DEFAULT_EMBEDDING)
    count = open_collection(EMBEDDING_MODELS[DEFAULT_EMBEDDING].collection).count()
    log.info("Index pret : %d extraits", count)


@mcp.tool()
def search_rental_law(query: str, top_k: int = 3, min_year: int | None = None) -> str:
    """Search the official ANIL documentation on French residential rental law
    (leases, deposits, notice periods, rent, charges, repairs, decency...).

    Returns the top_k most relevant excerpts, each with its article title, year
    and URL. Write the query in French. For a multi-part question, call this
    tool once per sub-question with a short, focused query rather than once
    with the whole question. Use min_year only to restrict to recent sources.
    """
    chunks = retrieve(query, top_k=top_k, min_year=min_year)
    if not chunks:
        return "No excerpt found."
    return "\n\n---\n\n".join(
        f"[{c['metadata'].get('article_title', '?')}] ({c['metadata'].get('year') or 'year unknown'})\n"
        f"URL: {c['metadata'].get('article_url', '')}\n"
        f"Similarity: {c['similarity']:.3f}\n\n{c['text']}"
        for c in chunks
    )


@mcp.tool()
def get_article(url: str) -> str:
    """Return the full text of an ANIL article, given the URL of one of its
    excerpts. Use it when an excerpt returned by search_rental_law is cut off
    or lacks the detail needed to answer.
    """
    collection = open_collection(EMBEDDING_MODELS[DEFAULT_EMBEDDING].collection)
    found = collection.get(where={"article_url": url}, include=["documents", "metadatas"])
    if not found["ids"]:
        return f"No article found for URL: {url}"

    parts = sorted(zip(found["metadatas"], found["documents"]), key=lambda p: p[0].get("chunk_index", 0))
    meta = parts[0][0]
    body, seen_sections = [], set()
    for chunk_meta, text in parts:
        section = chunk_meta.get("section_title") or ""
        if section and section not in seen_sections:
            body.append(f"## {section}")
            seen_sections.add(section)
        body.append(_HEADER_RE.sub("", text, count=1))

    article = f"[{meta.get('article_title', '?')}] ({meta.get('year') or 'year unknown'})\nURL: {url}\n\n" + "\n\n".join(body)
    if len(article) > MAX_ARTICLE_CHARS:
        article = article[:MAX_ARTICLE_CHARS] + "\n\n[... article truncated]"
    return article


@mcp.prompt()
def grounded_answer(question: str) -> str:
    """Instructions for a sourced answer to a rental-law question."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Use the search_rental_law tool (one call per sub-question) to gather the "
        "extracts, then answer.\n\n"
        f"Question : {question}"
    )


if __name__ == "__main__":
    _warm_up()
    mcp.run(transport="stdio")
