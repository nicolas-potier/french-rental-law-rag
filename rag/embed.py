"""
embed.py - Vectorisation des chunks et indexation dans ChromaDB.

Lit `data/chunks.jsonl` (produit par `rag.chunk`), calcule les embeddings et
les persiste dans une collection ChromaDB locale (distance cosinus), une
collection par modele d'embedding.

Utilisation :

    # Indexer avec le modele local par defaut (multilingual-e5, gratuit)
    python -m rag.embed

    # Indexer avec un modele OpenAI / Azure (ada002 / 3small / 3large)
    python -m rag.embed --embedding 3small

    # Tester sur 500 chunks / apercu sans calcul
    python -m rag.embed --max-chunks 500
    python -m rag.embed --dry-run
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from rag.config import CHUNKS_FILE, DEFAULT_EMBEDDING, EMBEDDING_MODELS, VECTOR_DB_DIR
from rag.embeddings import embed_passages

# Les chunks sont vectorises et inseres par tranches : la progression est
# visible et un arret en cours de route conserve les tranches deja indexees.
SLICE_SIZE = 1024


@dataclass
class Chunk:
    """Un chunk lu depuis le .jsonl, pret a etre embedde."""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


# Lecture du JSONL

def load_chunks_jsonl(path: Path) -> List[Chunk]:
    """Lit un fichier .jsonl ou chaque ligne est {"id", "text", "metadata"}."""
    chunks: List[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSON invalide ligne {line_num} de {path}: {e}"
                ) from e

            chunk_id = obj.get("id") or obj.get("chunk_id")
            text = obj.get("text") or obj.get("document")
            metadata = obj.get("metadata") or {}
            if not chunk_id or not text:
                raise ValueError(
                    f"Ligne {line_num} : champ 'id' ou 'text' manquant."
                )
            chunks.append(Chunk(id=chunk_id, text=text, metadata=metadata))
    return chunks


# Persistance ChromaDB

def get_or_create_collection(db_dir: Path, collection_name: str, reset: bool):
    """Ouvre (ou cree) la collection ChromaDB persistee dans db_dir."""
    import chromadb
    from chromadb.config import Settings

    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(db_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"Collection '{collection_name}' supprimee (--reset)")
        except Exception:
            pass
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def unique_ids(chunks: List[Chunk]) -> List[str]:
    """Securise les IDs dupliques eventuels du JSONL source."""
    seen: dict = {}
    ids = []
    for i, c in enumerate(chunks):
        base_id = str(c.id)
        if base_id not in seen:
            seen[base_id] = 0
            ids.append(base_id)
        else:
            seen[base_id] += 1
            ids.append(f"{base_id}_{seen[base_id]}_{i}")
    return ids


# Pipeline principal

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("chunks_file", type=Path, nargs="?", default=CHUNKS_FILE,
                        help="Fichier .jsonl des chunks (defaut : data/chunks.jsonl)")
    parser.add_argument("--embedding", default=DEFAULT_EMBEDDING, choices=list(EMBEDDING_MODELS),
                        help=f"Modele d'embedding (defaut : {DEFAULT_EMBEDDING})")
    parser.add_argument("--db-dir", type=Path, default=VECTOR_DB_DIR,
                        help="Dossier ChromaDB (defaut : data/vector_db)")
    parser.add_argument("--reset", action="store_true",
                        help="Supprime la collection avant de re-indexer")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Limite le nombre de chunks traites (utile pour tester)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Ne calcule aucun embedding, affiche un apercu des chunks")
    args = parser.parse_args()

    if not args.chunks_file.is_file():
        raise SystemExit(
            f"ERREUR : fichier introuvable : {args.chunks_file}. "
            "Lancez d'abord `python -m rag.chunk`."
        )

    print(f"Lecture des chunks depuis : {args.chunks_file}")
    chunks = load_chunks_jsonl(args.chunks_file)
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]
    print(f"  {len(chunks)} chunk(s) a indexer.")

    if args.dry_run:
        for ch in chunks[:3]:
            print(f"\nID : {ch.id}")
            print(f"  metadata     : {ch.metadata}")
            print(f"  texte (300c) : {ch.text[:300]}...")
        print(f"\n[dry-run] {len(chunks)} chunk(s) auraient ete indexe(s).")
        return

    model = EMBEDDING_MODELS[args.embedding]
    print(f"\nModele : {model.name}  |  Collection : {model.collection}")
    collection = get_or_create_collection(args.db_dir, model.collection, reset=args.reset)
    ids = unique_ids(chunks)

    for start in range(0, len(chunks), SLICE_SIZE):
        batch = chunks[start : start + SLICE_SIZE]
        collection.upsert(
            ids=ids[start : start + SLICE_SIZE],
            embeddings=embed_passages([c.text for c in batch], args.embedding),
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
        )
        print(f"  Indexe : {start + len(batch)}/{len(chunks)}")

    print(f"\nTermine. Collection '{model.collection}' : {collection.count()} document(s).")


if __name__ == "__main__":
    main()
