"""
embeddings.py - Calcul des embeddings, en local ou via l'API OpenAI / Azure.

Le modele local (multilingual-e5) attend des prefixes differents pour les
documents ("passage: ") et les requetes ("query: ") ; les vecteurs sont
normalises pour la distance cosinus.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import List

from rag.config import EMBEDDING_MODELS, get_client

API_BATCH_SIZE = 96
LOCAL_BATCH_SIZE = 32


@lru_cache(maxsize=None)
def _local_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cpu")


def embed_passages(texts: List[str], alias: str) -> List[List[float]]:
    model = EMBEDDING_MODELS[alias]
    if model.provider == "local":
        vectors = _local_model(model.name).encode(
            [f"passage: {t}" for t in texts],
            batch_size=LOCAL_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return vectors.tolist()
    return _embed_api(texts, model.name)


def embed_query(text: str, alias: str) -> List[float]:
    model = EMBEDDING_MODELS[alias]
    if model.provider == "local":
        return _local_model(model.name).encode(
            f"query: {text}", normalize_embeddings=True
        ).tolist()
    return get_client().embeddings.create(model=model.name, input=[text]).data[0].embedding


def _embed_api(texts: List[str], deployment: str, max_retries: int = 5) -> List[List[float]]:
    """Appelle l'API Embeddings par batches, avec retry exponentiel.

    Sur Azure, `deployment` est le nom du deploiement (identique au nom du
    modele dans ce projet) ; sur OpenAI, c'est le nom du modele.
    """
    client = get_client()
    all_embeddings: List[List[float]] = []
    n_batches = (len(texts) + API_BATCH_SIZE - 1) // API_BATCH_SIZE

    for start in range(0, len(texts), API_BATCH_SIZE):
        batch = texts[start : start + API_BATCH_SIZE]
        attempt = 0
        while True:
            try:
                response = client.embeddings.create(model=deployment, input=batch)
                all_embeddings.extend([d.embedding for d in response.data])
                print(f"  Embedded batch {start // API_BATCH_SIZE + 1}/{n_batches}  (+{len(batch)} chunks)")
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    raise
                wait = 2 ** attempt
                print(f"  Erreur embedding (tentative {attempt}/{max_retries}): {e}")
                print(f"  Retry dans {wait}s...")
                time.sleep(wait)
    return all_embeddings
