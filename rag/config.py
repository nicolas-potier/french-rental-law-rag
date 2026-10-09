"""
config.py - Configuration partagee : chemins, modeles et clients LLM.

Les identifiants sont lus depuis les variables d'environnement (ou un fichier
`.env` a la racine du projet, voir `.env.example`).

Embeddings :
  - "e5" (defaut) : modele open source execute en local, gratuit, sans cle ;
  - "ada002" / "3small" / "3large" : OpenAI ou Azure OpenAI (configuration
    d'origine du projet).

Generation (tous via des API compatibles OpenAI) :
  - groq   : GROQ_API_KEY   (offre gratuite, https://console.groq.com)
  - gemini : GEMINI_API_KEY (offre gratuite, https://aistudio.google.com)
  - openai : OPENAI_API_KEY, ou Azure OpenAI si AZURE_OPENAI_ENDPOINT est defini
             (AZURE_OPENAI_API_KEY)
  - azure_ai : modeles Mistral / Grok servis par Azure AI Foundry
             (AZURE_AI_INFERENCE_ENDPOINT + AZURE_OPENAI_API_KEY)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# Sur Windows, la console utilise par defaut cp1252 : on force l'UTF-8 pour
# afficher correctement les accents.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv est optionnel
    pass
else:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
VECTOR_DB_DIR = DATA_DIR / "vector_db"


# ---------------------------------------------------------------------------
# Modeles d'embedding
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingModel:
    name: str        # nom du modele (HuggingFace, OpenAI ou deploiement Azure)
    collection: str  # collection ChromaDB correspondante
    provider: str    # "local" ou "openai"


EMBEDDING_MODELS = {
    "e5": EmbeddingModel("intfloat/multilingual-e5-small", "anil_e5_small", "local"),
    "ada002": EmbeddingModel("text-embedding-ada-002", "anil_qna_ada002", "openai"),
    "3small": EmbeddingModel("text-embedding-3-small", "anil_qna_3small", "openai"),
    "3large": EmbeddingModel("text-embedding-3-large", "anil_qna_3large", "openai"),
}
DEFAULT_EMBEDDING = "e5"


# ---------------------------------------------------------------------------
# Modeles de generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChatModel:
    provider: str
    model_id: str
    params: dict = field(default_factory=dict)  # parametres d'API specifiques


CHAT_MODELS = {
    # Offres gratuites
    # Groq limite les tokens par minute (entree + max_tokens) : on plafonne la sortie.
    "gpt-oss-120b": ChatModel("groq", "openai/gpt-oss-120b", {"reasoning_effort": "low", "max_tokens": 2048}),
    "gpt-oss-20b": ChatModel("groq", "openai/gpt-oss-20b", {"reasoning_effort": "low", "max_tokens": 2048}),
    "qwen3.8-27b": ChatModel("groq", "qwen/qwen3.8-27b", {"max_tokens": 2048}),
    "gemma-4-31b": ChatModel("gemini", "gemma-4-31b-it"),
    "gemini-3.5-flash": ChatModel("gemini", "gemini-3.5-flash"),
    "gemini-3.8-flash": ChatModel("gemini", "gemini-3.8-flash"),
    # Modeles du benchmark d'origine (Azure)
    "gpt-4o-mini": ChatModel("openai", "gpt-4o-mini"),
    "gpt-5.4-nano": ChatModel("openai", "gpt-5.4-nano"),
    "mistral-small-2503": ChatModel("azure_ai", "mistral-small-2503"),
    "grok-4-1-fast-non-reasoning": ChatModel("azure_ai", "grok-4-1-fast-non-reasoning"),
}
DEFAULT_CHAT_MODEL = "gpt-oss-120b"
# Les modeles Gemini Flash sont limites a 20 requetes/jour en offre gratuite ;
# Gemma 4 (meme API) dispose d'un quota bien plus large.
DEFAULT_JUDGE_MODEL = "gemma-4-31b"

# Endpoints compatibles OpenAI des fournisseurs gratuits
OPENAI_COMPATIBLE = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
}

AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_AI_API_VERSION = "2024-05-01-preview"

# Les offres gratuites renvoient des erreurs 429 (quota par minute) ou 503
# (surcharge) : le SDK OpenAI re-essaie automatiquement en respectant l'en-tete
# retry-after. Le timeout evite qu'un appel bloque indefiniment.
MAX_RETRIES = 4
TIMEOUT_S = 90


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def get_client(provider: str = "openai"):
    """Retourne un client compatible OpenAI pour le fournisseur demande."""
    from openai import AzureOpenAI, OpenAI

    if provider in OPENAI_COMPATIBLE:
        base_url, key_var = OPENAI_COMPATIBLE[provider]
        return OpenAI(base_url=base_url, api_key=_require(key_var),
                      max_retries=MAX_RETRIES, timeout=TIMEOUT_S)

    if provider == "azure_ai":
        api_key = _require("AZURE_OPENAI_API_KEY")
        return OpenAI(
            base_url=_require("AZURE_AI_INFERENCE_ENDPOINT"),
            api_key=api_key,
            default_headers={"api-key": api_key},
            default_query={"api-version": AZURE_AI_API_VERSION},
            max_retries=MAX_RETRIES,
            timeout=TIMEOUT_S,
        )

    if os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=_require("AZURE_OPENAI_API_KEY"),
            api_version=AZURE_API_VERSION,
            max_retries=MAX_RETRIES,
            timeout=TIMEOUT_S,
        )
    return OpenAI(api_key=_require("OPENAI_API_KEY"), max_retries=MAX_RETRIES, timeout=TIMEOUT_S)


def _require(var: str) -> str:
    value = os.environ.get(var)
    if not value:
        raise SystemExit(
            f"ERREUR : variable d'environnement {var} manquante. "
            "Copiez .env.example en .env et renseignez vos identifiants."
        )
    return value
