# Configuration

Prerequisites, dependencies, environment variables, models, and file paths. Back to the [README](../README.md) | See also: [Usage](usage.md), [Evaluation](evaluation.md).

**Contents**

1. [Prerequisites](#1-prerequisites)
2. [Dependencies](#2-dependencies)
3. [Environment variables](#3-environment-variables)
4. [Models](#4-models)
5. [Paths](#5-paths)

## 1. Prerequisites

- Python 3.10 or later.
- About 2 GB of disk space for PyTorch (CPU) and the downloaded models.
- Internet access to download the embedding model on first use (Hugging Face) and to call LLM APIs. Retrieval itself runs locally.

## 2. Dependencies

Install with `pip install -r requirements.txt`.

| Package | Purpose |
|---------|---------|
| `openai>=1.40` | Client for all chat providers (Groq, Gemini, OpenAI, Azure) and API embeddings. |
| `chromadb>=0.5` | Local vector store (cosine distance). |
| `sentence-transformers>=3.0` | Local embeddings (`multilingual-e5-small`); pulls in PyTorch. |
| `python-docx>=1.1` | Writes the corpus (scrape) and reads it (chunk). |
| `python-dotenv>=1.0` | Loads the `.env` file. |
| `pandas>=2.0` | Benchmark data handling and `summary.csv`. |
| `openpyxl>=3.1` | Reads `evaluation/questions.xlsx`. |
| `requests>=2.31` | HTTP downloads for the scraper. |
| `beautifulsoup4>=4.12` | HTML parsing for the scraper. |

`sentence-transformers` installs PyTorch, whose default wheel is large. To save space, install the CPU wheel first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 3. Environment variables

Variables are read from the process environment, or from a `.env` file at the repository root, loaded with `python-dotenv` when `rag/config.py` is imported (python-dotenv is optional: without it only real environment variables are read). Copy the template and fill in what you need:

```bash
cp .env.example .env
```

Never commit `.env`. Local embeddings (the default) need no key.

| Variable | Required for | Where to get it |
|----------|--------------|-----------------|
| `GROQ_API_KEY` | Groq models: `gpt-oss-120b` (default), `gpt-oss-20b`, `qwen3.8-27b` | Free key at [console.groq.com/keys](https://console.groq.com/keys). |
| `GEMINI_API_KEY` | Gemini/Gemma models, including the default judge `gemma-4-31b` | Free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). |
| `OPENAI_API_KEY` | OpenAI models (`gpt-4o-mini`, `gpt-5.4-nano`) and `ada002` / `3small` / `3large` embeddings | OpenAI platform (paid). |
| `AZURE_OPENAI_ENDPOINT` | Same OpenAI models/embeddings, served by Azure OpenAI | Your Azure OpenAI resource, e.g. `https://<resource>.openai.azure.com/`. |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI, and also Azure AI Foundry models | Azure portal, resource keys. |
| `AZURE_AI_INFERENCE_ENDPOINT` | Azure AI Foundry models (`mistral-small-2503`, `grok-4-1-fast-non-reasoning`) | Your Foundry resource, e.g. `https://<resource>.services.ai.azure.com/models`. |
| `AZURE_OPENAI_API_VERSION` | Optional | Defaults to `2025-01-01-preview`. |

A missing required variable stops the program with an explicit message.

### Provider selection

Each chat model is bound to a provider in `rag/config.py`:

- `groq` and `gemini` use their OpenAI-compatible endpoints with their own key.
- `azure_ai` uses `AZURE_AI_INFERENCE_ENDPOINT` with `AZURE_OPENAI_API_KEY`.
- `openai` (and API embeddings) use **Azure OpenAI if `AZURE_OPENAI_ENDPOINT` is set**, otherwise plain OpenAI with `OPENAI_API_KEY`.

## 4. Models

### Embeddings

Select with `--embedding`. Each alias has its own ChromaDB collection, so indexes of different models coexist.

| Alias | Model | Collection | Provider |
|-------|-------|------------|----------|
| `e5` (default) | `intfloat/multilingual-e5-small` | `anil_e5_small` | local (CPU) |
| `ada002` | `text-embedding-ada-002` | `anil_qna_ada002` | OpenAI / Azure OpenAI |
| `3small` | `text-embedding-3-small` | `anil_qna_3small` | OpenAI / Azure OpenAI |
| `3large` | `text-embedding-3-large` | `anil_qna_3large` | OpenAI / Azure OpenAI |

The local model uses the `passage:` / `query:` prefixes expected by e5, with normalized vectors.

### Chat models

Select with `--model` (or `--models` / `--judge` in the benchmark). Default chat model: `gpt-oss-120b`. Default judge: `gemma-4-31b`.

| Name | Provider | API id | Notes |
|------|----------|--------|-------|
| `gpt-oss-120b` | groq | `openai/gpt-oss-120b` | Default. `reasoning_effort=low`, `max_tokens=2048`. |
| `gpt-oss-20b` | groq | `openai/gpt-oss-20b` | `reasoning_effort=low`, `max_tokens=2048`. |
| `qwen3.8-27b` | groq | `qwen/qwen3.8-27b` | `max_tokens=2048`. |
| `gemma-4-31b` | gemini | `gemma-4-31b-it` | Default judge. |
| `gemini-3.5-flash` | gemini | `gemini-3.5-flash` | Very small free quota. |
| `gemini-3.8-flash` | gemini | `gemini-3.8-flash` | Very small free quota. |
| `gpt-4o-mini` | openai | `gpt-4o-mini` | Original benchmark model (paid / Azure). |
| `gpt-5.4-nano` | openai | `gpt-5.4-nano` | Original benchmark model (paid / Azure). |
| `mistral-small-2503` | azure_ai | `mistral-small-2503` | Azure AI Foundry. |
| `grok-4-1-fast-non-reasoning` | azure_ai | `grok-4-1-fast-non-reasoning` | Azure AI Foundry. |

Output is capped at 2,048 tokens on Groq because its per-minute token limit counts input plus `max_tokens`.

### Free-tier limits observed

| Provider / model | Limit observed |
|------------------|----------------|
| Groq (free) | About 200k tokens per day per model, 8k tokens per minute. |
| Gemini Flash (free) | 20 requests per day per model. |
| Gemma via the Gemini API | Much larger quota, but frequent 503 overload errors. |

### Retries and timeout

All OpenAI-compatible clients are created with `MAX_RETRIES = 4` and `TIMEOUT_S = 90` (`rag/config.py`). The OpenAI SDK retries automatically on 429 (per-minute quota) and 503 (overload) errors, honouring the `retry-after` header; the timeout prevents a call from hanging. A daily quota error still fails after the retries: the benchmark then sets that model aside and resumes on the next run (see [Evaluation](evaluation.md#4-resume-and-quotas)). API embeddings are sent in batches of 96 with up to 5 exponential-backoff retries.

## 5. Paths

Defined in `rag/config.py`, relative to the repository root.

| Path | Content | Versioned |
|------|---------|-----------|
| `data/corpus/` | Scraped ANIL corpus, 5 `.docx` files | Yes |
| `data/chunks.jsonl` | Chunks with metadata (step 2) | No, generated |
| `data/vector_db/` | ChromaDB index (step 3) | No, generated |
| `data/anil_sections/` | Per-section `.docx` from the scraper | No, generated |
| `evaluation/questions.xlsx` | The 40 benchmark questions | Yes |
| `evaluation/results/` | Published benchmark run (`records.jsonl`, `summary.csv`) | Yes |
| `.env` | Your credentials | No, never commit |
