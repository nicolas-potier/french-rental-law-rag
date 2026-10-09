# Retrieval-Augmented Legal QA for French Rental Law

A RAG pipeline that answers French rental-law and housing-tax questions **with citations to official sources**, built on the public documentation of the [ANIL](https://www.anil.org) (Agence nationale pour l'information sur le logement).

Client project for a French legal-tech startup, CentraleSupélec, team of 5 (Oct 2025 – June 2026).

**My role:** I built the RAG pipeline (ANIL scraping, hierarchical chunking, embedding and ChromaDB indexing, grounded generation with citations) and designed the evaluation harness (reference questions, scoring grids, LLM-as-a-judge with hallucination penalty, RAG vs no-retrieval comparison).

**Original setup vs this repository.** During the project, the pipeline ran on the client's Azure OpenAI resource: ada-002 embeddings, four Azure-hosted models benchmarked in RAG vs no-retrieval mode, and GPT-4o-mini (temperature 0, JSON output) as the judge. That access ended with the project, so the original Azure runs cannot be reproduced. I therefore refactored the code so that anyone can run it for free (local embeddings, free-tier models, no client credentials) while keeping the Azure configuration available, and re-ran the full 40-question benchmark in October 2026. The [results](#results) below come from that re-run, not from the original Azure benchmark.

```
anil.org ──► scrape ──► .docx corpus ──► hierarchical chunking ──► embeddings ──► ChromaDB
                                          (+ article dating)                        │
                     question ──► embedding ──► top-k retrieval ◄───────────────────┘
                                               (optional recency filter)
                                                      │
                                                      ▼
                                   LLM answer grounded in the extracts + source URLs
```

## Highlights

- **Scraping**: concurrent crawler for 12 ANIL sections (legal analyses, case law, expert Q&A…), with text cleaning, navigation/footer removal and JSON checkpoints to resume after interruption.
- **Hierarchical chunking**: the whole article if it fits in 3,000 characters, otherwise one chunk per H2 section, otherwise paragraph-level splitting with a 500-character overlap. Each chunk is prefixed with `Title / URL / Section` so the LLM always knows where a passage comes from and can cite it.
- **Article dating & recency filter**: the corpus spans 30 years of legal analyses, many of which predate major reforms such as the 2014 ALUR law. Each article is dated (its "À jour au …" mention, else the year in its URL, else the decision date for case law; 91 % of chunks get a year) and `--min-year` can drop older and archived sources at retrieval time. The benchmark shows this filter does *not* help (see [Results](#results)), so it is off by default.
- **Vector search**: free local embeddings (`multilingual-e5-small`) by default, or OpenAI / Azure OpenAI embeddings, stored in a local ChromaDB collection (cosine distance).
- **Grounded generation**: the system prompt forces answers to rely only on the retrieved extracts, to cite the URL supporting each claim, and to say explicitly when the extracts do not contain the answer. Works with any OpenAI-compatible provider (Groq, Gemini, OpenAI, Azure).
- **Evaluation harness**: 40 reference questions with a scoring grid, graded against the rubric with a **−1 penalty per detected hallucination**; compares models in **no-retrieval baseline vs RAG vs RAG on recent sources** mode. With RAG, the average score rises from 54 % to 68 % and hallucinations drop from 111 to 5 over 120 answers.

## Repository layout

```
.
├── rag/
│   ├── config.py      # paths, models, API clients
│   ├── scrape.py      # 1. anil.org  -> data/corpus/*.docx
│   ├── chunk.py       # 2. .docx     -> data/chunks.jsonl (chunks + year metadata)
│   ├── embed.py       # 3. chunks    -> data/vector_db/ (ChromaDB)
│   ├── embeddings.py  #    local (sentence-transformers) or API embeddings
│   └── ask.py         # 4. question  -> retrieval + cited answer
├── evaluation/
│   ├── questions.xlsx # 40 questions, reference answers, scoring grids
│   ├── benchmark.py   # baseline vs RAG vs RAG (recent), LLM-as-a-judge
│   └── results/       # published run: every answer, its sources and its grade
├── data/
│   └── corpus/        # scraped ANIL corpus (5 .docx files, versioned)
├── requirements.txt
└── .env.example
```

`data/chunks.jsonl` and `data/vector_db/` are generated locally and are not versioned.

## Quickstart (free)

Requires Python 3.10+ and a free [Groq](https://console.groq.com/keys) API key (no credit card). Retrieval runs entirely locally.

```bash
git clone https://github.com/nicolas-potier/french-rental-law-rag.git
cd french-rental-law-rag
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then fill in GROQ_API_KEY
```

Build the index from the versioned corpus, then ask a question:

```bash
python -m rag.chunk                # ~1 min
python -m rag.embed                # local embeddings, 15-45 min on a laptop CPU
python -m rag.ask "Quel est le montant maximal du dépôt de garantie en location meublée ?"
```

To try it on a subset first: `python -m rag.embed --max-chunks 500`.

## Usage

| Step | Command | Main options |
|------|---------|--------------|
| Scrape ANIL (optional, corpus already provided) | `python -m rag.scrape --crawl` | `--sections`, `--workers`, `--delay`, `--max-articles` |
| Chunk the corpus | `python -m rag.chunk` | `--max-chars 3000`, `--overlap-chars 500`, `--min-chars 300`, `--preview` |
| Build the vector index | `python -m rag.embed` | `--embedding {e5,ada002,3small,3large}`, `--reset`, `--max-chunks`, `--dry-run` |
| Ask a question | `python -m rag.ask "..."` | `--model`, `--min-year`, `--top-k 3`, `--embedding`, `--full` |
| Run the benchmark | `python -m evaluation.benchmark` | `--models`, `--modes`, `--judge`, `--min-year`, `--limit`, `--no-judge`, `--summary-only` |

Every command accepts `--help`.

## Evaluation

`evaluation/questions.xlsx` contains 40 rental-law questions (security deposit, notice periods, rent revision, charges, repairs, …). Each one comes with a reference answer, a point-based scoring grid (5 points) and the official source URL.

For each question and each model, `benchmark.py`:

1. generates three answers: **baseline** (the model alone), **rag** (top-3 chunks as context) and **rag_recent** (top-3 chunks among sources dated 2014 or later, archives excluded);
2. asks a **judge** LLM (temperature 0, JSON output) to grade each answer against the grid and to flag hallucinations (invented law, article, figure or date), each one costing 1 point;
3. aggregates score (% of max), hallucination count, mean latency and verbosity per model × mode in `evaluation/results/summary.csv`.

Every answer is appended to `evaluation/results/records.jsonl` as soon as it is generated (with its sources), then updated with the grade and its justification. When a model or the judge hits its quota, it is skipped for the rest of the run; re-running the command resumes where it stopped, which makes it possible to spread a full run over several days of free-tier quotas.

```bash
python -m evaluation.benchmark --models gpt-oss-120b --limit 10   # quick run
python -m evaluation.benchmark                                    # full run (gpt-oss-120b, gpt-oss-20b, Qwen; Gemma judge)
python -m evaluation.benchmark --summary-only --judge claude-opus-5-5   # recompute the published summary, no API call
```

### Results

Published run: 40 questions × 3 free models (Groq) × 3 modes = 360 answers, local `multilingual-e5-small` retrieval, top-3 chunks.

| Model | Baseline | RAG | RAG (sources ≥ 2014) | Hallucinations (baseline → RAG) |
|-------|---------:|----:|---------------------:|:-------------------------------:|
| gpt-oss-120b | 61.2 % | **68.8 %** | 64.8 % | 39 → 1 |
| gpt-oss-20b | 42.0 % | **69.0 %** | 65.5 % | 36 → 4 |
| qwen3.8-27b | 57.8 % | **66.8 %** | 64.5 % | 36 → 0 |
| **Average** | 53.7 % | **68.2 %** | 64.9 % | 111 → 5 |

Takeaways:

- **Grounding removes almost all hallucinations.** Without retrieval, models routinely invent article numbers, case law and figures (111 penalised answers out of 120); with the extracts and the citation constraint, 5.
- **RAG levels the field**: the smallest model (gpt-oss-20b) gains 27 points and matches the largest one.
- **Remaining errors are retrieval misses, not reasoning errors.** RAG loses to the baseline on 13 of 40 questions, almost always because the top-3 chunks do not contain the answer and the model, as instructed, says so (≈ 18 % of RAG answers). Multi-part questions (e.g. "what is the deadline *and* what happens if it is missed?") are the typical case. Next steps: larger top-k, hybrid BM25 + vector search, query decomposition, or a reranker.
- **Filtering by date hurts (−3.3 points).** An old article is not necessarily an outdated rule: on Q23, the only relevant source is a 2004 analysis on rent-receipt fees that is still valid law, and the filter drops the score from 5/5 to 0/5. Keeping the filter as an opt-in was the right call.

**How the published run was graded.** The free Gemini Flash tier is capped at 20 requests per day and the free Gemma endpoint was too often overloaded to grade 360 answers. The published grades were therefore produced by Claude (Opus 5.5) acting as the judge in an assisted session, with the same rubric and hallucination penalty as the automated judge, on blind, shuffled answers (model and mode hidden). The judge name is stored with each record; running the benchmark with `--judge gemma-4-31b` re-grades every answer automatically.

## Configuration

Credentials are read from environment variables or from a `.env` file (see [.env.example](.env.example)). Models and providers are listed in [rag/config.py](rag/config.py).

| Provider | Variables | Models |
|----------|-----------|--------|
| Groq (free tier) | `GROQ_API_KEY` | `gpt-oss-120b`, `gpt-oss-20b`, `qwen3.8-27b` |
| Google AI Studio (free tier) | `GEMINI_API_KEY` | `gemma-4-31b` (default judge), `gemini-3.5-flash`, `gemini-3.8-flash` (20 requests/day on the free tier) |
| OpenAI / Azure OpenAI | `OPENAI_API_KEY`, or `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_API_KEY` | `gpt-4o-mini`, `gpt-5.4-nano`, `ada002` / `3small` / `3large` embeddings |
| Azure AI Foundry | `AZURE_AI_INFERENCE_ENDPOINT` | `mistral-small-2503`, `grok-4-1-fast-non-reasoning` |

The project was originally built on Azure OpenAI; the free providers and local embeddings were added so that it can be run without a paid account.

## Tech stack

Python · ChromaDB · sentence-transformers · OpenAI-compatible APIs (Groq, Gemini, OpenAI, Azure) · python-docx · BeautifulSoup · pandas

## Data & license

The corpus is built from content published on [anil.org](https://www.anil.org); all rights to that content belong to the ANIL. It is included here for research and demonstration only. Answers produced by this system are not legal advice.
