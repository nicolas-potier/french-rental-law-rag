# Evaluation protocol

How the benchmark compares models with and without retrieval, how answers are graded, and how to run or reproduce it. Back to the [README](../README.md) | See also: [Usage](usage.md), [Configuration](configuration.md).

Results: see the [README](../README.md).

**Contents**

1. [Question set](#1-question-set)
2. [Modes](#2-modes)
3. [Judge](#3-judge)
4. [Resume and quotas](#4-resume-and-quotas)
5. [Commands and options](#5-commands-and-options)
6. [Output files](#6-output-files)
7. [Reproduce and re-grade](#7-reproduce-and-re-grade)

## 1. Question set

`evaluation/questions.xlsx` holds 40 French rental-law questions. Columns:

| Column | Content |
|--------|---------|
| `id` | Question identifier. |
| `question` | The question asked to the model. |
| `reponse_reference` | Reference answer. |
| `elements_scoring` | Scoring grid: the elements an answer must contain. |
| `max_score` | Maximum score (5 points for every question). |
| `source_url` | Official source of the reference answer. |

## 2. Modes

Each question is asked to each model in three modes:

| Mode | Behaviour |
|------|-----------|
| `baseline` | The model answers alone, with a short legal-assistant system prompt and no context (temperature 0.1). |
| `rag` | The model answers from the top-k retrieved chunks (default 3). |
| `rag_recent` | Same, but retrieval excludes archives and sources dated before `--min-year` (default 2014, the ALUR law). Undated sources are kept. |

## 3. Judge

An LLM judge (default `gemma-4-31b`, temperature 0, JSON output) receives the question, the reference answer, the scoring grid, the maximum score and the answer to grade. Principles:

- score strictly against the grid: points for each correctly mentioned element;
- a **hallucination** (an invented law, figure or date) costs **1 point**, with a floor of 0;
- the reply must be valid JSON only: `score_obtenu`, `justification` (max 150 words), `hallucination_detectee` (true/false);
- the score is clamped between 0 and `max_score`; an empty answer scores 0.

## 4. Resume and quotas

The benchmark is designed to be spread over several days of free-tier quotas.

- Every record is appended to `evaluation/results/records.jsonl` as soon as the answer is generated, **before** grading. Grading then appends an updated line for the same (id, model, mode); the last line wins when the file is loaded.
- Re-running the same command skips answers that already exist and only generates and grades what is missing.
- An answer graded by a different judge than the current `--judge` is re-graded, so that all scores in a summary come from one judge.
- If a call fails with a 429 quota error, that model (or the judge) is set aside for the rest of the run, and the run continues with the others. "Request too large" 429 errors are not treated as quota exhaustion. Re-run later to resume.
- At the end, the number of answers still waiting for a grade is printed.

## 5. Commands and options

```bash
python -m evaluation.benchmark --models gpt-oss-120b --limit 10    # quick run
python -m evaluation.benchmark                                     # full run
python -m evaluation.benchmark --modes baseline rag_recent --judge gemma-4-31b
python -m evaluation.benchmark --no-judge                          # generate now, grade later
```

| Option | Default | Description |
|--------|---------|-------------|
| `--models M [M ...]` | `gpt-oss-120b qwen3.8-27b gpt-oss-20b` | Models to evaluate (any [chat model](configuration.md#chat-models)). |
| `--modes {baseline,rag,rag_recent} [...]` | all three | Modes to run. |
| `--judge` | `gemma-4-31b` | Judge model, one of the models in `rag/config.py`. With `--summary-only`, any name is accepted (e.g. `claude-opus-5-5`). |
| `--embedding {e5,ada002,3small,3large}` | `e5` | Embedding model / collection used for retrieval. |
| `--top-k` | `3` | Chunks retrieved in `rag` and `rag_recent`. |
| `--min-year` | `2014` | Minimum source year in `rag_recent`. |
| `--limit` | all 40 | Evaluate only the first N questions. |
| `--no-judge` | off | Generate answers without grading them. |
| `--summary-only` | off | Rebuild `summary.csv` from `records.jsonl`, no API call. |

The vector index must exist (see [Usage](usage.md#3-build-the-index)) for `rag` and `rag_recent`, and the relevant API keys must be set (see [Configuration](configuration.md#3-environment-variables)).

## 6. Output files

Both live in `evaluation/results/`.

**`records.jsonl`**: one JSON object per line, for each (question, model, mode). Re-graded records are appended again; the last line for a key wins.

| Field | Content |
|-------|---------|
| `id` | Question id. |
| `model`, `mode` | Model name and mode. |
| `answer` | The generated answer. |
| `latency_s` | Generation latency in seconds. |
| `words` | Number of words in the answer. |
| `max_score` | Maximum score of the question. |
| `sources` | Retrieved chunks: `title`, `url`, `year`, `similarity` (empty for baseline). |
| `judge` | Judge that graded it (`null` until graded). |
| `score` | Score obtained (`null` until graded). |
| `hallucination` | Whether a hallucination was flagged. |
| `justification` | The judge's short justification. |

**`summary.csv`**: one row per model and mode, UTF-8 with BOM.

| Column | Content |
|--------|---------|
| `model`, `mode` | Group keys. |
| `questions` | Number of graded answers. |
| `score` | Sum of scores. |
| `max_score` | Sum of maximum scores. |
| `hallucinations` | Number of answers flagged. |
| `latency_mean_s` | Mean latency in seconds. |
| `words_mean` | Mean answer length in words. |
| `score_pct` | `100 * score / max_score`, one decimal. |

## 7. Reproduce and re-grade

The published run is stored in `records.jsonl`. To recompute its summary with no API call and no key:

```bash
python -m evaluation.benchmark --summary-only --judge claude-opus-5-5
```

`--judge` selects which grades to keep. The published grades were produced by Claude (Opus 5.5) acting as judge in an assisted session, with the same rubric and hallucination penalty, on blind and shuffled answers; the judge name is stored on each record. The command rewrites `summary.csv` (identical content when run on the published records).

To re-grade the same answers with an automated judge (the default needs `GEMINI_API_KEY`):

```bash
python -m evaluation.benchmark --judge gemma-4-31b
```

Every answer graded by another judge is graded again, and the summary is computed from the new grades. The new grades are appended to `records.jsonl` and take precedence over the old ones, so keep a copy of the file first if you want to preserve the published grades (or restore it with `git checkout`). To generate fresh answers from scratch, move `records.jsonl` aside and run the full benchmark.
