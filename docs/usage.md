# Usage reference

Step-by-step reference for the four pipeline steps, with every command-line option. Back to the [README](../README.md) | See also: [Configuration](configuration.md), [Evaluation](evaluation.md).

**Contents**

1. [Scrape](#1-scrape-optional)
2. [Chunk](#2-chunk)
3. [Build the index](#3-build-the-index)
4. [Ask a question](#4-ask-a-question)

All commands run from the repository root. Every command accepts `--help`. Credentials and models are described in [Configuration](configuration.md).

```
scrape ──► data/corpus/*.docx ──► chunk ──► data/chunks.jsonl ──► embed ──► data/vector_db/ ──► ask
```

## 1. Scrape (optional)

Crawls 12 sections of [anil.org](https://www.anil.org) (4 Resource Centre sections and 8 "Parole d'expert" sections), cleans the text (non-breaking spaces, invisible characters, navigation and footer blocks), and merges the articles into Word files. Downloads are concurrent and checkpointed in JSON, so an interrupted crawl can resume.

The corpus is already versioned in `data/corpus/` (5 `.docx` files): **skip this step unless you want to refresh it.**

- Input: the ANIL website (network required).
- Output: one `.docx` per section in `data/anil_sections/`, then merged files `data/corpus/anil_complet*.docx`.

```bash
python -m rag.scrape --crawl --workers 4 --delay 0.25
```

| Option | Default | Description |
|--------|---------|-------------|
| `url` (positional) | none | Scrape a single article URL. |
| `output` (positional) | none | Output `.docx` for single-page mode. |
| `--crawl` | off | Scrape the selected sections. |
| `--sections S [S ...]` | all 12 | Sections to scrape: `analyses-juridiques`, `etudes-eclairages`, `habitat-actualite`, `jurisprudence`, `parole-achat-vente-immobiliere`, `parole-assurance`, `parole-copropriete`, `parole-habitat-indigne`, `parole-habiter-vie-pratique`, `parole-location`, `parole-renovation-energetique`, `parole-urbanisme`. |
| `--output-dir` | `data/anil_sections` | Folder for the per-section `.docx` files. |
| `--merged-output` | `data/corpus/anil_complet` | Base name (no extension) of the merged files. |
| `--merge-only` | off | Only merge existing `.docx` files, no scraping. |
| `--list-urls-only` | off | List detected article URLs without scraping them. |
| `--articles-per-file` | `800` | Maximum number of articles per merged Word file. |
| `--max-articles` | no limit | Limit of articles per section, for testing. |
| `--delay` | `0.25` | Delay per HTTP request, in seconds. |
| `--workers` | `4` | Number of concurrent downloads. |

Examples:

```bash
python -m rag.scrape --list-urls-only
python -m rag.scrape --crawl --sections parole-location --workers 4 --delay 0.25
python -m rag.scrape --merge-only --articles-per-file 800
```

## 2. Chunk

Reads the `.docx` corpus and splits each article into chunks sized for retrieval. Each chunk is prefixed with `Titre / URL / Section` so the LLM always knows where a passage comes from and can cite it. Expected runtime: about 1 minute (the published corpus gives 11,371 chunks).

- Input: `data/corpus/` (a folder of `.docx`, or a single `.docx`).
- Output: `data/chunks.jsonl`, one JSON chunk per line.

```bash
python -m rag.chunk
python -m rag.chunk --preview     # show the first 3 chunks, write nothing
```

| Option | Default | Description |
|--------|---------|-------------|
| `input` (positional) | `data/corpus` | Folder of `.docx` files, or a single `.docx`. |
| `--output`, `-o` | `data/chunks.jsonl` | Output `.jsonl` file. |
| `--max-chars` | `3000` | Maximum chunk size in characters. |
| `--overlap-chars` | `500` | Overlap in characters when a section is split further. |
| `--min-chars` | `300` | Minimum chunk size to be kept; `0` disables the filter. |
| `--preview` | off | Print the first 3 chunks instead of writing the file. |

### Chunking strategy

Rules apply in priority order:

1. If the whole article (title + URL + body) fits in `--max-chars` (3,000), it stays a single chunk, which preserves article context.
2. Otherwise, it is cut by H2 headings: the intro before the first H2, then each H2 section, become independent chunks.
3. If a section still exceeds `--max-chars`, it is re-split by accumulating paragraphs (separated by blank lines) with `--overlap-chars` (500) characters of overlap to keep continuity.

Chunks shorter than `--min-chars` (300) are dropped.

Chunk format:

```json
{"id": "...", "text": "Titre : ...\nURL : ...\nSection : ...\n\n<body>",
 "metadata": {"source_file": "", "article_title": "", "article_url": "",
              "section_title": "", "chunk_index": 0, "char_count": 0,
              "year": 0, "archived": false}}
```

### Article dating

The corpus spans 30 years of analyses, many predating reforms such as the 2014 ALUR law. Each article gets a `year` (`0` if unknown), by decreasing priority:

1. the "À jour au `<day> <month> <year>`" mention at the top of the article;
2. otherwise a year found in the URL (e.g. `.../analyses-juridiques-2010/...`);
3. otherwise, for case law, the date of the cited decision (e.g. "Cass. Civ III : 21.11.19").

The `archived` flag marks pages coming from ANIL archives. These two fields feed the recency filter at retrieval time (see `--min-year` in [step 4](#why---min-year-is-opt-in)).

## 3. Build the index

Embeds every chunk and stores the vectors in a local ChromaDB collection (cosine distance), one collection per embedding model. Upserts are idempotent: re-running updates existing chunks.

- Input: `data/chunks.jsonl`.
- Output: `data/vector_db/` (ChromaDB persistent store).

```bash
python -m rag.embed
python -m rag.embed --max-chunks 500      # quick test
```

Runtime: with the default local model (`multilingual-e5-small`), embedding the 11,371 chunks takes about 15 to 45 minutes on a laptop CPU. Use `--max-chunks 500` to check the whole pipeline quickly, or `--dry-run` to preview chunks without computing anything. API models (`ada002`, `3small`, `3large`) need credentials, see [Configuration](configuration.md#3-environment-variables).

| Option | Default | Description |
|--------|---------|-------------|
| `chunks_file` (positional) | `data/chunks.jsonl` | Chunks `.jsonl` file. |
| `--embedding {e5,ada002,3small,3large}` | `e5` | Embedding model (see [models](configuration.md#4-models)). |
| `--db-dir` | `data/vector_db` | ChromaDB folder. |
| `--reset` | off | Delete the collection before re-indexing. |
| `--max-chunks` | all | Process only the first N chunks (testing). |
| `--dry-run` | off | Compute no embedding, print a preview of the chunks. |

Use the same `--embedding` at query time as at indexing time: each model has its own collection.

## 4. Ask a question

Embeds the question with the same model as the index, retrieves the top-k closest chunks, then asks an LLM to answer using only those extracts and to cite the source URL for each claim. If the extracts do not contain the answer, the model says so.

- Input: the question, `data/vector_db/`, an API key for the chosen chat model.
- Output: the answer and the list of sources, printed to the terminal.

```bash
python -m rag.ask "Quel est le montant maximal du dépôt de garantie en location meublée ?"
python -m rag.ask "..." --model gpt-oss-20b --top-k 5 --full
python -m rag.ask "..." --min-year 2014
```

| Option | Default | Description |
|--------|---------|-------------|
| `question` (positional) | required | The question to ask. |
| `--model` | `gpt-oss-120b` | Generation model, see [chat models](configuration.md#chat-models). |
| `--embedding {e5,ada002,3small,3large}` | `e5` | Embedding model / collection. |
| `--top-k` | `3` | Number of chunks retrieved. |
| `--min-year` | off | Drop sources dated before this year and archive pages. Undated articles are kept. |
| `--db-dir` | `data/vector_db` | ChromaDB folder. |
| `--full` | off | Print the full text of the retrieved chunks. |

### Why `--min-year` is opt-in

Excluding pre-ALUR (2014) analyses looks sensible, since they may describe rules since changed. The benchmark shows the opposite on average: the filter lowers scores, because an old article is not necessarily an outdated rule (some of the best sources are old but still valid law). It is therefore off by default. Technically, the filter keeps chunks that are not archived and have `year >= min-year` or `year == 0` (unknown). Numbers: see the [README](../README.md) and the [evaluation protocol](evaluation.md).
