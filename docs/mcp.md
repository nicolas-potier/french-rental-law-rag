# MCP server

Expose the retrieval of this project as a Model Context Protocol (MCP) server, so that any MCP client (Claude Desktop, MCP Inspector, a custom agent) can search the ANIL rental-law index. Back to the [README](../README.md) | See also: [Usage](usage.md), [Configuration](configuration.md).

**Contents**

1. [Purpose](#1-purpose)
2. [Tools and prompt](#2-tools-and-prompt)
3. [Run the server](#3-run-the-server)
4. [Inspect it](#4-inspect-it)
5. [Use it in Claude Desktop](#5-use-it-in-claude-desktop)
6. [Design notes](#6-design-notes)
7. [Troubleshooting](#7-troubleshooting)

## 1. Purpose

The server (`rag/mcp_server.py`) gives an LLM client two search tools and one prompt. The client, not the pipeline, decides when to search, how many times, and with which query. This suits multi-part questions, where one search per sub-question retrieves better extracts than a single search with the whole question.

Retrieval is entirely local (`multilingual-e5-small` embeddings and ChromaDB), so the server needs **no API key**. The client's own model does the answering. A minimal agent built on this server is available as `python -m rag.agent` (see the [README](../README.md)).

Prerequisite: the index must exist. Build it first with `python -m rag.chunk` and `python -m rag.embed` (see [Usage](usage.md#3-build-the-index)).

## 2. Tools and prompt

| Name | Arguments | Returns | When the model should use it |
|------|-----------|---------|------------------------------|
| `search_rental_law` (tool) | `query: str`, `top_k: int = 3`, `min_year: int \| None = None` | The `top_k` closest excerpts, separated by `---`, each with article title, year, URL and similarity score. `No excerpt found.` if nothing matches. | For any rental-law question. The docstring tells the model to write the query in French and, for a multi-part question, to call the tool "once per sub-question with a short, focused query rather than once with the whole question". `min_year` only to restrict to recent sources (see [why it is opt-in](usage.md#why---min-year-is-opt-in)). |
| `get_article` (tool) | `url: str` | The full article text, rebuilt from all its chunks, with section headings. `No article found for URL: ...` if the URL is not in the index. | When an excerpt from `search_rental_law` is cut off or lacks the detail needed. Pass the `URL` value of an excerpt. |
| `grounded_answer` (prompt) | `question: str` | The project's answer-with-sources instructions followed by the question. | To start a conversation that answers only from retrieved extracts and cites their URLs. |

## 3. Run the server

From the repository root:

```bash
python -m rag.mcp_server
```

The server uses the stdio transport. It loads the embedding model, opens the index, logs the following line to **stderr**, then waits for JSON-RPC messages on stdin:

```
INFO Index pret : 11371 extraits
```

Run by hand, it looks idle: that is expected, a client is supposed to talk to it. Press Ctrl+C to stop it. Extra log lines (Hugging Face HTTP requests, a progress bar) may appear before it: they are harmless.

To check it end to end without any GUI, use the Python SDK as a client. This is the script that was used to validate this page:

```python
import asyncio, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command=sys.executable, args=["-m", "rag.mcp_server"], cwd=".")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "search_rental_law",
                {"query": "délai de restitution du dépôt de garantie", "top_k": 1},
            )
            print(result.content[0].text)

asyncio.run(main())
```

Output (shortened with `…`):

```
[Retenue sur le dépôt de garantie et dédommagement du bailleur] (2017)
URL: https://www.anil.org/documentation-experte/analyses-juridiques-jurisprudence/analyses-juridiques/analyses-juridiques-2010/retenue-sur-le-depot-de-garantie-et-dedommagement-du-bailleur/
Similarity: 0.940

Titre : Retenue sur le dépôt de garantie et dédommagement du bailleur
URL : https://www.anil.org/…/retenue-sur-le-depot-de-garantie-et-dedommagement-du-bailleur/
Section : Délai de restitution

Le dépôt de garantie doit être restitué dans le délai maximal de deux mois lorsque la comparaison entre l’état des lieux d’entrée et l’état des lieux de sortie fait apparaître des dégradations dont le locataire est responsable. Si ces documents sont conformes, le délai de restitution est ramené à un mois. …
```

## 4. Inspect it

The MCP Inspector is a browser UI to list the tools, call them with your own arguments, and read the raw responses. It needs Node.js.

```bash
# Option A: through the mcp CLI (needs Node.js and, in MCP SDK v2, uv)
mcp dev rag/mcp_server.py

# Option B: Inspector launched directly (needs Node.js only)
npx @modelcontextprotocol/inspector python -m rag.mcp_server
```

In the Inspector, connect, open the **Tools** tab, then **List Tools** and run `search_rental_law` with a French query.

What was tested on the development machine (Windows 11, `mcp` 2.3.0, no Node.js, no `uv`):

- **Not testable here:** neither option could be run to the end, because Node.js (`npx`) and `uv` are not installed and were not installed for this test. The Inspector UI itself was never opened.
- `mcp dev rag/mcp_server.py` launched from the repository root failed with `ModuleNotFoundError: No module named 'rag'`: the command imports the file directly, so the `rag` package must be importable. With `PYTHONPATH=.` (Git Bash) the import succeeded and the command then stopped with `npx not found`. If you run it, set `PYTHONPATH` to the repository root.
- In MCP SDK v2, `mcp dev` and `mcp install` run the server through `uv run --with ...`, so `uv` is expected. This was not verified here.
- Option B was not run. The server itself works over stdio (see [section 3](#3-run-the-server)), which is all the Inspector needs.

## 5. Use it in Claude Desktop

Claude Desktop starts the server itself as a child process. Edit its configuration file (create it if missing), then restart Claude Desktop completely.

| OS | Location |
|----|----------|
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |

```json
{
  "mcpServers": {
    "anil-rental-law": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "rag.mcp_server"],
      "env": {
        "PYTHONPATH": "/path/to/french-rental-law-rag",
        "PYTHONIOENCODING": "utf-8"
      }
    }
  }
}
```

- `command`: the Python of the environment where the project dependencies are installed. Use its absolute path. On Windows this is typically `C:/path/to/venv/Scripts/python.exe`.
- `PYTHONPATH`: the repository root, so that `rag` is importable.
- No `cwd` is used: Claude Desktop's documented server entries are `command`, `args` and `env`, and `cwd` support could not be confirmed. It is not needed, because `rag/config.py` resolves the data paths from the package location (verified: see below).

Windows paths in JSON need either forward slashes (`C:/Users/me/...`) or escaped backslashes (`C:\\Users\\me\\...`).

### What was tested

- **Pass:** the server started from another directory (`cd` to the home folder, `PYTHONPATH` set to the repository root, no `cwd`) and logged `Index pret : 11371 extraits`. The `env` approach above therefore works for the server itself.
- **Not testable here:** Claude Desktop is not available on the development machine, so the configuration snippet and the "Try it" scenario below were not run inside Claude Desktop.

### Try it

After restarting Claude Desktop, the tools should appear in the tools menu of a new conversation. Ask a question with several parts, in French:

> Mon propriétaire veut augmenter mon loyer et garder une partie de mon dépôt de garantie. Quel est le délai de restitution du dépôt, et dans quels cas peut-il augmenter le loyer en cours de bail ?

What to look for:

- Claude calls `search_rental_law` **several times**, once per sub-question (deposit return delay, rent increase during the lease), rather than once with the whole text.
- It may call `get_article` when an excerpt is incomplete.
- The final answer cites the ANIL **URLs** returned by the tools and says so when the extracts do not contain the answer.

To force the sourced style, pick the `grounded_answer` prompt from the attachments menu, if your client version lists MCP prompts.

## 6. Design notes

- **stdio and stderr.** On the stdio transport, stdout carries the JSON-RPC protocol. Anything printed there would corrupt it, so the server logs to stderr (`logging.basicConfig(stream=sys.stderr)`). In MCP SDK v2, stdout is also redirected while serving, but logs still go to stderr by design. Clients usually surface stderr in their own log files.
- **Warm start.** The embedding model is loaded and the index opened once at startup (`_warm_up`), not at the first query. The first start is slower; later calls only embed the query.
- **No API key.** Retrieval is local, so the server needs no credentials (importing `rag.config` loads an optional `.env`, but no key from it is used).
- **Article truncation.** `get_article` truncates a result beyond 20,000 characters (`MAX_ARTICLE_CHARS`) and appends `[... article truncated]`, to avoid saturating the client's context.
- **Absolute data paths.** `rag/config.py` resolves `data/` from the package location, not from the current directory, so the server works from any working directory as long as `rag` is importable.
- **SDK version.** The server uses `MCPServer` (`mcp.server.mcpserver`), the MCP SDK v2 name of what was `FastMCP` in v1.

## 7. Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| An error about a missing collection or an empty index at startup. | The index is not built. Run `python -m rag.chunk` then `python -m rag.embed` (see [Usage](usage.md)). The server exits with `ERREUR : collection ... introuvable` (from `rag.ask.open_collection`, not run here). |
| The first start takes a long time or seems stuck. | The embedding model (`multilingual-e5-small`) is downloaded from Hugging Face on first use. Wait for it to finish, or run the server once manually before connecting a client, because some clients time out the startup. |
| Claude Desktop shows the server as failed. | Check the logs in the Claude Desktop log folder (the `mcp-server-*.log` files). Typical causes: wrong `command` path, `PYTHONPATH` not set (`No module named 'rag'`), invalid JSON. |
| `No module named 'rag'` | `rag` is not importable. Run from the repository root, or set `PYTHONPATH` to it. |
| JSON parse error in the config file | On Windows, backslashes must be doubled (`C:\\Users\\...`) or replaced by forward slashes. |
| `mcp dev` fails or `npx not found` | Node.js is required for the Inspector, and `uv` for `mcp dev` in SDK v2. See [section 4](#4-inspect-it). |
