# Offline PDF semantic search

Goal: enter a natural-language question or sentence and retrieve the best 5–10
matching sentences, with similarity scores, PDF filenames, and page numbers,
using the local Qwen3-Embedding-0.6B model.

## Build sequence

1. **PDF sentence catalog (implemented).** Extract text and preserve citations.
2. **Embedding index (implemented).** Filter the catalog, encode surviving
   sentences with the local Qwen model, and persist normalized vectors with
   their catalog IDs.
3. **Search (implemented).** Encode a query, rank by cosine similarity, and
   display 5–10 hits.
4. **Eval (implemented).** Run known queries and check that the expected PDF
   appears in the top hits.
5. **SEARCH UI (implemented).** Local Streamlit page over the same index.

## Step 1: prepare the PDFs

Install dependencies once while connected:

```sh
.venv/bin/python -m pip install -r requirements-ingest.txt
```

Then run locally, without network access or model downloads:

```sh
.venv/bin/python prepare_documents.py
```

Outputs (regenerated on each run):

- `data/sentences.jsonl`: one sentence candidate per line, including `text`,
  `file`, `page`, `page_label`, and a stable content/page/position ID.
- `data/extraction_report.json`: document counts, empty pages, errors, and
  byte-identical duplicate PDFs skipped to avoid repeated search hits.

`page` is the physical PDF page, starting at 1, for navigation in a PDF viewer.
`page_label` is the PDF's embedded label, which may differ from printed numbering.
File paths are relative to `source/`. Use `--source`, `--output`, and `--language`
to override defaults; sentence splitting defaults to English and uses bundled
pySBD rules without downloading language data.

This first pass preserves page boundaries: sentences spanning pages will be
fragments. Headings and table content may also become sentence candidates.
Multi-column reading order, line-end hyphens, and repeated headers need review
before embedding. Image-only pages require a later OCR stage; empty pages are
reported rather than silently treated as searchable. Language selection applies
to the whole run, so mixed-language documents need further handling.

The existing `download_model.py` is a one-time online model setup script.
`test_model.py` remains the local embedding smoke test.

## Step 2: build the embedding index

Install embedding dependencies once while connected (already present if you
ran `test_model.py`):

```sh
.venv/bin/python -m pip install -r requirements-search.txt
```

Filter the catalog and report what would be indexed, without loading the model:

```sh
.venv/bin/python build_index.py --dry-run
```

Then encode locally. The script disables Hugging Face network access and loads
only `models/Qwen3-Embedding-0.6B`. On this Mac it uses MPS when available.

```sh
.venv/bin/python build_index.py
```

Outputs (regenerated when the catalog, model path, or filter rules change):

- `data/embeddings.npy`: L2-normalized `float32` matrix, one row per sentence.
- `data/index_meta.jsonl`: row-aligned `id`, `text`, `file`, `page`, `page_label`.
- `data/index_config.json`: model path, dimension, query prompt name, catalog
  hash, and filter settings. Rebuilds are skipped when this fingerprint matches.
- `data/index_report.json`: per-file kept/dropped counts.

Step 1.5 runs inside the indexer: line-end hyphens are joined, then short
fragments, dotted TOC lines, and scan/garbage text are dropped so they are not
embedded. Use `--force` to rebuild anyway, `--limit N` for a partial smoke
index, `--batch-size` (default 8), and `--device cpu` to avoid GPU/MPS. Long sentences
are truncated at 512 tokens so a table dump cannot exhaust MPS memory.

Documents are encoded with no instruct prefix. Queries use `prompt_name="query"`.

## Step 3: search

Requires the index from step 2. Runs offline against the local model:

```sh
.venv/bin/python search.py "How does the Federal Reserve monitor financial stability?"
.venv/bin/python search.py --top-k 5 "NIST AI risk management functions"
```

Each hit is a cosine score, the PDF filename, the 1-based physical page, and
the matching sentence. `--top-k` must be 5–10 (default 8). Use `--index` and
`--model` to override the default `data/` index and local Qwen path.

The score is ranking similarity, not factual confidence. SEARCH returns source
sentences only; it does not generate an answer.

## Step 4: evaluate ranking

Loads the model once and runs a fixed set of known queries (Fed, NIST, Columbia,
Alice, Fluke, FOMC, IRS Pub 15, OSHA handbook, rare-event paper, Archives
transcript, the unindexed dust scan, and one Chinese paraphrase):

```sh
.venv/bin/python eval_search.py
```

A case passes when the expected filename is in the top 8 hits. The dust case
passes if `3371COMBUSTIBLE-DUST.pdf` does not appear (it was never indexed).
The Chinese query is reported but not required. Writes `data/eval_report.json`.

## Step 5: SEARCH UI

Install the UI dependency once while connected:

```sh
.venv/bin/python -m pip install -r requirements-app.txt
```

Then start the local app (binds to localhost only):

```sh
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

The first load pulls the local model into memory. Type a query, choose 5–10
results, and Search. Each card shows the sentence, cosine score, filename, and
page. **Open PDF at page** writes a temporary copy with the matching sentence
highlighted, serves it from `http://127.0.0.1`, and opens that page in your
default browser. Preview cannot jump or highlight from a `file://` link.

On this LAN Mac, bind the UI so other machines can open it:

```sh
.venv/bin/python -m streamlit run app.py --server.address 0.0.0.0
```

## Live source updates

Drop a PDF into `source/` (or delete/replace one). The running Streamlit app
and FastAPI service watch the folder every 2 seconds, wait until the file
size is stable, then extract, filter, and embed only the new document. After
that, searches include it the same way as the original corpus. Removed PDFs
leave the index; byte-identical copies are skipped.

No full rebuild is required. A one-shot CLI search also syncs before ranking:

```sh
.venv/bin/python sync_index.py
.venv/bin/python search.py "your query"
```

## FastAPI for Express

The Streamlit app is a prototype UI. For a JavaScript + Express front end, run
this service instead. It loads the local model once and answers JSON only.

```sh
.venv/bin/python -m pip install -r requirements-api.txt
.venv/bin/python api.py
```

Listens on `0.0.0.0:8000` so other machines on the LAN can reach it. The
service also watches `source/` and hot-swaps the in-memory index when PDFs
are added or removed. CORS still allows only `localhost` / `127.0.0.1`
browser origins; have Express proxy `/api/search` if the front end is served
from another host.

```http
GET /api/health

POST /api/search
Content-Type: application/json

{ "query": "hybrid index for newspaper archives", "top_k": 8 }
```

`top_k` is 5–10 (default 8). Each hit includes `rank`, `score`, `text`, `file`,
`page`, `page_label`, and `id`. Express should proxy:

```js
const r = await fetch("http://127.0.0.1:8000/api/search", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query, top_k: 8 }),
});
```

## MySQL text search

PDF search stays on `source/`. A second corpus is a local MySQL table with
only `title` and `content` (plus an `id` and `updated_at` used for sync).
This Mac already has Homebrew MySQL 9 listening on `127.0.0.1:3306`. Create
the app database, user, table, and 12 sample rows:

```sh
.venv/bin/python -m pip install -r requirements-db.txt
.venv/bin/python text_db.py --init
.venv/bin/python search_db.py --sync
.venv/bin/python search_db.py "kelp fiber winter power"
```

Defaults: host `127.0.0.1`, database/user/password `docsearch`. Override with
`MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`.
Admin bootstrap uses `MYSQL_ADMIN_USER` / `MYSQL_ADMIN_PASSWORD` (root, empty
password on this machine).

The Streamlit **MySQL text** tab and `POST /api/search/db` use the same local
Qwen model. New or edited rows become searchable on the next 2-second poll.

```http
POST /api/search/db
Content-Type: application/json

{ "query": "kelp fiber winter power", "top_k": 8 }
```

Hits are `{ rank, score, id, title, content }`. Add or remove rows without
rebuilding:

```sh
.venv/bin/python text_db.py --add --title "New note" --content "Plain text only."
.venv/bin/python text_db.py --list
```

Optional: `docker compose up -d` if you would rather run MySQL in a container
than use the Homebrew server. Do not start both on port 3306.
