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
