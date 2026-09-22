# Offline PDF semantic search

Goal: enter a natural-language question or sentence and retrieve the best 5–10
matching sentences, with similarity scores, PDF filenames, and page numbers,
using the local Qwen3-Embedding-0.6B model.

## Build sequence

1. **PDF sentence catalog (implemented).** Extract text and preserve citations.
2. **Embedding index (next).** Encode sentences with the local Qwen model and
   persist normalized vectors together with their catalog IDs.
3. **Search.** Encode a query, rank by cosine similarity, and display 5–10 hits.

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
`test_model.py` remains the local embedding smoke test. No embedding index or
search command is implemented yet.
