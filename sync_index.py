"""Incrementally extract and embed PDFs when source/ changes."""

import argparse
import fcntl
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pysbd

from build_index import (
    encode_texts,
    fingerprint,
    sha256_file,
    write_json,
    write_jsonl,
)
from catalog_filter import FILTER_SPEC, accepted_record
from offline_runtime import (
    DEFAULT_MODEL,
    EMBEDDING_DIM,
    MAX_SEQ_LENGTH,
    QUERY_PROMPT_NAME,
    load_embedder,
)
from prepare_documents import extract_pdf

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
INDEX_DIR = ROOT / "data"


def _pdfs(source):
    return sorted(p for p in source.rglob("*") if p.suffix.lower() == ".pdf")


def _snapshot(source):
    rows = []
    for path in _pdfs(source):
        stat = path.stat()
        rows.append((path.relative_to(source).as_posix(), stat.st_size, stat.st_mtime_ns))
    return tuple(rows)


def _stable_paths(source, wait=0.5):
    first = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in _pdfs(source)}
    time.sleep(wait)
    stable = []
    for path in _pdfs(source):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        current = (path.stat().st_size, path.stat().st_mtime_ns)
        if first.get(path) == current:
            stable.append(path)
    return stable


def _load_report(output):
    path = output / "extraction_report.json"
    if not path.is_file():
        return {"source": "", "language": "en", "documents": [], "duplicates": [],
                "errors": [], "sentence_count": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path):
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _digest_of(row):
    return (row.get("id") or "").split(":", 1)[0]


def _keep(records):
    kept = []
    for record in records:
        accepted, _reason = accepted_record(record)
        if accepted:
            kept.append(accepted)
    return kept


@contextmanager
def _index_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    handle = (output / ".index.lock").open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def _write_config(output, catalog, model_path, device, source_count, kept_count):
    catalog_sha = sha256_file(catalog) if catalog.is_file() else ""
    resolved = Path(model_path).resolve()
    write_json(output / "index_config.json", {
        "fingerprint": fingerprint(catalog_sha, resolved, None),
        "model_path": str(resolved),
        "embedding_dim": EMBEDDING_DIM,
        "max_seq_length": MAX_SEQ_LENGTH,
        "normalize": True,
        "query_prompt_name": QUERY_PROMPT_NAME,
        "device": device,
        "catalog": str(catalog.resolve()) if catalog.is_file() else "",
        "catalog_sha256": catalog_sha,
        "source_sentence_count": source_count,
        "sentence_count": kept_count,
        "filter": FILTER_SPEC,
        "partial": False,
        "limit": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })


def sync_once(source=None, output=None, model=None, device="cpu",
              model_path=None, language="en", batch_size=8):
    """Apply source/ changes to the on-disk index. Returns (changed, payload)."""
    source = Path(source or SOURCE_DIR)
    output = Path(output or INDEX_DIR)
    model_path = Path(model_path or DEFAULT_MODEL)
    output.mkdir(parents=True, exist_ok=True)

    with _index_lock(output):
        return _sync_once_locked(
            source, output, model, device, model_path, language, batch_size,
        )


def _sync_once_locked(source, output, model, device, model_path, language, batch_size):
    report = _load_report(output)
    indexed = {doc["sha256"]: doc for doc in report.get("documents", [])}
    catalog = _read_jsonl(output / "sentences.jsonl")
    meta = _read_jsonl(output / "index_meta.jsonl")
    vectors_path = output / "embeddings.npy"
    if vectors_path.is_file() and meta:
        vectors = np.asarray(np.load(vectors_path), dtype=np.float32)
    else:
        vectors = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    if vectors.shape[0] != len(meta):
        raise ValueError(
            f"Index rows do not align: {vectors.shape[0]} vectors vs {len(meta)} metadata lines."
        )

    seen = {}
    present = []
    duplicates = []
    for path in _stable_paths(source):
        relative = path.relative_to(source).as_posix()
        digest = sha256_file(path)
        if digest in seen:
            duplicates.append({"file": relative, "same_as": seen[digest]})
            continue
        seen[digest] = relative
        present.append((relative, digest, path))

    present_hashes = {digest for _rel, digest, _path in present}
    removed = [digest for digest in indexed if digest not in present_hashes]
    added = [(rel, digest, path) for rel, digest, path in present if digest not in indexed]
    renamed = [(rel, digest) for rel, digest, _path in present
               if digest in indexed and indexed[digest]["file"] != rel]

    if not removed and not added and not renamed:
        return False, {
            "added": [], "removed": [], "renamed": [],
            "vectors": vectors, "meta": meta, "config": None,
            "sentence_count": len(meta),
        }

    if removed:
        remove = set(removed)
        catalog = [row for row in catalog if _digest_of(row) not in remove]
        keep_idx = [i for i, row in enumerate(meta) if _digest_of(row) not in remove]
        meta = [meta[i] for i in keep_idx]
        vectors = vectors[keep_idx] if keep_idx else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        report["documents"] = [doc for doc in report["documents"] if doc["sha256"] not in remove]

    if renamed:
        mapping = {digest: rel for rel, digest in renamed}
        for row in catalog:
            digest = _digest_of(row)
            if digest in mapping:
                row["file"] = mapping[digest]
        for row in meta:
            digest = _digest_of(row)
            if digest in mapping:
                row["file"] = mapping[digest]
        for doc in report["documents"]:
            if doc["sha256"] in mapping:
                doc["file"] = mapping[doc["sha256"]]

    new_kept = []
    errors = list(report.get("errors") or [])
    segmenter = pysbd.Segmenter(language=language, clean=False) if added else None
    for relative, digest, path in added:
        try:
            _digest, records, info, page_errors = extract_pdf(path, source, segmenter)
            errors.extend(page_errors)
            catalog.extend(records)
            kept = _keep(records)
            new_kept.extend(kept)
            report["documents"].append(info)
            print(f"Indexed {relative}: {info['pages']} pages, {len(kept)} searchable sentences",
                  flush=True)
        except Exception as exc:
            errors.append({"file": relative, "error": str(exc)})
            print(f"Failed to index {relative}: {exc}", flush=True)

    if new_kept:
        if model is None:
            raise ValueError("A loaded embedding model is required to index new PDFs.")
        extra = encode_texts(
            model, [row["text"] for row in new_kept], batch_size, device,
        )
        vectors = extra if vectors.shape[0] == 0 else np.vstack([vectors, extra])
        meta.extend(new_kept)

    report["source"] = str(source.resolve())
    report["language"] = language
    report["duplicates"] = duplicates
    report["errors"] = errors
    report["sentence_count"] = len(catalog)

    catalog_path = output / "sentences.jsonl"
    write_jsonl(catalog_path, catalog)
    write_jsonl(output / "index_meta.jsonl", meta)
    temporary = output / "embeddings.npy.tmp"
    with temporary.open("wb") as handle:
        np.save(handle, vectors)
    temporary.replace(output / "embeddings.npy")
    write_json(output / "extraction_report.json", report)
    _write_config(output, catalog_path, model_path, device, len(catalog), len(meta))
    config = json.loads((output / "index_config.json").read_text(encoding="utf-8"))
    return True, {
        "added": [rel for rel, _digest, _path in added],
        "removed": [indexed[digest]["file"] for digest in removed],
        "renamed": [{"from": indexed[digest]["file"], "to": rel} for rel, digest in renamed],
        "vectors": vectors,
        "meta": meta,
        "config": config,
        "sentence_count": len(meta),
    }


class SourceWatcher:
    def __init__(self, source=None, output=None, interval=2.0):
        self.source = Path(source or SOURCE_DIR)
        self.output = Path(output or INDEX_DIR)
        self.interval = interval
        self._snapshot = None

    def poll(self, model, device, model_path=None):
        snap = _snapshot(self.source)
        if snap == self._snapshot:
            return False, None
        changed, payload = sync_once(
            self.source, self.output, model=model, device=device,
            model_path=model_path,
        )
        self._snapshot = snap
        return changed, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=INDEX_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    model, device, resolved = load_embedder(args.model, args.device)
    changed, payload = sync_once(
        args.source, args.output, model=model, device=device, model_path=resolved,
    )
    if not changed:
        print(f"Index already matches {args.source} ({payload['sentence_count']} sentences).")
        return
    print(
        f"Added {payload['added'] or '[]'}; removed {payload['removed'] or '[]'}; "
        f"{payload['sentence_count']} searchable sentences."
    )


if __name__ == "__main__":
    main()
