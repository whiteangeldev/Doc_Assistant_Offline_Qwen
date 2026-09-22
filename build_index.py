"""Step 2: embed the filtered sentence catalog into a local search index."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from catalog_filter import FILTER_SPEC, accepted_record
from offline_runtime import (
    DEFAULT_MODEL,
    EMBEDDING_DIM,
    MAX_SEQ_LENGTH,
    QUERY_PROMPT_NAME,
)

ROOT = Path(__file__).resolve().parent


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(catalog_sha256, model_path, limit):
    payload = {
        "catalog_sha256": catalog_sha256,
        "model_path": str(Path(model_path).resolve()),
        "filter": FILTER_SPEC,
        "embedding_dim": EMBEDDING_DIM,
        "max_seq_length": MAX_SEQ_LENGTH,
        "normalize": True,
        "query_prompt_name": QUERY_PROMPT_NAME,
        "limit": limit,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def read_catalog(path):
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def filter_catalog(records, limit=None):
    kept = []
    reasons = Counter()
    per_file = defaultdict(lambda: Counter())
    for record in records:
        accepted, reason = accepted_record(record)
        name = record.get("file", "?")
        per_file[name]["input"] += 1
        if reason:
            reasons[reason] += 1
            per_file[name][reason] += 1
            continue
        per_file[name]["kept"] += 1
        kept.append(accepted)
        if limit and len(kept) >= limit:
            break
    return kept, reasons, per_file


def write_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def index_is_current(output, mark):
    config_path = output / "index_config.json"
    vectors = output / "embeddings.npy"
    meta = output / "index_meta.jsonl"
    if not (config_path.is_file() and vectors.is_file() and meta.is_file()):
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return config.get("fingerprint") == mark


def encode_texts(model, texts, batch_size, device):
    import numpy as np
    import torch

    model.max_seq_length = MAX_SEQ_LENGTH
    pieces = []
    step = 256
    current_batch = max(1, batch_size)
    index = 0
    while index < len(texts):
        chunk = texts[index:index + step]
        print(f"Embedding {index + 1}-{index + len(chunk)} / {len(texts)} "
              f"(batch_size={current_batch})", flush=True)
        try:
            vectors = model.encode(
                chunk,
                batch_size=current_batch,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            pieces.append(np.asarray(vectors, dtype=np.float32))
            index += len(chunk)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or current_batch <= 1:
                raise
            current_batch = max(1, current_batch // 2)
            print(f"Memory pressure; retrying chunk at batch_size={current_batch}", flush=True)
            if device == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
    return np.vstack(pieces)


def build_index(catalog, output, model_path, device=None, batch_size=8,
                limit=None, dry_run=False, force=False):
    if not catalog.is_file():
        raise FileNotFoundError(f"Sentence catalog not found: {catalog}")
    output.mkdir(parents=True, exist_ok=True)
    catalog_sha = sha256_file(catalog)
    records = read_catalog(catalog)
    if not records:
        raise ValueError(f"Catalog is empty: {catalog}")
    kept, reasons, per_file = filter_catalog(records, limit=limit)
    report = {
        "catalog": str(catalog.resolve()),
        "catalog_sha256": catalog_sha,
        "source_sentence_count": len(records),
        "kept_sentence_count": len(kept),
        "dropped": dict(reasons),
        "documents": [],
        "partial": bool(limit),
        "limit": limit,
    }
    for name, counts in per_file.items():
        info = {"file": name, "input": counts["input"], "kept": counts["kept"],
                "dropped": {key: counts[key] for key in ("short", "few_letters", "toc", "garbled")
                            if counts[key]}}
        report["documents"].append(info)
        dropped = counts["input"] - counts["kept"]
        print(f"{name}: {counts['kept']} kept / {counts['input']} "
              f"({dropped} dropped)", flush=True)
    print(f"Kept {len(kept)} of {len(records)} sentences "
          f"(dropped {dict(reasons) or 0}).", flush=True)
    write_json(output / "index_report.json", report)
    if dry_run:
        return report
    if not kept:
        raise ValueError("No sentences survived filtering; inspect index_report.json.")

    mark = fingerprint(catalog_sha, model_path, limit)
    if not force and index_is_current(output, mark):
        print(f"Index already current at {output}; skipping encode.")
        return report

    from offline_runtime import load_embedder
    import numpy as np

    model, resolved_device, resolved_model = load_embedder(model_path, device)
    print(f"Encoding {len(kept)} sentences on {resolved_device} "
          f"(batch_size={batch_size}, max_seq_length={MAX_SEQ_LENGTH})...", flush=True)
    vectors = encode_texts(
        model, [row["text"] for row in kept], batch_size, resolved_device,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(kept):
        raise ValueError(f"Unexpected embedding shape {vectors.shape}")
    if vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM} dimensions, got {vectors.shape[1]}")

    temporary = output / "embeddings.npy.tmp"
    with temporary.open("wb") as handle:
        np.save(handle, vectors)
    temporary.replace(output / "embeddings.npy")
    write_jsonl(output / "index_meta.jsonl", kept)
    write_json(output / "index_config.json", {
        "fingerprint": mark,
        "model_path": str(resolved_model),
        "embedding_dim": EMBEDDING_DIM,
        "max_seq_length": MAX_SEQ_LENGTH,
        "normalize": True,
        "query_prompt_name": QUERY_PROMPT_NAME,
        "device": resolved_device,
        "batch_size": batch_size,
        "catalog": str(catalog.resolve()),
        "catalog_sha256": catalog_sha,
        "source_sentence_count": len(records),
        "sentence_count": len(kept),
        "filter": FILTER_SPEC,
        "partial": bool(limit),
        "limit": limit,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    print(f"Saved {vectors.shape[0]} x {vectors.shape[1]} index to {output / 'embeddings.npy'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "sentences.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="cpu, mps, or cuda (default: auto)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Embed only the first N accepted sentences")
    parser.add_argument("--dry-run", action="store_true",
                        help="Filter and report only; do not load the model")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the index is current")
    args = parser.parse_args()
    try:
        report = build_index(
            args.catalog, args.output, args.model, device=args.device,
            batch_size=args.batch_size, limit=args.limit, dry_run=args.dry_run,
            force=args.force,
        )
    except (ValueError, OSError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    if not report["kept_sentence_count"]:
        parser.exit(1, "No sentences to index; inspect index_report.json.\n")


if __name__ == "__main__":
    main()
