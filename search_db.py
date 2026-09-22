"""Semantic search over the local MySQL title/content table."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pymysql

from build_index import encode_texts, write_json, write_jsonl
from offline_runtime import (
    DEFAULT_MODEL,
    EMBEDDING_DIM,
    QUERY_PROMPT_NAME,
    load_embedder,
)
from search import encode_query
from text_db import (
    connect,
    embed_text,
    fetch_records,
    init_db,
    row_digest,
    table_snapshot,
)

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "data"
VECTORS_NAME = "db_embeddings.npy"
META_NAME = "db_index_meta.jsonl"
CONFIG_NAME = "db_index_config.json"


def _paths(output):
    output = Path(output)
    return (
        output / VECTORS_NAME,
        output / META_NAME,
        output / CONFIG_NAME,
    )


def _empty_vectors():
    return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)


def load_db_index(output=None):
    output = Path(output or INDEX_DIR)
    vectors_path, meta_path, config_path = _paths(output)
    if not (vectors_path.is_file() and meta_path.is_file()):
        return _empty_vectors(), [], {}
    vectors = np.asarray(np.load(vectors_path), dtype=np.float32)
    meta = []
    with meta_path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                meta.append(json.loads(line))
    config = {}
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    if vectors.shape[0] != len(meta):
        raise ValueError(
            f"DB index rows do not align: {vectors.shape[0]} vectors vs {len(meta)} metadata lines."
        )
    return vectors, meta, config


def rank_db_hits(query_vector, vectors, meta, top_k):
    if vectors is None or getattr(vectors, "shape", (0,))[0] == 0 or not meta:
        return []
    scores = vectors @ query_vector
    count = min(top_k, scores.shape[0])
    order = np.argpartition(-scores, count - 1)[:count]
    order = order[np.argsort(-scores[order], kind="stable")]
    hits = []
    for rank, index in enumerate(order, start=1):
        row = meta[int(index)]
        hits.append({
            "rank": rank,
            "score": float(scores[int(index)]),
            "id": row["id"],
            "title": row["title"],
            "content": row["content"],
        })
    return hits


def _write_index(output, vectors, meta, model_path, device):
    vectors_path, meta_path, config_path = _paths(output)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / (VECTORS_NAME + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, vectors)
    temporary.replace(vectors_path)
    write_jsonl(meta_path, meta)
    write_json(config_path, {
        "source": "mysql",
        "model_path": str(Path(model_path).resolve()),
        "embedding_dim": EMBEDDING_DIM,
        "normalize": True,
        "query_prompt_name": QUERY_PROMPT_NAME,
        "device": device,
        "row_count": len(meta),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })


def sync_text_db(output=None, model=None, device="cpu", model_path=None, batch_size=8):
    """Embed new/changed MySQL rows. Returns (changed, payload)."""
    output = Path(output or INDEX_DIR)
    model_path = Path(model_path or DEFAULT_MODEL)
    init_db(seed=True)
    conn = connect()
    try:
        records = fetch_records(conn)
    finally:
        conn.close()

    vectors, meta, config = load_db_index(output)
    indexed = {row["id"]: row for row in meta}
    present = {}
    for record in records:
        present[record["id"]] = {
            "id": record["id"],
            "title": record["title"],
            "content": record["content"],
            "sha256": row_digest(record["title"], record["content"]),
        }

    removed = [row_id for row_id in indexed if row_id not in present]
    added = [
        row for row_id, row in present.items()
        if row_id not in indexed or indexed[row_id].get("sha256") != row["sha256"]
    ]
    stale = set(removed)
    stale.update(row["id"] for row in added)

    if not removed and not added:
        return False, {
            "added": [],
            "removed": [],
            "vectors": vectors,
            "meta": meta,
            "config": config or None,
            "row_count": len(meta),
        }

    if stale:
        keep_idx = [i for i, row in enumerate(meta) if row["id"] not in stale]
        meta = [meta[i] for i in keep_idx]
        vectors = vectors[keep_idx] if keep_idx else _empty_vectors()

    if added:
        if model is None:
            raise ValueError("A loaded embedding model is required to index new rows.")
        texts = [embed_text(row["title"], row["content"]) for row in added]
        extra = encode_texts(model, texts, batch_size, device)
        vectors = extra if vectors.shape[0] == 0 else np.vstack([vectors, extra])
        meta.extend(added)
        print(
            f"Indexed MySQL rows {', '.join(str(row['id']) for row in added)} "
            f"({len(added)} records)",
            flush=True,
        )

    _write_index(output, vectors, meta, model_path, device)
    config = json.loads((output / CONFIG_NAME).read_text(encoding="utf-8"))
    return True, {
        "added": [row["id"] for row in added],
        "removed": removed,
        "vectors": vectors,
        "meta": meta,
        "config": config,
        "row_count": len(meta),
    }


class TextDbWatcher:
    def __init__(self, output=None, interval=2.0):
        self.output = Path(output or INDEX_DIR)
        self.interval = interval
        self._snapshot = None

    def _snap(self):
        conn = connect()
        try:
            return table_snapshot(conn)
        finally:
            conn.close()

    def poll(self, model, device, model_path=None):
        snap = self._snap()
        if snap == self._snapshot and snap is not None:
            return False, None
        changed, payload = sync_text_db(
            self.output, model=model, device=device, model_path=model_path,
        )
        self._snapshot = self._snap()
        return changed, payload


def search_db(query, output=None, model=None, model_path=None, device=None, top_k=8):
    query = (query or "").strip()
    if not query:
        raise ValueError("Query is empty.")
    if model is None:
        model, device, model_path = load_embedder(model_path, device)
    watcher = TextDbWatcher(output)
    _changed, payload = watcher.poll(model, device, model_path)
    if payload is None:
        vectors, meta, config = load_db_index(output)
    else:
        vectors, meta, config = payload["vectors"], payload["meta"], payload["config"] or {}
    prompt_name = (config or {}).get("query_prompt_name") or QUERY_PROMPT_NAME
    query_vector = np.asarray(encode_query(model, query, prompt_name), dtype=np.float32)
    return rank_db_hits(query_vector, vectors, meta, top_k), device


def format_hit(hit):
    return (
        f"{hit['score']:.3f} | #{hit['id']} {hit['title']}\n"
        f"{hit['content']}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="Natural-language query")
    parser.add_argument("--output", type=Path, default=INDEX_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=8, dest="top_k")
    parser.add_argument("--sync", action="store_true", help="Index the table without searching")
    args = parser.parse_args()
    if args.top_k < 5 or args.top_k > 10:
        parser.error("--top-k must be between 5 and 10")
    try:
        if args.sync or not args.query:
            model, device, resolved = load_embedder(args.model, args.device)
            changed, payload = sync_text_db(
                args.output, model=model, device=device, model_path=resolved,
            )
            state = "updated" if changed else "current"
            print(f"MySQL index {state}: {payload['row_count']} rows ({device}).")
            if not args.query:
                return
            hits, device = search_db(
                " ".join(args.query), args.output, model=model,
                model_path=resolved, device=device, top_k=args.top_k,
            )
        else:
            hits, device = search_db(
                " ".join(args.query), args.output,
                model_path=args.model, device=args.device, top_k=args.top_k,
            )
    except (ValueError, OSError, FileNotFoundError, RuntimeError, pymysql.Error) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Top {len(hits)} MySQL hits ({device}):")
    print()
    print("\n\n".join(format_hit(hit) for hit in hits))


if __name__ == "__main__":
    main()
