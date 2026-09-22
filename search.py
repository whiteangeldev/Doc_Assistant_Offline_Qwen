"""Step 3: encode a query and rank the local sentence index by cosine similarity."""

import argparse
import json
from pathlib import Path

from offline_runtime import (
    DEFAULT_MODEL,
    EMBEDDING_DIM,
    MAX_SEQ_LENGTH,
    QUERY_PROMPT_NAME,
    load_embedder,
)

ROOT = Path(__file__).resolve().parent


def load_index(index_dir):
    config_path = index_dir / "index_config.json"
    vectors_path = index_dir / "embeddings.npy"
    meta_path = index_dir / "index_meta.jsonl"
    missing = [str(path) for path in (config_path, vectors_path, meta_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Search index is incomplete. Run build_index.py first. Missing: "
            + ", ".join(missing)
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("partial"):
        raise ValueError(
            "Index is a partial --limit build. Run build_index.py without --limit."
        )
    import numpy as np
    vectors = np.load(vectors_path, mmap_mode="r")
    meta = []
    with meta_path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                meta.append(json.loads(line))
    if vectors.ndim != 2 or vectors.shape[1] != config.get("embedding_dim", EMBEDDING_DIM):
        raise ValueError(f"Unexpected embedding shape {vectors.shape}")
    if vectors.shape[0] != len(meta):
        raise ValueError(
            f"Index rows do not align: {vectors.shape[0]} vectors vs {len(meta)} metadata lines."
        )
    if config.get("sentence_count") not in (None, len(meta)):
        raise ValueError("index_config.json sentence_count does not match the matrix.")
    return vectors, meta, config


def encode_query(model, query, prompt_name):
    model.max_seq_length = MAX_SEQ_LENGTH
    vector = model.encode(
        [query],
        prompt_name=prompt_name,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector[0]


def rank_hits(query_vector, vectors, meta, top_k):
    import numpy as np

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
            "text": row["text"],
            "file": row["file"],
            "page": row["page"],
            "page_label": row.get("page_label"),
            "id": row.get("id"),
        })
    return hits


def search(query, index_dir, model_path, top_k=8, device=None):
    import numpy as np

    query = (query or "").strip()
    if not query:
        raise ValueError("Query is empty.")
    vectors, meta, config = load_index(index_dir)
    prompt_name = config.get("query_prompt_name") or QUERY_PROMPT_NAME
    model, resolved_device, _ = load_embedder(model_path, device)
    query_vector = np.asarray(encode_query(model, query, prompt_name), dtype=np.float32)
    return rank_hits(query_vector, vectors, meta, top_k), resolved_device


def format_hit(hit):
    return (
        f"{hit['score']:.3f} | {hit['file']} page {hit['page']}\n"
        f"{hit['text']}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="+", help="Natural-language query")
    parser.add_argument("--index", type=Path, default=ROOT / "data")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="cpu, mps, or cuda (default: auto)")
    parser.add_argument("--top-k", type=int, default=8, dest="top_k",
                        help="Number of hits to show (5–10, default 8)")
    args = parser.parse_args()
    if args.top_k < 5 or args.top_k > 10:
        parser.error("--top-k must be between 5 and 10")
    try:
        hits, device = search(
            " ".join(args.query), args.index, args.model,
            top_k=args.top_k, device=args.device,
        )
    except (ValueError, OSError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Top {len(hits)} hits ({device}):")
    print()
    print("\n\n".join(format_hit(hit) for hit in hits))


if __name__ == "__main__":
    main()
