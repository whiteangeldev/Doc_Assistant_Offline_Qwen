"""Step 4: run known queries against the local index and check expected sources."""

import argparse
import json
from pathlib import Path

from offline_runtime import DEFAULT_MODEL, QUERY_PROMPT_NAME, load_embedder
from search import encode_query, load_index, rank_hits

ROOT = Path(__file__).resolve().parent

CASES = [
    {
        "id": "fed-stability",
        "query": "How does the Federal Reserve monitor financial stability?",
        "expect_file": "2024-annual-report.pdf",
        "expect_pages": [21, 22, 23, 27],
    },
    {
        "id": "nist-rmf",
        "query": "What are the NIST AI risk management functions?",
        "expect_file": "NIST.AI.100-1 (1).pdf",
    },
    {
        "id": "columbia",
        "query": "What caused the Space Shuttle Columbia accident?",
        "expect_file": "CAIBreportv1.pdf",
    },
    {
        "id": "alice",
        "query": "Alice falls down the rabbit hole in Wonderland",
        "expect_file": "Alice_in_Wonderland.pdf",
    },
    {
        "id": "fluke-warranty",
        "query": "How long is the Fluke 87V true-rms multimeter warranty?",
        "expect_file": "87vex___umeng0000.pdf",
        "expect_pages": [2],
    },
    {
        "id": "fomc-jan-2024",
        "query": "What did the FOMC discuss at the January 2024 meeting?",
        "expect_file": "fomcminutes20240131.pdf",
    },
    {
        "id": "irs-pub15",
        "query": "What is the employer social security tax rate in Publication 15?",
        "expect_file": "p15.pdf",
    },
    {
        "id": "osha-handbook",
        "query": "OSHA NIOSH small business safety and health handbook",
        "expect_file": "SMALL-BUSINESS.pdf",
    },
    {
        "id": "rare-event",
        "query": "Testing a battery management system with rare event simulation",
        "expect_file": "2107.00530v1 (1).pdf",
    },
    {
        "id": "archives-parade",
        "query": "National Archives film of the Washington parade and the Constitution",
        "expect_file": "transcript-archives-washington-parade-89855.pdf",
    },
    {
        "id": "dust-scan",
        "query": "What causes combustible dust explosions?",
        "expect_absent_file": "3371COMBUSTIBLE-DUST.pdf",
        "note": "That PDF is a scan and was not indexed.",
    },
    {
        "id": "fed-stability-zh",
        "query": "美联储如何监测金融稳定风险？",
        "expect_file": "2024-annual-report.pdf",
        "required": False,
        "note": "Cross-language check; failure is reported but not fatal.",
    },
]


def judge(case, hits):
    expect_file = case.get("expect_file")
    absent = case.get("expect_absent_file")
    pages = set(case.get("expect_pages") or [])
    matched = None
    if expect_file:
        matched = next((hit for hit in hits if hit["file"] == expect_file), None)
        if matched is None:
            return "FAIL", "expected file not in top hits", matched
        if pages and matched["page"] not in pages:
            return "WARN", f"file ok, page {matched['page']} not in {sorted(pages)}", matched
        return "PASS", "expected file in top hits", matched
    if absent:
        leaked = next((hit for hit in hits if hit["file"] == absent), None)
        if leaked:
            return "FAIL", f"unindexed file appeared: {absent}", leaked
        return "PASS", "unindexed scan did not appear", hits[0] if hits else None
    return "FAIL", "case has no expectation", None


def evaluate(index_dir, model_path, top_k, device=None):
    import numpy as np

    vectors, meta, config = load_index(index_dir)
    prompt_name = config.get("query_prompt_name") or QUERY_PROMPT_NAME
    model, resolved_device, _ = load_embedder(model_path, device)
    results = []
    for case in CASES:
        query_vector = np.asarray(
            encode_query(model, case["query"], prompt_name), dtype=np.float32,
        )
        hits = rank_hits(query_vector, vectors, meta, top_k)
        status, reason, matched = judge(case, hits)
        row = {
            "id": case["id"],
            "query": case["query"],
            "status": status,
            "reason": reason,
            "required": case.get("required", True),
            "note": case.get("note"),
            "expect_file": case.get("expect_file"),
            "expect_absent_file": case.get("expect_absent_file"),
            "top": {
                "score": hits[0]["score"],
                "file": hits[0]["file"],
                "page": hits[0]["page"],
                "text": hits[0]["text"],
            } if hits else None,
            "matched": {
                "rank": matched["rank"],
                "score": matched["score"],
                "file": matched["file"],
                "page": matched["page"],
                "text": matched["text"],
            } if matched else None,
        }
        results.append(row)
        pointer = row["matched"] or row["top"]
        where = f"{pointer['file']} page {pointer['page']}" if pointer else "no hits"
        score = f"{pointer['score']:.3f}" if pointer else "-"
        print(f"{status:4} {case['id']}: {score} | {where} ({reason})", flush=True)
    return results, resolved_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "data")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=8, dest="top_k")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "eval_report.json")
    args = parser.parse_args()
    if args.top_k < 5 or args.top_k > 10:
        parser.error("--top-k must be between 5 and 10")
    try:
        results, device = evaluate(args.index, args.model, args.top_k, args.device)
    except (ValueError, OSError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    counts = {label: sum(1 for row in results if row["status"] == label)
              for label in ("PASS", "WARN", "FAIL")}
    required_fail = [row["id"] for row in results
                     if row["status"] == "FAIL" and row["required"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "device": device,
        "top_k": args.top_k,
        "counts": counts,
        "required_failures": required_fail,
        "cases": results,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS {counts['PASS']}  WARN {counts['WARN']}  FAIL {counts['FAIL']}  ({device})")
    print(f"Wrote {args.output}")
    if required_fail:
        parser.exit(1, "Required cases failed: " + ", ".join(required_fail) + "\n")


if __name__ == "__main__":
    main()
