"""Step 1: build an offline, page-addressable sentence catalog from PDFs."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from pypdf import PdfReader
import pysbd

ROOT = Path(__file__).resolve().parent


def sentences(text, segmenter):
    # Join visual line wraps before sentence segmentation. Preserve punctuation.
    text = re.sub(r"\s+", " ", text.replace("\u00ad", "")).strip()
    return [s.strip() for s in segmenter.segment(text) if any(c.isalpha() for c in s)]


def prepare(source, output, language="en"):
    segmenter = pysbd.Segmenter(language=language, clean=False)
    files = sorted(p for p in source.rglob("*") if p.suffix.lower() == ".pdf")
    if not files:
        raise ValueError(f"No PDFs found in {source}")
    records = []
    report = {"source": str(source.resolve()), "language": language,
              "documents": [], "duplicates": [], "errors": []}
    seen = {}
    for path in files:
        relative = path.relative_to(source).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            report["duplicates"].append({"file": relative, "same_as": seen[digest]})
            continue
        try:
            reader = PdfReader(path)
            if reader.is_encrypted and not reader.decrypt(""):
                raise ValueError("Password-protected PDF")
            labels = reader.page_labels
            info = {"file": relative, "sha256": digest, "pages": len(reader.pages),
                    "sentences": 0, "empty_pages": []}
            for number, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                    if not text.strip():
                        info["empty_pages"].append(number)
                        continue
                    for index, sentence in enumerate(sentences(text, segmenter)):
                        records.append({
                            "id": f"{digest}:{number}:{index}",
                            "text": sentence, "file": relative,
                            "page": number, "page_label": labels[number - 1],
                        })
                        info["sentences"] += 1
                except Exception as exc:
                    report["errors"].append({"file": relative, "page": number,
                                             "error": str(exc)})
            seen[digest] = relative
            report["documents"].append(info)
            print(f"{relative}: {info['pages']} pages, {info['sentences']} sentences", flush=True)
        except Exception as exc:
            report["errors"].append({"file": relative, "error": str(exc)})
    output.mkdir(parents=True, exist_ok=True)
    catalog = output / "sentences.jsonl"
    temporary = output / "sentences.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(catalog)
    report["sentence_count"] = len(records)
    (output / "extraction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} sentences to {catalog}")
    print(f"Skipped {len(report['duplicates'])} identical PDFs; {len(report['errors'])} errors.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "source")
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument("--language", default="en", help="pySBD language code (default: en)")
    args = parser.parse_args()
    try:
        report = prepare(args.source, args.output, args.language)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    if report["errors"] or not report["sentence_count"]:
        parser.exit(1, "Extraction incomplete; inspect extraction_report.json.\n")


if __name__ == "__main__":
    main()
