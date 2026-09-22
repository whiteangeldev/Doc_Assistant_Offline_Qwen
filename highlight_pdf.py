"""Stamp a yellow highlight on the matching sentence and save a local copy."""

import re
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent
HIGHLIGHT_DIR = ROOT / "data" / "highlights"


def _snippets(text):
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    words = text.split()
    snippets = [text]
    for size in (12, 8, 5):
        if len(words) < size:
            continue
        step = max(1, size // 2)
        for start in range(0, len(words) - size + 1, step):
            snippets.append(" ".join(words[start:start + size]))
        break
    return snippets


def _dedupe(quads):
    seen = set()
    unique = []
    for quad in quads:
        key = tuple(round(value, 1) for value in quad.rect)
        if key not in seen:
            seen.add(key)
            unique.append(quad)
    return unique


def find_quads(page, text):
    flags = pymupdf.TEXTFLAGS_SEARCH
    for snippet in _snippets(text):
        if len(snippet) < 8:
            continue
        found = page.search_for(snippet, flags=flags, quads=True)
        if found:
            return _dedupe(found)
    return []


def highlight_hit(source, page, text, hit_id):
    HIGHLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", hit_id)
    destination = HIGHLIGHT_DIR / f"{safe}.pdf"
    document = pymupdf.open(source)
    index = int(page) - 1
    if index < 0 or index >= document.page_count:
        document.close()
        raise ValueError(f"Page {page} is out of range for {source}")
    pdf_page = document[index]
    quads = find_quads(pdf_page, text)
    if quads:
        pdf_page.add_highlight_annot(quads)
    document.save(destination, garbage=3, deflate=True)
    document.close()
    return destination, len(quads)
