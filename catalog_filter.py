"""Step 1.5: clean catalog sentences and drop ones that should not be embedded."""

import re

FILTER_SPEC = {
    "min_chars": 32,
    "min_letters": 16,
    "min_englishish_words": 2,
    "min_cjk_chars": 8,
    "dehyphenate": True,
    "drop_toc_dots": True,
}

HYPHEN_BREAK = re.compile(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])")
TOC_DOTS = re.compile(r"\.{4,}|(?:\.\s*){4,}|…{2,}")
CJK_CHAR = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
VOWEL = re.compile(r"[aeiouyAEIOUY]")
URL = re.compile(r"https?://", re.I)
ALNUM_FLIP = re.compile(r"[A-Za-z]\d|\d[A-Za-z]")
LONG_CONSONANT = re.compile(r"[bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ]{6,}")
COMMON_PUNCT = set(".,;:!?()[]'\"-%/$€£")


def clean_sentence(text):
    text = (text or "").replace("\u00ad", "")
    if FILTER_SPEC["dehyphenate"]:
        text = HYPHEN_BREAK.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _letters(text):
    return sum(c.isalpha() for c in text)


def _englishish(token):
    core = "".join(c for c in token if c.isalpha())
    if len(core) < 3 or any(c.isdigit() for c in token) or not VOWEL.search(core):
        return False
    if not core.isupper() and sum(c.isupper() for c in core[1:]) >= 2:
        return False
    return True


def _letter_spaced(text):
    tokens = text.split()
    if len(tokens) < 8:
        return False
    singles = 0
    for token in tokens:
        letters = "".join(c for c in token if c.isalpha())
        if len(letters) == 1:
            singles += 1
    return singles / len(tokens) >= 0.5


def _garbled(text):
    if URL.search(text) or len(CJK_CHAR.findall(text)) >= FILTER_SPEC["min_cjk_chars"]:
        return False
    if sum(1 for token in text.split() if _englishish(token)) < FILTER_SPEC["min_englishish_words"]:
        return True
    if _letter_spaced(text):
        return True
    n = len(text)
    flips = len(ALNUM_FLIP.findall(text))
    weird = sum(
        1 for c in text
        if not c.isalnum() and not c.isspace() and c not in COMMON_PUNCT
    )
    words = re.findall(r"[A-Za-z]{3,}", text)
    mixed_case = 0
    if words:
        mixed_case = sum(1 for word in words if sum(c.isupper() for c in word[1:]) >= 2)
        mixed_case /= len(words)
    score = 0
    if flips >= 8 or (n > 80 and flips / n > 0.04):
        score += 2
    if n and weird / n > 0.04:
        score += 1
    if mixed_case > 0.35:
        score += 2
    if n > 80 and text.count(" ") / n < 0.08:
        score += 1
    if LONG_CONSONANT.search(text):
        score += 1
    return score >= 3


def reject_reason(text):
    """Return a drop reason, or None if the cleaned sentence should be indexed."""
    if len(text) < FILTER_SPEC["min_chars"]:
        return "short"
    if _letters(text) < FILTER_SPEC["min_letters"]:
        return "few_letters"
    if FILTER_SPEC["drop_toc_dots"] and TOC_DOTS.search(text):
        return "toc"
    if _garbled(text):
        return "garbled"
    return None


def accepted_record(record):
    text = clean_sentence(record.get("text", ""))
    reason = reject_reason(text)
    if reason:
        return None, reason
    return {
        "id": record["id"],
        "text": text,
        "file": record["file"],
        "page": record["page"],
        "page_label": record.get("page_label"),
    }, None
