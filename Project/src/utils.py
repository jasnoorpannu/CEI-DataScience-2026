from __future__ import annotations

import re
from typing import Iterable

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE_PATTERN = re.compile(r"\b\+?\d[\d\s().-]{7,}\d\b")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")

SKILL_SEPARATORS = re.compile(r"[,;/|•]+|\s+and\s+")

STOPWORDS = set(
    """a an the and or but if then else for while of to in on at by with from as is are was were be been being
    have has had do does did will would shall should may might must can could not no nor so too very just
    it its this that these those he she they them their his her i you we our your us me my
    about above after again against all am any because before below between both each few more most other
    some such than through under until up when where why how what which who whom whose
    experience skills education summary candidate resume job position company work years month months
    including etc per via using use used working knowledge strong good great including""".split()
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def clean_text(text: str | None) -> str:
    text = (text or "").lower()
    text = _URL_PATTERN.sub(" ", text)
    text = _EMAIL_PATTERN.sub(" ", text)
    text = _PHONE_PATTERN.sub(" ", text)
    text = _NON_ALNUM.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str | None) -> str:
    text = (text or "").lower()
    text = _URL_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str | None) -> list[str]:
    return [tok for tok in clean_text(text).split() if tok not in STOPWORDS and len(tok) > 1]


def sentence_split(text: str | None) -> list[str]:
    parts = _SENTENCE_SPLIT.split((text or "").strip())
    out = []
    for part in parts:
        part = " ".join(part.split())
        if len(part) >= 8:
            out.append(part)
    return out


def chunks(text: str | None, size: int = 3) -> list[str]:
    sents = sentence_split(text)
    if not sents:
        return []
    return [" ".join(sents[i : i + size]) for i in range(0, len(sents), size)]


def top_n(items: Iterable[tuple[str, float]], n: int) -> list[tuple[str, float]]:
    return sorted(items, key=lambda x: x[1], reverse=True)[:n]
