from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.matching import extract_skills

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE = re.compile(r"(?<![0-9])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?![0-9])")
_LINK = re.compile(
    r"(?:linkedin\.com/in/[\w.-]+|github\.com/[\w.-]+|"
    r"(?:https?://)?(?:[\w-]+\.)+(?:com|io|dev|org|net|ai|in)/[^\s|]+)",
    re.I,
)
_YEARS = re.compile(r"(\d{1,2})(?:\s*\+)?\s*(?:years|yrs)\b", re.I)
_DEGREES = re.compile(
    r"\b(bachelor(?:'s|s)?|master(?:'s|s)?|ph\.?d|doctorate|b\.s\.?|m\.s\.?|b\.e\.?|m\.e\.?|"
    r"m\.tech|b\.tech|m\.ba|b\.a\.?|m\.a\.?|associate(?:'s|s)?|diploma)(?![a-z0-9])",
    re.I,
)
_HEADER_LINE = re.compile(r"@|linkedin\.com|github\.com|\|")
_SECTION_LABEL = re.compile(r"^\s*(education|skills|experience|summary|degree|certifications|qualification)\s*:\s*", re.I)


@dataclass
class CandidateProfile:
    email: str = ""
    phone: str = ""
    links: list[str] = field(default_factory=list)
    name: str = ""
    experience_years: float = 0.0
    education: list[str] = field(default_factory=list)
    top_skills: list[str] = field(default_factory=list)
    summary: str = ""

    def has_content(self) -> bool:
        return bool(
            self.email
            or self.phone
            or self.links
            or self.name
            or self.experience_years
            or self.education
            or self.top_skills
            or self.summary
        )


def extract_profile(text: str | None) -> CandidateProfile:
    text = (text or "").strip()
    profile = CandidateProfile()
    if not text:
        return profile
    email = _EMAIL.search(text)
    if email:
        profile.email = email.group(0)
    phone = _PHONE.search(text)
    if phone:
        profile.phone = phone.group(0)
    profile.links = list(dict.fromkeys(_LINK.findall(text)))
    profile.name = _guess_name(text, profile.email)
    profile.experience_years = _extract_years(text)
    profile.education = _extract_education(text)
    profile.top_skills = extract_skills(text)[:10]
    profile.summary = _extract_summary(text)
    return profile


def _guess_name(text: str, email: str = "") -> str:
    for line in text.splitlines()[:6]:
        line = line.strip().rstrip(".,;:|")
        if not line or _HEADER_LINE.search(line) or _PHONE.search(line) or _DEGREES.search(line):
            continue
        words = line.split()
        if 1 <= len(words) <= 4 and any(w[:1].isupper() for w in words):
            return line
    if email:
        local = email.split("@")[0].replace(".", " ").replace("_", " ").strip()
        return local.title() if local else ""
    return ""


def _extract_years(text: str) -> float:
    years = [int(m.group(1)) for m in _YEARS.finditer(text)]
    return float(max(years)) if years else 0.0


def _extract_education(text: str) -> list[str]:
    seen: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if _DEGREES.search(line):
            snippet = _SECTION_LABEL.sub("", " ".join(line.split()))[:140]
            if snippet not in seen:
                seen.append(snippet)
        if len(seen) >= 3:
            break
    return seen


def _extract_summary(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if not line or _HEADER_LINE.search(line) or _EMAIL.search(line) or _PHONE.search(line):
            continue
        if len(line.split()) >= 6:
            return line[:300]
    return ""
