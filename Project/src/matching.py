from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd

from src import config
from src.utils import clean_text, tokenize

_ALIAS_SKILL = "alias_to_skill"


@lru_cache(maxsize=1)
def load_lexicon() -> dict[str, list[str]]:
    with open(config.SKILLS_LEXICON, encoding="utf-8") as fh:
        data = json.load(fh)
    return {str(k): [str(a) for a in v] for k, v in data.items()}


@lru_cache(maxsize=1)
def _patterns() -> dict[str, re.Pattern]:
    lex = load_lexicon()
    pats: dict[str, re.Pattern] = {}
    for skill, aliases in lex.items():
        for alias in aliases:
            escaped = re.escape(alias.strip())
            pats[f"{skill}::{alias}"] = re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")
    return pats


@lru_cache(maxsize=1)
def canonical_skills() -> list[str]:
    return sorted(load_lexicon().keys())


def extract_skills(text: str | None, aliases: bool = False) -> list[str]:
    if not text:
        return []
    lowered = (text or "").lower()
    found: set[str] = set()
    for key, pat in _patterns().items():
        skill, alias = key.split("::", 1)
        if pat.search(lowered):
            found.add(skill)
    return sorted(found)


def extract_skills_with_aliases(text: str | None) -> list[str]:
    if not text:
        return []
    lowered = (text or "").lower()
    found: set[str] = set()
    for key, pat in _patterns().items():
        alias = key.split("::", 1)[1]
        if pat.search(lowered):
            found.add(alias)
    return sorted(found)


def parse_structured_skills(skills_text: str | None) -> list[str]:
    if not skills_text:
        return []
    parts = config.SKILL_SEPARATORS.split(skills_text)
    tokens = []
    for part in parts:
        part = part.strip().lower()
        if part:
            tokens.append(part)
    return tokens


def skill_overlap(resume_skills: list[str], required_skills: list[str]) -> list[str]:
    rs = set(resume_skills)
    return [s for s in required_skills if s in rs]


def skill_gaps(resume_skills: list[str], required_skills: list[str]) -> list[str]:
    rs = set(resume_skills)
    return [s for s in required_skills if s not in rs]


def skill_coverage(resume_skills: list[str], required_skills: list[str]) -> float:
    if not required_skills:
        return 0.0
    return len(skill_overlap(resume_skills, required_skills)) / len(required_skills)


def top_skills_by_category(df: pd.DataFrame, top_n: int = 20) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for category, group in df.groupby("Category"):
        counter: Counter[str] = Counter()
        for text in group["Text"].fillna(""):
            for skill in extract_skills(text):
                counter[skill] += 1
        out[str(category)] = [skill for skill, _ in counter.most_common(top_n)]
    return out


@dataclass
class JobRequirements:
    source_text: str = ""
    skills: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.source_text.strip()


def parse_job_requirements(job_text: str | None) -> JobRequirements:
    text = (job_text or "").strip()
    req = JobRequirements(source_text=text)
    if not text:
        return req
    req.skills = extract_skills(text)
    tokens = tokenize(text)
    counts: dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    top = sorted(counts.items(), key=lambda x: (x[1], len(x[0])), reverse=True)[:20]
    req.keywords = [tok for tok, _ in top]
    return req


def generate_requirements_for_category(
    category: str,
    top_skills: dict[str, Sequence[str]],
    n_skills: int = 14,
) -> str:
    skills = list(top_skills.get(category, []))[:n_skills]
    lines = [
        f"We are looking for a skilled {category} to join our team.",
        "Responsibilities:",
        f"- Design, develop, and maintain high-quality solutions in {category}.",
        "- Collaborate with cross-functional teams and deliver on time.",
        "- Follow best practices, code reviews, and agile ceremonies.",
        "Required skills:",
    ]
    for skill in skills:
        lines.append(f"- {skill}")
    lines.append("- Strong problem-solving and communication skills.")
    return "\n".join(lines)


@dataclass
class MatchComponents:
    skill_coverage: float = 0.0
    embedding_similarity: float = 0.0
    category_affinity: float = 0.0
    weights_version: str = config.MATCH_WEIGHT_VERSION

    def overall(self, weights: dict[str, float] | None = None) -> float:
        weights = weights or config.MATCH_WEIGHTS
        raw = (
            weights["skill_coverage"] * self.skill_coverage
            + weights["embedding_similarity"] * self.embedding_similarity
            + weights["category_affinity"] * self.category_affinity
        )
        return max(0.0, min(100.0, round(raw * 100.0, 1)))


@dataclass
class MatchResult:
    resume_text: str = ""
    job_text: str = ""
    requirements: JobRequirements = field(default_factory=JobRequirements)
    resume_skills: list[str] = field(default_factory=list)
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    extra_skills: list[str] = field(default_factory=list)
    resume_categories: list[tuple[str, float]] = field(default_factory=list)
    job_categories: list[tuple[str, float]] = field(default_factory=list)
    components: MatchComponents = field(default_factory=MatchComponents)
    semantic_matches: list[dict] = field(default_factory=list)
    transferable_skills: list[tuple[str, str, float]] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        return self.components.overall()

    def grade(self) -> str:
        score = self.overall_score
        if score >= 80:
            return "Strong match"
        if score >= 60:
            return "Good match"
        if score >= 40:
            return "Partial match"
        return "Weak match"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def _category_affinity(
    resume_cats: Sequence[tuple[str, float]], job_cats: Sequence[tuple[str, float]]
) -> float:
    if not resume_cats or not job_cats:
        return 0.0
    resume = {cat: prob for cat, prob in resume_cats}
    job = {cat: prob for cat, prob in job_cats}
    classes = set(resume) | set(job)
    rvec = np.array([resume.get(c, 0.0) for c in sorted(classes)])
    jvec = np.array([job.get(c, 0.0) for c in sorted(classes)])
    return _cosine(rvec, jvec)


def match_resume_to_job(
    resume_text: str,
    job_text: str | None,
    classifier,
    embedder,
    top_k: int = 5,
    skill_extractor=None,
    weights: dict[str, float] | None = None,
    weights_version: str | None = None,
) -> MatchResult:
    resume_text = (resume_text or "").strip()
    if not resume_text:
        raise ValueError("Resume text is required for matching.")

    requirements = parse_job_requirements(job_text)
    resume_skills = extract_skills(resume_text)
    job_skills = requirements.skills

    semantic_matches: list[dict] = []
    transferable: list[tuple[str, str, float]] = []
    if skill_extractor is not None and job_skills:
        result = skill_extractor.match(resume_text, job_skills)
        semantic_matches = [m.to_dict() for m in result.all()]
        if hasattr(skill_extractor, "transferable"):
            transferable = skill_extractor.transferable(resume_skills, job_skills)

    matched = skill_overlap(resume_skills, job_skills)
    matched_semantic = {m["skill"] for m in semantic_matches}
    matched = list(dict.fromkeys(matched + sorted(matched_semantic)))
    missing = skill_gaps(matched, job_skills)
    extra = [s for s in resume_skills if s not in job_skills]

    coverage = skill_coverage(matched, job_skills) if job_skills else 0.0

    resume_cats = classifier.predict_categories([resume_text])[0][:top_k]
    job_cats = []
    if requirements.is_empty:
        affinity = 0.0
    else:
        job_cats_full = classifier.predict_categories([requirements.source_text])[0]
        job_cats = job_cats_full[:top_k]
        affinity = _category_affinity(resume_cats, job_cats_full)

    sim = 0.0
    if not requirements.is_empty:
        rvec = embedder.encode_one(resume_text)
        jvec = embedder.encode_one(requirements.source_text)
        sim = _cosine(rvec, jvec)

    components = MatchComponents(
        skill_coverage=coverage,
        embedding_similarity=sim,
        category_affinity=affinity,
        weights_version=weights_version or config.MATCH_WEIGHT_VERSION,
    )
    components.overall(weights=weights)

    return MatchResult(
        resume_text=resume_text,
        job_text=requirements.source_text,
        requirements=requirements,
        resume_skills=resume_skills,
        matched_skills=matched,
        missing_skills=missing,
        extra_skills=extra,
        resume_categories=resume_cats,
        job_categories=job_cats,
        components=components,
        semantic_matches=semantic_matches,
        transferable_skills=transferable,
    )
