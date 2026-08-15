from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

import numpy as np

from src import config
from src.evidence import sentence_for_term
from src.matching import extract_skills, load_lexicon
from src.utils import normalize_text, sentence_split

_SKILL_ALIASES = {"ml": "machine learning", "ai": "artificial intelligence", "mlops": "ml operations"}


@dataclass
class SemanticSkillMatch:
    skill: str
    matched_terms: list[str] = field(default_factory=list)
    score: float = 0.0
    evidence: str = ""
    method: str = "semantic"

    def to_dict(self) -> dict:
        return {
            "skill": self.skill,
            "matched_terms": self.matched_terms,
            "score": round(self.score, 3),
            "evidence": self.evidence,
            "method": self.method,
        }


@dataclass
class SemanticMatchResult:
    lexical: list[SemanticSkillMatch] = field(default_factory=list)
    semantic: list[SemanticSkillMatch] = field(default_factory=list)
    implicit: list[SemanticSkillMatch] = field(default_factory=list)

    def all(self) -> list[SemanticSkillMatch]:
        seen: set[str] = set()
        out = []
        for match in self.lexical + self.semantic + self.implicit:
            if match.skill not in seen:
                seen.add(match.skill)
                out.append(match)
        return out

    def matched_skills(self) -> list[str]:
        return [m.skill for m in self.all()]


class SemanticSkillExtractor:
    def __init__(self, embedder, threshold: float = config.SEMANTIC_SKILL_THRESHOLD) -> None:
        self.embedder = embedder
        self.threshold = threshold

    @lru_cache(maxsize=1)
    def _lexicon_skills(self) -> list[str]:
        return sorted(load_lexicon().keys())

    def _resume_index(self, text: str):
        sents = sentence_split(text)
        if not sents:
            return [], np.zeros((0, self.embedder.embedding_dim), dtype=np.float32)
        return sents, self.embedder.encode(sents)

    def _sentence_match(self, query: str, sents: list[str], vecs: np.ndarray) -> SemanticSkillMatch:
        qvec = self.embedder.encode_one(query)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sims = (vecs / norms) @ qvec.ravel()
        best = int(np.argmax(sims))
        score = float(sims[best])
        return SemanticSkillMatch(
            skill=query,
            matched_terms=[query],
            score=score,
            evidence=sents[best] if score > 0 else "",
            method="implicit",
        )

    def match(self, resume_text: str, required_skills: Sequence[str], top_k: int = 3) -> SemanticMatchResult:
        normalized = normalize_text(resume_text)
        lexicon = set(self._lexicon_skills())
        lexical_skills = set(extract_skills(resume_text))
        for alias, canonical in _SKILL_ALIASES.items():
            if canonical not in lexical_skills and alias in normalized:
                lexical_skills.add(canonical)

        sents, vecs = self._resume_index(resume_text)
        result = SemanticMatchResult()

        for skill in required_skills:
            skill = skill.lower()
            if skill in lexical_skills:
                result.lexical.append(
                    SemanticSkillMatch(
                        skill=skill,
                        matched_terms=[skill],
                        score=1.0,
                        evidence=sentence_for_term(resume_text, skill) or "",
                        method="lexical",
                    )
                )
                continue
            if skill not in lexicon and len(sents):
                probe = self._sentence_match(skill, sents, vecs)
                if probe.score >= self.threshold:
                    result.implicit.append(probe)
                continue
            candidates = self.related(skill, top_k=top_k)
            best: SemanticSkillMatch | None = None
            for related, rel_score in candidates:
                if related in lexical_skills:
                    candidate = SemanticSkillMatch(
                        skill=skill,
                        matched_terms=[related],
                        score=rel_score,
                        evidence=sentence_for_term(resume_text, related) or "",
                        method="synonym",
                    )
                    if best is None or rel_score > best.score:
                        best = candidate
            if best is not None:
                result.semantic.append(best)
        return result

    def related(self, skill: str, top_k: int = config.SEMANTIC_RELATED_TOP_K) -> list[tuple[str, float]]:
        qvec = self.embedder.encode_one(skill)
        vecs = self._all_skill_vectors()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sims = ((vecs / norms) @ qvec.ravel()).ravel()
        order = np.argsort(sims)[::-1]
        skills = self._lexicon_skills()
        return [(skills[i], float(sims[i])) for i in order[:top_k] if skills[i] != skill]

    @lru_cache(maxsize=1)
    def _all_skill_vectors(self) -> np.ndarray:
        skills = self._lexicon_skills()
        return self.embedder.encode(skills)

    def transferable(self, resume_skills: Sequence[str], required_skills: Sequence[str], top_k: int = 3) -> list[tuple[str, str, float]]:
        resume_set = set(s.lower() for s in resume_skills)
        out = []
        for skill in required_skills:
            skill = skill.lower()
            if skill in resume_set:
                continue
            for related, score in self.related(skill, top_k=top_k):
                if related in resume_set:
                    out.append((skill, related, score))
                    break
        return out
