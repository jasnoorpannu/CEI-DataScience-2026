from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.utils import sentence_split

EVIDENCE_SOURCES = ("resume", "job", "reference", "skill", "recommendation", "synthetic")


@dataclass
class Evidence:
    text: str
    source: str = "resume"
    score: float = 0.0
    kind: str = "match"
    attribution: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "score": round(float(self.score), 3),
            "kind": self.kind,
            "attribution": self.attribution,
        }


def _top_sentence_hits(query_vec: np.ndarray, sent_vecs: np.ndarray, k: int = 3, threshold: float = 0.0):
    if len(sent_vecs) == 0:
        return []
    norms = np.linalg.norm(sent_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sims = (sent_vecs / norms) @ query_vec.ravel()
    order = np.argsort(sims)[::-1]
    hits = []
    for idx in order:
        score = float(sims[idx])
        if score < threshold:
            break
        hits.append((int(idx), score))
        if len(hits) >= k:
            break
    return hits


def find_evidence_sentences(
    query: str,
    text: str,
    embedder,
    k: int = 1,
    threshold: float = 0.0,
    source: str = "resume",
) -> list[Evidence]:
    sents = sentence_split(text)
    if not sents:
        return []
    qvec = embedder.encode_one(query)
    sent_vecs = embedder.encode(sents)
    hits = _top_sentence_hits(qvec, sent_vecs, k=k, threshold=threshold)
    return [Evidence(text=sents[idx], score=score, source=source) for idx, score in hits]


def sentence_for_term(text: str, term: str) -> str:
    lowered = term.lower()
    for sent in sentence_split(text):
        if lowered in sent.lower():
            return sent
    return ""


def evidence_for_skill(skill: str, resume_text: str, embedder) -> Evidence:
    sent = sentence_for_term(resume_text, skill)
    if sent:
        return Evidence(text=sent, source="resume", score=1.0, kind="lexical")
    hits = find_evidence_sentences(skill, resume_text, embedder, k=1)
    if hits:
        hits[0].kind = "semantic"
        return hits[0]
    return Evidence(text="", source="resume", score=0.0, kind="none")


def build_component_evidence(
    matched_skills: Sequence[str],
    missing_skills: Sequence[str],
    resume_text: str,
    job_text: str,
    embedder,
    top_n: int = 5,
) -> dict[str, list[Evidence]]:
    resume_evidence = []
    for skill in matched_skills[:top_n]:
        resume_evidence.append(evidence_for_skill(skill, resume_text, embedder))
    missing_evidence = []
    for skill in missing_skills[:top_n]:
        ev = evidence_for_skill(skill, resume_text, embedder)
        if ev.text:
            ev.kind = "missing"
            missing_evidence.append(ev)
    job_evidence = find_evidence_sentences("required skills", job_text, embedder, k=3, source="job")
    return {
        "matched_skills": resume_evidence,
        "missing_skills": missing_evidence,
        "job_requirements": job_evidence,
    }


def render_evidence(evidences: Sequence[Evidence], max_len: int = 200) -> str:
    parts = []
    for ev in evidences:
        snippet = ev.text[:max_len] if len(ev.text) > max_len else ev.text
        parts.append(f"[{ev.source}:{ev.kind}] \"{snippet}\" (score {ev.score:.2f})")
    return "\n".join(parts)
