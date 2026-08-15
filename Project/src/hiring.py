from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import pandas as pd

from src import config
from src.evidence import Evidence, build_component_evidence
from src.feedback import FeedbackReport
from src.logging_config import get_logger
from src.matching import extract_skills

logger = get_logger("resumefit.hiring")


@dataclass
class Candidate:
    resume_id: str
    name: str = ""
    category: str = ""
    resume_text: str = ""
    summary: str = ""
    skills: list[str] = field(default_factory=list)
    experience_years: float = 0.0
    location: str = ""
    email: str = ""
    phone: str = ""

    def to_dict(self) -> dict:
        return {
            "resume_id": self.resume_id,
            "name": self.name,
            "category": self.category,
            "summary": self.summary,
            "skills": self.skills,
            "experience_years": self.experience_years,
            "location": self.location,
        }


def candidate_from_row(row) -> Candidate:
    text = str(row.get("Text", ""))
    return Candidate(
        resume_id=str(row.get("ResumeID", "")),
        name=str(row.get("Name", "")),
        category=str(row.get("Category", "")),
        resume_text=text,
        summary=str(row.get("Summary", "")),
        skills=extract_skills(text),
        location=str(row.get("Location", "")),
        email=str(row.get("Email", "")),
        phone=str(row.get("Phone", "")),
    )


def candidates_from_dataframe(df: pd.DataFrame, n: int | None = None) -> list[Candidate]:
    frame = df if n is None else df.head(n)
    return [candidate_from_row(row) for _, row in frame.iterrows()]


def verdict_for_score(score: float) -> str:
    for verdict, threshold in config.SCREENING_THRESHOLDS:
        if score >= threshold:
            return verdict
    return "pass"


def _reasons(report: FeedbackReport) -> list[dict]:
    reasons = []
    if report.strengths:
        top = report.strengths[:3]
        reasons.append(
            {
                "text": f"Strong evidence for matched skills: {', '.join(s.skill for s in top)}.",
                "evidence": [s.evidence for s in top if s.evidence][:3],
                "source": "resume",
            }
        )
    if report.match.semantic_matches:
        semantic = [m["skill"] for m in report.match.semantic_matches]
        reasons.append(
            {
                "text": f"Semantic matching recovered implicit/synonym skills: {', '.join(semantic[:5])}.",
                "evidence": [m["evidence"] for m in report.match.semantic_matches if m.get("evidence")][:3],
                "source": "resume",
            }
        )
    if report.match.transferable_skills:
        transfer = [f"{t[0]}~{t[1]}" for t in report.match.transferable_skills[:3]]
        reasons.append(
            {
                "text": "Transferable skills bridge gaps: " + ", ".join(transfer) + ".",
                "evidence": [],
                "source": "skill",
            }
        )
    if report.gaps:
        reasons.append(
            {
                "text": "Primary gaps to probe in interview: "
                + ", ".join(g.skill for g in report.gaps[:3])
                + ".",
                "evidence": [g.suggestion for g in report.gaps[:3] if g.suggestion],
                "source": "recommendation",
            }
        )
    if report.match.components.embedding_similarity < 0.35:
        reasons.append(
            {
                "text": "Low semantic similarity to the job description; resume language may need tailoring.",
                "evidence": [],
                "source": "job",
            }
        )
    return reasons


@dataclass
class ScreeningItem:
    candidate: Candidate
    rank: int = 0
    score: float = 0.0
    grade: str = ""
    verdict: str = ""
    predicted_category: str = ""
    components: dict[str, float] = field(default_factory=dict)
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    reasons: list[dict] = field(default_factory=list)
    percentile: float = 0.0
    report: FeedbackReport | None = None

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "candidate": self.candidate.to_dict(),
            "score": round(self.score, 1),
            "grade": self.grade,
            "verdict": self.verdict,
            "predicted_category": self.predicted_category,
            "components": {k: round(v * 100, 1) for k, v in self.components.items()},
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "percentile": round(self.percentile, 1),
            "reasons": self.reasons,
        }


@dataclass
class ScreeningPool:
    job_text: str
    items: list[ScreeningItem] = field(default_factory=list)
    weights_version: str = config.MATCH_WEIGHT_VERSION
    semantic: bool = True

    def by_verdict(self, verdict: str) -> list[ScreeningItem]:
        return [i for i in self.items if i.verdict == verdict]

    def shortlist(self) -> list[ScreeningItem]:
        return self.by_verdict("strong_advance") + self.by_verdict("advance")

    def to_dict(self) -> dict:
        return {
            "weights_version": self.weights_version,
            "semantic": self.semantic,
            "job_text": self.job_text,
            "items": [i.to_dict() for i in self.items],
        }


class CandidateScreener:
    def __init__(self, pipeline, semantic: bool = True) -> None:
        self.pipeline = pipeline
        self.semantic = semantic
        self._semantic_extractor = None
        if semantic:
            from src.semantic_skills import SemanticSkillExtractor

            self._semantic_extractor = SemanticSkillExtractor(pipeline.embedder)

    def screen(self, candidates: Sequence[Candidate], job_text: str) -> ScreeningPool:
        logger.info("Screening %d candidates against a job description.", len(candidates))
        items = []
        for candidate in candidates:
            report = self.pipeline.feedback_with(
                candidate.resume_text,
                job_text or None,
                skill_extractor=self._semantic_extractor,
            )
            item = ScreeningItem(
                candidate=candidate,
                score=report.overall_score,
                grade=report.grade,
                verdict=verdict_for_score(report.overall_score),
                predicted_category=report.predicted_category,
                components={
                    "skill_coverage": report.match.components.skill_coverage,
                    "embedding_similarity": report.match.components.embedding_similarity,
                    "category_affinity": report.match.components.category_affinity,
                },
                matched_skills=report.match.matched_skills,
                missing_skills=report.match.missing_skills,
                reasons=_reasons(report),
                percentile=report.benchmark_percentile,
                report=report,
            )
            items.append(item)
        items.sort(key=lambda i: i.score, reverse=True)
        for rank, item in enumerate(items, start=1):
            item.rank = rank
        return ScreeningPool(
            job_text=job_text or "",
            items=items,
            weights_version=self.pipeline.weights_version,
            semantic=self.semantic,
        )


class CandidateComparator:
    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def compare(self, a: ScreeningItem, b: ScreeningItem) -> dict:
        a_comp = a.components
        b_comp = b.components
        rows = []
        for key, label in [
            ("skill_coverage", "Skill coverage"),
            ("embedding_similarity", "Semantic similarity"),
            ("category_affinity", "Category alignment"),
        ]:
            av = a_comp.get(key, 0.0)
            bv = b_comp.get(key, 0.0)
            rows.append(
                {
                    "criterion": label,
                    "a_value": round(av * 100, 1),
                    "b_value": round(bv * 100, 1),
                    "leader": a.candidate.resume_id if av > bv else (b.candidate.resume_id if bv > av else "tie"),
                }
            )
        a_matched = set(a.matched_skills)
        b_matched = set(b.matched_skills)
        only_a = sorted(a_matched - b_matched)
        only_b = sorted(b_matched - a_matched)
        a_gaps = set(a.missing_skills)
        b_gaps = set(b.missing_skills)
        summary = (
            f"{a.candidate.resume_id} leads overall ({a.score:.1f} vs {b.score:.1f}), "
            if a.score != b.score
            else f"{a.candidate.resume_id} and {b.candidate.resume_id} are tied at {a.score:.1f}. "
        )
        summary += (
            f"{a.candidate.resume_id} uniquely brings {', '.join(only_a[:4]) if only_a else 'no additional skills'}; "
            f"{b.candidate.resume_id} uniquely brings {', '.join(only_b[:4]) if only_b else 'no additional skills'}."
        )
        return {
            "candidate_a": a.candidate.resume_id,
            "candidate_b": b.candidate.resume_id,
            "score_a": round(a.score, 1),
            "score_b": round(b.score, 1),
            "criteria": rows,
            "unique_skills_a": only_a,
            "unique_skills_b": only_b,
            "gaps_a_only": sorted(a_gaps - b_gaps),
            "gaps_b_only": sorted(b_gaps - a_gaps),
            "summary": summary,
        }


class HiringWorkflow:
    def __init__(self, stages: Sequence[str] | None = None) -> None:
        self.stages = list(stages) if stages else list(config.HIRING_STAGES)
        self.records: dict[str, list[dict]] = {}

    def _init_if_needed(self, candidate_id: str) -> None:
        if candidate_id not in self.records:
            self.records[candidate_id] = [
                {
                    "stage": "sourced",
                    "note": "Candidate entered the hiring pipeline.",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ]

    def current_stage(self, candidate_id: str) -> str:
        self._init_if_needed(candidate_id)
        return self.records[candidate_id][-1]["stage"]

    def advance(self, candidate_id: str, stage: str, note: str = "") -> dict:
        self._init_if_needed(candidate_id)
        if stage not in self.stages:
            raise ValueError(f"Unknown stage '{stage}'. Valid: {self.stages}")
        current = self.current_stage(candidate_id)
        if self.stages.index(stage) <= self.stages.index(current):
            raise ValueError(f"Cannot move candidate {candidate_id} back from '{current}' to '{stage}'.")
        entry = {
            "stage": stage,
            "note": note,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.records[candidate_id].append(entry)
        logger.info("Candidate %s advanced to '%s' (%s).", candidate_id, stage, note or "no note")
        return entry

    def history(self, candidate_id: str) -> list[dict]:
        self._init_if_needed(candidate_id)
        return self.records[candidate_id]

    def next_actions(self, item: ScreeningItem) -> list[dict]:
        stage = self.current_stage(item.candidate.resume_id)
        score = item.score
        if stage in ("sourced",):
            return [{"action": "screen", "detail": "Evaluate the candidate against the job description."}]
        if stage == "screened":
            if item.verdict in ("strong_advance", "advance"):
                return [
                    {"action": "shortlist", "detail": "Move the candidate to the shortlist."},
                    {"action": "interview", "detail": "Generate interview questions for the candidate."},
                ]
            if item.verdict == "maybe":
                return [
                    {"action": "compare", "detail": "Compare this candidate with the shortlist before deciding."},
                    {"action": "shortlist", "detail": "Conditionally shortlist pending a screening call."},
                ]
            return [{"action": "reject", "detail": "Archive with a respectful rejection note."}]
        if stage == "shortlisted":
            return [
                {"action": "interview", "detail": "Generate interview questions and proceed."},
                {"action": "offer", "detail": "Fast-track to offer for exceptional profiles (score >= 85)."} if score >= 85 else {"action": "interview", "detail": "Proceed to interview."},
            ]
        if stage == "interview":
            return [{"action": "offer", "detail": "Extend an offer if interview feedback is positive."}, {"action": "reject", "detail": "Reject if interview feedback is negative."}]
        if stage == "offer":
            return [{"action": "hired", "detail": "Confirm hire and start onboarding."}, {"action": "rejected", "detail": "Close the loop if the offer is declined."}]
        return [{"action": "close", "detail": "Workflow is complete for this candidate."}]
