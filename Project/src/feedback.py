from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.matching import MatchResult, match_resume_to_job
from src.models import (
    EmbeddingGenerator,
    TFIDFClassifier,
    VectorStore,
    build_sentence_index,
    search_sentences,
)
from src.utils import sentence_split


@dataclass
class SkillEvidence:
    skill: str
    evidence: str
    importance: float = 0.5


@dataclass
class Gap:
    skill: str
    importance: float = 0.5
    suggestion: str = ""
    peer_evidence: str = ""


@dataclass
class SimilarCandidate:
    resume_id: str
    category: str
    score: float


@dataclass
class FeedbackReport:
    match: MatchResult
    predicted_category: str = ""
    predicted_confidence: float = 0.0
    top_terms: list[tuple[str, float]] = field(default_factory=list)
    strengths: list[SkillEvidence] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    summary: str = ""
    similar_candidates: list[SimilarCandidate] = field(default_factory=list)
    benchmark_percentile: float = 0.0

    @property
    def overall_score(self) -> float:
        return self.match.overall_score

    @property
    def grade(self) -> str:
        return self.match.grade()

    def to_dict(self) -> dict:
        return {
            "score": self.overall_score,
            "grade": self.grade,
            "predicted_category": self.predicted_category,
            "predicted_confidence": round(self.predicted_confidence, 3),
            "top_terms": self.top_terms,
            "components": {
                "skill_coverage": round(self.match.components.skill_coverage * 100, 1),
                "embedding_similarity": round(self.match.components.embedding_similarity * 100, 1),
                "category_affinity": round(self.match.components.category_affinity * 100, 1),
            },
            "strengths": [
                {"skill": s.skill, "evidence": s.evidence, "importance": round(s.importance, 3)}
                for s in self.strengths
            ],
            "gaps": [
                {
                    "skill": g.skill,
                    "importance": round(g.importance, 3),
                    "suggestion": g.suggestion,
                    "peer_evidence": g.peer_evidence,
                }
                for g in self.gaps
            ],
            "recommendations": self.recommendations,
            "summary": self.summary,
            "similar_candidates": [
                {"resume_id": c.resume_id, "category": c.category, "score": round(c.score, 3)}
                for c in self.similar_candidates
            ],
            "benchmark_percentile": round(self.benchmark_percentile, 1),
        }


class FeedbackGenerator:
    def __init__(
        self,
        classifier: TFIDFClassifier,
        embedder: EmbeddingGenerator,
        vector_store: VectorStore,
        category_embeddings: np.ndarray,
        records: Sequence[dict],
        top_skills_by_cat: dict[str, list[str]],
    ) -> None:
        self.classifier = classifier
        self.embedder = embedder
        self.store = vector_store
        self.category_embeddings = category_embeddings
        self.records = records
        self.top_skills_by_cat = top_skills_by_cat
        self.category_centroids = self._centroids()

    def _centroids(self) -> dict[str, np.ndarray]:
        centroids: dict[str, np.ndarray] = {}
        for rec in self.records:
            cat = rec.get("Category", "")
            idx = int(rec.get("_idx", -1))
            if idx < 0:
                continue
            vec = self.store.vectors[idx]
            if cat not in centroids:
                centroids[cat] = vec.copy()
            else:
                centroids[cat] += vec
        for cat in centroids:
            norm = np.linalg.norm(centroids[cat])
            if norm > 0:
                centroids[cat] = centroids[cat] / norm
        return centroids

    def _skill_sentence(self, skill: str, text: str) -> str:
        sents = sentence_split(text)
        if not sents:
            return ""
        for sent in sents:
            if skill in sent:
                return sent
        return ""

    def _peer_evidence(self, skill: str, category: str, k: int = 3) -> str:
        count = 0
        for rec in self.records:
            if rec.get("Category", "") != category:
                continue
            text = rec.get("Text", "")
            sent = self._skill_sentence(skill, text)
            if sent:
                return sent
            count += 1
            if count > k:
                break
        return ""

    def generate(
        self,
        resume_text: str,
        job_text: str | None,
        top_k: int = 5,
    ) -> FeedbackReport:
        match = match_resume_to_job(resume_text, job_text, self.classifier, self.embedder, top_k=top_k)

        predicted_category, predicted_confidence = match.resume_categories[0] if match.resume_categories else ("", 0.0)
        top_terms = self.classifier.top_terms(resume_text)

        resume_vec = self.embedder.encode_one(resume_text)
        neighbors = self.store.search(resume_vec, k=6)
        similar = [
            SimilarCandidate(
                resume_id=neighbor.payload.get("ResumeID", ""),
                category=neighbor.payload.get("Category", ""),
                score=neighbor.score,
            )
            for neighbor in neighbors
            if neighbor.payload.get("Text") not in (resume_text,)
        ][:top_k]

        sents, sent_vecs = build_sentence_index(resume_text, self.embedder)

        strengths: list[SkillEvidence] = []
        for skill in match.matched_skills:
            skill_vec = self.embedder.encode_one(skill)
            evidence = ""
            if len(sent_vecs) > 0:
                hits = search_sentences(skill_vec, sent_vecs, k=1)
                if hits:
                    evidence = sents[hits[0][0]]
            if not evidence:
                evidence = self._skill_sentence(skill, resume_text)
            importance = self._importance(skill, predicted_category)
            strengths.append(SkillEvidence(skill=skill, evidence=evidence, importance=importance))

        strengths.sort(key=lambda s: s.importance, reverse=True)

        gaps: list[Gap] = []
        for skill in match.missing_skills:
            importance = self._importance(skill, predicted_category)
            suggestion = f"Add or highlight your experience with '{skill}' in the summary, skills, and work-history sections."
            peer = self._peer_evidence(skill, predicted_category)
            gaps.append(Gap(skill=skill, importance=importance, suggestion=suggestion, peer_evidence=peer))
        gaps.sort(key=lambda g: g.importance, reverse=True)

        recommendations = self._recommendations(match, predicted_category, gaps)
        percentile = self._benchmark(match, resume_vec)
        summary = self._build_summary(match, predicted_category, predicted_confidence, gaps)

        report = FeedbackReport(
            match=match,
            predicted_category=predicted_category,
            predicted_confidence=predicted_confidence,
            top_terms=top_terms,
            strengths=strengths,
            gaps=gaps,
            recommendations=recommendations,
            summary=summary,
            similar_candidates=similar,
            benchmark_percentile=percentile,
        )
        return self._maybe_rewrite(report)

    def _importance(self, skill: str, category: str) -> float:
        top = self.top_skills_by_cat.get(category, [])
        if not top:
            return 0.5
        if skill in top[:5]:
            return 1.0
        if skill in top[:10]:
            return 0.8
        if skill in top:
            return 0.6
        return 0.4

    def _benchmark(self, match: MatchResult, resume_vec: np.ndarray) -> float:
        category = match.resume_categories[0][0] if match.resume_categories else ""
        centroid = self.category_centroids.get(category)
        if centroid is None:
            return 0.0
        candidate_sim = float(resume_vec @ centroid)
        members = [
            self.store.vectors[int(rec["_idx"])]
            for rec in self.records
            if rec.get("Category", "") == category and int(rec.get("_idx", -1)) >= 0
        ]
        if not members:
            return 0.0
        member_sims = [float(m @ centroid) for m in members]
        return float(np.mean([sim <= candidate_sim for sim in member_sims]) * 100)

    def _recommendations(
        self, match: MatchResult, predicted_category: str, gaps: Sequence[Gap]
    ) -> list[str]:
        recs = []
        if gaps:
            important = [g.skill for g in gaps if g.importance >= 0.6][:3]
            if important:
                recs.append(
                    f"Prioritize addressing the highest-impact gaps: {', '.join(important)}. "
                    "These appear frequently in profiles for this role."
                )
        top_terms = self.classifier.top_terms(match.resume_text, top_n=5)
        if top_terms and predicted_category:
            terms = ", ".join(term for term, _ in top_terms[:3])
            recs.append(
                f"Your resume signals '{predicted_category}' strongly through terms like {terms}. "
                "Make sure these keywords appear in your summary and headline."
            )
        if match.components.embedding_similarity < 0.35:
            recs.append(
                "Your overall text similarity to the job description is low. "
                "Tailor your summary and bullet points to mirror the job description's language."
            )
        if not match.resume_skills:
            recs.append("No clear technical skills detected. Add a dedicated Skills section with concrete technologies.")
        return recs[:5]

    def _build_summary(
        self, match: MatchResult, category: str, confidence: float, gaps: Sequence[Gap]
    ) -> str:
        matched = len(match.matched_skills)
        total = len(match.missing_skills) + matched
        parts = []
        if category:
            parts.append(
                f"The profile is best classified as **{category}** "
                f"with {confidence * 100:.0f}% confidence."
            )
        if total:
            parts.append(f"The candidate matches {matched} of {total} required skills.")
        parts.append(
            f"Skill coverage is {match.components.skill_coverage * 100:.0f}%, "
            f"text similarity is {match.components.embedding_similarity * 100:.0f}%, "
            f"and category alignment is {match.components.category_affinity * 100:.0f}%."
        )
        if gaps:
            parts.append(f"Key gaps: {', '.join(g.skill for g in gaps[:4])}.")
        return " ".join(parts)

    def _maybe_rewrite(self, report: FeedbackReport) -> FeedbackReport:
        endpoint = os.environ.get("FEEDBACK_LLM_ENDPOINT", "")
        api_key = os.environ.get("FEEDBACK_LLM_API_KEY", "")
        if not endpoint or not api_key:
            return report
        try:
            import requests

            prompt = self._llm_prompt(report)
            resp = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": os.environ.get("FEEDBACK_LLM_MODEL", "gpt-4o-mini"),
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.4,
                },
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                report.summary = text
        except Exception:
            pass
        return report

    def _llm_prompt(self, report: FeedbackReport) -> str:
        data = report.to_dict()
        lines = [
            "Rewrite the following resume-evaluation summary into 3-4 clear, professional sentences. "
            "Keep it factual and do not invent information.",
            f"Predicted role: {data['predicted_category']}",
            f"Overall score: {data['score']}/100 ({data['grade']}).",
            f"Component scores: {data['components']}.",
            f"Strengths: {', '.join(s['skill'] for s in data['strengths'][:6])}.",
            f"Gaps: {', '.join(g['skill'] for g in data['gaps'][:6])}.",
        ]
        return "\n".join(lines)
