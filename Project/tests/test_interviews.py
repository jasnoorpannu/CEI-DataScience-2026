from __future__ import annotations

from src.hiring import Candidate
from src.interviews import QuestionGenerator


class _FakeMatch:
    missing_skills = ["kubernetes"]
    matched_skills = ["docker"]
    requirements = type("R", (), {"skills": ["docker", "kubernetes", "terraform"]})()


class _FakeReport:
    match = _FakeMatch()
    strengths = []
    semantic_matches = []


class _FakePipeline:
    def requirements_for_category(self, category: str) -> str:
        return f"Requirements for {category}"


class _FakeItem:
    def __init__(self, resume_id: str, score: float, category: str = "DevOps") -> None:
        self.candidate = Candidate(resume_id=resume_id, category=category)
        self.score = score
        self.predicted_category = category
        self.report = _FakeReport()


def test_question_generation_kinds():
    generator = QuestionGenerator(_FakePipeline())
    questions = generator.generate(_FakeItem("R1", 90), "docker kubernetes terraform")
    kinds = {q.kind for q in questions}
    assert "technical" in kinds
    assert "evidence_gap" in kinds
    assert "behavioral" in kinds
    assert "screening" in kinds


def test_questions_tie_to_gaps():
    generator = QuestionGenerator(_FakePipeline())
    questions = generator.generate(_FakeItem("R1", 90), "docker kubernetes terraform")
    gap_q = [q for q in questions if q.kind == "evidence_gap"]
    assert gap_q and "kubernetes" in gap_q[0].question
