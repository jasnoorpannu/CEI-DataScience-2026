from __future__ import annotations

from src.evidence import (
    Evidence,
    build_component_evidence,
    evidence_for_skill,
    find_evidence_sentences,
    sentence_for_term,
)
from src.models import EmbeddingGenerator

CORPUS = [
    "python developer building apis with fastapi and docker containers",
    "data scientist using pandas numpy and scikit-learn for ml models",
    "backend engineer with spring boot hibernate and mysql databases",
    "devops engineer managing kubernetes clusters with terraform and ansible",
    "frontend developer using react typescript and redux state management",
    "machine learning engineer building transformers with pytorch training",
    "android developer writing kotlin apps with jetpack compose and coroutines",
    "data engineer orchestrating airflow pipelines with spark and sql",
    "security engineer performing penetration testing and vulnerability scanning",
    "product manager driving roadmap planning and stakeholder communication",
]


def _embedder():
    return EmbeddingGenerator(backend="tfidf-svd").fit_fallback(CORPUS)


def test_evidence_dataclass():
    ev = Evidence(text="strong python", source="resume", score=0.9)
    data = ev.to_dict()
    assert data["text"] == "strong python"
    assert data["source"] == "resume"


def test_sentence_for_term():
    text = "Five years building microservices. Expert in java and spring boot."
    assert sentence_for_term(text, "java") == "Expert in java and spring boot."


def test_evidence_for_skill_lexical():
    embedder = _embedder()
    ev = evidence_for_skill("docker", "worked with docker and aws daily.", embedder)
    assert ev.text and ev.kind == "lexical"


def test_evidence_for_skill_semantic():
    embedder = _embedder()
    ev = evidence_for_skill("kubernetes", "operated production clusters with helm charts.", embedder)
    assert ev.text
    assert ev.kind in ("lexical", "semantic")


def test_find_evidence_sentences():
    embedder = _embedder()
    hits = find_evidence_sentences("python", "used python for data analysis. strong sql too.", embedder, k=1)
    assert len(hits) == 1
    assert hits[0].source == "resume"


def test_build_component_evidence():
    embedder = _embedder()
    evidence = build_component_evidence(
        matched_skills=["python"],
        missing_skills=["kubernetes"],
        resume_text="python and docker experience.",
        job_text="Requires python and kubernetes.",
        embedder=embedder,
    )
    assert "matched_skills" in evidence
    assert "job_requirements" in evidence
    assert all(isinstance(e, Evidence) for e in evidence["matched_skills"])
