from __future__ import annotations

import numpy as np

from src.models import EmbeddingGenerator
from src.semantic_skills import SemanticSkillExtractor

TEXTS = [
    "python developer building apis with fastapi and docker containers",
    "data engineer orchestrating airflow pipelines with pyspark and sql",
    "devops engineer managing kubernetes clusters with terraform and ansible",
    "machine learning engineer building models with pytorch and scikit-learn",
    "backend engineer with spring boot hibernate and mysql databases",
    "frontend developer using react typescript and redux state management",
    "data scientist using pandas numpy and statistical modeling",
    "security engineer performing penetration testing and vulnerability scanning",
    "android developer writing kotlin apps with jetpack compose",
    "product manager driving roadmap planning and stakeholder communication",
    "data engineer processing real-time streams with pyspark and kafka",
]


def _extractor():
    embedder = EmbeddingGenerator(backend="tfidf-svd").fit_fallback(TEXTS)
    return SemanticSkillExtractor(embedder, threshold=0.1)


def test_lexical_match():
    extractor = _extractor()
    result = extractor.match(TEXTS[0], ["python", "machine learning"])
    assert any(m.skill == "python" and m.method == "lexical" for m in result.all())


def test_implicit_match_for_skill_outside_lexicon():
    extractor = _extractor()
    resume = "built real-time pipelines with pyspark on aws"
    result = extractor.match(resume, ["pyspark"])
    matches = [m for m in result.all() if m.method == "implicit"]
    assert matches
    assert matches[0].score >= extractor.threshold
    assert matches[0].evidence


def test_synonym_match_through_related_skill():
    class FakeEmbedder:
        def __init__(self) -> None:
            self.dim = 8
            self.vector = np.eye(8)[0]

        @property
        def embedding_dim(self) -> int:
            return self.dim

        def encode_one(self, text: str) -> np.ndarray:
            if text.lower().strip() in ("docker", "containerization"):
                return self.vector.copy()
            return np.zeros(self.dim, dtype=np.float32)

        def encode(self, texts) -> np.ndarray:
            return np.array([self.encode_one(t) for t in texts], dtype=np.float32)

    extractor = SemanticSkillExtractor(FakeEmbedder(), threshold=0.5)
    result = extractor.match("worked with docker for microservices", ["containerization"])
    synonyms = [m for m in result.all() if m.method == "synonym"]
    assert synonyms
    assert synonyms[0].skill == "containerization"
    assert synonyms[0].matched_terms == ["docker"]


def test_no_false_positive_when_threshold_high():
    embedder = EmbeddingGenerator(backend="tfidf-svd").fit_fallback(TEXTS)
    extractor = SemanticSkillExtractor(embedder, threshold=0.99)
    result = extractor.match("painting and drawing daily", ["python"])
    assert "python" not in result.matched_skills()


def test_related_returns_scored_skills():
    extractor = _extractor()
    related = extractor.related("python", top_k=3)
    assert related
    assert all(isinstance(score, float) for _, score in related)


def test_transferable_skills():
    extractor = _extractor()
    pairs = extractor.transferable(["python", "sql"], ["airflow", "machine learning"], top_k=3)
    assert isinstance(pairs, list)
    for target, source, score in pairs:
        assert source in ("python", "sql")
        assert target != source
        assert isinstance(score, float)
