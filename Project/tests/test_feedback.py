from __future__ import annotations

import numpy as np

from src.feedback import FeedbackGenerator
from src.models import EmbeddingGenerator, TFIDFClassifier, VectorStore

SAMPLE_DATA = {
    "Java Developer": [
        "java developer using spring boot and hibernate with mysql and docker",
        "java engineer building microservices with spring mvc and kafka",
        "senior java developer expert in j2ee, servlets and oracle sql",
    ],
    "Data Science": [
        "data scientist using python pandas and machine learning with scikit-learn",
        "data analyst using numpy matplotlib and statistical modeling",
        "machine learning engineer with tensorflow keras and feature engineering",
    ],
    "DevOps": [
        "devops engineer managing docker and kubernetes with terraform",
        "cloud engineer with aws lambda and jenkins for ci cd",
        "site reliability engineer with prometheus and grafana monitoring",
    ],
}


def _build_generator():
    texts = [t for group in SAMPLE_DATA.values() for t in group]
    labels = [cat for cat, group in SAMPLE_DATA.items() for _ in group]
    classifier = TFIDFClassifier().fit(texts, labels)
    embedder = EmbeddingGenerator(backend="tfidf-svd").fit_fallback(texts)
    vectors = embedder.encode(texts)
    records = [
        {"ResumeID": f"R{i}", "Category": labels[i], "Text": texts[i], "_idx": i}
        for i in range(len(texts))
    ]
    store = VectorStore(backend="numpy").build(vectors, payloads=[dict(r) for r in records])
    top_skills = {
        "Java Developer": ["spring boot", "hibernate", "mysql", "docker", "j2ee"],
        "Data Science": ["python", "pandas", "machine learning", "scikit-learn", "numpy"],
        "DevOps": ["docker", "kubernetes", "terraform", "jenkins", "aws"],
    }
    generator = FeedbackGenerator(
        classifier=classifier,
        embedder=embedder,
        vector_store=store,
        category_embeddings=np.zeros((3, vectors.shape[1]), dtype=np.float32),
        records=records,
        top_skills_by_cat=top_skills,
    )
    return generator


def test_generate_report_structure():
    generator = _build_generator()
    resume = SAMPLE_DATA["DevOps"][0]
    job = "We need a devops engineer with docker, kubernetes, terraform and aws."
    report = generator.generate(resume, job)

    assert 0.0 <= report.overall_score <= 100.0
    assert report.predicted_category
    assert report.summary
    assert any(s.skill == "docker" for s in report.strengths)
    assert report.recommendations


def test_generate_report_without_job():
    generator = _build_generator()
    report = generator.generate(SAMPLE_DATA["Data Science"][0], None)
    assert report.predicted_category == "Data Science"
    assert report.match.components.skill_coverage == 0.0


def test_report_to_dict():
    generator = _build_generator()
    report = generator.generate(SAMPLE_DATA["DevOps"][0], None)
    data = report.to_dict()
    for key in ["score", "grade", "predicted_category", "components", "summary"]:
        assert key in data


def test_similar_candidates_returned():
    generator = _build_generator()
    report = generator.generate(SAMPLE_DATA["DevOps"][0], None)
    assert len(report.similar_candidates) >= 1
