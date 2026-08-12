from __future__ import annotations

import numpy as np
import pandas as pd

from src.matching import (
    MatchComponents,
    extract_skills,
    generate_requirements_for_category,
    match_resume_to_job,
    parse_job_requirements,
    skill_coverage,
    skill_gaps,
    skill_overlap,
    top_skills_by_category,
)
from src.models import EmbeddingGenerator, TFIDFClassifier

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


def _build_tools():
    texts = [t for group in SAMPLE_DATA.values() for t in group]
    labels = [cat for cat, group in SAMPLE_DATA.items() for _ in group]
    classifier = TFIDFClassifier().fit(texts, labels)
    embedder = EmbeddingGenerator(backend="tfidf-svd").fit_fallback(texts)
    return classifier, embedder, texts, labels


def test_extract_single_word_skills():
    text = "I use python, kubernetes, docker and mysql every day."
    skills = extract_skills(text)
    assert "python" in skills
    assert "kubernetes" in skills
    assert "docker" in skills
    assert "mysql" in skills


def test_extract_multi_word_skills():
    text = "Experienced with spring boot, machine learning and rest api design."
    skills = extract_skills(text)
    assert "spring boot" in skills
    assert "machine learning" in skills
    assert "rest api" in skills


def test_extract_does_not_match_substrings():
    skills = extract_skills("I like painting and drawing.")
    assert "art" not in skills


def test_extract_c_sharp_and_c_plus_plus():
    skills = extract_skills("C# and C++ are both used here. Node.js too.")
    assert "c#" in skills
    assert "c++" in skills
    assert "node.js" in skills


def test_skill_helpers():
    resume = ["python", "docker", "aws"]
    required = ["python", "kubernetes", "aws", "terraform"]
    assert skill_overlap(resume, required) == ["python", "aws"]
    assert skill_gaps(resume, required) == ["kubernetes", "terraform"]
    assert skill_coverage(resume, required) == 0.5


def test_top_skills_by_category():
    df = pd.DataFrame(
        {
            "Category": ["DevOps", "DevOps", "DevOps"],
            "Text": [
                "I know docker and kubernetes.",
                "I know kubernetes and terraform.",
                "I know docker and aws.",
            ],
        }
    )
    top = top_skills_by_category(df, top_n=5)
    assert "kubernetes" in top["DevOps"]
    assert top["DevOps"][0] in {"docker", "kubernetes"}


def test_parse_job_requirements():
    req = parse_job_requirements("We need a developer with python, docker and kubernetes experience.")
    assert "python" in req.skills
    assert "docker" in req.skills
    assert "kubernetes" in req.skills
    assert req.keywords


def test_parse_empty_requirements():
    req = parse_job_requirements("   ")
    assert req.is_empty


def test_generate_requirements_for_category():
    top = {"DevOps": ["docker", "kubernetes", "terraform", "jenkins"]}
    text = generate_requirements_for_category("DevOps", top)
    assert "DevOps" in text
    assert "docker" in text


def test_match_resume_to_job():
    classifier, embedder, texts, labels = _build_tools()
    resume = SAMPLE_DATA["DevOps"][0]
    job = "We need a devops engineer with docker, kubernetes and terraform."
    result = match_resume_to_job(resume, job, classifier, embedder)
    assert result.overall_score > 0
    assert "docker" in result.matched_skills
    assert "kubernetes" in result.matched_skills
    assert result.components.skill_coverage > 0


def test_match_predicts_category():
    classifier, embedder, texts, labels = _build_tools()
    result = match_resume_to_job(SAMPLE_DATA["Data Science"][0], None, classifier, embedder)
    cats = [c for c, _ in result.resume_categories]
    assert "Data Science" in cats


def test_components_overall_bounds():
    comp = MatchComponents(skill_coverage=1.0, embedding_similarity=1.0, category_affinity=1.0)
    assert comp.overall() == 100.0
    comp2 = MatchComponents(skill_coverage=0.0, embedding_similarity=0.0, category_affinity=0.0)
    assert comp2.overall() == 0.0


def test_embedding_vectors_normalized():
    classifier, embedder, texts, labels = _build_tools()
    v = embedder.encode_one(SAMPLE_DATA["DevOps"][0])
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-4)
