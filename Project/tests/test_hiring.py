from __future__ import annotations

import numpy as np

from src.feedback import FeedbackGenerator, FeedbackReport
from src.hiring import (
    Candidate,
    CandidateComparator,
    CandidateScreener,
    HiringWorkflow,
    candidate_from_row,
    candidates_from_dataframe,
    verdict_for_score,
)
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


class _FakePipeline:
    def __init__(self) -> None:
        texts = [t for group in SAMPLE_DATA.values() for t in group]
        labels = [cat for cat, group in SAMPLE_DATA.items() for _ in group]
        self.classifier = TFIDFClassifier().fit(texts, labels)
        self.embedder = EmbeddingGenerator(backend="tfidf-svd").fit_fallback(texts)
        vectors = self.embedder.encode(texts)
        records = [
            {"ResumeID": f"R{i}", "Category": labels[i], "Text": texts[i], "_idx": i}
            for i in range(len(texts))
        ]
        self.store = VectorStore(backend="numpy").build(vectors, payloads=[dict(r) for r in records])
        self.records = records
        self.weights_version = "test"
        self.calibrated_weights = None
        self.feedback_generator = FeedbackGenerator(
            classifier=self.classifier,
            embedder=self.embedder,
            vector_store=self.store,
            category_embeddings=np.zeros((3, vectors.shape[1]), dtype=np.float32),
            records=records,
            top_skills_by_cat={
                "Java Developer": ["spring boot", "hibernate", "mysql"],
                "Data Science": ["python", "pandas", "machine learning"],
                "DevOps": ["docker", "kubernetes", "terraform"],
            },
        )

    def feedback_with(self, resume_text, job_text, skill_extractor=None, **kwargs) -> FeedbackReport:
        return self.feedback_generator.generate(
            resume_text, job_text, skill_extractor=skill_extractor,
            weights=kwargs.get("weights"), weights_version=kwargs.get("weights_version"),
        )


def _candidates():
    return [
        Candidate(resume_id="A", name="James Smith", resume_text=SAMPLE_DATA["Data Science"][0]),
        Candidate(resume_id="B", name="Mary Jones", resume_text=SAMPLE_DATA["DevOps"][0]),
        Candidate(resume_id="C", name="Raj Patel", resume_text=SAMPLE_DATA["Java Developer"][0]),
    ]


def test_verdict_for_score():
    assert verdict_for_score(85) == "strong_advance"
    assert verdict_for_score(70) == "advance"
    assert verdict_for_score(55) == "maybe"
    assert verdict_for_score(10) == "pass"


def test_candidate_from_row():
    candidate = candidate_from_row({"ResumeID": "R1", "Name": "Jane", "Text": "python developer", "Location": "NYC"})
    assert candidate.resume_id == "R1"
    assert candidate.name == "Jane"
    assert candidate.location == "NYC"


def test_screening_ranks_and_verdicts():
    import pandas as pd

    df = pd.DataFrame(
        [
            {"ResumeID": "A", "Name": "James", "Text": SAMPLE_DATA["Data Science"][0]},
            {"ResumeID": "B", "Name": "Mary", "Text": SAMPLE_DATA["DevOps"][0]},
            {"ResumeID": "C", "Name": "Raj", "Text": SAMPLE_DATA["Java Developer"][0]},
        ]
    )
    candidates = candidates_from_dataframe(df)
    pool = CandidateScreener(_FakePipeline(), semantic=False).screen(
        candidates, "We need a devops engineer with docker and kubernetes."
    )
    assert len(pool.items) == 3
    assert [i.rank for i in pool.items] == [1, 2, 3]
    assert pool.items[0].score >= pool.items[1].score >= pool.items[2].score
    assert all(i.verdict for i in pool.items)
    assert pool.items[0].reasons


def test_compare():
    pool = CandidateScreener(_FakePipeline(), semantic=False).screen(
        _candidates(), "We need a devops engineer with docker and kubernetes."
    )
    comparison = CandidateComparator(_FakePipeline()).compare(pool.items[0], pool.items[1])
    assert "candidate_a" in comparison
    assert comparison["criteria"]
    assert comparison["summary"]


def test_workflow_advance_and_guardrails():
    workflow = HiringWorkflow()
    assert workflow.current_stage("A") == "sourced"
    workflow.advance("A", "screened", note="passed screening")
    assert workflow.current_stage("A") == "screened"
    assert workflow.history("A")[-1]["note"] == "passed screening"
    try:
        workflow.advance("A", "sourced")
        assert False, "should reject backward move"
    except ValueError:
        pass


def test_workflow_next_actions():
    from src.hiring import ScreeningItem

    item = ScreeningItem(candidate=Candidate(resume_id="A"), score=90, verdict="strong_advance")
    workflow = HiringWorkflow()
    workflow.advance("A", "screened")
    actions = [a["action"] for a in workflow.next_actions(item)]
    assert "interview" in actions or "shortlist" in actions
