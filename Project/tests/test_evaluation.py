from __future__ import annotations

import numpy as np

from src.evaluation import (
    EvaluationReport,
    classification_metrics,
    evaluate_matching,
    evaluate_retrieval,
)


def test_classification_metrics():
    y_true = ["a", "a", "a", "b", "b", "b"]
    y_pred = ["a", "a", "b", "b", "b", "b"]
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 5 / 6
    assert set(metrics["per_class"]) == {"a", "b"}
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics


class _FakeRecord:
    def __init__(self, category: str) -> None:
        self.payload = {"Category": category}


class _FakeStore:
    def __init__(self, categories: list[str]) -> None:
        self.categories = categories

    def search(self, vec, k: int):
        return [_FakeRecord(cat) for cat in self.categories[:k]]


class _FakeEmbedder:
    def encode_one(self, text: str) -> np.ndarray:
        return np.zeros(8, dtype=np.float32)


class _FakePipelineRetrieval:
    def __init__(self, store: _FakeStore) -> None:
        self.store = store
        self.embedder = _FakeEmbedder()
        self.records = [{"Category": cat} for cat in store.categories]


def test_retrieval_metrics_perfect():
    import pandas as pd

    store = _FakeStore(["DS", "DS", "DS", "DS", "DS"])
    pipeline = _FakePipelineRetrieval(store)
    df = pd.DataFrame([{"Text": "t", "Category": "DS"}])
    metrics = evaluate_retrieval(pipeline, df, k=5)
    assert metrics["precision_at_5"] == 1.0
    assert metrics["mrr"] == 1.0


def test_retrieval_metrics_mrr():
    import pandas as pd

    store = _FakeStore(["Other", "DS", "DS", "DS", "DS"])
    pipeline = _FakePipelineRetrieval(store)
    df = pd.DataFrame([{"Text": "t", "Category": "DS"}])
    metrics = evaluate_retrieval(pipeline, df, k=5)
    assert metrics["mrr"] == 0.5


def test_evaluation_report_markdown():
    report = EvaluationReport(
        classification={"accuracy": 0.8, "macro_f1": 0.7, "weighted_f1": 0.75, "per_class": {}},
        matching={"ndcg_10": 0.5, "hit_at_1": 0.3, "pairwise": 0.6, "mean_score": 50.0},
        retrieval={"precision_at_5": 0.4, "recall_at_5": 0.5, "mrr": 0.3},
        pipeline_meta={"model_id": "abc", "version": "1", "backend": "tfidf-svd"},
        sampled=10,
    )
    md = report.to_markdown()
    assert "Role classification" in md
    assert "NDCG@10" in md
    assert "MRR" in md


def test_evaluation_report_save(tmp_path):
    report = EvaluationReport()
    path = report.save(tmp_path / "eval.json")
    assert path.exists()
    assert path.with_suffix(".md").exists()
