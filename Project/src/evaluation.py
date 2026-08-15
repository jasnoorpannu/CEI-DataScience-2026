from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from src import config
from src.calibration import ndcg_at_k, pairwise_accuracy


@dataclass
class EvaluationReport:
    classification: dict = field(default_factory=dict)
    matching: dict = field(default_factory=dict)
    retrieval: dict = field(default_factory=dict)
    pipeline_meta: dict = field(default_factory=dict)
    sampled: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = ["# Evaluation report", ""]
        meta = self.pipeline_meta
        lines.append(
            f"- Pipeline: model `{meta.get('model_id', '?')}` v{meta.get('version', '?')} "
            f"({meta.get('backend', '?')} embeddings), sampled on {self.sampled} resumes."
        )
        lines.append("")

        cls = self.classification
        if cls:
            lines.append("## 1. Role classification")
            lines.append(f"- Accuracy: **{cls.get('accuracy', float('nan')):.3f}**")
            lines.append(f"- Macro F1: **{cls.get('macro_f1', float('nan')):.3f}**")
            lines.append(f"- Weighted F1: **{cls.get('weighted_f1', float('nan')):.3f}**")
            if cls.get("note"):
                lines.append(f"- *{cls['note']}*")
            lines.append("")
            lines.append("| Category | Precision | Recall | F1 | Support |")
            lines.append("|---|---|---|---|---|")
            for cat, metrics in sorted(cls.get("per_class", {}).items()):
                lines.append(
                    f"| {cat} | {metrics['precision']:.3f} | {metrics['recall']:.3f} "
                    f"| {metrics['f1']:.3f} | {int(metrics['support'])} |"
                )
            lines.append("")

        match = self.matching
        if match:
            lines.append("## 2. Resume-JD matching (ranking quality)")
            lines.append(f"- NDCG@10: **{match.get('ndcg_10', float('nan')):.3f}**")
            lines.append(f"- Hit@1 (correct role in top prediction): **{match.get('hit_at_1', float('nan')):.3f}**")
            lines.append(f"- Pairwise accuracy: **{match.get('pairwise', float('nan')):.3f}**")
            lines.append(f"- Mean overall score: **{match.get('mean_score', float('nan')):.1f}**")
            lines.append("")

        ret = self.retrieval
        if ret:
            lines.append("## 3. Retrieval quality (similar-profile search)")
            lines.append(f"- Precision@5: **{ret.get('precision_at_5', float('nan')):.3f}**")
            lines.append(f"- Recall@5: **{ret.get('recall_at_5', float('nan')):.3f}**")
            lines.append(f"- MRR@10: **{ret.get('mrr', float('nan')):.3f}**")
            lines.append(f"- Same-category rate @6: **{ret.get('same_category_rate', float('nan')):.3f}**")
            lines.append("")
        return "\n".join(lines)

    def save(self, path: Path | None = None, markdown: bool = True) -> Path:
        path = path or config.ARTIFACT_EVAL_REPORT
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        if markdown:
            md_path = path.with_suffix(".md")
            md_path.write_text(self.to_markdown(), encoding="utf-8")
        return path


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    macro = report["macro avg"]
    weighted = report["weighted avg"]
    per_class = {}
    for label in sorted(set(y_true)):
        per_class[label] = {
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "f1": report[label]["f1-score"],
            "support": report[label]["support"],
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": macro["f1-score"],
        "weighted_f1": weighted["f1-score"],
        "per_class": per_class,
    }


def evaluate_matching(pipeline, sample: pd.DataFrame, k: int = 10) -> dict:
    scores = []
    labels = []
    errors = 0
    for _, row in sample.iterrows():
        true_cat = str(row["Category"])
        job = pipeline.requirements_for_category(true_cat)
        try:
            report = pipeline.feedback(str(row["Text"]), job)
        except Exception:
            errors += 1
            continue
        predicted_cat = report.predicted_category
        scores.append(report.overall_score)
        labels.append(1 if predicted_cat == true_cat else 0)
    if not scores:
        return {"errors": errors}
    scores_arr = np.asarray(scores, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)
    return {
        "n": len(scores_arr),
        "errors": errors,
        "mean_score": float(scores_arr.mean()),
        "ndcg_10": ndcg_at_k(scores_arr, labels_arr, k=k),
        "hit_at_1": float(np.mean(labels_arr)),
        "pairwise": pairwise_accuracy(scores_arr, labels_arr),
    }


def evaluate_retrieval(pipeline, sample: pd.DataFrame, k: int = 5) -> dict:
    precisions = []
    recalls = []
    reciprocal_ranks = []
    same_category = []
    total = 0
    correct = 0
    for _, row in sample.head(200).iterrows():
        vec = pipeline.embedder.encode_one(str(row["Text"]))
        neighbors = pipeline.store.search(vec, k=max(k, 6))
        true_cat = str(row["Category"])
        hits = [n for n in neighbors if n.payload.get("Category") == true_cat]
        relevant = sum(1 for rec in pipeline.records if rec.get("Category") == true_cat)
        precisions.append(len(hits[:k]) / k)
        recalls.append(len(hits[:k]) / max(relevant, 1))
        for rank, neighbor in enumerate(neighbors, start=1):
            if neighbor.payload.get("Category") == true_cat:
                reciprocal_ranks.append(1.0 / rank)
                break
        for neighbor in neighbors[1:]:
            total += 1
            if neighbor.payload.get("Category") == true_cat:
                correct += 1
        same_category.append(
            sum(1 for n in neighbors[1:] if n.payload.get("Category") == true_cat) / max(len(neighbors) - 1, 1)
        )
    return {
        "n": len(precisions),
        "precision_at_5": float(np.mean(precisions)),
        "recall_at_5": float(np.mean(recalls)),
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan"),
        "same_category_rate": float(np.mean(same_category)) if same_category else float("nan"),
    }


def run_full_evaluation(pipeline, df: pd.DataFrame, sample_size: int = 60) -> EvaluationReport:
    sample = df.sample(n=min(sample_size, len(df)), random_state=config.RANDOM_SEED)
    preds = pipeline.classifier.predict(df["Text"].tolist())
    classification = classification_metrics(df["Category"].tolist(), preds)
    classification["note"] = (
        "Classification metrics are in-sample (all resumes); "
        "held-out test accuracy is recorded in models/metadata.json."
    )
    matching = evaluate_matching(pipeline, sample)
    retrieval = evaluate_retrieval(pipeline, sample)
    meta = pipeline.metadata
    return EvaluationReport(
        classification=classification,
        matching=matching,
        retrieval=retrieval,
        pipeline_meta={
            "model_id": meta.get("model_id", ""),
            "version": meta.get("version", ""),
            "backend": meta.get("backend", ""),
            "num_resumes": meta.get("num_resumes", ""),
            "num_categories": meta.get("num_categories", ""),
        },
        sampled=int(len(sample)),
    )
