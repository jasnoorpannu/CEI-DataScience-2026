from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import config
from src.logging_config import get_logger
from src.matching import MatchComponents, match_resume_to_job

logger = get_logger("resumefit.calibration")

COMPONENT_NAMES = ["skill_coverage", "embedding_similarity", "category_affinity"]
METHOD_LOGISTIC = "logistic_regression"
METHOD_GRID = "grid_search"


def _normalize_weights(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, v) for v in values.values())
    if total <= 0:
        return {k: 1.0 / len(values) for k in values}
    return {k: max(0.0, v) / total for k, v in values.items()}


def ndcg_at_k(scores: Sequence[float], labels: Sequence[int], k: int = 10) -> float:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    dcg = sum((2 ** int(labels[i]) - 1) / np.log2(pos + 2) for pos, i in enumerate(order))
    ideal = sum((2 ** int(l) - 1) / np.log2(pos + 2) for pos, l in enumerate(sorted(labels, reverse=True)[:k]))
    return float(dcg / ideal) if ideal > 0 else 0.0


def pairwise_accuracy(scores: Sequence[float], labels: Sequence[int]) -> float:
    pairs = 0
    correct = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if labels[i] == labels[j]:
                continue
            pairs += 1
            agreement = (scores[i] - scores[j]) * (labels[i] - labels[j]) > 0
            correct += int(agreement)
    return correct / pairs if pairs else 0.0


def _metrics(scores: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    if len(np.unique(outcomes)) < 2:
        return {"roc_auc": float("nan"), "ndcg": ndcg_at_k(scores, outcomes), "pairwise": pairwise_accuracy(scores, outcomes)}
    return {
        "roc_auc": float(roc_auc_score(outcomes, scores)),
        "ndcg": ndcg_at_k(scores, outcomes),
        "pairwise": pairwise_accuracy(scores, outcomes),
    }


@dataclass
class CalibratedWeights:
    weights: dict[str, float]
    method: str = METHOD_LOGISTIC
    metrics: dict[str, float] = field(default_factory=dict)
    version: str = ""
    fitted_on: str = ""
    n_samples: int = 0
    coefficients: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CalibratedWeights":
        return cls(
            weights=dict(data.get("weights", config.MATCH_WEIGHTS)),
            method=data.get("method", METHOD_LOGISTIC),
            metrics=dict(data.get("metrics", {})),
            version=data.get("version", ""),
            fitted_on=data.get("fitted_on", ""),
            n_samples=int(data.get("n_samples", 0)),
            coefficients=dict(data.get("coefficients", {})),
        )


def fit_logistic_weights(components: np.ndarray, outcomes: np.ndarray) -> CalibratedWeights:
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=config.RANDOM_SEED)
    clf.fit(components, outcomes)
    raw = {name: float(coef) for name, coef in zip(COMPONENT_NAMES, clf.coef_[0])}
    weights = _normalize_weights(raw)
    scores = clf.predict_proba(components)[:, 1]
    return CalibratedWeights(
        weights=weights,
        method=METHOD_LOGISTIC,
        metrics=_metrics(scores, outcomes),
        coefficients=raw,
    )


def _simplex(step: float = 0.1):
    start = 0.0
    while start <= 1.0:
        for i, j in itertools.product(np.arange(0.0, 1.0 - start + 1e-9, step), repeat=2):
            k = 1.0 - start - i - j
            if k < -1e-9:
                continue
            yield {"skill_coverage": float(start), "embedding_similarity": float(i), "category_affinity": float(k)}
        start += step


def grid_search_weights(components: np.ndarray, outcomes: np.ndarray, step: float = 0.1) -> CalibratedWeights:
    best: CalibratedWeights | None = None
    best_auc = -1.0
    for weights in _simplex(step):
        scores = MatchComponents(
            skill_coverage=components[:, 0],
            embedding_similarity=components[:, 1],
            category_affinity=components[:, 2],
        )
        raw_scores = (
            weights["skill_coverage"] * components[:, 0]
            + weights["embedding_similarity"] * components[:, 1]
            + weights["category_affinity"] * components[:, 2]
        )
        if len(np.unique(outcomes)) < 2:
            continue
        auc = float(roc_auc_score(outcomes, raw_scores))
        if auc > best_auc:
            best_auc = auc
            best = CalibratedWeights(
                weights=dict(weights),
                method=METHOD_GRID,
                metrics=_metrics(raw_scores, outcomes),
                coefficients={k: v for k, v in weights.items()},
            )
    if best is None:
        raise ValueError("Grid search requires both positive and negative outcomes.")
    return best


def build_labelled_dataset(df, pipeline, n_per_category: int = 3, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed or config.RANDOM_SEED)
    rows: list[dict] = []
    for category, group in df.groupby("Category"):
        sample = group.sample(n=min(n_per_category, len(group)), random_state=config.RANDOM_SEED)
        others = [c for c in df["Category"].unique() if c != category]
        for _, row in sample.iterrows():
            resume = str(row["Text"])
            positive_job = pipeline.requirements_for_category(category)
            negative_job = pipeline.requirements_for_category(str(rng.choice(others)))
            rows.append({"resume_text": resume, "job_text": positive_job, "hired": 1})
            rows.append({"resume_text": resume, "job_text": negative_job, "hired": 0})

    components = np.zeros((len(rows), 3), dtype=float)
    outcomes = np.zeros(len(rows), dtype=int)
    for i, row in enumerate(rows):
        result = pipeline.match(row["resume_text"], row["job_text"])
        components[i] = [
            result.components.skill_coverage,
            result.components.embedding_similarity,
            result.components.category_affinity,
        ]
        outcomes[i] = int(row["hired"])
    return components, outcomes, rows


def calibrate(labelled_path: Path | None = None, pipeline=None, df=None) -> tuple[CalibratedWeights, CalibratedWeights]:
    if pipeline is None:
        from src.pipeline import Pipeline

        pipeline = Pipeline.load()
    if labelled_path is None:
        labelled_path = config.LABELLED_HIRING_DATA

    if labelled_path.exists():
        rows = [json.loads(line) for line in labelled_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        components = np.zeros((len(rows), 3), dtype=float)
        outcomes = np.zeros(len(rows), dtype=int)
        for i, row in enumerate(rows):
            result = pipeline.match(str(row["resume_text"]), str(row["job_text"]))
            components[i] = [
                result.components.skill_coverage,
                result.components.embedding_similarity,
                result.components.category_affinity,
            ]
            outcomes[i] = int(row["hired"])
        source = str(labelled_path)
    elif df is not None:
        components, outcomes, _ = build_labelled_dataset(df, pipeline)
        source = f"generated:{len(outcomes)} rows"
    else:
        raise FileNotFoundError("No labelled hiring data available. Pass labelled_path or df.")

    logistic = fit_logistic_weights(components, outcomes)
    grid = grid_search_weights(components, outcomes)
    logistic.metrics["cross_val_auc"] = _cross_val_auc(components, outcomes)
    logistic.n_samples = int(len(outcomes))
    grid.n_samples = int(len(outcomes))
    logistic.fitted_on = source
    grid.fitted_on = source
    logistic.version = "logistic-1"
    grid.version = "grid-1"
    return logistic, grid


def _cross_val_auc(components: np.ndarray, outcomes: np.ndarray) -> float:
    if len(np.unique(outcomes)) < 2 or len(outcomes) < 10:
        return float("nan")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    clf = LogisticRegression(C=1.0, max_iter=2000)
    try:
        scores = cross_val_score(clf, components, outcomes, cv=cv, scoring="roc_auc")
        return float(scores.mean())
    except Exception:
        return float("nan")


def save_calibrated_weights(weights: CalibratedWeights, path: Path | None = None) -> Path:
    path = path or config.ARTIFACT_CALIBRATED_WEIGHTS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights.to_dict(), indent=2), encoding="utf-8")
    logger.info("Saved calibrated weights (%s) to %s", weights.method, path)
    return path


def load_calibrated_weights(path: Path | None = None) -> CalibratedWeights | None:
    path = path or config.ARTIFACT_CALIBRATED_WEIGHTS
    if not path.exists():
        return None
    return CalibratedWeights.from_dict(json.loads(path.read_text(encoding="utf-8")))
