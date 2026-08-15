from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.calibration import calibrate, load_calibrated_weights, save_calibrated_weights
from src.pipeline import Pipeline


def _fixed_weights_metrics(pipeline, rows: list[dict]) -> dict:
    scores = []
    labels = []
    for row in rows:
        result = pipeline.match(str(row["resume_text"]), str(row["job_text"]))
        scores.append(result.overall_score)
        labels.append(int(row["hired"]))
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    return {"roc_auc": float(roc_auc_score(labels, scores)), "n": len(scores)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate match weights from labelled hiring outcomes")
    parser.add_argument("--data", default=str(config.LABELLED_HIRING_DATA), help="Path to labelled hiring outcomes")
    parser.add_argument("--save", action="store_true", default=True, help="Save the calibrated weights")
    args = parser.parse_args()

    pipeline = Pipeline.load()
    print(f"Pipeline loaded with fixed weights {pipeline.weights_version}: {config.MATCH_WEIGHTS}")

    logistic, grid = calibrate(labelled_path=Path(args.data), pipeline=pipeline)
    fixed = _fixed_weights_metrics(pipeline, [dict(line) for line in map(_parse, Path(args.data).read_text(encoding="utf-8").splitlines())])

    print("\nWeight calibration on labelled hiring outcomes")
    print(f"Source: {args.data} | samples: {logistic.n_samples} (1=hired, 0=rejected)")
    print("\nMethod            skill_coverage  embedding_similarity  category_affinity   ROC-AUC   NDCG")
    for name, cal, fixed_auc in [
        ("fixed (manual)", None, fixed["roc_auc"]),
        (logistic.method, logistic, None),
        (grid.method, grid, None),
    ]:
        if cal is None:
            print(f"{name:16s} {config.MATCH_WEIGHTS['skill_coverage']:11.3f} {config.MATCH_WEIGHTS['embedding_similarity']:16.3f} {config.MATCH_WEIGHTS['category_affinity']:12.3f}  {fixed_auc:.3f}")
        else:
            w = cal.weights
            print(f"{name:16s} {w['skill_coverage']:11.3f} {w['embedding_similarity']:16.3f} {w['category_affinity']:12.3f}  {cal.metrics.get('roc_auc', float('nan')):.3f}  {cal.metrics.get('ndcg', float('nan')):.3f}")

    if args.save:
        save_calibrated_weights(logistic)
        print(f"\nSaved logistic-calibrated weights to {config.ARTIFACT_CALIBRATED_WEIGHTS}")
        print("The pipeline will now use these weights automatically (weights_version='logistic-1').")


def _parse(line: str) -> dict:
    import json

    return json.loads(line)


if __name__ == "__main__":
    main()
