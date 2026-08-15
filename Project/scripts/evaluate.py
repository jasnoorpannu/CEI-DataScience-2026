from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import load_dataset
from src.evaluation import run_full_evaluation
from src.pipeline import Pipeline


def evaluate(args: argparse.Namespace) -> None:
    pipeline = Pipeline.load()
    df = load_dataset(args.data)
    print(f"Loaded {len(df)} resumes across {df['Category'].nunique()} categories.")
    print(f"Pipeline: model {pipeline.metadata.get('model_id', '?')} "
          f"v{pipeline.metadata.get('version', '?')}, "
          f"weights={pipeline.weights_version}")

    report = run_full_evaluation(pipeline, df, sample_size=args.sample)
    report.save()

    cls = report.classification
    print("\n=== 1. Role classification ===")
    print(f"Accuracy: {cls['accuracy']:.4f} | Macro F1: {cls['macro_f1']:.4f} | Weighted F1: {cls['weighted_f1']:.4f}")
    print(f"Note: {cls.get('note', '')}")
    print(f"Per-class precision/recall/F1 for the 10 lowest-F1 classes:")
    worst = sorted(cls["per_class"].items(), key=lambda kv: kv[1]["f1"])[:10]
    for cat, m in worst:
        print(f"  {cat:28s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={int(m['support'])})")

    match = report.matching
    if match:
        print("\n=== 2. Resume-JD matching (ranking quality) ===")
        print(f"Sampled {match['n']} resumes (errors: {match.get('errors', 0)})")
        print(f"NDCG@10: {match['ndcg_10']:.4f} | Hit@1: {match['hit_at_1']:.4f} | Pairwise: {match['pairwise']:.4f} | Mean score: {match['mean_score']:.1f}")

    ret = report.retrieval
    if ret:
        print("\n=== 3. Retrieval quality ===")
        print(f"Precision@5: {ret['precision_at_5']:.4f} | Recall@5: {ret['recall_at_5']:.4f} | MRR: {ret['mrr']:.4f} | Same-category@6: {ret['same_category_rate']:.4f}")

    print(f"\nFull report written to:\n  {config.ARTIFACT_EVAL_REPORT}\n  {config.ARTIFACT_EVAL_REPORT_MD}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the resume pipeline end to end")
    parser.add_argument("--data", default=str(config.DEFAULT_DATASET), help="Path to dataset")
    parser.add_argument("--sample", type=int, default=60, help="Number of resumes for matching/retrieval eval")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
