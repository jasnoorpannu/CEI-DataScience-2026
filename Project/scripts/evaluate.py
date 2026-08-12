from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import load_dataset
from src.pipeline import Pipeline


def evaluate(args: argparse.Namespace) -> None:
    pipeline = Pipeline.load()
    df = load_dataset(args.data)
    print(f"Loaded {len(df)} resumes.")

    preds = pipeline.classifier.predict(df["Text"].tolist())
    acc = accuracy_score(df["Category"], preds)
    print(f"\nML classifier category accuracy: {acc:.4f}")
    print(
        classification_report(
            df["Category"], preds, output_dict=False, zero_division=0
        )
    )

    sample = df.sample(n=args.sample, random_state=config.RANDOM_SEED)
    score_rows = []
    errors = 0
    for _, row in sample.iterrows():
        job = pipeline.requirements_for_category(row["Category"])
        try:
            report = pipeline.feedback(row["Text"], job)
            score_rows.append(
                {
                    "resume_id": row["ResumeID"],
                    "true_category": row["Category"],
                    "predicted_category": report.predicted_category,
                    "score": report.overall_score,
                    "grade": report.grade,
                    "matched": len(report.match.matched_skills),
                    "missing": len(report.match.missing_skills),
                    "benchmark_percentile": report.benchmark_percentile,
                }
            )
        except Exception as exc:
            errors += 1
            print(f"Feedback failed for {row['ResumeID']}: {exc}")

    if score_rows:
        df_scores = pd.DataFrame(score_rows)
        print(f"\nFeedback run on {len(df_scores)} resumes ({errors} errors).")
        print(f"Mean overall score: {df_scores['score'].mean():.1f}")
        print(f"Mean matched skills: {df_scores['matched'].mean():.1f}")
        print(f"Mean missing skills: {df_scores['missing'].mean():.1f}")
        print(f"Mean benchmark percentile: {df_scores['benchmark_percentile'].mean():.1f}")
        print(f"Score distribution:\n{df_scores['score'].describe()}")

    retrieval_correct = 0
    retrieval_total = 0
    for _, row in sample.head(50).iterrows():
        vec = pipeline.embedder.encode_one(row["Text"])
        neighbors = pipeline.store.search(vec, k=6)
        for n in neighbors[1:]:
            retrieval_total += 1
            if n.payload.get("Category") == row["Category"]:
                retrieval_correct += 1
    if retrieval_total:
        print(f"\nSimilar-profile retrieval same-category rate: {retrieval_correct / retrieval_total:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the resume pipeline")
    parser.add_argument("--data", default=str(config.DEFAULT_DATASET), help="Path to dataset")
    parser.add_argument("--sample", type=int, default=40, help="Number of resumes for feedback eval")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
