from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import load_dataset
from src.matching import top_skills_by_category
from src.models import EmbeddingGenerator, TFIDFClassifier


def _prepare_outputs() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _dataset_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _save_records(df: pd.DataFrame) -> None:
    records = []
    for idx, row in df.iterrows():
        records.append(
            {
                "ResumeID": str(row.get("ResumeID", f"R{idx:05d}")),
                "Category": str(row["Category"]),
                "Text": str(row["Text"]),
                "Summary": str(row.get("Summary", "")),
                "Skills": str(row.get("Skills", "")),
                "Experience": str(row.get("Experience", "")),
                "Education": str(row.get("Education", "")),
                "_idx": int(idx),
            }
        )
    with open(config.ARTIFACT_RECORDS, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def train(args: argparse.Namespace) -> dict:
    _prepare_outputs()
    df = load_dataset(args.data)
    print(f"Loaded {len(df)} resumes across {df['Category'].nunique()} categories.")

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=config.RANDOM_SEED, stratify=df["Category"]
    )

    classifier = TFIDFClassifier().fit(train_df["Text"], train_df["Category"])
    classifier.save(config.ARTIFACT_CLASSIFIER_DIR)
    test_pred = classifier.predict(test_df["Text"])
    test_acc = float((test_pred == test_df["Category"]).mean())
    report = classification_report(
        test_df["Category"], test_pred, output_dict=True, zero_division=0
    )
    print(f"ML classifier test accuracy: {test_acc:.4f}")

    embedder = EmbeddingGenerator(backend=args.backend)
    backend_used = args.backend
    try:
        embedder.encode(train_df["Text"].head(2).tolist())
        print(f"Using embedding backend: {backend_used}")
    except Exception as exc:
        print(f"Backend {args.backend} unavailable ({exc}); falling back to tfidf-svd.")
        embedder = EmbeddingGenerator(backend="tfidf-svd")
        backend_used = "tfidf-svd"
    if embedder.backend != "sentence-transformers":
        embedder.fit_fallback(df["Text"])
    embedder.save(config.ARTIFACT_EMBEDDER_DIR)

    embeddings = embedder.encode(df["Text"].tolist())
    _save_records(df)
    np.save(config.ARTIFACT_EMBEDDINGS, embeddings)
    print(f"Embeddings saved: {embeddings.shape}")

    top_skills = top_skills_by_category(df)
    class_terms = {
        str(cat): [t for t, _ in classifier.class_top_terms(cat, top_n=20)]
        for cat in classifier.encoder.classes_
    }
    metadata = {
        "model_id": uuid.uuid4().hex[:12],
        "version": "2.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha": _dataset_sha(Path(args.data)),
        "runtime": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "num_resumes": int(len(df)),
        "num_categories": int(df["Category"].nunique()),
        "categories": [str(c) for c in classifier.encoder.classes_],
        "backend": backend_used,
        "embedding_dim": int(embeddings.shape[1]),
        "test_accuracy": test_acc,
        "classification_report": report,
        "top_skills": top_skills,
        "class_top_terms": class_terms,
        "trained_on": args.data,
    }

    with open(config.ARTIFACT_METADATA, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    print("Metadata saved.")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train resume-evaluation models")
    parser.add_argument("--data", default=str(config.DEFAULT_DATASET), help="Path to dataset (jsonl/csv)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction")
    parser.add_argument("--backend", choices=["sentence-transformers", "tfidf-svd"], default="sentence-transformers")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
