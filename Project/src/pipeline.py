from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.chat import ResumeAssistant
from src.feedback import FeedbackGenerator, FeedbackReport
from src.matching import MatchResult, generate_requirements_for_category, match_resume_to_job
from src.models import EmbeddingGenerator, TFIDFClassifier, VectorStore


class Pipeline:
    def __init__(self, artifacts_dir: Path | None = None) -> None:
        self.dir = Path(artifacts_dir) if artifacts_dir else config.MODELS_DIR
        self.classifier: TFIDFClassifier | None = None
        self.embedder: EmbeddingGenerator | None = None
        self.store: VectorStore | None = None
        self.records: list[dict] = []
        self.metadata: dict[str, Any] = {}
        self.feedback_generator: FeedbackGenerator | None = None
        self.assistant: ResumeAssistant | None = None
        self._ready = False

    @classmethod
    def load(cls, artifacts_dir: Path | None = None) -> "Pipeline":
        pipeline = cls(artifacts_dir)
        pipeline._load_all()
        return pipeline

    def _load_all(self) -> None:
        classifier_dir = self.dir / "classifier"
        if (classifier_dir / "model.joblib").exists():
            self.classifier = TFIDFClassifier.load(classifier_dir)
        else:
            raise FileNotFoundError(
                "No trained classifier found. Run `python scripts/train.py` first."
            )

        self.embedder = EmbeddingGenerator().load(self.dir / "embedder")

        if config.ARTIFACT_EMBEDDINGS.exists():
            self.store = VectorStore().load(
                config.ARTIFACT_EMBEDDINGS, self._load_records()
            )
        else:
            raise FileNotFoundError(
                "No embedding index found. Run `python scripts/train.py` first."
            )

        self.records = self._load_records()
        self.metadata = self._load_metadata()

        self.feedback_generator = FeedbackGenerator(
            classifier=self.classifier,
            embedder=self.embedder,
            vector_store=self.store,
            category_embeddings=np.asarray(
                self.metadata.get("category_embeddings", []), dtype=np.float32
            ),
            records=self.records,
            top_skills_by_cat=self.metadata.get("top_skills", {}),
        )
        self.assistant = ResumeAssistant(self)
        self._ready = True

    def _load_records(self) -> list[dict]:
        if config.ARTIFACT_RECORDS.exists():
            with open(config.ARTIFACT_RECORDS, encoding="utf-8") as fh:
                return [json.loads(line) for line in fh if line.strip()]
        return []

    def _load_metadata(self) -> dict[str, Any]:
        if config.ARTIFACT_METADATA.exists():
            with open(config.ARTIFACT_METADATA, encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    @property
    def is_ready(self) -> bool:
        return self._ready

    def require(self) -> None:
        if not self._ready:
            raise RuntimeError("Pipeline not loaded.")

    def categories(self) -> list[str]:
        self.require()
        return list(self.classifier.encoder.classes_)

    def predict_category(self, text: str | list[str]) -> list[tuple[str, float]] | list[list[tuple[str, float]]]:
        self.require()
        texts = [text] if isinstance(text, str) else list(text)
        result = self.classifier.predict_categories(texts)
        return result[0] if isinstance(text, str) else result

    def class_top_terms(self, category: str, top_n: int = 15) -> list[tuple[str, float]]:
        self.require()
        return self.classifier.class_top_terms(category, top_n=top_n)

    def match(self, resume_text: str, job_text: str | None) -> MatchResult:
        self.require()
        return match_resume_to_job(
            resume_text, job_text, self.classifier, self.embedder, top_k=5
        )

    def feedback(self, resume_text: str, job_text: str | None) -> FeedbackReport:
        self.require()
        return self.feedback_generator.generate(resume_text, job_text)

    def requirements_for_category(self, category: str) -> str:
        self.require()
        return generate_requirements_for_category(
            category, self.metadata.get("top_skills", {})
        )

    def top_skills_table(self, n: int = 12) -> pd.DataFrame:
        self.require()
        top = self.metadata.get("top_skills", {})
        categories = list(top.keys())
        rows = []
        for cat in categories:
            skills = top[cat][:n]
            rows.append({"Category": cat, "Skills": ", ".join(skills)})
        return pd.DataFrame(rows)

    def chat(self, message: str, context: dict[str, Any] | None = None) -> str:
        self.require()
        return self.assistant.respond(message, context or {})

    def embedding_similarity(self, a: str, b: str) -> float:
        self.require()
        va = self.embedder.encode_one(a)
        vb = self.embedder.encode_one(b)
        return float(va @ vb)
