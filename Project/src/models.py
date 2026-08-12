from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, Normalizer

from src import config
from src.utils import clean_text, sentence_split


class TFIDFClassifier:
    def __init__(self, max_features: int = config.TFIDF_MAX_FEATURES) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            min_df=config.TFIDF_MIN_DF,
            sublinear_tf=True,
            stop_words="english",
        )
        self.model = LogisticRegression(C=1.0, max_iter=2000, random_state=config.RANDOM_SEED)
        self.encoder = LabelEncoder()
        self._feature_names: np.ndarray | None = None

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "TFIDFClassifier":
        clean = [clean_text(t) for t in texts]
        x = self.vectorizer.fit_transform(clean)
        self._feature_names = self.vectorizer.get_feature_names_out()
        self.encoder.fit(labels)
        y = self.encoder.transform(labels)
        self.model.fit(x, y)
        return self

    def _matrix(self, texts: Sequence[str]):
        return self.vectorizer.transform([clean_text(t) for t in texts])

    def predict(self, texts: Sequence[str]) -> list[str]:
        y = self.model.predict(self._matrix(texts))
        return list(self.encoder.inverse_transform(y))

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.predict_proba(self._matrix(texts))

    def predict_categories(self, texts: Sequence[str]) -> list[list[tuple[str, float]]]:
        proba = self.predict_proba(texts)
        out = []
        for row in proba:
            pairs = list(zip(self.encoder.classes_, row))
            pairs.sort(key=lambda x: x[1], reverse=True)
            out.append(pairs)
        return out

    def top_terms(self, text: str, top_n: int = 15) -> list[tuple[str, float]]:
        clean = clean_text(text)
        x = self.vectorizer.transform([clean])
        predicted = self.model.predict(x)[0]
        coefs = self.model.coef_[predicted]
        score = x.toarray().ravel() * coefs
        order = np.argsort(score)[::-1]
        terms = []
        for idx in order:
            if score[idx] <= 0:
                break
            terms.append((str(self._feature_names[idx]), float(score[idx])))
            if len(terms) >= top_n:
                break
        return terms

    def class_top_terms(self, class_label: str, top_n: int = 15) -> list[tuple[str, float]]:
        if class_label not in self.encoder.classes_:
            return []
        class_idx = int(np.where(self.encoder.classes_ == class_label)[0][0])
        coefs = self.model.coef_[class_idx]
        order = np.argsort(coefs)[::-1]
        return [(str(self._feature_names[i]), float(coefs[i])) for i in order[:top_n]]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, directory / "tfidf.joblib")
        joblib.dump(self.model, directory / "model.joblib")
        joblib.dump(self.encoder, directory / "encoder.joblib")

    @classmethod
    def load(cls, directory: Path) -> "TFIDFClassifier":
        obj = cls()
        obj.vectorizer = joblib.load(directory / "tfidf.joblib")
        obj.model = joblib.load(directory / "model.joblib")
        obj.encoder = joblib.load(directory / "encoder.joblib")
        obj._feature_names = obj.vectorizer.get_feature_names_out()
        return obj


class EmbeddingGenerator:
    def __init__(
        self,
        backend: str = "sentence-transformers",
        model_name: str = config.EMBEDDING_MODEL_NAME,
        dimension: int = config.SVD_COMPONENTS,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.dimension = dimension
        self._st_model = None
        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None
        self._normalizer = Normalizer(norm="l2")

    @property
    def embedding_dim(self) -> int:
        if self.backend == "sentence-transformers":
            if self._st_model is not None:
                return self._st_model.get_sentence_embedding_dimension()
            return config.EMBEDDING_DIM
        return self.dimension

    def _load_st(self):
        from sentence_transformers import SentenceTransformer

        if self._st_model is None:
            self._st_model = SentenceTransformer(self.model_name)
        return self._st_model

    def fit_fallback(self, texts: Sequence[str]) -> "EmbeddingGenerator":
        self._vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            stop_words="english",
        )
        x = self._vectorizer.fit_transform([clean_text(t) for t in texts])
        n_components = min(self.dimension, x.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=n_components, random_state=config.RANDOM_SEED)
        self._svd.fit(x)
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        if self.backend == "sentence-transformers":
            try:
                model = self._load_st()
                vectors = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
                return np.asarray(vectors, dtype=np.float32)
            except Exception:
                if self._svd is None:
                    raise
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("Fallback embedder not fitted; call fit_fallback first.")
        x = self._vectorizer.transform([clean_text(t) for t in texts])
        vectors = self._svd.transform(x)
        return self._normalizer.transform(vectors).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._vectorizer, directory / "fallback_tfidf.joblib")
        joblib.dump(self._svd, directory / "fallback_svd.joblib")
        with open(directory / "backend.txt", "w", encoding="utf-8") as fh:
            fh.write(self.backend)

    def load(self, directory: Path) -> "EmbeddingGenerator":
        backend_file = directory / "backend.txt"
        if backend_file.exists():
            with open(backend_file, encoding="utf-8") as fh:
                self.backend = fh.read().strip()
        vec_file = directory / "fallback_tfidf.joblib"
        if vec_file.exists():
            self._vectorizer = joblib.load(vec_file)
        svd_file = directory / "fallback_svd.joblib"
        if svd_file.exists():
            self._svd = joblib.load(svd_file)
        return self


@dataclass
class Neighbor:
    index: int
    score: float
    payload: dict


class VectorStore:
    def __init__(self, backend: str = "numpy") -> None:
        self.backend = backend
        self.vectors: np.ndarray | None = None
        self.payloads: list[dict] = []
        self._faiss_index = None

    def build(self, vectors: np.ndarray, payloads: list[dict] | None = None) -> "VectorStore":
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.payloads = payloads or [{} for _ in range(len(self.vectors))]
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = self.vectors / norms
        if self.backend == "faiss":
            try:
                import faiss

                self._faiss_index = faiss.IndexFlatIP(self.vectors.shape[1])
                self._faiss_index.add(self.vectors)
            except Exception:
                self._faiss_index = None
        return self

    def search(self, query_vector: np.ndarray, k: int = 10) -> list[Neighbor]:
        if self.vectors is None:
            raise RuntimeError("Vector store not built.")
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        qnorm = np.linalg.norm(query)
        if qnorm > 0:
            query = query / qnorm

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query, k)
            scores = scores[0].tolist()
            indices = indices[0].tolist()
        else:
            sims = self.vectors @ query.T
            sims = sims.ravel()
            order = np.argsort(sims)[::-1][:k]
            indices = order.tolist()
            scores = sims[order].tolist()

        return [
            Neighbor(index=int(idx), score=float(score), payload=self.payloads[idx])
            for idx, score in zip(indices, scores)
            if idx >= 0
        ]

    def save(self, path: str) -> None:
        np.save(path, self.vectors)

    def load(self, path: str, payloads: list[dict]) -> "VectorStore":
        self.vectors = np.load(path, allow_pickle=True)
        self.payloads = payloads
        return self


def build_sentence_index(
    text: str, generator: EmbeddingGenerator
) -> tuple[list[str], np.ndarray]:
    sents = sentence_split(text)
    if not sents:
        return [], np.zeros((0, generator.embedding_dim), dtype=np.float32)
    vectors = generator.encode(sents)
    return sents, vectors


def search_sentences(
    query_vec: np.ndarray, vectors: np.ndarray, k: int = 5, threshold: float = 0.15
) -> list[tuple[int, float]]:
    if len(vectors) == 0:
        return []
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sims = (vectors / norms) @ query_vec.ravel()
    order = np.argsort(sims)[::-1]
    results = []
    for idx in order:
        score = float(sims[idx])
        if score < threshold:
            break
        results.append((int(idx), score))
        if len(results) >= k:
            break
    return results


def retrieve_evidence(
    query_vec: np.ndarray,
    sents: list[str],
    vectors: np.ndarray,
    k: int = 5,
    threshold: float = 0.15,
) -> list[tuple[str, float]]:
    hits = search_sentences(query_vec, vectors, k=k, threshold=threshold)
    return [(sents[idx], score) for idx, score in hits]
