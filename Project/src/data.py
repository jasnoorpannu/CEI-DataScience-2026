from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import config

_JSONL_SUFFIXES = {".jsonl", ".ndjson", ".json"}
_CSV_SUFFIXES = {".csv", ".tsv"}


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    for cand in candidates:
        if cand in columns:
            return cand
    return None


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    text_col = _pick_column(list(df.columns), config.TEXT_COLUMNS)
    label_col = _pick_column(list(df.columns), config.LABEL_COLUMNS)
    skill_col = _pick_column(list(df.columns), config.SKILL_COLUMNS)
    summary_col = _pick_column(list(df.columns), config.SUMMARY_COLUMNS)
    experience_col = _pick_column(list(df.columns), config.EXPERIENCE_COLUMNS)
    education_col = _pick_column(list(df.columns), config.EDUCATION_COLUMNS)

    if text_col is None:
        raise ValueError(
            f"No resume-text column found in columns: {list(df.columns)}. "
            f"Expected one of {config.TEXT_COLUMNS}."
        )
    if label_col is None:
        raise ValueError(
            f"No label column found in columns: {list(df.columns)}. "
            f"Expected one of {config.LABEL_COLUMNS}."
        )

    def _as_str(series: pd.Series | None, default: str = "") -> pd.Series:
        if series is None:
            return pd.Series([default] * len(df), index=df.index, dtype=str)
        return series.fillna("").astype(str)

    out = pd.DataFrame(
        {
            "Category": _as_str(df[label_col]).str.strip(),
            "Text": _as_str(df[text_col]),
            "Summary": _as_str(df[summary_col]),
            "Skills": _as_str(df[skill_col]),
            "Experience": _as_str(df[experience_col]),
            "Education": _as_str(df[education_col]),
        }
    )
    for extra in ("Name", "Email", "Phone", "Location", "Gender"):
        if extra in df.columns:
            out[extra] = _as_str(df[extra])
    out = out[out["Text"].str.len() >= 20].reset_index(drop=True)
    return out


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix in _JSONL_SUFFIXES:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        df = pd.DataFrame(records)
    elif path.suffix in _CSV_SUFFIXES:
        sep = "\t" if path.suffix == ".tsv" else ","
        df = pd.read_csv(path, encoding="utf-8", sep=sep, low_memory=False)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    normalized = _normalize_schema(df)
    if "ResumeID" not in normalized.columns and "ResumeID" in df.columns:
        normalized.insert(0, "ResumeID", df["ResumeID"].fillna("").astype(str))
    if "ResumeID" not in normalized.columns:
        normalized.insert(0, "ResumeID", [f"R{i:05d}" for i in range(len(normalized))])
    return normalized


def load_default_dataset() -> pd.DataFrame:
    return load_dataset(config.DEFAULT_DATASET)


def category_stats(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["Category"].value_counts().rename("count")
    return counts.to_frame().sort_values("count", ascending=False)
