from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

DEFAULT_DATASET = DATA_DIR / "resumes_dataset.jsonl"

ARTIFACT_TFIDF = MODELS_DIR / "tfidf.joblib"
ARTIFACT_CLASSIFIER_DIR = MODELS_DIR / "classifier"
ARTIFACT_EMBEDDER_DIR = MODELS_DIR / "embedder"
ARTIFACT_METADATA = MODELS_DIR / "metadata.json"
ARTIFACT_EMBEDDINGS = MODELS_DIR / "resume_embeddings.npy"
ARTIFACT_RECORDS = MODELS_DIR / "resume_records.jsonl"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

TEXT_COLUMNS = ["Text", "Resume", "text", "resume", "Resume_str"]
LABEL_COLUMNS = ["Category", "category", "Job", "job", "Job_Area", "Label"]
SKILL_COLUMNS = ["Skills", "skills", "Skill", "skill"]
SUMMARY_COLUMNS = ["Summary", "summary"]
EXPERIENCE_COLUMNS = ["Experience", "experience"]
EDUCATION_COLUMNS = ["Education", "education"]

DEFAULT_JOB_REQUIREMENTS = "Full Stack Developer"

SKILLS_LEXICON = ASSETS_DIR / "skills.json"

TFIDF_MAX_FEATURES = 30000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2

SVD_COMPONENTS = 256

MATCH_WEIGHTS = {
    "skill_coverage": 0.45,
    "embedding_similarity": 0.30,
    "category_affinity": 0.25,
}

RANDOM_SEED = 42

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
