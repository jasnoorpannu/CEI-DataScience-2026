# ResumeFit AI — Hiring Assistant

An end-to-end **hiring assistant** that screens and ranks candidates against a job
description, explains every decision with **evidence** from the resume and job text,
generates **interview questions**, and drives the **recruiter workflow**. Built for the
CEI Data Science course project.

## How it works

1. **Classify** — a TF-IDF + Logistic Regression model (`src/models.py`) predicts the
   best-fit role from resume text (36 categories).
2. **Match** — the resume is scored 0-100 against the job description by blending three
   signals (`src/matching.py`):
   - **Skill coverage** — which required skills appear on the resume (lexicon-based
     extraction from `assets/skills.json`).
   - **Semantic similarity** — cosine similarity of sentence embeddings between the
     resume and the job text.
   - **Category alignment** — agreement between the predicted role of the resume and
     the job description.
   - The blend can use the **manual fixed weights (45/30/25)** or **calibrated weights
     (logistic-1)** learned from labelled hiring outcomes (`data/hiring_outcomes.jsonl`,
     see `scripts/calibrate.py`). The pipeline auto-loads `models/calibrated_weights.json`
     when present.
3. **Semantic skills** (`src/semantic_skills.py`) — embedding-based matching recovers
   synonyms, implicit skills and **transferable skills** beyond the curated lexicon.
4. **Evidence** (`src/evidence.py`) — every score component and recommendation cites the
   resume/JD sentences behind it.
5. **Screen & rank** (`src/hiring.py`) — a candidate pool (uploaded or from the dataset)
   is ranked with explainable verdicts (`pass / maybe / advance / strong_advance`) and a
   stage-based workflow with next hiring actions.
6. **Compare & interview** (`src/interviews.py`) — pairwise candidate comparison and
   evidence-grounded interview questions.
7. **Traceability** — every score and recommendation cites the resume/JD evidence behind it.

## Evaluation

`python scripts/evaluate.py` writes `models/eval_report.json` (and `.md`) with:

- **Role classification**: accuracy, macro/weighted F1, per-class precision/recall/F1.
- **Matching quality**: NDCG@10, Hit@1, pairwise accuracy over a sampled resume-JD set.
- **Retrieval quality**: Precision@5, Recall@5, MRR for the similar-profile search.

`python scripts/calibrate.py` fits match weights from labelled hiring outcomes, compares
learned vs fixed weights on ROC-AUC / NDCG, and saves `models/calibrated_weights.json`.

## Project layout

- `assets/skills.json` — skill lexicon used for extraction.
- `data/` — training dataset and labelled hiring outcomes.
- `models/` — trained artifacts (created by `scripts/train.py`, committed so the app runs
  without retraining), plus `calibrated_weights.json` and `eval_report.json`.
- `src/` — source code, one module per concept:
  - `config.py` — paths and hyper-parameters.
  - `utils.py` — text cleaning and chunking helpers.
  - `data.py` — dataset loading and schema normalization.
  - `matching.py` — skill extraction, job requirement parsing, scoring.
  - `semantic_skills.py` — implicit/synonym/transferable skill matching.
  - `models.py` — TF-IDF classifier, embeddings, vector store, model versioning.
  - `evidence.py` — evidence sentences for every decision.
  - `feedback.py` — explainable report generation.
  - `calibration.py` — learned match weights from labelled outcomes.
  - `evaluation.py` — classification / matching / retrieval metrics.
  - `hiring.py` — candidate screening, ranking, comparison, workflow.
  - `interviews.py` — evidence-grounded interview question generation.
  - `auth.py` — user store (PBKDF2) and Streamlit login.
  - `logging_config.py` — rotating-file logging.
  - `chat.py` — intent detection and the conversational assistant.
  - `pipeline.py` — orchestrates loading all models and running evaluations.
  - `app.py` — Streamlit UI (auth-gated recruiter workspace).
- `scripts/` — `train.py`, `evaluate.py`, `calibrate.py`, `run_app.py`.
- `tests/` — pytest suite.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r src/requirements.txt

# train (or reuse the existing models/ artifacts)
python scripts/train.py

# run the test suite
python -m pytest tests/ -q

# evaluate the model quantitatively
python scripts/evaluate.py

# calibrate match weights from labelled hiring outcomes
python scripts/calibrate.py

# launch the web app
python scripts/run_app.py
```

Sign in with the default account **`demo` / `demo123`** (or register a new user).

## Deploy

### Streamlit Community Cloud

1. Push this repo to GitHub (trained `models/` and `data/` are committed on purpose).
2. New app → **Main file path**: `Project/src/app.py`, **Branch**: `main`.
3. First load downloads the MiniLM embedding model from Hugging Face.

### Docker

```bash
docker build -t resumefit .
docker run -p 8501:8501 resumefit
```

## Interview talking points

- **Feature extraction**: TF-IDF with n-grams plus a curated skill lexicon.
- **Classification**: Logistic Regression over TF-IDF for role prediction (~87.9% test
  accuracy, per-class metrics in `models/eval_report.json`).
- **Semantic search**: MiniLM sentence embeddings + NumPy/FAISS vector store.
- **Calibration**: match weights fit from labelled hiring outcomes and compared against
  the fixed 45/30/25 baseline (ROC-AUC, NDCG) — replacing guesswork with learned priors.
- **Explainability & traceability**: every score decomposes into weighted components and
  every recommendation cites its resume/JD evidence sentences.
- **Production readiness**: auth, versioned models, rotating logs, Docker, CI-ready tests.
- **Design**: a single `Pipeline` class hides model-loading behind one API shared by the
  CLI and the UI.
