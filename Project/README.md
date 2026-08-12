# ResumeFit AI

An end-to-end system that evaluates a candidate's resume against a job description,
predicts the best-fit role, and produces explainable feedback (strengths, gaps,
recommendations). It also offers a retrieval-based chat assistant and a Streamlit web app.

## How it works

1. **Classify** — a TF-IDF + Logistic Regression model (`src/models.py`) predicts the
   best-fit role from the resume text.
2. **Match** — the resume is scored 0-100 against the job description by blending three
   signals (`src/matching.py`):
   - **Skill coverage** (45%): which required skills appear on the resume (lexicon-based
     extraction from `assets/skills.json`).
   - **Semantic similarity** (30%): cosine similarity of sentence embeddings between the
     resume and the job text.
   - **Category alignment** (25%): agreement between the predicted role of the resume and
     the job description.
3. **Feedback** — a report is generated (`src/feedback.py`) with evidence sentences pulled
   from the resume and from similar reference profiles (retrieval over a vector store of
   all resumes).
4. **Converse** — an intent-based assistant (`src/chat.py`) answers questions about the
   score, skills, gaps, and role requirements.

## Project layout

- `assets/skills.json` — skill lexicon used for extraction.
- `data/` — training dataset (resumes + job category labels).
- `models/` — trained artifacts (created by `scripts/train.py`, git-ignored).
- `src/` — source code, one module per concept:
  - `config.py` — paths and hyper-parameters.
  - `utils.py` — text cleaning and chunking helpers.
  - `data.py` — dataset loading and schema normalization.
  - `matching.py` — skill extraction, job requirement parsing, and scoring.
  - `models.py` — TF-IDF classifier, embeddings, vector store, evidence retrieval.
  - `feedback.py` — explainable report generation.
  - `chat.py` — intent detection and the conversational assistant.
  - `pipeline.py` — orchestrates loading all models and running evaluations.
  - `app.py` — Streamlit UI.
- `scripts/` — `train.py`, `evaluate.py`, `run_app.py`.
- `tests/` — pytest test suite.

## Quick start

```bash
pip install -r src/requirements.txt

# train (or reuse the existing models/ artifacts)
python scripts/train.py

# launch the web app
python scripts/run_app.py
```

You can also evaluate a single resume from the command line with `python scripts/evaluate.py`.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (the trained `models/` folder is committed on purpose so the app runs without retraining).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → connect the GitHub repo.
3. Set:
   - **Repository**: `jasnoorpannu/CEI-DataScience-2026`
   - **Branch**: `main`
   - **Main file path**: `Project/src/app.py`
4. Deploy. The first load downloads the MiniLM embedding model from Hugging Face.

## Interview talking points

- **Feature extraction**: TF-IDF with n-grams and a curated skill lexicon for keyword matching.
- **Classification**: Logistic Regression over TF-IDF for role prediction (~accuracy in `models/metadata.json`).
- **Semantic search**: sentence embeddings (MiniLM) for similarity; a NumPy/FAISS vector store for finding similar profiles.
- **Explainability**: every score is decomposed into weighted signals with retrieved evidence sentences.
- **Design**: a single `Pipeline` class hides the model-loading complexity behind a simple API used by both the CLI and the UI.
