from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.auth import AuthManager
from src.feedback import FeedbackReport
from src.hiring import Candidate, CandidateComparator, CandidateScreener, HiringWorkflow, candidates_from_dataframe
from src.interviews import QuestionGenerator

st.set_page_config(page_title="ResumeFit AI - Hiring Assistant", layout="wide")


def _grade_color(score: float) -> str:
    if score >= 80:
        return "#28a745"
    if score >= 60:
        return "#17a2b8"
    if score >= 40:
        return "#ffc107"
    return "#dc3545"


def score_gauge(score: float, grade: str) -> None:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 48}},
            title={"text": f"<b>{grade}</b>", "font": {"size": 22}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": _grade_color(score)},
                "steps": [
                    {"range": [0, 40], "color": "#fde2e2"},
                    {"range": [40, 60], "color": "#fff3cd"},
                    {"range": [60, 80], "color": "#d4edda"},
                    {"range": [80, 100], "color": "#c3e6cb"},
                ],
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)


def components_radar(components: dict[str, float]) -> None:
    labels = ["Skill coverage", "Text similarity", "Category alignment"]
    values = [components["skill_coverage"], components["embedding_similarity"], components["category_affinity"]]
    fig = go.Figure(go.Scatterpolar(r=values, theta=labels, fill="toself", name="Components"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_report(report: FeedbackReport) -> None:
    col1, col2 = st.columns([1, 1.2])
    with col1:
        score_gauge(report.overall_score, report.grade)
    with col2:
        st.subheader("Component breakdown")
        components_radar(
            {
                "skill_coverage": report.match.components.skill_coverage * 100,
                "embedding_similarity": report.match.components.embedding_similarity * 100,
                "category_affinity": report.match.components.category_affinity * 100,
            }
        )

    st.markdown("### Summary")
    st.markdown(report.summary)

    if report.match.semantic_matches:
        semantic = ", ".join(m["skill"] for m in report.match.semantic_matches[:6])
        st.info(f"Semantic matching recovered implicit/synonym skills: **{semantic}**")
    if report.match.transferable_skills:
        transfer = ", ".join(f"{s}~{t}" for s, t, _ in report.match.transferable_skills[:4])
        st.info(f"Transferable skills bridging gaps: **{transfer}**")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Strengths")
        for item in report.strengths[:8]:
            st.markdown(f"- **{item.skill}** - *{item.evidence[:200]}*")
    with c2:
        st.markdown("### Gaps")
        for gap in report.gaps[:8]:
            st.markdown(f"- **{gap.skill}** (importance {gap.importance * 100:.0f}%)")
            if gap.suggestion:
                st.markdown(f"  - {gap.suggestion}")

    st.markdown("### Recommendations (evidence-linked)")
    for i, rec in enumerate(report.recommendation_details, start=1):
        st.markdown(f"{i}. {rec.text}")
        if rec.evidence and rec.evidence.text:
            st.caption(f"   Evidence [{rec.evidence.source}]: {rec.evidence.text[:220]}")

    st.markdown("### Benchmark & similar profiles")
    st.markdown(
        f"**{report.benchmark_percentile:.0f}%** of reference profiles in "
        f"**{report.predicted_category}** score lower on similarity."
    )
    if report.similar_candidates:
        cols = st.columns(min(5, len(report.similar_candidates)))
        for i, cand in enumerate(report.similar_candidates[:5]):
            with cols[i]:
                st.metric(cand.resume_id, cand.category, f"{cand.score * 100:.0f}%")


@st.cache_resource(show_spinner=False)
def get_pipeline():
    from src.pipeline import Pipeline

    try:
        return Pipeline.load()
    except FileNotFoundError as exc:
        return exc


def pipeline_ready():
    pipeline = get_pipeline()
    return isinstance(pipeline, Exception) is False


def require_pipeline():
    pipeline = get_pipeline()
    if isinstance(pipeline, Exception):
        st.error(f"Models are not trained yet.\n\n{get_pipeline()}")
        st.code("python scripts/train.py\npython -m streamlit run src/app.py", language="bash")
        st.stop()
    return pipeline


def page_overview():
    st.title("ResumeFit AI - Hiring Assistant")
    st.markdown(
        """
        An end-to-end **hiring assistant** built on the resume-evaluation engine. It screens and
        ranks candidates against a job description, explains every decision with **evidence** from
        the resume and job text, generates **interview questions**, and drives the **recruiter
        workflow**.
        """
    )

    if pipeline_ready():
        pipeline = get_pipeline()
        meta = pipeline.metadata
        c1, c2, c3 = st.columns(3)
        c1.metric("Resumes in dataset", meta.get("num_resumes", "?"))
        c2.metric("Job categories", meta.get("num_categories", "?"))
        c3.metric("Classifier accuracy", f"{meta.get('test_accuracy', 0.0) * 100:.1f}%")
        st.success(
            f"Model ready with `{meta.get('backend', '?')}` embeddings, calibrated match weights (`{pipeline.weights_version}`)."
        )
        with st.expander("System details"):
            st.markdown(f"Model ID: `{meta.get('model_id', '?')}` v{meta.get('version', '?')}")
            st.markdown(f"Trained: {meta.get('trained_at', '?')}")
            st.markdown(f"Match weights: {pipeline.calibrated_weights or 'manual fixed weights'}")
    else:
        st.warning("No trained models found yet - run `python scripts/train.py` first.")

    st.markdown("### How it works")
    st.markdown(
        """
        1. **Classify** - TF-IDF + Logistic Regression predicts the best-fit role (36 categories).
        2. **Match** - score 0-100 from skill coverage, semantic similarity, category alignment,
           using either manual or **calibrated (learned) weights**.
        3. **Semantic skills** - embedding-based matching recovers synonyms, implicit and transferable skills.
        4. **Screen & rank** - a candidate pool is ranked with explainable verdicts.
        5. **Compare & interview** - pairwise comparison and evidence-based interview questions.
        6. **Traceability** - every score and recommendation cites the resume/JD evidence behind it.
        """
    )


def _extract_upload(uploaded) -> str:
    name = uploaded.name.lower()
    if name.endswith(".txt"):
        return uploaded.getvalue().decode("utf-8", errors="ignore")
    if name.endswith(".docx"):
        import io

        try:
            from docx import Document

            doc = Document(io.BytesIO(uploaded.getvalue()))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            st.info("Install `python-docx` to extract .docx files.")
            return ""
    if name.endswith(".pdf"):
        import io

        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(uploaded.getvalue()))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            st.info("Install `pypdf` to extract PDF text, or paste the text manually.")
            return ""
    return ""


def page_evaluate():
    st.title("Evaluate a resume")
    pipeline = require_pipeline()

    tab_paste, tab_upload = st.tabs(["Paste text", "Upload file"])
    resume_text = ""
    with tab_paste:
        resume_text = st.text_area("Resume text", height=280, placeholder="Paste the full resume text here...")
    with tab_upload:
        uploaded = st.file_uploader("Upload resume (txt, docx, pdf)", type=["txt", "docx", "pdf"])
        if uploaded is not None:
            resume_text = _extract_upload(uploaded)
            st.text_area("Extracted text", resume_text, height=280)

    st.divider()
    st.markdown("**Job description** *(optional - leave empty to auto-generate from the predicted role)*")
    job_text = st.text_area("Job description", height=150, placeholder="Paste the job description...")
    if not job_text.strip():
        chosen = st.selectbox("Or auto-generate requirements for role", [""] + pipeline.categories())
        if chosen:
            job_text = pipeline.requirements_for_category(chosen)

    if st.button("Run evaluation", type="primary", use_container_width=True):
        if not resume_text.strip():
            st.warning("Provide a resume first.")
            st.stop()
        with st.spinner("Scoring the resume..."):
            report = pipeline.feedback(resume_text, job_text or None)
        st.session_state["last_report"] = report
        st.session_state["last_resume"] = resume_text
        st.session_state["last_job"] = job_text
        render_report(report)


def page_chat():
    st.title("Conversational assistant")
    pipeline = require_pipeline()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! I'm your hiring copilot. Screen a pool in the Recruiter workspace and I can "
                "rank it, compare candidates, generate interview questions, and recommend the next "
                "hiring action.",
            }
        ]
    if "chat_resume" not in st.session_state:
        st.session_state.chat_resume = ""
    if "chat_job" not in st.session_state:
        st.session_state.chat_job = ""

    with st.sidebar:
        st.markdown("### Chat context")
        st.session_state.chat_resume = st.text_area(
            "Resume text (optional)", value=st.session_state.chat_resume, height=180, key="chat_resume_input"
        )
        st.session_state.chat_job = st.text_area(
            "Job description (optional)", value=st.session_state.chat_job, height=120, key="chat_job_input"
        )
        if st.session_state.get("last_resume") and not st.session_state.chat_resume:
            st.caption("Using the resume evaluated on the Evaluate page.")
        pool = st.session_state.get("pool")
        if pool is not None:
            st.caption(f"Pool loaded: {len(pool.items)} candidates (`{pool.weights_version}`).")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about candidates, screening, interviews, next steps...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        resume = st.session_state.chat_resume or st.session_state.get("last_resume", "")
        job = st.session_state.chat_job or st.session_state.get("last_job", "")
        context = {
            "resume": resume,
            "job": job,
            "report": st.session_state.get("last_report"),
            "pool": st.session_state.get("pool"),
            "workflow": st.session_state.get("workflow"),
            "focus_item": st.session_state.get("focus_item"),
        }
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = pipeline.chat(prompt, context)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


def page_recruiter():
    st.title("Recruiter workspace")
    pipeline = require_pipeline()

    with st.form("screen_form"):
        c1, c2, c3 = st.columns([1.6, 1.2, 1])
        source = c1.radio("Candidate source", ["Upload resumes", "Dataset pool"], horizontal=True)
        job_mode = c1.radio("Job definition", ["Category", "Custom description"], horizontal=True)
        category = None
        job_text = ""
        uploaded_files = []
        if source == "Upload resumes":
            uploaded_files = c1.file_uploader(
                "Upload resumes (txt, pdf, docx)",
                type=["txt", "docx", "pdf"],
                accept_multiple_files=True,
                help="Each uploaded file is treated as one candidate.",
            )
        else:
            n_candidates = c2.slider("Candidates to screen", 4, 20, 8, step=2)
        if job_mode == "Category":
            category = c1.selectbox("Role", [""] + pipeline.categories(), index=1)
        else:
            job_text = c1.text_area("Job description", height=100)
        use_semantic = c3.checkbox("Semantic skill matching", value=True)
        submitted = st.form_submit_button("Screen candidates", type="primary", use_container_width=True)

    if submitted:
        if not category and not job_text.strip():
            st.warning("Choose a role or paste a job description.")
            st.stop()
        if source == "Upload resumes":
            if not uploaded_files:
                st.warning("Upload at least one resume file.")
                st.stop()
            candidates = []
            skipped = 0
            for f in uploaded_files:
                text = _extract_upload(f)
                if not text.strip():
                    skipped += 1
                    continue
                candidates.append(Candidate(resume_id=f.name, name=f.name, resume_text=text))
            if not candidates:
                st.error("None of the uploaded files contained extractable text (PDFs may need `pypdf`, DOCX `python-docx`).")
                st.stop()
            if skipped:
                st.caption(f"Skipped {skipped} file(s) with no extractable text.")
            job = job_text if job_text.strip() else _focused_job(pipeline, category)
        else:
            df = _load_dataframe()
            if category:
                pool_df = df[df["Category"] == category]
                if len(pool_df) < n_candidates:
                    pool_df = df.sample(n=n_candidates, random_state=42)
                job = job_text if job_text.strip() else _focused_job(pipeline, category)
            else:
                pool_df = df.sample(n=n_candidates, random_state=42)
                job = job_text
            candidates = candidates_from_dataframe(pool_df, n=n_candidates)
        with st.spinner(f"Screening {len(candidates)} candidates (semantic={use_semantic})..."):
            screener = CandidateScreener(pipeline, semantic=use_semantic)
            pool = screener.screen(candidates, job)
        st.session_state["pool"] = pool
        st.session_state["focus_item"] = pool.items[0] if pool.items else None
        st.session_state["workflow"] = HiringWorkflow()
        st.success("Screening complete.")

    pool = st.session_state.get("pool")
    if pool is None:
        st.info("Run a screening above to see the ranked candidate pool.")
        return

    st.markdown(f"### Ranked pool with `{pool.weights_version}` weights, semantic={pool.semantic}")
    rows = [
        {
            "Rank": i.rank,
            "Candidate": i.candidate.resume_id,
            "Score": f"{i.score:.1f}",
            "Verdict": i.verdict,
            "Predicted role": i.predicted_category,
            "Matched": ", ".join(i.matched_skills[:5]),
            "Missing": ", ".join(i.missing_skills[:5]),
        }
        for i in pool.items
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    focus_id = st.selectbox(
        "Focus candidate", [i.candidate.resume_id for i in pool.items],
        index=0, key="focus_select",
    )
    focus_item = next(i for i in pool.items if i.candidate.resume_id == focus_id)
    st.session_state["focus_item"] = focus_item
    with st.expander("Why this ranking? (evidence for focus candidate)"):
        for reason in focus_item.reasons:
            st.markdown(f"- {reason['text']}")
            for ev in reason["evidence"][:2]:
                st.caption(f"   Evidence: {ev[:180]}")

    tab1, tab2 = st.tabs(["Compare", "Interview questions"])

    with tab1:
        if len(pool.items) >= 2:
            pair_b = st.selectbox(
                "Compare focus vs", [i.candidate.resume_id for i in pool.items if i.candidate.resume_id != focus_id]
            )
            other = next(i for i in pool.items if i.candidate.resume_id == pair_b)
            comparison = CandidateComparator(pipeline).compare(focus_item, other)
            st.markdown(f"**{comparison['candidate_a']} ({comparison['score_a']}) vs "
                        f"{comparison['candidate_b']} ({comparison['score_b']})**")
            for row in comparison["criteria"]:
                st.markdown(f"- **{row['criterion']}**: {row['a_value']:.0f}% vs {row['b_value']:.0f}% → {row['leader']}")
            st.info(comparison["summary"])

    with tab2:
        if st.button("Generate interview questions", type="primary"):
            generator = QuestionGenerator(pipeline)
            questions = generator.generate(focus_item, pool.job_text)
            for i, q in enumerate(questions, start=1):
                st.markdown(f"**{i}. [{q.kind}]** {q.question}")
                if q.source_evidence:
                    st.caption(f"Evidence: {q.source_evidence[:200]}")
                st.markdown("")

    st.markdown("### Next hiring actions")
    workflow = st.session_state.get("workflow") or HiringWorkflow()
    st.session_state["workflow"] = workflow
    for item in pool.items[:5]:
        actions = workflow.next_actions(item)
        steps = "; ".join(f"**{a['action']}** ({a['detail']})" for a in actions)
        st.markdown(f"- **{item.candidate.resume_id}**: {steps}")


def _load_dataframe():
    from src.data import load_dataset

    return load_dataset("data/resumes_dataset.jsonl")


def _focused_job(pipeline, category: str) -> str:
    generic = {
        "agile", "scrum", "pmbok", "onboarding", "mentoring", "team leadership", "storage",
        "testing", "reporting", "documentation", "microsoft office", "excel", "wan", "monitoring",
        "code review", "linux", "git", "github",
    }
    skills = [s for s in pipeline.metadata.get("top_skills", {}).get(category, []) if s not in generic][:10]
    return "We are hiring a " + category + ". Required skills: " + ", ".join(skills) + "."


def page_evaluation():
    st.title("Model evaluation")
    pipeline = require_pipeline()

    meta = pipeline.metadata
    st.markdown(
        f"**Model** `{meta.get('model_id', '?')}` v{meta.get('version', '?')} "
        f"| dataset sha `{meta.get('dataset_sha', '?')}` | trained {meta.get('trained_at', '?')}"
    )
    st.markdown(f"**Weights in use**: `{pipeline.weights_version}` = {pipeline.calibrated_weights or config_fixed()}")
    st.markdown(
        "Run `python scripts/evaluate.py` to regenerate the full report, and "
        "`python scripts/calibrate.py` to re-fit the match weights from labelled hiring outcomes."
    )

    report_path = config.ARTIFACT_EVAL_REPORT
    if report_path.exists():
        import json

        report = json.loads(report_path.read_text(encoding="utf-8"))
        cls, match, ret = report["classification"], report["matching"], report["retrieval"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Classification accuracy", f"{cls.get('accuracy', 0):.3f}")
        c2.metric("Macro F1", f"{cls.get('macro_f1', 0):.3f}")
        c3.metric("Matching NDCG@10", f"{match.get('ndcg_10', 0):.3f}")
        c4.metric("Retrieval MRR", f"{ret.get('mrr', 0):.3f}")

        st.markdown("### Per-class precision / recall / F1")
        per = sorted(cls.get("per_class", {}).items(), key=lambda kv: kv[1]["f1"])
        st.dataframe(
            pd.DataFrame(
                [
                    {"Category": cat, "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"], "Support": int(m["support"])}
                    for cat, m in per
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No `models/eval_report.json` yet. Run `python scripts/evaluate.py`.")


def config_fixed():
    from src import config

    return config.MATCH_WEIGHTS


PAGES = {
    "Overview": page_overview,
    "Evaluate resume": page_evaluate,
    "Chat assistant": page_chat,
    "Recruiter workspace": page_recruiter,
    "Model evaluation": page_evaluation,
}


def main() -> None:
    auth = AuthManager()
    auth.store.ensure_default_user()
    if not auth.is_authenticated():
        if not auth.render_login():
            st.stop()

    user = auth.current_user()
    with st.sidebar:
        st.title("ResumeFit AI")
        choice = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
        st.markdown("---")
        st.caption(f"Signed in as **{user.username}** ({user.role})")
        if st.button("Sign out"):
            auth.logout()
            st.rerun()
    PAGES[choice]()


if __name__ == "__main__":
    main()
