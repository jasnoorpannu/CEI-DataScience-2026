from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import plotly.graph_objects as go

from src.feedback import FeedbackReport

st.set_page_config(page_title="ResumeFit AI", layout="wide")


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
    fig = go.Figure(
        go.Scatterpolar(r=values, theta=labels, fill="toself", name="Components")
    )
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

    st.markdown("### Recommendations")
    for i, rec in enumerate(report.recommendations, start=1):
        st.markdown(f"{i}. {rec}")

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
        st.code(
            "python scripts/train.py\npython -m streamlit run src/app.py",
            language="bash",
        )
        st.stop()
    return pipeline


def page_overview():
    st.title("ResumeFit AI")
    st.markdown(
        """
        An end-to-end system that **evaluates candidate resumes against job requirements**
        using machine learning, then produces **personalized, explainable feedback**
        through a **retrieval-based text generation pipeline**, plus a
        **conversational interface** for candidates and recruiters.
        """
    )

    if pipeline_ready():
        pipeline = get_pipeline()
        meta = pipeline.metadata
        c1, c2, c3 = st.columns(3)
        c1.metric("Resumes in dataset", meta.get("num_resumes", "?"))
        c2.metric("Job categories", meta.get("num_categories", "?"))
        c3.metric("Classifier test accuracy", f"{meta.get('test_accuracy', 0.0) * 100:.1f}%")
        st.success(
            f"Models loaded: {meta.get('backend', '?')} embeddings, "
            f"{meta.get('num_resumes', '?')} reference profiles indexed for retrieval."
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Resumes in dataset", "3,500")
        c2.metric("Job categories", "36")
        c3.metric("Pipeline", "ML + RAG")
        st.warning("No trained models found yet - run `python scripts/train.py` first.")

    st.markdown("### How it works")
    st.markdown(
        """
        1. **Classify** - a TF-IDF + Logistic Regression model predicts the best-fit role from the resume text.
        2. **Match** - the resume is scored 0-100 against the job requirements, blending skill coverage, semantic text similarity, and category alignment.
        3. **Feedback** - evidence sentences are retrieved from the resume and similar profiles to explain strengths, gaps, and recommendations.
        4. **Converse** - a retrieval-based assistant answers questions about the score, skills, gaps, and role requirements.
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
        resume_text = st.text_area(
            "Resume text",
            height=280,
            placeholder="Paste the full resume text here...",
        )
    with tab_upload:
        uploaded = st.file_uploader("Upload resume (txt, docx, pdf)", type=["txt", "docx", "pdf"])
        if uploaded is not None:
            resume_text = _extract_upload(uploaded)
            st.text_area("Extracted text", resume_text, height=280)

    st.divider()
    st.markdown("**Job description** *(optional - leave empty to auto-generate from the predicted role)*")
    job_text = st.text_area(
        "Job description",
        height=150,
        placeholder="Paste the job description...",
    )
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
                "content": "Hi! Paste your resume in the panel on the left (or ask about a role's requirements), then ask me to evaluate it, explain the score, list skills/gaps, or compare candidates.",
            }
        ]
    if "chat_resume" not in st.session_state:
        st.session_state.chat_resume = ""
    if "chat_job" not in st.session_state:
        st.session_state.chat_job = ""

    with st.sidebar:
        st.markdown("### Chat context")
        st.session_state.chat_resume = st.text_area(
            "Resume text (optional)",
            value=st.session_state.chat_resume,
            height=200,
            key="chat_resume_input",
        )
        st.session_state.chat_job = st.text_area(
            "Job description (optional)",
            value=st.session_state.chat_job,
            height=120,
            key="chat_job_input",
        )
        if st.session_state.get("last_resume") and not st.session_state.chat_resume:
            st.caption("Using the resume evaluated on the Evaluate page.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about the resume or a role...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        resume = st.session_state.chat_resume or st.session_state.get("last_resume", "")
        job = st.session_state.chat_job or st.session_state.get("last_job", "")
        report = st.session_state.get("last_report")
        context = {
            "resume": resume,
            "job": job,
            "report": report,
        }
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = pipeline.chat(prompt, context)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


PAGES = {
    "Overview": page_overview,
    "Evaluate resume": page_evaluate,
    "Chat assistant": page_chat,
}


def main() -> None:
    with st.sidebar:
        st.title("ResumeFit AI")
        choice = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
        st.markdown("---")
        st.caption("ML resume matching with retrieval-based explainable feedback.")
    PAGES[choice]()


if __name__ == "__main__":
    main()
