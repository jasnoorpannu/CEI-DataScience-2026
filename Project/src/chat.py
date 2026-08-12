from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.pipeline import Pipeline

_PATTERNS: list[tuple[str, list[re.Pattern]]] = [
    (
        "greet",
        [re.compile(r"\b(hi|hello|hey|namaste|good morning|good afternoon|good evening)\b", re.I)],
    ),
    (
        "help",
        [re.compile(r"\b(help|what can you do|how do you work|capabilit)\b", re.I)],
    ),
    (
        "explain",
        [
            re.compile(r"\b(why|explain|reason|because|how is)\b", re.I),
            re.compile(r"\bhow did you (score|rate|decide)\b", re.I),
        ],
    ),
    (
        "evaluate",
        [
            re.compile(r"\b(evaluate|assess|score|rate|screen|shortlist|rank)\b", re.I),
            re.compile(r"\bhow.*(fit|match)\b", re.I),
        ],
    ),
    (
        "skills",
        [re.compile(r"\bskills?\b", re.I), re.compile(r"\b(technolog|stack|keywords|expertise)\b", re.I)],
    ),
    (
        "gaps",
        [
            re.compile(r"\b(gap|missing|lack|shortfall|weakness|improve|better|strengthen)\b", re.I),
        ],
    ),
    (
        "category",
        [
            re.compile(r"\b(role|job title|category|position|best fit|which job|what job)\b", re.I),
        ],
    ),
    (
        "requirements",
        [
            re.compile(r"\b(require|requirements|need|needed|must have|qualification)\b", re.I),
        ],
    ),
    (
        "compare",
        [
            re.compile(r"\b(compare|versus|vs\.?|similar|candidate|benchmark|peers|percentile)\b", re.I),
        ],
    ),
    (
        "similar",
        [re.compile(r"\b(find|show|look for).*(similar|other|profiles|resumes|candidates)\b", re.I)],
    ),
    (
        "resume_qa",
        [
            re.compile(r"\b(summary|experience|education|work history|background|about|tell me)\b", re.I),
            re.compile(r"\b(build|built|projects?|degree|qualification)\b", re.I),
        ],
    ),
    (
        "identity",
        [re.compile(r"\b(what is your name|who are you|your name|about yourself)\b", re.I)],
    ),
    (
        "out_of_scope",
        [
            re.compile(
                r"\b(president|prime minister|capital of|population|weather|currency|history|"
                r"politics|sports|celebrity|famous|invented|discovered|national anthem|flag of)\b",
                re.I,
            ),
        ],
    ),
]


def detect_intent(message: str) -> str:
    for intent, patterns in _PATTERNS:
        for pat in patterns:
            if pat.search(message):
                return intent
    return "fallback"


class ResumeAssistant:
    def __init__(self, pipeline: "Pipeline") -> None:
        self.pipeline = pipeline

    def respond(self, message: str, context: dict[str, Any]) -> str:
        intent = detect_intent(message)
        handler = getattr(self, f"_on_{intent}", self._on_fallback)
        try:
            return handler(message, context)
        except Exception as exc:
            return f"I ran into an issue: {exc}. Could you rephrase or provide your resume text?"

    def _context_resume(self, context: dict[str, Any]) -> str:
        return (context.get("resume") or "").strip()

    def _context_report(self, context: dict[str, Any]):
        return context.get("report")

    def _on_greet(self, message: str, context: dict[str, Any]) -> str:
        return (
            "Hi! I can help you evaluate resumes against job requirements. "
            "Share a resume (paste text or a file), then ask me to score it, explain the score, "
            "list skills or gaps, compare with similar candidates, or check what a role requires."
        )

    def _on_identity(self, message: str, context: dict[str, Any]) -> str:
        return (
            "I'm the ResumeFit AI assistant. I help evaluate resumes against job requirements - "
            "ask me to score one, explain a score, list skills or gaps, or ask what a role requires."
        )

    def _on_out_of_scope(self, message: str, context: dict[str, Any]) -> str:
        return (
            "I can only answer questions about a resume or a job's requirements, so I don't have an answer for that. "
            "Try asking me to evaluate a resume, explain a score, list skills or gaps, or check a role's requirements."
        )

    def _on_help(self, message: str, context: dict[str, Any]) -> str:
        return (
            "I can help you with:\n"
            "- **Evaluate/score**: match a resume against a job description\n"
            "- **Explain**: why a score or category was chosen\n"
            "- **Skills & gaps**: what the resume covers and what is missing\n"
            "- **Requirements**: typical skills for a role like 'Machine Learning Engineer'\n"
            "- **Compare**: benchmark against similar candidates in the dataset\n"
            "Paste a resume or ask for a role's requirements to get started."
        )

    def _on_evaluate(self, message: str, context: dict[str, Any]) -> str:
        resume = self._context_resume(context)
        if not resume:
            return "I need a resume first. Paste your resume text or upload a file, then ask me to evaluate it."
        job = (context.get("job") or "").strip()
        report = self.pipeline.feedback(resume, job or None)
        context["report"] = report
        return (
            f"**Overall score: {report.overall_score}/100 - {report.grade}**\n\n"
            f"Predicted category: **{report.predicted_category}** "
            f"({report.predicted_confidence * 100:.0f}% confidence)\n\n"
            f"Breakdown:\n"
            f"- Skill coverage: {report.match.components.skill_coverage * 100:.0f}%\n"
            f"- Text similarity: {report.match.components.embedding_similarity * 100:.0f}%\n"
            f"- Category alignment: {report.match.components.category_affinity * 100:.0f}%\n\n"
            f"{report.summary}"
        )

    def _on_explain(self, message: str, context: dict[str, Any]) -> str:
        resume = self._context_resume(context)
        if not resume:
            return "I need a resume first to explain the scoring."
        job = (context.get("job") or "").strip()
        report = context.get("report") or self.pipeline.feedback(resume, job or None)
        context["report"] = report
        lines = [
            f"The score is a weighted blend of three signals: "
            f"skill coverage ({report.match.components.skill_coverage * 100:.0f}%), "
            f"semantic similarity to the job text ({report.match.components.embedding_similarity * 100:.0f}%), "
            f"and category alignment ({report.match.components.category_affinity * 100:.0f}%).",
        ]
        if report.top_terms:
            terms = ", ".join(t for t, _ in report.top_terms[:8])
            lines.append(f"The classifier leaned toward **{report.predicted_category}** because of terms like: {terms}.")
        if report.strengths:
            strong = ", ".join(s.skill for s in report.strengths[:5])
            lines.append(f"Strongest matched skills: {strong}.")
        if report.gaps:
            miss = ", ".join(g.skill for g in report.gaps[:5])
            lines.append(f"Main gaps: {miss}.")
        return "\n\n".join(lines)

    def _on_skills(self, message: str, context: dict[str, Any]) -> str:
        resume = self._context_resume(context)
        if not resume:
            return "I need a resume to extract skills from."
        report = context.get("report") or self.pipeline.feedback(resume, context.get("job") or None)
        context["report"] = report
        lines = ["**Skills detected on the resume:**"]
        lines.append(", ".join(sorted(report.match.resume_skills)) if report.match.resume_skills else "None detected.")
        if report.match.matched_skills:
            lines.append(f"\n**Matched with the job:** {', '.join(report.match.matched_skills)}")
        if report.match.missing_skills:
            lines.append(f"\n**Required but missing:** {', '.join(report.match.missing_skills)}")
        return "\n".join(lines)

    def _on_gaps(self, message: str, context: dict[str, Any]) -> str:
        resume = self._context_resume(context)
        if not resume:
            return "I need a resume first to analyze gaps."
        job = (context.get("job") or "").strip()
        report = context.get("report") or self.pipeline.feedback(resume, job or None)
        context["report"] = report
        if not report.gaps:
            return "No major gaps detected - the resume covers the required skills well."
        lines = ["Here are the gaps, ordered by importance:\n"]
        for gap in report.gaps[:8]:
            lines.append(f"- **{gap.skill}** (importance {gap.importance * 100:.0f}%)")
            if gap.suggestion:
                lines.append(f"  → {gap.suggestion}")
            if gap.peer_evidence:
                lines.append(f"  Example from a peer profile: \"{gap.peer_evidence[:180]}\"")
        return "\n".join(lines)

    def _on_category(self, message: str, context: dict[str, Any]) -> str:
        resume = self._context_resume(context)
        if not resume:
            return "I need a resume to determine the best-fitting role."
        cats = self.pipeline.predict_category(resume)
        lines = ["Predicted roles (confidence):"]
        for cat, prob in cats[:5]:
            lines.append(f"- **{cat}**: {prob * 100:.0f}%")
        return "\n".join(lines)

    def _on_requirements(self, message: str, context: dict[str, Any]) -> str:
        m = message.lower()
        resume = self._context_resume(context)
        category = None
        if "resume" in m or "my" in m or "this" in m:
            if resume:
                report = context.get("report") or self.pipeline.feedback(resume, context.get("job") or None)
                category = report.predicted_category
        if not category:
            from src.matching import parse_job_requirements

            req = parse_job_requirements(message)
            if req.skills:
                return f"Required skills I detected: {', '.join(req.skills)}."
            cats = self.pipeline.categories()
            matched = [c for c in cats if c.lower() in m]
            if matched:
                category = matched[0]
        if category:
            text = self.pipeline.requirements_for_category(category)
            return f"Typical requirements for **{category}**:\n\n{text}"
        return (
            "Tell me a role name (e.g., 'Machine Learning Engineer') or ask 'what does this job require?' "
            "with a job description."
        )

    def _on_compare(self, message: str, context: dict[str, Any]) -> str:
        resume = self._context_resume(context)
        if not resume:
            return "I need a resume to benchmark against similar candidates."
        job = (context.get("job") or "").strip()
        report = context.get("report") or self.pipeline.feedback(resume, job or None)
        context["report"] = report
        lines = [
            f"You are in the top {report.benchmark_percentile:.0f}% of similar profiles "
            f"for **{report.predicted_category}** in the reference dataset."
        ]
        if report.similar_candidates:
            lines.append("\nMost similar profiles in the dataset:")
            for cand in report.similar_candidates[:5]:
                lines.append(f"- {cand.resume_id} ({cand.category}) similarity {cand.score * 100:.0f}%")
        return "\n".join(lines)

    def _on_similar(self, message: str, context: dict[str, Any]) -> str:
        return self._on_compare(message, context)

    def _on_resume_qa(self, message: str, context: dict[str, Any]) -> str:
        resume = self._context_resume(context)
        if not resume:
            return "I need a resume to answer questions about it."
        answer = self._rag_answer(message, resume, threshold=0.15)
        if answer is None:
            return (
                "I couldn't find a clear answer in the resume. "
                "Try asking about the summary, skills, education, or experience."
            )
        return answer

    def _on_fallback(self, message: str, context: dict[str, Any]) -> str:
        resume = self._context_resume(context)
        if not resume:
            return (
                "I didn't fully catch that. You can ask me to evaluate a resume, explain a score, "
                "list skills/gaps, compare candidates, or ask about a role's requirements. "
                "You can also paste a resume so I can answer specific questions."
            )
        answer = self._rag_answer(message, resume)
        if answer is None:
            return self._on_out_of_scope(message, context)
        return answer

    def _rag_answer(self, query: str, resume: str, threshold: float = 0.3) -> str | None:
        from src.models import build_sentence_index, search_sentences

        sents, vecs = build_sentence_index(resume, self.pipeline.embedder)
        if len(sents) == 0:
            return None
        qvec = self.pipeline.embedder.encode_one(query)
        hits = search_sentences(qvec, vecs, k=3, threshold=threshold)
        if not hits:
            return None
        evidence = " ".join(sents[idx] for idx, _ in hits)
        return (
            "Based on the resume:\n\n"
            f"> {evidence}\n\n"
            "Want me to score this resume against a job description or list its skills and gaps?"
        )
