from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from src.profile import extract_profile

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
        "screen",
        [
            re.compile(r"\bscreen(?:ing)?\b", re.I),
            re.compile(r"\bshortlist\b", re.I),
            re.compile(r"\bpool\b", re.I),
            re.compile(r"\brank(?:ing)?\b", re.I),
            re.compile(r"\bwho.*(rank|list).*candidates?\b", re.I),
            re.compile(r"\b(top|best)\s+candidates\b", re.I),
        ],
    ),
    (
        "evaluate",
        [
            re.compile(r"\b(evaluate|assess|score|rate)\b", re.I),
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
            re.compile(r"\b(compare|versus|vs\.?)\b", re.I),
            re.compile(r"\b(which candidate is (better|stronger)|who is (better|stronger))\b", re.I),
            re.compile(r"\bcandidate\s+comparison\b", re.I),
        ],
    ),
    (
        "interview_questions",
        [
            re.compile(r"\binterview\s+questions?\b", re.I),
            re.compile(r"\bquestions?\s+(to\s+|for\s+)?ask\b", re.I),
        ],
    ),
    (
        "next_action",
        [
            re.compile(r"\bnext\s+(step|action)\b", re.I),
            re.compile(r"\bwhat.*next\b", re.I),
            re.compile(r"\bworkflow\b", re.I),
            re.compile(r"\brecommend.*(action|hire|next)\b", re.I),
        ],
    ),
    (
        "similar",
        [re.compile(r"\b(find|show|look for).*(similar|other|profiles|resumes|candidates)\b", re.I)],
    ),
    (
        "contact",
        [
            re.compile(r"\b(contact|email|e-?mail|phone|mobile|reach|linkedin|github|portfolio|website|url)\b", re.I),
        ],
    ),
    (
        "experience",
        [
            re.compile(r"\byears?\s+of\s+experience\b", re.I),
            re.compile(r"\bhow (many|long).*experience\b", re.I),
            re.compile(r"\bwork history\b", re.I),
            re.compile(r"\bexperience level\b", re.I),
        ],
    ),
    (
        "education",
        [
            re.compile(r"\b(education|degree|university|college|school|qualification)\b", re.I),
        ],
    ),
    (
        "background",
        [
            re.compile(r"\b(background|overview|profile|introduce)\b", re.I),
            re.compile(r"\bwho (is|'s) (she|he|this|the candidate|them|her|him|the person)\b", re.I),
        ],
    ),
    (
        "resume_qa",
        [
            re.compile(r"\b(summary|about|tell me|projects?|build|built|what did)\b", re.I),
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
            "- **Profile**: contact details, experience, education, and a candidate summary\n"
            "- **Requirements**: typical skills for a role like 'Machine Learning Engineer'\n"
            "- **Compare**: benchmark against similar candidates, or pairwise-compare two shortlisted candidates\n"
            "- **Screen/rank**: rank a whole candidate pool against a job description\n"
            "- **Interview questions**: generate questions with resume/JD evidence for a candidate\n"
            "- **Next action**: recommend the next step in the hiring workflow\n"
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
            f"The score is a weighted blend of three signals "
            f"(weights: `{report.match.components.weights_version}`): "
            f"skill coverage ({report.match.components.skill_coverage * 100:.0f}%), "
            f"semantic similarity to the job text ({report.match.components.embedding_similarity * 100:.0f}%), "
            f"and category alignment ({report.match.components.category_affinity * 100:.0f}%).",
        ]
        if report.match.semantic_matches:
            semantic = ", ".join(m["skill"] for m in report.match.semantic_matches[:5])
            lines.append(f"Semantic matching recovered skills beyond the exact lexicon: {semantic}.")
        if report.match.transferable_skills:
            transfer = ", ".join(f"{s}~{t}" for s, t, _ in report.match.transferable_skills[:3])
            lines.append(f"Transferable skills bridging gaps: {transfer}.")
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
        from src.hiring import CandidateComparator, ScreeningPool

        pool = context.get("pool")
        if isinstance(pool, ScreeningPool) and len(pool.items) >= 2:
            comparator = CandidateComparator(self.pipeline)
            first, second = pool.items[:2]
            comparison = comparator.compare(first, second)
            lines = [
                f"**Comparing {comparison['candidate_a']} vs {comparison['candidate_b']}** "
                f"({comparison['score_a']:.1f} vs {comparison['score_b']:.1f})\n"
            ]
            for row in comparison["criteria"]:
                lines.append(
                    f"- **{row['criterion']}**: {row['a_value']:.0f}% vs {row['b_value']:.0f}% "
                    f"→ leader {row['leader']}"
                )
            if comparison["unique_skills_a"]:
                lines.append(f"- {comparison['candidate_a']} uniquely has: {', '.join(comparison['unique_skills_a'][:5])}")
            if comparison["unique_skills_b"]:
                lines.append(f"- {comparison['candidate_b']} uniquely has: {', '.join(comparison['unique_skills_b'][:5])}")
            lines.append(f"\n{comparison['summary']}")
            return "\n".join(lines)

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

    def _on_screen(self, message: str, context: dict[str, Any]) -> str:
        from src.hiring import CandidateScreener, ScreeningPool

        pool = context.get("pool")
        if not isinstance(pool, ScreeningPool):
            if self._context_resume(context):
                return self._on_evaluate(message, context)
            return (
                "I can rank a candidate pool against a job description. "
                "Ask me to 'screen the candidates' from the Recruiter workspace, or provide candidates and a job first."
            )
        lines = [f"**Ranked pool** ({pool.weights_version} weights, semantic matching {'on' if pool.semantic else 'off'}):\n"]
        for item in pool.items[:10]:
            lines.append(
                f"- #{item.rank} **{item.candidate.resume_id}** - {item.score:.1f}/100 "
                f"({item.verdict}, {item.predicted_category})"
            )
        shortlist = pool.shortlist()
        if shortlist:
            lines.append(
                f"\nShortlist: {', '.join(i.candidate.resume_id for i in shortlist)} "
                f"({len(shortlist)} candidates advance)."
            )
        return "\n".join(lines)

    def _on_interview_questions(self, message: str, context: dict[str, Any]) -> str:
        from src.hiring import ScreeningPool
        from src.interviews import QuestionGenerator

        pool = context.get("pool")
        if not isinstance(pool, ScreeningPool):
            return "Screen candidates first (ask me to 'screen the candidates'), then ask for interview questions."
        item = context.get("focus_item") or (pool.items[0] if pool.items else None)
        if item is None:
            return "No candidate selected. Screen a pool first."
        generator = QuestionGenerator(self.pipeline)
        questions = generator.generate(item, pool.job_text)
        lines = [f"**Interview questions for {item.candidate.resume_id}**\n"]
        for i, q in enumerate(questions, start=1):
            ev = f"\n  - Evidence: {q.source_evidence[:160]}" if q.source_evidence else ""
            lines.append(f"{i}. [{q.kind}] {q.question}{ev}")
        return "\n\n".join(lines)

    def _on_next_action(self, message: str, context: dict[str, Any]) -> str:
        from src.hiring import HiringWorkflow, ScreeningPool

        pool = context.get("pool")
        workflow = context.get("workflow") or HiringWorkflow()
        context["workflow"] = workflow
        if not isinstance(pool, ScreeningPool) or not pool.items:
            return "Screen candidates first, then I can recommend the next hiring action per candidate."
        lines = []
        for item in pool.items[:5]:
            actions = workflow.next_actions(item)
            steps = ", ".join(f"**{a['action']}** ({a['detail']})" for a in actions)
            lines.append(f"- {item.candidate.resume_id} (stage: {workflow.current_stage(item.candidate.resume_id)}): {steps}")
        return "\n".join(lines)

    def _on_contact(self, message: str, context: dict[str, Any]) -> str:
        profile = self._cached_profile(context)
        if profile is None:
            return "I need a resume first to pull contact details from."
        lines = ["**Contact details**"]
        if profile.name:
            lines.append(f"- Name: {profile.name}")
        if profile.email:
            lines.append(f"- Email: {profile.email}")
        if profile.phone:
            lines.append(f"- Phone: {profile.phone}")
        if profile.links:
            lines.append("- Links: " + ", ".join(profile.links))
        if len(lines) == 1:
            return "I couldn't find contact details in this resume."
        return "\n".join(lines)

    def _on_experience(self, message: str, context: dict[str, Any]) -> str:
        profile = self._cached_profile(context)
        if profile is None:
            return "I need a resume first to look at work experience."
        if profile.experience_years:
            return (
                f"Based on the resume, this candidate has about "
                f"**{profile.experience_years:.0f} years** of experience."
            )
        answer = self._rag_answer(message, self._context_resume(context), context, threshold=0.2)
        if answer is None:
            return "I couldn't find an explicit experience count. Ask what they built or worked on."
        return answer

    def _on_education(self, message: str, context: dict[str, Any]) -> str:
        profile = self._cached_profile(context)
        if profile is None:
            return "I need a resume first to look at education."
        if profile.education:
            return "**Education**\n- " + "\n- ".join(profile.education)
        answer = self._rag_answer(message, self._context_resume(context), context, threshold=0.2)
        if answer is None:
            return "I couldn't find education details in this resume."
        return answer

    def _on_background(self, message: str, context: dict[str, Any]) -> str:
        profile = self._cached_profile(context)
        if profile is None:
            return "I need a resume first to summarize the candidate."
        lines = []
        head = profile.name or "This candidate"
        if profile.summary:
            lines.append(f"{head}: {profile.summary}")
        if profile.top_skills:
            lines.append(f"\nTop skills: {', '.join(profile.top_skills[:8])}")
        if profile.experience_years:
            lines.append(f"\nExperience: {profile.experience_years:.0f} years")
        if profile.education:
            lines.append(f"\nEducation: {profile.education[0]}")
        if lines:
            return "\n".join(lines)
        return "I couldn't extract a profile summary from this resume."

    def _cached_profile(self, context: dict[str, Any]):
        resume = self._context_resume(context)
        if not resume:
            return None
        profile = context.get("_profile")
        if profile is None:
            profile = extract_profile(resume)
            context["_profile"] = profile
        return profile

    def _on_resume_qa(self, message: str, context: dict[str, Any]) -> str:
        resume = self._context_resume(context)
        if not resume:
            return "I need a resume to answer questions about it."
        answer = self._rag_answer(message, resume, context, threshold=0.15)
        if answer is None:
            return (
                "I couldn't find a clear answer in the resume. "
                "Try asking about contact details, experience, education, skills, or the summary."
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
        answer = self._rag_answer(message, resume, context)
        if answer is None:
            return self._on_out_of_scope(message, context)
        return answer

    def _rag_answer(self, query: str, resume: str, context: dict[str, Any], threshold: float = 0.3) -> str | None:
        from src.models import build_sentence_index, search_sentences

        sents, vecs = build_sentence_index(resume, self.pipeline.embedder)
        if len(sents) == 0:
            return None
        qvec = self.pipeline.embedder.encode_one(query)
        hits = search_sentences(qvec, vecs, k=2, threshold=threshold)
        if not hits:
            return None
        excerpt = " … ".join(sents[idx] for idx, _ in hits)
        return f"Based on the resume:\n\n> {excerpt}\n\n{self._followup(context)}"

    def _followup(self, context: dict[str, Any]) -> str:
        options = []
        if self._context_resume(context) and context.get("report") is None:
            options.append("score this resume against a job description")
        options.append("list its skills and gaps")
        if options:
            return "Want me to " + " or ".join(options) + "?"
        return "Anything else you'd like to know about this candidate?"
