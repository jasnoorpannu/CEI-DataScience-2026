from __future__ import annotations

from dataclasses import dataclass

from src.hiring import ScreeningItem


@dataclass
class InterviewQuestion:
    kind: str
    question: str
    rationale: str = ""
    target_skill: str = ""
    source_evidence: str = ""
    difficulty: str = "medium"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "question": self.question,
            "rationale": self.rationale,
            "target_skill": self.target_skill,
            "source_evidence": self.source_evidence,
            "difficulty": self.difficulty,
        }


class QuestionGenerator:
    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def generate(self, item: ScreeningItem, job_text: str, n_technical: int = 5, n_behavioral: int = 2) -> list[InterviewQuestion]:
        questions: list[InterviewQuestion] = []
        report = item.report
        required = report.match.missing_skills if report else []
        matched = report.match.matched_skills if report else []

        if report:
            for skill in report.match.requirements.skills[:n_technical]:
                evidence = next((s.evidence for s in report.strengths if s.skill == skill), "")
                questions.append(
                    InterviewQuestion(
                        kind="technical",
                        question=f"Walk me through a project where you used **{skill}**. What problem did it solve and how did you evaluate the result?",
                        rationale="Validates hands-on experience with a core required skill.",
                        target_skill=skill,
                        source_evidence=evidence or "Required in the job description.",
                        difficulty="high" if skill in required else "medium",
                    )
                )
        for skill in required[:3]:
            questions.append(
                InterviewQuestion(
                    kind="evidence_gap",
                    question=f"The resume does not show direct evidence of **{skill}**, which is required for this role. Can you describe a specific way you have used {skill} in a real project?",
                    rationale="Probes a missing required skill and gives the candidate a chance to demonstrate transferable knowledge.",
                    target_skill=skill,
                    source_evidence=f"Missing from resume; required by the job description.",
                    difficulty="medium",
                )
            )
        behavioral = self._behavioral(item.predicted_category or item.candidate.category)
        for bq in behavioral[:n_behavioral]:
            questions.append(
                InterviewQuestion(
                    kind="behavioral",
                    question=bq,
                    rationale="Assesses collaboration and delivery behaviour for the target role.",
                    difficulty="medium",
                )
            )
        questions.append(
            InterviewQuestion(
                kind="screening",
                question="Are you available to start within the expected timeframe, and do you require work authorization support?",
                rationale="Knock-out question aligned with the hiring timeline.",
                difficulty="low",
            )
        )
        return questions

    def _behavioral(self, category: str) -> list[str]:
        base = [
            "Describe a time you had to deliver under a tight deadline. What did you prioritise and how did it end?",
            "Tell me about a disagreement with a teammate on approach. How did you resolve it?",
            "Describe a difficult problem you solved recently and the steps you took.",
        ]
        if category:
            return [
                f"Describe a project you delivered as a {category}. What was your specific contribution?",
                "How do you keep your skills current, and what is the most recent thing you learned?",
            ] + base
        return base
