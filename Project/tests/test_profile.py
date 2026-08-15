from __future__ import annotations

from src.profile import extract_profile

HEADER = (
    "seivarya.in | seivarya.he@gmail.com | linkedin.com/in/seivarya | github.com/seivarya\n"
    "Technical Contributor, GigaVector (Open Source)\n\n"
    "5 years of experience building retrieval pipelines in Python, PyTorch and Elasticsearch.\n\n"
    "Education: B.Tech in Computer Science, IIT Delhi\n"
    "M.S. in Data Science, Georgia Tech\n\n"
    "Built a semantic search service used by 40+ teams."
)


def test_email_extracted():
    assert extract_profile(HEADER).email == "seivarya.he@gmail.com"


def test_links_extracted():
    profile = extract_profile(HEADER)
    assert "linkedin.com/in/seivarya" in profile.links
    assert "github.com/seivarya" in profile.links


def test_phone_extracted():
    text = "Reach me at +1 555-123-4567 or seivarya.he@gmail.com"
    assert extract_profile(text).phone == "+1 555-123-4567"


def test_name_guess_from_email():
    assert extract_profile(HEADER).name == "Seivarya He"


def test_name_line_kept():
    text = "Alex Chen\nMachine Learning Engineer\n3 years experience\n"
    assert extract_profile(text).name == "Alex Chen"


def test_experience_years():
    assert extract_profile(HEADER).experience_years == 5.0


def test_education_lines():
    profile = extract_profile(HEADER)
    assert len(profile.education) == 2
    assert any("B.Tech" in e for e in profile.education)
    assert any("M.S." in e for e in profile.education)


def test_top_skills_limited():
    skills = extract_profile(HEADER).top_skills
    assert isinstance(skills, list)
    assert len(skills) <= 10


def test_summary_line():
    assert "retrieval pipelines" in extract_profile(HEADER).summary


def test_empty_text():
    profile = extract_profile("")
    assert not profile.has_content()
