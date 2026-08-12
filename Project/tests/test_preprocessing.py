from __future__ import annotations

from src.utils import clean_text, normalize_text, sentence_split, tokenize, chunks


def test_clean_text_removes_contacts_and_punctuation():
    text = "Contact me at jane@x.com or +1 555 123 4567 https://example.com"
    cleaned = clean_text(text)
    assert "jane" not in cleaned
    assert "555" not in cleaned
    assert "http" not in cleaned


def test_clean_text_lowercases():
    assert clean_text("Python Developer AWS") == "python developer aws"


def test_tokenize_filters_stopwords():
    tokens = tokenize("I am a senior java developer with strong skills")
    assert "java" in tokens
    assert "developer" in tokens
    assert "am" not in tokens
    assert "a" not in tokens


def test_sentence_split():
    sents = sentence_split("First sentence here about something. Second sentence longer. Third one too.")
    assert len(sents) == 3
    assert sents[0].startswith("First")


def test_normalize_text_preserves_internal_punctuation():
    assert normalize_text("C++ / .NET  Developer") == "c++ / .net developer"


def test_chunks_returns_at_least_one_chunk():
    assert chunks("") == []
    assert len(chunks("One two three. Four five six. Seven eight nine.", size=3)) == 1