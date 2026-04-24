"""Tests for ssdiff/utils/text.py and ssdiff/lang_config.py."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from ssdiff.lang_config import (
    LANG_TO_MODEL,
    LANGUAGES,
    get_config,
    lang_to_model,
)
from ssdiff.utils.text import (
    PreprocessedDoc,
    SpacyModelNotInstalledError,
    build_docs_from_preprocessed,
    load_spacy,
    load_stopwords,
    preprocess_texts,
)


# ---------------------------------------------------------------------------
# load_stopwords
# ---------------------------------------------------------------------------


def test_load_stopwords_pl_returns_list():
    """load_stopwords('pl') returns a list of strings."""
    sw = load_stopwords("pl")
    assert isinstance(sw, list)
    assert all(isinstance(w, str) for w in sw)


def test_load_stopwords_pl_known_words():
    """Polish stopwords include 'jest' and 'i'."""
    sw = load_stopwords("pl")
    assert "jest" in sw, "'jest' must be a Polish stopword"
    assert "i" in sw, "'i' must be a Polish stopword"


def test_load_stopwords_en_known_word():
    """English stopwords (spaCy built-in) include 'the'."""
    sw = load_stopwords("en")
    assert isinstance(sw, list)
    assert "the" in sw, "'the' must be in English stopwords"


def test_load_stopwords_lowercase():
    """load_stopwords returns all-lowercase entries by default."""
    sw = load_stopwords("pl")
    for w in sw:
        assert w == w.lower(), f"Stopword {w!r} is not lowercase"


# ---------------------------------------------------------------------------
# lang_to_model — aliasing and normalisation
# ---------------------------------------------------------------------------


def test_lang_to_model_iso_code():
    assert lang_to_model("pl") == "pl_core_news_lg"


def test_lang_to_model_full_name():
    """Full English name 'polish' (case-insensitive) resolves correctly."""
    assert lang_to_model("polish") == "pl_core_news_lg"


def test_lang_to_model_uppercase():
    """Uppercase 'PL' resolves to correct model."""
    assert lang_to_model("PL") == "pl_core_news_lg"


def test_lang_to_model_whitespace():
    """Leading/trailing whitespace is stripped."""
    assert lang_to_model("  pl  ") == "pl_core_news_lg"


def test_lang_to_model_unknown_raises():
    """Unknown language code raises ValueError."""
    with pytest.raises(ValueError):
        lang_to_model("unknown_lang_xyz")


# ---------------------------------------------------------------------------
# get_config and bad_token_re
# ---------------------------------------------------------------------------


def test_get_config_pl_has_bad_token_re():
    """get_config('pl') returns a LangConfig with a valid bad_token_re."""
    cfg = get_config("pl")
    assert cfg.bad_token_re is not None


def test_get_config_bad_token_re_matches_digit_token():
    """bad_token_re matches tokens containing digits like 'ABC123'."""
    cfg = get_config("pl")
    assert cfg.bad_token_re.match("ABC123"), "ABC123 must match bad_token_re"


def test_get_config_bad_token_re_no_match_lowercase():
    """bad_token_re does not match plain lowercase tokens like 'kraj'."""
    cfg = get_config("pl")
    assert not cfg.bad_token_re.match("kraj"), "'kraj' must not match bad_token_re"


# ---------------------------------------------------------------------------
# LANG_TO_MODEL / LANGUAGES consistency
# ---------------------------------------------------------------------------


def test_lang_to_model_keys_subset_of_languages():
    """LANG_TO_MODEL.keys() is a subset of LANGUAGES.keys()."""
    assert set(LANG_TO_MODEL.keys()) <= set(LANGUAGES.keys())


# ---------------------------------------------------------------------------
# load_spacy — error path (no actual model download)
# ---------------------------------------------------------------------------


def test_load_spacy_nonexistent_model_raises():
    """load_spacy raises SpacyModelNotInstalledError for unknown models."""
    with pytest.raises(SpacyModelNotInstalledError) as exc_info:
        load_spacy("nonexistent_model_xyz_abc")
    assert exc_info.value.model == "nonexistent_model_xyz_abc"


# ---------------------------------------------------------------------------
# preprocess_texts — flat mode using fake_nlp
# ---------------------------------------------------------------------------


def test_preprocess_texts_basic(fake_nlp):
    """Stopword-filtered preprocess_texts returns correct doc_lemmas."""
    result = preprocess_texts(
        ["Kraj jest piekny."],
        nlp=fake_nlp,
        stopwords=["jest"],
    )
    assert len(result) == 1
    doc = result[0]
    assert isinstance(doc, PreprocessedDoc)
    # fake_nlp splits on whitespace → tokens are: Kraj jest piekny.
    # lowercased lemmas: kraj, jest, piekny.
    # "jest" filtered out; "piekny." is not a stopword
    assert "kraj" in doc.doc_lemmas
    assert "jest" not in doc.doc_lemmas


def test_preprocess_texts_empty_string(fake_nlp):
    """Empty string input yields PreprocessedDoc with empty doc_lemmas."""
    result = preprocess_texts([""], nlp=fake_nlp, stopwords=[])
    assert len(result) == 1
    doc = result[0]
    assert isinstance(doc, PreprocessedDoc)
    assert doc.doc_lemmas == []


def test_preprocess_texts_none_input(fake_nlp):
    """None inputs are sanitized (converted to empty string) — not dropped."""
    result = preprocess_texts([None], nlp=fake_nlp, stopwords=[])
    assert len(result) == 1
    doc = result[0]
    assert isinstance(doc, PreprocessedDoc)
    assert doc.doc_lemmas == []


def test_preprocess_texts_nan_input(fake_nlp):
    """NaN float is misclassified as profile mode by _is_profile_input and raises TypeError.

    This is a known limitation: NaN handling only works for flat-list strings, not
    for raw float NaN values at the top level. Only str/None/bytes are sanitized in
    flat mode. NaN as a non-string, non-bytes element triggers profile mode and crashes.
    """
    with pytest.raises(TypeError):
        preprocess_texts([float("nan")], nlp=fake_nlp, stopwords=[])


def test_preprocess_texts_returns_one_result_per_input(fake_nlp):
    """Output length matches input length."""
    texts = ["Kraj jest piekny.", "Narod i dom.", "Szybki bieg."]
    result = preprocess_texts(texts, nlp=fake_nlp, stopwords=[])
    assert len(result) == len(texts)
    for doc in result:
        assert isinstance(doc, PreprocessedDoc)


# ---------------------------------------------------------------------------
# build_docs_from_preprocessed
# ---------------------------------------------------------------------------


def test_build_docs_from_preprocessed_returns_flat_lemmas(sample_preprocessed_docs):
    """build_docs_from_preprocessed returns list of doc_lemmas lists."""
    docs = build_docs_from_preprocessed(sample_preprocessed_docs)
    assert len(docs) == len(sample_preprocessed_docs)
    for i, lemmas in enumerate(docs):
        assert lemmas == sample_preprocessed_docs[i].doc_lemmas


def test_build_docs_from_preprocessed_empty_input():
    """Empty input returns empty list."""
    result = build_docs_from_preprocessed([])
    assert result == []
