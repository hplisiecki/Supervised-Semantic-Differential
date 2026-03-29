"""Tests for ssdlite/utils/text.py — preprocessing pipeline."""

from __future__ import annotations

import pytest

from ssdlite.utils.text import (
    PreprocessedDoc,
    PreprocessedProfile,
    _keep_token,
    _is_profile_input,
    _sanitize_posts,
    preprocess_texts,
    build_docs_from_preprocessed,
    load_stopwords,
)
from tests.conftest import MockToken, FakeNlp


# ---------------------------------------------------------------------------
# _keep_token
# ---------------------------------------------------------------------------

class TestKeepToken:
    """Token-level filter logic."""

    def test_normal_word_kept(self):
        tok = MockToken(text="dom", lemma_="dom")
        assert _keep_token(tok, set()) is True

    def test_space_rejected(self):
        tok = MockToken(text=" ", lemma_=" ", is_space=True)
        assert _keep_token(tok, set()) is False

    def test_punct_rejected(self):
        tok = MockToken(text=".", lemma_=".", is_punct=True)
        assert _keep_token(tok, set()) is False

    def test_quote_rejected(self):
        tok = MockToken(text='"', lemma_='"', is_quote=True)
        assert _keep_token(tok, set()) is False

    def test_currency_rejected(self):
        tok = MockToken(text="$", lemma_="$", is_currency=True)
        assert _keep_token(tok, set()) is False

    def test_digit_rejected(self):
        tok = MockToken(text="123", lemma_="123", is_digit=True)
        assert _keep_token(tok, set()) is False

    def test_url_rejected(self):
        tok = MockToken(text="https://example.com", lemma_="https://example.com")
        assert _keep_token(tok, set()) is False

    def test_http_url_rejected(self):
        tok = MockToken(text="http://foo.bar/baz", lemma_="http://foo.bar/baz")
        assert _keep_token(tok, set()) is False

    def test_at_mention_rejected(self):
        tok = MockToken(text="@user123", lemma_="@user123")
        assert _keep_token(tok, set()) is False

    def test_stopword_rejected(self):
        tok = MockToken(text="jest", lemma_="jest")
        assert _keep_token(tok, {"jest"}) is False

    def test_empty_lemma_rejected(self):
        tok = MockToken(text="", lemma_="")
        assert _keep_token(tok, set()) is False

    def test_stopword_case_insensitive(self):
        tok = MockToken(text="Jest", lemma_="Jest")
        assert _keep_token(tok, {"jest"}) is False


# ---------------------------------------------------------------------------
# _is_profile_input / _sanitize_posts
# ---------------------------------------------------------------------------

class TestIsProfileInput:
    def test_flat_strings(self):
        assert _is_profile_input(["hello", "world"]) is False

    def test_nested_lists(self):
        assert _is_profile_input([["post1", "post2"], ["post3"]]) is True

    def test_empty_list(self):
        assert _is_profile_input([]) is False

    def test_none_entries_skipped(self):
        assert _is_profile_input([None, "hello"]) is False

    def test_none_then_list(self):
        assert _is_profile_input([None, ["post"]]) is True


class TestSanitizePosts:
    def test_normal_strings(self):
        assert _sanitize_posts(["hello", "world"]) == ["hello", "world"]

    def test_strips_whitespace(self):
        assert _sanitize_posts(["  hello  ", " world "]) == ["hello", "world"]

    def test_empty_strings_dropped(self):
        assert _sanitize_posts(["hello", "", "  ", "world"]) == ["hello", "world"]

    def test_bytes_decoded(self):
        assert _sanitize_posts([b"hello"]) == ["hello"]

    def test_none_input(self):
        assert _sanitize_posts(None) == []

    def test_none_entries_skipped(self):
        assert _sanitize_posts(["hello", None, "world"]) == ["hello", "world"]


# ---------------------------------------------------------------------------
# preprocess_texts — flat mode
# ---------------------------------------------------------------------------

class TestPreprocessTextsFlat:
    def test_basic_flat(self, fake_nlp):
        texts = ["kraj piekny dom", "narod wielki"]
        result = preprocess_texts(texts, fake_nlp)
        assert len(result) == 2
        assert all(isinstance(r, PreprocessedDoc) for r in result)

    def test_doc_lemmas_lowercased(self, fake_nlp):
        result = preprocess_texts(["Kraj Piekny"], fake_nlp)
        assert result[0].doc_lemmas == ["kraj", "piekny"]

    def test_raw_preserved(self, fake_nlp):
        result = preprocess_texts(["hello world"], fake_nlp)
        assert result[0].raw == "hello world"

    def test_sents_surface(self, fake_nlp):
        result = preprocess_texts(["hello world"], fake_nlp)
        assert result[0].sents_surface == ["hello world"]

    def test_none_becomes_empty(self, fake_nlp):
        result = preprocess_texts([None, "word"], fake_nlp)
        assert len(result) == 2
        assert result[0].doc_lemmas == []

    def test_nan_coerced_to_string(self, fake_nlp):
        # NaN float after a string stays in flat mode and is coerced to ""
        # (the isinstance(t, float) and t != t check on line 258 catches it)
        result = preprocess_texts(["word", float("nan")], fake_nlp)
        assert len(result) == 2
        assert result[1].doc_lemmas == []

    def test_bytes_decoded(self, fake_nlp):
        result = preprocess_texts([b"hello world"], fake_nlp)
        assert len(result) == 1
        assert "hello" in result[0].doc_lemmas

    def test_stopwords_filtered(self, fake_nlp):
        result = preprocess_texts(["kraj jest dom"], fake_nlp, stopwords=["jest"])
        lemmas = result[0].doc_lemmas
        assert "jest" not in lemmas
        assert "kraj" in lemmas

    def test_token_to_sent_mapping(self, fake_nlp):
        result = preprocess_texts(["a b c"], fake_nlp)
        # FakeNlp produces one sentence per text
        assert result[0].token_to_sent == [0, 0, 0]

    def test_nlp_none_raises(self):
        with pytest.raises(ValueError, match="nlp is None"):
            preprocess_texts(["hello"], None)


# ---------------------------------------------------------------------------
# preprocess_texts — profile mode
# ---------------------------------------------------------------------------

class TestPreprocessTextsProfile:
    def test_basic_profile(self, fake_nlp):
        texts = [["post one", "post two"], ["post three"]]
        result = preprocess_texts(texts, fake_nlp)
        assert len(result) == 2
        assert all(isinstance(r, PreprocessedProfile) for r in result)

    def test_profile_raw_posts(self, fake_nlp):
        texts = [["hello world", "foo bar"]]
        result = preprocess_texts(texts, fake_nlp)
        assert result[0].raw_posts == ["hello world", "foo bar"]

    def test_profile_post_count(self, fake_nlp):
        texts = [["a", "b", "c"]]
        result = preprocess_texts(texts, fake_nlp)
        assert len(result[0].post_doc_lemmas) == 3

    def test_empty_profile(self, fake_nlp):
        texts = [[], ["hello"]]
        result = preprocess_texts(texts, fake_nlp)
        assert len(result) == 2
        # Empty profile
        assert result[0].raw_posts == []
        assert result[0].post_doc_lemmas == []

    def test_profile_empty_strings_dropped(self, fake_nlp):
        texts = [["hello", "", "  ", "world"]]
        result = preprocess_texts(texts, fake_nlp)
        assert result[0].raw_posts == ["hello", "world"]


# ---------------------------------------------------------------------------
# build_docs_from_preprocessed
# ---------------------------------------------------------------------------

class TestBuildDocsFromPreprocessed:
    def test_from_docs(self, sample_preprocessed_docs):
        result = build_docs_from_preprocessed(sample_preprocessed_docs)
        assert len(result) == len(sample_preprocessed_docs)
        assert result[0] == sample_preprocessed_docs[0].doc_lemmas

    def test_empty_input(self):
        assert build_docs_from_preprocessed([]) == []

    def test_from_profiles(self):
        profile = PreprocessedProfile(
            raw_posts=["a b", "c d"],
            post_sents_surface=[["a b"], ["c d"]],
            post_sents_lemmas=[[["a", "b"]], [["c", "d"]]],
            post_doc_lemmas=[["a", "b"], ["c", "d"]],
            post_sent_char_spans=[[(0, 3)], [(0, 3)]],
            post_token_to_sent=[[0, 0], [0, 0]],
            post_sents_kept_idx=[[[0, 1]], [[0, 1]]],
        )
        result = build_docs_from_preprocessed([profile])
        assert result == [[["a", "b"], ["c", "d"]]]


# ---------------------------------------------------------------------------
# load_stopwords (bundled Polish)
# ---------------------------------------------------------------------------

class TestLoadStopwords:
    def test_polish_returns_list(self):
        sw = load_stopwords("pl")
        assert isinstance(sw, list)
        assert len(sw) > 0

    def test_polish_lowercase(self):
        sw = load_stopwords("pl", lowercase=True)
        assert all(w == w.lower() for w in sw)

    def test_default_is_polish(self):
        sw_default = load_stopwords()
        sw_pl = load_stopwords("pl")
        assert sw_default == sw_pl

    def test_english_from_spacy(self):
        sw = load_stopwords("en")
        assert isinstance(sw, list)
        assert len(sw) > 0

    def test_unknown_lang_raises(self):
        with pytest.raises(ValueError, match="Unknown language"):
            load_stopwords("zzzz_nonexistent")
