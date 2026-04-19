"""Tests for ssdiff.corpus — Corpus class."""

import pytest

from ssdiff.corpus import Corpus


class TestCorpusPretokenized:
    def test_basic(self):
        docs = [["hello", "world"], ["foo", "bar", "baz"]]
        corpus = Corpus(docs, pretokenized=True)
        assert len(corpus) == 2
        assert corpus.docs == docs
        assert corpus.pre_docs is None
        assert corpus.n_texts == 2

    def test_repr(self):
        corpus = Corpus([["a", "b"], ["c"]], pretokenized=True)
        r = repr(corpus)
        assert "Corpus" in r
        assert "n=2" in r
        assert ".docs" in r
        assert ".suggest_lexicon(...)" in r


class TestCorpusValidation:
    def test_no_lang_no_model_raises(self):
        with pytest.raises(ValueError, match="Provide lang="):
            Corpus(["hello world"], pretokenized=False)

    def test_pretokenized_skips_nlp(self):
        # Should not raise even without lang/model
        corpus = Corpus([["a", "b"]], pretokenized=True)
        assert len(corpus) == 1


class TestCorpusNonPretokenizedPath:
    """Corpus(pretokenized=False, nlp=...) — exercises the preprocess pipeline."""

    def test_builds_docs_from_raw_text(self, fake_nlp):
        raw = [
            "kraj jest piekny i silny",
            "narod jest wielki",
            "panstwo i szkola",
        ]
        c = Corpus(raw, pretokenized=False, nlp=fake_nlp, lang="pl")
        # c.docs is list[list[str]] — lemmatized token lists
        assert len(c.docs) == 3
        # Token membership: at least the seeds should be in doc 0
        # (fake_nlp lowercases each word as its lemma)
        all_tokens_doc0 = set(c.docs[0])
        assert "kraj" in all_tokens_doc0

    def test_non_pretokenized_requires_nlp(self):
        # With no nlp/lang/model provided, Corpus raises ValueError
        with pytest.raises(ValueError):
            Corpus(["kraj"], pretokenized=False, lang=None)
