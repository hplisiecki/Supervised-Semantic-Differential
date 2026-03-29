"""Tests for ssdlite.corpus — Corpus class."""

import pytest

from ssdlite.corpus import Corpus


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
        assert "2 docs" in repr(corpus)


class TestCorpusValidation:
    def test_no_lang_no_model_raises(self):
        with pytest.raises(ValueError, match="Provide lang="):
            Corpus(["hello world"], pretokenized=False)

    def test_pretokenized_skips_nlp(self):
        # Should not raise even without lang/model
        corpus = Corpus([["a", "b"]], pretokenized=True)
        assert len(corpus) == 1
