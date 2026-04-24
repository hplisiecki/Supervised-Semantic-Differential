"""Tests for ssdiff.corpus.Corpus construction and basic API."""

from __future__ import annotations

import pytest

from ssdiff.corpus import Corpus


# ---------------------------------------------------------------------------
# 1. Pretokenized construction — basic len and docs
# ---------------------------------------------------------------------------

def test_pretokenized_len(sample_docs):
    corpus = Corpus(sample_docs, pretokenized=True)
    assert len(corpus) == len(sample_docs)


def test_pretokenized_docs_content(sample_docs):
    corpus = Corpus(sample_docs, pretokenized=True)
    # .docs must be a list equal to the input
    assert corpus.docs == sample_docs


def test_pretokenized_pre_docs_is_none(sample_docs):
    # pretokenized=True skips spaCy — pre_docs is None, not populated
    corpus = Corpus(sample_docs, pretokenized=True)
    assert corpus.pre_docs is None


def test_pretokenized_n_texts_matches_len(sample_docs):
    corpus = Corpus(sample_docs, pretokenized=True)
    assert corpus.n_texts == len(corpus)


# ---------------------------------------------------------------------------
# 2. Raw-text construction via fake_nlp
# ---------------------------------------------------------------------------

def test_raw_text_construction_with_nlp(fake_nlp):
    texts = ["Kraj jest piekny", "Narod jest wielki"]
    corpus = Corpus(texts, nlp=fake_nlp)
    assert len(corpus) == 2
    # FakeNlp lowercases each space-split word as the lemma
    assert corpus.docs is not None
    # Each doc must be a non-empty list of strings
    for doc in corpus.docs:
        assert isinstance(doc, list)
        assert all(isinstance(t, str) for t in doc)


def test_raw_text_pre_docs_populated(fake_nlp):
    from ssdiff.utils.text import PreprocessedDoc

    texts = ["Kraj jest piekny", "Narod jest wielki"]
    corpus = Corpus(texts, nlp=fake_nlp)
    assert corpus.pre_docs is not None
    assert len(corpus.pre_docs) == 2
    for pd in corpus.pre_docs:
        assert isinstance(pd, PreprocessedDoc)


def test_raw_text_lemmas_lowercased(fake_nlp):
    # FakeNlp uses lemma_=w.lower() for each token — verify round-trip
    texts = ["Kraj Piekny"]
    corpus = Corpus(texts, nlp=fake_nlp)
    # The exact lemma content depends on stopword filtering in preprocess_texts.
    # We only assert that all returned tokens are lowercase strings.
    for tok in corpus.docs[0]:
        assert tok == tok.lower()


# ---------------------------------------------------------------------------
# 3. Empty docs list
# ---------------------------------------------------------------------------

def test_empty_pretokenized_zero_len():
    corpus = Corpus([], pretokenized=True)
    assert len(corpus) == 0
    assert corpus.docs == []


def test_empty_raw_text_raises_without_nlp():
    # No nlp / lang / model provided → ValueError
    with pytest.raises(ValueError, match="lang=|model=|nlp="):
        Corpus(["hello world"])


# ---------------------------------------------------------------------------
# 4. Invalid input — missing nlp/lang/model
# ---------------------------------------------------------------------------

def test_no_nlp_lang_model_raises_valueerror():
    with pytest.raises(ValueError):
        Corpus(["some text", "another text"])


# ---------------------------------------------------------------------------
# 5. repr contains doc count
# ---------------------------------------------------------------------------

def test_repr_contains_doc_count(sample_docs):
    corpus = Corpus(sample_docs, pretokenized=True)
    r = repr(corpus)
    assert str(len(sample_docs)) in r


def test_repr_contains_n_equals(sample_docs):
    corpus = Corpus(sample_docs, pretokenized=True)
    r = repr(corpus)
    # repr format: "Corpus  n=8  lang=..."
    assert "n=" in r
