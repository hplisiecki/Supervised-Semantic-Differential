"""Tests for ssdiff/utils/text.py."""

from __future__ import annotations

from ssdiff.utils.text import build_docs_from_preprocessed

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
