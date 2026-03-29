"""Tests for ssdlite/utils/snippets.py — snippet extraction helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ssdlite.utils.text import PreprocessedDoc
from ssdlite.utils.snippets import (
    _centroid_unit_from_cluster_words,
    _iter_doclikes,
    _build_global_sif,
    _make_snippet_anchor,
    _DocLike,
    _sort_rows_by_cosine_desc,
    _sort_rows_by_label_and_cosine,
    _top_per_group,
)


# ---------------------------------------------------------------------------
# _centroid_unit_from_cluster_words
# ---------------------------------------------------------------------------

class TestCentroidUnit:
    def test_dict_format(self, tiny_kv):
        words = [{"word": "kraj"}, {"word": "narod"}]
        c = _centroid_unit_from_cluster_words(words, tiny_kv)
        assert c.shape == (tiny_kv.vector_size,)
        assert np.linalg.norm(c) == pytest.approx(1.0, abs=1e-6)

    def test_tuple_format(self, tiny_kv):
        words = [("kraj", 0.9, 1), ("narod", 0.8, 2)]
        c = _centroid_unit_from_cluster_words(words, tiny_kv)
        assert c.shape == (tiny_kv.vector_size,)
        assert np.linalg.norm(c) == pytest.approx(1.0, abs=1e-6)

    def test_all_oov_returns_zero(self, tiny_kv):
        words = [{"word": "zzz_not_in_kv"}, {"word": "yyy_not_in_kv"}]
        c = _centroid_unit_from_cluster_words(words, tiny_kv)
        assert np.allclose(c, 0.0)

    def test_empty_words(self, tiny_kv):
        c = _centroid_unit_from_cluster_words([], tiny_kv)
        assert np.allclose(c, 0.0)

    def test_single_word(self, tiny_kv):
        words = [{"word": "kraj"}]
        c = _centroid_unit_from_cluster_words(words, tiny_kv)
        assert np.linalg.norm(c) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# _iter_doclikes
# ---------------------------------------------------------------------------

class TestIterDoclikes:
    def test_empty_input(self):
        result = list(_iter_doclikes([]))
        assert result == []

    def test_yields_doclikes(self, sample_preprocessed_docs):
        result = list(_iter_doclikes(sample_preprocessed_docs))
        assert len(result) == len(sample_preprocessed_docs)
        for dl in result:
            assert isinstance(dl, _DocLike)

    def test_preserves_lemmas(self, sample_preprocessed_docs):
        result = list(_iter_doclikes(sample_preprocessed_docs))
        assert result[0].doc_lemmas == sample_preprocessed_docs[0].doc_lemmas

    def test_profile_id_sequential(self, sample_preprocessed_docs):
        result = list(_iter_doclikes(sample_preprocessed_docs))
        ids = [dl.profile_id for dl in result]
        assert ids == list(range(len(sample_preprocessed_docs)))

    def test_post_id_always_zero(self, sample_preprocessed_docs):
        result = list(_iter_doclikes(sample_preprocessed_docs))
        assert all(dl.post_id == 0 for dl in result)


# ---------------------------------------------------------------------------
# _build_global_sif
# ---------------------------------------------------------------------------

class TestBuildGlobalSif:
    def test_returns_counts_and_total(self, sample_preprocessed_docs):
        wc, total = _build_global_sif(sample_preprocessed_docs)
        assert isinstance(wc, dict)
        assert isinstance(total, int)
        assert total > 0
        # "kraj" appears in first doc
        assert "kraj" in wc
        assert wc["kraj"] >= 1


# ---------------------------------------------------------------------------
# _make_snippet_anchor
# ---------------------------------------------------------------------------

class TestMakeSnippetAnchor:
    def _make_dl(self):
        return _DocLike(
            profile_id=0,
            post_id=0,
            sents_surface=["First sentence.", "Second sentence.", "Third sentence."],
            doc_lemmas=["a", "b", "c", "d", "e", "f"],
            token_to_sent=[0, 0, 1, 1, 2, 2],
        )

    def test_within_same_sentence(self):
        dl = self._make_dl()
        snippet, s_min, s_max = _make_snippet_anchor(dl, i=0, start_tok=0, end_tok=1)
        assert snippet == "First sentence."
        assert s_min == 0
        assert s_max == 0

    def test_spans_previous_sentence(self):
        dl = self._make_dl()
        # Token 2 is in sentence 1, but context window reaches sentence 0
        snippet, s_min, s_max = _make_snippet_anchor(dl, i=2, start_tok=0, end_tok=3)
        assert "First sentence." in snippet
        assert "Second sentence." in snippet
        assert s_min == 0
        assert s_max == 1

    def test_spans_next_sentence(self):
        dl = self._make_dl()
        # Token 3 is in sentence 1, but context window reaches sentence 2
        snippet, s_min, s_max = _make_snippet_anchor(dl, i=3, start_tok=2, end_tok=5)
        assert "Second sentence." in snippet
        assert "Third sentence." in snippet

    def test_boundary_clamped(self):
        dl = self._make_dl()
        # start_tok beyond range
        snippet, _, _ = _make_snippet_anchor(dl, i=0, start_tok=-5, end_tok=1)
        assert isinstance(snippet, str)
        assert len(snippet) > 0

    def test_last_sentence_no_next(self):
        dl = self._make_dl()
        snippet, s_min, s_max = _make_snippet_anchor(dl, i=5, start_tok=4, end_tok=5)
        assert snippet == "Third sentence."


# ---------------------------------------------------------------------------
# Sorting / grouping helpers
# ---------------------------------------------------------------------------

class TestSortHelpers:
    def test_sort_by_cosine_desc(self):
        rows = [
            {"cosine": 0.3, "id": 1},
            {"cosine": 0.9, "id": 2},
            {"cosine": 0.5, "id": 3},
        ]
        result = _sort_rows_by_cosine_desc(rows)
        assert [r["id"] for r in result] == [2, 3, 1]

    def test_sort_by_label_and_cosine(self):
        rows = [
            {"centroid_label": "b", "cosine": 0.5},
            {"centroid_label": "a", "cosine": 0.3},
            {"centroid_label": "a", "cosine": 0.8},
            {"centroid_label": "b", "cosine": 0.9},
        ]
        result = _sort_rows_by_label_and_cosine(rows)
        labels = [r["centroid_label"] for r in result]
        # 'a' group first, then 'b'
        assert labels == ["a", "a", "b", "b"]
        # Within 'a' group, higher cosine first
        assert result[0]["cosine"] == 0.8
        assert result[1]["cosine"] == 0.3


class TestTopPerGroup:
    def test_basic(self):
        rows = [
            {"g": "A", "cosine": 0.9},
            {"g": "A", "cosine": 0.8},
            {"g": "A", "cosine": 0.7},
            {"g": "B", "cosine": 0.6},
            {"g": "B", "cosine": 0.5},
        ]
        result = _top_per_group(rows, "g", n=2)
        a_rows = [r for r in result if r["g"] == "A"]
        b_rows = [r for r in result if r["g"] == "B"]
        assert len(a_rows) == 2
        assert len(b_rows) == 2

    def test_n_larger_than_group(self):
        rows = [{"g": "A", "cosine": 0.9}]
        result = _top_per_group(rows, "g", n=10)
        assert len(result) == 1

    def test_empty_input(self):
        assert _top_per_group([], "g", n=5) == []
