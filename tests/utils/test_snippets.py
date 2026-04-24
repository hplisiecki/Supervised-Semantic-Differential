"""Tests for ssdiff/utils/snippets.py

Tests cover:
- _make_snippet_anchor: token-span clamping and multi-sentence window
- _iter_doclikes: yields _DocLike from PreprocessedDoc list
- _top_per_group: top-k per group from sorted rows
- snippets_along_beta: returns pos/neg dicts, cosine signs, sorted descending
- cluster_snippets_by_centroids: 2-cluster case returns 2-keyed structure
"""

from __future__ import annotations

import numpy as np
import pytest

from ssdiff.utils.snippets import (
    _DocLike,
    _iter_doclikes,
    _make_snippet_anchor,
    _top_per_group,
    cluster_snippets_by_centroids,
    snippets_along_beta,
)
from ssdiff.utils.text import PreprocessedDoc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doclike(
    sents_surface: list[str],
    doc_lemmas: list[str],
    token_to_sent: list[int],
    profile_id: int = 0,
    post_id: int = 0,
) -> _DocLike:
    return _DocLike(
        profile_id=profile_id,
        post_id=post_id,
        sents_surface=sents_surface,
        doc_lemmas=doc_lemmas,
        token_to_sent=token_to_sent,
    )


class _FakeSSD:
    """Minimal duck-type satisfying snippets_along_beta and cluster_snippets_by_centroids."""

    def __init__(self, embeddings, gradient, lexicon=None):
        self.embeddings = embeddings
        self.gradient = gradient
        self.beta = gradient  # fallback attr
        self.lexicon = lexicon or []


# ---------------------------------------------------------------------------
# _make_snippet_anchor: token-span clamping
# ---------------------------------------------------------------------------

class TestMakeSnippetAnchorClamping:
    """Invariant 1: token-span clamping when start/end exceed doc bounds."""

    def test_end_tok_beyond_last_token_clamped(self):
        """end_tok past last token index is clamped to len(doc_lemmas)-1."""
        D = _make_doclike(
            sents_surface=["Hello world."],
            doc_lemmas=["hello", "world"],
            token_to_sent=[0, 0],
        )
        # anchor token i=1, window [0, 99] — 99 must clamp to 1
        snippet, s_start, s_end = _make_snippet_anchor(D, i=1, start_tok=0, end_tok=99)
        assert snippet == "Hello world."
        assert s_start == 0
        assert s_end == 0

    def test_start_tok_negative_clamped_to_zero(self):
        """start_tok < 0 is clamped to 0."""
        D = _make_doclike(
            sents_surface=["Kraj jest piekny."],
            doc_lemmas=["kraj", "jest", "piekny"],
            token_to_sent=[0, 0, 0],
        )
        snippet, s_start, s_end = _make_snippet_anchor(D, i=1, start_tok=-5, end_tok=2)
        assert snippet == "Kraj jest piekny."
        assert s_start == 0
        assert s_end == 0

    def test_single_token_doc_any_window_returns_single_sentence(self):
        """Single-token doc: any window must still yield the only sentence."""
        D = _make_doclike(
            sents_surface=["Word."],
            doc_lemmas=["word"],
            token_to_sent=[0],
        )
        snippet, s_start, s_end = _make_snippet_anchor(D, i=0, start_tok=-10, end_tok=100)
        assert snippet == "Word."
        assert s_start == 0
        assert s_end == 0


# ---------------------------------------------------------------------------
# _make_snippet_anchor: multi-sentence window
# ---------------------------------------------------------------------------

class TestMakeSnippetAnchorMultiSentence:
    """Invariant 2: multi-sentence window spans across sentence boundaries."""

    def test_window_reaches_previous_sentence(self):
        """When start_tok falls in a previous sentence, prev sentence is prepended."""
        # Doc: sent0 = tokens 0-1, sent1 = tokens 2-3
        D = _make_doclike(
            sents_surface=["Narod jest wielki.", "Kraj jest silny."],
            doc_lemmas=["narod", "jest", "kraj", "silny"],
            token_to_sent=[0, 0, 1, 1],
        )
        # anchor at tok 2 (sent 1), window start at tok 0 (sent 0)
        snippet, s_start, s_end = _make_snippet_anchor(D, i=2, start_tok=0, end_tok=3)
        # Should include previous sentence
        assert "Narod jest wielki." in snippet
        assert "Kraj jest silny." in snippet
        assert s_start == 0
        assert s_end == 1

    def test_window_reaches_next_sentence(self):
        """When end_tok falls in a next sentence, next sentence is appended."""
        D = _make_doclike(
            sents_surface=["Narod jest wielki.", "Kraj jest silny."],
            doc_lemmas=["narod", "jest", "kraj", "silny"],
            token_to_sent=[0, 0, 1, 1],
        )
        # anchor at tok 1 (sent 0), window end at tok 3 (sent 1)
        snippet, s_start, s_end = _make_snippet_anchor(D, i=1, start_tok=0, end_tok=3)
        # Because start_sent == anchor sent (both 0), the code checks next sentence
        assert "Narod jest wielki." in snippet
        assert "Kraj jest silny." in snippet
        assert s_start == 0
        assert s_end == 1

    def test_anchor_in_middle_no_cross_boundary_stays_single(self):
        """If window stays within anchor sentence, returns just that sentence."""
        D = _make_doclike(
            sents_surface=["Narod jest wielki.", "Kraj jest silny."],
            doc_lemmas=["narod", "jest", "kraj", "silny"],
            token_to_sent=[0, 0, 1, 1],
        )
        # anchor at tok 0 (sent 0), window within [0,1] — same sentence
        snippet, s_start, s_end = _make_snippet_anchor(D, i=0, start_tok=0, end_tok=1)
        assert snippet == "Narod jest wielki."
        assert s_start == 0
        assert s_end == 0


# ---------------------------------------------------------------------------
# _iter_doclikes
# ---------------------------------------------------------------------------

class TestIterDoclikes:
    """Invariants 5 & 6: _iter_doclikes over PreprocessedDoc list."""

    def test_yields_doclikes_for_each_preprocessed_doc(self, sample_preprocessed_docs):
        """Invariant 5: yields one _DocLike per PreprocessedDoc, unchanged content."""
        result = list(_iter_doclikes(sample_preprocessed_docs))
        assert len(result) == len(sample_preprocessed_docs)
        for dl, pd in zip(result, sample_preprocessed_docs):
            assert dl.doc_lemmas == pd.doc_lemmas
            assert dl.sents_surface == pd.sents_surface
            assert dl.token_to_sent == pd.token_to_sent

    def test_profile_id_is_sequential_index(self, sample_preprocessed_docs):
        """profile_id is assigned as the enumeration index."""
        result = list(_iter_doclikes(sample_preprocessed_docs))
        for i, dl in enumerate(result):
            assert dl.profile_id == i

    def test_post_id_is_zero_for_single_docs(self, sample_preprocessed_docs):
        """post_id is always 0 for non-profile (plain PreprocessedDoc) input."""
        result = list(_iter_doclikes(sample_preprocessed_docs))
        assert all(dl.post_id == 0 for dl in result)

    def test_empty_list_yields_nothing(self):
        """Empty input produces empty output."""
        result = list(_iter_doclikes([]))
        assert result == []


# ---------------------------------------------------------------------------
# _top_per_group
# ---------------------------------------------------------------------------

class TestTopPerGroup:
    """Invariant 8: _top_per_group returns top-k per group key."""

    def _make_rows(self, items):
        """items: list of (group, cosine) tuples."""
        return [{"centroid_label": g, "cosine": c} for g, c in items]

    def test_top1_per_group(self):
        """k=1 returns the first (highest cosine) row per group."""
        rows = self._make_rows([
            ("A", 0.9), ("A", 0.5), ("A", 0.1),
            ("B", 0.8), ("B", 0.3),
        ])
        result = _top_per_group(rows, "centroid_label", 1)
        a_rows = [r for r in result if r["centroid_label"] == "A"]
        b_rows = [r for r in result if r["centroid_label"] == "B"]
        assert len(a_rows) == 1
        assert a_rows[0]["cosine"] == 0.9
        assert len(b_rows) == 1
        assert b_rows[0]["cosine"] == 0.8

    def test_top2_per_group_caps_at_available(self):
        """k=2 returns up to 2 per group, even if fewer exist."""
        rows = self._make_rows([
            ("X", 0.9), ("X", 0.7),
            ("Y", 0.6),
        ])
        result = _top_per_group(rows, "centroid_label", 2)
        x_rows = [r for r in result if r["centroid_label"] == "X"]
        y_rows = [r for r in result if r["centroid_label"] == "Y"]
        assert len(x_rows) == 2
        assert len(y_rows) == 1

    def test_preserves_order(self):
        """Input order within group is preserved (assumes pre-sorted by cosine desc)."""
        rows = self._make_rows([
            ("A", 0.95), ("A", 0.80), ("A", 0.50),
        ])
        result = _top_per_group(rows, "centroid_label", 2)
        cosines = [r["cosine"] for r in result]
        assert cosines == [0.95, 0.80]


# ---------------------------------------------------------------------------
# snippets_along_beta
# ---------------------------------------------------------------------------

class TestSnippetsAlongBeta:
    """Invariants 3 & 4: snippets_along_beta returns k_per_side snippets, sorted."""

    def test_returns_pos_and_neg_keys(self, sample_preprocessed_docs, tiny_kv):
        """Function returns a dict with 'pos' and 'neg' keys."""
        gradient = tiny_kv.get_vector("kraj", norm=True)
        ssd = _FakeSSD(tiny_kv, gradient, lexicon=["kraj", "narod", "panstwo"])
        result = snippets_along_beta(
            pre_docs=sample_preprocessed_docs,
            ssd=ssd,
            token_window=2,
            top_per_side=10,
        )
        assert "pos" in result
        assert "neg" in result

    def test_pos_snippets_sorted_descending(self, sample_preprocessed_docs, tiny_kv):
        """Invariant 4: pos snippets sorted by cosine descending."""
        gradient = tiny_kv.get_vector("kraj", norm=True)
        ssd = _FakeSSD(tiny_kv, gradient, lexicon=["kraj", "narod", "panstwo"])
        result = snippets_along_beta(
            pre_docs=sample_preprocessed_docs,
            ssd=ssd,
            token_window=2,
            top_per_side=20,
        )
        pos = result["pos"]
        if len(pos) > 1:
            cosines = [r["cosine"] for r in pos]
            assert cosines == sorted(cosines, reverse=True)

    def test_neg_snippets_sorted_descending(self, sample_preprocessed_docs, tiny_kv):
        """Neg side also sorted descending (neg cosines are -pos, sorted desc)."""
        gradient = tiny_kv.get_vector("kraj", norm=True)
        ssd = _FakeSSD(tiny_kv, gradient, lexicon=["kraj", "narod", "panstwo"])
        result = snippets_along_beta(
            pre_docs=sample_preprocessed_docs,
            ssd=ssd,
            token_window=2,
            top_per_side=20,
        )
        neg = result["neg"]
        if len(neg) > 1:
            cosines = [r["cosine"] for r in neg]
            assert cosines == sorted(cosines, reverse=True)

    def test_pos_neg_cosines_are_negatives_of_each_other(self, sample_preprocessed_docs, tiny_kv):
        """Invariant 3: for same occurrence, pos_cosine == -neg_cosine."""
        gradient = tiny_kv.get_vector("kraj", norm=True)
        ssd = _FakeSSD(tiny_kv, gradient, lexicon=["kraj"])
        result = snippets_along_beta(
            pre_docs=sample_preprocessed_docs,
            ssd=ssd,
            token_window=2,
            top_per_side=200,
        )
        pos = result["pos"]
        neg = result["neg"]
        # Both sides should have same number of snippets (no threshold)
        assert len(pos) == len(neg)
        # Sort both by (profile_id, seed) to align matching rows
        pos_sorted = sorted(pos, key=lambda r: (r["profile_id"], r["seed"]))
        neg_sorted = sorted(neg, key=lambda r: (r["profile_id"], r["seed"]))
        for p, n in zip(pos_sorted, neg_sorted):
            assert abs(p["cosine"] + n["cosine"]) < 1e-9

    def test_top_per_side_limits_output(self, sample_preprocessed_docs, tiny_kv):
        """top_per_side caps the number of returned snippets per side."""
        gradient = tiny_kv.get_vector("kraj", norm=True)
        ssd = _FakeSSD(tiny_kv, gradient, lexicon=["kraj", "narod", "panstwo"])
        result = snippets_along_beta(
            pre_docs=sample_preprocessed_docs,
            ssd=ssd,
            token_window=2,
            top_per_side=1,
        )
        assert len(result["pos"]) <= 1
        assert len(result["neg"]) <= 1

    def test_snippet_dict_has_required_keys(self, sample_preprocessed_docs, tiny_kv):
        """Each snippet dict contains the documented keys."""
        gradient = tiny_kv.get_vector("kraj", norm=True)
        ssd = _FakeSSD(tiny_kv, gradient, lexicon=["kraj", "narod", "panstwo"])
        result = snippets_along_beta(
            pre_docs=sample_preprocessed_docs,
            ssd=ssd,
            token_window=2,
            top_per_side=10,
        )
        required_keys = {
            "side", "profile_id", "post_id", "cosine", "seed",
            "start_token_idx", "end_token_idx", "start_sent_idx", "end_sent_idx",
            "snippet_anchor", "essay_text_surface", "essay_text_lemmas",
        }
        for snippet in result["pos"] + result["neg"]:
            assert required_keys.issubset(snippet.keys())


# ---------------------------------------------------------------------------
# cluster_snippets_by_centroids
# ---------------------------------------------------------------------------

class TestClusterSnippetsByCentroids:
    """Invariant 7: 2-cluster case returns dict with 2 keys."""

    def test_two_clusters_returns_pos_and_neg_keys(self, sample_preprocessed_docs, tiny_kv):
        """With pos_clusters and neg_clusters each having 1 cluster, returns pos/neg."""
        gradient = tiny_kv.get_vector("kraj", norm=True)
        ssd = _FakeSSD(tiny_kv, gradient, lexicon=["kraj", "narod", "panstwo"])

        pos_clusters = [{"words": [{"word": "kraj"}, {"word": "narod"}]}]
        neg_clusters = [{"words": [{"word": "panstwo"}, {"word": "dom"}]}]

        result = cluster_snippets_by_centroids(
            pre_docs=sample_preprocessed_docs,
            ssd=ssd,
            pos_clusters=pos_clusters,
            neg_clusters=neg_clusters,
            token_window=2,
            seeds=["kraj", "narod", "panstwo"],
            top_per_cluster=50,
        )
        assert "pos" in result
        assert "neg" in result
        assert isinstance(result["pos"], list)
        assert isinstance(result["neg"], list)

    def test_cluster_snippet_has_centroid_label(self, sample_preprocessed_docs, tiny_kv):
        """Each cluster snippet has a centroid_label key."""
        gradient = tiny_kv.get_vector("kraj", norm=True)
        ssd = _FakeSSD(tiny_kv, gradient, lexicon=["kraj", "narod", "panstwo"])

        pos_clusters = [{"words": [{"word": "kraj"}]}]
        neg_clusters = [{"words": [{"word": "panstwo"}]}]

        result = cluster_snippets_by_centroids(
            pre_docs=sample_preprocessed_docs,
            ssd=ssd,
            pos_clusters=pos_clusters,
            neg_clusters=neg_clusters,
            token_window=2,
            seeds=["kraj", "narod", "panstwo"],
            top_per_cluster=50,
        )
        for snippet in result["pos"]:
            assert snippet["centroid_label"].startswith("pos_")
        for snippet in result["neg"]:
            assert not snippet["centroid_label"].startswith("pos_")
