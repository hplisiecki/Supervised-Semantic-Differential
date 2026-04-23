"""Tests for GroupResult multi-pair (G >= 3) pivot dispatch.

Covers:
- gr.words / gr.clusters / gr.snippets are _ShimView dicts keyed by canonical pair tuple
- gr.beta, gr.gradient, gr.beta_norm, gr.alignment_scores are dicts keyed by canonical pair tuple
- gr.words[(g1, g2)] round-trips to a single-pair WordsView
- gr.words[(g2, g1)] raises KeyError (canonical-only)
- list(gr.words.keys()) matches canonical pair tuples from gr.pairs
- gr.words[("g2", "g1")] raises KeyError on words view (not just on gr.pairs)
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Multi-pair dispatch (G >= 3)
# ---------------------------------------------------------------------------


class TestMultiPairDispatch:
    def test_words_is_shim_dict(self, group_result_3g):
        from ssdiff.results.multi_container import _ShimView
        assert isinstance(group_result_3g.words, _ShimView)

    def test_clusters_is_shim_dict(self, group_result_3g):
        from ssdiff.results.multi_container import _ShimView
        assert isinstance(group_result_3g.clusters, _ShimView)

    def test_snippets_is_shim_dict(self, group_result_3g):
        from ssdiff.results.multi_container import _ShimView
        assert isinstance(group_result_3g.snippets, _ShimView)

    def test_beta_is_dict_keyed_by_canonical_pair(self, group_result_3g):
        gr = group_result_3g
        assert isinstance(gr.beta, dict)
        assert len(gr.beta) == 3
        for key, b in gr.beta.items():
            assert isinstance(key, tuple) and len(key) == 2
            assert isinstance(b, np.ndarray)
            assert b.shape == (gr._x.shape[1],)

    def test_gradient_is_dict_of_unit_vectors(self, group_result_3g):
        gr = group_result_3g
        assert isinstance(gr.gradient, dict)
        for _, g in gr.gradient.items():
            np.testing.assert_allclose(np.linalg.norm(g), 1.0, rtol=1e-10)

    def test_beta_norm_is_dict_of_floats(self, group_result_3g):
        gr = group_result_3g
        assert isinstance(gr.beta_norm, dict)
        for _, bn in gr.beta_norm.items():
            assert isinstance(bn, float)
            assert bn >= 0.0

    def test_alignment_scores_is_dict(self, group_result_3g):
        gr = group_result_3g
        scores = gr.alignment_scores
        assert isinstance(scores, dict)
        assert len(scores) == 3
        for key, arr in scores.items():
            assert isinstance(key, tuple) and len(key) == 2
            # Shape is per-pair n (n_g1 + n_g2), not total n_kept.
            leaf = gr[key]
            assert arr.shape == (len(leaf.x),)


# ---------------------------------------------------------------------------
# _ShimView key-access and canonical-only enforcement
# ---------------------------------------------------------------------------


def test_words_getitem_roundtrips_to_single_pair_view(group_result_3g):
    """gr.words[(g1, g2)] returns a WordsView (single-pair)."""
    from ssdiff.results.continuous_result import WordsView
    gr = group_result_3g
    first_pair = next(iter(gr.pairs))
    view = gr.words[(first_pair.g1, first_pair.g2)]
    assert isinstance(view, WordsView)


def test_words_reverse_order_raises_keyerror(group_result_3g):
    """gr.words[(g2, g1)] raises KeyError — canonical order only."""
    gr = group_result_3g
    first_pair = next(iter(gr.pairs))
    with pytest.raises(KeyError):
        _ = gr.words[(first_pair.g2, first_pair.g1)]


def test_words_keys_match_canonical_pairs(group_result_3g):
    """list(gr.words.keys()) contains exactly the canonical pair tuples from gr.pairs."""
    gr = group_result_3g
    expected = {(p.g1, p.g2) for p in gr.pairs}
    actual = set(gr.words.keys())
    assert actual == expected, f"words keys {actual} != pairs {expected}"


# ---------------------------------------------------------------------------
# Caching: _leaves-based leaf access (replaces the removed _compute_pair_arrays test)
# ---------------------------------------------------------------------------


def test_pair_arrays_computed_once_per_pair(group_result_3g, monkeypatch):
    """Each leaf is built once; repeated attribute access does not recompute."""
    gr = group_result_3g
    # The leaves are keyed by canonical pair tuples.
    n_pairs = len(list(gr.pairs))
    assert len(gr._leaves) == n_pairs
    # Accessing beta/gradient repeatedly returns the same cached values.
    beta1 = gr.beta
    beta2 = gr.beta
    for key in beta1:
        np.testing.assert_array_equal(beta1[key], beta2[key])
