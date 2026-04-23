"""Tests for GroupResult single-pair (G == 2) dispatch.

Covers:
- gr.words / gr.clusters / gr.snippets are _ShimView; leaf access via gr[pair].words etc.
- gr.beta, gr.gradient, gr.beta_norm, gr.alignment_scores are always dicts (even for G==2)
- gr.pairs tuple lookup returns a Pair dataclass; reverse order raises KeyError
- gr.words.save(path) writes a flat file (no subfolder) for a single-pair result
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Single-pair dispatch (G == 2) — dict-valued arrays
# ---------------------------------------------------------------------------


class TestSinglePairDispatch:
    def test_words_leaf_is_single_view(self, group_result_2g):
        """gr[pair].words returns WordsView; gr.words[pair] is the power-user shortcut."""
        from ssdiff.results.continuous_result import WordsView
        gr = group_result_2g
        pair = next(iter(gr.pairs))
        assert isinstance(gr[(pair.g1, pair.g2)].words, WordsView)
        assert isinstance(gr.words[(pair.g1, pair.g2)], WordsView)

    def test_clusters_leaf_has_pos_neg(self, group_result_2g):
        from ssdiff.results.continuous_result import ClustersViewSided
        gr = group_result_2g
        pair = next(iter(gr.pairs))
        clusters = gr[(pair.g1, pair.g2)].clusters
        assert isinstance(clusters.pos, ClustersViewSided)
        assert isinstance(clusters.neg, ClustersViewSided)

    def test_snippets_leaf_is_single_view(self, group_result_2g):
        from ssdiff.results.continuous_result import SnippetsView
        gr = group_result_2g
        pair = next(iter(gr.pairs))
        assert isinstance(gr[(pair.g1, pair.g2)].snippets, SnippetsView)

    def test_beta_is_dict(self, group_result_2g):
        gr = group_result_2g
        assert isinstance(gr.beta, dict)
        assert len(gr.beta) == 1
        for key, b in gr.beta.items():
            assert isinstance(key, tuple) and len(key) == 2
            assert isinstance(b, np.ndarray)
            assert b.shape == (gr._x.shape[1],)

    def test_gradient_is_dict_of_unit_vectors(self, group_result_2g):
        gr = group_result_2g
        assert isinstance(gr.gradient, dict)
        for _, g in gr.gradient.items():
            np.testing.assert_allclose(np.linalg.norm(g), 1.0, rtol=1e-10)

    def test_beta_norm_is_dict_of_floats(self, group_result_2g):
        gr = group_result_2g
        assert isinstance(gr.beta_norm, dict)
        for _, bn in gr.beta_norm.items():
            assert isinstance(bn, float)
            assert bn >= 0.0

    def test_alignment_scores_is_dict(self, group_result_2g):
        gr = group_result_2g
        scores = gr.alignment_scores
        assert isinstance(scores, dict)
        assert len(scores) == 1
        for key, arr in scores.items():
            assert isinstance(key, tuple) and len(key) == 2
            assert arr.shape == (gr.n_kept,)


# ---------------------------------------------------------------------------
# PairsListView direct Pair return
# ---------------------------------------------------------------------------


class TestPairsListViewReturnsPair:
    def test_tuple_lookup_returns_pair_dataclass(self, group_result_2g):
        from ssdiff.results.schema import Pair
        gr = group_result_2g
        pair = next(iter(gr.pairs))
        looked_up = gr.pairs[(pair.g1, pair.g2)]
        assert isinstance(looked_up, Pair)
        assert looked_up is pair

    def test_reverse_order_raises_keyerror(self, group_result_2g):
        gr = group_result_2g
        pair = next(iter(gr.pairs))
        with pytest.raises(KeyError):
            _ = gr.pairs[(pair.g2, pair.g1)]


# ---------------------------------------------------------------------------
# Regression: group_labels values are plain str, not numpy.str_
# ---------------------------------------------------------------------------


def test_group_labels_values_are_plain_str(group_result_2g):
    gr = group_result_2g
    for k, v in gr.group_labels.items():
        assert type(v) is str, f"expected str, got {type(v).__name__}"


# ---------------------------------------------------------------------------
# Single-pair save: flat file, no subfolder
# ---------------------------------------------------------------------------


def test_words_save_flat_file_no_folder(group_result_2g, tmp_path):
    """gr.words.save(path) on a 2-group result writes a flat file, not a folder."""
    target = tmp_path / "w.csv"
    group_result_2g.words.save(target)
    assert target.is_file(), "expected flat file at target path"
    # No subfolder should be created
    assert not (tmp_path / "words").exists(), "unexpected 'words' subfolder for single-pair result"
