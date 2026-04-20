"""Tests for GroupResult single-pair (G == 2) pivot dispatch.

Covers:
- gr.words / gr.clusters / gr.snippets return single-view types (same as ContinuousResult)
- gr.beta, gr.gradient, gr.beta_norm, gr.alignment_scores are plain arrays/float
- gr.pairs tuple lookup returns a Pair dataclass; reverse order raises KeyError
- gr.words.save(path) writes a flat file (no subfolder) for a single-pair result
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Single-pair dispatch (G == 2)
# ---------------------------------------------------------------------------


class TestSinglePairDispatch:
    def test_words_is_single_view(self, group_result_2g):
        from ssdiff.results.continuous_result import WordsView
        assert isinstance(group_result_2g.words, WordsView)

    def test_clusters_has_pos_neg(self, group_result_2g):
        from ssdiff.results.continuous_result import ClustersViewSided
        clusters = group_result_2g.clusters
        assert isinstance(clusters.pos, ClustersViewSided)
        assert isinstance(clusters.neg, ClustersViewSided)

    def test_snippets_is_single_view(self, group_result_2g):
        from ssdiff.results.continuous_result import SnippetsView
        assert isinstance(group_result_2g.snippets, SnippetsView)

    def test_beta_is_ndarray(self, group_result_2g):
        gr = group_result_2g
        assert isinstance(gr.beta, np.ndarray)
        assert gr.beta.shape == (gr.x.shape[1],)

    def test_gradient_is_unit_ndarray(self, group_result_2g):
        gr = group_result_2g
        assert isinstance(gr.gradient, np.ndarray)
        np.testing.assert_allclose(np.linalg.norm(gr.gradient), 1.0, rtol=1e-10)

    def test_beta_norm_is_float(self, group_result_2g):
        gr = group_result_2g
        assert isinstance(gr.beta_norm, float)
        assert gr.beta_norm >= 0.0

    def test_alignment_scores_is_1d_ndarray(self, group_result_2g):
        gr = group_result_2g
        scores = gr.alignment_scores
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (gr.n_kept,)


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
