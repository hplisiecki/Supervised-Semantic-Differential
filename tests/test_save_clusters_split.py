"""Tests for cluster save behaviour: flat file (single-pair) via leaf path.

Post-refactor: gr.clusters is a _ShimView with no .pos/.neg attributes.
Use gr[pair].clusters.pos / .neg to access sided cluster views.

Single-pair (group_result_2g):
  gr[pair].clusters.pos.save(path.csv) → flat file at path, no subfolder.

Multi-pair (group_result_3g):
  gr[pair].clusters.pos.save(path.csv) → flat file for that specific pair.
  (Multi-pair sided fan-out from a single call is a removed feature.)
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Single-pair (2-group): flat csv via leaf path
# ---------------------------------------------------------------------------


def test_clusters_pos_single_pair_flat_file(group_result_2g, tmp_path):
    """Single-pair: gr[pair].clusters.pos.save → flat file at the given path."""
    gr = group_result_2g
    pair = next(iter(gr.pairs))
    leaf = gr[(pair.g1, pair.g2)]
    target = tmp_path / "clusters_pos.csv"
    leaf.clusters.pos.save(target)

    assert target.is_file(), "expected flat csv at target path"
    assert not (tmp_path / "clusters_pos").exists(), \
        "unexpected subfolder for single-pair result"


def test_clusters_neg_single_pair_flat_file(group_result_2g, tmp_path):
    """Single-pair: gr[pair].clusters.neg.save → flat file at the given path."""
    gr = group_result_2g
    pair = next(iter(gr.pairs))
    leaf = gr[(pair.g1, pair.g2)]
    target = tmp_path / "clusters_neg.csv"
    leaf.clusters.neg.save(target)

    assert target.is_file(), "expected flat csv at target path"
    assert not (tmp_path / "clusters_neg").exists(), \
        "unexpected subfolder for single-pair result"


# ---------------------------------------------------------------------------
# Multi-pair (3-group): per-pair leaf save
# ---------------------------------------------------------------------------


def test_clusters_pos_multi_pair_each_leaf_saves_flat(group_result_3g, tmp_path):
    """Multi-pair: iterating leaves and saving per pair produces one flat file each."""
    gr = group_result_3g
    for (g1, g2), leaf in gr._leaves.items():
        out = tmp_path / f"clusters_pos_{g1}_{g2}.csv"
        leaf.clusters.pos.save(out)
        assert out.is_file(), f"expected flat file at {out}"


def test_clusters_neg_multi_pair_each_leaf_saves_flat(group_result_3g, tmp_path):
    """Multi-pair: iterating leaves and saving neg per pair produces one flat file each."""
    gr = group_result_3g
    for (g1, g2), leaf in gr._leaves.items():
        out = tmp_path / f"clusters_neg_{g1}_{g2}.csv"
        leaf.clusters.neg.save(out)
        assert out.is_file(), f"expected flat file at {out}"
