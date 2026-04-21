"""Tests for cluster save behaviour: flat file (single-pair) vs subfolder fan-out (multi-pair).

Single-pair (group_result_2g):
  gr.clusters.pos.save(path.csv) → flat file at path, no subfolder.

Multi-pair (group_result_3g):
  gr.clusters.pos.save(path.csv) → subfolder name hardcoded to 'clusters_pos',
  one csv per pair inside.

Mirrors the same tests for .neg.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Single-pair (2-group): flat csv
# ---------------------------------------------------------------------------


def test_clusters_pos_single_pair_flat_file(group_result_2g, tmp_path):
    """Single-pair: clusters.pos.save → flat file at the given path, no subfolder."""
    target = tmp_path / "clusters_pos.csv"
    group_result_2g.clusters.pos.save(target)

    assert target.is_file(), "expected flat csv at target path"
    assert not (tmp_path / "clusters_pos").exists(), \
        "unexpected subfolder for single-pair result"


def test_clusters_neg_single_pair_flat_file(group_result_2g, tmp_path):
    """Single-pair: clusters.neg.save → flat file at the given path, no subfolder."""
    target = tmp_path / "clusters_neg.csv"
    group_result_2g.clusters.neg.save(target)

    assert target.is_file(), "expected flat csv at target path"
    assert not (tmp_path / "clusters_neg").exists(), \
        "unexpected subfolder for single-pair result"


# ---------------------------------------------------------------------------
# Multi-pair (3-group): subfolder fan-out, name hardcoded
# ---------------------------------------------------------------------------


def test_clusters_pos_multi_pair_subfolder(group_result_3g, tmp_path):
    """Multi-pair: clusters.pos.save → tmp/clusters_pos/gi_gj.csv for each pair."""
    target = tmp_path / "clusters_pos.csv"
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        group_result_3g.clusters.pos.save(target)

    # Original path NOT written
    assert not target.exists(), "original csv path should not be created"

    # Subfolder name is 'clusters_pos' (hardcoded), not derived from caller path
    folder = tmp_path / "clusters_pos"
    assert folder.is_dir(), "subfolder 'clusters_pos' should be created"

    for pair in group_result_3g.pairs:
        expected = folder / f"{pair.g1}_{pair.g2}.csv"
        assert expected.is_file(), f"expected {expected}"

    assert any(issubclass(w.category, UserWarning) for w in recorded), \
        "expected a UserWarning on csv fan-out"


def test_clusters_neg_multi_pair_subfolder(group_result_3g, tmp_path):
    """Multi-pair: clusters.neg.save → tmp/clusters_neg/gi_gj.csv for each pair."""
    target = tmp_path / "clusters_neg.csv"
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        group_result_3g.clusters.neg.save(target)

    assert not target.exists()

    folder = tmp_path / "clusters_neg"
    assert folder.is_dir(), "subfolder 'clusters_neg' should be created"

    for pair in group_result_3g.pairs:
        expected = folder / f"{pair.g1}_{pair.g2}.csv"
        assert expected.is_file(), f"expected {expected}"

    assert any(issubclass(w.category, UserWarning) for w in recorded)


def test_clusters_pos_multi_pair_subfolder_name_hardcoded(group_result_3g, tmp_path):
    """Subfolder name is always 'clusters_pos', regardless of the caller's path stem."""
    target = tmp_path / "anything.csv"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        group_result_3g.clusters.pos.save(target)

    assert not target.exists()
    assert (tmp_path / "clusters_pos").is_dir(), \
        "subfolder must be 'clusters_pos', not 'anything'"
    assert not (tmp_path / "anything").exists(), "unexpected 'anything' subfolder"


def test_clusters_neg_multi_pair_subfolder_name_hardcoded(group_result_3g, tmp_path):
    """Subfolder name is always 'clusters_neg', regardless of the caller's path stem."""
    target = tmp_path / "anything.csv"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        group_result_3g.clusters.neg.save(target)

    assert not target.exists()
    assert (tmp_path / "clusters_neg").is_dir(), \
        "subfolder must be 'clusters_neg', not 'anything'"
    assert not (tmp_path / "anything").exists()
