"""Tests for ClusterWordsViewSided / ClusterWordsView unified accessors.

Covers:
- clusters.pos.words / .neg.words → ClusterWordsViewSided (property)
- clusters.pos.words(cluster_id) → filtered ClusterWordsView (back-compat)
- clusters.words → combined ClusterWordsView across both sides
- Default save path: cluster_words_pos.csv / cluster_words_neg.csv / cluster_words.csv
- DEFAULT_COLS["ClusterWordsViewSided"] includes ``side``, excludes ``contrast``
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ssdiff.results.continuous_result import (
    ClusterWordsView,
    ClusterWordsViewSided,
    ClustersViewSided,
)
from ssdiff.results.schema import Cluster, ClusterWord


# ---------------------------------------------------------------------------
# Minimal row helpers
# ---------------------------------------------------------------------------

def _cw(cluster_id: int, side: str, word: str = "kraj") -> ClusterWord:
    return ClusterWord(
        cluster_id=cluster_id, side=side, word=word,
        cos_centroid=0.8, cos_beta=0.5 if side == "pos" else -0.5,
        contrast=None,
    )


def _cluster(cluster_id: int, side: str) -> Cluster:
    return Cluster(
        cluster_id=cluster_id, side=side, size=3, coherence=0.7,
        centroid_cos_beta=0.6 if side == "pos" else -0.6, contrast=None,
    )


def _sided(side: str, cluster_ids=(0, 1)) -> ClustersViewSided:
    rows = [_cluster(cid, side) for cid in cluster_ids]
    words_rows = [_cw(cid, side, word=f"w{cid}") for cid in cluster_ids]
    return ClustersViewSided(
        parent=None, side=side, rows=rows, words_rows=words_rows,
        params={},
    )


# ---------------------------------------------------------------------------
# Property form
# ---------------------------------------------------------------------------

def test_words_is_property_returning_sided_view():
    sv = _sided("pos")
    assert isinstance(sv.words, ClusterWordsViewSided)
    # All rows visible without filtering.
    assert len(list(sv.words)) == 2
    for w in sv.words:
        assert w.side == "pos"


def test_words_callable_filters_back_to_cluster_words_view():
    sv = _sided("pos", cluster_ids=(0, 1, 2))
    filtered = sv.words(1)
    assert isinstance(filtered, ClusterWordsView)
    # Must NOT be a ClusterWordsViewSided — it's the back-compat single-cluster form.
    assert not isinstance(filtered, ClusterWordsViewSided)
    rows = list(filtered)
    assert len(rows) == 1
    assert rows[0].cluster_id == 1


# ---------------------------------------------------------------------------
# Default save paths
# ---------------------------------------------------------------------------

def test_sided_pos_default_save_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sv = _sided("pos")
    sv.words.save()
    out = tmp_path / "cluster_words_pos.csv"
    assert out.exists()


def test_sided_neg_default_save_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sv = _sided("neg")
    sv.words.save()
    out = tmp_path / "cluster_words_neg.csv"
    assert out.exists()


def test_sided_default_cols_has_side_no_contrast(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sv = _sided("pos")
    sv.words.save()
    out = tmp_path / "cluster_words_pos.csv"
    with open(out) as f:
        reader = csv.DictReader(f)
        assert "side" in reader.fieldnames
        assert "contrast" not in reader.fieldnames
        assert "cluster_id" in reader.fieldnames
        assert "word" in reader.fieldnames


# ---------------------------------------------------------------------------
# Combined (no side) — via fixture
# ---------------------------------------------------------------------------

def test_clusters_words_combined_default_path(pls_result, tmp_path, monkeypatch):
    """pls_result.clusters.words.save() writes ./cluster_words.csv (both sides)."""
    monkeypatch.chdir(tmp_path)
    pls_result.clusters.words.save()
    out = tmp_path / "cluster_words.csv"
    assert out.exists()
    # Must be a plain ClusterWordsView (has no _side) — combined rows.
    assert isinstance(pls_result.clusters.words, ClusterWordsView)
    assert not isinstance(pls_result.clusters.words, ClusterWordsViewSided)


def test_clusters_words_combined_includes_both_sides(pls_result):
    rows = list(pls_result.clusters.words)
    sides = {r.side for r in rows}
    assert sides == {"pos", "neg"}


# ---------------------------------------------------------------------------
# Paired fan-out: gr.clusters.pos.words / gr.clusters.words
# ---------------------------------------------------------------------------

def test_paired_cluster_words_single_pair(group_result_2g, tmp_path, monkeypatch):
    """2-group: gr.clusters.pos.words.save() writes a flat cluster_words_pos.csv."""
    monkeypatch.chdir(tmp_path)
    with pytest.warns(UserWarning, match="fans out") if len(group_result_2g.pairs) > 1 else _nullctx():
        group_result_2g.clusters.pos.words.save()
    # With a single pair the paired view delegates to the child, flat filename.
    out = tmp_path / "cluster_words_pos.csv"
    assert out.exists()


def test_paired_cluster_words_multi_pair_csv_fans_out(group_result_3g, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.warns(UserWarning, match="fans out"):
        group_result_3g.clusters.pos.words.save()
    subfolder = tmp_path / "cluster_words_pos"
    assert subfolder.is_dir()
    files = sorted(p.name for p in subfolder.glob("*.csv"))
    assert len(files) == 3
    for name in files:
        assert name.startswith("g") and "_g" in name


def test_paired_cluster_words_combined_multi_pair_csv_fans_out(
    group_result_3g, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    with pytest.warns(UserWarning, match="fans out"):
        group_result_3g.clusters.words.save()
    subfolder = tmp_path / "cluster_words"
    assert subfolder.is_dir()
    assert len(list(subfolder.glob("*.csv"))) == 3


def test_paired_cluster_words_xlsx_multi_pair(group_result_3g, tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "cw.xlsx"
    group_result_3g.clusters.pos.words.save(str(out))
    assert out.exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from contextlib import contextmanager


@contextmanager
def _nullctx():
    yield
