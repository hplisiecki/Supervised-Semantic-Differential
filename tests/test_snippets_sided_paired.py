"""Tests for SnippetsViewSidedPaired on multi-pair GroupResult."""

import pytest

from ssdiff.results.paired_view import SnippetsViewSidedPaired


def test_snippets_pos_is_sided_paired(group_result_3g):
    pv = group_result_3g.snippets.pos
    assert isinstance(pv, SnippetsViewSidedPaired)


def test_snippets_neg_is_sided_paired(group_result_3g):
    pv = group_result_3g.snippets.neg
    assert isinstance(pv, SnippetsViewSidedPaired)


def test_snippets_pos_per_pair_index(group_result_3g):
    pv = group_result_3g.snippets.pos
    child = pv[("g1", "g2")]
    from ssdiff.results.continuous_result import SnippetsViewSided
    assert isinstance(child, SnippetsViewSided)
    assert all(s.side == "pos" for s in child)


def test_snippets_pos_save_fanout(tmp_path, group_result_3g):
    pv = group_result_3g.snippets.pos
    target = tmp_path / "snippets_pos.csv"
    pv.save(str(target))
    out_dir = tmp_path / "snippets_pos"
    assert out_dir.exists()
    produced = sorted(p.name for p in out_dir.glob("*.csv"))
    assert produced == ["g1_g2.csv", "g1_g3.csv", "g2_g3.csv"]
