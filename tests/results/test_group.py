"""Tests for GroupResult, PairResult, and related views.

Invariants covered:
- Container structure (pairs count, canonical keys, _canonical_pair_key order)
- Per-pair accessor equivalence
- beta dict shape
- PairResult.beta arithmetic
- alignment_scores contract
- save fan-out behavior (csv subfolder, xlsx multi-sheet)
- report text markers
"""

from __future__ import annotations

import csv
import math
import warnings

import numpy as np
import pytest

from ssdiff.results.group_result import PairResult, _canonical_pair_key

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _n_pairs(n_groups: int) -> int:
    """C(n, 2)."""
    return math.comb(n_groups, 2)


# ---------------------------------------------------------------------------
# pairs count == C(G, 2)
# ---------------------------------------------------------------------------

def test_pairs_count_3g(group_result_3g):
    gr = group_result_3g
    assert len(gr.pairs) == _n_pairs(3)


# ---------------------------------------------------------------------------
# canonical pair order and key lookup
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a, b, expected", [
    ("B", "A", ("A", "B")),
    ("A", "B", ("A", "B")),
    ("g1", "g2", ("g1", "g2")),
    ("g10", "g2", ("g2", "g10")),  # numeric suffix, not lexicographic
])
def test_canonical_pair_key(a, b, expected):
    assert _canonical_pair_key(a, b) == expected


def test_canonical_key_lookup_3g(group_result_3g):
    gr = group_result_3g
    expected = {("g1", "g2"), ("g1", "g3"), ("g2", "g3")}
    assert set(gr.keys()) == expected
    assert gr.group_labels == {"g1": "X", "g2": "Y", "g3": "Z"}
    for key in expected:
        pr = gr[key]
        assert isinstance(pr, PairResult)
        assert pr.contrast == f"{key[0]}_{key[1]}"


# ---------------------------------------------------------------------------
# reverse-order tuple raises KeyError mentioning "canonical"
# ---------------------------------------------------------------------------

def test_reverse_order_raises_key_error(group_result_2g):
    gr = group_result_2g
    assert list(gr.keys()) == [("g1", "g2")]
    with pytest.raises(KeyError, match="canonical"):
        gr[("g2", "g1")]


# ---------------------------------------------------------------------------
# unknown group raises KeyError
# ---------------------------------------------------------------------------

def test_unknown_group_raises_key_error(group_result_2g):
    gr = group_result_2g
    with pytest.raises(KeyError):
        gr[("g1", "Unknown")]


# ---------------------------------------------------------------------------
# gr.words[pair] same object as gr[pair].words
# ---------------------------------------------------------------------------

def test_words_shim_matches_pair_words(group_result_2g):
    gr = group_result_2g
    assert gr.words.keys() == gr.keys()
    for key in gr.keys():
        rows = list(gr.words[key])
        assert rows
        assert rows == list(gr[key].words)


# ---------------------------------------------------------------------------
# gr.beta is dict[pair_key, np.ndarray of shape (D,)]
# ---------------------------------------------------------------------------

def test_beta_dict_shape_3g(group_result_3g):
    gr = group_result_3g
    D = gr.x.shape[1]
    assert set(gr.beta) == set(gr.keys())
    for key, arr in gr.beta.items():
        assert arr.shape == (D,)
        np.testing.assert_array_equal(arr, gr[key].beta)


# ---------------------------------------------------------------------------
# PairResult.beta == mean(g1) - mean(g2) from raw x/groups
# ---------------------------------------------------------------------------

def test_pair_result_beta_arithmetic(group_result_2g):
    gr = group_result_2g
    pair_key = list(gr._leaves.keys())[0]
    g1, g2 = pair_key
    pr = gr[pair_key]

    # Reconstruct from container's x and groups
    x = gr._x
    groups = gr._groups
    mean_g1 = x[groups == g1].mean(axis=0)
    mean_g2 = x[groups == g2].mean(axis=0)
    expected_beta = mean_g1 - mean_g2

    np.testing.assert_allclose(pr.beta, expected_beta, atol=1e-10)


# ---------------------------------------------------------------------------
# alignment_scores per pair has length n_kept and values in [-1, 1]
# ---------------------------------------------------------------------------

def test_alignment_scores_2g(group_result_2g):
    gr = group_result_2g
    assert set(gr.alignment_scores) == set(gr.keys())
    for key, scores in gr.alignment_scores.items():
        pr = gr[key]
        x_unit = pr.x / np.linalg.norm(pr.x, axis=1, keepdims=True)
        expected = x_unit @ (pr.beta / np.linalg.norm(pr.beta))
        np.testing.assert_allclose(scores, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# 3-group gr.words.save("out.xlsx") → single workbook, one sheet per pair
# ---------------------------------------------------------------------------

def test_words_save_xlsx_3g(group_result_3g, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    gr = group_result_3g
    out = tmp_path / "words.xlsx"
    gr.words.save(str(out))
    wb = openpyxl.load_workbook(str(out))
    assert set(wb.sheetnames) == {"g1_g2", "g1_g3", "g2_g3"}


# ---------------------------------------------------------------------------
# shim-view save(k=N) works for clusters, not just words
# ---------------------------------------------------------------------------
# The fan-out tests below also cover ``gr.clusters``, not just ``gr.words``:
# its leaves need the per-leaf ``_resized(k)`` step. ``gr.snippets`` is left
# out: the group fixtures produce no snippets, so its files hold only a header.

_SHIM_NAMES = ["words", "clusters"]


@pytest.mark.parametrize("shim_name", _SHIM_NAMES)
def test_shim_save_csv_2g_with_k(group_result_2g, shim_name, tmp_path):
    """2-group: shim.save(out.csv, k=N) delegates to the single child; flat file."""
    shim = getattr(group_result_2g, shim_name)
    out = tmp_path / f"{shim_name}.csv"
    shim.save(str(out), k=2)
    assert out.read_text(encoding="utf-8").strip()


@pytest.mark.parametrize("k", [None, 2])
@pytest.mark.parametrize("shim_name", _SHIM_NAMES)
def test_shim_save_csv_3g_fanout(group_result_3g, shim_name, k, tmp_path):
    """3-group: shim.save(out.csv, k=k) fans out to one file per pair, <= k rows each."""
    gr = group_result_3g
    shim = getattr(gr, shim_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shim.save(str(tmp_path / f"{shim_name}.csv"), k=k)
    assert sorted(f.name for f in (tmp_path / shim_name).glob("*.csv")) == [
        f"{a}_{b}.csv" for a, b in gr.keys()
    ]
    for a, b in gr.keys():
        available = len(list(shim[(a, b)]))
        with open(tmp_path / shim_name / f"{a}_{b}.csv", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert reader.fieldnames
        assert len(rows) == (available if k is None else min(k, available))


# ---------------------------------------------------------------------------
# report().to_text() contains "Omnibus" and "Pairwise"
# ---------------------------------------------------------------------------

def test_report_text_contains_omnibus_pairwise(group_result_2g):
    gr = group_result_2g
    text = gr.report().to_text()
    assert "Omnibus" in text, "report().to_text() missing 'Omnibus'"
    assert "Pairwise" in text, "report().to_text() missing 'Pairwise'"


# ---------------------------------------------------------------------------
# report().to_text() lists every pair and the group labels
# ---------------------------------------------------------------------------

def test_report_text_lists_every_pair_and_group_labels(group_result_3g):
    gr = group_result_3g
    text = gr.report().to_text()
    for a, b in gr.keys():
        assert f"{a}_{b}" in text
    assert "Group labels" in text
    for label in gr.group_labels.values():
        assert label in text
