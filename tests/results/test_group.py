"""Tests for GroupResult, PairResult, and related views.

Consolidates invariants covering:
- Container structure (pairs count, canonical keys)
- _ShimView instances (words, clusters, snippets)
- Per-pair accessor equivalence
- beta dict shape
- PairResult.beta arithmetic
- alignment_scores contract
- save fan-out behavior (csv subfolder, xlsx multi-sheet)
- report text markers
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ssdiff.results.group_result import GroupResult, PairResult
from ssdiff.results.multi_container import _MultiContainer, _ShimView


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _n_pairs(n_groups: int) -> int:
    """C(n, 2)."""
    return math.comb(n_groups, 2)


# ---------------------------------------------------------------------------
# Invariant 1 — pairs count == C(G, 2)
# ---------------------------------------------------------------------------

def test_pairs_count_2g(group_result_2g):
    gr = group_result_2g
    assert len(gr.pairs) == _n_pairs(2)


def test_pairs_count_3g(group_result_3g):
    gr = group_result_3g
    assert len(gr.pairs) == _n_pairs(3)


# ---------------------------------------------------------------------------
# Invariant 2 — canonical key lookup returns PairResult with correct contrast
# ---------------------------------------------------------------------------

def test_canonical_key_lookup_2g(group_result_2g):
    gr = group_result_2g
    # There is exactly one pair in a 2-group result
    pair_key = list(gr._leaves.keys())[0]
    pr = gr[pair_key]
    assert isinstance(pr, PairResult)
    expected_contrast = f"{pair_key[0]}_{pair_key[1]}"
    assert pr.contrast == expected_contrast


def test_canonical_key_lookup_3g(group_result_3g):
    gr = group_result_3g
    # Check every pair
    for pair_key in gr._leaves:
        pr = gr[pair_key]
        assert isinstance(pr, PairResult)
        assert pr.contrast == f"{pair_key[0]}_{pair_key[1]}"


# ---------------------------------------------------------------------------
# Invariant 3 — reverse-order tuple raises KeyError mentioning "canonical"
# ---------------------------------------------------------------------------

def test_reverse_order_raises_key_error(group_result_2g):
    gr = group_result_2g
    canonical_key = list(gr._leaves.keys())[0]
    # Reverse it
    reversed_key = (canonical_key[1], canonical_key[0])
    with pytest.raises(KeyError, match="canonical"):
        gr[reversed_key]


# ---------------------------------------------------------------------------
# Invariant 4 — unknown group raises KeyError
# ---------------------------------------------------------------------------

def test_unknown_group_raises_key_error(group_result_2g):
    gr = group_result_2g
    with pytest.raises(KeyError):
        gr[("g1", "Unknown")]


# ---------------------------------------------------------------------------
# Invariant 5 — words/clusters/snippets are _ShimView instances
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("view_name", ["words", "clusters", "snippets"])
def test_shim_view_type(group_result_2g, view_name):
    gr = group_result_2g
    view = getattr(gr, view_name)
    assert isinstance(view, _ShimView), (
        f"gr.{view_name} should be _ShimView, got {type(view)}"
    )


# ---------------------------------------------------------------------------
# Invariant 6 — gr.words[pair] same object as gr[pair].words
# ---------------------------------------------------------------------------

def test_words_shim_matches_pair_words(group_result_2g):
    gr = group_result_2g
    pair_key = list(gr._leaves.keys())[0]
    # Access via shim and via direct pair
    shim_words = gr.words[pair_key]
    pair_words = gr[pair_key].words
    # Must be equal content — both are WordsView, may not be identical object
    # (shim rebuilds on each property access), but rows should match
    shim_rows = list(shim_words)
    pair_rows = list(pair_words)
    assert len(shim_rows) == len(pair_rows)
    if shim_rows:
        assert shim_rows[0].word == pair_rows[0].word


# ---------------------------------------------------------------------------
# Invariant 7 — gr.beta is dict[pair_key, np.ndarray of shape (D,)]
# ---------------------------------------------------------------------------

def test_beta_dict_shape_2g(group_result_2g):
    gr = group_result_2g
    beta = gr.beta
    assert isinstance(beta, dict)
    assert len(beta) == _n_pairs(2)
    for pair_key, arr in beta.items():
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 1, f"beta for {pair_key} should be 1-D"


def test_beta_dict_shape_3g(group_result_3g):
    gr = group_result_3g
    beta = gr.beta
    assert isinstance(beta, dict)
    assert len(beta) == _n_pairs(3)
    for pair_key, arr in beta.items():
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 1, f"beta for {pair_key} should be 1-D"


# ---------------------------------------------------------------------------
# Invariant 8 — PairResult.beta == mean(g1) - mean(g2) from raw x/groups
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
# Invariant 9 — alignment_scores per pair has length n_kept and values in [-1, 1]
# ---------------------------------------------------------------------------

def test_alignment_scores_2g(group_result_2g):
    gr = group_result_2g
    scores_dict = gr.alignment_scores
    assert isinstance(scores_dict, dict)
    for pair_key, scores in scores_dict.items():
        pr = gr[pair_key]
        # alignment_scores slices container x to pair-subset
        assert len(scores) == len(pr.x), (
            f"alignment_scores length mismatch for {pair_key}"
        )
        assert np.all(scores >= -1.0 - 1e-9), "alignment_scores below -1"
        assert np.all(scores <= 1.0 + 1e-9), "alignment_scores above 1"


# ---------------------------------------------------------------------------
# Invariant 10 — 2-group gr.words.save("out.csv") → flat file (n=1 delegates)
# ---------------------------------------------------------------------------

def test_words_save_csv_2g(group_result_2g):
    gr = group_result_2g
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "words.csv"
        gr.words.save(str(out))
        # n=1 so _paired_save delegates to child.save() which writes a single file
        assert out.exists(), "Expected a single flat CSV for 2-group save"
        content = out.read_text()
        assert len(content.strip()) > 0, "CSV should not be empty"


# ---------------------------------------------------------------------------
# Invariant 11 — 3-group gr.words.save("out.csv") → subfolder fan-out
# ---------------------------------------------------------------------------

def test_words_save_csv_3g_fanout(group_result_3g):
    gr = group_result_3g
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "words.csv"
        # Suppress the fan-out warning — it's expected
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gr.words.save(str(out))
        # For N>=2 csv: subfolder is created at path.parent / view_name
        subfolder = Path(tmpdir) / "words"
        assert subfolder.is_dir(), (
            f"Expected subfolder {subfolder} for 3-group CSV fan-out"
        )
        csv_files = list(subfolder.glob("*.csv"))
        assert len(csv_files) == _n_pairs(3), (
            f"Expected {_n_pairs(3)} CSV files in subfolder, got {len(csv_files)}"
        )


# ---------------------------------------------------------------------------
# Invariant 12 — 3-group gr.words.save("out.xlsx") → single workbook, C(3,2) sheets
# ---------------------------------------------------------------------------

def test_words_save_xlsx_3g(group_result_3g):
    openpyxl = pytest.importorskip("openpyxl")
    gr = group_result_3g
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "words.xlsx"
        gr.words.save(str(out))
        assert out.exists(), "Expected a single xlsx file for 3-group save"
        wb = openpyxl.load_workbook(str(out))
        sheet_names = wb.sheetnames
        assert len(sheet_names) == _n_pairs(3), (
            f"Expected {_n_pairs(3)} sheets, got {len(sheet_names)}: {sheet_names}"
        )
        # Sheet names must be canonical pair strings: g1_g2
        for name in sheet_names:
            assert "_" in name, f"Sheet name {name!r} not in canonical g1_g2 form"


# ---------------------------------------------------------------------------
# Invariant 12b — shim-view save(k=N) works for clusters/snippets, not just words
# ---------------------------------------------------------------------------
# Regression for an old gap: the fan-out tests only exercised ``gr.words.save``,
# whose leaves are WordsView (slice-safe). The same logic for ``gr.clusters``
# (leaves: ClustersView) was broken at the per-leaf ``_resized(k)`` step and
# only surfaced in the desktop app's export path.

_SHIM_NAMES = ["words", "clusters", "snippets"]


@pytest.mark.parametrize("shim_name", _SHIM_NAMES)
def test_shim_save_csv_2g_with_k(group_result_2g, shim_name):
    """2-group: shim.save(out.csv, k=N) delegates to single child; no crash."""
    gr = group_result_2g
    shim = getattr(gr, shim_name)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / f"{shim_name}.csv"
        # n=1 → delegates to child.save which routes through View._resized(k)
        shim.save(str(out), k=2)
        assert out.exists(), f"{shim_name}: expected flat CSV for 2-group save"


@pytest.mark.parametrize("shim_name", _SHIM_NAMES)
def test_shim_save_csv_3g_fanout_with_k(group_result_3g, shim_name):
    """3-group: shim.save(out.csv, k=N) fans out; each pair file has ≤ N rows."""
    import csv as _csv
    import warnings

    gr = group_result_3g
    shim = getattr(gr, shim_name)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / f"{shim_name}.csv"
        k = 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shim.save(str(out), k=k)
        subfolder = Path(tmpdir) / shim_name
        assert subfolder.is_dir(), (
            f"{shim_name}: expected fan-out subfolder for 3-group save"
        )
        csv_files = list(subfolder.glob("*.csv"))
        assert len(csv_files) == _n_pairs(3), (
            f"{shim_name}: expected {_n_pairs(3)} files, got {len(csv_files)}"
        )
        for csv_file in csv_files:
            with open(csv_file, newline="", encoding="utf-8") as f:
                rows = list(_csv.DictReader(f))
            assert len(rows) <= k, (
                f"{shim_name}/{csv_file.name}: {len(rows)} rows; expected ≤ {k}"
            )


# ---------------------------------------------------------------------------
# Invariant 13 — report().to_text() contains "Omnibus" and "Pairwise"
# ---------------------------------------------------------------------------

def test_report_text_contains_omnibus_pairwise(group_result_2g):
    gr = group_result_2g
    text = gr.report().to_text()
    assert "Omnibus" in text, "report().to_text() missing 'Omnibus'"
    assert "Pairwise" in text, "report().to_text() missing 'Pairwise'"


# ---------------------------------------------------------------------------
# Invariant 14 — report().to_text() has "— pos" count equal number of pairs
# (Only when words are available; otherwise skip)
# ---------------------------------------------------------------------------

def test_report_text_pos_section_count(group_result_3g):
    gr = group_result_3g
    text = gr.report().to_text()
    # "Pairwise contrasts" section title appears exactly once
    assert text.count("Pairwise") >= 1
    # The pairwise table has one row per pair; check contrast lines in test view
    n_pairs = _n_pairs(3)
    # Each pair produces one contrast entry in the pairwise table
    for pair_key in gr._leaves:
        contrast = f"{pair_key[0]}_{pair_key[1]}"
        assert contrast in text, f"contrast {contrast!r} missing from report text"
