"""Tests for ssdiff.results.format and display-layer repr/save-hint helpers.

Covers:
- format.py primitives: fmt_p, fmt_r, fmt_d, fmt_sig, fmt_count, fmt_pct, truncate,
  fmt_table, default_alignment
- display.py: set_repr_hints
- Result class reprs: PLSResult, PCAOLSResult, GroupResult, LexiconResult
- View repr save hints: WordsView, DocsView, SnippetsView, ClustersView
"""

from __future__ import annotations

import pytest

from ssdiff.results.display import TABULAR_EXTS, set_repr_hints
from ssdiff.results.format import (
    ELLIPSIS,
    MINUS,
    default_alignment,
    fmt_count,
    fmt_d,
    fmt_p,
    fmt_pct,
    fmt_r,
    fmt_sig,
    fmt_table,
    truncate,
)

# ---------------------------------------------------------------------------
# fmt_p
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p,expected", [
    (0.0001, "<.001"),       # very small → threshold symbol
    (0.0009, "<.001"),       # just under threshold
    (0.001, ".001"),         # exactly at threshold → decimal (no leading zero)
    (0.05, ".050"),          # typical value
    (0.5, ".500"),           # mid-range
    (1.0, "1.000"),          # upper bound — keeps leading 1
])
def test_fmt_p(p, expected):
    assert fmt_p(p) == expected


@pytest.mark.parametrize("p", [float("nan"), float("inf")])
def test_fmt_p_nan(p):
    assert fmt_p(p) == "nan"


# ---------------------------------------------------------------------------
# fmt_r
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r,digits,signed,expected", [
    (0.856,  2, False, ".86"),          # positive, no sign, strips leading zero
    (-0.856, 2, False, f"{MINUS}.86"),  # negative, uses MINUS sign
    (0.856,  3, False, ".856"),         # 3-decimal precision
    (0.856,  2, True,  "+.86"),         # signed=True adds +
    (-0.5,   2, True,  f"{MINUS}.50"),  # signed=True, negative stays MINUS
    (1.0,    2, False, "1.00"),         # value >= 1 keeps leading digit
    (0.0,    2, False, ".00"),          # zero
])
def test_fmt_r(r, digits, signed, expected):
    assert fmt_r(r, digits=digits, signed=signed) == expected


# ---------------------------------------------------------------------------
# fmt_d
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x,digits,expected", [
    (1.234,  2, "1.23"),
    (-0.567, 2, "-0.57"),  # unbounded: leading zero kept
    (0.0,    2, "0.00"),
    (10.0,   2, "10.00"),
    (float("nan"), 2, "nan"),
])
def test_fmt_d(x, digits, expected):
    assert fmt_d(x, digits=digits) == expected


# ---------------------------------------------------------------------------
# fmt_sig
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x,digits,expected", [
    (3.14159,  3, "3.14"),         # float → significant figures
    (42,       3, "42"),           # int stays int (no decimal)
    (None,     3, ""),             # None → empty string
    ("hello",  3, "hello"),        # string passthrough
    (True,     3, "True"),         # bool passthrough (not treated as int)
    (0.00123,  3, "0.00123"),      # small float — 3 sig figs
])
def test_fmt_sig(x, digits, expected):
    assert fmt_sig(x, digits=digits) == expected


# ---------------------------------------------------------------------------
# fmt_count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [
    (0,       "0"),
    (999,     "999"),
    (1000,    "1,000"),
    (1234567, "1,234,567"),
])
def test_fmt_count(n, expected):
    assert fmt_count(n) == expected


# ---------------------------------------------------------------------------
# fmt_pct
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x,digits,expected", [
    (0.0,   1, "0.0%"),
    (0.5,   1, "50.0%"),
    (1.0,   1, "100.0%"),
    (0.123, 2, "12.30%"),
])
def test_fmt_pct(x, digits, expected):
    assert fmt_pct(x, digits=digits) == expected


# ---------------------------------------------------------------------------
# truncate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,width,expected", [
    ("abc", 5, "abc"),
    ("abcde", 5, "abcde"),
    ("abcdefgh", 5, "abcd" + ELLIPSIS),
])
def test_truncate(text, width, expected):
    assert truncate(text, width) == expected


# ---------------------------------------------------------------------------
# fmt_table
# ---------------------------------------------------------------------------

def test_fmt_table_right_aligns_numeric_column():
    result = fmt_table(
        [["a", "1"], ["bb", "22"]], headers=["label", "n"], numeric=[False, True]
    )
    assert result == "label   n\na       1\nbb     22"


def test_fmt_table_empty_rows():
    result = fmt_table([], headers=["col1", "col2"], numeric=[False, True])
    assert "empty" in result.lower()


def test_fmt_table_text_truncate():
    rows = [["very_long_word_here", "1"]]
    headers = ["label", "n"]
    numeric = [False, True]
    result = fmt_table(rows, headers=headers, numeric=numeric, text_truncate=8)
    # The long word should be truncated to 8 chars (7 + ellipsis)
    assert "very_lo" + ELLIPSIS in result


# ---------------------------------------------------------------------------
# default_alignment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [
    (0, ()),
    (1, (False,)),
    (4, (False, True, True, True)),
])
def test_default_alignment(n, expected):
    assert default_alignment(n) == expected


# ---------------------------------------------------------------------------
# set_repr_hints toggle
# ---------------------------------------------------------------------------

def test_repr_hints_toggle_off(pls_result):
    set_repr_hints(False)
    try:
        r = repr(pls_result)
        assert "Save:" not in r
    finally:
        set_repr_hints(True)  # restore


@pytest.mark.parametrize("value", ["maybe", 1])
def test_repr_hints_bad_type_raises(value):
    with pytest.raises(TypeError):
        set_repr_hints(value)


# ---------------------------------------------------------------------------
# Result-class repr examples (one per class)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture, head, fact", [
    ("pls_result", "PLSResult  r²=", "p="),
    ("pcaols_result", "PCAOLSResult  r²=", "p="),
    ("group_result_2g", "GroupResult  G=2  n=50", "omnibus p="),
])
def test_result_repr(request, fixture, head, fact):
    set_repr_hints(True)
    r = repr(request.getfixturevalue(fixture))
    assert r.startswith(head) and fact in r.splitlines()[0]
    assert "Save:" in r


def test_lexicon_result_repr():
    """Construct a minimal LexiconResult in-memory (no fixture needed)."""
    from ssdiff.results.lexicon_result import LexiconResult
    from ssdiff.results.schema import Suggestion

    sug = Suggestion(
        token="kraj",
        freq=10,
        cov_all=0.8,
        cov_bal=0.7,
        corr=0.5,
        pvalue=0.01,
        direction="pos",
        rank=0.9,
    )
    lr = LexiconResult(var_type="continuous", n_docs=100, n_tokens=3, suggestions=[sug])
    set_repr_hints(True)
    r = repr(lr)
    assert r.startswith("LexiconResult  n_tokens=3  n_docs=100")
    assert "Save:" in r


# ---------------------------------------------------------------------------
# Suggestions of a LexiconResult with seeds
# ---------------------------------------------------------------------------

def _seeded_suggestions(n_rows: int):
    from ssdiff.results.lexicon_result import LexiconResult
    from ssdiff.results.schema import Suggestion

    suggestions = [
        Suggestion(
            token=f"tok_{i}", freq=10, cov_all=0.8, cov_bal=0.7, corr=0.5,
            pvalue=0.01, direction="positive", rank=0.9, seed="kraj", cos=1.0 - i / 100,
        )
        for i in range(n_rows)
    ]
    lr = LexiconResult(
        var_type="continuous", n_docs=100, n_tokens=n_rows, suggestions=suggestions,
        seeded=True,
    )
    return lr.suggestions


def test_seeded_suggestions_print_seed_first_and_cos_after_token():
    view = _seeded_suggestions(3)
    assert view.columns == (
        "seed", "token", "cos", "freq", "cov_all", "cov_bal", "corr", "pvalue",
        "direction", "rank",
    )
    header = view.to_text().splitlines()[0].split()
    assert header == ["seed", "rank", "token", "cos", "freq", "corr", "pvalue", "direction"]


def test_seeded_suggestions_slice_prints_every_row_with_the_seed_column():
    view = _seeded_suggestions(25)
    assert "... 5 more rows" in view.to_text()
    lines = view[:].to_text().splitlines()
    assert lines[0].split()[0] == "seed"
    assert "tok_24" in lines[-1]


def test_seeded_suggestions_save_to_lexicon_suggestions_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seeded_suggestions(3).save()
    assert (tmp_path / "lexicon_suggestions.csv").exists()


# ---------------------------------------------------------------------------
# View repr save hints
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("view_name, tail", [
    ("words", "Sides:"),
    ("docs", "Slice:"),
    ("snippets", "Size:"),
])
def test_view_repr_save_hint(pls_result, view_name, tail):
    set_repr_hints(True)
    r = repr(getattr(pls_result, view_name))
    assert f".save('{view_name}.csv'" in r
    listed = r.split("extensions:")[1].split("\n")[0].split()
    assert all(ext in listed for ext in TABULAR_EXTS)
    assert tail in r


def test_clusters_repr_save_hint(pls_result):
    set_repr_hints(True)
    r = repr(pls_result.clusters)
    assert ".save('clusters.csv')" in r and ".pos.save('clusters_pos.csv')" in r
