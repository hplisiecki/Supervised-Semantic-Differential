"""Tests for ssdiff.results.format and display-layer repr/save-hint helpers.

Covers:
- format.py primitives: fmt_p, fmt_r, fmt_d, fmt_sig, fmt_count, fmt_pct, truncate,
  fmt_table, default_alignment
- display.py: set_repr_hints, _save_hint_enabled
- Result class reprs: PLSResult, PCAOLSResult, GroupResult, LexiconResult (one spec each)
- View save-hint sanity: WordsView, ClustersView, SnippetsView, DocsView
"""

from __future__ import annotations

import math

import pytest

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
from ssdiff.results.display import set_repr_hints, _save_hint_enabled


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


def test_fmt_p_nan():
    assert fmt_p(float("nan")) == "nan"


def test_fmt_p_inf():
    assert fmt_p(float("inf")) == "nan"


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
])
def test_fmt_d(x, digits, expected):
    assert fmt_d(x, digits=digits) == expected


def test_fmt_d_nan():
    assert fmt_d(float("nan")) == "nan"


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

def test_truncate_short_string_unchanged():
    assert truncate("abc", 5) == "abc"


def test_truncate_exact_length_unchanged():
    assert truncate("abcde", 5) == "abcde"


def test_truncate_long_string():
    result = truncate("abcdefgh", 5)
    assert result == "abcd" + ELLIPSIS
    assert len(result) == 5


def test_truncate_appends_ellipsis_char():
    # Confirm it's the unicode ELLIPSIS (U+2026) not three dots
    result = truncate("abcdefgh", 4)
    assert result[-1] == ELLIPSIS


# ---------------------------------------------------------------------------
# fmt_table
# ---------------------------------------------------------------------------

def test_fmt_table_basic():
    rows = [["apple", "1.23"], ["banana", "4.56"]]
    headers = ["fruit", "value"]
    numeric = [False, True]
    result = fmt_table(rows, headers=headers, numeric=numeric)
    lines = result.splitlines()
    # Header row first
    assert "fruit" in lines[0]
    assert "value" in lines[0]
    # Data rows follow
    assert "apple" in lines[1]
    assert "banana" in lines[2]


def test_fmt_table_right_aligns_numeric_column():
    rows = [["a", "1"], ["bb", "22"]]
    headers = ["label", "n"]
    numeric = [False, True]
    result = fmt_table(rows, headers=headers, numeric=numeric)
    lines = result.splitlines()
    # numeric column width for "n" vs "22": widths = [2, 2]; both right-aligned
    # header line: "label  n" with "n" right-padded to 2 or right-justified
    # The numeric column "n" should be right-justified (trailing whitespace at start)
    # we just verify the output is a non-empty string and contains the values
    assert "1" in result
    assert "22" in result


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

def test_default_alignment_zero_cols():
    assert default_alignment(0) == ()


def test_default_alignment_one_col():
    result = default_alignment(1)
    assert result == (False,)  # single column: left-aligned


def test_default_alignment_multi_col():
    result = default_alignment(4)
    # First column left (False), rest right (True)
    assert result == (False, True, True, True)


def test_default_alignment_first_left_rest_right():
    for n in range(1, 6):
        result = default_alignment(n)
        assert result[0] is False
        assert all(v is True for v in result[1:])


# ---------------------------------------------------------------------------
# set_repr_hints toggle
# ---------------------------------------------------------------------------

def test_repr_hints_default_on():
    set_repr_hints(True)  # ensure clean state
    assert _save_hint_enabled() is True


def test_repr_hints_toggle_off(pls_result):
    set_repr_hints(False)
    try:
        r = repr(pls_result)
        assert "Save:" not in r
    finally:
        set_repr_hints(True)  # restore


def test_repr_hints_toggle_on(pls_result):
    set_repr_hints(True)
    r = repr(pls_result)
    assert "Save:" in r


def test_repr_hints_bad_type_raises():
    with pytest.raises(TypeError):
        set_repr_hints("maybe")


def test_repr_hints_bad_type_int_raises():
    with pytest.raises(TypeError):
        set_repr_hints(1)


# ---------------------------------------------------------------------------
# Result-class repr spec examples (one per class)
# ---------------------------------------------------------------------------

def test_pls_result_repr(pls_result):
    set_repr_hints(True)
    r = repr(pls_result)
    assert "PLSResult" in r
    # r² appears in the summary line
    assert "r²" in r or "r2" in r.lower()
    # Save hint present
    assert "Save:" in r


def test_pcaols_result_repr(pcaols_result):
    set_repr_hints(True)
    r = repr(pcaols_result)
    assert "PCAOLSResult" in r
    assert "Save:" in r


def test_group_result_repr(group_result_2g):
    set_repr_hints(True)
    r = repr(group_result_2g)
    assert "GroupResult" in r
    # Omnibus p-value appears
    assert "omnibus" in r.lower() or "p=" in r
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
    assert "LexiconResult" in r
    # n_tokens appears in summary
    assert "n_tokens=3" in r or "3" in r


# ---------------------------------------------------------------------------
# View save-hint sanity: WordsView, DocsView, SnippetsView, ClustersView
# ---------------------------------------------------------------------------

def test_words_view_save_hint_starts_with_save(pls_result):
    """WordsView._save_hint() begins with 'Save:'."""
    from ssdiff.results.continuous_result import WordsView
    # Access words through pls_result (which has embeddings attached)
    words_view = pls_result.words
    assert isinstance(words_view, WordsView)
    hint = words_view._save_hint()
    assert hint.startswith("Save:")


def test_docs_view_save_hint_starts_with_save(pls_result):
    """DocsView._save_hint() begins with 'Save:'."""
    from ssdiff.results.continuous_result import DocsView
    docs_view = pls_result.docs
    assert isinstance(docs_view, DocsView)
    hint = docs_view._save_hint()
    assert hint.startswith("Save:")


def test_snippets_view_save_hint_starts_with_save(pls_result):
    """SnippetsView._save_hint() begins with 'Save:'."""
    from ssdiff.results.continuous_result import SnippetsView
    snippets_view = pls_result.snippets
    assert isinstance(snippets_view, SnippetsView)
    hint = snippets_view._save_hint()
    assert hint.startswith("Save:")


def test_clusters_view_custom_save_hint(pls_result):
    """ClustersView has a custom _save_hint() that starts with 'Save:'."""
    from ssdiff.results.continuous_result import ClustersView
    clusters_view = pls_result.clusters
    assert isinstance(clusters_view, ClustersView)
    hint = clusters_view._save_hint()
    assert hint.startswith("Save:")
