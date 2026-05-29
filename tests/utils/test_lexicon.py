"""Tests for ssdiff/utils/lexicon.py and Corpus lexicon methods."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import pointbiserialr

from ssdiff.corpus import Corpus
from ssdiff.utils.lexicon import (
    _as_float_array,
    _chi2_pvalue,
    _cramers_v,
    _crosstab,
    _effect_direction,
    _pointbiserial_pvalue,
    _quantile_bins,
    _validate_var_type,
)


# ---------------------------------------------------------------------------
# _as_float_array
# ---------------------------------------------------------------------------

def test_as_float_array_basic():
    result = _as_float_array([1, 2, 3])
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_as_float_array_none_becomes_nan():
    result = _as_float_array([1.0, None, 3.0])
    assert np.isnan(result[1])
    assert result[0] == 1.0
    assert result[2] == 3.0


# ---------------------------------------------------------------------------
# _quantile_bins
# ---------------------------------------------------------------------------

def test_quantile_bins_5_elements_2_bins():
    # edges = np.percentile([1,2,3,4,5], [0, 50, 100]) = [1, 3, 5]
    # unique edges = [1, 3, 5]; interior = [3]
    # searchsorted([3], [1,2,3,4,5], side='right') = [0, 0, 1, 1, 1]
    result = _quantile_bins(np.array([1, 2, 3, 4, 5]), n_bins=2)
    expected = np.array([0, 0, 1, 1, 1])
    np.testing.assert_array_equal(result, expected)


def test_quantile_bins_returns_integer_labels():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    result = _quantile_bins(y, n_bins=2)
    assert result.dtype in (np.int64, np.int32, np.intp)
    assert set(result).issubset({0, 1})


def test_quantile_bins_constant_falls_back_to_median_split():
    # All same value — can't form edges, should fallback to median split
    y = np.array([5.0, 5.0, 5.0, 5.0])
    result = _quantile_bins(y, n_bins=4)
    # Fallback: (arr > median).astype(int), median=5, so all False → all 0
    assert all(r == 0 for r in result)


# ---------------------------------------------------------------------------
# _crosstab
# ---------------------------------------------------------------------------

def test_crosstab_2x2_exact():
    # a=[0,0,1,1], b=[0,1,0,1] → row_labels=[0,1], col_labels=[0,1]
    # table[0,0]=1, table[0,1]=1, table[1,0]=1, table[1,1]=1
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 1, 0, 1])
    table, row_labels, col_labels = _crosstab(a, b)
    assert row_labels == [0, 1]
    assert col_labels == [0, 1]
    np.testing.assert_array_equal(table, [[1, 1], [1, 1]])


def test_crosstab_perfect_association():
    # a=[0,0,1,1], b=[0,0,1,1] → only diagonal cells filled
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 0, 1, 1])
    table, row_labels, col_labels = _crosstab(a, b)
    np.testing.assert_array_equal(table, [[2, 0], [0, 2]])


# ---------------------------------------------------------------------------
# _cramers_v
# ---------------------------------------------------------------------------

def test_cramers_v_uniform_table_is_zero():
    # 2x2 table all 25 → no association → V=0
    presence = np.array([0, 0, 1, 1] * 25)
    groups = np.array([0, 1, 0, 1] * 25)
    v = _cramers_v(presence, groups)
    assert abs(v) < 1e-12


def test_cramers_v_perfect_association_is_one():
    # Perfect 2x2: [0,0] and [1,1] exclusively → V=1
    presence = np.array([0, 0, 1, 1])
    groups = np.array([0, 0, 1, 1])
    v = _cramers_v(presence, groups)
    assert abs(v - 1.0) < 1e-12


def test_cramers_v_single_group_returns_zero():
    # Only one group — crosstab is 2x1 → V=0.0 (fallback)
    presence = np.array([0, 1, 0, 1])
    groups = np.array([0, 0, 0, 0])
    v = _cramers_v(presence, groups)
    assert v == 0.0


# ---------------------------------------------------------------------------
# _effect_direction
# ---------------------------------------------------------------------------

def test_effect_direction_categorical_positive():
    # groups: A and B; B (last sorted) has higher presence → positive
    presence = np.array([0.0, 0.0, 1.0, 1.0])
    groups = np.array(["A", "A", "B", "B"], dtype=object)
    direction = _effect_direction(presence, groups, categorical=True)
    assert direction == "positive"


def test_effect_direction_categorical_negative():
    # A has higher presence than B → negative (last < first)
    presence = np.array([1.0, 1.0, 0.0, 0.0])
    groups = np.array(["A", "A", "B", "B"], dtype=object)
    direction = _effect_direction(presence, groups, categorical=True)
    assert direction == "negative"


def test_effect_direction_continuous_positive():
    # presence correlates positively with y
    presence = np.array([0.0, 0.0, 1.0, 1.0])
    y = [1.0, 2.0, 3.0, 4.0]
    direction = _effect_direction(presence, y, categorical=False)
    assert direction == "positive"


def test_effect_direction_continuous_none_constant_presence():
    # Constant presence → std < 1e-12 → "none"
    presence = np.array([1.0, 1.0, 1.0, 1.0])
    y = [1.0, 2.0, 3.0, 4.0]
    direction = _effect_direction(presence, y, categorical=False)
    assert direction == "none"


# ---------------------------------------------------------------------------
# _chi2_pvalue (real signature: presence, groups)
# ---------------------------------------------------------------------------

def test_chi2_pvalue_no_association_returns_large_p():
    # Uniform table → chi2~0 → p close to 1
    presence = np.array([0, 0, 1, 1] * 25)
    groups = np.array([0, 1, 0, 1] * 25)
    p = _chi2_pvalue(presence, groups)
    assert np.isfinite(p)
    assert p > 0.5


def test_chi2_pvalue_perfect_association_returns_small_p():
    # Perfect association → large chi2 → p very small
    presence = np.array([0] * 50 + [1] * 50)
    groups = np.array([0] * 50 + [1] * 50)
    p = _chi2_pvalue(presence, groups)
    assert np.isfinite(p)
    assert p < 0.001


def test_chi2_pvalue_single_group_returns_nan():
    # Only one group → crosstab is 2x1 → shape[1] < 2 → nan
    presence = np.array([0, 1, 0, 1])
    groups = np.array([0, 0, 0, 0])
    p = _chi2_pvalue(presence, groups)
    assert math.isnan(p)


# ---------------------------------------------------------------------------
# _pointbiserial_pvalue (vs. scipy)
# ---------------------------------------------------------------------------

def test_pointbiserial_pvalue_matches_scipy():
    presence = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    y = np.array([1.0, 1.2, 0.9, 0.8, 1.5, 1.1, 0.7, 1.3])
    our_p = _pointbiserial_pvalue(presence, y)
    _, scipy_p = pointbiserialr(presence, y)
    assert abs(our_p - scipy_p) < 1e-10


def test_pointbiserial_pvalue_constant_returns_nan():
    presence = np.array([1.0, 1.0, 1.0, 1.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = _pointbiserial_pvalue(presence, y)
    assert math.isnan(p)


# ---------------------------------------------------------------------------
# _validate_var_type
# ---------------------------------------------------------------------------

def test_validate_var_type_valid():
    _validate_var_type("continuous")
    _validate_var_type("categorical")


def test_validate_var_type_invalid():
    with pytest.raises(ValueError, match="var_type"):
        _validate_var_type("ordinal")


# ---------------------------------------------------------------------------
# Corpus.suggest_lexicon
# ---------------------------------------------------------------------------

def test_suggest_lexicon_returns_lexicon_result(sample_docs, sample_y):
    from ssdiff.results.lexicon_result import LexiconResult
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    result = corpus.suggest_lexicon(sample_y, top_k=5, min_docs=1)
    assert isinstance(result, LexiconResult)


def test_suggest_lexicon_top_k_respected(sample_docs, sample_y):
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    result = corpus.suggest_lexicon(sample_y, top_k=3, min_docs=1)
    assert len(result.tokens) <= 3


def test_suggest_lexicon_tokens_list_not_empty(sample_docs, sample_y):
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    result = corpus.suggest_lexicon(sample_y, min_docs=1)
    assert len(result.tokens) > 0
    assert all(isinstance(t, str) for t in result.tokens)


# ---------------------------------------------------------------------------
# Corpus.token_stats
# ---------------------------------------------------------------------------

def test_token_stats_returns_list_of_dicts(sample_docs, sample_y):
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    lex = ["kraj", "narod", "panstwo"]
    rows = corpus.token_stats(sample_y, lex)
    assert isinstance(rows, list)
    assert len(rows) == 3
    for row in rows:
        assert isinstance(row, dict)
        assert "token" in row
        assert "freq" in row
        assert "cov_all" in row
        assert "rank" in row
        assert "direction" in row


def test_token_stats_freq_correct(sample_docs, sample_y):
    # From conftest: "kraj" appears in docs 0, 3, 6 → freq=3
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    rows = corpus.token_stats(sample_y, ["kraj"])
    kraj_row = next(r for r in rows if r["token"] == "kraj")
    assert kraj_row["freq"] == 3


def test_token_stats_cov_all_correct(sample_docs, sample_y):
    # 8 docs, "kraj" in 3 → cov_all = 3/8 = 0.375
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    rows = corpus.token_stats(sample_y, ["kraj"])
    kraj_row = next(r for r in rows if r["token"] == "kraj")
    assert abs(kraj_row["cov_all"] - 3 / 8) < 1e-10


# ---------------------------------------------------------------------------
# Corpus.coverage_summary
# ---------------------------------------------------------------------------

def test_coverage_summary_returns_dict(sample_docs, sample_y):
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    summary = corpus.coverage_summary(sample_y, {"kraj"})
    assert isinstance(summary, dict)
    assert "docs_any" in summary
    assert "cov_all" in summary
    assert "q1" in summary
    assert "q4" in summary


def test_coverage_summary_kraj_fraction(sample_docs, sample_y):
    # "kraj" appears in docs 0, 3, 6 → 3/8 docs = 0.375
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    summary = corpus.coverage_summary(sample_y, {"kraj"})
    assert summary["docs_any"] == 3
    assert abs(summary["cov_all"] - 3 / 8) < 1e-10


def test_coverage_summary_full_lexicon_all_docs(sample_docs, sample_y):
    # All docs contain at least one of kraj/narod/panstwo → docs_any=8
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    summary = corpus.coverage_summary(sample_y, {"kraj", "narod", "panstwo"})
    assert summary["docs_any"] == 8
    assert abs(summary["cov_all"] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Corpus.evaluate_lexicon
# ---------------------------------------------------------------------------

def test_evaluate_lexicon_returns_lexicon_result(sample_docs, sample_y):
    from ssdiff.results.lexicon_result import LexiconResult
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    result = corpus.evaluate_lexicon(sample_y, ["kraj", "narod"])
    assert isinstance(result, LexiconResult)
    assert result.summary is not None


def test_evaluate_lexicon_empty_lexicon_graceful(sample_docs, sample_y):
    # Empty lexicon should not raise — zero-token case
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    result = corpus.evaluate_lexicon(sample_y, [])
    assert result is not None
    assert len(result.tokens) == 0
    assert result.summary is not None
    # cov_all=0 since no tokens; docs_any=0
    assert result.summary.docs_any == 0
    assert result.summary.cov_all == 0.0


def test_evaluate_lexicon_summary_matches_coverage_summary(sample_docs, sample_y):
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    lex = ["kraj", "narod", "panstwo"]
    result = corpus.evaluate_lexicon(sample_y, lex)
    direct = corpus.coverage_summary(sample_y, lex)
    assert result.summary.docs_any == direct["docs_any"]
    assert abs(result.summary.cov_all - direct["cov_all"]) < 1e-12
