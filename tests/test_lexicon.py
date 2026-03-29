"""Tests for ssdlite/utils/lexicon.py — lexicon suggestion and coverage."""

from __future__ import annotations

import numpy as np
import pytest

from ssdlite.utils.lexicon import (
    _as_float_array,
    _texts_to_token_lists,
    _quantile_bins,
    _crosstab,
    _cramers_v,
    _effect_direction,
    _validate_var_type,
    suggest_lexicon,
    token_presence_stats,
    coverage_by_lexicon,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestAsFloatArray:
    def test_ints(self):
        result = _as_float_array([1, 2, 3])
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_none_becomes_nan(self):
        result = _as_float_array([1.0, None, 3.0])
        assert np.isnan(result[1])

    def test_mixed_types(self):
        result = _as_float_array([1, 2.5, "3"])
        assert result[0] == 1.0
        assert result[1] == 2.5
        assert result[2] == 3.0


class TestTextsToTokenLists:
    def test_strings_split(self):
        result = _texts_to_token_lists(["hello world", "foo bar"])
        assert result == [["hello", "world"], ["foo", "bar"]]

    def test_already_tokenized(self):
        result = _texts_to_token_lists([["hello", "world"], ["foo"]])
        assert result == [["hello", "world"], ["foo"]]

    def test_empty(self):
        assert _texts_to_token_lists([]) == []


class TestQuantileBins:
    def test_basic(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        bins = _quantile_bins(y, n_bins=4)
        assert len(bins) == len(y)
        assert bins.min() >= 0

    def test_all_same_value(self):
        y = np.array([5.0, 5.0, 5.0, 5.0])
        bins = _quantile_bins(y, n_bins=4)
        assert len(bins) == 4


class TestCrosstab:
    def test_basic(self):
        a = np.array([0, 0, 1, 1])
        b = np.array(["A", "B", "A", "B"])
        table, rows, cols = _crosstab(a, b)
        assert table.shape == (2, 2)
        assert rows == [0, 1]
        assert cols == ["A", "B"]
        assert table.sum() == 4

    def test_counts_correct(self):
        a = np.array([1, 1, 1, 0, 0])
        b = np.array(["X", "X", "Y", "X", "Y"])
        table, _, _ = _crosstab(a, b)
        # row 0 = a==0, row 1 = a==1
        assert table[1, 0] == 2  # a=1, b=X
        assert table[0, 1] == 1  # a=0, b=Y


class TestCramersV:
    def test_perfect_association(self):
        presence = np.array([1, 1, 0, 0])
        groups = np.array(["A", "A", "B", "B"])
        v = _cramers_v(presence, groups)
        assert v == pytest.approx(1.0)

    def test_no_association(self):
        presence = np.array([1, 0, 1, 0])
        groups = np.array(["A", "A", "B", "B"])
        v = _cramers_v(presence, groups)
        assert v == pytest.approx(0.0)

    def test_single_row_returns_zero(self):
        presence = np.array([1, 1, 1, 1])
        groups = np.array(["A", "B", "A", "B"])
        v = _cramers_v(presence, groups)
        assert v == 0.0


class TestEffectDirection:
    def test_positive_continuous(self):
        presence = np.array([0, 0, 0, 1, 1, 1])
        y = np.array([1.0, 2.0, 3.0, 7.0, 8.0, 9.0])
        assert _effect_direction(presence, y, categorical=False) == "positive"

    def test_negative_continuous(self):
        presence = np.array([1, 1, 1, 0, 0, 0])
        y = np.array([1.0, 2.0, 3.0, 7.0, 8.0, 9.0])
        assert _effect_direction(presence, y, categorical=False) == "negative"

    def test_none_when_constant(self):
        presence = np.array([1, 1, 1, 1])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert _effect_direction(presence, y, categorical=False) == "none"

    def test_categorical_direction(self):
        presence = np.array([0, 0, 1, 1])
        groups = np.array(["A", "A", "B", "B"], dtype=object)
        # Token more present in later group (B) → positive
        assert _effect_direction(presence, groups, categorical=True) == "positive"


class TestValidateVarType:
    def test_valid_continuous(self):
        _validate_var_type("continuous")  # no error

    def test_valid_categorical(self):
        _validate_var_type("categorical")  # no error

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="var_type must be"):
            _validate_var_type("ordinal")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_texts():
    return ["alpha beta gamma", "alpha delta", "beta gamma epsilon",
            "alpha beta", "gamma delta epsilon", "alpha gamma"]


@pytest.fixture
def simple_y():
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


class TestSuggestLexicon:
    def test_returns_list(self, simple_texts, simple_y):
        result = suggest_lexicon(
            (simple_texts, simple_y), top_k=10, min_docs=1
        )
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(t, str) for t in result)

    def test_top_k_limit(self, simple_texts, simple_y):
        result = suggest_lexicon(
            (simple_texts, simple_y), top_k=2, min_docs=1
        )
        assert len(result) <= 2

    def test_min_docs_filter(self, simple_texts, simple_y):
        result = suggest_lexicon(
            (simple_texts, simple_y), top_k=100, min_docs=100
        )
        assert result == []

    def test_dict_input(self, simple_texts, simple_y):
        data = {"text": simple_texts, "score": simple_y.tolist()}
        result = suggest_lexicon(
            data, text_col="text", score_col="score", top_k=10, min_docs=1
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_categorical(self):
        texts = ["alpha beta", "alpha gamma", "beta delta", "gamma delta"]
        groups = ["A", "A", "B", "B"]
        result = suggest_lexicon(
            (texts, groups), top_k=10, min_docs=1, var_type="categorical"
        )
        assert isinstance(result, list)

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            suggest_lexicon("not a valid input")

    def test_dict_missing_cols_raises(self):
        with pytest.raises(ValueError, match="text_col and score_col"):
            suggest_lexicon({"text": ["a"]})

    def test_nan_y_filtered(self, simple_texts):
        y = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
        result = suggest_lexicon(
            (simple_texts, y), top_k=10, min_docs=1
        )
        assert isinstance(result, list)


class TestTokenPresenceStats:
    def test_basic_continuous(self, simple_texts, simple_y):
        result = token_presence_stats(simple_texts, simple_y, "alpha")
        assert len(result) == 1
        d = result[0]
        assert d["token"] == "alpha"
        assert isinstance(d["frequency"], int)
        assert d["frequency"] > 0
        assert "association" in d
        assert "pvalue" in d
        assert d["effect_direction"] in ("positive", "negative", "none")

    def test_missing_token(self, simple_texts, simple_y):
        result = token_presence_stats(simple_texts, simple_y, "zzzznotfound")
        assert result[0]["frequency"] == 0

    def test_categorical(self):
        texts = ["alpha beta", "alpha", "beta gamma", "gamma"]
        groups = np.array(["A", "A", "B", "B"], dtype=object)
        result = token_presence_stats(
            texts, groups, "alpha", var_type="categorical"
        )
        assert result[0]["token"] == "alpha"
        assert result[0]["frequency"] == 2

    def test_nan_y_filtered(self, simple_texts):
        y = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
        result = token_presence_stats(simple_texts, y, "alpha")
        assert result[0]["frequency"] >= 0


class TestCoverageByLexicon:
    def test_basic(self, simple_texts, simple_y):
        summary, per_token = coverage_by_lexicon(
            (simple_texts, simple_y), lexicon=["alpha", "beta"]
        )
        assert isinstance(summary, dict)
        assert "docs_any" in summary
        assert "cov_all" in summary
        assert summary["docs_any"] > 0
        assert 0.0 <= summary["cov_all"] <= 1.0
        assert isinstance(per_token, list)
        assert len(per_token) == 2

    def test_per_token_keys(self, simple_texts, simple_y):
        _, per_token = coverage_by_lexicon(
            (simple_texts, simple_y), lexicon=["alpha"]
        )
        d = per_token[0]
        assert d["token"] == "alpha"
        assert "frequency" in d
        assert "association" in d
        assert "pvalue" in d
        assert "effect_direction" in d

    def test_empty_lexicon(self, simple_texts, simple_y):
        summary, per_token = coverage_by_lexicon(
            (simple_texts, simple_y), lexicon=[]
        )
        assert summary["docs_any"] == 0
        assert per_token == []

    def test_dict_input(self, simple_texts, simple_y):
        data = {"text": simple_texts, "score": simple_y.tolist()}
        summary, _ = coverage_by_lexicon(
            data, text_col="text", score_col="score", lexicon=["alpha"]
        )
        assert summary["docs_any"] > 0

    def test_categorical(self):
        texts = ["alpha beta", "alpha gamma", "beta delta", "gamma delta"]
        groups = ["A", "A", "B", "B"]
        summary, per_token = coverage_by_lexicon(
            (texts, groups),
            lexicon=["alpha", "beta"],
            var_type="categorical",
        )
        assert "group_cov" in summary
        assert isinstance(summary["group_cov"], dict)

    def test_hits_and_types_stats(self, simple_texts, simple_y):
        summary, _ = coverage_by_lexicon(
            (simple_texts, simple_y), lexicon=["alpha", "beta"]
        )
        assert "hits_mean" in summary
        assert "hits_median" in summary
        assert "types_mean" in summary
        assert "types_median" in summary
        assert summary["hits_mean"] >= 0

    def test_all_nan_y(self):
        texts = ["alpha beta", "gamma delta"]
        y = np.array([np.nan, np.nan])
        summary, per_token = coverage_by_lexicon(
            (texts, y), lexicon=["alpha"]
        )
        assert summary["docs_any"] == 0
