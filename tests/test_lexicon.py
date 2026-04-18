"""Tests for lexicon utilities — private helpers and Corpus lexicon methods."""

from __future__ import annotations

import numpy as np
import pytest

from ssdiff.corpus import Corpus
from ssdiff.results.lexicon_result import LexiconResult
from ssdiff.results.schema import Suggestion
from ssdiff.utils.lexicon import (
    _as_float_array,
    _cramers_v,
    _crosstab,
    _effect_direction,
    _quantile_bins,
    _texts_to_token_lists,
    _validate_var_type,
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
# Unified Suggestion fields used by all Corpus methods
# ---------------------------------------------------------------------------

EXPECTED_SUGGESTION_FIELDS = {"token", "freq", "cov_all", "cov_bal", "corr", "rank", "pvalue", "direction"}


# ---------------------------------------------------------------------------
# Corpus.suggest_lexicon
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_corpus():
    """Corpus built from pre-tokenized docs."""
    docs = [
        ["alpha", "beta", "gamma"],
        ["alpha", "delta"],
        ["beta", "gamma", "epsilon"],
        ["alpha", "beta"],
        ["gamma", "delta", "epsilon"],
        ["alpha", "gamma"],
    ]
    return Corpus(docs, pretokenized=True)


@pytest.fixture
def simple_y():
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


class TestCorpusSuggestLexicon:
    def test_returns_lexicon_result(self, simple_corpus, simple_y):
        import dataclasses
        result = simple_corpus.suggest_lexicon(simple_y, top_k=10, min_docs=1)
        assert isinstance(result, LexiconResult)
        # suggestions is the view
        assert len(result.suggestions) > 0
        # Each suggestion is a Suggestion dataclass with expected fields
        for s in result.suggestions:
            assert isinstance(s, Suggestion)
            fields = {f.name for f in dataclasses.fields(s)}
            assert fields == EXPECTED_SUGGESTION_FIELDS

    def test_tokens_property(self, simple_corpus, simple_y):
        result = simple_corpus.suggest_lexicon(simple_y, top_k=10, min_docs=1)
        assert isinstance(result.tokens, list)
        assert all(isinstance(t, str) for t in result.tokens)
        assert len(result.tokens) == len(result.suggestions)

    def test_top_k_limit(self, simple_corpus, simple_y):
        result = simple_corpus.suggest_lexicon(simple_y, top_k=2, min_docs=1)
        assert len(result.suggestions) <= 2

    def test_min_docs_filter(self, simple_corpus, simple_y):
        result = simple_corpus.suggest_lexicon(simple_y, top_k=100, min_docs=100)
        assert len(result.suggestions) == 0

    def test_categorical(self):
        import dataclasses
        docs = [["alpha", "beta"], ["alpha", "gamma"],
                ["beta", "delta"], ["gamma", "delta"]]
        corpus = Corpus(docs, pretokenized=True)
        groups = np.array(["A", "A", "B", "B"], dtype=object)
        result = corpus.suggest_lexicon(groups, top_k=10, min_docs=1, var_type="categorical")
        assert isinstance(result, LexiconResult)
        for s in result.suggestions:
            fields = {f.name for f in dataclasses.fields(s)}
            assert fields == EXPECTED_SUGGESTION_FIELDS

    def test_nan_y_filtered(self, simple_corpus):
        y = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
        result = simple_corpus.suggest_lexicon(y, top_k=10, min_docs=1)
        assert isinstance(result, LexiconResult)

    def test_sorted_by_rank_descending(self, simple_corpus, simple_y):
        result = simple_corpus.suggest_lexicon(simple_y, top_k=10, min_docs=1)
        if len(result.suggestions) > 1:
            ranks = [s.rank for s in result.suggestions]
            assert ranks == sorted(ranks, reverse=True)

    def test_report(self, simple_corpus, simple_y):
        result = simple_corpus.suggest_lexicon(simple_y, top_k=5, min_docs=1)
        text = result.report().to_text()
        assert isinstance(text, str)
        assert "Suggestions" in text


# ---------------------------------------------------------------------------
# Corpus.token_stats
# ---------------------------------------------------------------------------

class TestCorpusTokenStats:
    def test_basic_continuous(self, simple_corpus, simple_y):
        result = simple_corpus.token_stats(simple_y, ["alpha", "beta"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(set(r.keys()) == EXPECTED_SUGGESTION_FIELDS for r in result)

    def test_single_token(self, simple_corpus, simple_y):
        result = simple_corpus.token_stats(simple_y, ["alpha"])
        assert len(result) == 1
        d = result[0]
        assert d["token"] == "alpha"
        assert isinstance(d["freq"], int)
        assert d["freq"] > 0
        assert d["direction"] in ("positive", "negative", "none")

    def test_missing_token(self, simple_corpus, simple_y):
        result = simple_corpus.token_stats(simple_y, ["zzzznotfound"])
        assert result[0]["freq"] == 0

    def test_categorical(self):
        docs = [["alpha", "beta"], ["alpha"], ["beta", "gamma"], ["gamma"]]
        corpus = Corpus(docs, pretokenized=True)
        groups = np.array(["A", "A", "B", "B"], dtype=object)
        result = corpus.token_stats(groups, ["alpha"], var_type="categorical")
        assert result[0]["token"] == "alpha"
        assert result[0]["freq"] == 2

    def test_nan_y_filtered(self, simple_corpus):
        y = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
        result = simple_corpus.token_stats(y, ["alpha"])
        assert result[0]["freq"] >= 0


# ---------------------------------------------------------------------------
# Corpus.coverage_summary
# ---------------------------------------------------------------------------

class TestCorpusCoverageSummary:
    def test_basic(self, simple_corpus, simple_y):
        summary = simple_corpus.coverage_summary(simple_y, ["alpha", "beta"])
        assert isinstance(summary, dict)
        assert "docs_any" in summary
        assert "cov_all" in summary
        assert summary["docs_any"] > 0
        assert 0.0 <= summary["cov_all"] <= 1.0

    def test_empty_lexicon(self, simple_corpus, simple_y):
        summary = simple_corpus.coverage_summary(simple_y, [])
        assert summary["docs_any"] == 0

    def test_categorical(self):
        docs = [["alpha", "beta"], ["alpha", "gamma"],
                ["beta", "delta"], ["gamma", "delta"]]
        corpus = Corpus(docs, pretokenized=True)
        groups = np.array(["A", "A", "B", "B"], dtype=object)
        summary = corpus.coverage_summary(groups, ["alpha", "beta"], var_type="categorical")
        assert "group_cov" in summary
        assert isinstance(summary["group_cov"], dict)

    def test_hits_and_types_stats(self, simple_corpus, simple_y):
        summary = simple_corpus.coverage_summary(simple_y, ["alpha", "beta"])
        assert "hits_mean" in summary
        assert "hits_median" in summary
        assert "types_mean" in summary
        assert "types_median" in summary
        assert summary["hits_mean"] >= 0

    def test_all_nan_y(self):
        docs = [["alpha", "beta"], ["gamma", "delta"]]
        corpus = Corpus(docs, pretokenized=True)
        y = np.array([np.nan, np.nan])
        summary = corpus.coverage_summary(y, ["alpha"])
        assert summary["docs_any"] == 0


# ---------------------------------------------------------------------------
# Corpus.evaluate_lexicon
# ---------------------------------------------------------------------------

class TestCorpusEvaluateLexicon:
    def test_returns_lexicon_result_with_summary(self, simple_corpus, simple_y):
        result = simple_corpus.evaluate_lexicon(simple_y, ["alpha", "beta"])
        assert isinstance(result, LexiconResult)
        # summary is a SummaryView, not None
        assert result.summary is not None
        assert result.summary.docs_any is not None
        assert len(result.suggestions) == 2

    def test_tokens_property(self, simple_corpus, simple_y):
        result = simple_corpus.evaluate_lexicon(simple_y, ["alpha", "beta"])
        assert set(result.tokens) == {"alpha", "beta"}

    def test_report_includes_summary(self, simple_corpus, simple_y):
        result = simple_corpus.evaluate_lexicon(simple_y, ["alpha", "beta"])
        text = result.report().to_text()
        assert "Docs with any hit" in text or "docs_any" in text
        # Check for coverage info in report
        assert "Coverage summary" in text or "cov_all" in text

    def test_categorical(self):
        docs = [["alpha", "beta"], ["alpha", "gamma"],
                ["beta", "delta"], ["gamma", "delta"]]
        corpus = Corpus(docs, pretokenized=True)
        groups = np.array(["A", "A", "B", "B"], dtype=object)
        result = corpus.evaluate_lexicon(groups, ["alpha"], var_type="categorical")
        assert result.summary is not None
        # group_cov is exposed via summary view
        assert result.summary.group_cov is not None or result.summary.docs_any is not None
