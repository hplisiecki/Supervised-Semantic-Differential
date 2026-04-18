"""LexiconResult view contract."""

import pytest

from ssdiff.results.lexicon_result import LexiconResult
from ssdiff.results.schema import Suggestion


def _mk(summary=None):
    rows = [
        Suggestion(rank=1, token="excellent", freq=87, cov_all=0.07,
                   cov_bal=0.073, corr=0.42, pvalue=1e-9, direction="+"),
        Suggestion(rank=2, token="terrible", freq=62, cov_all=0.05,
                   cov_bal=0.048, corr=-0.38, pvalue=1e-9, direction="-"),
    ]
    return LexiconResult(
        var_type="continuous", n_docs=1240, n_tokens=2,
        suggestions=rows, summary=summary, corpus=None,
    )


def test_stats_exposes_fields():
    lex = _mk()
    assert lex.stats.var_type == "continuous"
    assert lex.stats.n_docs == 1240
    assert lex.stats.n_tokens == 2


def test_suggestions_view_iterates_rows():
    lex = _mk()
    assert len(lex.suggestions) == 2
    tokens = [s.token for s in lex.suggestions]
    assert tokens == ["excellent", "terrible"]


def test_tokens_property_is_rank_ordered_list():
    lex = _mk()
    assert lex.tokens == ["excellent", "terrible"]


def test_summary_absent_when_none():
    lex = _mk(summary=None)
    assert lex.summary is None


def test_summary_present_when_provided():
    from ssdiff.results.schema import Summary
    s = Summary(docs_any=464, cov_all=0.374, q1=0.183, q4=0.612,
                corr_any=0.31, hits_mean=0.82, hits_median=0.0,
                types_mean=0.52, types_median=0.0, group_cov=None)
    lex = _mk(summary=s)
    assert abs(lex.summary.cov_all - 0.374) < 1e-9


def test_report_contains_suggestions_table():
    lex = _mk()
    text = lex.report().to_text()
    assert "excellent" in text
    assert "terrible" in text
