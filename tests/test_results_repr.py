"""Tests for console-accessible repr API (spec_console_repr.md)."""

from __future__ import annotations

import numpy as np
import pytest

import ssdiff
from ssdiff.results.continuous_result import WordsView
from ssdiff.results.display import _save_hint_enabled
from ssdiff.results.group_result import PairsListView
from ssdiff.results.schema import Pair, Word


def test_set_repr_hints_default_true():
    # Default state at import time
    assert _save_hint_enabled() is True


def test_set_repr_hints_toggle_off_then_on():
    ssdiff.set_repr_hints(False)
    assert _save_hint_enabled() is False
    ssdiff.set_repr_hints(True)
    assert _save_hint_enabled() is True


def test_set_repr_hints_returns_none():
    assert ssdiff.set_repr_hints(True) is None


def test_set_repr_hints_validates_bool():
    with pytest.raises(TypeError):
        ssdiff.set_repr_hints("yes")  # type: ignore[arg-type]


def _make_words(n: int) -> list[Word]:
    return [Word(side="pos", rank=i, word=f"w{i}",
                 cos_beta=float(i) / 100, contrast=None)
            for i in range(n)]


def test_view_slice_returns_same_type():
    v = WordsView(_make_words(50))
    sliced = v[:5]
    assert isinstance(sliced, WordsView)
    assert len(sliced) == 5
    assert [w.word for w in sliced] == [f"w{i}" for i in range(5)]


def test_view_slice_full_range_no_truncation_flag():
    v = WordsView(_make_words(50))
    full = v[:]
    assert isinstance(full, WordsView)
    assert len(full) == 50
    assert full._no_trunc is True


def test_view_slice_middle_range():
    v = WordsView(_make_words(50))
    sub = v[5:10]
    assert isinstance(sub, WordsView)
    assert len(sub) == 5
    assert [w.rank for w in sub] == [5, 6, 7, 8, 9]
    assert sub._no_trunc is True


def test_view_int_index_unchanged():
    v = WordsView(_make_words(3))
    assert v[0].word == "w0"
    assert v[2].word == "w2"


def test_pairs_list_view_tuple_lookup_unaffected_by_slice_dispatch():
    p1 = Pair(contrast="a_b", g1="a", g2="b", T=1.0, p_raw=0.01,
              p_corrected=0.01, cohens_d=0.5, n_g1=10, n_g2=10,
              contrast_norm=1.0)
    p2 = Pair(contrast="a_c", g1="a", g2="c", T=2.0, p_raw=0.02,
              p_corrected=0.02, cohens_d=0.6, n_g1=10, n_g2=10,
              contrast_norm=1.0)
    view = PairsListView([p1, p2])
    looked_up = view["a", "b"]
    # Tuple lookup now returns the Pair dataclass directly, not a wrapper.
    assert isinstance(looked_up, Pair)
    sliced = view[:1]
    assert isinstance(sliced, PairsListView)
    assert len(sliced) == 1


def test_scalar_view_slice_is_noop():
    """ScalarView[:] returns self (1-row degenerate slice)."""
    from ssdiff.results.continuous_result import StatsView
    from ssdiff.results.schema import Stats
    sv = StatsView(Stats(
        backend="PLS", r2=0.3, r2_adj=0.29, pvalue=0.001,
        n_raw=100, n_kept=100, n_dropped=0,
        y_mean=0.0, y_std=1.0, beta_norm=1.0,
        delta=0.1, iqr_effect=0.1, y_corr_pred=0.5,
    ))
    assert sv[:] is sv


def test_view_to_text_truncates_at_max_rows_with_footer():
    v = WordsView(_make_words(50))
    out = v.to_text(max_rows=20)
    lines = out.splitlines()
    assert len(lines) == 22
    assert lines[-1] == "... 30 more rows"


def test_view_to_text_no_footer_when_within_limit():
    v = WordsView(_make_words(10))
    out = v.to_text(max_rows=20)
    lines = out.splitlines()
    assert len(lines) == 11
    assert "more rows" not in out


def test_view_to_text_skips_footer_when_no_trunc():
    """Sliced views never show the footer (length is exact by construction)."""
    v = WordsView(_make_words(50))
    full = v[:]
    out = full.to_text(max_rows=20)
    assert "more rows" not in out
    lines = out.splitlines()
    assert len(lines) == 51


def test_view_to_text_uses_first_left_rest_right():
    v = WordsView(_make_words(3))
    out = v.to_text()
    lines = out.splitlines()
    assert lines[0].startswith("side")


def test_view_drops_to_latex():
    v = WordsView(_make_words(2))
    assert not hasattr(v, "to_latex")


def test_view_drops_to_docx():
    v = WordsView(_make_words(2))
    assert not hasattr(v, "to_docx")


def test_view_drops_to_markdown():
    v = WordsView(_make_words(2))
    assert not hasattr(v, "to_markdown")


def test_view_drops_to_excel():
    v = WordsView(_make_words(2))
    assert not hasattr(v, "to_excel")


def test_scalar_view_drops_data_export_extras():
    from ssdiff.results.continuous_result import StatsView
    from ssdiff.results.schema import Stats
    sv = StatsView(Stats(
        backend="PLS", r2=0.3, pvalue=0.001,
        n_raw=100, n_kept=100, n_dropped=0,
        y_mean=0.0, y_std=1.0, beta_norm=1.0,
        delta=0.1, iqr_effect=0.1, y_corr_pred=0.5,
    ))
    for name in ("to_latex", "to_docx", "to_markdown", "to_excel",
                 "to_csv", "to_json"):
        assert not hasattr(sv, name), f"ScalarView still has .{name}"
    # save() is the canonical file-output surface:
    assert hasattr(sv, "save")


def test_result_to_text_wraps_summary_with_access_and_save_hint():
    from ssdiff.results.core import Result

    class _FakeResult(Result):
        _access = ("stats", "words")

        def _summary(self) -> str:
            return "FakeResult  metric=42"

        def _summary_html(self) -> str:
            return "<p>FakeResult metric=42</p>"

        def _save_hint(self) -> str:
            return "Save:  .save('out.md')"

        def _save_hint_html(self) -> str:
            return "<pre>Save:  .save('out.md')</pre>"

    r = _FakeResult()
    text = r.to_text()
    assert "FakeResult" in text
    assert "views:" in text
    assert "stats" in text
    assert "words" in text
    assert "Save:" not in text


def test_result_repr_appends_save_hint():
    ssdiff.set_repr_hints(True)
    from ssdiff.results.core import Result

    class _FakeResult(Result):
        _access = ("stats",)
        def _summary(self): return "FakeResult"
        def _summary_html(self): return "<p>FakeResult</p>"
        def _save_hint(self): return "Save:  .save('out.md')"
        def _save_hint_html(self): return "<pre>Save:  .save('out.md')</pre>"

    r = _FakeResult()
    out = repr(r)
    assert "FakeResult" in out
    assert out.rstrip().endswith(".save('out.md')")


def test_result_repr_omits_save_hint_when_disabled():
    ssdiff.set_repr_hints(False)
    try:
        from ssdiff.results.core import Result

        class _FakeResult(Result):
            _access = ("stats",)
            def _summary(self): return "FakeResult"
            def _summary_html(self): return "<p>FakeResult</p>"
            def _save_hint(self): return "Save:  .save('out.md')"
            def _save_hint_html(self): return "<pre>save</pre>"

        r = _FakeResult()
        assert "Save:" not in repr(r)
    finally:
        ssdiff.set_repr_hints(True)


def test_result_repr_html_symmetric_with_repr():
    """Critical: __repr__ and _repr_html_ are both compact (not asymmetric)."""
    ssdiff.set_repr_hints(True)
    from ssdiff.results.core import Result

    class _FakeResult(Result):
        _access = ("stats",)
        def _summary(self): return "FakeResult"
        def _summary_html(self): return "<p>FakeResult</p>"
        def _save_hint(self): return "Save: hint"
        def _save_hint_html(self): return "<pre>save hint</pre>"

    r = _FakeResult()
    text = repr(r)
    html = r._repr_html_()
    assert "FakeResult" in text
    assert "FakeResult" in html
    assert "Plisiecki" not in text
    assert "Plisiecki" not in html


def test_continuous_result_drops_formatted_export_methods():
    """Result has no bundle-save or file-export methods — save() lives per-view."""
    from ssdiff.results.continuous_result import PLSResult
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 3))
    beta = rng.normal(size=3)
    y = x @ beta
    pls = PLSResult(
        x=x, beta=beta, keep_mask=np.ones(10, dtype=bool),
        n_raw=10, n_kept=10, n_dropped=0,
        y=y, _y_mean=np.array([y.mean()]),
        _y_scale=np.array([y.std() or 1.0]),
        r2=0.5, pvalue=0.01,
    )
    assert hasattr(pls, "to_text")
    assert hasattr(pls, "to_html")
    for name in ("to_markdown", "to_latex", "to_docx",
                 "to_csv", "to_json", "to_excel", "save"):
        assert not hasattr(pls, name), f"PLSResult unexpectedly has .{name}"
    # per-view save is the canonical file-output surface:
    assert hasattr(pls.stats, "save")
    assert hasattr(pls.docs, "save")
    assert hasattr(pls.report(), "save")


def test_continuous_result_to_text_is_compact_not_full_report():
    """Result.to_text returns the compact summary, not the full report."""
    from ssdiff.results.continuous_result import PLSResult
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 3))
    beta = rng.normal(size=3)
    y = x @ beta
    pls = PLSResult(
        x=x, beta=beta, keep_mask=np.ones(10, dtype=bool),
        n_raw=10, n_kept=10, n_dropped=0,
        y=y, _y_mean=np.array([y.mean()]),
        _y_scale=np.array([y.std() or 1.0]),
        r2=0.5, r2_adj=0.4, pvalue=0.01,
    )
    text = pls.to_text()
    assert "PLSResult" in text
    assert "Plisiecki" not in text


def test_clusters_index_repr_when_uncached():
    """If clusters not yet computed, repr says '(call to compute)' — never trigger compute."""
    from ssdiff.results.continuous_result import PLSResult
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 3))
    beta = rng.normal(size=3)
    y = x @ beta
    pls = PLSResult(
        x=x, beta=beta, keep_mask=np.ones(10, dtype=bool),
        n_raw=10, n_kept=10, n_dropped=0,
        y=y, _y_mean=np.array([y.mean()]),
        _y_scale=np.array([y.std() or 1.0]),
        r2=0.5, r2_adj=0.4, pvalue=0.01,
    )
    text = repr(pls.clusters)
    assert "ClustersView" in text
    assert ".pos" in text
    assert ".neg" in text
    assert "call to compute" in text
    assert not any(name == "clusters" for (name, _) in pls._cache)


def test_clusters_index_save_hint_present():
    from ssdiff.results.continuous_result import PLSResult
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 3))
    beta = rng.normal(size=3)
    y = x @ beta
    pls = PLSResult(
        x=x, beta=beta, keep_mask=np.ones(10, dtype=bool),
        n_raw=10, n_kept=10, n_dropped=0,
        y=y, _y_mean=np.array([y.mean()]),
        _y_scale=np.array([y.std() or 1.0]),
        r2=0.5, r2_adj=0.4, pvalue=0.01,
    )
    out = repr(pls.clusters)
    assert "clusters_pos.csv" in out
    assert "clusters_neg.csv" in out


def test_report_save_docx_writes_file(tmp_path):
    pytest.importorskip("docx")
    from ssdiff.results.report import Report, Section
    rep = Report(title="T", subtitle=None,
                 sections=[Section(title="S", kind="kv", rows=[("k", "v")])])
    out = tmp_path / "r.docx"
    rep.save(str(out))
    assert out.exists()


# ---------- spec acceptance tests (Examples 1-12 from spec_console_repr.md) ----------



def _shared_pls():
    """Reuse the same builder as save-hint tests."""
    from ssdiff.results.continuous_result import PLSResult
    rng = np.random.default_rng(0)
    n, d = 30, 4
    x = rng.normal(size=(n, d))
    beta = rng.normal(size=d)
    y = x @ beta
    return PLSResult(
        x=x, beta=beta, keep_mask=np.ones(n, dtype=bool),
        n_raw=n, n_kept=n, n_dropped=0,
        y=y, _y_mean=np.array([y.mean()]),
        _y_scale=np.array([y.std() or 1.0]),
        r2=0.30, r2_adj=0.29, pvalue=4.83e-08,
        test_name="split", test_info={"pvalue": 4.83e-08, "split_r2": 0.33,
                                       "n_splits": 30, "split_ratio": 0.5,
                                       "random_state": None},
    )


def test_spec_example_1_pls_repr_shape():
    """Example 1: `pls = ssd.fit_pls(); pls`"""
    import ssdiff
    ssdiff.set_repr_hints(True)
    pls = _shared_pls()
    text = repr(pls)
    # Header line
    assert "PLSResult" in text
    assert "r²" in text
    assert "backend=PLS" in text
    # Access lines
    for v in ("stats", "fit_info", "words", "clusters",
              "snippets", "docs", "test"):
        assert v in text
    for m in ("report()", "test(", "attach("):
        assert m in text
    # Save hint canonical lines
    assert "report().save('report.md')" in text
    assert "words.save('words.csv')" in text
    assert "docs.to_df()" in text


def test_spec_example_2_pls_stats_repr_shape():
    """Example 2: `pls.stats` — narrowed defaults."""
    pls = _shared_pls()
    text = repr(pls.stats)
    body = text.split("Save:")[0]
    # Default columns for StatsView per docs/results_tables.md
    for k in ("backend", "r2", "pvalue", "n_kept", "iqr_effect"):
        assert k in body
    # PLS: no r2_adj (OLS-only statistic)
    assert "r2_adj" not in body
    # Non-default columns must not appear in narrowed repr body
    for k in ("n_raw", "n_dropped", "y_mean", "y_std",
             "beta_norm", "delta", "y_corr_pred"):
        assert k not in body
    assert "Save:" in text
    assert ".to_dict()" in text
    assert ".to_df()" in text


def test_spec_example_3_pls_test_repr_shape():
    """Example 3: `pls.test` — narrowed defaults."""
    pls = _shared_pls()
    text = repr(pls.test)
    body = text.split("Save:")[0]
    for k in ("name", "pvalue", "split_r2"):
        assert k in body
    for k in ("n_splits", "split_ratio", "random_state"):
        assert k not in body
    assert "Save:" in text
    assert "Rerun:" in text
    assert "n_perm=" in text
    assert "n_splits=" in text


def test_spec_example_4_pls_words_repr_shows_both_sides():
    """`pls.words` splits max_rows: top N pos + top N neg."""
    from ssdiff.results.continuous_result import WordsView
    from ssdiff.results.schema import Word
    rows = [Word(side="pos", rank=i, word=f"p{i}",
                 cos_beta=0.32 - 0.001 * i, contrast=None)
            for i in range(100)]
    rows += [Word(side="neg", rank=i, word=f"n{i}",
                  cos_beta=0.30 - 0.001 * i, contrast=None)
             for i in range(100)]
    v = WordsView(rows)
    text = repr(v)
    for h in ("side", "rank", "word", "cos_beta"):
        assert h in text
    # Both sides visible, footer mentions hidden rows
    assert "pos" in text and "neg" in text
    assert "p0" in text and "n0" in text
    assert "more rows" in text
    assert ".save('words.csv'," in text


def test_spec_example_5_clusters_index_uncached():
    """Example 5: `pls.clusters` — uncached path."""
    pls = _shared_pls()
    text = repr(pls.clusters)
    assert "ClustersView" in text
    assert ".pos" in text and ".neg" in text
    assert "call to compute" in text
    assert "clusters_pos.csv" in text
    assert "clusters_neg.csv" in text


def test_spec_example_6_sided_clusters_view_includes_words_hint():
    """Example 6: `pls.clusters.pos` → ClustersViewSided."""
    from ssdiff.results.continuous_result import ClustersViewSided
    from ssdiff.results.schema import Cluster
    rows = [
        Cluster(cluster_id=i, side="pos", size=8 - i,
                coherence=0.71 - 0.05 * i,
                centroid_cos_beta=0.28 - 0.02 * i, contrast=None)
        for i in range(4)
    ]
    v = ClustersViewSided(parent=None, side="pos", rows=rows,
                          words_rows=[], params={})
    text = repr(v)
    for h in ("cluster_id", "size", "coherence", "centroid_cos_beta"):
        assert h in text
    assert "Words:" in text
    assert "(0).words" in text
    assert "(1).words" in text


def test_spec_example_7_docs_view_pos_repr_includes_slice_hint():
    """`pls.docs.pos(3)` renders with slice hint."""
    pls = _shared_pls()
    text = repr(pls.docs.pos(3))
    for h in ("doc_id", "y_true", "y_hat", "residual", "alignment_score"):
        assert h in text
    assert ".pos()" in text
    assert ".neg()" in text


def _shared_group():
    from ssdiff.results.group_result import GroupResult
    from ssdiff.results.schema import Pair
    return GroupResult(
        G=2, n_kept=597, n_perm=5000, correction="holm", random_state=2137,
        omnibus_T=0.842, omnibus_p=0.0002,
        pairs=[Pair(contrast="low_high", g1="low", g2="high",
                    T=0.842, p_raw=0.0001, p_corrected=0.0002,
                    cohens_d=0.512, n_g1=297, n_g2=300, contrast_norm=1.0)],
        x=None, groups=None,
    )


def test_spec_example_8_group_result_repr_shape():
    """Example 8: `gr`."""
    gr = _shared_group()
    text = repr(gr)
    assert "GroupResult" in text
    assert "G=2" in text
    for v in ("stats", "test", "pairs"):
        assert v in text
    assert "report().save('report.md')" in text
    assert "pairs.save('pairs.csv')" in text


def test_spec_example_9_group_test_view_pairwise():
    """Example 9: `gr.test` — narrowed defaults."""
    gr = _shared_group()
    text = repr(gr.test)
    body = text.split("Save:")[0]
    for k in ("name", "pvalue", "omnibus_T"):
        assert k in body
    for k in ("omnibus_p", "G", "n_kept", "n_perm", "correction"):
        assert k not in body
    assert "pairwise:" in text
    assert "low_high" in text
    assert "Rerun:" in text
    assert "correction=" in text


def test_spec_example_10_pair_tuple_lookup_returns_pair_dataclass():
    """Example 10: ``gr.pairs[(g1, g2)]`` returns a ``Pair`` dataclass directly."""
    from ssdiff.results.schema import Pair
    gr = _shared_group()
    # _shared_group uses original labels "low"/"high"; groups=None so pairs are not relabeled.
    pair = next(iter(gr.pairs))
    looked_up = gr.pairs[(pair.g1, pair.g2)]
    assert isinstance(looked_up, Pair)
    assert looked_up.contrast == pair.contrast
    # Reverse-order lookup raises KeyError (no sign-flip).
    with pytest.raises(KeyError):
        _ = gr.pairs[(pair.g2, pair.g1)]


def test_spec_example_11_lexicon_result_repr_shape():
    """Example 11: `lex`."""
    from ssdiff.results.lexicon_result import LexiconResult
    lex = LexiconResult(var_type="continuous", n_docs=541, n_tokens=15,
                        suggestions=[], summary=None)
    text = repr(lex)
    assert "LexiconResult" in text
    assert "n_tokens=15" in text or "n_tokens=0" in text
    # Stats says n_tokens=15 (the 'declared' count); suggestions is empty here.
    for v in ("stats", "suggestions", "tokens", "summary"):
        assert v in text
    assert "report().save('lexicon.md')" in text
    assert "suggestions.save('suggestions.csv')" in text


def test_spec_example_12_pls_report_repr_shape():
    """Example 12: `pls.report()`."""
    pls = _shared_pls()
    rep = pls.report()
    text = repr(rep)
    # Title
    assert "PLSResult" in text
    # Stats section
    assert "Stats" in text
    # Citation (full path)
    assert "Plisiecki" in text
    # Save hint — dispatch is by extension
    assert "Save:" in text
    assert ".save('report.md')" in text
    assert "md txt html tex docx json" in text
