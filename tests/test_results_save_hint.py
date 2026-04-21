"""Save-hint contents and repr/html dunder behavior."""

from __future__ import annotations

import ssdiff
from ssdiff.results.continuous_result import WordsView
from ssdiff.results.schema import Word


def _words(n: int = 3) -> list[Word]:
    return [Word(side="pos", rank=i, word=f"w{i}",
                 cos_beta=float(i) / 10, contrast=None)
            for i in range(n)]


def test_view_default_save_hint_text():
    v = WordsView(_words())
    hint = v._save_hint()
    assert hint.startswith("Save:")
    assert ".save('words.csv'," in hint
    assert ".to_df()" in hint
    # Enumerates supported extensions:
    assert "csv" in hint and "json" in hint and "xlsx" in hint


def test_view_repr_includes_table_and_hint_when_enabled():
    ssdiff.set_repr_hints(True)
    v = WordsView(_words())
    text = repr(v)
    assert "side" in text
    assert "w0" in text
    assert ".save('words.csv'," in text
    assert "cols=" in text


def test_view_repr_omits_hint_when_disabled():
    ssdiff.set_repr_hints(False)
    try:
        v = WordsView(_words())
        text = repr(v)
        assert "side" in text
        assert "Save:" not in text
    finally:
        ssdiff.set_repr_hints(True)


def test_view_repr_html_includes_table_and_hint_when_enabled():
    ssdiff.set_repr_hints(True)
    v = WordsView(_words())
    html = v._repr_html_()
    assert "<table" in html
    assert "Save:" in html


def test_view_repr_html_omits_hint_when_disabled():
    ssdiff.set_repr_hints(False)
    try:
        v = WordsView(_words())
        html = v._repr_html_()
        assert "<table" in html
        assert "Save:" not in html
    finally:
        ssdiff.set_repr_hints(True)


def test_scalar_view_to_text_kv_layout():
    from ssdiff.results.continuous_result import StatsView
    from ssdiff.results.schema import Stats
    sv = StatsView(Stats(
        backend="PLS", r2=0.30, r2_adj=0.29, pvalue=0.001,
        n_raw=597, n_kept=597, n_dropped=0,
        y_mean=5.12, y_std=1.34, beta_norm=0.821,
        delta=0.110, iqr_effect=0.083, y_corr_pred=0.549,
    ))
    text = sv.to_text()
    lines = text.splitlines()
    assert any(line.startswith("backend") for line in lines)
    backend_line = next(line for line in lines if line.startswith("backend"))
    assert "PLS" in backend_line


def test_scalar_view_default_save_hint():
    from ssdiff.results.continuous_result import StatsView
    from ssdiff.results.schema import Stats
    sv = StatsView(Stats(
        backend="PLS", r2=0.30, r2_adj=0.29, pvalue=0.001,
        n_raw=597, n_kept=597, n_dropped=0,
        y_mean=5.12, y_std=1.34, beta_norm=0.821,
        delta=0.110, iqr_effect=0.083, y_corr_pred=0.549,
    ))
    hint = sv._save_hint()
    assert hint.startswith("Save:")
    assert ".to_dict()" in hint
    assert ".to_df()" in hint
    assert "cols=" in hint


def test_scalar_view_repr_includes_kv_and_hint():
    ssdiff.set_repr_hints(True)
    from ssdiff.results.continuous_result import StatsView
    from ssdiff.results.schema import Stats
    sv = StatsView(Stats(
        backend="PLS", r2=0.30, r2_adj=0.29, pvalue=0.001,
        n_raw=597, n_kept=597, n_dropped=0,
        y_mean=5.12, y_std=1.34, beta_norm=0.821,
        delta=0.110, iqr_effect=0.083, y_corr_pred=0.549,
    ))
    text = repr(sv)
    assert "backend" in text
    assert "PLS" in text
    assert ".to_df()" in text
    assert "cols=" in text


def test_pls_test_view_save_hint_includes_rerun_line():
    from ssdiff.results.continuous_result import PLSTestView
    tv = PLSTestView(parent=None, name="split", info={
        "pvalue": 4.83e-08, "split_r2": 0.3289, "n_splits": 30,
        "split_ratio": 0.5, "random_state": None,
    })
    hint = tv._save_hint()
    assert "Save:" in hint
    assert "Rerun:" in hint
    assert ".test('perm'" in hint or ".test('split'" in hint
    assert "n_perm=" in hint
    assert "n_splits=" in hint


def test_pcaols_test_view_save_hint_includes_rerun_line():
    from ssdiff.results.continuous_result import PCAOLSTestView
    tv = PCAOLSTestView(parent=None, name="f_test", info={"pvalue": 0.01})
    hint = tv._save_hint()
    assert "Rerun:" in hint
    assert ".test('f_test')" in hint


def test_group_test_view_save_hint_includes_rerun_line():
    from ssdiff.results.group_result import GroupTestView
    tv = GroupTestView(parent=None, name="permutation", info={
        "pvalue": 0.0002, "omnibus_T": 0.842, "omnibus_p": 0.0002,
        "G": 2, "n_kept": 597, "n_perm": 5000,
        "correction": "holm", "random_state": 2137,
    })
    hint = tv._save_hint()
    assert "Rerun:" in hint
    assert "n_perm=" in hint
    assert "correction=" in hint


def test_group_test_view_to_text_appends_pairwise_section():
    from ssdiff.results.group_result import GroupTestView, PairsListView
    from ssdiff.results.schema import Pair

    class _FakeParent:
        pass
    parent = _FakeParent()
    parent.pairs = PairsListView([
        Pair(contrast="low_high", g1="low", g2="high",
             T=0.842, p_raw=0.0001, p_corrected=0.0002, cohens_d=0.512,
             n_g1=297, n_g2=300, contrast_norm=1.0),
    ])
    tv = GroupTestView(parent=parent, name="permutation", info={
        "pvalue": 0.0002, "omnibus_T": 0.842, "omnibus_p": 0.0002,
        "G": 2, "n_kept": 597, "n_perm": 5000,
        "correction": "holm", "random_state": 2137,
    })
    text = tv.to_text()
    assert "permutation" in text
    assert "omnibus_T" in text
    assert "pairwise:" in text
    assert "low_high" in text
    assert "T=" in text
    assert "p=" in text
    assert "d=" in text


def test_sided_clusters_view_save_hint_includes_words_line():
    from ssdiff.results.continuous_result import ClustersViewSided
    v = ClustersViewSided(parent=None, side="pos", rows=[],
                          words_rows=[], params={})
    hint = v._save_hint()
    assert "Save:" in hint
    # .words is now a property → ClusterWordsViewSided; .words(cluster_id) still works.
    assert ".words" in hint
    assert "ClusterWordsViewSided" in hint or "ClusterWordsView" in hint


def test_snippets_view_save_hint_includes_sides_line():
    """SnippetsView hint advertises .pos/.neg (not a broken side/cluster_id filter)."""
    from ssdiff.results.continuous_result import SnippetsView
    v = SnippetsView([])
    hint = v._save_hint()
    assert "Save:" in hint
    assert ".pos" in hint and ".neg" in hint
    # The old misleading filter line must be gone.
    assert "side='pos'" not in hint


def test_docs_view_save_hint_includes_slice_line():
    from ssdiff.results.continuous_result import DocsView
    v = DocsView([])
    hint = v._save_hint()
    assert "Save:" in hint
    assert ".pos()" in hint
    assert ".neg()" in hint
    assert ".id(" in hint
    assert ".misdiagnosed(" in hint


def test_pairs_list_view_save_hint_includes_lookup_line():
    from ssdiff.results.group_result import PairsListView
    from ssdiff.results.schema import Pair

    # Empty case: shows generic placeholder so the shape is still obvious.
    v = PairsListView([])
    hint = v._save_hint()
    assert "Save:" in hint
    assert "Lookup:" in hint
    assert "Pair" in hint
    assert "<group1>" in hint and "<group2>" in hint

    # Populated case: hint substitutes real group names so copy-paste works.
    pair = Pair(contrast="high_low", g1="high", g2="low",
                T=0.0, p_raw=1.0, p_corrected=1.0, cohens_d=0.0,
                n_g1=10, n_g2=10, contrast_norm=0.0)
    hint = PairsListView([pair])._save_hint()
    assert "view['high', 'low']" in hint


def _make_pls_result():
    """Build a PLSResult from synthetic arrays (no embeddings, no corpus)."""
    import numpy as np

    from ssdiff.results.continuous_result import PLSResult
    rng = np.random.default_rng(0)
    n, d = 30, 4
    x = rng.normal(size=(n, d))
    beta = rng.normal(size=d)
    y = x @ beta
    keep_mask = np.ones(n, dtype=bool)
    return PLSResult(
        x=x, beta=beta, keep_mask=keep_mask,
        n_raw=n, n_kept=n, n_dropped=0,
        y=y, _y_mean=np.array([y.mean()]),
        _y_scale=np.array([y.std() or 1.0]),
        r2=0.30, r2_adj=0.29, pvalue=4.83e-08,
        test_name="split", test_info={"pvalue": 4.83e-08, "split_r2": 0.33,
                                       "n_splits": 30, "split_ratio": 0.5,
                                       "random_state": None},
    )


def test_pls_result_summary_includes_r2_n_backend():
    pls = _make_pls_result()
    text = pls._summary()
    assert "PLSResult" in text
    assert "r²" in text or "r2" in text
    assert "0.30" in text or ".30" in text
    assert "n=30" in text
    assert "backend=PLS" in text


def test_pls_result_save_hint_lists_canonical_lines():
    pls = _make_pls_result()
    hint = pls._save_hint()
    assert "report().save('report.md')" in hint
    assert "words.save('words.csv')" in hint
    assert "docs.to_df()" in hint


def test_pls_result_repr_is_compact_no_citation():
    import ssdiff
    ssdiff.set_repr_hints(True)
    pls = _make_pls_result()
    out = repr(pls)
    assert "Plisiecki" not in out
    assert "PLSResult" in out
    assert "Save:" in out


def test_pls_result_access_hint_shows_views_and_methods():
    pls = _make_pls_result()
    text = pls.to_text()
    assert "views:" in text
    for name in ("stats", "fit_info", "words", "clusters",
                 "snippets", "docs", "test"):
        assert name in text
    assert "methods:" in text
    for name in ("report()", "test(", "attach("):
        assert name in text


def _make_group_result():
    from ssdiff.results.group_result import GroupResult
    from ssdiff.results.schema import Pair
    return GroupResult(
        G=2, n_kept=597, n_perm=5000, correction="holm", random_state=2137,
        omnibus_T=0.842, omnibus_p=0.0002,
        pairs=[Pair(contrast="low_high", g1="low", g2="high",
                    T=0.842, p_raw=0.0001, p_corrected=0.0002,
                    cohens_d=0.512, n_g1=297, n_g2=300, contrast_norm=1.0)],
        words_rows=[], cluster_rows=[], cluster_words_rows=[], snippets_rows=[],
        x=None, groups=None,
    )


def test_group_result_summary_includes_g_n_p():
    gr = _make_group_result()
    text = gr._summary()
    assert "GroupResult" in text
    assert "G=2" in text
    assert "n=597" in text
    assert "omnibus" in text or "p=" in text


def test_group_result_save_hint_lists_canonical_lines():
    gr = _make_group_result()
    hint = gr._save_hint()
    assert "report().save('report.md')" in hint
    assert "pairs.save('pairs.csv')" in hint


def _make_lexicon_result():
    from ssdiff.results.lexicon_result import LexiconResult
    return LexiconResult(var_type="continuous", n_docs=541, n_tokens=15,
                         suggestions=[], summary=None)


def test_lexicon_result_summary_includes_n_tokens_n_docs():
    lex = _make_lexicon_result()
    text = lex._summary()
    assert "LexiconResult" in text
    assert "n_tokens=15" in text
    assert "n_docs=541" in text


def test_lexicon_result_save_hint_lists_canonical_lines():
    lex = _make_lexicon_result()
    hint = lex._save_hint()
    assert "report().save('lexicon.md')" in hint
    assert "suggestions.save('suggestions.csv')" in hint


def test_report_repr_includes_save_hint():
    import ssdiff
    ssdiff.set_repr_hints(True)
    from ssdiff.results.report import Report, Section
    rep = Report(
        title="Test report",
        subtitle=None,
        sections=[Section(title="S", kind="kv", rows=[("k", "v")])],
    )
    text = repr(rep)
    assert "Plisiecki" in text
    assert "Save:" in text
    assert ".save('report.md')" in text


def test_report_repr_omits_save_hint_when_disabled():
    import ssdiff
    ssdiff.set_repr_hints(False)
    try:
        from ssdiff.results.report import Report, Section
        rep = Report(
            title="Test report",
            subtitle=None,
            sections=[Section(title="S", kind="kv", rows=[("k", "v")])],
        )
        text = repr(rep)
        assert "Plisiecki" in text
        assert "Save:" not in text
    finally:
        ssdiff.set_repr_hints(True)


def test_report_to_text_does_not_include_save_hint():
    """to_text is the renderer for save('out.txt') — must be hint-free."""
    from ssdiff.results.report import Report, Section
    rep = Report(
        title="Test report",
        subtitle=None,
        sections=[Section(title="S", kind="kv", rows=[("k", "v")])],
    )
    text = rep.to_text()
    assert "Plisiecki" in text
    assert "Save:" not in text
