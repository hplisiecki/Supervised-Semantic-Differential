"""Tests for result objects returned by SSD.fit_pls() and SSD.fit_ols()."""

import numpy as np
import pytest

from ssdiff.results.report import Report


class TestPLSResultAttributes:
    def test_has_fit_stats(self, pls_result):
        import math
        assert isinstance(pls_result.stats.r2, float)
        assert 0.0 <= pls_result.stats.r2 <= 1.0
        assert math.isfinite(pls_result.stats.r2)
        assert isinstance(pls_result.stats.pvalue, float)
        assert 0.0 <= pls_result.stats.pvalue <= 1.0

    def test_has_beta(self, pls_result):
        assert pls_result.beta.ndim == 1
        assert pls_result.gradient.ndim == 1

    def test_has_pls_specific(self, pls_result):
        assert isinstance(pls_result.fit_info.n_components, int)
        assert pls_result.fit_info.n_components >= 1
        assert pls_result.perm_null is not None
        assert pls_result.perm_null.shape == (pls_result.fit_info.n_perm,)

    def test_has_doc_info(self, pls_result):
        assert pls_result.stats.n_kept > 0
        assert pls_result.stats.n_kept + pls_result.stats.n_dropped == pls_result.stats.n_raw


class TestPCAOLSResultAttributes:
    def test_has_fit_stats(self, pcaols_result):
        import math
        assert isinstance(pcaols_result.stats.r2, float)
        assert 0.0 <= pcaols_result.stats.r2 <= 1.0
        assert math.isfinite(pcaols_result.stats.r2)
        assert 0.0 <= pcaols_result.stats.pvalue <= 1.0

    def test_has_pcaols_specific(self, pcaols_result):
        # sweep_result is None when fixed_k is explicit
        assert hasattr(pcaols_result, "sweep_result")

    def test_no_pls_diagnostics(self, pcaols_result):
        assert not hasattr(pcaols_result, "perm_null")


class TestResultInterpretation:
    """Both result types share words + docs views."""

    def test_words_view(self, pls_result):
        words = list(pls_result.words)
        assert len(words) > 0
        w = words[0]
        assert w.side in ("pos", "neg")
        assert isinstance(w.word, str) and len(w.word) > 0
        assert isinstance(w.cos_beta, float)
        sides = {w.side for w in words}
        assert sides == {"pos", "neg"}
        # cos_beta is cosine with a unit gradient → must be in [-1, 1]
        assert all(-1.0 - 1e-9 <= x.cos_beta <= 1.0 + 1e-9 for x in words)
        # Pos words have positive cos_beta; neg words have negative cos_beta.
        for x in words:
            if x.side == "pos":
                assert x.cos_beta >= 0, x
            else:
                assert x.cos_beta <= 0, x
        # Within each side, words must be sorted by |cos_beta| descending.
        pos = [x.cos_beta for x in words if x.side == "pos"]
        neg = [abs(x.cos_beta) for x in words if x.side == "neg"]
        assert pos == sorted(pos, reverse=True), "pos words not sorted desc"
        assert neg == sorted(neg, reverse=True), "neg words not sorted desc"

    def test_words_sliced_by_side(self, pls_result):
        """Top-N words per side via list comprehension."""
        pos_words = [w for w in pls_result.words if w.side == "pos"][:5]
        neg_words = [w for w in pls_result.words if w.side == "neg"][:5]
        assert len(pos_words) == 5
        assert len(neg_words) == 5

    def test_docs_view(self, pls_result):
        docs = list(pls_result.docs)
        n = pls_result.stats.n_kept
        assert len(docs) == n
        d = docs[0]
        assert hasattr(d, "doc_id")
        assert hasattr(d, "y_true")
        assert hasattr(d, "y_hat")
        assert hasattr(d, "alignment_score")


class TestResultRepr:
    def test_pls_repr(self, pls_result):
        r = repr(pls_result)
        assert "PLSResult" in r
        assert "r²" in r

    def test_pcaols_repr(self, pcaols_result):
        r = repr(pcaols_result)
        assert "PCAOLSResult" in r
        assert "r²" in r


class TestEffectSizes:
    def test_y_mean_y_std(self, pls_result):
        assert isinstance(pls_result.stats.y_mean, float)
        assert isinstance(pls_result.stats.y_std, float)
        assert pls_result.stats.y_std > 0

    def test_alignment_score_on_docs(self, pls_result):
        """alignment_score is per-doc, accessed via docs view."""
        scores = np.array([d.alignment_score for d in pls_result.docs])
        assert scores.shape == (pls_result.stats.n_kept,)
        assert np.all(scores >= -1.0 - 1e-10)
        assert np.all(scores <= 1.0 + 1e-10)

    def test_y_corr_pred(self, pls_result):
        assert 0 <= pls_result.stats.y_corr_pred <= 1

    def test_delta(self, pls_result):
        assert isinstance(pls_result.stats.delta, float)
        expected = 0.10 * pls_result.stats.beta_norm * pls_result.stats.y_std
        assert abs(pls_result.stats.delta - expected) < 1e-12

    def test_iqr_effect(self, pls_result):
        assert isinstance(pls_result.stats.iqr_effect, float)
        assert pls_result.stats.iqr_effect >= 0

    def test_pcaols_has_effect_sizes(self, pcaols_result):
        assert hasattr(pcaols_result.stats, "delta")
        assert hasattr(pcaols_result.stats, "iqr_effect")
        assert hasattr(pcaols_result.stats, "y_corr_pred")


class TestExtremeDocs:
    """docs.pos(k) / docs.neg(k) expose the β-aligned ends of y_hat."""

    def test_pos_docs_are_doc_objects(self, pls_result):
        pos = list(pls_result.docs.pos(2))
        assert len(pos) == 2
        assert all(hasattr(d, "doc_id") for d in pos)

    def test_pos_docs_sorted_descending(self, pls_result):
        pos = list(pls_result.docs.pos(3))
        yhats = [d.y_hat for d in pos]
        assert yhats == sorted(yhats, reverse=True)

    def test_neg_docs_sorted_ascending(self, pls_result):
        neg = list(pls_result.docs.neg(3))
        yhats = [d.y_hat for d in neg]
        assert yhats == sorted(yhats)

    def test_k_clamped_to_n_docs(self, pls_result):
        pos = list(pls_result.docs.pos(9999))
        assert len(pos) <= pls_result.stats.n_kept

    def test_no_overlap_pos_neg(self, pls_result):
        pos = {d.doc_id for d in pls_result.docs.pos(2)}
        neg = {d.doc_id for d in pls_result.docs.neg(2)}
        # With n=8 docs and k=2, they should not overlap
        assert len(pos) == 2
        assert len(neg) == 2

    def test_k_zero_returns_empty(self, pls_result):
        pos = list(pls_result.docs.pos(0))
        assert pos == []


class TestFitInfo:
    """PLS/PCAOLSResult expose FitInfoView with all hyperparams."""

    def test_pls_fit_info_n_components(self, pls_result):
        assert pls_result.fit_info.n_components == 2

    def test_pls_fit_info_n_perm(self, pls_result):
        assert pls_result.fit_info.n_perm == 50

    def test_pls_fit_info_n_splits(self, pls_result):
        assert pls_result.fit_info.n_splits == 50  # default

    def test_pls_fit_info_split_ratio(self, pls_result):
        assert pls_result.fit_info.split_ratio == 0.5  # default

    def test_pls_fit_info_random_state(self, pls_result):
        assert pls_result.fit_info.random_state == 42

    def test_pcaols_fit_info_k_params_when_explicit(self, pcaols_result):
        """When fixed_k is explicit, k_min/k_max/k_step are None."""
        assert pcaols_result.fit_info.k_min is None
        assert pcaols_result.fit_info.k_max is None
        assert pcaols_result.fit_info.k_step is None

    def test_pcaols_fit_info_k_params_when_sweep(self, pcaols_result_sweep):
        """When sweep is used, k_min/k_max/k_step are stored."""
        assert pcaols_result_sweep.fit_info.k_min == 2
        assert pcaols_result_sweep.fit_info.k_max == 6
        assert pcaols_result_sweep.fit_info.k_step == 1


class TestR2AdjPlacement:
    def test_pls_stats_has_no_r2_adj(self, pls_result):
        """PLS stats must not expose r2_adj — adjusted R² is OLS-only."""
        with pytest.raises(AttributeError):
            _ = pls_result.stats.r2_adj
        assert "r2_adj" not in pls_result.stats.columns

    def test_pcaols_has_r2_adj(self, pcaols_result):
        """PCAOLSResult should still have r2_adj in stats."""
        assert pcaols_result.stats.r2_adj is not None
        assert isinstance(pcaols_result.stats.r2_adj, float)


class TestReport:
    """report() method on regression results returns a Report object."""

    def test_pls_report_returns_report(self, pls_result):
        report = pls_result.report()
        assert isinstance(report, Report)

    def test_pls_report_to_text_contains_pls(self, pls_result):
        text = pls_result.report().to_text()
        assert "PLS" in text

    def test_pls_report_to_text_contains_top_words_section(self, pls_result):
        text = pls_result.report().to_text()
        assert "Top words" in text

    def test_pcaols_report_to_text_contains_pca(self, pcaols_result):
        text = pcaols_result.report().to_text()
        assert "PCA" in text or "PCAOLSResult" in text

    def test_pcaols_report_has_top_words(self, pcaols_result):
        text = pcaols_result.report().to_text()
        assert "Top words" in text

    def test_top_words_none_skips_section(self, pls_result):
        text = pls_result.report(top_words=None).to_text()
        assert "Top words" not in text

    def test_top_words_custom_n(self, pls_result):
        text = pls_result.report(top_words=3).to_text()
        assert "n=3" in text

    def test_clusters_section(self, pls_result):
        text = pls_result.report(clusters=10).to_text()
        assert "Clusters" in text
        assert "coherence" in text

    def test_extreme_docs_section(self, pls_result):
        text = pls_result.report(extreme_docs=2).to_text()
        assert "Docs" in text

    def test_stats_always_present(self, pls_result):
        text = pls_result.report(top_words=None).to_text()
        assert "r²" in text or "r2" in text.lower()
        assert "n_kept" in text

    def test_report_citation_always_present(self, pls_result):
        """Every report ends with the APA citation."""
        text = pls_result.report(top_words=None).to_text()
        assert "Plisiecki" in text

    def test_report_not_auto_printed(self, pls_result, capsys):
        """report() does NOT auto-print to stdout; use .to_text() explicitly."""
        _ = pls_result.report()
        captured = capsys.readouterr()
        assert captured.out == ""


class TestPlotSweep:
    """PCAOLSResult.plot_sweep() smoke tests."""

    def test_returns_png_bytes_with_path(self, pcaols_result_sweep, tmp_path):
        path = tmp_path / "sweep.png"
        png = pcaols_result_sweep.plot_sweep(path=str(path))
        assert isinstance(png, bytes)
        assert png[:4] == b"\x89PNG"
        assert path.exists()
        assert path.read_bytes() == png

    def test_returns_png_bytes_without_path(self, pcaols_result_sweep, monkeypatch):
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda: None)
        png = pcaols_result_sweep.plot_sweep()
        assert isinstance(png, bytes)
        assert png[:4] == b"\x89PNG"

    def test_no_sweep_result_raises(self, pcaols_result):
        assert pcaols_result.sweep_result is None
        with pytest.raises(RuntimeError, match="No sweep data"):
            pcaols_result.plot_sweep()


class TestPlotSweepMatplotlibOptional:
    def test_raises_import_error_without_matplotlib(self, pcaols_result_sweep, monkeypatch):
        """plot_sweep() raises ImportError when matplotlib is not installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "matplotlib.pyplot" or name == "matplotlib":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="matplotlib is required"):
            pcaols_result_sweep.plot_sweep()


class TestGroupResultHyperparams:
    def test_stores_random_state(self, group_result_2g):
        assert group_result_2g.random_state == 42

    def test_stores_random_state_3g(self, group_result_3g):
        assert group_result_3g.random_state == 42


class TestGroupResultStats:
    """GroupResult exposes G, n_kept, pairs, omnibus via new views."""

    def test_group_result_g(self, group_result_2g):
        assert group_result_2g.G == 2

    def test_group_result_n_kept(self, group_result_2g):
        assert group_result_2g.n_kept > 0

    def test_group_omnibus(self, group_result_2g):
        assert np.isfinite(group_result_2g.test.omnibus_T)
        assert 0 <= group_result_2g.test.omnibus_p <= 1

    def test_pairs_count_2g(self, group_result_2g):
        assert len(group_result_2g.pairs) == 1

    def test_pairs_count_3g(self, group_result_3g):
        assert len(group_result_3g.pairs) == 3  # C(3,2)=3

    def test_pair_access(self, group_result_2g):
        pair = list(group_result_2g.pairs)[0]
        assert np.isfinite(pair.T)
        assert 0 <= pair.p_raw <= 1
        assert 0 <= pair.p_corrected <= 1
        assert np.isfinite(pair.cohens_d)

    def test_correction_default_holm(self, group_result_2g):
        assert group_result_2g.correction == "holm"
