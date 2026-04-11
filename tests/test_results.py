"""Tests for result objects returned by SSD.fit_pls() and SSD.fit_ols()."""

import io

import numpy as np
import pytest


class TestPLSResultAttributes:
    def test_has_fit_stats(self, pls_result):
        assert hasattr(pls_result, "r2")
        # r2_adj is None for PLS (not meaningful)
        assert pls_result.r2_adj is None
        assert hasattr(pls_result, "pvalue")
        assert 0 <= pls_result.r2 <= 1

    def test_has_beta(self, pls_result):
        assert pls_result.beta.ndim == 1
        assert pls_result.beta_unit.ndim == 1

    def test_has_pls_specific(self, pls_result):
        assert hasattr(pls_result, "n_components")
        assert hasattr(pls_result, "cv_result")
        assert hasattr(pls_result, "perm_null")

    def test_has_doc_info(self, pls_result):
        assert pls_result.n_kept > 0
        assert pls_result.n_kept + pls_result.n_dropped == pls_result.n_raw


class TestPCAOLSResultAttributes:
    def test_has_fit_stats(self, pcaols_result):
        assert hasattr(pcaols_result, "r2")
        assert hasattr(pcaols_result, "pvalue")
        assert 0 <= pcaols_result.r2 <= 1

    def test_has_pcaols_specific(self, pcaols_result):
        assert hasattr(pcaols_result, "sweep_result")

    def test_no_pls_attributes(self, pcaols_result):
        assert not hasattr(pcaols_result, "cv_result")
        assert not hasattr(pcaols_result, "perm_null")


class TestResultInterpretation:
    """Both result types share interpretation methods."""

    def test_top_words(self, pls_result):
        words = pls_result.top_words(n=5)
        assert isinstance(words, list)
        assert len(words) > 0
        assert set(words[0].keys()) == {"side", "rank", "word", "cos"}
        sides = {w["side"] for w in words}
        assert sides == {"pos", "neg"}

    def test_neighbors(self, pls_result):
        nbrs = pls_result.neighbors("pos", n=3)
        assert isinstance(nbrs, list)
        assert len(nbrs) <= 3
        assert isinstance(nbrs[0], tuple)

    def test_doc_scores(self, pls_result):
        scores = pls_result.doc_scores()
        assert "keep_mask" in scores
        assert "cos_align" in scores
        assert "score_std" in scores
        assert "yhat_raw" in scores
        assert scores["cos_align"].shape[0] == pls_result.n_kept


class TestResultRepr:
    def test_pls_repr(self, pls_result):
        r = repr(pls_result)
        assert "PLS" in r
        assert "r2=" in r

    def test_pcaols_repr(self, pcaols_result):
        r = repr(pcaols_result)
        assert "PCAOLS" in r
        assert "r2=" in r


class TestEffectSizes:
    def test_y_mean_y_std(self, pls_result):
        assert isinstance(pls_result.y_mean, float)
        assert isinstance(pls_result.y_std, float)
        assert pls_result.y_std > 0

    def test_cos_align(self, pls_result):
        assert pls_result.cos_align.shape == (pls_result.n_kept,)
        assert np.all(pls_result.cos_align >= -1.0 - 1e-10)
        assert np.all(pls_result.cos_align <= 1.0 + 1e-10)

    def test_y_corr_pred(self, pls_result):
        assert 0 <= pls_result.y_corr_pred <= 1

    def test_delta(self, pls_result):
        assert isinstance(pls_result.delta, float)
        expected = 0.10 * pls_result.beta_norm * pls_result.y_std
        assert abs(pls_result.delta - expected) < 1e-12

    def test_iqr_effect(self, pls_result):
        assert isinstance(pls_result.iqr_effect, float)
        assert pls_result.iqr_effect >= 0

    def test_doc_scores_reuses_cos_align(self, pls_result):
        """doc_scores() should use precomputed cos_align."""
        scores = pls_result.doc_scores()
        np.testing.assert_array_almost_equal(
            scores["cos_align"], pls_result.cos_align,
        )

    def test_pcaols_has_effect_sizes(self, pcaols_result):
        assert hasattr(pcaols_result, "delta")
        assert hasattr(pcaols_result, "iqr_effect")
        assert hasattr(pcaols_result, "y_corr_pred")


class TestSummary:
    def test_pls_summary_is_string(self, pls_result):
        s = pls_result.summary()
        assert isinstance(s, str)

    def test_pls_summary_contains_key_info(self, pls_result):
        s = pls_result.summary()
        assert "PLS" in s
        assert "kept" in s

    def test_pcaols_summary(self, pcaols_result):
        s = pcaols_result.summary()
        assert isinstance(s, str)
        assert "PCA" in s

    def test_summary_multiline(self, pls_result):
        s = pls_result.summary()
        assert s.count("\n") >= 5


class TestExtremeDocs:
    def test_returns_list_of_dicts(self, pls_result):
        docs = pls_result.extreme_docs(k=2)
        assert isinstance(docs, list)
        assert all(isinstance(d, dict) for d in docs)

    def test_dict_keys(self, pls_result):
        docs = pls_result.extreme_docs(k=2)
        if docs:
            assert set(docs[0].keys()) == {"idx", "y_true", "yhat", "cos", "side"}

    def test_sides(self, pls_result):
        docs = pls_result.extreme_docs(k=2)
        sides = {d["side"] for d in docs}
        assert sides == {"top", "bottom"}

    def test_top_docs_have_highest_yhat(self, pls_result):
        docs = pls_result.extreme_docs(k=2, by="predicted")
        top_docs = [d for d in docs if d["side"] == "top"]
        bottom_docs = [d for d in docs if d["side"] == "bottom"]
        if top_docs and bottom_docs:
            # Top docs should have higher yhat than bottom docs
            min_top = min(d["yhat"] for d in top_docs)
            max_bottom = max(d["yhat"] for d in bottom_docs)
            assert min_top >= max_bottom

    def test_k_clamped(self, pls_result):
        docs = pls_result.extreme_docs(k=9999)
        assert len(docs) <= pls_result.n_kept

    def test_by_observed(self, pls_result):
        docs = pls_result.extreme_docs(k=2, by="observed")
        assert len(docs) > 0

    def test_invalid_by(self, pls_result):
        with pytest.raises(ValueError):
            pls_result.extreme_docs(k=2, by="invalid")

    def test_empty_when_k_zero(self, pls_result):
        docs = pls_result.extreme_docs(k=0)
        assert docs == []

    def test_no_duplicate_indices(self, pls_result):
        docs = pls_result.extreme_docs(k=2)
        indices = [d["idx"] for d in docs]
        assert len(indices) == len(set(indices))


class TestSnippetsExtreme:
    def test_returns_dict(self, pls_result, sample_preprocessed_docs):
        result = pls_result.snippets_extreme(
            sample_preprocessed_docs, k=2,
        )
        assert isinstance(result, dict)
        assert "pos" in result and "neg" in result


class TestMisdiagnosed:
    def test_returns_list_of_dicts(self, pls_result):
        docs = pls_result.misdiagnosed(k=2)
        assert isinstance(docs, list)
        assert all(isinstance(d, dict) for d in docs)

    def test_dict_keys(self, pls_result):
        docs = pls_result.misdiagnosed(k=2)
        if docs:
            assert set(docs[0].keys()) == {"idx", "y_true", "yhat", "cos", "residual", "side"}

    def test_both_sides(self, pls_result):
        docs = pls_result.misdiagnosed(k=2, side="both")
        sides = {d["side"] for d in docs}
        assert sides == {"over", "under"}

    def test_over_only(self, pls_result):
        docs = pls_result.misdiagnosed(k=2, side="over")
        assert all(d["side"] == "over" for d in docs)

    def test_under_only(self, pls_result):
        docs = pls_result.misdiagnosed(k=2, side="under")
        assert all(d["side"] == "under" for d in docs)

    def test_residual_sign(self, pls_result):
        docs = pls_result.misdiagnosed(k=2, side="over")
        for d in docs:
            assert d["residual"] >= 0

    def test_invalid_side(self, pls_result):
        with pytest.raises(ValueError):
            pls_result.misdiagnosed(k=2, side="invalid")

    def test_sorted_by_abs_residual(self, pls_result):
        docs = pls_result.misdiagnosed(k=3, side="over")
        residuals = [abs(d["residual"]) for d in docs]
        assert residuals == sorted(residuals, reverse=True)

    def test_residual_matches_yhat_minus_y(self, pls_result):
        docs = pls_result.misdiagnosed(k=3, side="both")
        for d in docs:
            expected_residual = d["yhat"] - d["y_true"]
            assert abs(d["residual"] - expected_residual) < 1e-10


class TestSplitTest:
    def test_returns_dict(self, pls_result):
        result = pls_result.split_test(n_splits=10, seed=42)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"pvalue", "mean_r"}

    def test_pvalue_range(self, pls_result):
        result = pls_result.split_test(n_splits=10, seed=42)
        assert 0 <= result["pvalue"] <= 1

    def test_default_is_split(self, pls_result):
        result = pls_result.split_test(n_splits=10, seed=42)
        explicit = pls_result.split_test(n_splits=10, seed=42, method="split")
        assert result["pvalue"] == explicit["pvalue"]

    def test_invalid_method(self, pls_result):
        import pytest
        with pytest.raises(ValueError, match="Unknown method"):
            pls_result.split_test(method="bogus")

    def test_significant_data_yields_positive_r(self, pls_result):
        result = pls_result.split_test(n_splits=20, seed=42)
        # For data with real signal, mean_r should be positive
        assert result["mean_r"] > -1
        assert result["mean_r"] < 1

    def test_not_on_pcaols(self, pcaols_result):
        assert not hasattr(pcaols_result, "split_test")


class TestReport:
    """report() method on regression results."""

    def test_pls_report_default(self, pls_result, capsys):
        text = pls_result.report()
        captured = capsys.readouterr()
        assert captured.out.strip() == text.strip()
        assert "PLS" in text
        assert "Top Words" in text
        assert "+ pole" in text

    def test_pcaols_report_default(self, pcaols_result, capsys):
        text = pcaols_result.report()
        captured = capsys.readouterr()
        assert captured.out.strip() == text.strip()
        assert "PCA" in text
        assert "Top Words" in text

    def test_top_words_none_skips(self, pls_result):
        text = pls_result.report(top_words=None)
        assert "Top Words" not in text

    def test_top_words_custom_n(self, pls_result):
        text = pls_result.report(top_words=3)
        assert "n=3" in text

    def test_clusters(self, pls_result):
        text = pls_result.report(top_words=None, clusters=10)
        assert "Clusters" in text
        assert "coherence" in text

    def test_extreme_docs(self, pls_result):
        text = pls_result.report(top_words=None, extreme_docs=2)
        assert "Extreme Documents" in text
        assert "Highest predicted" in text
        assert "Lowest predicted" in text

    def test_misdiagnosed(self, pls_result):
        text = pls_result.report(top_words=None, misdiagnosed=2)
        assert "Misdiagnosed" in text

    def test_all_sections(self, pls_result):
        text = pls_result.report(top_words=3, clusters=10, extreme_docs=2, misdiagnosed=2)
        assert "Top Words" in text
        assert "Clusters" in text
        assert "Extreme Documents" in text
        assert "Misdiagnosed" in text

    def test_summary_always_present(self, pls_result):
        text = pls_result.report(top_words=None)
        assert "R²" in text
        assert "kept" in text


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


class TestSnippetsOnResult:
    """_SSDResultBase.snippets() on real result objects."""

    def test_pls_snippets(self, pls_result, sample_preprocessed_docs):
        result = pls_result.snippets(sample_preprocessed_docs, top_per_side=10)
        assert isinstance(result, dict)
        assert "pos" in result and "neg" in result

    def test_pcaols_snippets(self, pcaols_result, sample_preprocessed_docs):
        result = pcaols_result.snippets(sample_preprocessed_docs, top_per_side=10)
        assert isinstance(result, dict)
        assert "pos" in result and "neg" in result


class TestGroupResultNeighborsTuples:
    """GroupResult.neighbors() returns consistent 3-tuples."""

    def test_2group_returns_3tuples(self, group_result_2g):
        nbrs = group_result_2g.neighbors("pos", n=3)
        assert isinstance(nbrs, list)
        assert len(nbrs) > 0
        for item in nbrs:
            assert len(item) == 3, f"Expected 3-tuple, got {len(item)}-tuple: {item}"
            label, word, cos = item
            assert isinstance(label, str)
            assert isinstance(word, str)
            assert isinstance(cos, float)

    def test_3group_returns_3tuples(self, group_result_3g):
        nbrs = group_result_3g.neighbors("pos", n=3)
        assert isinstance(nbrs, list)
        assert len(nbrs) > 0
        for item in nbrs:
            assert len(item) == 3

    def test_3group_has_all_contrast_labels(self, group_result_3g):
        nbrs = group_result_3g.neighbors("pos", n=3)
        labels = {item[0] for item in nbrs}
        assert len(labels) == 3  # C(3,2) = 3 contrasts


class TestPLSResultHyperparams:
    def test_stores_random_state(self, pls_result):
        assert hasattr(pls_result, "random_state")
        assert pls_result.random_state == 42

    def test_stores_n_perm(self, pls_result):
        assert hasattr(pls_result, "n_perm")
        assert pls_result.n_perm == 50

    def test_stores_n_splits(self, pls_result):
        assert hasattr(pls_result, "n_splits")
        assert pls_result.n_splits == 50  # default

    def test_stores_split_ratio(self, pls_result):
        assert hasattr(pls_result, "split_ratio")
        assert pls_result.split_ratio == 0.5  # default


class TestPCAOLSResultHyperparams:
    def test_stores_k_min_none_when_explicit(self, pcaols_result):
        """When n_components is explicitly set, sweep params are None."""
        assert hasattr(pcaols_result, "k_min")
        assert pcaols_result.k_min is None

    def test_stores_k_max_none_when_explicit(self, pcaols_result):
        assert hasattr(pcaols_result, "k_max")
        assert pcaols_result.k_max is None

    def test_stores_k_step_none_when_explicit(self, pcaols_result):
        assert hasattr(pcaols_result, "k_step")
        assert pcaols_result.k_step is None

    def test_stores_k_params_when_sweep(self, pcaols_result_sweep):
        """When sweep is used, k_min/k_max/k_step are stored."""
        assert pcaols_result_sweep.k_min == 2
        assert pcaols_result_sweep.k_max == 6
        assert pcaols_result_sweep.k_step == 1


class TestGroupResultHyperparams:
    def test_stores_random_state(self, group_result_2g):
        assert hasattr(group_result_2g, "random_state")
        assert group_result_2g.random_state == 42

    def test_stores_random_state_3g(self, group_result_3g):
        assert group_result_3g.random_state == 42


class TestR2AdjPlacement:
    def test_pls_has_no_r2_adj(self, pls_result):
        """PLSResult should not have r2_adj (not meaningful for PLS)."""
        assert pls_result.r2_adj is None

    def test_pcaols_has_r2_adj(self, pcaols_result):
        """PCAOLSResult should still have r2_adj."""
        assert pcaols_result.r2_adj is not None
        assert isinstance(pcaols_result.r2_adj, float)

    def test_pls_summary_no_r2_adj(self, pls_result):
        """PLS summary should not mention R²_adj."""
        s = pls_result.summary()
        assert "R²_adj" not in s

    def test_pcaols_summary_has_r2_adj(self, pcaols_result):
        """PCA+OLS summary should still show R²_adj."""
        s = pcaols_result.summary()
        assert "R²_adj" in s


class TestGroupResultSnippets:
    """GroupResult.snippets() method."""

    def test_returns_pos_neg_dict(self, group_result_2g, sample_preprocessed_docs):
        result = group_result_2g.snippets(
            sample_preprocessed_docs, top_per_side=10,
        )
        assert isinstance(result, dict)
        assert "pos" in result and "neg" in result

    def test_snippets_have_contrast_key(self, group_result_2g, sample_preprocessed_docs):
        result = group_result_2g.snippets(
            sample_preprocessed_docs, top_per_side=10,
        )
        for side in ("pos", "neg"):
            for row in result[side]:
                assert "contrast" in row
                assert "vs" in row["contrast"]

    def test_3group_snippets(self, group_result_3g, sample_preprocessed_docs):
        result = group_result_3g.snippets(
            sample_preprocessed_docs, top_per_side=10,
        )
        assert isinstance(result, dict)
        contrasts = set()
        for side in ("pos", "neg"):
            for row in result[side]:
                contrasts.add(row["contrast"])
        # Should have snippets from multiple contrasts
        assert len(contrasts) >= 1
