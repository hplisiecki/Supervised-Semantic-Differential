"""Tests for result objects returned by SSD.fit_pls() and SSD.fit_ols()."""

import numpy as np
import pytest


class TestPLSResultAttributes:
    def test_has_fit_stats(self, pls_result):
        assert hasattr(pls_result, "r2")
        assert hasattr(pls_result, "r2_adj")
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
        assert np.all(pls_result.cos_align >= -1.01)
        assert np.all(pls_result.cos_align <= 1.01)

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

    def test_not_on_pcaols(self, pcaols_result):
        assert not hasattr(pcaols_result, "split_test")
