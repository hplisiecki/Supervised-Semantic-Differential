"""Tests for ssdiff.ssd — SSD class and method-based API."""

import numpy as np
import pytest

from ssdiff.corpus import Corpus
from ssdiff.results import PCAOLSResult, PLSResult
from ssdiff.ssd import SSD


class TestSSDConstructor:
    """SSD.__init__ builds doc vectors without fitting."""

    def test_creates_doc_vectors(self, tiny_kv, sample_docs, sample_y, lexicon):
        corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, sample_y, lexicon)
        assert ssd.n_kept > 0
        assert ssd.n_kept + ssd.n_dropped == ssd.n_raw
        assert ssd.x.ndim == 2
        assert ssd.x.shape[0] == ssd.n_kept

    def test_no_fit_attributes(self, tiny_kv, sample_docs, sample_y, lexicon):
        corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, sample_y, lexicon)
        assert not hasattr(ssd, "r2")
        assert not hasattr(ssd, "beta")

    def test_no_ys_on_init(self, tiny_kv, sample_docs, sample_y, lexicon):
        """y standardization is deferred — ys should not exist after init."""
        corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, sample_y, lexicon)
        assert not hasattr(ssd, "ys")

    def test_nan_y_filtered(self, tiny_kv, sample_docs, sample_y_with_nan, lexicon):
        corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, sample_y_with_nan, lexicon)
        assert ssd.n_raw < len(sample_y_with_nan)

    def test_repr(self, tiny_kv, sample_docs, sample_y, lexicon):
        corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, sample_y, lexicon)
        r = repr(ssd)
        assert "n_kept=" in r


class TestSSDPLS:
    """SSD.fit_pls() returns a PLSResult."""

    def test_returns_pls_result(self, ssd_instance):
        result = ssd_instance.fit_pls(n_components=2, p_method="perm", n_perm=50, random_state=42)
        assert isinstance(result, PLSResult)

    def test_fit_stats(self, pls_result):
        assert 0 <= pls_result.r2 <= 1
        assert np.isfinite(pls_result.pvalue)
        assert 0 <= pls_result.pvalue <= 1
        assert pls_result.beta.ndim == 1
        assert pls_result.beta.shape[0] > 0  # Not empty
        assert pls_result.beta_unit.ndim == 1
        # beta_unit should be a unit vector
        assert np.linalg.norm(pls_result.beta_unit) == pytest.approx(1.0, abs=1e-6)
        # r2_adj is None for PLS (not meaningful)
        assert pls_result.r2_adj is None

    def test_pls_specific(self, pls_result):
        assert pls_result.n_components == 2
        assert pls_result.perm_null is not None
        assert pls_result.perm_null.shape == (50,)

    def test_top_words(self, pls_result):
        words = pls_result.top_words(n=5)
        assert isinstance(words, list)
        assert len(words) > 0
        sides = {w["side"] for w in words}
        assert sides == {"pos", "neg"}

    def test_doc_scores(self, pls_result):
        scores = pls_result.doc_scores()
        assert scores["cos_align"].shape[0] == pls_result.n_kept

    def test_no_pmethod_gives_nan_pvalue(self, ssd_instance):
        result = ssd_instance.fit_pls(n_components=2, p_method=None)
        assert result.perm_null is None
        assert np.isnan(result.pvalue)

    def test_auto_components(self, ssd_instance):
        result = ssd_instance.fit_pls(n_components=None, p_method=None)
        assert result.n_components >= 1
        assert result.cv_result is not None

    def test_result_type(self, pls_result):
        assert pls_result.result_type == "pls"

    def test_repr(self, pls_result):
        r = repr(pls_result)
        assert "PLS" in r
        assert "r2=" in r


class TestSSDPCAOLS:
    """SSD.fit_ols() returns a PCAOLSResult."""

    def test_returns_pcaols_result(self, ssd_instance):
        result = ssd_instance.fit_ols(n_components=3)
        assert isinstance(result, PCAOLSResult)
        # Should have actual fit stats
        assert 0 <= result.r2 <= 1
        assert result.beta.ndim == 1
        assert result.beta.shape[0] > 0
        assert np.isfinite(result.pvalue)

    def test_fit_stats(self, pcaols_result):
        assert 0 <= pcaols_result.r2 <= 1
        assert np.isfinite(pcaols_result.pvalue)

    def test_top_words(self, pcaols_result):
        words = pcaols_result.top_words(n=3)
        assert len(words) > 0

    def test_result_type(self, pcaols_result):
        assert pcaols_result.result_type == "pca_ols"

    def test_repr(self, pcaols_result):
        r = repr(pcaols_result)
        assert "PCAOLSResult" in r


class TestSSDPCAOLSSweep:
    """SSD.fit_ols() with auto-sweep (n_components=None)."""

    def test_auto_sweep_returns_result(self, ssd_instance):
        result = ssd_instance.fit_ols(
            n_components=None,
            k_min=2, k_max=6, k_step=2,
            verbose=False,
        )
        assert isinstance(result, PCAOLSResult)
        assert result.sweep_result is not None
        assert result.n_components == result.sweep_result.best_k

    def test_auto_sweep_selects_valid_k(self, ssd_instance):
        result = ssd_instance.fit_ols(
            n_components=None,
            k_min=2, k_max=6, k_step=2,
            verbose=False,
        )
        assert 2 <= result.n_components <= 6

    def test_auto_sweep_has_fit_stats(self, ssd_instance):
        result = ssd_instance.fit_ols(
            n_components=None,
            k_min=2, k_max=6, k_step=2,
            verbose=False,
        )
        assert 0 <= result.r2 <= 1
        assert result.beta.ndim == 1


class TestSSDValidation:
    """Input validation."""

    def test_y_docs_length_mismatch(self, tiny_kv, sample_docs, lexicon):
        corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
        y_wrong = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="len"):
            SSD(tiny_kv, corpus, y_wrong, lexicon)


class TestSSDReuse:
    """SSD instance can be reused for multiple fits."""

    def test_both_backends(self, ssd_instance):
        pls = ssd_instance.fit_pls(n_components=2, p_method=None)
        pcaols = ssd_instance.fit_ols(n_components=3)
        assert 0 <= pls.r2 <= 1
        assert 0 <= pcaols.r2 <= 1
        assert pls is not pcaols
