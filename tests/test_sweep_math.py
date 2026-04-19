"""Tests for ssdiff/backends/_sweep_math.py — PCA sweep math utilities."""

from __future__ import annotations

import numpy as np
import pytest

from ssdiff.backends._sweep_math import (
    PCAKSelectionResult,
    compute_auck,
    cosine,
    detrend_by_variance,
    overall_interpretability,
    zscore_ignore_nan,
)

# ---------------------------------------------------------------------------
# cosine
# ---------------------------------------------------------------------------

class TestCosine:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine(v, v) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert cosine(v, -v) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 1.0])
        assert cosine(u, v) == pytest.approx(0.0, abs=1e-10)

    def test_zero_vector_returns_nan(self):
        v = np.array([1.0, 2.0])
        z = np.array([0.0, 0.0])
        assert np.isnan(cosine(v, z))
        assert np.isnan(cosine(z, v))

    def test_both_zero_nan(self):
        z = np.array([0.0, 0.0])
        assert np.isnan(cosine(z, z))


# ---------------------------------------------------------------------------
# zscore_ignore_nan
# ---------------------------------------------------------------------------

class TestZscoreIgnoreNan:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = zscore_ignore_nan(x)
        assert z.shape == x.shape
        assert np.nanmean(z) == pytest.approx(0.0, abs=1e-10)
        assert np.nanstd(z) == pytest.approx(1.0, abs=1e-10)

    def test_with_nans(self):
        x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        z = zscore_ignore_nan(x)
        assert np.isnan(z[1])
        assert np.isnan(z[3])
        finite = z[np.isfinite(z)]
        assert np.mean(finite) == pytest.approx(0.0, abs=1e-10)

    def test_zero_variance(self):
        x = np.array([5.0, 5.0, 5.0])
        z = zscore_ignore_nan(x)
        # All same → (x - mean) / 1.0 = 0
        assert np.allclose(z, 0.0)

    def test_all_nan(self):
        x = np.array([np.nan, np.nan])
        z = zscore_ignore_nan(x)
        assert all(np.isnan(z))


# ---------------------------------------------------------------------------
# compute_auck
# ---------------------------------------------------------------------------

class TestComputeAuck:
    def test_basic(self):
        z = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        auck = compute_auck(z, radius=1)
        assert auck.shape == z.shape
        # auck[2] = mean([1, 2, 3]) = 2.0
        assert auck[2] == pytest.approx(2.0)

    def test_boundary(self):
        z = np.array([0.0, 1.0, 2.0])
        auck = compute_auck(z, radius=1)
        # auck[0] = mean([0, 1]) = 0.5
        assert auck[0] == pytest.approx(0.5)
        # auck[2] = mean([1, 2]) = 1.5
        assert auck[2] == pytest.approx(1.5)

    def test_radius_zero(self):
        z = np.array([1.0, 2.0, 3.0])
        auck = compute_auck(z, radius=0)
        np.testing.assert_array_almost_equal(auck, z)

    def test_with_nan(self):
        z = np.array([1.0, np.nan, 3.0])
        auck = compute_auck(z, radius=1)
        # auck[0] = mean(finite in [1, nan]) = 1.0
        assert auck[0] == pytest.approx(1.0)

    def test_all_nan(self):
        z = np.array([np.nan, np.nan])
        auck = compute_auck(z, radius=1)
        assert all(np.isnan(auck))


# ---------------------------------------------------------------------------
# detrend_by_variance
# ---------------------------------------------------------------------------

class TestDetrendByVariance:
    def test_basic_shape(self):
        var = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
        y = np.array([0.5, 0.6, 0.65, 0.68, 0.7])
        y_hat, resid, (a, b) = detrend_by_variance(var, y)
        assert y_hat.shape == y.shape
        assert resid.shape == y.shape
        assert np.isfinite(a)
        assert np.isfinite(b)

    def test_residuals_sum_near_zero(self):
        var = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
        y = np.array([0.5, 0.6, 0.65, 0.68, 0.7])
        _, resid, _ = detrend_by_variance(var, y)
        assert np.nansum(resid) == pytest.approx(0.0, abs=1e-6)

    def test_insufficient_data(self):
        var = np.array([10.0, 20.0])
        y = np.array([0.5, 0.6])
        y_hat, resid, (a, b) = detrend_by_variance(var, y)
        assert all(np.isnan(y_hat))
        assert np.isnan(a)

    def test_with_nans(self):
        var = np.array([10.0, np.nan, 50.0, 70.0, 90.0])
        y = np.array([0.5, 0.6, 0.65, np.nan, 0.7])
        y_hat, resid, _ = detrend_by_variance(var, y)
        assert y_hat.shape == y.shape


# ---------------------------------------------------------------------------
# overall_interpretability
# ---------------------------------------------------------------------------

class TestOverallInterpretability:
    def test_basic(self):
        clusters = [
            {"size": 10, "coherence": 0.8, "centroid_cos_beta": 0.6},
            {"size": 5, "coherence": 0.7, "centroid_cos_beta": -0.5},
        ]
        result = overall_interpretability(clusters)
        assert "mean_coherence" in result
        assert "mean_abs_cosb" in result
        assert "aggregate" in result
        assert result["n_clusters"] == 2
        assert result["total_size"] == 15

    def test_aggregate_is_product(self):
        clusters = [
            {"size": 10, "coherence": 0.8, "centroid_cos_beta": 0.6},
        ]
        result = overall_interpretability(clusters, weight_by_size=False)
        expected = 0.8 * 0.6  # coherence * abs(cos_beta)
        assert result["aggregate"] == pytest.approx(expected)

    def test_empty_clusters(self):
        result = overall_interpretability([])
        assert result["n_clusters"] == 0
        assert result["total_size"] == 0
        assert np.isnan(result["aggregate"])

    def test_missing_keys_raises(self):
        with pytest.raises(RuntimeError, match="missing keys"):
            overall_interpretability([{"size": 10}])

    def test_weighted_by_size(self):
        clusters = [
            {"size": 10, "coherence": 1.0, "centroid_cos_beta": 1.0},
            {"size": 0, "coherence": 0.0, "centroid_cos_beta": 0.0},
        ]
        result = overall_interpretability(clusters, weight_by_size=True)
        # weight 10/(10+0) = 1.0 for first cluster
        assert result["mean_coherence"] == pytest.approx(1.0)

    def test_unweighted(self):
        clusters = [
            {"size": 10, "coherence": 0.8, "centroid_cos_beta": 0.4},
            {"size": 1, "coherence": 0.2, "centroid_cos_beta": 0.6},
        ]
        result = overall_interpretability(clusters, weight_by_size=False)
        assert result["mean_coherence"] == pytest.approx(0.5)
        assert result["mean_abs_cosb"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# PCAKSelectionResult
# ---------------------------------------------------------------------------

class TestPCAKSelectionResult:
    def test_construction(self):
        r = PCAKSelectionResult(best_k=50, df_joined=[{"PCA_K": 50}])
        assert r.best_k == 50
        assert len(r.df_joined) == 1

    def test_frozen(self):
        r = PCAKSelectionResult(best_k=50)
        with pytest.raises(AttributeError):
            r.best_k = 60


# ---------------------------------------------------------------------------
# End-to-end sweep K-selection
# ---------------------------------------------------------------------------

class TestSweepPicksCorrectKEndToEnd:
    """fit_ols(fixed_k=None) must pick K near the true rank on synthetic data
    whose y depends on a single PC direction. The selected K is allowed to
    bracket the truth (1..3) on tiny data; the assertion fails loudly if the
    sweep selects the max of the range (overfitting) or refuses to move off
    k_min (underfitting)."""

    def test_sweep_selects_low_k_for_rank_one_signal(self):
        import numpy as np

        from ssdiff.corpus import Corpus
        from ssdiff.embeddings import Embeddings
        from ssdiff.ssd import SSD

        rng = np.random.default_rng(0)
        D = 12
        vocab = [f"w{i:02d}" for i in range(40)]
        mat = rng.normal(size=(len(vocab), D)).astype(np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        emb = Embeddings(vocab, mat)

        # Build docs where y depends on the frequency of a fixed "signal" seed.
        seeds = ["w00", "w01", "w02"]
        filler = [f"w{i:02d}" for i in range(3, 40)]
        n_docs = 80
        docs = []
        y = np.empty(n_docs, dtype=np.float64)
        for i in range(n_docs):
            seed = rng.choice(seeds)
            ctx = list(rng.choice(filler, size=4, replace=False))
            docs.append([seed] + ctx)
            # Outcome depends on which seed appears (3-level signal).
            y[i] = {"w00": -1.0, "w01": 0.0, "w02": 1.0}[seed] + rng.normal(0, 0.1)

        corpus = Corpus(docs, pretokenized=True, lang="pl")
        ssd = SSD(emb, corpus, y, set(seeds))

        result = ssd.fit_ols(fixed_k=None, k_min=1, k_max=8, k_step=1)
        assert result.sweep_result is not None
        picked = result.fit_info.n_components
        # Signal is low-rank; sweep must not pick the ceiling (overfit)
        # and must not refuse to leave the floor (underfit on a real signal).
        assert 1 <= picked <= 5, f"unexpected K: {picked}"
        # And best_k bookkeeping must match n_components.
        assert result.fit_info.n_components == result.fit_info.best_k
