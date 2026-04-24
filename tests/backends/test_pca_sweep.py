"""Tests for ssdiff.backends._sweep_math utilities.

Public surface under test:
  - cosine (cosine similarity)
  - zscore_ignore_nan
  - compute_auck
  - detrend_by_variance
  - overall_interpretability

Note: pca_sweep itself requires real Embeddings objects for clustering, so
it is integration-tested elsewhere. All math primitives are pure numpy and
fully testable here.
"""

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
    def test_identical_vectors_returns_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine(v, v) == pytest.approx(1.0, abs=1e-15)

    def test_antiparallel_returns_minus_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine(v, -v) == pytest.approx(-1.0, abs=1e-15)

    def test_zero_vector_returns_nan(self):
        zero = np.array([0.0, 0.0, 0.0])
        v = np.array([1.0, 2.0, 3.0])
        assert np.isnan(cosine(zero, v))

    def test_both_zero_vectors_returns_nan(self):
        zero = np.array([0.0, 0.0])
        assert np.isnan(cosine(zero, zero))

    def test_orthogonal_returns_zero(self):
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 1.0])
        assert cosine(u, v) == pytest.approx(0.0, abs=1e-15)

    def test_known_angle(self):
        # 45-degree angle: cosine = sqrt(2)/2
        u = np.array([1.0, 0.0])
        v = np.array([1.0, 1.0])
        expected = 1.0 / np.sqrt(2.0)
        assert cosine(u, v) == pytest.approx(expected, abs=1e-14)


# ---------------------------------------------------------------------------
# zscore_ignore_nan
# ---------------------------------------------------------------------------

class TestZscoreIgnoreNan:
    def test_output_mean_zero_std_one(self):
        x = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        z = zscore_ignore_nan(x)
        assert np.nanmean(z) == pytest.approx(0.0, abs=1e-12)
        assert np.nanstd(z) == pytest.approx(1.0, abs=1e-12)

    def test_nan_positions_preserved(self):
        x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        z = zscore_ignore_nan(x)
        assert np.isnan(z[1])
        assert np.isnan(z[3])
        assert np.isfinite(z[0])
        assert np.isfinite(z[2])
        assert np.isfinite(z[4])

    def test_nan_excluded_from_stats(self):
        # NaN should not affect the z-score of finite values
        x_full = np.array([1.0, 3.0, 5.0])
        x_nan = np.array([1.0, np.nan, 3.0, 5.0])
        z_full = zscore_ignore_nan(x_full)
        z_nan = zscore_ignore_nan(x_nan)
        assert z_nan[0] == pytest.approx(z_full[0], abs=1e-12)
        assert z_nan[2] == pytest.approx(z_full[1], abs=1e-12)
        assert z_nan[3] == pytest.approx(z_full[2], abs=1e-12)

    def test_all_nan_returns_all_nan(self):
        x = np.array([np.nan, np.nan, np.nan])
        z = zscore_ignore_nan(x)
        assert np.all(np.isnan(z))

    def test_constant_input_does_not_crash(self):
        # std == 0 branch: should not raise, result is well-defined
        x = np.array([5.0, 5.0, 5.0])
        z = zscore_ignore_nan(x)
        # With s clamped to 1.0: z = (5 - 5) / 1 = 0
        assert np.allclose(z, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# compute_auck
# ---------------------------------------------------------------------------

class TestComputeAuck:
    def test_constant_ones_mean_equals_one(self):
        # Constant array — every window has mean 1.0
        z = np.array([1.0, 1.0, 1.0])
        auck = compute_auck(z, radius=3)
        assert auck[0] == pytest.approx(1.0, abs=1e-12)
        assert auck[1] == pytest.approx(1.0, abs=1e-12)
        assert auck[2] == pytest.approx(1.0, abs=1e-12)

    def test_linear_ramp_center_mean(self):
        # [1, 2, 3] with radius=3: all windows cover the full array → mean=2.0
        z = np.array([1.0, 2.0, 3.0])
        auck = compute_auck(z, radius=3)
        assert auck[2] == pytest.approx(2.0, abs=1e-12)

    def test_radius_zero_returns_self(self):
        # With radius=0 each window is just the element itself
        z = np.array([1.0, 5.0, 3.0])
        auck = compute_auck(z, radius=0)
        np.testing.assert_allclose(auck, z, atol=1e-12)

    def test_nan_in_window_excluded(self):
        z = np.array([1.0, np.nan, 3.0])
        auck = compute_auck(z, radius=1)
        # Position 1: window = [z[0], z[1], z[2]] = [1, nan, 3] → mean([1,3])=2
        assert auck[1] == pytest.approx(2.0, abs=1e-12)

    def test_all_nan_returns_all_nan(self):
        z = np.array([np.nan, np.nan, np.nan])
        auck = compute_auck(z, radius=2)
        assert np.all(np.isnan(auck))


# ---------------------------------------------------------------------------
# detrend_by_variance
# ---------------------------------------------------------------------------

class TestDetrendByVariance:
    def test_all_nan_var_returns_nan_residuals(self):
        var = np.array([np.nan, np.nan, np.nan, np.nan])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        y_hat, resid, (a, b) = detrend_by_variance(var, y)
        assert np.all(np.isnan(y_hat))
        assert np.all(np.isnan(resid))
        assert np.isnan(a)
        assert np.isnan(b)

    def test_insufficient_valid_points_returns_nan(self):
        # Only 2 valid points → function returns NaN (needs at least 3)
        var = np.array([10.0, 20.0, np.nan, np.nan])
        y = np.array([1.0, 2.0, np.nan, np.nan])
        y_hat, resid, (a, b) = detrend_by_variance(var, y)
        assert np.all(np.isnan(y_hat))

    def test_zero_slope_when_y_is_constant(self):
        # When y is constant, OLS slope b ≈ 0, residuals ≈ 0
        var = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        y_hat, resid, (a, b) = detrend_by_variance(var, y)
        assert b == pytest.approx(0.0, abs=1e-10)
        assert a == pytest.approx(5.0, abs=1e-10)
        np.testing.assert_allclose(resid, 0.0, atol=1e-10)

    def test_perfect_log_linear_fit_zero_residuals(self):
        # y = 2 + 3*log(var) → residuals should be ~0
        rng = np.random.default_rng(42)
        var = np.linspace(10.0, 100.0, 20)
        a_true, b_true = 2.0, 3.0
        y = a_true + b_true * np.log(var)
        y_hat, resid, (a, b) = detrend_by_variance(var, y)
        np.testing.assert_allclose(resid, 0.0, atol=1e-10)
        assert a == pytest.approx(a_true, abs=1e-10)
        assert b == pytest.approx(b_true, abs=1e-10)

    def test_residuals_sum_to_zero(self):
        # OLS residuals always sum to zero (intercept included)
        rng = np.random.default_rng(0)
        var = np.linspace(5.0, 50.0, 10)
        y = rng.standard_normal(10)
        _, resid, _ = detrend_by_variance(var, y)
        finite_resid = resid[np.isfinite(resid)]
        assert np.sum(finite_resid) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# overall_interpretability
# ---------------------------------------------------------------------------

class TestOverallInterpretability:
    def _make_clusters(self, sizes, coherences, cos_betas):
        return [
            {"size": s, "coherence": c, "centroid_cos_beta": cb}
            for s, c, cb in zip(sizes, coherences, cos_betas)
        ]

    def test_empty_clusters_returns_nan(self):
        result = overall_interpretability([])
        assert np.isnan(result["mean_coherence"])
        assert np.isnan(result["mean_abs_cosb"])
        assert np.isnan(result["aggregate"])
        assert result["n_clusters"] == 0
        assert result["total_size"] == 0

    def test_weighted_mean_coherence(self):
        # sizes=[10,20], coherences=[0.5, 0.8], cos_beta=1.0 for both
        # weighted mean_coherence = (10*0.5 + 20*0.8) / 30 = (5 + 16) / 30 = 0.7
        clusters = self._make_clusters([10, 20], [0.5, 0.8], [1.0, 1.0])
        result = overall_interpretability(clusters, weight_by_size=True)
        assert result["mean_coherence"] == pytest.approx(0.7, abs=1e-15)

    def test_unweighted_mean_coherence(self):
        # unweighted: (0.5 + 0.8) / 2 = 0.65
        clusters = self._make_clusters([10, 20], [0.5, 0.8], [1.0, 1.0])
        result = overall_interpretability(clusters, weight_by_size=False)
        assert result["mean_coherence"] == pytest.approx(0.65, abs=1e-15)

    def test_aggregate_is_coherence_times_abs_cosb(self):
        clusters = self._make_clusters([10, 10], [0.6, 0.6], [0.8, 0.8])
        result = overall_interpretability(clusters, weight_by_size=True)
        expected_aggregate = result["mean_coherence"] * result["mean_abs_cosb"]
        assert result["aggregate"] == pytest.approx(expected_aggregate, abs=1e-14)

    def test_centroid_cos_beta_takes_abs_value(self):
        # Negative and positive cos_beta with same magnitude → same mean_abs_cosb
        clusters_pos = self._make_clusters([10], [0.5], [0.8])
        clusters_neg = self._make_clusters([10], [0.5], [-0.8])
        result_pos = overall_interpretability(clusters_pos)
        result_neg = overall_interpretability(clusters_neg)
        assert result_pos["mean_abs_cosb"] == pytest.approx(
            result_neg["mean_abs_cosb"], abs=1e-14
        )

    def test_n_clusters_and_total_size(self):
        clusters = self._make_clusters([5, 10, 15], [0.5, 0.6, 0.7], [0.3, 0.4, 0.5])
        result = overall_interpretability(clusters)
        assert result["n_clusters"] == 3
        assert result["total_size"] == 30

    def test_missing_key_raises_runtime_error(self):
        bad_cluster = [{"size": 10, "coherence": 0.5}]  # missing centroid_cos_beta
        with pytest.raises(RuntimeError, match="missing keys"):
            overall_interpretability(bad_cluster)


# ---------------------------------------------------------------------------
# PCAKSelectionResult dataclass (basic sanity — schema tests live elsewhere)
# ---------------------------------------------------------------------------

class TestPCAKSelectionResult:
    def test_frozen_dataclass_fields(self):
        result = PCAKSelectionResult(best_k=5, df_joined=[{"PCA_K": 5}])
        assert result.best_k == 5
        assert len(result.df_joined) == 1
        with pytest.raises((TypeError, AttributeError)):
            result.best_k = 10  # type: ignore[misc]
