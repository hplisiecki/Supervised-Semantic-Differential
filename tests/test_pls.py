"""Tests for ssdiff.backends.pls — PLS1 NIPALS backend."""

import numpy as np
import pytest

from ssdiff.backends.pls import pls1_cv_select, pls1_fit, pls1_permutation_test


class TestPLS1Fit:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 5))
        y = X @ rng.normal(size=5) + rng.normal(size=30) * 0.1
        T, P, W, Q, coef = pls1_fit(X, y, 3)
        assert T.shape == (30, 3)
        assert P.shape == (5, 3)
        assert W.shape == (5, 3)
        assert Q.shape == (3,)
        assert coef.shape == (5,)

        # Verify prediction quality: PLS coefficients should reconstruct well
        y_pred = X @ coef
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.8  # Good fit for low-noise data

    def test_prediction_quality(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 5))
        beta_true = np.array([1.0, -0.5, 0.3, 0.0, 0.8])
        y = X @ beta_true
        _, _, _, _, coef = pls1_fit(X, y, 5)
        y_pred = X @ coef
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
        assert r2 > 0.99  # Near-perfect fit for noiseless data

    def test_truncation(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(10, 5))
        y = rng.normal(size=10)
        # Request more components than possible
        T, P, W, Q, coef = pls1_fit(X, y, 20)
        assert W.shape[1] <= min(9, 5)  # Truncated to min(n-1, D)


class TestPLS1CVSelect:
    def test_returns_result(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(40, 5))
        y = X @ rng.normal(size=5) + rng.normal(size=40) * 0.5
        result = pls1_cv_select(X, y, max_components=5, n_folds=5, seed=42)
        assert 1 <= result.best_n_components <= 5
        assert isinstance(result.cv_scores, dict)
        assert isinstance(result.cv_scores_se, dict)
        assert np.isfinite(result.best_cv_r2)

class TestPermutationTest:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 5))
        y = X @ rng.normal(size=5) + rng.normal(size=30) * 0.1
        p, cv_r2_obs, null = pls1_permutation_test(
            X, y, n_components=2, n_perm=50, seed=42,
        )
        assert 0 <= p <= 1
        assert isinstance(cv_r2_obs, float)
        assert null.shape == (50,)

        # Null distribution should have real variation
        assert np.std(null) > 0
        # p-value should be computed as (b+1)/(m+1) formula
        expected_p = (np.sum(null >= cv_r2_obs) + 1) / (50 + 1)
        assert p == pytest.approx(expected_p, abs=1e-10)

    def test_random_data_high_pvalue(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 5))
        y = rng.normal(size=30)  # No real signal
        p, _, _ = pls1_permutation_test(X, y, n_components=2, n_perm=99, seed=42)
        assert p > 0.05  # Should not be significant


class TestPLS1SplitTest:
    def test_basic(self):
        from ssdiff.backends.pls import pls1_split_test
        rng = np.random.default_rng(42)
        n, D = 60, 10
        X = rng.normal(size=(n, D))
        beta_true = rng.normal(size=D)
        y = X @ beta_true + rng.normal(size=n) * 0.5

        p_split, mean_r = pls1_split_test(
            X, y, n_components=2, n_splits=20, seed=42,
        )
        assert 0 <= p_split <= 1
        assert -1 <= mean_r <= 1
        assert p_split < 0.1

    def test_null(self):
        from ssdiff.backends.pls import pls1_split_test
        rng = np.random.default_rng(99)
        n, D = 60, 10
        X = rng.normal(size=(n, D))
        y = rng.normal(size=n)

        p_split, _ = pls1_split_test(
            X, y, n_components=2, n_splits=20, seed=99,
        )
        assert 0 <= p_split <= 1
        assert p_split > 0.01


class TestSplitTestCalibrated:
    def test_basic(self):
        from ssdiff.backends.pls import pls1_split_test_calibrated
        rng = np.random.default_rng(42)
        n, D = 60, 10
        X = rng.normal(size=(n, D))
        beta_true = rng.normal(size=D)
        y = X @ beta_true + rng.normal(size=n) * 0.5

        p_cal, mean_r = pls1_split_test_calibrated(
            X, y, n_components=2, n_splits=20, n_perm=50, seed=42,
        )
        assert 0 <= p_cal <= 1
        assert -1 <= mean_r <= 1
        # Real signal should yield low p-value
        assert p_cal < 0.2

    def test_null_data(self):
        from ssdiff.backends.pls import pls1_split_test_calibrated
        rng = np.random.default_rng(99)
        n, D = 60, 10
        X = rng.normal(size=(n, D))
        y = rng.normal(size=n)

        p_cal, mean_r = pls1_split_test_calibrated(
            X, y, n_components=2, n_splits=20, n_perm=50, seed=99,
        )
        assert 0 <= p_cal <= 1
        # No signal: p-value should not be small
        assert p_cal > 0.01


class TestTSF:
    def test_zero(self):
        from ssdiff.utils.math import t_sf
        assert t_sf(0.0, 10.0) == 0.5

    def test_large_positive(self):
        from ssdiff.utils.math import t_sf
        p = t_sf(5.0, 30.0)
        assert 0 < p < 0.001

    def test_negative_t(self):
        from ssdiff.utils.math import t_sf
        p = t_sf(-2.0, 10.0)
        assert p > 0.95

    def test_symmetry(self):
        from ssdiff.utils.math import t_sf
        p_pos = t_sf(2.0, 20.0)
        p_neg = t_sf(-2.0, 20.0)
        assert abs(p_pos + p_neg - 1.0) < 1e-10
