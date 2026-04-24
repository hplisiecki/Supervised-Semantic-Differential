"""Tests for ssdiff.backends.pls — pure-numpy NIPALS PLS1."""

from __future__ import annotations

import numpy as np
import pytest

from ssdiff.backends.pls import (
    _pls1_coef_at_k,
    pls1_cv_select,
    pls1_fit,
    pls1_permutation_test,
    pls1_split_test,
    pls1_split_test_calibrated,
)

RNG_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal_data(n=60, D=10, n_latent=3, seed=RNG_SEED):
    """Dense signal: y is a linear combination of first n_latent features + noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, D))
    beta_true = np.zeros(D)
    beta_true[:n_latent] = rng.standard_normal(n_latent) * 3.0
    y = X @ beta_true + 0.1 * rng.standard_normal(n)
    return X, y


def _make_null_data(n=60, D=10, seed=RNG_SEED + 1):
    """Pure noise: y is independent of X."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, D))
    y = rng.standard_normal(n)
    return X, y


# ---------------------------------------------------------------------------
# pls1_fit
# ---------------------------------------------------------------------------

class TestPLS1Fit:
    def test_recovers_exact_beta_on_orthogonal_noise_free(self):
        """On noise-free orthogonal design, beta must equal the true coefficients."""
        D = 5
        X = np.eye(D)
        true_beta = np.arange(D, dtype=float)
        y = X @ true_beta

        _T, _P, _W, _Q, coef = pls1_fit(X, y, n_components=D)

        np.testing.assert_allclose(coef, true_beta, atol=1e-10)

    def test_r2_near_one_on_noise_free(self):
        """R² on training data should be essentially 1 for noise-free fit."""
        D = 5
        X = np.eye(D)
        true_beta = np.arange(D, dtype=float)
        y = X @ true_beta

        _T, _P, _W, _Q, coef = pls1_fit(X, y, n_components=D)

        y_pred = X @ coef
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        assert r2 > 0.9999

    def test_rank1_signal_beta_collinear_with_direction(self):
        """For a rank-1 signal matrix, beta should be collinear with the signal direction."""
        rng = np.random.default_rng(RNG_SEED)
        n, D = 80, 10
        u = rng.standard_normal(n)
        v = rng.standard_normal(D)
        v /= np.linalg.norm(v)
        X = np.outer(u, v) + 0.01 * rng.standard_normal((n, D))
        y = u + 0.01 * rng.standard_normal(n)

        # Centre manually (pls1_fit expects centred input)
        Xc = X - X.mean(axis=0)
        yc = y - y.mean()

        _T, _P, _W, _Q, coef = pls1_fit(Xc, yc, n_components=1)

        coef_norm = coef / np.linalg.norm(coef)
        cos_sim = abs(float(coef_norm @ v))
        assert cos_sim > 0.99, f"|cos(beta, v)| = {cos_sim:.4f} < 0.99"

    def test_centering_matters_for_non_zero_mean_data(self):
        """Fitting on zero-mean vs non-zero-mean data should produce different betas."""
        rng = np.random.default_rng(RNG_SEED)
        n, D = 30, 4
        X = rng.standard_normal((n, D)) + 5.0  # large mean shift
        y = X[:, 0] + 0.1 * rng.standard_normal(n) + 3.0

        # Uncentered fit
        _T, _P, _W, _Q, coef_raw = pls1_fit(X, y, n_components=2)

        # Centered fit
        Xc = X - X.mean(axis=0)
        yc = y - y.mean()
        _T, _P, _W, _Q, coef_centered = pls1_fit(Xc, yc, n_components=2)

        assert not np.allclose(coef_raw, coef_centered, atol=1e-6), (
            "Expected uncentered and centered betas to differ on non-zero-mean data"
        )

    def test_returns_correct_shapes(self):
        """Output arrays must have consistent shapes."""
        n, D, K = 20, 6, 3
        rng = np.random.default_rng(RNG_SEED)
        X = rng.standard_normal((n, D))
        y = rng.standard_normal(n)

        T, P, W, Q, coef = pls1_fit(X, y, n_components=K)

        assert T.shape[0] == n
        assert P.shape[0] == D
        assert W.shape[0] == D
        assert T.shape[1] == P.shape[1] == W.shape[1] == len(Q)
        assert coef.shape == (D,)

    def test_n_components_capped_at_rank(self):
        """n_components is silently capped at min(n-1, D)."""
        n, D = 5, 3
        rng = np.random.default_rng(RNG_SEED)
        X = rng.standard_normal((n, D))
        y = rng.standard_normal(n)

        T, P, W, Q, coef = pls1_fit(X, y, n_components=100)

        # Should not raise and should have at most min(n-1, D) = 3 components
        assert T.shape[1] <= min(n - 1, D)


# ---------------------------------------------------------------------------
# pls1_permutation_test
# ---------------------------------------------------------------------------

class TestPLS1PermutationTest:
    def test_signal_data_p_below_threshold(self):
        """Signal data should yield p < 0.05."""
        X, y = _make_signal_data(n=60, D=10)
        p, cv_r2_obs, cv_r2_null = pls1_permutation_test(
            X, y, n_components=3, n_perm=200, seed=RNG_SEED
        )
        assert p < 0.05, f"Expected p < 0.05 on signal data, got p={p:.4f}"

    def test_null_data_p_not_too_small(self):
        """Null data should not be flagged as significant (p > 0.3 with 200 perms)."""
        X, y = _make_null_data(n=60, D=10)
        p, cv_r2_obs, cv_r2_null = pls1_permutation_test(
            X, y, n_components=1, n_perm=200, seed=RNG_SEED
        )
        assert p > 0.3, f"Expected p > 0.3 on null data, got p={p:.4f}"

    def test_returns_valid_p_value(self):
        """p-value must be in [0, 1]."""
        X, y = _make_signal_data()
        p, cv_r2_obs, cv_r2_null = pls1_permutation_test(
            X, y, n_components=2, n_perm=50, seed=RNG_SEED
        )
        assert 0.0 <= p <= 1.0

    def test_pca_k_preprocessing_runs_and_valid(self):
        """pca_k preprocessing path should run without error and return valid p."""
        n, D = 60, 20
        rng = np.random.default_rng(RNG_SEED)
        X = rng.standard_normal((n, D))
        y = X[:, 0] * 2 + X[:, 1] + 0.1 * rng.standard_normal(n)

        p, cv_r2_obs, cv_r2_null = pls1_permutation_test(
            X, y, n_components=2, n_perm=50, seed=RNG_SEED, pca_k=5
        )
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# pls1_split_test
# ---------------------------------------------------------------------------

class TestPLS1SplitTest:
    def test_signal_p_below_threshold(self):
        """Split-half test on signal data: p < 0.05."""
        X, y = _make_signal_data(n=60, D=10)
        p, mean_r = pls1_split_test(X, y, n_components=3, seed=RNG_SEED)
        assert p < 0.05, f"Expected p < 0.05 on signal data, got p={p:.4f}"

    def test_null_p_not_too_small(self):
        """Split-half test on null data: p > 0.2."""
        X, y = _make_null_data(n=60, D=10)
        p, mean_r = pls1_split_test(X, y, n_components=1, seed=RNG_SEED)
        assert p > 0.2, f"Expected p > 0.2 on null data, got p={p:.4f}"

    def test_returns_valid_p_and_mean_r(self):
        """Return values must be valid: p in [0,1], mean_r in [-1,1]."""
        X, y = _make_signal_data()
        p, mean_r = pls1_split_test(X, y, n_components=2, seed=RNG_SEED)
        assert 0.0 <= p <= 1.0
        assert -1.0 <= mean_r <= 1.0


# ---------------------------------------------------------------------------
# pls1_split_test_calibrated
# ---------------------------------------------------------------------------

class TestPLS1SplitTestCalibrated:
    def test_signal_p_below_threshold(self):
        """Calibrated split test on signal data: p < 0.05."""
        X, y = _make_signal_data(n=60, D=10)
        p, mean_r = pls1_split_test_calibrated(
            X, y, n_components=3, n_perm=100, seed=RNG_SEED
        )
        assert p < 0.05, f"Expected p < 0.05 on signal data, got p={p:.4f}"

    def test_null_p_not_too_small(self):
        """Calibrated split test on null data: p > 0.2."""
        X, y = _make_null_data(n=60, D=10)
        p, mean_r = pls1_split_test_calibrated(
            X, y, n_components=1, n_perm=100, seed=RNG_SEED
        )
        assert p > 0.2, f"Expected p > 0.2 on null data, got p={p:.4f}"


# ---------------------------------------------------------------------------
# pls1_cv_select
# ---------------------------------------------------------------------------

class TestPLS1CVSelect:
    def test_selects_reasonable_k_on_signal(self):
        """CV selection on 3-latent-dim signal: returned k in [2, max_components]."""
        X, y = _make_signal_data(n=80, D=10, n_latent=3)
        max_comp = 6
        result = pls1_cv_select(X, y, max_components=max_comp, n_folds=5, seed=RNG_SEED)

        assert result.best_n_components >= 2, (
            f"Expected k >= 2, got {result.best_n_components}"
        )
        assert result.best_n_components <= max_comp, (
            f"Expected k <= {max_comp}, got {result.best_n_components}"
        )

    def test_raises_on_single_sample(self):
        """CV with n=1 must raise ValueError."""
        X = np.array([[1.0, 2.0]])
        y = np.array([1.0])
        with pytest.raises(ValueError):
            pls1_cv_select(X, y, max_components=2, n_folds=5, seed=RNG_SEED)

    def test_result_has_expected_fields(self):
        """PLSCVResult must expose best_n_components, cv_scores, cv_scores_se, best_cv_r2."""
        X, y = _make_signal_data(n=40, D=6)
        result = pls1_cv_select(X, y, max_components=3, n_folds=5, seed=RNG_SEED)

        assert isinstance(result.best_n_components, int)
        assert isinstance(result.cv_scores, dict)
        assert isinstance(result.cv_scores_se, dict)
        assert isinstance(result.best_cv_r2, float)
