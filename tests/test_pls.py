"""Tests for ssdlite.backends.pls — PLS1 NIPALS backend."""

import numpy as np

from ssdlite.backends.pls import pls1_fit, pls1_cv_select, pls1_permutation_test


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

    def test_1se_rule_parsimonious(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(40, 5))
        y = X[:, 0] + rng.normal(size=40) * 0.1  # Only 1 real component
        result = pls1_cv_select(X, y, max_components=5, n_folds=5, seed=42, use_1se_rule=True)
        # 1SE rule should pick fewer components
        assert result.best_n_components <= 3


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

    def test_random_data_high_pvalue(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 5))
        y = rng.normal(size=30)  # No real signal
        p, _, _ = pls1_permutation_test(X, y, n_components=2, n_perm=99, seed=42)
        assert p > 0.05  # Should not be significant


class TestPLS1SplitTest:
    def test_basic(self):
        from ssdlite.backends.pls import pls1_split_test
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
        from ssdlite.backends.pls import pls1_split_test
        rng = np.random.default_rng(99)
        n, D = 60, 10
        X = rng.normal(size=(n, D))
        y = rng.normal(size=n)

        p_split, _ = pls1_split_test(
            X, y, n_components=2, n_splits=20, seed=99,
        )
        assert 0 <= p_split <= 1
        assert p_split > 0.01


class TestTSF:
    def test_zero(self):
        from ssdlite.utils.math import t_sf
        assert t_sf(0.0, 10.0) == 0.5

    def test_large_positive(self):
        from ssdlite.utils.math import t_sf
        p = t_sf(5.0, 30.0)
        assert 0 < p < 0.001

    def test_negative_t(self):
        from ssdlite.utils.math import t_sf
        p = t_sf(-2.0, 10.0)
        assert p > 0.95

    def test_symmetry(self):
        from ssdlite.utils.math import t_sf
        p_pos = t_sf(2.0, 20.0)
        p_neg = t_sf(-2.0, 20.0)
        assert abs(p_pos + p_neg - 1.0) < 1e-10
