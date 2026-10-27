"""Tests for ssdiff.utils.math — pure-numpy math utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from ssdiff.utils.math import (
    chi2_sf,
    kmeans,
    l2_normalize_rows_inplace,
    pairwise_euclidean,
    standardize,
    t_sf,
    unit_vector,
)

# ---------------------------------------------------------------------------
# standardize
# ---------------------------------------------------------------------------


class TestStandardize:
    def test_output_has_zero_mean_unit_std(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 4))
        Xs, _, _ = standardize(X)
        col_means = Xs.mean(axis=0)
        assert np.all(np.abs(col_means) < 1e-12), f"means: {col_means}"
        col_stds = Xs.std(axis=0, ddof=0)
        assert np.allclose(col_stds, 1.0, atol=1e-12), f"stds: {col_stds}"

    def test_returns_mean_and_scale(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Xs, mean, scale = standardize(X)
        assert mean.shape == (2,)
        assert scale.shape == (2,)
        np.testing.assert_allclose(mean, X.mean(axis=0), atol=1e-14)
        np.testing.assert_allclose(scale, X.std(axis=0, ddof=0), atol=1e-14)

    def test_constant_column_no_exception_scale_is_one(self):
        # Constant columns must not raise and scale is clamped to 1.0
        X = np.ones((10, 2))
        Xs, mean, scale = standardize(X)
        # scale must be 1.0 (clamped), not 0
        assert np.all(scale == 1.0), f"scale: {scale}"
        # output is (X - mean) / 1.0 = zeros
        assert np.all(Xs == 0.0), f"Xs: {Xs}"


# ---------------------------------------------------------------------------
# unit_vector
# ---------------------------------------------------------------------------


class TestUnitVector:
    def test_arbitrary_vector_becomes_unit(self):
        v = np.array([3.0, 4.0])
        result = unit_vector(v)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-15)
        np.testing.assert_allclose(result, np.array([0.6, 0.8]), atol=1e-15)

    def test_near_zero_norm_returns_zero_vector(self):
        v = np.array([1e-15, 1e-15])
        result = unit_vector(v)
        assert np.all(result == 0.0), f"expected zero vector, got {result}"


# ---------------------------------------------------------------------------
# l2_normalize_rows_inplace
# ---------------------------------------------------------------------------


class TestL2NormalizeRowsInplace:
    def test_rows_have_unit_norm_float64(self):
        rng = np.random.default_rng(42)
        V = rng.standard_normal((10, 5))
        l2_normalize_rows_inplace(V)
        norms = np.linalg.norm(V, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_rows_have_unit_norm_float32(self):
        rng = np.random.default_rng(42)
        V = rng.standard_normal((10, 5)).astype(np.float32)
        l2_normalize_rows_inplace(V)
        norms = np.linalg.norm(V.astype(np.float64), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# kmeans
# ---------------------------------------------------------------------------


class TestKmeans:
    @pytest.fixture
    def two_gaussians(self):
        rng = np.random.default_rng(7)
        cluster_a = rng.normal([0.0, 0.0], 0.3, size=(20, 2))
        cluster_b = rng.normal([10.0, 10.0], 0.3, size=(20, 2))
        X = np.vstack([cluster_a, cluster_b])
        return X

    def test_recovers_two_blobs(self, two_gaussians):
        labels, centers, inertia = kmeans(two_gaussians, k=2, random_state=0)
        assert len(np.unique(labels)) == 2
        assert labels.shape == (40,)
        assert centers.shape == (2, 2)
        # Sort centers by distance from origin so order is deterministic
        order = np.argsort(np.linalg.norm(centers, axis=1))
        centers_sorted = centers[order]
        true_means = np.array([[0.0, 0.0], [10.0, 10.0]])
        # atol=0.5: sample means from n=20, std=0.3 can be ~0.2 off the true mean
        np.testing.assert_allclose(centers_sorted, true_means, atol=0.5)


# ---------------------------------------------------------------------------
# pairwise_euclidean
# ---------------------------------------------------------------------------


class TestPairwiseEuclidean:
    @pytest.fixture
    def X(self):
        rng = np.random.default_rng(11)
        return rng.standard_normal((15, 4))

    def test_matches_scipy_pdist(self, X):
        D = pairwise_euclidean(X)
        D_scipy = squareform(pdist(X, metric="euclidean"))
        # Off-diagonal matches to 1e-12; diagonal may have ~1e-8 FP residuals
        # from the BLAS trick (positive D_sq before sqrt not clamped to 0).
        np.testing.assert_allclose(D, D_scipy, atol=1e-6)
        np.testing.assert_array_equal(D, D.T)


# ---------------------------------------------------------------------------
# Boundary cases
# ---------------------------------------------------------------------------


def test_t_sf_df_zero_returns_nan():
    result = t_sf(1.0, df=0)
    assert math.isnan(result), f"expected nan, got {result}"


def test_chi2_sf_x_zero_returns_one():
    result = chi2_sf(0.0, df=1)
    assert result == 1.0, f"expected 1.0, got {result}"
