"""Tests for ssdiff.utils.math — pure-numpy math utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.distance import squareform, pdist
from scipy.stats import f as sp_f, t as sp_t, chi2 as sp_chi2
from sklearn.metrics import silhouette_score as sk_silhouette
from sklearn.decomposition import PCA

from ssdiff.utils.math import (
    standardize,
    unit_vector,
    l2_normalize_rows_inplace,
    pca_fit_transform,
    kmeans,
    silhouette_score,
    pairwise_euclidean,
    t_sf,
    f_sf,
    chi2_sf,
)


# ---------------------------------------------------------------------------
# standardize
# ---------------------------------------------------------------------------


class TestStandardize:
    def test_output_mean_near_zero(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 4))
        Xs, _, _ = standardize(X)
        col_means = Xs.mean(axis=0)
        assert np.all(np.abs(col_means) < 1e-12), f"means: {col_means}"

    def test_output_std_near_one(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 4))
        Xs, _, _ = standardize(X)
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
    def test_unit_input_unchanged(self):
        v = np.array([1.0, 0.0, 0.0])
        result = unit_vector(v)
        np.testing.assert_allclose(result, v, atol=1e-15)

    def test_arbitrary_vector_becomes_unit(self):
        v = np.array([3.0, 4.0])
        result = unit_vector(v)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-15)
        np.testing.assert_allclose(result, np.array([0.6, 0.8]), atol=1e-15)

    def test_near_zero_norm_returns_zero_vector(self):
        v = np.array([1e-15, 1e-15])
        result = unit_vector(v)
        assert np.all(result == 0.0), f"expected zero vector, got {result}"

    def test_exact_zero_returns_zero_vector(self):
        v = np.zeros(3)
        result = unit_vector(v)
        assert np.all(result == 0.0)


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

    def test_modifies_in_place(self):
        V = np.array([[3.0, 4.0], [1.0, 0.0]])
        V_id = id(V)
        l2_normalize_rows_inplace(V)
        assert id(V) == V_id  # same object
        np.testing.assert_allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# pca_fit_transform
# ---------------------------------------------------------------------------


class TestPcaFitTransform:
    @pytest.fixture
    def X_symmetric(self):
        return np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

    def test_components_orthonormal(self, X_symmetric):
        _, components, _ = pca_fit_transform(X_symmetric, n_components=2)
        G = components @ components.T
        np.testing.assert_allclose(G, np.eye(2), atol=1e-12)

    def test_evr_matches_sklearn(self, X_symmetric):
        _, _, evr = pca_fit_transform(X_symmetric, n_components=1)
        sk = PCA(n_components=1).fit(X_symmetric)
        np.testing.assert_allclose(evr, sk.explained_variance_ratio_, atol=1e-10)

    def test_evr_sums_to_one_for_full_rank(self, X_symmetric):
        _, _, evr = pca_fit_transform(X_symmetric, n_components=2)
        np.testing.assert_allclose(evr.sum(), 1.0, atol=1e-12)

    def test_projection_shape(self, X_symmetric):
        z, components, evr = pca_fit_transform(X_symmetric, n_components=1)
        assert z.shape == (4, 1)
        assert components.shape == (1, 2)
        assert evr.shape == (1,)


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

    def test_returns_two_clusters(self, two_gaussians):
        labels, centers, inertia = kmeans(two_gaussians, k=2, random_state=0)
        assert len(np.unique(labels)) == 2

    def test_centers_close_to_true_means(self, two_gaussians):
        labels, centers, inertia = kmeans(two_gaussians, k=2, random_state=0)
        # Sort centers by distance from origin so order is deterministic
        order = np.argsort(np.linalg.norm(centers, axis=1))
        centers_sorted = centers[order]
        true_means = np.array([[0.0, 0.0], [10.0, 10.0]])
        # atol=0.5: sample means from n=20, std=0.3 can be ~0.2 off the true mean
        np.testing.assert_allclose(centers_sorted, true_means, atol=0.5)

    def test_inertia_positive(self, two_gaussians):
        _, _, inertia = kmeans(two_gaussians, k=2, random_state=0)
        assert inertia >= 0.0

    def test_labels_shape(self, two_gaussians):
        labels, centers, inertia = kmeans(two_gaussians, k=2, random_state=0)
        assert labels.shape == (40,)
        assert centers.shape == (2, 2)


# ---------------------------------------------------------------------------
# silhouette_score
# ---------------------------------------------------------------------------


class TestSilhouetteScore:
    def test_parity_with_sklearn(self):
        rng = np.random.default_rng(99)
        X = rng.standard_normal((30, 3))
        labels = np.array([0] * 15 + [1] * 15)
        ours = silhouette_score(X, labels)
        theirs = sk_silhouette(X, labels, metric="euclidean")
        assert abs(ours - theirs) < 1e-9, f"ours={ours}, theirs={theirs}"


# ---------------------------------------------------------------------------
# pairwise_euclidean
# ---------------------------------------------------------------------------


class TestPairwiseEuclidean:
    @pytest.fixture
    def X(self):
        rng = np.random.default_rng(11)
        return rng.standard_normal((15, 4))

    def test_diagonal_near_zero(self, X):
        # BLAS dot-product trick can leave small FP residuals on the diagonal
        # before sqrt; source clamps negatives but positive residuals may survive.
        D = pairwise_euclidean(X)
        np.testing.assert_allclose(D.diagonal(), 0.0, atol=1e-6)

    def test_symmetric(self, X):
        D = pairwise_euclidean(X)
        np.testing.assert_array_equal(D, D.T)

    def test_matches_scipy_pdist(self, X):
        D = pairwise_euclidean(X)
        D_scipy = squareform(pdist(X, metric="euclidean"))
        # Off-diagonal matches to 1e-12; diagonal may have ~1e-8 FP residuals
        # from the BLAS trick (positive D_sq before sqrt not clamped to 0).
        np.testing.assert_allclose(D, D_scipy, atol=1e-6)

    def test_output_shape(self, X):
        D = pairwise_euclidean(X)
        assert D.shape == (15, 15)


# ---------------------------------------------------------------------------
# t_sf / f_sf / chi2_sf — scipy parity
# ---------------------------------------------------------------------------


T_VALUES = [0.5, 1.0, 2.0, 3.5, 5.0]
F_VALUES = [0.5, 1.0, 2.5, 4.0, 7.0]
CHI2_VALUES = [0.1, 1.0, 3.84, 7.0, 12.0]


@pytest.mark.parametrize("t", T_VALUES)
def test_t_sf_parity_scipy(t):
    df = 10
    ours = t_sf(t, df)
    theirs = float(sp_t.sf(t, df))
    assert abs(ours - theirs) < 1e-14, f"t={t}, df={df}: ours={ours}, theirs={theirs}"


@pytest.mark.parametrize("f", F_VALUES)
def test_f_sf_parity_scipy(f):
    dfn, dfd = 3, 20
    ours = f_sf(f, dfn, dfd)
    theirs = float(sp_f.sf(f, dfn, dfd))
    assert abs(ours - theirs) < 1e-14, f"f={f}: ours={ours}, theirs={theirs}"


@pytest.mark.parametrize("x", CHI2_VALUES)
def test_chi2_sf_parity_scipy(x):
    df = 5
    ours = chi2_sf(x, df)
    theirs = float(sp_chi2.sf(x, df))
    # gammainc series converges to ~12 sig digits; allow 1e-13 for edge cases
    assert abs(ours - theirs) < 1e-13, f"x={x}: ours={ours}, theirs={theirs}"


# ---------------------------------------------------------------------------
# Boundary / edge cases
# ---------------------------------------------------------------------------


def test_t_sf_df_zero_returns_nan():
    result = t_sf(1.0, df=0)
    assert math.isnan(result), f"expected nan, got {result}"


def test_chi2_sf_x_zero_returns_one():
    result = chi2_sf(0.0, df=1)
    assert result == 1.0, f"expected 1.0, got {result}"
