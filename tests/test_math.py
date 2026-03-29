"""Tests for ssdlite.utils.math — pure-numpy math routines."""

import numpy as np
import pytest

from ssdlite.utils.math import (
    standardize,
    pca_fit_transform,
    kmeans,
    kmeans_auto_k,
    silhouette_score,
    f_sf,
    l2_normalize_rows_inplace,
)


class TestStandardize:
    def test_basic(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Xs, mean, scale = standardize(X)
        assert np.allclose(Xs.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(Xs.std(axis=0, ddof=0), 1, atol=1e-10)

    def test_zero_variance_column(self):
        X = np.array([[1.0, 5.0], [1.0, 3.0], [1.0, 7.0]])
        Xs, mean, scale = standardize(X)
        assert np.allclose(Xs[:, 0], 0)  # zero-variance → all zeros
        assert scale[0] == 1.0  # clamped to 1


class TestPCA:
    def test_shape(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 10))
        z, comp, evr = pca_fit_transform(X, 5)
        assert z.shape == (30, 5)
        assert comp.shape == (5, 10)
        assert evr.shape == (5,)

    def test_variance_explained_sums_to_less_than_one(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 20))
        _, _, evr = pca_fit_transform(X, 10)
        assert 0 < evr.sum() <= 1.0

    def test_components_orthogonal(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 10))
        _, comp, _ = pca_fit_transform(X, 5)
        gram = comp @ comp.T
        assert np.allclose(gram, np.eye(5), atol=1e-10)


class TestKMeans:
    def test_basic(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 0.3, (20, 2)), rng.normal(3, 0.3, (20, 2))])
        labels, centers, inertia = kmeans(X, k=2, random_state=42)
        assert labels.shape == (40,)
        assert centers.shape == (2, 2)
        assert len(set(labels)) == 2
        assert inertia >= 0

    def test_auto_k(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 0.3, (15, 2)), rng.normal(3, 0.3, (15, 2))])
        labels, centers, inertia, best_k = kmeans_auto_k(X, k_min=2, k_max=5, random_state=42)
        assert best_k >= 2
        assert labels.shape == (30,)

    def test_k_too_large(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="Cannot request k=5"):
            kmeans(X, k=5)


class TestSilhouette:
    def test_perfect_clusters(self):
        X = np.array([[0, 0], [0.1, 0], [10, 10], [10.1, 10]], dtype=float)
        labels = np.array([0, 0, 1, 1])
        s = silhouette_score(X, labels)
        assert s > 0.9

    def test_single_cluster(self):
        X = np.array([[0, 0], [1, 1]], dtype=float)
        labels = np.array([0, 0])
        s = silhouette_score(X, labels)
        assert s == 0.0


class TestFSurvival:
    def test_basic(self):
        p = f_sf(3.0, 2, 17)
        assert 0 < p < 1

    def test_zero_f(self):
        assert f_sf(0.0, 2, 10) == 1.0

    def test_large_f(self):
        p = f_sf(100.0, 5, 50)
        assert p < 0.001


class TestL2Normalize:
    def test_inplace(self):
        V = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float64)
        l2_normalize_rows_inplace(V)
        for i in range(2):
            assert np.allclose(np.linalg.norm(V[i]), 1.0, atol=1e-10)


def test_chi2_sf_df_zero():
    """chi2_sf with df=0 should return 1.0 (no degrees of freedom)."""
    from ssdlite.utils.math import chi2_sf
    assert chi2_sf(5.0, 0) == 1.0
