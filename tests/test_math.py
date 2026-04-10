"""Tests for ssdiff.utils.math — pure-numpy math routines."""

import numpy as np
import pytest

from ssdiff.utils.math import (
    f_sf,
    kmeans,
    kmeans_auto_k,
    l2_normalize_rows_inplace,
    pca_fit_transform,
    silhouette_score,
    standardize,
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

    def test_variance_decreasing_and_centered(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 20))
        z, _, evr = pca_fit_transform(X, 10)
        # Variance explained per component is in decreasing order
        assert all(evr[i] >= evr[i + 1] for i in range(len(evr) - 1))
        # Projected data is centered (mean ≈ 0 along each component)
        assert np.allclose(z.mean(axis=0), 0, atol=1e-10)


class TestKMeans:
    def test_basic(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 0.3, (20, 2)), rng.normal(3, 0.3, (20, 2))])
        labels, centers, inertia = kmeans(X, k=2, random_state=42)
        assert labels.shape == (40,)
        assert centers.shape == (2, 2)
        assert len(set(labels)) == 2
        assert inertia >= 0
        # Cluster purity: the two Gaussians at 0 and 3 (std 0.3) should separate nearly perfectly
        label_a = labels[0]  # label assigned to first group
        label_b = labels[20]  # label assigned to second group
        assert label_a != label_b, "Two clusters should get different labels"
        purity_a = np.sum(labels[:20] == label_a) / 20
        purity_b = np.sum(labels[20:] == label_b) / 20
        assert purity_a > 0.9, f"Cluster 0 purity too low: {purity_a}"
        assert purity_b > 0.9, f"Cluster 1 purity too low: {purity_b}"
        # Inertia should be reasonably low for 40 points with std 0.3
        assert inertia < 20
        # Centers should be near the true means [0,0] and [3,3]
        sorted_centers = centers[np.argsort(centers[:, 0])]
        assert np.allclose(sorted_centers[0], [0, 0], atol=1.0)
        assert np.allclose(sorted_centers[1], [3, 3], atol=1.0)

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
        assert s > 0.99

    def test_single_cluster(self):
        X = np.array([[0, 0], [1, 1]], dtype=float)
        labels = np.array([0, 0])
        s = silhouette_score(X, labels)
        assert s == 0.0

    def test_bad_clustering_low_score(self):
        # Well-separated data with wrong (shuffled) labels → low silhouette
        X = np.array(
            [[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1],
             [10, 10], [10.1, 10], [10, 10.1], [10.1, 10.1]],
            dtype=float,
        )
        rng = np.random.default_rng(123)
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        rng.shuffle(labels)
        s = silhouette_score(X, labels)
        assert s < 0.5

    def test_matches_manual_computation(self):
        # Manual silhouette for [[0,0],[1,0],[5,0],[6,0]], labels=[0,0,1,1]
        # Distances: d01=1, d02=5, d03=6, d12=4, d13=5, d23=1
        # a(0)=d(0,1)/(2-1)=1, b(0)=mean(d(0,2),d(0,3))=(5+6)/2=5.5, s(0)=4.5/5.5
        # a(1)=d(1,0)/(2-1)=1, b(1)=mean(d(1,2),d(1,3))=(4+5)/2=4.5, s(1)=3.5/4.5
        # a(2)=d(2,3)/(2-1)=1, b(2)=mean(d(2,0),d(2,1))=(5+4)/2=4.5, s(2)=3.5/4.5
        # a(3)=d(3,2)/(2-1)=1, b(3)=mean(d(3,0),d(3,1))=(6+5)/2=5.5, s(3)=4.5/5.5
        # Mean = (4.5/5.5 + 3.5/4.5 + 3.5/4.5 + 4.5/5.5) / 4 ≈ 0.797979...
        X = np.array([[0, 0], [1, 0], [5, 0], [6, 0]], dtype=float)
        labels = np.array([0, 0, 1, 1])
        s = silhouette_score(X, labels)
        expected = (4.5 / 5.5 + 3.5 / 4.5 + 3.5 / 4.5 + 4.5 / 5.5) / 4
        assert s == pytest.approx(expected, abs=1e-10)


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
    from ssdiff.utils.math import chi2_sf
    assert chi2_sf(5.0, 0) == 1.0
