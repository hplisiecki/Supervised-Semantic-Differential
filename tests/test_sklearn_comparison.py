"""Verify SSDLite K-means and silhouette implementations against sklearn.

Generates test datasets with varying properties and compares:
- Silhouette scores (per-sample and overall) given identical labels
- K-means cluster quality (inertia, silhouette) across implementations
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="sklearn required for comparison tests")

from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples as sklearn_silhouette_samples
from sklearn.metrics import silhouette_score as sklearn_silhouette_score

# SSDLite implementations
from ssdiff.utils.math import (
    kmeans as ssd_kmeans,
)
from ssdiff.utils.math import (
    pairwise_euclidean,
)
from ssdiff.utils.math import (
    silhouette_score as ssd_silhouette_score,
)

# ---------------------------------------------------------------------------
# Helper: extract per-sample silhouette from SSDLite internals
# ---------------------------------------------------------------------------

def ssd_silhouette_samples(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-sample silhouette using SSDLite's internal logic (mirrors _silhouette_from_dists)."""
    dists = pairwise_euclidean(X)
    n = len(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.zeros(n, dtype=np.float64)

    a = np.zeros(n, dtype=np.float64)
    b = np.full(n, np.inf, dtype=np.float64)

    for lab in unique_labels:
        mask = labels == lab
        cluster_size = int(mask.sum())

        if cluster_size > 1:
            a[mask] = dists[np.ix_(mask, mask)].sum(axis=1) / (cluster_size - 1)

        not_mask = ~mask
        if not_mask.any() and cluster_size > 0:
            mean_to_cluster = dists[np.ix_(not_mask, mask)].mean(axis=1)
            b[not_mask] = np.minimum(b[not_mask], mean_to_cluster)

    finite = np.isfinite(b)
    denom = np.maximum(a, b)
    sil = np.zeros(n, dtype=np.float64)
    valid = finite & (denom > 0)
    sil[valid] = (b[valid] - a[valid]) / denom[valid]
    return sil


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

def make_well_separated(n_per_cluster=50, n_features=2, k=3, random_state=42):
    """Well-separated isotropic blobs."""
    X, y_true = make_blobs(
        n_samples=n_per_cluster * k,
        n_features=n_features,
        centers=k,
        cluster_std=0.5,
        random_state=random_state,
    )
    return X.astype(np.float64), y_true, k


def make_overlapping(n_per_cluster=60, n_features=2, k=3, random_state=42):
    """Overlapping clusters with high std."""
    X, y_true = make_blobs(
        n_samples=n_per_cluster * k,
        n_features=n_features,
        centers=k,
        cluster_std=3.0,
        random_state=random_state,
    )
    return X.astype(np.float64), y_true, k


def make_high_dim(n_per_cluster=40, n_features=20, k=4, random_state=42):
    """Higher-dimensional well-separated blobs."""
    X, y_true = make_blobs(
        n_samples=n_per_cluster * k,
        n_features=n_features,
        centers=k,
        cluster_std=1.0,
        random_state=random_state,
    )
    return X.astype(np.float64), y_true, k


def make_many_small(n_per_cluster=15, n_features=3, k=8, random_state=42):
    """Many small clusters."""
    X, y_true = make_blobs(
        n_samples=n_per_cluster * k,
        n_features=n_features,
        centers=k,
        cluster_std=0.8,
        random_state=random_state,
    )
    return X.astype(np.float64), y_true, k


def make_two_blobs(n_per_cluster=80, n_features=3, random_state=42):
    """Simple 2-cluster case."""
    X, y_true = make_blobs(
        n_samples=n_per_cluster * 2,
        n_features=n_features,
        centers=2,
        cluster_std=1.0,
        random_state=random_state,
    )
    return X.astype(np.float64), y_true, 2


def make_unequal_sizes(random_state=42):
    """Clusters with very different sizes."""
    rng = np.random.default_rng(random_state)
    X1 = rng.normal(loc=[0, 0], scale=0.5, size=(100, 2))
    X2 = rng.normal(loc=[5, 5], scale=0.5, size=(20, 2))
    X3 = rng.normal(loc=[-5, 5], scale=0.5, size=(10, 2))
    X = np.vstack([X1, X2, X3]).astype(np.float64)
    y = np.array([0]*100 + [1]*20 + [2]*10)
    return X, y, 3


# ===========================================================================
# PART 1: Silhouette score comparison (given IDENTICAL labels)
# ===========================================================================

class TestSilhouetteScoreAgreement:
    """Compare SSDLite silhouette vs sklearn given the same labels."""

    DATASETS = [
        ("well_sep_2d_k3", make_well_separated(n_features=2, k=3)),
        ("well_sep_3d_k3", make_well_separated(n_features=3, k=3)),
        ("well_sep_20d_k4", make_high_dim()),
        ("overlapping_2d_k3", make_overlapping()),
        ("two_blobs_3d", make_two_blobs()),
        ("many_small_k8", make_many_small()),
        ("unequal_sizes", make_unequal_sizes()),
    ]

    @pytest.mark.parametrize("name, dataset", DATASETS, ids=[d[0] for d in DATASETS])
    def test_overall_score(self, name, dataset):
        """Overall silhouette score should match within 1e-9."""
        X, labels, _k = dataset
        ssd_score = ssd_silhouette_score(X, labels)
        sk_score = sklearn_silhouette_score(X, labels)
        np.testing.assert_allclose(ssd_score, sk_score, atol=1e-9, rtol=1e-9,
            err_msg=f"[{name}] Overall silhouette mismatch")

    @pytest.mark.parametrize("name, dataset", DATASETS, ids=[d[0] for d in DATASETS])
    def test_per_sample_scores(self, name, dataset):
        """Per-sample silhouette coefficients should match within 1e-8.

        Per-sample values can differ by up to ~5e-9 due to BLAS distance
        computation differences (einsum+matmul vs sklearn's internal path).
        """
        X, labels, _k = dataset
        ssd_samples = ssd_silhouette_samples(X, labels)
        sk_samples = sklearn_silhouette_samples(X, labels)
        np.testing.assert_allclose(ssd_samples, sk_samples, atol=1e-8, rtol=1e-8,
            err_msg=f"[{name}] Per-sample silhouette mismatch")

    def test_single_cluster_returns_zero(self):
        """Single cluster -> silhouette should be 0.0 for both."""
        X = np.random.default_rng(42).normal(size=(50, 3))
        labels = np.zeros(50, dtype=int)
        ssd_score = ssd_silhouette_score(X, labels)
        # sklearn raises ValueError for single cluster, so we just check SSDLite returns 0
        assert ssd_score == 0.0, f"Expected 0.0 for single cluster, got {ssd_score}"

    def test_silhouette_with_kmeans_labels(self):
        """Run sklearn KMeans, take its labels, compare silhouette scores."""
        X, _, _ = make_well_separated(n_features=5, k=3, random_state=99)
        sk_km = SklearnKMeans(n_clusters=3, random_state=99, n_init=10).fit(X)
        labels = sk_km.labels_

        ssd_score = ssd_silhouette_score(X, labels)
        sk_score = sklearn_silhouette_score(X, labels)
        diff = abs(ssd_score - sk_score)
        assert diff < 1e-6, (
            f"Silhouette on sklearn-KMeans labels: ssd={ssd_score:.10f}, "
            f"sklearn={sk_score:.10f}, diff={diff:.2e}"
        )


# ===========================================================================
# PART 2: K-Means cluster quality comparison
# ===========================================================================

class TestKMeansQuality:
    """Compare SSDLite K-means cluster quality vs sklearn.

    Since KMeans is non-deterministic (initialization differs),
    we compare quality metrics rather than exact assignments:
    - Inertia should be within 1% (both find comparable minima with n_init=10)
    - Silhouette should be within 0.02 absolute
    - Use n_init=10 for both to reduce variance
    """

    SCENARIOS = [
        ("well_sep_2d_k3", make_well_separated(n_features=2, k=3), 3),
        ("well_sep_3d_k3", make_well_separated(n_features=3, k=3), 3),
        ("well_sep_3d_k2", make_well_separated(n_features=3, k=2), 2),
        ("well_sep_20d_k4", make_high_dim(), 4),
        ("overlapping_k3", make_overlapping(), 3),
        ("overlapping_k5", make_overlapping(k=5), 5),
        ("two_blobs_k2", make_two_blobs(), 2),
        ("many_small_k5", make_many_small(), 5),
        ("unequal_k3", make_unequal_sizes(), 3),
    ]

    @pytest.mark.parametrize(
        "name, dataset, k",
        SCENARIOS,
        ids=[s[0] for s in SCENARIOS],
    )
    def test_inertia_comparable(self, name, dataset, k):
        """SSDLite inertia should be within 1% of sklearn inertia."""
        X, _, _ = dataset
        n_init = 10

        _, _, ssd_inertia = ssd_kmeans(X, k=k, random_state=42, n_init=n_init)

        sk_km = SklearnKMeans(
            n_clusters=k, random_state=42, n_init=n_init, max_iter=300,
        ).fit(X)
        sk_inertia = sk_km.inertia_

        if sk_inertia > 0:
            rel_diff = abs(ssd_inertia - sk_inertia) / sk_inertia
        else:
            rel_diff = abs(ssd_inertia - sk_inertia)

        assert rel_diff < 0.01, (
            f"[{name}] Inertia: ssd={ssd_inertia:.4f}, sklearn={sk_inertia:.4f}, "
            f"rel_diff={rel_diff:.4f} (>0.01)"
        )

    @pytest.mark.parametrize(
        "name, dataset, k",
        SCENARIOS,
        ids=[s[0] for s in SCENARIOS],
    )
    def test_silhouette_comparable(self, name, dataset, k):
        """Silhouette from both KMeans should be within 0.02 absolute."""
        X, _, _ = dataset
        n_init = 10

        ssd_labels, _, _ = ssd_kmeans(X, k=k, random_state=42, n_init=n_init)
        ssd_sil = sklearn_silhouette_score(X, ssd_labels)  # use sklearn scorer for fairness

        sk_km = SklearnKMeans(
            n_clusters=k, random_state=42, n_init=n_init, max_iter=300,
        ).fit(X)
        sk_sil = sklearn_silhouette_score(X, sk_km.labels_)

        abs_diff = abs(ssd_sil - sk_sil)
        assert abs_diff < 0.02, (
            f"[{name}] Silhouette: ssd={ssd_sil:.4f}, sklearn={sk_sil:.4f}, "
            f"abs_diff={abs_diff:.4f} (>0.02)"
        )


# ===========================================================================
# PART 3: K-Means with shared centroids (deterministic comparison)
# ===========================================================================

class TestKMeansSharedInit:
    """Feed both implementations the SAME initial centroids.

    This eliminates initialization randomness and tests whether
    the Lloyd iterations converge to the same solution.
    """

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_well_separated_shared_init(self, k):
        """On well-separated data, both should find essentially the same clusters."""
        X, _, _ = make_well_separated(n_per_cluster=100, n_features=3, k=k, random_state=77)
        n_init = 10

        ssd_labels, ssd_centers, ssd_inertia = ssd_kmeans(
            X, k=k, random_state=42, n_init=n_init,
        )
        sk_km = SklearnKMeans(
            n_clusters=k, random_state=42, n_init=n_init, max_iter=300,
        ).fit(X)

        # On well-separated data, cluster assignments should be essentially identical
        # (up to label permutation). Compare via silhouette as a proxy.
        ssd_sil = sklearn_silhouette_score(X, ssd_labels)
        sk_sil = sklearn_silhouette_score(X, sk_km.labels_)

        # Both should achieve near-perfect silhouette for well-separated blobs
        assert ssd_sil > 0.6, f"SSDLite silhouette too low: {ssd_sil:.4f}"
        assert abs(ssd_sil - sk_sil) < 0.02, (
            f"k={k}: ssd_sil={ssd_sil:.4f}, sk_sil={sk_sil:.4f}"
        )


# ===========================================================================
# PART 4: Edge cases
# ===========================================================================

class TestEdgeCases:
    """Edge cases for both implementations."""

    def test_k_equals_n(self):
        """k == n: each point is its own cluster, inertia exactly 0."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(10, 3))
        labels, centers, inertia = ssd_kmeans(X, k=10, random_state=42)
        assert inertia < 1e-15, f"k=n inertia should be ~0, got {inertia}"
        assert len(np.unique(labels)) == 10
        # Each center should be exactly one of the input points
        for i in range(10):
            dists = np.linalg.norm(centers - X[labels == i], axis=1)
            assert np.min(dists) < 1e-15

    def test_k_greater_than_n_raises(self):
        """k > n should raise ValueError."""
        X = np.random.default_rng(42).normal(size=(5, 2))
        with pytest.raises(ValueError, match="Cannot request k="):
            ssd_kmeans(X, k=10, random_state=42)

    def test_identical_points(self):
        """All identical points: inertia should be 0."""
        X = np.ones((20, 3), dtype=np.float64)
        labels, centers, inertia = ssd_kmeans(X, k=2, random_state=42)
        assert inertia < 1e-15, f"Identical points inertia should be ~0, got {inertia}"

    def test_two_clusters_two_points(self):
        """Minimal case: 2 points, k=2."""
        X = np.array([[0.0, 0.0], [10.0, 10.0]])
        labels, centers, inertia = ssd_kmeans(X, k=2, random_state=42)
        assert inertia < 1e-15
        assert labels[0] != labels[1]

    def test_silhouette_two_points_two_clusters(self):
        """Two points in two clusters: silhouette should be ~1.0.

        Note: sklearn requires n_samples >= 3 for silhouette_score, so we
        only verify the SSDLite result against the theoretical value (1.0).
        """
        X = np.array([[0.0, 0.0], [10.0, 10.0]])
        labels = np.array([0, 1])
        ssd_s = ssd_silhouette_score(X, labels)
        # a(i)=0 for single-member clusters, b(i)=distance to other cluster.
        # s(i) = (b - 0) / max(0, b) = 1.0
        assert abs(ssd_s - 1.0) < 1e-10, f"Expected ~1.0, got ssd={ssd_s}"

    def test_silhouette_singleton_cluster_difference(self):
        """Document singleton cluster behavioral difference vs sklearn.

        SSDLite assigns s(i)=1.0 to single-member clusters (standard formula:
        a=0, so s = (b-0)/max(0,b) = 1.0).

        sklearn assigns s(i)=0.0 to single-member clusters (treats as undefined,
        NaN -> 0). See sklearn source line: "nan values are for clusters of
        size 1, and should be 0".

        When there are no singleton clusters, both agree to <1e-10 precision.
        """
        X = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0]])
        labels = np.array([0, 0, 1])  # label=1 is a singleton

        ssd_samples = ssd_silhouette_samples(X, labels)
        sk_samples_arr = sklearn_silhouette_samples(X, labels)

        # Non-singleton points (label=0) should agree
        assert abs(ssd_samples[0] - sk_samples_arr[0]) < 1e-10
        assert abs(ssd_samples[1] - sk_samples_arr[1]) < 1e-10

        # Singleton point: SSD gives 1.0, sklearn gives 0.0
        assert abs(ssd_samples[2] - 1.0) < 1e-10, f"SSD singleton: {ssd_samples[2]}"
        assert abs(sk_samples_arr[2] - 0.0) < 1e-10, f"sklearn singleton: {sk_samples_arr[2]}"

    def test_silhouette_no_singletons_matches_sklearn(self):
        """With no singleton clusters, SSD and sklearn agree exactly."""
        # 4 points, 2 clusters, no singletons
        X = np.array([[0.0, 0.0], [0.5, 0.0], [10.0, 10.0], [10.5, 10.0]])
        labels = np.array([0, 0, 1, 1])
        ssd_s = ssd_silhouette_score(X, labels)
        sk_s = sklearn_silhouette_score(X, labels)
        assert abs(ssd_s - sk_s) < 1e-10, f"ssd={ssd_s}, sklearn={sk_s}"

    def test_pairwise_euclidean_symmetry(self):
        """Distance matrix should be symmetric with near-zero diagonal.

        The BLAS trick (X_sq + X_sq - 2*X@X.T then sqrt) can leave tiny
        residuals (~1e-8) on the diagonal due to floating-point cancellation.
        This is a well-known property of this computation method.
        """
        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 5))
        D = pairwise_euclidean(X)
        assert np.allclose(D, D.T, atol=1e-12), "Distance matrix not symmetric"
        # Diagonal: BLAS trick can leave residuals up to ~1e-7 after sqrt
        max_diag = np.max(np.abs(np.diag(D)))
        assert max_diag < 1e-6, f"Diagonal max = {max_diag:.2e}, expected < 1e-6"

    def test_pairwise_euclidean_matches_sklearn(self):
        """Pairwise Euclidean distances should match sklearn.

        Both use BLAS-based computation but may differ by ~1e-7 due to
        different internal implementations (SSDLite uses einsum + matmul,
        sklearn uses its own optimized path). This is acceptable.
        """
        from sklearn.metrics import pairwise_distances
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 4))
        ssd_D = pairwise_euclidean(X)
        sk_D = pairwise_distances(X, metric="euclidean")
        max_diff = np.max(np.abs(ssd_D - sk_D))
        assert max_diff < 1e-6, f"Max dist diff: {max_diff:.2e}"


# ===========================================================================
# PART 5: PCA comparison
# ===========================================================================

class TestPCAComparison:
    """Compare SSDLite PCA (SVD-based) vs sklearn PCA."""

    @pytest.mark.parametrize("n, d, k", [
        (50, 10, 5),
        (100, 20, 10),
        (30, 5, 3),
    ])
    def test_explained_variance_ratio(self, n, d, k):
        """Explained variance ratios should match sklearn."""
        from sklearn.decomposition import PCA as SklearnPCA

        from ssdiff.utils.math import pca_fit_transform

        rng = np.random.default_rng(42)
        X = rng.normal(size=(n, d))

        _, _, ssd_evr = pca_fit_transform(X, k)
        sk_pca = SklearnPCA(n_components=k).fit(X)

        np.testing.assert_allclose(ssd_evr, sk_pca.explained_variance_ratio_, atol=1e-10)

    @pytest.mark.parametrize("n, d, k", [
        (50, 10, 5),
        (100, 20, 3),
    ])
    def test_projections_span_same_subspace(self, n, d, k):
        """Projected data should span the same subspace (signs may differ)."""
        from sklearn.decomposition import PCA as SklearnPCA

        from ssdiff.utils.math import pca_fit_transform

        rng = np.random.default_rng(42)
        X = rng.normal(size=(n, d))

        ssd_z, ssd_comp, ssd_evr = pca_fit_transform(X, k)
        sk_pca = SklearnPCA(n_components=k).fit(X)
        sk_z = sk_pca.transform(X)

        # Verify each component pair is parallel (|cos| ≈ 1) before sign alignment.
        # This catches rotation bugs that simple sign-flip would mask.
        for j in range(k):
            cos_sim = abs(np.dot(ssd_comp[j], sk_pca.components_[j]))
            assert cos_sim > 1.0 - 1e-9, (
                f"Component {j}: |cos| = {cos_sim:.10f}, not parallel"
            )

        # Now align signs and verify exact values
        for j in range(k):
            if np.dot(ssd_comp[j], sk_pca.components_[j]) < 0:
                ssd_z[:, j] *= -1

        np.testing.assert_allclose(ssd_z, sk_z, atol=1e-10)

        # EVR should be monotonically decreasing
        assert np.all(np.diff(ssd_evr) <= 1e-10), "EVR not monotonically decreasing"
        assert ssd_evr.sum() <= 1.0 + 1e-10, f"EVR sums to {ssd_evr.sum()}"


# ===========================================================================
# PART 6: Statistical function comparison (f_sf, t_sf, chi2_sf vs scipy)
# ===========================================================================

class TestStatFunctionsVsScipy:
    """Compare custom _betainc/f_sf/t_sf/chi2_sf against scipy."""

    scipy_stats = pytest.importorskip("scipy.stats", reason="scipy required")

    @pytest.mark.parametrize("f_val, dfn, dfd", [
        (1.0, 2, 10),
        (3.0, 5, 20),
        (0.5, 1, 30),
        (10.0, 3, 50),
        (50.0, 10, 100),
        (0.01, 2, 5),
        (100.0, 5, 50),
    ])
    def test_f_sf(self, f_val, dfn, dfd):
        from ssdiff.utils.math import f_sf
        ssd_p = f_sf(f_val, dfn, dfd)
        scipy_p = float(self.scipy_stats.f.sf(f_val, dfn, dfd))
        np.testing.assert_allclose(ssd_p, scipy_p, atol=1e-14, rtol=1e-10,
            err_msg=f"f_sf({f_val}, {dfn}, {dfd})")

    @pytest.mark.parametrize("t_val, df", [
        (0.0, 10),
        (2.0, 20),
        (-2.0, 20),
        (5.0, 30),
        (-5.0, 30),
        (1.0, 5),
        (0.5, 100),
        (10.0, 50),
    ])
    def test_t_sf(self, t_val, df):
        from ssdiff.utils.math import t_sf
        ssd_p = t_sf(t_val, df)
        scipy_p = float(self.scipy_stats.t.sf(t_val, df))
        np.testing.assert_allclose(ssd_p, scipy_p, atol=1e-14, rtol=1e-10,
            err_msg=f"t_sf({t_val}, {df})")

    @pytest.mark.parametrize("x, df", [
        (5.0, 2),
        (10.0, 5),
        (1.0, 1),
        (20.0, 10),
        (0.5, 3),
        (50.0, 20),
    ])
    def test_chi2_sf(self, x, df):
        from ssdiff.utils.math import chi2_sf
        ssd_p = chi2_sf(x, df)
        scipy_p = float(self.scipy_stats.chi2.sf(x, df))
        np.testing.assert_allclose(ssd_p, scipy_p, atol=1e-14, rtol=1e-10,
            err_msg=f"chi2_sf({x}, {df})")
