"""Pure-numpy replacements for sklearn StandardScaler, PCA, KMeans, silhouette_score."""

from __future__ import annotations

import numpy as np


def unit_vector(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return unit vector in direction of v, or zero vector if ||v|| < eps."""
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score standardize columns. Returns (X_standardized, mean, scale)."""
    mean = X.mean(axis=0, dtype=np.float64)
    scale = X.std(axis=0, dtype=np.float64, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    Xs = (X - mean) / scale
    return Xs, mean, scale


def pca_fit_transform(
    X: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA via full SVD (matches sklearn svd_solver='full').

    Parameters
    ----------
    X : (n, d) array.  Centering is applied internally.
    n_components : number of components to keep.

    Returns
    -------
    z : (n, n_components) projected data
    components : (n_components, d) principal axes
    explained_variance_ratio : (n_components,) fraction of variance per component
    """
    n = X.shape[0]
    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    components = Vt[:n_components]
    z = Xc @ components.T

    explained_var = (S**2) / (n - 1)
    total_var = explained_var.sum()
    evr = (
        explained_var[:n_components] / total_var
        if total_var > 0
        else np.zeros(n_components)
    )

    return z, components, evr


def l2_normalize_rows_inplace(V: np.ndarray) -> None:
    """L2-normalize each row of *V* in-place."""
    norms = np.sqrt(np.einsum("ij,ij->i", V, V))[:, None]
    np.maximum(norms, 1e-12, out=norms)
    if norms.dtype != V.dtype:
        norms = norms.astype(V.dtype, copy=False)
    V /= norms


# ---------------------------------------------------------------------------
# KMeans
# ---------------------------------------------------------------------------


def _sq_dists(X: np.ndarray, C: np.ndarray, X_sq: np.ndarray | None = None) -> np.ndarray:
    """Squared Euclidean distances between rows of X (n,d) and C (k,d)."""
    if X_sq is None:
        X_sq = np.einsum("ij,ij->i", X, X)
    C_sq = np.einsum("ij,ij->i", C, C)
    D = X_sq[:, None] + C_sq[None, :] - 2.0 * (X @ C.T)
    np.maximum(D, 0.0, out=D)
    return D


def _kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """K-means++ initialization with multi-candidate sampling."""
    n = X.shape[0]
    n_local_trials = 2 + int(np.log(k))
    first = rng.integers(n)
    centers = [first]
    min_dists = np.full(n, np.inf, dtype=np.float64)

    for _ in range(1, k):
        last = X[centers[-1]]
        d = np.sum((X - last) ** 2, axis=1)
        np.minimum(min_dists, d, out=min_dists)

        total = min_dists.sum()
        if total == 0:
            idx = rng.integers(n)
        else:
            probs = min_dists / total
            candidates = rng.choice(n, size=n_local_trials, p=probs, replace=True)
            best_idx = candidates[0]
            best_potential = np.inf
            for c in candidates:
                potential = np.minimum(min_dists, np.sum((X - X[c]) ** 2, axis=1)).sum()
                if potential < best_potential:
                    best_potential = potential
                    best_idx = c
            idx = best_idx
        centers.append(idx)

    return X[centers].copy()


def _update_centers(X: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute new cluster centers as mean of assigned points."""
    d = X.shape[1]
    new_centers = np.zeros((k, d), dtype=np.float64)
    counts = np.bincount(labels, minlength=k)
    for j in range(k):
        if counts[j] > 0:
            new_centers[j] = X[labels == j].mean(axis=0)
    return new_centers, counts


def kmeans(
    X: np.ndarray,
    k: int,
    *,
    random_state: int | None = None,
    max_iter: int = 300,
    n_init: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """K-Means clustering via Lloyd's algorithm with k-means++ initialization.

    Pure-numpy implementation (no scikit-learn dependency).  Empty clusters
    are handled by re-seeding from the farthest point.

    Parameters
    ----------
    X : (n, d) ndarray
        Data matrix with one sample per row.
    k : int
        Number of clusters.
    random_state : int or None, default None
        Seed for the random number generator (reproducibility).
    max_iter : int, default 300
        Maximum number of Lloyd iterations per run.
    n_init : int, default 1
        Number of independent runs with different initializations;
        the result with the lowest inertia is returned.

    Returns
    -------
    labels : (n,) ndarray of int
        Cluster assignment for each sample.
    centers : (k, d) ndarray
        Final cluster centroids.
    inertia : float
        Sum of squared distances from each sample to its assigned centroid
        (lower is better).
    """
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    if k > n:
        raise ValueError(f"Cannot request k={k} clusters from n={n} samples.")
    best_inertia = np.inf
    best_labels = np.zeros(n, dtype=np.intp)
    best_centers = np.zeros((k, d), dtype=np.float64)
    X_sq = np.einsum("ij,ij->i", X, X)

    for _ in range(n_init):
        centers = _kmeans_plus_plus(X, k, rng)
        prev_labels = np.full(n, -1, dtype=np.intp)

        for _ in range(max_iter):
            dists = _sq_dists(X, centers, X_sq)
            labels = np.argmin(dists, axis=1)

            if np.array_equal(labels, prev_labels):
                break
            prev_labels = labels

            new_centers, counts = _update_centers(X, labels, k)

            empty = counts == 0
            if empty.any():
                min_d = np.min(dists, axis=1)
                for j in np.where(empty)[0]:
                    farthest = np.argmax(min_d)
                    new_centers[j] = X[farthest]
                    min_d[farthest] = -1.0

            centers = new_centers

        dists = _sq_dists(X, centers, X_sq)
        labels = np.argmin(dists, axis=1)
        inertia = float(np.min(dists, axis=1).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_centers, best_inertia


# ---------------------------------------------------------------------------
# Silhouette score
# ---------------------------------------------------------------------------


def pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix (n, n) via BLAS trick."""
    X_sq = np.einsum("ij,ij->i", X, X)
    D_sq = X_sq[:, None] + X_sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(D_sq, 0.0, out=D_sq)
    return np.sqrt(D_sq)


def _silhouette_from_dists(dists: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient from precomputed distance matrix."""
    n = len(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

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
    return float(sil.mean())


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient (Euclidean)."""
    return _silhouette_from_dists(pairwise_euclidean(X), labels)


def kmeans_auto_k(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 5,
    *,
    random_state: int | None = None,
    n_init: int = 1,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """K-Means with automatic *k* selection via silhouette score.

    Runs ``kmeans`` for every *k* in ``[k_min, k_max]`` and picks the
    *k* that maximizes the mean silhouette coefficient (Euclidean).

    Parameters
    ----------
    X : (n, d) ndarray
        Data matrix with one sample per row.
    k_min : int, default 2
        Minimum number of clusters to evaluate.
    k_max : int, default 5
        Maximum number of clusters to evaluate (clamped to ``n - 1``).
    random_state : int or None, default None
        Seed for the random number generator (reproducibility).
    n_init : int, default 1
        Number of independent K-Means runs per candidate *k*.

    Returns
    -------
    labels : (n,) ndarray of int
        Cluster assignments for the best *k*.
    centers : (best_k, d) ndarray
        Cluster centroids for the best *k*.
    inertia : float
        Sum of squared distances for the best *k*.
    best_k : int
        Selected number of clusters.
    """
    upper = min(k_max, max(k_min, X.shape[0] - 1))
    pw_dists = pairwise_euclidean(X)

    best_k = max(2, k_min)
    best_s = -1.0
    best_labels = None
    best_centers = None
    best_inertia = np.inf

    for k in range(max(2, k_min), max(2, upper) + 1):
        labels, centers, inertia = kmeans(X, k=k, random_state=random_state, n_init=n_init)
        if len(set(labels)) <= 1 or np.max(np.bincount(labels)) <= 1:
            continue
        s = _silhouette_from_dists(pw_dists, labels)
        if s > best_s:
            best_s = s
            best_k = k
            best_labels = labels
            best_centers = centers
            best_inertia = inertia

    if best_labels is None:
        best_labels, best_centers, best_inertia = kmeans(
            X, k=best_k, random_state=random_state,
        )

    return best_labels, best_centers, best_inertia, best_k


# ---------------------------------------------------------------------------
# F-distribution survival function (replaces scipy.stats.f.cdf)
# ---------------------------------------------------------------------------


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b) via continued fraction.

    Implements the Lentz continued-fraction algorithm (Numerical Recipes,
    Press et al.).  Uses the identity I_x(a, b) = 1 - I_{1-x}(b, a) when
    x > (a + 1) / (a + b + 2) to ensure the continued fraction converges
    from the favorable side.  Converges to approximately 14 significant
    digits (EPS = 1e-14).

    Used internally by ``f_sf`` and ``t_sf`` for p-value computation without
    a SciPy dependency.
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    if x > (a + 1) / (a + b + 2):
        return 1.0 - _betainc(b, a, 1.0 - x)

    from math import lgamma, exp, log

    lbeta_ab = lgamma(a) + lgamma(b) - lgamma(a + b)
    front = exp(a * log(x) + b * log(1.0 - x) - lbeta_ab) / a

    TINY = 1e-30
    EPS = 1e-14

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < TINY:
        d = TINY
    d = 1.0 / d
    h = d

    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < TINY:
            d = TINY
        c = 1.0 + aa / c
        if abs(c) < TINY:
            c = TINY
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < TINY:
            d = TINY
        c = 1.0 + aa / c
        if abs(c) < TINY:
            c = TINY
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < EPS:
            break

    return front * h


def _gammainc_lower(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x) = gamma(a,x) / Gamma(a).

    Uses the series representation when x < a + 1, and the continued-
    fraction representation for Q(a, x) = 1 - P(a, x) otherwise.
    Converges to approximately 12 significant digits.

    Used internally by ``chi2_sf`` for p-value computation without a
    SciPy dependency.
    """
    if x <= 0:
        return 0.0

    from math import lgamma, exp, log

    if x < a + 1:
        # series representation
        ap = a
        s = 1.0 / a
        d = s
        for _ in range(200):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-12:
                break
        return s * exp(-x + a * log(x) - lgamma(a))

    # continued fraction for Q(a, x), return 1 - Q
    _FPMIN = 1e-30
    b_val = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b_val
    h = d
    for i in range(1, 200):
        an = -i * (i - a)
        b_val += 2.0
        d = an * d + b_val
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b_val + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return 1.0 - exp(-x + a * log(x) - lgamma(a)) * h


def chi2_sf(x: float, df: int) -> float:
    """Survival function (1 - CDF) of the chi-squared distribution."""
    if x <= 0 or df <= 0:
        return 1.0
    return 1.0 - _gammainc_lower(df / 2.0, x / 2.0)


def f_sf(f_val: float, dfn: float, dfd: float) -> float:
    """Survival function (1 - CDF) of the F-distribution."""
    if f_val <= 0 or dfn <= 0 or dfd <= 0:
        return 1.0
    x = dfn * f_val / (dfn * f_val + dfd)
    return 1.0 - _betainc(dfn / 2.0, dfd / 2.0, x)


def t_sf(t_val: float, df: float) -> float:
    """Survival function P(T > t) for Student's t-distribution."""
    if df <= 0:
        return float("nan")
    if t_val == 0.0:
        return 0.5
    x = df / (df + t_val * t_val)
    half_p = 0.5 * _betainc(df / 2.0, 0.5, x)
    return half_p if t_val > 0 else 1.0 - half_p
