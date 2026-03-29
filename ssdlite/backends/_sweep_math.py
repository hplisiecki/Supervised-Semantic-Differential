"""Shared math utilities for PCA-K sweep backends."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PCAKSelectionResult:
    """Result from PCA-K sweep selection.

    Attributes
    ----------
    best_k : int
        Selected PCA component count.
    df_joined : list[dict]
        Per-K metrics table with keys: PCA_K, var_explained, mean_coherence,
        mean_abs_cosb, aggregate, n_clusters, total_size,
        beta_delta_1_minus_cos, interp_auck, stab_auck_raw, joint_score.
    """
    best_k: int
    df_joined: list[dict] = field(default_factory=list)


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns NaN if either vector is zero."""
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def zscore_ignore_nan(x: np.ndarray) -> np.ndarray:
    """Z-score normalization ignoring NaNs."""
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-12:
        s = 1.0
    return (x - m) / s



def compute_auck(z: np.ndarray, radius: int) -> np.ndarray:
    """Area Under the Curve, K-neighborhood (AUCK) -- a smoothed local score.

    For each position, computes the mean of z-scores within a symmetric
    window of the given radius. Used to identify locally good regions in
    sweep curves.

    Parameters
    ----------
    z : ndarray, shape (n,)
        1-D array of z-scores.
    radius : int
        Half-window size for the symmetric neighborhood.

    Returns
    -------
    ndarray, shape (n,)
        AUCK values (same length as *z*). NaN where no finite values
        exist in the window.
    """
    z = np.asarray(z, dtype=float)
    n = len(z)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        w = z[lo:hi]
        w = w[np.isfinite(w)]
        if len(w) == 0:
            continue
        out[i] = float(np.nanmean(w))
    return out


def detrend_by_variance(var_explained_percent: np.ndarray, y: np.ndarray):
    """Remove the log-linear trend of *y* on variance explained.

    Fits ``y ~ a + b * log(var_explained_percent)`` via OLS and returns
    the predicted values, residuals, and regression coefficients.

    Parameters
    ----------
    var_explained_percent : ndarray, shape (n,)
        Percentage of variance explained at each K.
    y : ndarray, shape (n,)
        Metric values to detrend.

    Returns
    -------
    y_hat : ndarray, shape (n,)
        Predicted values from the OLS fit.
    residuals : ndarray, shape (n,)
        ``y - y_hat``.
    (a, b) : tuple of float
        Intercept and slope of the log-linear regression.
        NaN where inputs are invalid.
    """
    v = np.asarray(var_explained_percent, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(v) & np.isfinite(y) & (v > 0)
    v2 = v[mask]
    y2 = y[mask]

    if len(v2) < 3:
        y_hat = np.full_like(y, np.nan, dtype=float)
        resid = np.full_like(y, np.nan, dtype=float)
        return y_hat, resid, (float("nan"), float("nan"))

    X = np.column_stack([np.ones_like(v2), np.log(v2)])
    coef, _, _, _ = np.linalg.lstsq(X, y2, rcond=None)
    a, b = coef

    y_hat = np.full_like(y, np.nan, dtype=float)
    y_hat[mask] = a + b * np.log(v2)
    resid = y - y_hat
    return y_hat, resid, (float(a), float(b))


def overall_interpretability(
    clusters: list[dict], weight_by_size: bool = True
) -> dict:
    """Compute aggregate interpretability from a list of cluster dicts.

    Each dict must have keys: 'size', 'coherence', 'centroid_cos_beta'.
    """
    if not clusters:
        return dict(
            mean_coherence=np.nan,
            mean_abs_cosb=np.nan,
            aggregate=np.nan,
            n_clusters=0,
            total_size=0,
        )

    need_keys = {"size", "coherence", "centroid_cos_beta"}
    for i, c in enumerate(clusters):
        missing = need_keys - set(c.keys())
        if missing:
            raise RuntimeError(f"cluster[{i}] missing keys: {missing}")

    sizes = np.array([c["size"] for c in clusters], dtype=float)
    coherence = np.array([c["coherence"] for c in clusters], dtype=float)
    abs_cosb = np.abs(np.array([c["centroid_cos_beta"] for c in clusters], dtype=float))

    if weight_by_size:
        wsum = np.nansum(sizes)
        if wsum > 0:
            w = sizes / wsum
            mean_coh = float(np.nansum(coherence * w))
            mean_abs = float(np.nansum(abs_cosb * w))
        else:
            mean_coh = float(np.nanmean(coherence))
            mean_abs = float(np.nanmean(abs_cosb))
    else:
        mean_coh = float(np.nanmean(coherence))
        mean_abs = float(np.nanmean(abs_cosb))

    return dict(
        mean_coherence=mean_coh,
        mean_abs_cosb=mean_abs,
        aggregate=float(mean_coh * mean_abs),
        n_clusters=int(len(clusters)),
        total_size=int(np.nansum(sizes)),
    )
