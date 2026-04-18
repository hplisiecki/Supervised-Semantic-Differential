"""Group comparison backend — unified permutation tests for fit_groups().

Implements:
- Median split with tied-value equalization
- Small-group filtering
- Single-pass permutation test (omnibus + all pairwise from one shuffle loop)
- P-value correction (Holm, Bonferroni, FDR-BH, none)
- Cohen's d for each pairwise contrast
"""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np

from ssdiff.utils.math import unit_vector

# ---------------------------------------------------------------------------
# Median split
# ---------------------------------------------------------------------------

def median_split(y: np.ndarray, random_state: int = 2137) -> np.ndarray:
    """Split continuous y into "low"/"high" labels at the median.

    Docs at exactly the median are distributed between groups to equalize
    sizes, assigned randomly using *random_state*.

    Raises
    ------
    ValueError
        If all y values are identical.
    """
    if np.max(y) - np.min(y) == 0:
        raise ValueError(
            f"Cannot median-split: all y values are identical ({y[0]})"
        )

    med = float(np.median(y))
    labels = np.empty(len(y), dtype=object)
    labels[y < med] = "low"
    labels[y > med] = "high"

    tied = y == med
    n_tied = int(tied.sum())
    if n_tied > 0:
        n_below = int((y < med).sum())
        target = len(y) // 2
        n_low_needs = max(0, target - n_below)
        rng = np.random.default_rng(random_state)
        tied_indices = np.where(tied)[0]
        rng.shuffle(tied_indices)
        labels[tied_indices[:n_low_needs]] = "low"
        labels[tied_indices[n_low_needs:]] = "high"

    return labels


# ---------------------------------------------------------------------------
# Small-group filtering
# ---------------------------------------------------------------------------

MIN_GROUP_SIZE = 20


def filter_small_groups(
    x: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop groups with < MIN_GROUP_SIZE docs. Returns (x, groups, n_dropped).

    Operates on copies — never mutates inputs.
    """
    unique, counts = np.unique(groups, return_counts=True)
    small = unique[counts < MIN_GROUP_SIZE]
    n_dropped = 0
    if len(small) > 0:
        for label in small:
            n = int(counts[unique == label][0])
            warnings.warn(
                f"Group '{label}' dropped: only {n} docs (minimum {MIN_GROUP_SIZE})",
                stacklevel=2,
            )
        keep = np.isin(groups, small, invert=True)
        n_dropped = int((~keep).sum())
        x = x[keep]
        groups = groups[keep]
    return x, groups, n_dropped


# ---------------------------------------------------------------------------
# Centroid computation
# ---------------------------------------------------------------------------

def _compute_centroids_matrix(
    x: np.ndarray, group_idx: np.ndarray, G: int,
) -> np.ndarray:
    """Centroid computation via group-loop. Returns (G, D) unit-normed matrix.

    Loops over G groups (typically 2-10) rather than D dimensions (100-300),
    using vectorised mean per group for ~4-6x speedup in the permutation loop.
    """
    counts = np.bincount(group_idx, minlength=G)
    centroids = np.zeros((G, x.shape[1]), dtype=np.float64)
    for g in range(G):
        if counts[g] > 0:
            centroids[g] = x[group_idx == g].mean(axis=0)
    norms = np.maximum(
        np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12,
    )
    centroids /= norms
    return centroids


# ---------------------------------------------------------------------------
# P-value correction
# ---------------------------------------------------------------------------

_VALID_CORRECTIONS = ("holm", "bonferroni", "fdr_bh", "none")


def correct_pvalues(
    p_raw: np.ndarray, method: str = "holm",
) -> np.ndarray:
    """Apply multiple-comparison correction to raw p-values.

    Parameters
    ----------
    p_raw : 1-D array of raw p-values
    method : one of "holm", "bonferroni", "fdr_bh", "none"

    Returns
    -------
    1-D array of corrected p-values (capped at 1.0)
    """
    if method not in _VALID_CORRECTIONS:
        raise ValueError(
            f"Unknown correction {method!r}. "
            f"Choose from {_VALID_CORRECTIONS}"
        )
    p = np.asarray(p_raw, dtype=np.float64)
    m = len(p)
    if m == 0 or method == "none":
        return p.copy()

    if method == "bonferroni":
        return np.minimum(p * m, 1.0)

    if method == "holm":
        order = np.argsort(p)
        corrected = np.empty(m, dtype=np.float64)
        cummax = 0.0
        for rank, idx in enumerate(order):
            adj = p[idx] * (m - rank)
            cummax = max(cummax, adj)
            corrected[idx] = cummax
        return np.minimum(corrected, 1.0)

    # fdr_bh (Benjamini-Hochberg)
    order = np.argsort(p)
    corrected = np.empty(m, dtype=np.float64)
    cummin = 1.0
    for i in range(m - 1, -1, -1):
        idx = order[i]
        adj = p[idx] * m / (i + 1)
        cummin = min(cummin, adj)
        corrected[idx] = cummin
    return np.minimum(corrected, 1.0)


# ---------------------------------------------------------------------------
# Unified permutation test
# ---------------------------------------------------------------------------

def unified_permutation_test(
    x: np.ndarray,
    groups: np.ndarray,
    group_labels: list,
    *,
    n_perm: int = 5000,
    correction: str = "holm",
    random_state: int = 2137,
    verbose: bool = False,
) -> dict:
    """Run omnibus + all pairwise permutation tests in a single pass.

    Returns
    -------
    dict with keys:
        omnibus_T, omnibus_p,
        pairwise: dict[(g1,g2)] -> {
            beta_unit, T, p_raw, p_corrected, cohens_d,
            n_g1, n_g2, contrast_norm
        },
        group_labels, G, correction
    """
    G = len(group_labels)
    pairs = list(combinations(group_labels, 2))
    n_pairs = len(pairs)
    # Map labels to integer indices
    label_to_idx = {g: i for i, g in enumerate(group_labels)}
    group_idx = np.array(
        [label_to_idx[g] for g in groups], dtype=np.intp,
    )
    pair_int = np.array(
        [(label_to_idx[a], label_to_idx[b]) for a, b in pairs],
        dtype=np.intp,
    )

    # --- Observed statistics ---
    centroids_obs = _compute_centroids_matrix(x, group_idx, G)

    # Pairwise cosine distances
    T_pairwise_obs = np.empty(n_pairs, dtype=np.float64)
    for k, (ia, ib) in enumerate(pair_int):
        dot = float(np.clip(np.dot(centroids_obs[ia], centroids_obs[ib]), -1, 1))
        T_pairwise_obs[k] = 1.0 - dot

    # Omnibus T = mean of pairwise distances
    T_omnibus_obs = float(T_pairwise_obs.mean()) if n_pairs > 0 else 0.0

    # --- Permutation loop (single pass) ---
    from ssdiff.utils import _progress

    rng = np.random.default_rng(random_state)
    null_omnibus = np.empty(n_perm, dtype=np.float64)
    null_pairwise = np.empty((n_perm, n_pairs), dtype=np.float64)

    perm_group_idx = group_idx.copy()
    for p in _progress(range(n_perm), verbose=verbose, total=n_perm,
                       desc="Group permutation test"):
        rng.shuffle(perm_group_idx)
        centroids_perm = _compute_centroids_matrix(x, perm_group_idx, G)
        for k, (ia, ib) in enumerate(pair_int):
            dot = float(np.clip(
                np.dot(centroids_perm[ia], centroids_perm[ib]), -1, 1,
            ))
            null_pairwise[p, k] = 1.0 - dot
        null_omnibus[p] = float(null_pairwise[p].mean()) if n_pairs > 0 else 0.0

    # --- P-values ---
    from ssdiff.utils import _diagnostic
    from ssdiff.results.format import fmt_p

    omnibus_p = float((np.sum(null_omnibus >= T_omnibus_obs) + 1) / (n_perm + 1))
    _diagnostic(verbose, f"[groups] omnibus p={fmt_p(omnibus_p)} ({G} groups, {n_perm} perms)")

    p_raw_arr = np.empty(n_pairs, dtype=np.float64)
    for k in range(n_pairs):
        p_raw_arr[k] = float(
            (np.sum(null_pairwise[:, k] >= T_pairwise_obs[k]) + 1) / (n_perm + 1)
        )

    p_corrected_arr = correct_pvalues(p_raw_arr, method=correction)

    # --- Contrast vectors + Cohen's d ---
    centroids_dict = {g: centroids_obs[label_to_idx[g]] for g in group_labels}
    pairwise = {}
    for k, (g1, g2) in enumerate(pairs):
        contrast_raw = centroids_dict[g1] - centroids_dict[g2]
        contrast_norm = float(np.linalg.norm(contrast_raw))
        contrast_unit = unit_vector(contrast_raw)

        # Cohen's d from projections onto contrast vector
        proj = (x @ contrast_unit).ravel()
        proj_g1 = proj[groups == g1]
        proj_g2 = proj[groups == g2]
        dof = len(proj_g1) + len(proj_g2) - 2
        if dof > 0:
            pooled_std = np.sqrt(
                ((len(proj_g1) - 1) * np.var(proj_g1, ddof=1)
                 + (len(proj_g2) - 1) * np.var(proj_g2, ddof=1))
                / dof
            )
        else:
            pooled_std = 0.0
        cohens_d = float(
            (np.mean(proj_g1) - np.mean(proj_g2)) / max(pooled_std, 1e-12)
        )

        pairwise[(g1, g2)] = {
            "beta_unit": contrast_unit,
            "T": float(T_pairwise_obs[k]),
            "p_raw": float(p_raw_arr[k]),
            "p_corrected": float(p_corrected_arr[k]),
            "cohens_d": cohens_d,
            "n_g1": int((groups == g1).sum()),
            "n_g2": int((groups == g2).sum()),
            "contrast_norm": contrast_norm,
        }

    return {
        "omnibus_T": T_omnibus_obs,
        "omnibus_p": omnibus_p,
        "pairwise": pairwise,
        "group_labels": group_labels,
        "G": G,
        "correction": correction,
    }
