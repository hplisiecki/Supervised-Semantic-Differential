# ssdiff/utils/lexicon.py
"""Lexicon suggestion and coverage utilities (pandas-free)."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np

from .math import _categorical_mask

__all__: list[str] = []

# -------------------------
# Helpers: inputs & metrics
# -------------------------


def _as_float_array(y: Iterable) -> np.ndarray:
    """Standardize *y* to a 1-D float64 ndarray, coercing non-numeric to NaN."""
    arr = np.asarray(y, dtype=object)
    return np.array(
        [float(v) if v is not None else np.nan for v in arr],
        dtype=np.float64,
    )


def _to_unit_tokens(unit) -> list[str]:
    """Convert a single text unit to a flat token list.

    Handles str, list[str], list[list[str]] (profiles), and None.
    """
    if unit is None:
        return []
    if isinstance(unit, str):
        return unit.split()
    if isinstance(unit, (list, tuple)):
        if not unit:
            return []
        first = unit[0]
        if isinstance(first, str):
            return list(unit)
        if isinstance(first, (list, tuple)):
            out: list[str] = []
            for post in unit:
                if isinstance(post, (list, tuple)):
                    out.extend(t for t in post if isinstance(t, str))
                elif isinstance(post, str):
                    out.extend(post.split())
            return out
    return str(unit).split()


def _texts_to_token_lists(texts: Sequence) -> list[list[str]]:
    """Batch-convert text units to token lists.

    Fast path for homogeneous str or list[str] inputs;
    falls back to per-element :func:`_to_unit_tokens` for profiles
    and mixed inputs.
    """
    if not texts:
        return []
    first = texts[0]
    if isinstance(first, (list, tuple)):
        if first and isinstance(first[0], str):
            return [list(map(str, t)) for t in texts]
    elif isinstance(first, str):
        return [str(t).split() for t in texts]
    return [_to_unit_tokens(t) for t in texts]


def _token_sets(texts: Sequence) -> list[set[str]]:
    """Token lists → per-doc sets (unique presence)."""
    return [set(toks) for toks in _texts_to_token_lists(texts)]


def _quantile_bins(y: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """
    Return integer bin labels (0..k-1) via quantiles; fallback: median split.
    """
    arr = _as_float_array(y)
    try:
        # Compute quantile edges and digitize
        valid = arr[np.isfinite(arr)]
        edges = np.percentile(valid, np.linspace(0, 100, n_bins + 1))
        # Remove duplicate edges
        edges = np.unique(edges)
        if len(edges) < 2:
            raise ValueError("Not enough unique edges")
        # np.searchsorted gives bin indices; clip to valid range
        bins = np.searchsorted(edges[1:-1], arr, side="right")
        return bins
    except Exception:
        med = float(np.nanmedian(arr))
        return (arr > med).astype(int)


def _z(v: Iterable) -> np.ndarray:
    """Z-score to float np.ndarray with ddof=0; protects zero variance."""
    arr = _as_float_array(v)
    sd = np.std(arr, ddof=0)
    if not np.isfinite(sd) or sd < 1e-12:
        sd = 1.0
    mu = float(np.nanmean(arr))
    return (arr - mu) / sd


def _validate_var_type(var_type: str) -> None:
    """Raise ValueError if *var_type* is not 'continuous' or 'categorical'."""
    if var_type not in ("continuous", "categorical"):
        raise ValueError(
            f"var_type must be 'continuous' or 'categorical', got {var_type!r}"
        )


def _crosstab(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, list, list]:
    """
    Pure-numpy contingency table for two 1-D arrays.

    Returns (table, row_labels, col_labels) where table[i, j] is the count
    of co-occurrences of row_labels[i] and col_labels[j].
    """
    a = np.asarray(a)
    b = np.asarray(b)
    row_labels = sorted(set(a.tolist()))
    col_labels = sorted(set(b.tolist()))
    row_map = {v: i for i, v in enumerate(row_labels)}
    col_map = {v: j for j, v in enumerate(col_labels)}
    table = np.zeros((len(row_labels), len(col_labels)), dtype=np.float64)
    for ai, bi in zip(a, b):
        table[row_map[ai], col_map[bi]] += 1
    return table, row_labels, col_labels


def _cramers_v(presence: np.ndarray, groups: np.ndarray) -> float:
    """Cramér's V between binary presence (0/1) and group labels."""
    ct, row_labels, col_labels = _crosstab(presence, groups)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    n = ct.sum()
    row_sums = ct.sum(axis=1)
    col_sums = ct.sum(axis=0)
    expected = np.outer(row_sums, col_sums) / n
    nonzero = expected > 0
    chi2 = float(np.sum((ct[nonzero] - expected[nonzero]) ** 2 / expected[nonzero]))
    k = min(ct.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n * k > 0 else 0.0


def _chi2_pvalue(presence: np.ndarray, groups: np.ndarray) -> float:
    """P-value from chi-squared test of independence (pure numpy)."""
    from .math import chi2_sf

    ct, _, _ = _crosstab(presence, groups)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return float("nan")
    n = ct.sum()
    if n == 0:
        return float("nan")
    expected = np.outer(ct.sum(axis=1), ct.sum(axis=0)) / n
    nz = expected > 0
    # Yates' correction for 2x2 tables (matches scipy default)
    if ct.shape == (2, 2):
        diff = np.maximum(np.abs(ct - expected) - 0.5, 0.0)
        chi2 = float(np.sum(diff[nz] ** 2 / expected[nz]))
    else:
        chi2 = float(np.sum((ct[nz] - expected[nz]) ** 2 / expected[nz]))
    df = (ct.shape[0] - 1) * (ct.shape[1] - 1)
    if df < 1 or chi2 < 0:
        return float("nan")
    return chi2_sf(chi2, df)


def _pointbiserial_pvalue(presence: np.ndarray, y: np.ndarray) -> float:
    """P-value from point-biserial correlation (pure numpy)."""
    from .math import t_sf

    if np.std(presence) < 1e-12:
        return float("nan")
    n = len(presence)
    if n < 3:
        return float("nan")
    r = float(np.corrcoef(presence, y)[0, 1])
    if not np.isfinite(r):
        return float("nan")
    if abs(r) >= 1.0:
        return 0.0
    df = n - 2
    t = r * np.sqrt(df / (1.0 - r * r))
    # two-tailed p-value
    return 2.0 * t_sf(abs(t), df)


def _effect_direction(
    presence: np.ndarray,
    y,
    categorical: bool,
) -> str:
    """Return 'positive', 'negative', or 'none' for the association direction."""
    if categorical:
        groups = np.asarray(y, dtype=object)
        group_labels = sorted(set(groups))
        if len(group_labels) < 2:
            return "none"
        covs = {}
        for g in group_labels:
            idx = np.where(groups == g)[0]
            covs[g] = float(presence[idx].mean()) if len(idx) else 0.0
        vals = list(covs.values())
        if max(vals) - min(vals) < 1e-9:
            return "none"
        # Positive = token more present in later (higher-sorted) group
        first, last = vals[0], vals[-1]
        return "positive" if last > first else "negative"
    else:
        y_arr = _as_float_array(y)
        if np.std(presence) < 1e-12:
            return "none"
        c = float(np.corrcoef(presence.astype(float), y_arr)[0, 1])
        if not np.isfinite(c) or abs(c) < 1e-9:
            return "none"
        return "positive" if c > 0 else "negative"


def _rank_for_token_stats(
    presence_vec: np.ndarray,
    y: np.ndarray,
    n_bins: int = 4,
    corr_cap: float = 0.30,
    categorical: bool = False,
) -> tuple[float, float, float, float]:
    """
    presence_vec: 0/1 per document
    Returns: (cov_all, cov_bal, corr, rank)
    rank = balanced_coverage * (1 - min(1, |corr|/corr_cap))

    When categorical=True, bins are group labels and corr is Cramér's V.
    """
    presence_vec = presence_vec.astype(float)
    cov_all = float(np.mean(presence_vec)) if len(presence_vec) else 0.0

    if categorical:
        groups = np.asarray(y, dtype=object)
        cov_per_group = []
        for g in sorted(set(groups)):
            idx = np.where(groups == g)[0]
            cov_per_group.append(
                float(np.mean(presence_vec[idx])) if len(idx) else 0.0
            )
        cov_bal = float(np.mean(cov_per_group)) if cov_per_group else 0.0
        corr = _cramers_v(presence_vec.astype(int), groups)
    else:
        bins = _quantile_bins(y, n_bins=n_bins)
        cov_per_bin = []
        for b in sorted(np.unique(bins)):
            idx = np.where(bins == b)[0]
            cov_per_bin.append(
                float(np.mean(presence_vec[idx])) if len(idx) else 0.0
            )
        cov_bal = float(np.mean(cov_per_bin)) if cov_per_bin else 0.0
        y_std = _z(y)
        if np.std(presence_vec) < 1e-12:
            corr = 0.0
        else:
            c = float(np.corrcoef(presence_vec, y_std)[0, 1])
            corr = c if np.isfinite(c) else 0.0

    pen = min(1.0, abs(corr) / corr_cap)
    rank = cov_bal * (1.0 - pen)
    return cov_all, cov_bal, corr, rank


# -------------------------
# Shared helpers (used by Corpus methods)
# -------------------------


def _filter_y(
    docs: list[list[str]], y, *, var_type: str = "continuous",
) -> tuple[list[list[str]], np.ndarray]:
    """Filter docs and y for valid entries. Returns (docs, y_clean)."""
    _validate_var_type(var_type)
    if var_type == "categorical":
        y_arr = np.asarray(y, dtype=object)
        mask = _categorical_mask(y_arr)
        if not mask.all():
            docs = [docs[i] for i in range(len(docs)) if mask[i]]
            y_arr = y_arr[mask]
        return docs, y_arr
    else:
        y_arr = _as_float_array(y)
        mask = np.isfinite(y_arr)
        if not mask.all():
            docs = [docs[i] for i in range(len(docs)) if mask[i]]
            y_arr = y_arr[mask]
        return docs, y_arr


def _rank_tokens(
    token_sets: list[set[str]],
    y: np.ndarray,
    *,
    top_k: int = 150,
    min_docs: int = 5,
    n_bins: int = 4,
    corr_cap: float = 0.30,
    var_type: str = "continuous",
) -> list[dict]:
    """Rank tokens by balanced coverage with association penalty.

    Parameters
    ----------
    token_sets : list of sets of str
        Per-doc unique token sets.
    y : ndarray
        Outcome variable (already cleaned).
    top_k, min_docs, n_bins, corr_cap, var_type :
        Same as ``Corpus.suggest_lexicon``.

    Returns
    -------
    list[dict]
        Dicts with keys ``token``, ``freq``, ``cov_all``, ``cov_bal``,
        ``corr``, ``rank``, ``pvalue``, ``direction``, sorted by
        descending rank, at most *top_k*.
    """
    is_categorical = var_type == "categorical"

    df_counts: Counter = Counter()
    for ts in token_sets:
        df_counts.update(ts)
    vocab = [t for t, c in df_counts.items() if c >= min_docs]
    if not vocab:
        return []

    rows: list[dict] = []
    for t in vocab:
        pres = np.fromiter(
            (1 if t in ts else 0 for ts in token_sets),
            dtype=np.int8,
            count=len(token_sets),
        )
        cov_all, cov_bal, corr, rank = _rank_for_token_stats(
            pres,
            y,
            n_bins=n_bins,
            corr_cap=corr_cap,
            categorical=is_categorical,
        )
        if is_categorical:
            pval = _chi2_pvalue(pres.astype(int), y)
            direction = _effect_direction(pres, y, categorical=True)
        else:
            pval = _pointbiserial_pvalue(pres.astype(float), y)
            direction = _effect_direction(pres, y, categorical=False)
        rows.append(dict(
            token=t,
            freq=int(pres.sum()),
            cov_all=cov_all,
            cov_bal=cov_bal,
            corr=corr,
            rank=rank,
            pvalue=pval,
            direction=direction,
        ))

    rows.sort(key=lambda r: (-r["rank"], -r["cov_bal"], -r["freq"]))
    return rows[:top_k]

