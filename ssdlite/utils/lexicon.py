# ssdlite/utils/lexicon.py
"""Lexicon suggestion and coverage utilities (pandas-free)."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "suggest_lexicon",
    "token_presence_stats",
    "coverage_by_lexicon",
]

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


def _texts_to_token_lists(texts: Sequence) -> list[list[str]]:
    """
    Normalize texts into token lists:
      - list[list[str]] → passthrough
      - list[str]       → split on whitespace
    """
    if not texts:
        return []
    first = texts[0]
    if isinstance(first, (list, tuple)):
        return [list(map(str, t)) for t in texts]
    return [str(t).split() for t in texts]


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
    if var_type not in ("continuous", "categorical"):
        raise ValueError(
            f"var_type must be 'continuous' or 'categorical', got {var_type!r}"
        )


def _categorical_mask(y) -> np.ndarray:
    """Boolean mask: True for valid categorical entries (not None/NaN/empty)."""
    arr = np.asarray(y, dtype=object)
    return np.array(
        [
            g is not None
            and g != ""
            and (not isinstance(g, float) or np.isfinite(g))
            for g in arr
        ],
        dtype=bool,
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
# Public API
# -------------------------


def suggest_lexicon(
    df_or_texts,
    text_col: str | None = None,
    score_col: str | None = None,
    *,
    top_k: int = 150,
    min_docs: int = 5,
    n_bins: int = 4,
    corr_cap: float = 0.30,
    var_type: str = "continuous",
) -> list[str]:
    """
    Suggest candidate tokens ranked by coverage with a mild penalty for strong
    association with *y*.

    Parameters
    ----------
    df_or_texts : dict-of-lists | Sequence[str] | Sequence[list[str]]
        If a dict with string keys (column-oriented table), also pass
        *text_col* and *score_col*.
        Otherwise pass ``(texts, y)`` as a tuple where *texts* is
        ``list[str]`` or ``list[list[str]]``.
    text_col : str | None
        Key with preprocessed text (space-separated) if dict provided.
    score_col : str | None
        Key with outcome variable if dict provided (numeric for continuous,
        any hashable for categorical).
    var_type : str
        ``'continuous'`` (default) for numeric outcomes or ``'categorical'``
        for group labels.

    Returns
    -------
    list[str]
        Token strings sorted by descending rank, at most *top_k*.
    """
    _validate_var_type(var_type)
    is_categorical = var_type == "categorical"

    # Allow passing a tuple (texts, y) directly
    if isinstance(df_or_texts, dict):
        if not text_col or not score_col:
            raise ValueError(
                "Provide text_col and score_col when using a dict table."
            )
        raw_texts = df_or_texts[text_col]
        raw_y = df_or_texts[score_col]
        # Apply fillna equivalent and cast to str
        raw_texts = [str(t) if t is not None else "" for t in raw_texts]
        if is_categorical:
            y = np.asarray(raw_y, dtype=object)
            mask = _categorical_mask(y)
            texts = _texts_to_token_lists(
                [raw_texts[i] for i in range(len(raw_texts)) if mask[i]]
            )
            y = y[mask]
        else:
            y = _as_float_array(raw_y)
            mask = np.isfinite(y)
            texts = _texts_to_token_lists(
                [raw_texts[i] for i in range(len(raw_texts)) if mask[i]]
            )
            y = y[mask]
    elif isinstance(df_or_texts, tuple) and len(df_or_texts) == 2:
        texts, y = df_or_texts
        texts = _texts_to_token_lists(texts)
        if is_categorical:
            y = np.asarray(y, dtype=object)
            mask = _categorical_mask(y)
            if not mask.all():
                texts = [texts[i] for i in range(len(texts)) if mask[i]]
                y = y[mask]
        else:
            y = _as_float_array(y)
            mask = np.isfinite(y)
            if not mask.all():
                texts = [texts[i] for i in range(len(texts)) if mask[i]]
                y = y[mask]
    else:
        raise ValueError(
            "Pass either a dict table with text_col/score_col, "
            "or a (texts, y) tuple."
        )

    # Build doc-frequency counts
    token_sets = _token_sets(texts)
    df_counts: Counter = Counter()
    for ts in token_sets:
        df_counts.update(ts)
    vocab = [t for t, c in df_counts.items() if c >= min_docs]
    if not vocab:
        return []

    rows: list[tuple[str, float, float, int]] = []
    for t in vocab:
        pres = np.fromiter(
            (1 if t in ts else 0 for ts in token_sets),
            dtype=np.int8,
            count=len(token_sets),
        )
        _cov_all, cov_bal, _corr, rank = _rank_for_token_stats(
            pres,
            y,
            n_bins=n_bins,
            corr_cap=corr_cap,
            categorical=is_categorical,
        )
        docs = int(pres.sum())
        rows.append((t, rank, cov_bal, docs))

    # Sort by (rank desc, cov_bal desc, docs desc)
    rows.sort(key=lambda r: (-r[1], -r[2], -r[3]))
    return [r[0] for r in rows[:top_k]]


def token_presence_stats(
    texts: Iterable[object],
    y: np.ndarray,
    token: str,
    *,
    n_bins: int = 4,
    corr_cap: float = 0.30,
    verbose: bool = False,
    var_type: str = "continuous",
) -> list[dict]:
    """
    Compute presence statistics for a single token across documents.

    Returns
    -------
    list[dict]
        A single-element list.  Each dict has keys:
        ``token``, ``frequency``, ``association``, ``pvalue``,
        ``effect_direction``.

    Notes
    -----
    *pvalue* is computed with pure-numpy implementations of the
    incomplete beta / gamma functions (no scipy required).

    When *var_type='categorical'*, *association* is Cramér's V and *pvalue*
    comes from χ² test of independence.
    When *var_type='continuous'*, *association* is Pearson correlation
    between presence and z-scored outcome, and *pvalue* comes from a
    point-biserial correlation test.
    """
    _validate_var_type(var_type)
    is_categorical = var_type == "categorical"

    def _doc_token_set(doc) -> set[str]:
        if isinstance(doc, str):
            return set(doc.split())
        toks: list[str] = []
        stack = [doc]
        while stack:
            cur = stack.pop()
            if cur is None:
                continue
            if isinstance(cur, (list, tuple)):
                stack.extend(cur)
            elif isinstance(cur, set):
                stack.extend(list(cur))
            elif isinstance(cur, (str, bytes)):
                toks.append(cur.decode() if isinstance(cur, bytes) else cur)
        return set(toks)

    token = str(token)
    texts_list = list(texts)

    if is_categorical:
        y_arr = np.asarray(y, dtype=object)
        mask = _categorical_mask(y_arr)
        if not mask.all():
            texts_list = [texts_list[i] for i in range(len(texts_list)) if mask[i]]
            y_arr = y_arr[mask]
        if len(texts_list) != len(y_arr):
            raise ValueError(
                f"Length mismatch: texts={len(texts_list)} vs y={len(y_arr)}"
            )

        pres = np.fromiter(
            (1 if token in _doc_token_set(doc) else 0 for doc in texts_list),
            dtype=np.int8,
            count=len(texts_list),
        )

        _cov_all, _cov_bal, corr, _rank = _rank_for_token_stats(
            pres, y_arr, n_bins=n_bins, corr_cap=corr_cap, categorical=True,
        )

        frequency = int(pres.sum())
        association = float(corr)
        pvalue = _chi2_pvalue(pres.astype(int), y_arr)
        direction = _effect_direction(pres, y_arr, categorical=True)

    else:
        y_arr = _as_float_array(y)
        mask = np.isfinite(y_arr)
        if not mask.all():
            texts_list = [texts_list[i] for i in range(len(texts_list)) if mask[i]]
            y_arr = y_arr[mask]
        if len(texts_list) != len(y_arr):
            raise ValueError(
                f"Length mismatch: texts={len(texts_list)} vs y={len(y_arr)}"
            )

        pres = np.fromiter(
            (1 if token in _doc_token_set(doc) else 0 for doc in texts_list),
            dtype=np.int8,
            count=len(texts_list),
        )

        _cov_all, _cov_bal, corr, _rank = _rank_for_token_stats(
            pres, y_arr, n_bins=n_bins, corr_cap=corr_cap, categorical=False,
        )

        frequency = int(pres.sum())
        association = float(corr)
        pvalue = _pointbiserial_pvalue(pres.astype(float), y_arr)
        direction = _effect_direction(pres, y_arr, categorical=False)

    out = dict(
        token=token,
        frequency=frequency,
        association=association,
        pvalue=pvalue,
        effect_direction=direction,
    )

    if verbose:
        print(
            f"[token] '{token}': "
            f"freq={frequency} | assoc={association:.3f} | "
            f"p={pvalue:.4g} | dir={direction}"
        )

    return [out]


def coverage_by_lexicon(
    df_or_texts,
    text_col: str | None = None,
    score_col: str | None = None,
    lexicon: Iterable[str] = (),
    *,
    n_bins: int = 4,
    verbose: bool = False,
    var_type: str = "continuous",
) -> tuple[dict, list[dict]]:
    """
    Summarize coverage for a given lexicon.

    Accepts:
      - dict table + (text_col, score_col) where text_col values may be:
          * raw strings
          * token lists: list[str]
          * profiles: list[list[str]]  (multiple independent posts per unit)
      - Tuple (texts, y), where texts is a Sequence of the same forms above.

    Parameters
    ----------
    var_type : str
        ``'continuous'`` (default) or ``'categorical'``.

    Returns
    -------
    summary : dict
        Keys: docs_any, cov_all, q1, q4, corr_any, hits_mean, hits_median,
        types_mean, types_median [, group_cov].
    per_token : list[dict]
        Each dict has keys: ``token``, ``frequency``, ``association``,
        ``pvalue``, ``effect_direction``.
    """
    _validate_var_type(var_type)
    is_categorical = var_type == "categorical"

    # --- small internal adapters (robust to nested inputs) --------------------

    def _to_unit_tokens(unit) -> list[str]:
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
                        out.extend([t for t in post if isinstance(t, str)])
                    elif isinstance(post, str):
                        out.extend(post.split())
                return out
        return str(unit).split()

    def _local_texts_to_token_lists(texts_like) -> list[list[str]]:
        return [_to_unit_tokens(u) for u in texts_like]

    def _local_token_sets(text_lists: list[list[str]]) -> list[set[str]]:
        return [set(toks) if toks else set() for toks in text_lists]

    # --- coerce inputs --------------------------------------------------------
    if isinstance(df_or_texts, dict):
        if not text_col or not score_col:
            raise ValueError(
                "Provide text_col and score_col when using a dict table."
            )
        s = df_or_texts[text_col]
        if is_categorical:
            y = np.asarray(df_or_texts[score_col], dtype=object)
            mask = _categorical_mask(y)
            s = [s[i] for i in range(len(s)) if mask[i]]
            y = y[mask]
            texts = _local_texts_to_token_lists(s)
        else:
            y = _as_float_array(df_or_texts[score_col])
            mask = np.isfinite(y)
            s = [s[i] for i in range(len(s)) if mask[i]]
            y = y[mask]
            texts = _local_texts_to_token_lists(s)
    elif isinstance(df_or_texts, tuple) and len(df_or_texts) == 2:
        texts, y = df_or_texts
        texts = _local_texts_to_token_lists(texts)
        if is_categorical:
            y = np.asarray(y, dtype=object)
            mask = _categorical_mask(y)
            if not mask.all():
                texts = [texts[i] for i in range(len(texts)) if mask[i]]
                y = y[mask]
        else:
            y = _as_float_array(y)
            mask = np.isfinite(y)
            if not mask.all():
                texts = [texts[i] for i in range(len(texts)) if mask[i]]
                y = y[mask]
    else:
        raise ValueError(
            "Pass either a dict table with text_col/score_col, "
            "or a (texts, y) tuple."
        )

    # guard: empty after filtering
    if len(texts) == 0 or len(y) == 0:
        summary = dict(
            docs_any=0,
            cov_all=0.0,
            q1=0.0,
            q4=0.0,
            corr_any=0.0,
            hits_mean=0.0,
            hits_median=0.0,
            types_mean=0.0,
            types_median=0.0,
        )
        if is_categorical:
            summary["group_cov"] = {}
        return summary, []

    # --- prep features --------------------------------------------------------
    lex = [str(w) for w in lexicon]
    token_sets = _local_token_sets(texts)

    # presence of ANY lexicon word per unit
    pres_any = np.fromiter(
        (1 if any((w in ts) for w in lex) else 0 for ts in token_sets),
        dtype=np.int8,
        count=len(token_sets),
    )

    overall = float(pres_any.mean()) if len(pres_any) else 0.0
    docs_any = int(pres_any.sum())

    if is_categorical:
        groups = y  # already np.ndarray of object dtype
        group_labels = sorted(set(groups))

        group_cov_any: dict = {}
        for g in group_labels:
            idx = np.where(groups == g)[0]
            group_cov_any[g] = float(pres_any[idx].mean()) if len(idx) else 0.0
        q1 = min(group_cov_any.values()) if group_cov_any else 0.0
        q4 = max(group_cov_any.values()) if group_cov_any else 0.0

        corr_any = _cramers_v(pres_any.astype(int), groups)

        # per-token stats
        rows: list[dict] = []
        for w in lex:
            pres = np.fromiter(
                ((1 if w in ts else 0) for ts in token_sets),
                dtype=np.int8,
                count=len(token_sets),
            )
            assoc = _cramers_v(pres.astype(int), groups)
            pval = _chi2_pvalue(pres.astype(int), groups)
            direction = _effect_direction(pres, groups, categorical=True)
            rows.append(
                dict(
                    token=w,
                    frequency=int(pres.sum()),
                    association=assoc,
                    pvalue=pval,
                    effect_direction=direction,
                )
            )
    else:
        bins = _quantile_bins(y, n_bins=n_bins)
        low_idx = np.where(bins == bins.min())[0]
        high_idx = np.where(bins == bins.max())[0]

        y_std = _z(y)
        if pres_any.std() < 1e-12:
            corr_any = 0.0
        else:
            c = float(np.corrcoef(pres_any, y_std)[0, 1])
            corr_any = c if np.isfinite(c) else 0.0

        q1 = float(pres_any[low_idx].mean()) if len(low_idx) else 0.0
        q4 = float(pres_any[high_idx].mean()) if len(high_idx) else 0.0

        # per-token stats
        rows = []
        for w in lex:
            pres = np.fromiter(
                ((1 if w in ts else 0) for ts in token_sets),
                dtype=np.int8,
                count=len(token_sets),
            )
            assoc = (
                float(np.corrcoef(pres, y_std)[0, 1])
                if pres.std() > 0
                else 0.0
            )
            pval = _pointbiserial_pvalue(pres.astype(float), y)
            direction = _effect_direction(pres, y, categorical=False)
            rows.append(
                dict(
                    token=w,
                    frequency=int(pres.sum()),
                    association=assoc,
                    pvalue=pval,
                    effect_direction=direction,
                )
            )

    # Sort per-token by frequency descending
    rows.sort(key=lambda r: (-r["frequency"],))

    # --- whole-profile lexicon frequency stats (DV-agnostic) ------------------
    lex_set = set(lex)
    hits_per_unit = np.array(
        [sum(1 for t in toks if t in lex_set) for toks in texts], dtype=np.int32
    )
    types_per_unit = np.array(
        [len(set(toks) & lex_set) for toks in texts], dtype=np.int32
    )

    hits_mean = float(hits_per_unit.mean()) if len(hits_per_unit) else 0.0
    hits_median = float(np.median(hits_per_unit)) if len(hits_per_unit) else 0.0
    types_mean = float(types_per_unit.mean()) if len(types_per_unit) else 0.0
    types_median = float(np.median(types_per_unit)) if len(types_per_unit) else 0.0

    summary = dict(
        docs_any=docs_any,
        cov_all=overall,
        q1=q1,
        q4=q4,
        corr_any=corr_any,
        hits_mean=hits_mean,
        hits_median=hits_median,
        types_mean=types_mean,
        types_median=types_median,
    )
    if is_categorical:
        summary["group_cov"] = group_cov_any

    if verbose:
        print("[lexicon] summary:")
        print(
            f"  texts={len(texts)} | lexicon_size={len(lex)} | "
            f"docs_any={docs_any} | cov_all={overall:.3f} | "
            f"q1={q1:.3f} | q4={q4:.3f} | corr_any={corr_any:.3f}"
        )
        if is_categorical:
            parts = " | ".join(
                f"{g}={v:.3f}" for g, v in group_cov_any.items()
            )
            print(f"  group_cov: {parts}")
        print(
            f"  hits_mean={hits_mean:.2f} | hits_median={hits_median:.2f} | "
            f"types_mean={types_mean:.2f} | types_median={types_median:.2f}"
        )
        if rows:
            print("\n  per-token:")
            for r in rows[:10]:
                print(
                    f"    {r['token']:>20s}  freq={r['frequency']:>5d}  "
                    f"assoc={r['association']:.3f}  p={r['pvalue']:.4g}  "
                    f"dir={r['effect_direction']}"
                )
        print("-" * 72)

    return summary, rows
