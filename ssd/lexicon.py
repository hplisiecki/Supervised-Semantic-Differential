# ssd/lexicon.py
from __future__ import annotations
from typing import Iterable, Sequence
import numpy as np
import pandas as pd

__all__ = [
    "suggest_lexicon",
    "token_presence_stats",
    "coverage_by_lexicon",
]

# -------------------------
# Helpers: inputs & metrics
# -------------------------

def _as_series_1d(y: Iterable) -> pd.Series:
    """Standardize y to a 1D float Series (no index semantics)."""
    if isinstance(y, pd.Series):
        return pd.to_numeric(y, errors="coerce")
    return pd.to_numeric(pd.Series(list(y)), errors="coerce")

def _texts_to_token_lists(texts: Sequence) -> list[list[str]]:
    """
    Normalize texts into token lists:
      - list[list[str]] → passthrough
      - list[str] / Series[str] → split on whitespace
    """
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    if not texts:
        return []
    first = texts[0]
    if isinstance(first, (list, tuple)):
        # assume already tokenized
        return [list(map(str, t)) for t in texts]
    # assume space-separated strings
    return [str(t).split() for t in texts]

def _token_sets(texts: Sequence) -> list[set[str]]:
    """Token lists → per-doc sets (unique presence)."""
    return [set(toks) for toks in _texts_to_token_lists(texts)]

def _quantile_bins(y: pd.Series | np.ndarray, n_bins: int = 4) -> np.ndarray:
    """
    Return integer bin labels (0..k-1) via quantiles; fallback: median split.
    Works for numpy arrays and pandas Series.
    """
    ys = _as_series_1d(y)
    try:
        bins = pd.qcut(ys, q=n_bins, labels=False, duplicates="drop")
        return bins.to_numpy()
    except Exception:
        med = float(np.nanmedian(ys.to_numpy()))
        return (ys.to_numpy() > med).astype(int)

def _z(v: pd.Series | np.ndarray) -> np.ndarray:
    """Z-score to float np.ndarray with ddof=0; protects zero variance."""
    arr = _as_series_1d(v).to_numpy(dtype=float)
    sd = np.std(arr, ddof=0)
    if sd <= 0 or not np.isfinite(sd):
        sd = 1.0
    mu = float(np.nanmean(arr))
    return (arr - mu) / sd

def _rank_for_token_stats(
    presence_vec: np.ndarray,
    y: pd.Series | np.ndarray,
    n_bins: int = 4,
    corr_cap: float = 0.30,
) -> tuple[float, float, float, float]:
    """
    presence_vec: 0/1 per document
    Returns: (cov_all, cov_bal, corr, rank)
    rank = balanced_coverage * (1 - min(1, |corr|/corr_cap))
    """
    bins = _quantile_bins(y, n_bins=n_bins)
    presence_vec = presence_vec.astype(float)
    cov_all = float(np.mean(presence_vec)) if len(presence_vec) else 0.0

    # balanced coverage: mean coverage within each bin
    cov_per_bin = []
    for b in sorted(np.unique(bins)):
        idx = np.where(bins == b)[0]
        cov_per_bin.append(float(np.mean(presence_vec[idx])) if len(idx) else 0.0)
    cov_bal = float(np.mean(cov_per_bin)) if cov_per_bin else 0.0

    y_std = _z(y)
    # guard zero variance in presence
    corr = float(np.corrcoef(presence_vec, y_std)[0, 1]) if np.std(presence_vec) > 0 else 0.0
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
) -> pd.DataFrame:
    """
    Suggest candidate tokens ranked by coverage with a mild penalty for strong correlation with y.

    Parameters
    ----------
    df_or_texts : DataFrame | Sequence[str] | Sequence[list[str]]
        If DataFrame, also pass text_col and score_col.
        Otherwise, pass texts and y separately using keyword-only variant:
          suggest_lexicon((texts, y), ...)   # where texts is list[str] or list[list[str]]
    text_col : str | None
        Column name with preprocessed text (space-separated) if df provided.
    score_col : str | None
        Column name with numeric outcome if df provided.

    Returns
    -------
    DataFrame with columns: token, docs, cov_all, cov_bal, corr, rank (sorted desc).
    """
    # Allow passing a tuple (texts, y) directly
    if not isinstance(df_or_texts, pd.DataFrame):
        if isinstance(df_or_texts, tuple) and len(df_or_texts) == 2:
            texts, y = df_or_texts
            texts = _texts_to_token_lists(texts)
            y = _as_series_1d(y)
        else:
            raise ValueError("If not passing a DataFrame, pass (texts, y) as a tuple.")
    else:
        if not text_col or not score_col:
            raise ValueError("Provide text_col and score_col when using a DataFrame.")
        s = df_or_texts[text_col].fillna("").astype(str)
        y = _as_series_1d(df_or_texts[score_col])
        mask = ~y.isna()
        texts = _texts_to_token_lists(s[mask].tolist())
        y = y[mask]

    # Build doc-frequency counts
    token_sets = _token_sets(texts)
    from collections import Counter
    df_counts = Counter()
    for ts in token_sets:
        df_counts.update(ts)
    vocab = [t for t, c in df_counts.items() if c >= min_docs]
    if not vocab:
        return pd.DataFrame(columns=["token", "docs", "cov_all", "cov_bal", "corr", "rank"])

    rows = []
    y_clean = y.reset_index(drop=True)
    for t in vocab:
        pres = np.fromiter((1 if t in ts else 0 for ts in token_sets), dtype=np.int8, count=len(token_sets))
        cov_all, cov_bal, corr, rank = _rank_for_token_stats(pres, y_clean, n_bins=n_bins, corr_cap=corr_cap)
        rows.append(dict(token=t, docs=int(pres.sum()), cov_all=cov_all, cov_bal=cov_bal, corr=corr, rank=rank))

    out = pd.DataFrame(rows)
    return (out
            .sort_values(["rank", "cov_bal", "docs"], ascending=[False, False, False])
            .head(top_k)
            .reset_index(drop=True))

def token_presence_stats(
    texts: Iterable[str],
    y: pd.Series | np.ndarray,
    token: str,
    *,
    n_bins: int = 4,
    corr_cap: float = 0.30,
    verbose: bool = False,
) -> dict:
    """
    Compute docs count, coverage, balanced coverage, correlation, and rank for a single token.
    Now with 'verbose' printing. Also returns q1/q4 (quartile coverages).
    """
    # --- coerce inputs ---
    token = str(token)
    if isinstance(y, np.ndarray):
        y_series = pd.Series(y)
    else:
        y_series = y.copy()

    texts_list = list(texts)
    pres = np.fromiter((1 if token in t.split() else 0 for t in texts_list),
                       dtype=np.int8, count=len(texts_list))

    # --- core stats (existing behavior) ---
    cov_all, cov_bal, corr, rank = _rank_for_token_stats(pres, y_series, n_bins=n_bins, corr_cap=corr_cap)

    # quartiles (for interpretability)
    bins = _quantile_bins(y_series, n_bins=n_bins)
    low = np.where(bins == bins.min())[0]
    high = np.where(bins == bins.max())[0]
    q1 = float(pres[low].mean()) if len(low) else 0.0
    q4 = float(pres[high].mean()) if len(high) else 0.0

    out = dict(
        token=token,
        docs=int(pres.sum()),
        cov_all=float(cov_all),
        cov_bal=float(cov_bal),
        corr=float(corr),
        rank=float(rank),
        q1=q1,
        q4=q4,
    )

    if verbose:
        print(f"[token] '{token}': "
              f"docs={out['docs']} | cov_all={out['cov_all']:.3f} | cov_bal={out['cov_bal']:.3f} | "
              f"q1={out['q1']:.3f} | q4={out['q4']:.3f} | corr={out['corr']:.3f} | rank={out['rank']:.3f}")

    return out


def coverage_by_lexicon(
    df_or_texts,
    text_col: str | None = None,
    score_col: str | None = None,
    lexicon: Iterable[str] = (),
    *,
    n_bins: int = 4,
    verbose: bool = False,
) -> tuple[dict, pd.DataFrame]:
    """
    Summarize coverage for a given lexicon.

    Parameters
    ----------
    df_or_texts : DataFrame | Sequence[str] | Sequence[list[str]]
        If DataFrame, also pass text_col and score_col.
        Otherwise, pass (texts, y) as a tuple like in suggest_lexicon.
    lexicon : iterable of str
        Candidate lexicon words (must match preprocessed tokens).

    Returns
    -------
    summary : dict(docs_any, cov_all, q1, q4, corr_any)
    per_token_df : DataFrame(word, docs, cov_all, q1, q4, corr)
    """
    if not isinstance(df_or_texts, pd.DataFrame):
        if isinstance(df_or_texts, tuple) and len(df_or_texts) == 2:
            texts, y = df_or_texts
            texts = _texts_to_token_lists(texts)
            y = _as_series_1d(y)
        else:
            raise ValueError("If not passing a DataFrame, pass (texts, y) as a tuple.")
    else:
        if not text_col or not score_col:
            raise ValueError("Provide text_col and score_col when using a DataFrame.")
        s = df_or_texts[text_col].fillna("").astype(str)
        y = _as_series_1d(df_or_texts[score_col])
        mask = ~y.isna()
        texts = _texts_to_token_lists(s[mask].tolist())
        y = y[mask]

    bins = _quantile_bins(y, n_bins=n_bins)
    low = np.where(bins == bins.min())[0]
    high = np.where(bins == bins.max())[0]

    lex = [str(w) for w in lexicon]
    token_sets = _token_sets(texts)

    pres_any = np.array([1 if any(w in ts for w in lex) else 0 for ts in token_sets], dtype=np.int8)
    y_std = _z(y)
    corr_any = float(np.corrcoef(pres_any, y_std)[0, 1]) if pres_any.std() > 0 else 0.0

    overall = float(pres_any.mean()) if len(pres_any) else 0.0
    q1 = float(pres_any[low].mean()) if len(low) else 0.0
    q4 = float(pres_any[high].mean()) if len(high) else 0.0
    docs_any = int(pres_any.sum())

    rows = []
    for w in lex:
        pres = np.array([1 if w in ts else 0 for ts in token_sets], dtype=np.int8)
        corr = float(np.corrcoef(pres, y_std)[0, 1]) if pres.std() > 0 else 0.0
        rows.append(dict(
            word=w,
            docs=int(pres.sum()),
            cov_all=float(pres.mean()) if len(pres) else 0.0,
            q1=float(pres[low].mean()) if len(low) else 0.0,
            q4=float(pres[high].mean()) if len(high) else 0.0,
            corr=corr,
        ))
    per_token = pd.DataFrame(rows, columns=["word", "docs", "cov_all", "q1", "q4", "corr"])



    summary = dict(
        docs_any=docs_any,
        cov_all=overall,
        q1=q1,
        q4=q4,
        corr_any=corr_any,
    )

    per_token = per_token.sort_values(
        ["cov_all", "docs"], ascending=[False, False]
    ).reset_index(drop=True)

    if verbose:
        print("[lexicon] summary:")
        print(
            f"  texts={len(texts)} | lexicon_size={len(lex)} | "
            f"docs_any={docs_any} | cov_all={overall:.3f} | "
            f"q1={q1:.3f} | q4={q4:.3f} | corr_any={corr_any:.3f}"
        )
        if not per_token.empty:
            preview_cols = ["word", "docs", "cov_all", "q1", "q4", "corr"]
            print("\n  per-token:")
            print(per_token.loc[:, preview_cols].head(10).to_string(index=False))
        print("-" * 72)



    return summary, per_token
