"""PCA-K sweep and selection for SSD.

Single-pass sweep over PCA_K values evaluating interpretability (cluster-based)
and beta stability, then selects the best K via a joint AUCK score.

The per-K cost is dominated by the cluster step (vocab GEMV + kmeans on 100×D
points, twice per K).  The sweep runs in three passes so the costly GEMV
collapses into a single batched GEMM:

  Pass 1 — slice the cached SVD, fit OLS, orient β, compute β-stability.
  Pass 2 — one (V, D) × (D, 2·N_K) GEMM yields all neighbor similarities.
  Pass 3 — argpartition / regex-filter / kmeans per (K, side); the cluster
           inputs come from indexing the cached normed vocab matrix, no
           per-word dict lookups.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ssdiff.backends._sweep_math import (
    PCAKSelectionResult,
)
from ssdiff.backends._sweep_math import (
    compute_auck as _compute_auck,
)
from ssdiff.backends._sweep_math import (
    cosine as _cosine,
)
from ssdiff.backends._sweep_math import (
    detrend_by_variance as _detrend_by_variance,
)
from ssdiff.backends._sweep_math import (
    overall_interpretability as _overall_interpretability,
)
from ssdiff.backends._sweep_math import (
    zscore_ignore_nan as _zscore_ignore_nan,
)
from ssdiff.lang_config import get_config as _get_lang_config
from ssdiff.utils import _diagnostic
from ssdiff.utils.math import kmeans_auto_k, unit_vector

# Restrict neighbor search to the top-N most-frequent vocab rows; matches the
# cluster_top_neighbors default used on the public path.
_RESTRICT_VOCAB = 50_000
# Candidate pool size before regex filtering — must be >= cluster_topn.
_NEIGHBOR_CAND = 2000
# kmeans seed and minimum cluster size — match cluster_top_neighbors defaults.
_KMEANS_SEED = 2137
_MIN_CLUSTER_SIZE = 2


def _top_indices_filtered(
    sim_col: np.ndarray,
    keys: list[str],
    bad_re,
    topn: int,
    cand: int,
) -> np.ndarray:
    """Return top-``topn`` filtered vocab indices from a similarity column.

    Mirrors :func:`~ssdiff.utils.neighbors.filtered_neighbors` but takes a
    precomputed similarity column so a batched GEMM can fan out to many K's.
    """
    V = sim_col.shape[0]
    cand = min(cand, V)
    if cand <= 0:
        return np.empty(0, dtype=np.intp)
    raw = np.argpartition(-sim_col, cand - 1)[:cand]
    raw = raw[np.argsort(-sim_col[raw])]
    out: list[int] = []
    for i in raw:
        if not bad_re.match(keys[int(i)]):
            out.append(int(i))
            if len(out) >= topn:
                break
    return np.asarray(out, dtype=np.intp)


def _cluster_from_indices(
    emb_n: np.ndarray,
    indices: np.ndarray,
    beta_unit: np.ndarray,
    *,
    side: str,
    k_min: int,
    k_max: int,
) -> list[dict]:
    """Cluster the rows of ``emb_n`` selected by ``indices``.

    Returns the keys consumed by
    :func:`~ssdiff.backends._sweep_math.overall_interpretability` —
    ``side``, ``size``, ``centroid_cos_beta``, ``coherence``.  Drops
    ``words`` because the sweep aggregates only.
    """
    if len(indices) < max(2, k_min):
        raise ValueError("Not enough neighbors to cluster.")
    W = emb_n[indices].astype(np.float64, copy=False)
    labels, _centers, _inertia, _k_use = kmeans_auto_k(
        W, k_min=k_min, k_max=min(k_max, len(W)), random_state=_KMEANS_SEED,
    )
    clusters: list[dict] = []
    for cid in sorted(set(labels)):
        idx = np.where(labels == cid)[0]
        if len(idx) < _MIN_CLUSTER_SIZE:
            continue
        Wc = W[idx]
        centroid = unit_vector(Wc.mean(axis=0))
        clusters.append({
            "side": side,
            "size": int(len(idx)),
            "centroid_cos_beta": float(centroid @ beta_unit),
            "coherence": float(np.mean((Wc @ centroid).astype(float))),
        })
    return clusters


def pca_sweep(
    *,
    Xs: np.ndarray,
    X_scale: np.ndarray,
    x: np.ndarray,
    ys: np.ndarray,
    embeddings,
    pca_k_values: Sequence[int] | None = None,
    cluster_topn: int = 100,
    cluster_k_min: int = 2,
    cluster_k_max: int = 5,
    weight_by_size: bool = True,
    auck_radius: int = 3,
    save_tables: bool = False,
    out_dir: str | None = None,
    prefix: str = "pca_k",
    verbose: bool = False,
    lang: str = "pl",
) -> PCAKSelectionResult:
    """Single-pass sweep over PCA_K on pre-standardized doc vectors.

    For each candidate K the function fits PCA(K) → OLS → beta, then
    evaluates interpretability (via cluster-based scoring on BOTH poles) and
    beta stability (cosine change between consecutive K).  The best K is
    chosen by a joint AUCK score.

    Parameters
    ----------
    Xs : (n, D) array
        Standardized document vectors.
    X_scale : (D,) array
        Column standard deviations (``scaler.scale_``).
    x : (n, D) array
        Raw (un-standardized) document vectors — used for beta orientation.
    ys : (n,) array
        Standardized outcome variable.
    embeddings : Embeddings
        Word embeddings for neighbor lookup and clustering.
    pca_k_values : sequence of int, optional
        PCA_K values to try.  Default ``range(20, 121, 2)``.
    cluster_topn : int
        Top neighbors to cluster per side (default 100).
    cluster_k_min, cluster_k_max : int
        Range for auto-selecting number of clusters.
    weight_by_size : bool
        Weight interpretability means by cluster size.
    auck_radius : int
        Radius for AUCK computation (default 3).
    save_tables : bool
        Save result table to Excel (requires pandas).
    out_dir : str or None
        Directory for optional table output.
    prefix : str
        File name prefix for output files.
    verbose : bool
        If True, show a tqdm progress bar over K values and print
        diagnostic messages (skipped K, best-K summary, saved files).

    Returns
    -------
    PCAKSelectionResult
        ``best_k`` and ``df_joined`` (list of row-dicts with all metrics).
    """
    if pca_k_values is None:
        pca_k_values = list(range(20, 121, 2))

    if save_tables and out_dir is None:
        raise ValueError("save_tables=True requires out_dir.")

    n, D = Xs.shape
    X_scale_safe = np.where(X_scale > 1e-12, X_scale, 1.0)

    # Precompute full SVD once — each K just slices the first K components.
    U_full, S_full, Vt_full = np.linalg.svd(Xs, full_matrices=False)
    explained_var_full = (S_full ** 2) / (n - 1)
    total_var_full = float(explained_var_full.sum())

    # ---- Pass 1: per-K linear algebra (cheap, no embedding lookups) -------
    # Each record is (ok, K, var_expl, gradient, beta_delta).
    # Failed K's keep ok=False and are skipped in Pass 2/3.
    records: list[tuple[bool, int, float, np.ndarray | None, float]] = []
    beta_prev: np.ndarray | None = None
    for K in pca_k_values:
        try:
            max_k = min(K, n - 1, D)
            if max_k < 1:
                raise ValueError(f"PCA_K={K} too large for data (n={n}, D={D})")

            components_k = Vt_full[:max_k]               # (max_k, D)
            z = Xs @ components_k.T                       # (n, max_k)
            var_expl = (
                float(explained_var_full[:max_k].sum() / total_var_full * 100)
                if total_var_full > 0 else 0.0
            )

            # OLS in PCA space (normal equations, matches official)
            w_reg = np.linalg.solve(z.T @ z, z.T @ ys)

            # Back-project to document space
            beta = (components_k.T @ w_reg) / X_scale_safe

            # Orient beta so higher alignment → higher outcome
            yhat = (x @ beta).ravel()
            denom = float(np.std(ys) * np.std(yhat))
            if denom > 0:
                c = float(np.corrcoef(ys, yhat)[0, 1])
                corr = c if np.isfinite(c) else 0.0
            else:
                corr = 0.0
            if corr < 0:
                beta = -beta

            gradient = unit_vector(beta)

            if beta_prev is not None:
                beta_delta = 1.0 - _cosine(beta_prev, gradient)
            else:
                beta_delta = float("nan")
            beta_prev = gradient

            records.append((True, int(K), var_expl, gradient, float(beta_delta)))
        except (np.linalg.LinAlgError, ValueError) as e:
            _diagnostic(verbose, f"[sweep] PCA_K={K} skipped: {type(e).__name__}: {e}")
            records.append((False, int(K), float("nan"), None, float("nan")))
            beta_prev = None

    # ---- Pass 2: one batched GEMM for all valid K's, both sides -----------
    # emb_n is the cached, restricted, L2-normalized vocab matrix; the matmul
    # is float32 to match the per-K path's dtype (similar_by_vector casts to
    # float32), so argpartition produces identical rankings.
    emb_n_full = embeddings.vectors
    restrict = min(_RESTRICT_VOCAB, emb_n_full.shape[0])
    emb_n = emb_n_full[:restrict]
    keys = embeddings.index_to_key
    bad_re = _get_lang_config(lang).bad_token_re

    valid_records = [r for r in records if r[0]]
    if valid_records:
        cols = []
        for _ok, _K, _var, gradient, _bd in valid_records:
            cols.append(gradient)
            cols.append(-gradient)
        # Stack as (D, 2 * N_valid) and cast once to embedding dtype.
        B = np.stack(cols, axis=1).astype(emb_n.dtype, copy=False)
        sims_all = emb_n @ B                              # (V, 2 * N_valid)
    else:
        sims_all = None

    # ---- Pass 3: filter + cluster per (K, side); aggregate per K ----------
    from ssdiff.utils import _progress

    rows: list[dict] = []
    col = 0
    for ok, K, var_expl, gradient, beta_delta in _progress(
        records, verbose=verbose, total=len(records), desc="PCA sweep",
    ):
        if not ok:
            rows.append(dict(
                PCA_K=K,
                var_explained=float("nan"),
                mean_coherence=float("nan"),
                mean_abs_cosb=float("nan"),
                aggregate=float("nan"),
                n_clusters=0,
                total_size=0,
                beta_delta_1_minus_cos=float("nan"),
            ))
            continue

        clusters: list[dict] = []
        for side, side_off in (("pos", 0), ("neg", 1)):
            sim_col = sims_all[:, col + side_off]
            try:
                indices = _top_indices_filtered(
                    sim_col, keys, bad_re,
                    topn=cluster_topn, cand=_NEIGHBOR_CAND,
                )
                clusters.extend(_cluster_from_indices(
                    emb_n, indices, gradient,
                    side=side,
                    k_min=cluster_k_min, k_max=cluster_k_max,
                ))
            except ValueError:
                continue
        col += 2

        overall = _overall_interpretability(clusters, weight_by_size=weight_by_size)
        rows.append(dict(
            PCA_K=K,
            var_explained=var_expl,
            mean_coherence=overall["mean_coherence"],
            mean_abs_cosb=overall["mean_abs_cosb"],
            aggregate=overall["aggregate"],
            n_clusters=overall["n_clusters"],
            total_size=overall["total_size"],
            beta_delta_1_minus_cos=(
                float(beta_delta) if np.isfinite(beta_delta) else float("nan")
            ),
        ))

    # Sort rows by PCA_K
    rows.sort(key=lambda r: r["PCA_K"])

    # Extract columns as numpy arrays for vectorized scoring
    pca_ks = np.array([r["PCA_K"] for r in rows], dtype=int)
    var_explained = np.array([r["var_explained"] for r in rows], dtype=float)
    aggregate = np.array([r["aggregate"] for r in rows], dtype=float)
    beta_delta_arr = np.array(
        [r["beta_delta_1_minus_cos"] for r in rows], dtype=float,
    )

    # --- Interpretability: detrend by log(var_explained) → z → AUCK ---
    interp_hat, interp_resid, _ = _detrend_by_variance(var_explained, aggregate)
    interp_z = _zscore_ignore_nan(interp_resid)
    interp_auck = _compute_auck(interp_z, radius=auck_radius)

    # --- Stability: smaller delta = better → z → AUCK ---
    stab_good_raw = -beta_delta_arr
    stab_z_raw = _zscore_ignore_nan(stab_good_raw)
    stab_auck_raw = _compute_auck(stab_z_raw, radius=auck_radius)

    # --- Joint score ---
    # Interpretability is the parsimony-aware channel; stability alone rises
    # with K and would pick the ceiling if used on its own. If interp is
    # fully unavailable (e.g., clustering failed on a tiny vocab), leave
    # joint_score all-NaN and let the selector fall back to k_min.
    interp_any = np.any(np.isfinite(interp_auck))
    stab_any = np.any(np.isfinite(stab_auck_raw))
    if interp_any and stab_any:
        joint_score = 0.5 * (interp_auck + stab_auck_raw)
    elif interp_any:
        joint_score = interp_auck
    else:
        joint_score = np.full_like(var_explained, np.nan)

    # Enrich row dicts with computed columns
    for i, r in enumerate(rows):
        r["interp_hat"] = float(interp_hat[i]) if np.isfinite(interp_hat[i]) else np.nan
        r["interp_resid"] = float(interp_resid[i]) if np.isfinite(interp_resid[i]) else np.nan
        r["interp_resid_z"] = float(interp_z[i]) if np.isfinite(interp_z[i]) else np.nan
        r["interp_auck"] = float(interp_auck[i]) if np.isfinite(interp_auck[i]) else np.nan
        r["stab_good_raw"] = float(stab_good_raw[i]) if np.isfinite(stab_good_raw[i]) else np.nan
        r["stab_z_raw"] = float(stab_z_raw[i]) if np.isfinite(stab_z_raw[i]) else np.nan
        r["stab_auck_raw"] = float(stab_auck_raw[i]) if np.isfinite(stab_auck_raw[i]) else np.nan
        r["joint_score"] = float(joint_score[i]) if np.isfinite(joint_score[i]) else np.nan

    # --- Choose best K ---
    finite_mask = np.isfinite(joint_score)
    if finite_mask.any():
        joint_vals = joint_score[finite_mask]
        ks = pca_ks[finite_mask]
        best_val = float(np.nanmax(joint_vals))
        tied = ks[np.isclose(joint_vals, best_val, rtol=0, atol=1e-12)]
        best_k = int(np.min(tied))
        _diagnostic(verbose, f"[sweep] best PCA_K={best_k} (joint={best_val:.4f})")
    else:
        # No differentiating signal (tiny corpus, degenerate clustering AND
        # no stability gradient). Fall back to the most parsimonious K.
        best_k = int(pca_ks.min())
        _diagnostic(verbose,
                    f"[sweep] no finite joint_score; falling back to k_min={best_k}")

    # --- Optional table output ---
    if save_tables and out_dir is not None:
        import os

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "save_tables=True requires 'pandas' (pip install pandas) "
                "for Excel export."
            ) from exc
        os.makedirs(out_dir, exist_ok=True)
        out_xlsx = os.path.join(out_dir, f"{prefix}_pca_k_joint_auck_table.xlsx")
        pd.DataFrame(rows).to_excel(out_xlsx, index=False)
        _diagnostic(verbose, f"[sweep] saved table -> {out_xlsx}")

    return PCAKSelectionResult(best_k=best_k, df_joined=rows)
