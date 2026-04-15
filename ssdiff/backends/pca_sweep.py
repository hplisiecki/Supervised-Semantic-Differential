"""PCA-K sweep and selection for SSD.

Single-pass sweep over PCA_K values evaluating interpretability (cluster-based)
and beta stability, then selects the best K via a joint AUCK score.
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
from ssdiff.utils import _diagnostic
from ssdiff.utils.math import unit_vector
from ssdiff.utils.neighbors import cluster_top_neighbors


def _cluster_both_sides(
    kv,
    beta: np.ndarray,
    *,
    topn: int = 100,
    k_min: int = 2,
    k_max: int = 5,
    restrict_vocab: int = 50000,
    random_state: int = 2137,
    lang: str = "pl",
    min_cluster_size: int = 2,
) -> list[dict]:
    """Cluster top neighbors of both +beta and -beta poles.

    Wraps :func:`~ssdiff.utils.neighbors.cluster_top_neighbors` (pure numpy)
    for both poles and returns a combined list of cluster dicts compatible
    with :func:`~ssdiff.backends._sweep_math.overall_interpretability`.
    """
    all_clusters: list[dict] = []

    for side in ("pos", "neg"):
        try:
            clusters = cluster_top_neighbors(
                kv, beta,
                topn=topn,
                k=None,
                k_min=k_min,
                k_max=k_max,
                restrict_vocab=restrict_vocab,
                random_state=random_state,
                min_cluster_size=min_cluster_size,
                side=side,
                lang=lang,
            )
        except ValueError:
            continue

        for c in clusters:
            all_clusters.append({
                "side": side,
                "size": c["size"],
                "centroid_cos_beta": c["centroid_cos_beta"],
                "coherence": c["coherence"],
            })

    return all_clusters


def pca_sweep(
    *,
    Xs: np.ndarray,
    X_scale: np.ndarray,
    x: np.ndarray,
    ys: np.ndarray,
    kv,
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
    kv : Embeddings
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

    rows: list[dict] = []
    beta_prev: np.ndarray | None = None

    # Precompute full SVD once — each K just slices the first K components.
    U_full, S_full, Vt_full = np.linalg.svd(Xs, full_matrices=False)
    explained_var_full = (S_full ** 2) / (n - 1)
    total_var_full = float(explained_var_full.sum())

    from ssdiff.utils import _progress

    for K in _progress(pca_k_values, verbose=verbose,
                       total=len(pca_k_values), desc="PCA sweep"):

        try:
            max_k = min(K, n - 1, D)
            if max_k < 1:
                raise ValueError(f"PCA_K={K} too large for data (n={n}, D={D})")

            # Slice precomputed SVD
            components_k = Vt_full[:max_k]          # (max_k, D)
            z = Xs @ components_k.T                  # (n, max_k)
            var_expl = float(explained_var_full[:max_k].sum() / total_var_full * 100) if total_var_full > 0 else 0.0

            # OLS in PCA space (normal equations, matches official)
            w_reg = np.linalg.solve(z.T @ z, z.T @ ys)

            # Back-project to document space
            beta_std = components_k.T @ w_reg
            beta = beta_std / X_scale_safe

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

            beta_unit = unit_vector(beta)

            # Beta stability
            if beta_prev is not None:
                beta_delta = 1.0 - _cosine(beta_prev, beta_unit)
            else:
                beta_delta = np.nan
            beta_prev = beta_unit

            # Interpretability via clustering BOTH sides (matches official)
            clusters = _cluster_both_sides(
                kv, beta,
                topn=cluster_topn,
                k_min=cluster_k_min,
                k_max=cluster_k_max,
                lang=lang,
            )
            overall = _overall_interpretability(
                clusters, weight_by_size=weight_by_size,
            )

            rows.append(dict(
                PCA_K=int(K),
                var_explained=var_expl,
                mean_coherence=overall["mean_coherence"],
                mean_abs_cosb=overall["mean_abs_cosb"],
                aggregate=overall["aggregate"],
                n_clusters=overall["n_clusters"],
                total_size=overall["total_size"],
                beta_delta_1_minus_cos=(
                    float(beta_delta) if np.isfinite(beta_delta) else np.nan
                ),
            ))

        except (np.linalg.LinAlgError, ValueError) as e:
            _diagnostic(verbose, f"[sweep] PCA_K={K} skipped: {type(e).__name__}: {e}")
            rows.append(dict(
                PCA_K=int(K),
                var_explained=np.nan,
                mean_coherence=np.nan,
                mean_abs_cosb=np.nan,
                aggregate=np.nan,
                n_clusters=0,
                total_size=0,
                beta_delta_1_minus_cos=np.nan,
            ))
            beta_prev = None

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
    joint_score = 0.5 * (interp_auck + stab_auck_raw)

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
    if not finite_mask.any():
        raise RuntimeError("No finite joint_score values; cannot select best PCA_K.")

    joint_vals = joint_score[finite_mask]
    ks = pca_ks[finite_mask]

    best_val = float(np.nanmax(joint_vals))
    tied = ks[np.isclose(joint_vals, best_val, rtol=0, atol=1e-12)]
    best_k = int(np.min(tied))

    _diagnostic(verbose, f"[sweep] best PCA_K={best_k} (joint={best_val:.4f})")

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
