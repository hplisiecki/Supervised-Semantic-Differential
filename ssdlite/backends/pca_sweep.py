"""PCA-K sweep and selection for SSD (sklearn backend).

Single-pass sweep over PCA_K values evaluating interpretability (cluster-based)
and beta stability, then selects the best K via a joint AUCK score.

Matches the algorithm in the official ``ssdiff`` package exactly:
uses sklearn StandardScaler, PCA, KMeans, and silhouette_score.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

from ssdlite.backends._sweep_math import (
    PCAKSelectionResult,
    cosine as _cosine,
    zscore_ignore_nan as _zscore_ignore_nan,
    compute_auck as _compute_auck,
    detrend_by_variance as _detrend_by_variance,
    overall_interpretability as _overall_interpretability,
)
from ssdlite.utils.math import unit_vector
from ssdlite.utils.neighbors import filtered_neighbors

warnings.filterwarnings(
    "ignore",
    message=r"KMeans is known to have a memory leak on Windows with MKL.*",
    category=UserWarning,
    module=r"sklearn\.cluster\._kmeans",
)


def _require_sklearn():
    try:
        from sklearn.preprocessing import StandardScaler  # noqa: F401
        from sklearn.decomposition import PCA  # noqa: F401
        from sklearn.cluster import KMeans  # noqa: F401
        from sklearn.metrics import silhouette_score  # noqa: F401
    except ImportError:
        raise ImportError(
            "scikit-learn is required for the PCA sweep backend. "
            "Install: pip install scikit-learn"
        )


def _cluster_both_sides(
    kv,
    beta: np.ndarray,
    *,
    topn: int = 100,
    k_min: int = 2,
    k_max: int = 5,
    restrict_vocab: int = 50000,
    random_state: int = 13,
    lang: str = "pl",
    min_cluster_size: int = 2,
) -> list[dict]:
    """Cluster top neighbors of both +beta and -beta using sklearn KMeans.

    Matches the official ``ssd.cluster_neighbors()`` behavior: clusters both
    poles and returns a combined list of cluster dicts.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    bu = unit_vector(beta)
    all_clusters: list[dict] = []

    for side, vec in (("pos", bu), ("neg", -bu)):
        pairs = filtered_neighbors(kv, vec, topn=topn, restrict=restrict_vocab, lang=lang)
        words = [w for (w, _s) in pairs]
        if len(words) < max(2, k_min):
            continue

        W = np.vstack(
            [kv.get_vector(w, norm=True).astype(np.float64) for w in words]
        )

        # Auto-select k via silhouette score (matching official)
        upper = min(k_max, max(k_min, W.shape[0] - 1))
        best_s, best_labels = -1.0, None
        for kk in range(max(2, k_min), max(2, upper) + 1):
            km = KMeans(n_clusters=kk, random_state=random_state, n_init="auto")
            labels = km.fit_predict(W)
            if len(set(labels)) <= 1 or np.max(np.bincount(labels)) <= 1:
                continue
            s = silhouette_score(W, labels)
            if s > best_s:
                best_s, best_labels = s, labels

        if best_labels is None:
            km = KMeans(n_clusters=max(2, k_min), random_state=random_state, n_init="auto")
            best_labels = km.fit_predict(W)

        for cid in sorted(set(best_labels)):
            idx = np.where(best_labels == cid)[0]
            if len(idx) < min_cluster_size:
                continue
            Wc = W[idx]
            centroid = unit_vector(Wc.mean(axis=0))
            cos_beta = float(centroid @ bu)
            cos_to_centroid = (Wc @ centroid).astype(float)
            coherence = float(np.mean(cos_to_centroid))

            all_clusters.append({
                "side": side,
                "size": int(len(idx)),
                "centroid_cos_beta": cos_beta,
                "coherence": coherence,
            })

    return all_clusters


def pca_sweep(
    *,
    Xs: np.ndarray,
    X_scale: np.ndarray,
    x: np.ndarray,
    ys: np.ndarray,
    kv,
    use_unit_beta: bool = True,
    pca_k_values: Sequence[int] | None = None,
    cluster_topn: int = 100,
    cluster_k_min: int = 2,
    cluster_k_max: int = 5,
    weight_by_size: bool = True,
    auck_radius: int = 3,
    save_tables: bool = False,
    out_dir: str | None = None,
    prefix: str = "pca_k",
    verbose: bool = True,
    lang: str = "pl",
) -> PCAKSelectionResult:
    """Single-pass sweep over PCA_K on pre-standardized doc vectors.

    For each candidate K the function fits sklearn PCA(K) → OLS → beta, then
    evaluates interpretability (via cluster-based scoring on BOTH poles) and
    beta stability (cosine change between consecutive K).  The best K is
    chosen by a joint AUCK score.

    Parameters
    ----------
    Xs : (n, D) array
        Standardized document vectors (from sklearn StandardScaler).
    X_scale : (D,) array
        Column standard deviations (``scaler.scale_``).
    x : (n, D) array
        Raw (un-standardized) document vectors — used for beta orientation.
    ys : (n,) array
        Standardized outcome variable.
    kv : Embeddings
        Word embeddings for neighbor lookup and clustering.
    use_unit_beta : bool
        Whether to use unit-normed beta for neighbor search.
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
        Print progress.

    Returns
    -------
    PCAKSelectionResult
        ``best_k`` and ``df_joined`` (list of row-dicts with all metrics).
    """
    _require_sklearn()

    if pca_k_values is None:
        pca_k_values = list(range(20, 121, 2))

    if save_tables and out_dir is None:
        raise ValueError("save_tables=True requires out_dir.")

    n, D = Xs.shape
    X_scale_safe = np.where(X_scale > 1e-12, X_scale, 1.0)

    rows: list[dict] = []
    beta_prev: np.ndarray | None = None

    # Precompute full SVD once — each K just slices the first K components.
    # Equivalent to sklearn PCA(svd_solver='full') which also uses LAPACK SVD.
    U_full, S_full, Vt_full = np.linalg.svd(Xs, full_matrices=False)
    explained_var_full = (S_full ** 2) / (n - 1)
    total_var_full = float(explained_var_full.sum())

    for K in pca_k_values:
        if verbose:
            print(f"  [pca_sweep] PCA_K={K}")

        try:
            max_k = min(K, n - 1, D)
            if max_k < 1:
                raise ValueError(f"PCA_K={K} too large for data (n={n}, D={D})")

            # Slice precomputed SVD (equivalent to sklearn PCA with full SVD)
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
            if verbose:
                print(f"    [skip] PCA_K={K} failed: {type(e).__name__}: {e}")
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

    if verbose:
        print("\n=== BEST PCA_K (JOINT AUCK, single-pass) ===")
        print(f"PCA_K        : {best_k}")
        print(f"joint_score  : {best_val:.6f}")
        best_row = next(r for r in rows if r["PCA_K"] == best_k)
        print(f"interp_auck  : {best_row['interp_auck']:.6f}")
        print(f"stab_auck_raw: {best_row['stab_auck_raw']:.6f}")

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
        if verbose:
            print(f"Saved table -> {out_xlsx}")

    return PCAKSelectionResult(best_k=best_k, df_joined=rows)
