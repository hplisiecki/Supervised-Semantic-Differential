"""PLS backend — orchestration helpers over plskit.

* ``mpls_fit`` — multi-component PLS1 + W-subspace rotation (varimax / raw),
  reordered by ``|corr(t_i, y)|`` with sign-flip so ``corr > 0``.
* ``confirmatory_test`` — wrap ``plskit.pls1_confirmatory_test``;
  returns the resolved test name and an info dict ready for ``TestView``.
* ``find_k_optimal_with_fwer`` — wrap ``plskit.pls1_find_k_optimal``
  with ``return_fwer=True``; returns ``(k_star, fwer_alpha, info, raw)``.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np

import plskit


__all__ = (
    "confirmatory_test",
    "find_k_optimal_with_fwer",
    "mpls_fit",
    "TEST_METHODS",
    "FWER_METHODS",
)


TEST_METHODS = ("raw_perm", "split_nb", "split_perm", "score", "e")
FWER_METHODS = ("raw_perm", "split_nb", "split_perm", "e")


def _confirmatory_args(
    method: str, *, n_perm: int, n_splits: int,
) -> dict | None:
    if method == "raw_perm":
        return {"n_perm": int(n_perm)}
    if method == "split_nb":
        return {"n_splits": int(n_splits)}
    if method == "split_perm":
        return {"n_perm": int(n_perm), "n_splits": int(n_splits)}
    return None  # score, e take no args


def _split_r2_from(r) -> float | None:
    # split_nb / split_perm: statistic is the back-transformed mean Fisher-z.
    if r.method in ("split_nb", "split_perm"):
        return float(r.statistic)
    return None


def confirmatory_test(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    *,
    method: Literal["raw_perm", "split_nb", "split_perm", "score", "e"],
    n_perm: int = 1000,
    n_splits: int = 50,
    pre_standardized_X: bool = False,
    seed: int | None = None,
    verbose: bool = False,
) -> tuple[str, dict]:
    """Run ``plskit.pls1_confirmatory_test`` and assemble a TestView-ready dict.

    Parameters
    ----------
    X, y : ndarray
        Feature matrix ``(n, D)`` and outcome ``(n,)``.
    k : int
        Component count to test.
    method : str
        ``"raw_perm" | "split_nb" | "split_perm" | "score" | "e"``.
    n_perm, n_splits : int
        Resampling kwargs (used only by the methods that consume them).
    pre_standardized_X : bool
        Forwarded to plskit; default ``False`` so plskit standardises
        internally (necessary for honest CV / split resampling).
    seed : int or None
    verbose : bool

    Returns
    -------
    method : str
        Resolved test name (echoes ``method``).
    info : dict
        ``pvalue``, ``statistic``, ``split_r2`` (Optional), ``n_perm``,
        ``n_splits``, ``random_state``.
    """
    if method not in TEST_METHODS:
        raise ValueError(
            f"Unknown test method {method!r}. Choose from {TEST_METHODS}."
        )
    r = plskit.pls1_confirmatory_test(
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        int(k),
        method=method,
        args=_confirmatory_args(method, n_perm=n_perm, n_splits=n_splits),
        pre_standardized_X=pre_standardized_X,
        seed=seed,
        verbose=verbose,
    )
    return method, {
        "pvalue": float(r.pvalue),
        "statistic": float(r.statistic),
        "split_r2": _split_r2_from(r),
        "n_perm": r.n_perm,
        "n_splits": r.n_splits,
        "random_state": r.seed,
    }


def find_k_optimal_with_fwer(
    Xs: np.ndarray,
    ys: np.ndarray,
    k_max: int,
    *,
    fwer_method: Literal["raw_perm", "split_nb", "split_perm", "e"] = "split_nb",
    selector: Literal["r2_se", "r2_max", "bic"] = "r2_se",
    n_folds: int = 5,
    n_perm: int = 1000,
    n_splits: int = 50,
    seed: int | None = None,
    verbose: bool = False,
) -> tuple[int, float, dict, Any]:
    """Run ``plskit.pls1_find_k_optimal(return_fwer=True)``.

    Inputs ``Xs, ys`` are assumed already-standardised (forwarded with
    ``pre_standardized_X=True``).

    Returns
    -------
    k_star : int
        Selected component count.
    fwer_alpha : float
        FWER-corrected p-value at ``k_star`` — used as the SSD p-value.
    info : dict
        ``pvalue`` (= ``fwer_alpha``), ``fwer_method``, ``selector``,
        ``k_star``, plus ``n_perm`` / ``n_splits`` when applicable, and
        ``random_state``.
    raw : plskit.FindKOptimalResult
        Full plskit output — kept around for diagnostics.
    """
    if fwer_method not in FWER_METHODS:
        raise ValueError(
            f"Unknown fwer_method {fwer_method!r}. "
            f"Choose from {FWER_METHODS}."
        )
    args: dict = {}
    if selector != "bic":
        args["n_folds"] = int(n_folds)
    if fwer_method in ("raw_perm", "split_perm"):
        args["fwer_n_perm"] = int(n_perm)
    if fwer_method in ("split_nb", "split_perm"):
        args["fwer_n_splits"] = int(n_splits)

    r = plskit.pls1_find_k_optimal(
        np.asarray(Xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        int(k_max),
        selector=selector,
        return_fwer=True,
        fwer_method=fwer_method,
        args=args or None,
        pre_standardized_X=True,
        seed=seed,
        verbose=verbose,
    )

    info: dict = {
        "pvalue": float(r.fwer_alpha),
        "split_r2": None,
        "fwer_method": r.fwer_method,
        "selector": r.selector,
        "k_star": int(r.k_star),
        "random_state": r.seed,
    }
    if fwer_method in ("raw_perm", "split_perm"):
        info["n_perm"] = int(n_perm)
    if fwer_method in ("split_nb", "split_perm"):
        info["n_splits"] = int(n_splits)
    return int(r.k_star), float(r.fwer_alpha), info, r


def mpls_fit(
    Xs: np.ndarray,
    ys: np.ndarray,
    *,
    n_components: int,
    rotate: Literal["raw", "varimax"],
    E_target: np.ndarray | Callable[[np.ndarray], np.ndarray],
) -> dict:
    """Fit PLS1 on already-standardised inputs, then rotate the W-subspace.

    Parameters
    ----------
    Xs, ys : ndarray
        Already-standardised ``X`` / ``y``. The caller standardises
        upstream (matches ``plskit.pls1_fit``'s ``pre_standardized_X=True``
        path).
    n_components : int
        Exact number of components. Raises if NIPALS deflation collapses.
    rotate : {"raw", "varimax"}
        Rotation applied to W. ``"raw"`` still reorders by
        ``|corr(t_i, y)|`` and sign-flips.
    E_target : ndarray ``(V, D)`` or callable ``(W) -> (V, k)``
        Loading basis (or projector) used as the simple-structure target.
        Callable form lets the caller fold standardisation into the
        matmul to avoid materialising a full ``(V, D)`` standardised copy.

    Returns
    -------
    dict
        Keys: ``W``, ``P``, ``Q``, ``W_rot``, ``T_rot``, ``beta_combined``,
        ``order``, ``signs``, ``rotation_meta``.
    """
    Xs = np.asarray(Xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    try:
        model = plskit.pls1_fit(Xs, ys, k=n_components, pre_standardized_X=True)
    except plskit.PlsKitError as exc:
        raise ValueError(
            f"plskit.pls1_fit rejected n_components={n_components}: {exc}. "
            f"Reduce n_components or check for near-duplicate rows / "
            f"near-zero variance columns in X."
        ) from exc

    actual_k = model.W.shape[1]
    if actual_k < n_components:
        raise ValueError(
            f"plskit.pls1_fit returned {actual_k} components but "
            f"n_components={n_components} was requested (NIPALS deflation "
            f"collapsed). Reduce n_components or check for near-duplicate "
            f"rows / near-zero variance columns in X."
        )

    if rotate == "raw":
        R_rot = np.eye(n_components)
        W_pre = model.W.copy()
        rot_meta_pre: dict = {"rotate": "raw", "sweeps": 0,
                              "V_converged": 0.0, "kaiser_normalized": False}
    elif rotate == "varimax":
        if callable(E_target):
            L = np.asarray(E_target(model.W), dtype=np.float64)
        else:
            E_arr = np.asarray(E_target)
            L = (E_arr @ model.W.astype(E_arr.dtype, copy=False)).astype(
                np.float64, copy=False
            )
        rot = plskit.rotate(model.W, method="varimax", L=L)
        R_rot = rot.spec.R
        W_pre = rot.W_rot
        rot_meta_pre = {
            "rotate": "varimax",
            "sweeps": rot.spec.sweeps,
            "V_converged": rot.spec.V_converged,
            "kaiser_normalized": rot.spec.args["kaiser_normalize"],
        }
    else:
        raise ValueError(f"rotate must be 'raw' or 'varimax'; got {rotate!r}")

    # Order by |corr(T_pre[:, i], y)| desc, sign-flip so corr > 0.
    T_pre = Xs @ W_pre
    y_c = ys - ys.mean()
    y_norm = float(np.linalg.norm(y_c))
    if y_norm < 1e-12:
        y_norm = 1.0

    corrs = np.zeros(n_components)
    signs = np.ones(n_components)
    for i in range(n_components):
        ti = T_pre[:, i] - T_pre[:, i].mean()
        t_norm = float(np.linalg.norm(ti))
        if t_norm < 1e-12:
            corrs[i] = 0.0
            signs[i] = 1.0
        else:
            c = float(ti @ y_c) / (t_norm * y_norm)
            corrs[i] = abs(c)
            signs[i] = 1.0 if c >= 0 else -1.0

    order = np.argsort(-corrs)
    R_rot = R_rot[:, order] * signs[order][np.newaxis, :]

    W_rot = W_pre[:, order] * signs[order][np.newaxis, :]
    T_rot = T_pre[:, order] * signs[order][np.newaxis, :]

    rotation_meta = {**rot_meta_pre, "R": R_rot,
                     "order": order, "signs": signs[order]}

    return {
        "W": model.W, "P": model.P, "Q": model.Q,
        "W_rot": W_rot, "T_rot": T_rot,
        "beta_combined": model.coef,
        "order": order, "signs": signs[order],
        "rotation_meta": rotation_meta,
    }
