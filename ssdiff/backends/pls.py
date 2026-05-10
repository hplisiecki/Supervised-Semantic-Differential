"""PLS backend — multi-component PLS1 + W-subspace rotation.

Owns ``mpls_fit`` only: PLS1 fit + varimax/raw rotation, with reordering
by ``|corr(t_i, y)|`` and sign-flipping so ``corr > 0``. Significance
testing and k-selection live in ``ssdiff.ssd`` and call ``plskit``
directly.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

import plskit


__all__ = ("mpls_fit",)


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
        upstream (matches ``plskit.pls1_fit``'s ``pre_standardized=True``
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
        model = plskit.pls1_fit(Xs, ys, k=n_components, pre_standardized=True)
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
