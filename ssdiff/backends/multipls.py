"""Multi-dimensional PLS — rotation math + orchestrator.

Pure numpy. No result-class knowledge.

Implements the W-subspace rotation described in
`docs/propositions/2026-04-23-varimax-rotation-on-pls-dimensions.md`:
varimax (orthogonal) and promax (oblique), with an orchestrator
`mpls_fit` that wraps `pls1_fit` and returns everything a
MultiPLSResult needs (rotated weights, dim scores, combined beta,
rotation metadata, sign/order bookkeeping).
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def varimax_angle_2d(L: np.ndarray) -> float:
    """Closed-form varimax rotation angle for a 2-column loadings matrix.

    Maximises ``Σ_j Var((L @ R)[:, j]²)`` over 2D rotations ``R``.
    Derived in Kaiser (1958); see
    `docs/propositions/2026-04-23-varimax-rotation-on-pls-dimensions.md`.

    Parameters
    ----------
    L : ndarray of shape (n, 2)
        Loadings on two axes.

    Returns
    -------
    float
        Rotation angle θ (radians) such that ``L @ rot2d(θ)`` has
        simpler structure. Lies in ``(-π/4, π/4]``.
    """
    a, b = L[:, 0], L[:, 1]
    u = a * a - b * b
    v = 2 * a * b
    n = len(L)
    A = float((u * u - v * v).sum() - (u.sum() ** 2 - v.sum() ** 2) / n)
    B = float(2 * ((u * v).sum() - u.sum() * v.sum() / n))
    return float(np.arctan2(B, A) / 4.0)


def varimax_kaiser_sweep(
    L: np.ndarray,
    *,
    tol: float = 1e-8,
    max_sweeps: int = 50,
    kaiser_normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pairwise Kaiser sweeps for k-dimensional varimax.

    Each sweep applies ``varimax_angle_2d`` to every column pair in turn;
    iteration stops when the criterion ``V(L) = Σ_j Var(L² col j)`` changes
    by less than ``tol``. Kaiser row-normalisation (row h²-scaling before
    rotation, restore after) is on by default.

    Parameters
    ----------
    L : ndarray of shape (n, k)
        Loadings to rotate.
    tol : float
        Convergence tolerance on V(L).
    max_sweeps : int
        Hard cap on sweep count.
    kaiser_normalize : bool
        If True, divide each row by its norm before sweeping and
        multiply back at the end.

    Returns
    -------
    L_rot : ndarray of shape (n, k)
        Rotated loadings.
    R : ndarray of shape (k, k)
        Orthogonal rotation such that ``L @ R = L_rot``.
    meta : dict
        ``{"sweeps": int, "V_converged": float, "kaiser_normalized": bool}``.
    """
    n, k = L.shape
    if k < 2:
        return L.copy(), np.eye(k), {
            "sweeps": 0,
            "V_converged": float(np.sum(np.var(L ** 2, axis=0))),
            "kaiser_normalized": bool(kaiser_normalize),
        }

    if kaiser_normalize:
        row_norms = np.linalg.norm(L, axis=1, keepdims=True)
        row_norms = np.where(row_norms > 1e-12, row_norms, 1.0)
        L_work = L / row_norms
    else:
        L_work = L.copy()

    R = np.eye(k)
    V_prev = float(np.sum(np.var(L_work ** 2, axis=0)))

    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        for p in range(k - 1):
            for q in range(p + 1, k):
                pair = L_work[:, [p, q]]
                theta = varimax_angle_2d(pair)
                c, s = np.cos(theta), np.sin(theta)
                # Apply rotation to columns p, q of L_work AND R.
                Lp, Lq = L_work[:, p].copy(), L_work[:, q].copy()
                L_work[:, p] = c * Lp + s * Lq
                L_work[:, q] = -s * Lp + c * Lq
                Rp, Rq = R[:, p].copy(), R[:, q].copy()
                R[:, p] = c * Rp + s * Rq
                R[:, q] = -s * Rp + c * Rq
        V_new = float(np.sum(np.var(L_work ** 2, axis=0)))
        if V_new - V_prev < tol:
            break
        V_prev = V_new

    if kaiser_normalize:
        L_rot = L @ R  # restore original row magnitudes via the rotated original L
    else:
        L_rot = L_work

    return L_rot, R, {
        "sweeps": sweeps,
        "V_converged": V_prev,
        "kaiser_normalized": bool(kaiser_normalize),
    }


def promax_fit(
    L_varimax: np.ndarray,
    *,
    kappa: int | float = 4,
) -> dict:
    """Oblique promax rotation applied to a varimax-rotated loadings matrix.

    Standard Hendrickson & White (1964) procedure:

    1. ``P = sign(L) * |L|**kappa`` — exaggerated target.
    2. ``T = (L.T @ L)^{-1} @ L.T @ P`` — least-squares transform.
    3. Rescale columns of ``T`` by ``sqrt(diag((T.T @ T)^{-1}))`` so the
       rescaled transform satisfies ``diag((T.T @ T)^{-1}) = 1`` — i.e. the
       factor-correlation matrix has unit diagonal.
    4. ``pattern = L @ T``;  ``phi = inv(T.T @ T)``;  ``structure = pattern @ phi``.

    At κ=1, the target equals the input, the transform is identity, and
    the result degenerates to the varimax input (pattern = structure,
    phi = I). This degeneracy is asserted in tests.

    Parameters
    ----------
    L_varimax : ndarray of shape (n, k)
        Varimax-rotated loadings. Column scaling is assumed to be
        whatever ``varimax_kaiser_sweep`` produced.
    kappa : int or float
        Exaggeration exponent. Common values: 2 (mild), 4 (default).

    Returns
    -------
    dict with keys
        pattern   : (n, k) unique-contribution loadings (use for .words)
        structure : (n, k) correlation of each item with each factor
        phi       : (k, k) factor correlation matrix
        transform : (k, k) oblique transform (L_varimax @ transform = pattern)
        kappa     : echoed back
    """
    L = np.asarray(L_varimax, dtype=float)
    _, k = L.shape

    # Step 1: exaggerated target.
    P = np.sign(L) * (np.abs(L) ** kappa)

    # Step 2: least-squares transform.
    # (L'L) may be singular in degenerate cases; solve via lstsq for safety.
    T, *_ = np.linalg.lstsq(L, P, rcond=None)

    # Step 3: rescale columns of T so diag((T_new'T_new)^-1) == 1.
    # With d = sqrt(diag((T'T)^-1)) and T_new = T * d, we get
    # (T_new'T_new)^-1 = diag(1/d) (T'T)^-1 diag(1/d), whose diagonal is
    # (1/d_i^2) * d_i^2 = 1.
    TtT_inv = np.linalg.inv(T.T @ T)
    d = np.sqrt(np.diag(TtT_inv))
    d = np.where(d > 1e-12, d, 1.0)
    T = T * d[np.newaxis, :]

    pattern = L @ T
    phi = np.linalg.inv(T.T @ T)
    structure = pattern @ phi

    return {
        "pattern": pattern,
        "structure": structure,
        "phi": phi,
        "transform": T,
        "kappa": kappa,
    }


def mpls_fit(
    Xs: np.ndarray,
    ys: np.ndarray,
    *,
    n_components: int,
    rotate: Literal["raw", "varimax", "promax"],
    E_target: np.ndarray,
    kappa: int | float = 4,
) -> dict:
    """Fit PLS1 on already-standardised inputs, then rotate the W-subspace.

    The caller is responsible for standardising ``X`` and ``y`` before
    passing them in — this matches how ``pls1_fit`` expects its inputs
    and avoids double-standardisation when the caller does PCA
    preprocessing.

    Parameters
    ----------
    Xs : ndarray of shape (n, D)
        Already-standardised document vectors (or PCA scores, if the
        caller applied PCA preprocessing — in that case D is the number
        of PCA components).
    ys : ndarray of shape (n,)
        Already-standardised outcome.
    n_components : int
        Exact number of components to extract. Raises if NIPALS deflation
        collapses and fewer are returned.
    rotate : {"raw", "varimax", "promax"}
        Rotation scheme applied to W. ``"raw"`` still reorders and
        sign-flips the columns; the underlying subspace is preserved.
    E_target : ndarray of shape (V, D)
        Full-vocabulary matrix in the same column space as ``Xs`` —
        typically ``(E - x_mean) / x_scale`` (no PCA) or that same
        matrix further projected into PCA space (with PCA). Used as the
        rotation target.
    kappa : int or float
        Promax exaggeration exponent (ignored for non-promax rotations).

    Returns
    -------
    dict with keys:
        W, P, Q          : unrotated PLS1 outputs (for .test and reconstruction)
        W_rot            : (D, k) rotated + ordered + sign-flipped weights
        T_rot            : (n, k) dim scores on UN-deflated Xs
        beta_combined    : (D,) unrotated PLS coefficient (rotation-invariant)
        order            : (k,) permutation applied for ordering
        signs            : (k,) +1/−1 applied after ordering
        rotation_meta    : dict with rotation diagnostics
    """
    from ssdiff.backends.pls import pls1_fit

    Xs = np.asarray(Xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    _, P, W, Q, coef = pls1_fit(Xs, ys, n_components)

    actual_k = W.shape[1]
    if actual_k < n_components:
        raise ValueError(
            f"pls1_fit returned {actual_k} components but n_components={n_components} "
            f"was requested (NIPALS deflation collapsed). Reduce n_components or "
            f"check for near-duplicate rows / near-zero variance columns in X."
        )

    # -- Rotation ----------------------------------------------------------
    # Project the full vocabulary onto the W-subspace.
    # Downcast W to E_target's dtype for the matmul so we don't allocate a
    # second full (V, D) block via np.asarray(..., dtype=float64). The
    # (V, k) result is tiny — upcasting it to float64 for downstream
    # rotation math is essentially free.
    E_asarray = np.asarray(E_target)
    L = (E_asarray @ W.astype(E_asarray.dtype, copy=False)).astype(
        np.float64, copy=False
    )

    if rotate == "raw":
        R_rot = np.eye(n_components)
        W_pre = W.copy()
        rot_diag: dict = {
            "kaiser_normalized": False,
            "sweeps": 0,
            "V_converged": float(np.sum(np.var(L ** 2, axis=0))),
            "kappa": None,
            "pattern": None,
            "structure": None,
            "phi": None,
        }
    elif rotate == "varimax":
        _, R_rot, meta_v = varimax_kaiser_sweep(L)
        W_pre = W @ R_rot
        rot_diag = {
            "kaiser_normalized": meta_v["kaiser_normalized"],
            "sweeps": meta_v["sweeps"],
            "V_converged": meta_v["V_converged"],
            "kappa": None,
            "pattern": None,
            "structure": None,
            "phi": None,
        }
    elif rotate == "promax":
        L_var, R_v, meta_v = varimax_kaiser_sweep(L)
        prom = promax_fit(L_var, kappa=kappa)
        # Oblique transform from the varimax-rotated W.
        R_rot = R_v @ prom["transform"]
        W_pre = W @ R_rot
        rot_diag = {
            "kaiser_normalized": meta_v["kaiser_normalized"],
            "sweeps": meta_v["sweeps"],
            "V_converged": meta_v["V_converged"],
            "kappa": kappa,
            "pattern": prom["pattern"],
            "structure": prom["structure"],
            "phi": prom["phi"],
        }
    else:
        raise ValueError(
            f"rotate must be 'raw', 'varimax', or 'promax'; got {rotate!r}"
        )

    # -- Order by |corr(T_rot_i, y)| desc, sign-flip so corr > 0 -----------
    T_pre = Xs @ W_pre  # dim scores on un-deflated Xs (per roadmap)
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

    order = np.argsort(-corrs)  # descending |corr|
    R_rot = R_rot[:, order] * signs[order][np.newaxis, :]

    # Apply ordering/sign flips to the pre-ordered outputs.
    W_rot = W_pre[:, order] * signs[order][np.newaxis, :]
    T_rot = T_pre[:, order] * signs[order][np.newaxis, :]

    # If rotate == "promax", also reorder pattern/structure/phi to match.
    # phi transforms under both permutation and sign flip: when a factor's
    # sign is flipped, both its row and column in phi flip, so
    # phi_new[i,j] = signs[i]*signs[j]*phi_old[order[i], order[j]].
    if rotate == "promax":
        signs_ord = signs[order]
        rot_diag["pattern"] = rot_diag["pattern"][:, order] * signs_ord[np.newaxis, :]
        rot_diag["structure"] = rot_diag["structure"][:, order] * signs_ord[np.newaxis, :]
        rot_diag["phi"] = (
            rot_diag["phi"][np.ix_(order, order)] * np.outer(signs_ord, signs_ord)
        )

    rotation_meta = {
        "rotate": rotate, "R": R_rot,
        "order": order, "signs": signs[order],
        **rot_diag,
    }

    return {
        "W": W, "P": P, "Q": Q,
        "W_rot": W_rot,
        "T_rot": T_rot,
        "beta_combined": coef,
        "order": order,
        "signs": signs[order],
        "rotation_meta": rotation_meta,
    }
