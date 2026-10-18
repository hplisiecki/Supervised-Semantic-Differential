"""PLS backend — rotation of PLS1 weights and of a PLS3 saliency pair.

Owns ``word_target``, ``pls1_rotate`` and ``pls3_rotate``. Fitting,
significance testing and k-selection live in ``ssdiff.ssd`` and call
``plskit`` directly.
"""

from __future__ import annotations

import numpy as np
import plskit

__all__ = ("pls1_rotate", "pls3_rotate", "word_target")


def word_target(vectors, W, X_mean, X_scale):
    """Varimax target for ``rotate="words"``: the leading vocabulary rows projected onto ``W``.

    Each of the first ``_RESTRICT_VOCAB`` rows of ``vectors`` is centred by
    ``X_mean``, scaled by ``X_scale`` (as the doc vectors were) and projected
    onto the columns of ``W`` (D × k). Returns an (n_rot, k) float64 array.
    """
    from ssdiff.backends.pca_sweep import _RESTRICT_VOCAB

    emb_dtype = vectors.dtype
    n_rot = min(_RESTRICT_VOCAB, int(vectors.shape[0]))
    # Fold standardisation into the matmul instead of materialising a
    # standardised (n_rot, D) copy of the vocabulary.
    M = (W / X_scale.astype(emb_dtype, copy=False)[:, None]).astype(emb_dtype, copy=False)
    L = vectors[:n_rot] @ M
    L -= X_mean.astype(emb_dtype, copy=False) @ M
    return L.astype(np.float64, copy=False)


def _varimax(W, L):
    """Orthogonal ``R`` (k × k) from varimax of ``W`` against target ``L``, and plskit's diagnostics."""
    rot = plskit.rotate(W, method="varimax", L=L)
    meta = {
        "sweeps": rot.spec.sweeps,
        "V_converged": rot.spec.V_converged,
        "kaiser_normalized": rot.spec.args["kaiser_normalize"],
    }
    return np.asarray(rot.spec.R, dtype=np.float64), meta


_NO_ROTATION_META = {"sweeps": 0, "V_converged": 0.0, "kaiser_normalized": False}


def pls1_rotate(W, Xs, ys, *, rotate, L=None):
    """Rotate PLS1 weights ``W`` (D × N, orthonormal) and score each dimension.

    ``rotate="words"`` runs varimax against ``L``, orders the dimensions by
    ``|corr(Xs @ W_rot[:, i], ys)|``, descending, and flips each so that
    correlation is positive. ``rotate=None`` keeps the NIPALS order and
    signs: NIPALS gives ``w_1 ∝ Xs' ys`` and orthonormal W, so dimensions
    2…N have zero covariance with ``ys``, and ordering or signing them by it
    would follow rounding noise.

    ``cov_share[i]`` is ``(W_rot[:, i]' Xs' ys)²`` over its sum across i;
    the sum, ``‖W' Xs' ys‖²``, is the same for every rotation.
    """
    k = W.shape[1]
    if rotate is None:
        R, meta = np.eye(k), dict(_NO_ROTATION_META)
    else:
        R, meta = _varimax(W, L)
    W_rot = W @ R
    score_corr = np.array([np.corrcoef(Xs @ W_rot[:, i], ys)[0, 1] for i in range(k)])
    if rotate is None:
        order, signs = np.arange(k), np.ones(k)
    else:
        order = np.argsort(-np.abs(score_corr), kind="stable")
        signs = np.where(score_corr[order] < 0, -1.0, 1.0)
    W_rot, R = W_rot[:, order] * signs, R[:, order] * signs
    c = (W_rot.T @ (Xs.T @ ys)) ** 2
    return {
        "W_rot": W_rot, "R": R, "order": order, "signs": signs,
        "score_corr": score_corr[order] * signs, "cov_share": c / c.sum(),
        "rotation_meta": meta,
    }


def pls3_rotate(U, V, sigma, *, rotate, L=None):
    """Rotate a PLS3 saliency pair with one shared orthogonal ``R``.

    ``U_rot = U @ R`` and ``V_rot = V @ R``; ``c[i] = (R' Σ² R)_ii`` is the
    squared cross-covariance of dimension i and sums to ``Σσ²`` for any R.
    Varimax modes reorder by ``c``; every mode pins the sign so the
    largest-|·| entry of each ``V_rot`` column is positive (plskit pins on
    ``U``, an arbitrary embedding coordinate).

    ``rotate="outcomes"`` with ``L=None`` targets the outcome loadings ``V``.
    When V is square (k = q) it takes ``R = V'`` without varimax: the
    rotation varimax converges to, each dimension loading 1 on one outcome.
    """
    k = U.shape[1]
    if rotate is None:
        R, meta = np.eye(k), dict(_NO_ROTATION_META)
    elif rotate == "outcomes" and L is None and V.shape[0] == k:
        R, meta = V.T.copy(), dict(_NO_ROTATION_META)
    else:
        R, meta = _varimax(U, V if L is None else L)
    U_rot, V_rot = U @ R, V @ R
    c = np.diag(R.T @ np.diag(sigma ** 2) @ R).copy()
    order = np.arange(k) if rotate is None else np.argsort(-c, kind="stable")
    U_rot, V_rot, R, c = U_rot[:, order], V_rot[:, order], R[:, order], c[order]
    idx = np.argmax(np.abs(V_rot), axis=0)
    signs = np.where(V_rot[idx, np.arange(k)] < 0, -1.0, 1.0)
    return {
        "U_rot": U_rot * signs, "V_rot": V_rot * signs, "R": R * signs,
        "c": c, "order": order, "signs": signs, "rotation_meta": meta,
    }
