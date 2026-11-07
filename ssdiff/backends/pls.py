"""PLS backend — gradient leaves of one beta and rotation of a PLS3 saliency pair.

Owns ``split_beta`` (the split of a beta, through ``plskit.profiles`` or
``plskit.factors``), ``split_components`` (the split along PLS1 weights),
``gradient_dims``, ``word_target`` and ``pls3_rotate``.
Fitting, significance testing and k-selection live in ``ssdiff.ssd`` and call
``plskit`` directly.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import plskit

__all__ = ("gradient_dims", "pls3_rotate", "split_beta", "split_components", "word_target")


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


def split_beta(model_or_coef, X, n, gradients):
    """The *n* parts of a fit's standardized coefficient, as a ``plskit.FactorsResult``.

    ``"factors"`` is ``plskit.factors``: *n* consecutive blocks along the PCA
    basis of ``X`` with about equal shares of the squared coefficients, in
    block order. ``"profiles"`` is ``plskit.profiles``: the same parts rotated
    by varimax on the scores ``X W``, largest first. ``X`` is standardized.
    """
    if gradients == "profiles":
        return plskit.profiles(model_or_coef, X, n, pre_standardized=True).factors
    return plskit.factors(model_or_coef, X, n, pre_standardized=True)


def split_components(W, coef, L=None):
    """``coef`` split along the k PLS1 weight columns ``W`` (D × k), shaped like a ``plskit.FactorsResult``.

    Part i is ``F_i = w_i (w_i' coef)``. PLS1 weights are orthonormal and
    ``coef`` lies in their span, so the parts sum to ``coef``; a rotation
    ``W R`` with R orthogonal keeps both. With a varimax target ``L``
    (``word_target``) the columns are rotated against it and ordered by size,
    largest first; without one they keep the PLS order. The sign of ``w_i``
    does not matter: ``F_i' coef = (w_i' coef)² ≥ 0``.
    """
    spec = None
    if L is not None:
        rot = plskit.rotate(W, method="varimax", L=L)
        W, spec = rot.W_rot, rot.spec
    proj = W.T @ coef
    if spec is not None:
        order = np.argsort(-np.abs(proj), kind="stable")
        W, proj = W[:, order], proj[order]
    return SimpleNamespace(F=W * proj, W=W, coef=coef, size=np.abs(proj), rotation_spec=spec)


def gradient_dims(f, Xs, ys, scale, rotate="docs"):
    """The ``dims`` dict the result classes read, from a ``plskit.FactorsResult``.

    The columns of ``f.F`` sum to ``f.coef`` and all are divided by the same
    ``scale``, so the leaves' betas sum to ``f.coef / scale``, the fit's beta.
    A rotated ``f`` has its leaves ordered by ``f.size``, in standardized
    space, largest first; an unrotated one is in block order. Each leaf keeps
    the model's sign: ``F_i' coef = size_i² ≥ 0``.
    ``score_corr`` is reported as it comes and can be negative.

    ``cov_share[i]`` is ``(w_i' Xs' ys)²`` over its sum across i, with
    ``w_i = f.W[:, i]``. ``rotate`` names the varimax target of a rotated ``f``.
    """
    # unit_vector turns a zero beta into zeros, which would otherwise pass silently.
    if np.any(f.size <= 1e-12 * np.linalg.norm(f.coef)):
        raise ValueError("ssd_gradients: a gradient is zero. Use a smaller ssd_gradients.")
    scores = Xs @ f.F
    c = (f.W.T @ (Xs.T @ ys)) ** 2
    spec = f.rotation_spec
    return {
        "beta": f.F / scale[:, None],
        "score_corr": np.array([np.corrcoef(s, ys)[0, 1] for s in scores.T]),
        "cov_share": c / c.sum(),
        "rotate": None if spec is None else rotate,
        "rotation_meta": dict(_NO_ROTATION_META) if spec is None else {
            "sweeps": spec.sweeps,
            "V_converged": spec.V_converged,
            "kaiser_normalized": spec.args["kaiser_normalize"],
        },
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
