"""PLS backend — rotation of a PLS3 saliency pair.

Owns ``pls3_rotate`` only. Fitting, significance testing and k-selection
live in ``ssdiff.ssd`` and call ``plskit`` directly.
"""

from __future__ import annotations

import numpy as np

import plskit


__all__ = ("pls3_rotate",)


def pls3_rotate(U, V, sigma, *, rotate, L=None):
    """Rotate a PLS3 saliency pair with one shared orthogonal ``R``.

    ``U_rot = U @ R`` and ``V_rot = V @ R``; ``c[i] = (R' Σ² R)_ii`` is the
    squared cross-covariance of dimension i and sums to ``Σσ²`` for any R.
    Varimax modes reorder by ``c``; every mode pins the sign so the
    largest-|·| entry of each ``V_rot`` column is positive (plskit pins on
    ``U``, an arbitrary embedding coordinate).
    """
    k = U.shape[1]
    if rotate is None:
        R = np.eye(k)
        meta = {"sweeps": 0, "V_converged": 0.0, "kaiser_normalized": False}
    else:
        rot = plskit.rotate(U, method="varimax", L=L)
        R = np.asarray(rot.spec.R, dtype=np.float64)
        meta = {
            "sweeps": rot.spec.sweeps,
            "V_converged": rot.spec.V_converged,
            "kaiser_normalized": rot.spec.args["kaiser_normalize"],
        }
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
