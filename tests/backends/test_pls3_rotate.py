import numpy as np
import pytest
from ssdiff.backends.pls import pls3_rotate


def _svd(seed=0, D=30, q=6, k=3):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(D, q))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k], Vt.T[:, :k], s[:k]


@pytest.mark.parametrize("rotate", [None, "outcomes"])
def test_orthonormal_and_total_preserved(rotate):
    U, V, s = _svd()
    out = pls3_rotate(U, V, s, rotate=rotate, L=V)
    k = U.shape[1]
    np.testing.assert_allclose(out["U_rot"].T @ out["U_rot"], np.eye(k), atol=1e-10)
    np.testing.assert_allclose(out["V_rot"].T @ out["V_rot"], np.eye(k), atol=1e-10)
    np.testing.assert_allclose(out["c"].sum(), (s ** 2).sum(), rtol=1e-12)
    np.testing.assert_allclose(out["U_rot"], U @ out["R"], atol=1e-12)
    np.testing.assert_allclose(out["V_rot"], V @ out["R"], atol=1e-12)


def test_c_is_squared_cross_cov_per_dim():
    U, V, s = _svd()
    A = U @ np.diag(s) @ V.T
    out = pls3_rotate(U, V, s, rotate="outcomes", L=V)
    per_dim = np.sum((A.T @ out["U_rot"]) ** 2, axis=0)
    np.testing.assert_allclose(out["c"], per_dim, rtol=1e-10)


def test_rotated_order_descending():
    U, V, s = _svd()
    c = pls3_rotate(U, V, s, rotate="outcomes", L=V)["c"]
    assert np.all(np.diff(c) <= 1e-12)


@pytest.mark.parametrize("rotate", [None, "outcomes"])
def test_sign_rule_largest_V_entry_positive(rotate):
    U, V, s = _svd()
    Vr = pls3_rotate(-U, -V, s, rotate=rotate, L=-V)["V_rot"]
    idx = np.argmax(np.abs(Vr), axis=0)
    assert np.all(Vr[idx, np.arange(Vr.shape[1])] > 0)


def test_none_is_identity_up_to_sign():
    U, V, s = _svd()
    out = pls3_rotate(U, V, s, rotate=None, L=None)
    np.testing.assert_allclose(np.abs(out["R"]), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(out["c"], s ** 2, rtol=1e-12)
    assert out["rotation_meta"]["sweeps"] == 0
