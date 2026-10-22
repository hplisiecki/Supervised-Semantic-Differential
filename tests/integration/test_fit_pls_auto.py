"""fit_pls(k='auto') reports an honest k=1 confirmatory p-value."""
from __future__ import annotations

import dataclasses

import numpy as np
import plskit
import pytest

from ssdiff.utils.math import standardize


def _xy(ssd_instance, parity_df):
    X, has_vector, _ = ssd_instance._vectors()
    return X, parity_df["y"].to_numpy()[has_vector]


def _fit_and_compare(ssd_instance, parity_df, *, k_max=3, n_splits=20, seed=0):
    res = ssd_instance.fit_pls(
        parity_df["y"], k="auto", k_max=k_max, n_splits=n_splits, random_state=seed,
    )
    x, y = _xy(ssd_instance, parity_df)
    Xs, _, _ = standardize(x)
    ys, _, _ = standardize(y.reshape(-1, 1))
    ys = ys.ravel()
    direct = plskit.pls1_confirmatory_test(
        Xs, ys, 1,
        test_method="auto",
        args={"n_splits": n_splits},
        pre_standardized=True,
        seed=seed,
    )
    return res, float(direct.pvalue)


def test_fit_pls_auto_reports_honest_k1_pvalue(ssd_instance, parity_df):
    """stats.pvalue equals a direct k=1 test for any k_star."""
    res, direct_p = _fit_and_compare(ssd_instance, parity_df, k_max=3, n_splits=20, seed=0)
    assert abs(res.stats.pvalue - direct_p) < 1e-9, (
        f"stats.pvalue={res.stats.pvalue} != direct k=1 p={direct_p}"
    )


def test_fit_pls_auto_p_at_k_is_one(ssd_instance, parity_df):
    """p_at_k is always 1 for k='auto'."""
    res = ssd_instance.fit_pls(parity_df["y"], k="auto", k_max=3, n_splits=20, random_state=0)
    assert res.fit_info.p_at_k == 1


def test_fit_pls_auto_components_match_direct_fit(ssd_instance, parity_df):
    """β / r² match plskit.pls1_fit at the resolved k_star."""
    res = ssd_instance.fit_pls(parity_df["y"], k="auto", k_max=3, n_splits=20, random_state=0)
    x, y = _xy(ssd_instance, parity_df)
    Xs, _, X_scale = standardize(x)
    ys, _, _ = standardize(y.reshape(-1, 1))
    ys = ys.ravel()
    m = plskit.pls1_fit(
        Xs, ys, k=res.fit_info.n_components,
        pre_standardized=True, seed=0,
    )
    scale = np.where(X_scale > 1e-12, X_scale, 1.0)
    direct_beta = m.coef / scale
    cos = abs(
        float(np.dot(res.beta, direct_beta))
        / max(np.linalg.norm(res.beta) * np.linalg.norm(direct_beta), 1e-12)
    )
    assert cos > 0.95, f"|cos(beta, direct)|={cos:.4f}"


def test_k_auto_k_star_zero_raises(ssd_instance, parity_df, monkeypatch):
    real = plskit.pls1_find_k_optimal
    def fake(*a, **kw):
        return dataclasses.replace(real(*a, **kw), k_star=0)
    monkeypatch.setattr(plskit, "pls1_find_k_optimal", fake)
    with pytest.raises(ValueError, match="k_star"):
        ssd_instance.fit_pls(parity_df["y"], k="auto", n_splits=10)
