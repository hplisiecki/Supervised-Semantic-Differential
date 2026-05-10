"""fit_pls(k='auto') reports an honest k=1 confirmatory split_nb p-value."""
from __future__ import annotations

import numpy as np
import plskit

from ssdiff.utils.math import standardize


def _fit_and_compare(ssd_instance, *, k_max=3, n_splits=20, seed=0):
    res = ssd_instance.fit_pls(
        k="auto", k_max=k_max, n_splits=n_splits, random_state=seed,
    )
    Xs, _, _ = standardize(ssd_instance.x)
    ys, _, _ = standardize(ssd_instance.y.reshape(-1, 1))
    ys = ys.ravel()
    direct = plskit.pls1_confirmatory_test(
        Xs, ys, 1,
        method="split_nb",
        args={"n_splits": n_splits},
        pre_standardized=True,
        seed=seed,
    )
    return res, float(direct.pvalue)


def test_fit_pls_auto_reports_honest_k1_pvalue(ssd_instance):
    """When k_star > 1, stats.pvalue equals a direct k=1 split_nb test."""
    res, direct_p = _fit_and_compare(ssd_instance, k_max=3, n_splits=20, seed=0)
    if res.fit_info.n_components > 1:
        assert abs(res.stats.pvalue - direct_p) < 1e-9, (
            f"stats.pvalue={res.stats.pvalue} != direct k=1 p={direct_p}"
        )


def test_fit_pls_auto_p_at_k_is_one(ssd_instance):
    """p_at_k is always 1 for k='auto'."""
    res = ssd_instance.fit_pls(k="auto", k_max=3, n_splits=20, random_state=0)
    assert res.fit_info.p_at_k == 1


def test_fit_pls_auto_components_match_direct_fit(ssd_instance):
    """β / r² match plskit.pls1_fit at the resolved k_star."""
    res = ssd_instance.fit_pls(k="auto", k_max=3, n_splits=20, random_state=0)
    Xs, _, X_scale = standardize(ssd_instance.x)
    ys, _, _ = standardize(ssd_instance.y.reshape(-1, 1))
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
