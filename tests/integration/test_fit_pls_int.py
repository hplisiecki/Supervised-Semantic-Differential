"""fit_pls(k=int) uses pls1_confirmatory_test at the requested k."""
from __future__ import annotations

import plskit

from ssdiff.utils.math import standardize


def test_fit_pls_int_pvalue_matches_direct_confirmatory(ssd_instance, parity_df):
    res = ssd_instance.fit_pls(parity_df["y"], k=2, n_splits=20, random_state=0)
    X, has_vector, _ = ssd_instance._vectors()
    Xs, _, _ = standardize(X)
    ys, _, _ = standardize(parity_df["y"].to_numpy()[has_vector].reshape(-1, 1))
    ys = ys.ravel()
    direct = plskit.pls1_confirmatory_test(
        Xs, ys, 2,
        test_method="auto",
        args={"n_splits": 20},
        pre_standardized=True,
        seed=0,
    )
    assert abs(res.stats.pvalue - float(direct.pvalue)) < 1e-9
    assert res.fit_info.p_at_k == 2
    assert res.fit_info.n_components == 2
