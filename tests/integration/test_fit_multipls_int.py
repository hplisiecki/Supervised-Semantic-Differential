"""fit_multipls(k=int): the container p comes from one confirmatory_test at
n_comp; per-dim p-values come from n_comp individual confirmatory_test calls
(k=1..n_comp), remapped via order. No find_k_optimal call in this branch."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import plskit

from ssdiff.utils.math import standardize


def test_fit_multipls_int_call_counts(ssd_instance):
    """k=int branch: n_comp+1 confirmatory calls, 0 find_k_optimal calls."""
    real_ct = plskit.pls1_confirmatory_test
    real_fk = plskit.pls1_find_k_optimal
    counters = {"ct": 0, "fk": 0}

    def counting_ct(*a, **kw):
        counters["ct"] += 1
        return real_ct(*a, **kw)

    def counting_fk(*a, **kw):
        counters["fk"] += 1
        return real_fk(*a, **kw)

    with patch("plskit.pls1_confirmatory_test", new=counting_ct), \
         patch("plskit.pls1_find_k_optimal", new=counting_fk):
        res = ssd_instance.fit_multipls(
            k=2, n_splits=20, random_state=0,
        )

    # 1 main confirmatory at k=n_comp, plus n_comp confirmatory chain calls.
    assert counters["ct"] == res.n_components + 1, counters
    assert counters["fk"] == 0, counters
    assert res.n_components == 2


def test_fit_multipls_int_container_pvalue(ssd_instance):
    """Container stats.pvalue matches a direct confirmatory at n_comp."""
    res = ssd_instance.fit_multipls(k=2, n_splits=20, random_state=0)

    Xs, _, _ = standardize(ssd_instance.x)
    ys, _, _ = standardize(ssd_instance.y.reshape(-1, 1))
    ys = ys.ravel()
    direct = plskit.pls1_confirmatory_test(
        Xs, ys, 2,
        method="split_nb",
        args={"n_splits": 20},
        pre_standardized=True,
        seed=0,
    )
    assert abs(res.stats.pvalue - float(direct.pvalue)) < 1e-9


def test_fit_multipls_int_per_dim_matches_chain_reorder(ssd_instance):
    """per_dim_pvalues equals chain (k=1..n_comp via individual
    confirmatory_test calls) reordered by mpls_fit's order permutation."""
    res = ssd_instance.fit_multipls(k=2, n_splits=20, random_state=0)

    Xs, _, _ = standardize(ssd_instance.x)
    ys, _, _ = standardize(ssd_instance.y.reshape(-1, 1))
    ys = ys.ravel()
    chain = np.array([
        float(plskit.pls1_confirmatory_test(
            Xs, ys, ki,
            method="split_nb",
            args={"n_splits": 20},
            pre_standardized=True,
            seed=0,
        ).pvalue)
        for ki in (1, 2)
    ], dtype=float)
    order = np.asarray(next(iter(res.pls_info))["order"], dtype=int)
    np.testing.assert_allclose(res.per_dim_pvalues, chain[order])
