"""fit_multipls(k='auto'): per-dim p-values come from the same find_k chain
remapped via mpls_fit's order; combined.pvalue == container.stats.pvalue."""
from __future__ import annotations

import numpy as np
import plskit

from ssdiff.utils.math import standardize


def test_fit_multipls_auto_per_dim_pvalues_shape_and_source(ssd_instance):
    res = ssd_instance.fit_multipls(
        k="auto", k_max=3, n_splits=20, random_state=0,
    )
    n = res.n_components
    assert res.per_dim_pvalues.shape == (n,)

    # Reproduce the find_k call used inside fit_multipls.
    Xs, _, _ = standardize(ssd_instance.x)
    ys, _, _ = standardize(ssd_instance.y.reshape(-1, 1))
    ys = ys.ravel()
    fk = plskit.pls1_find_k_optimal(
        Xs, ys, max(min(3, Xs.shape[0] - 1, Xs.shape[1]), 1),
        selector="r2_se",
        diagnostic="split_nb",
        args={"n_folds": 5, "n_splits": 20},
        pre_standardized=True,
        seed=0,
    )
    chain = np.asarray(fk.pvalues[:n], dtype=float)

    # The container does not store `order` directly; recover via pls_info.
    order = np.asarray(next(iter(res.pls_info))["order"], dtype=int)
    expected = chain[order]
    np.testing.assert_allclose(res.per_dim_pvalues, expected, rtol=0, atol=0)


def test_fit_multipls_auto_leaf_pvalues_match_container(ssd_instance):
    res = ssd_instance.fit_multipls(
        k="auto", k_max=3, n_splits=20, random_state=0,
    )
    for i in range(res.n_components):
        leaf = res[f"dim-{i+1}"]
        assert leaf.pvalue == float(res.per_dim_pvalues[i])
    assert res["combined"].pvalue == float(res.stats.pvalue)
