"""result.test(...) rerun: auto only; method-name argument raises."""
from __future__ import annotations

import pytest


def test_pls_test_rerun_propagates_pvalue(ssd_instance, parity_df):
    res = ssd_instance.fit_pls(parity_df["y"], k=2, n_splits=20, random_state=0)
    old_p = res.stats.pvalue
    res.test(n_splits=40, seed=42)
    assert res.test.pvalue is not None
    assert res.stats.pvalue == res.test.pvalue
    assert isinstance(old_p, float)


def test_pls_test_rerun_rejects_method_name(ssd_instance, parity_df):
    res = ssd_instance.fit_pls(parity_df["y"], k=2, n_splits=20, random_state=0)
    with pytest.raises(TypeError, match="auto"):
        res.test("raw_perm", n_splits=10)
    with pytest.raises(TypeError, match="auto"):
        res.test("score")
