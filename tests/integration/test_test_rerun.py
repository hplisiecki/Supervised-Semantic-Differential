"""result.test(...) rerun: split_nb only; method-name argument raises."""
from __future__ import annotations

import pytest


def test_pls_test_rerun_propagates_pvalue(ssd_instance):
    res = ssd_instance.fit_pls(k=2, n_splits=20, random_state=0)
    old_p = res.stats.pvalue
    res.test(n_splits=40, seed=42)
    assert res.test.pvalue is not None
    assert res.stats.pvalue == res.test.pvalue
    assert isinstance(old_p, float)


def test_pls_test_rerun_rejects_method_name(ssd_instance):
    res = ssd_instance.fit_pls(k=2, n_splits=20, random_state=0)
    with pytest.raises(TypeError, match="split_nb"):
        res.test("raw_perm", n_splits=10)
    with pytest.raises(TypeError, match="split_nb"):
        res.test("score")


def test_multipls_test_rerun_signature(ssd_instance):
    res = ssd_instance.fit_multipls(k=2, n_splits=20, random_state=0)
    res.test(n_splits=40, seed=42)
    assert res.stats.pvalue == res.test.pvalue


def test_multipls_test_rerun_rejects_method_name(ssd_instance):
    res = ssd_instance.fit_multipls(k=2, n_splits=20, random_state=0)
    with pytest.raises(TypeError, match="split_nb"):
        res.test("raw_perm", n_splits=10)
