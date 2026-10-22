"""fit_pls against plskit: k='auto' and k=int p-values and fits, and result.test() reruns."""
from __future__ import annotations

import dataclasses

import numpy as np
import plskit
import pytest

from ssdiff.ssd import SSD
from ssdiff.utils.math import standardize
from tests.conftest import _make_large_docs


def _direct(ssd, y):
    """Standardized X and y as fit_pls feeds them to plskit, and X's scale."""
    X, has_vector, _ = ssd._vectors()
    Xs, _, scale = standardize(X)
    ys = standardize(np.asarray(y, float)[has_vector].reshape(-1, 1))[0].ravel()
    return Xs, ys, np.where(scale > 1e-12, scale, 1.0)


@pytest.fixture(scope="module")
def planted(tiny_kv_large, lexicon):
    """y built from two embedding dimensions, so k='auto' picks more than one component."""
    docs = _make_large_docs(150, np.random.default_rng(0))
    ssd = SSD(tiny_kv_large, docs, lexicon)
    X, has_vector, _ = ssd._vectors()
    rng = np.random.default_rng(1)
    y = np.full(len(docs), np.nan)
    y[has_vector] = X[:, 0] + X[:, 1] + rng.normal(size=len(X)) * X[:, 0].std()
    return ssd, y


def test_fit_pls_auto_reports_honest_k1_pvalue(ssd_instance, parity_df):
    """k='auto' reports a fresh k=1 confirmatory test, not the diagnostic chain's p."""
    res = ssd_instance.fit_pls(parity_df["y"], k="auto", k_max=3, n_splits=20, random_state=0)
    Xs, ys, _ = _direct(ssd_instance, parity_df["y"])
    direct = plskit.pls1_confirmatory_test(
        Xs, ys, 1, test_method="auto", args={"n_splits": 20},
        pre_standardized=True, seed=0)
    assert res.test.name == direct.test_method
    assert res.stats.pvalue == pytest.approx(float(direct.pvalue), abs=1e-9)
    assert res.fit_info.p_at_k == 1


def test_fit_pls_auto_components_match_direct_fit(planted):
    ssd, y = planted
    res = ssd.fit_pls(y, k="auto", k_max=4, n_splits=20, random_state=0)
    k = res.fit_info.n_components
    assert k > 1
    Xs, ys, scale = _direct(ssd, y)
    m = plskit.pls1_fit(Xs, ys, k=k, pre_standardized=True, seed=0)
    np.testing.assert_allclose(res.beta, m.coef / scale, rtol=1e-8, atol=1e-12)
    # PLS fitted values project onto the score space, so r2 is the squared correlation.
    assert res.stats.r2 == pytest.approx(np.corrcoef(res.y, res.x @ res.beta)[0, 1] ** 2)


def test_k_auto_k_star_zero_raises(ssd_instance, parity_df, monkeypatch):
    real = plskit.pls1_find_k_optimal
    def fake(*a, **kw):
        return dataclasses.replace(real(*a, **kw), k_star=0)
    monkeypatch.setattr(plskit, "pls1_find_k_optimal", fake)
    with pytest.raises(ValueError, match="k_star"):
        ssd_instance.fit_pls(parity_df["y"], k="auto", n_splits=10)


def test_fit_pls_int_pvalue_matches_direct_confirmatory(ssd_instance, parity_df):
    res = ssd_instance.fit_pls(parity_df["y"], k=2, n_splits=20, random_state=0)
    Xs, ys, _ = _direct(ssd_instance, parity_df["y"])
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
    # Fit state lives on the result; the SSD object stays reusable.
    assert not hasattr(ssd_instance, "fit_info")


def test_pls_test_rerun_matches_direct_test(ssd_instance, parity_df):
    res = ssd_instance.fit_pls(parity_df["y"], k=2, n_splits=20, random_state=0)
    res.test(n_splits=40, seed=42)
    Xs, ys, _ = _direct(ssd_instance, parity_df["y"])
    direct = plskit.pls1_confirmatory_test(
        Xs, ys, 2, test_method="auto", args={"n_splits": 40},
        pre_standardized=True, seed=42)
    assert res.test.pvalue == pytest.approx(float(direct.pvalue))
    assert res.test.params["n_splits"] == 40
    assert res.stats.pvalue == res.test.pvalue


def test_pls_test_rerun_rejects_method_name(ssd_instance, parity_df):
    res = ssd_instance.fit_pls(parity_df["y"], k=2, n_splits=20, random_state=0)
    with pytest.raises(TypeError, match="auto"):
        res.test("raw_perm", n_splits=10)
    with pytest.raises(TypeError, match="auto"):
        res.test("score")
