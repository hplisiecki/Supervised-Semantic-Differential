"""statsmodels names on fit_regression results: params, pvalues, fvalue, llr, resid, ..."""

from __future__ import annotations

import glmm
import numpy as np
import pandas as pd
import pytest

import ssdiff
from ssdiff.utils.math import chi2_sf, f_sf, pca_fit_transform, standardize

K = 3


def _ols(ssd, parity_df):
    """``y ~ cov + ssd()`` by numpy: design, coefficients, RSS and the kept rows."""
    keep = ssd._vectors()[1]
    y = parity_df["y"].to_numpy()[keep]
    z = pca_fit_transform(standardize(ssd._vectors()[0])[0], K)[0]
    X = np.column_stack([np.ones(len(y)), parity_df["cov"].to_numpy()[keep], z])
    b, rss = np.linalg.lstsq(X, y, rcond=None)[:2]
    return X, y, b, float(rss[0])


def test_ols_names_match_numpy(ssd_instance, parity_df):
    res = ssd_instance.fit_regression("y ~ cov + ssd()", data=parity_df, fixed_k=K)
    X, y, b, rss = _ols(ssd_instance, parity_df)
    n, p = X.shape
    se = np.sqrt(np.diag(rss / (n - p) * np.linalg.inv(X.T @ X)))

    assert list(res.params.index) == ["(Intercept)", "cov"]
    np.testing.assert_allclose(res.params, b[:2], rtol=1e-6)
    np.testing.assert_allclose(res.bse, se[:2], rtol=1e-6)
    np.testing.assert_allclose(res.tvalues, b[:2] / se[:2], rtol=1e-6)
    np.testing.assert_allclose(np.sqrt(np.diag(res.cov_params())), se[:2], rtol=1e-6)
    ci = res.conf_int(alpha=0.1)
    assert list(ci.columns) == [0, 1] and (ci[0] < res.params).all() and (res.params < ci[1]).all()

    assert (res.nobs, res.df_model, res.df_resid) == (n, p - 1, n - p)
    tss = float(np.sum((y - y.mean()) ** 2))
    assert res.rsquared == pytest.approx(1 - rss / tss, rel=1e-6)
    f = (tss - rss) / (p - 1) / (rss / (n - p))
    assert res.fvalue == pytest.approx(f, rel=1e-6)
    assert res.f_pvalue == pytest.approx(f_sf(f, p - 1, n - p), rel=1e-4)
    assert np.isnan(res.llr) and np.isnan(res.llr_pvalue)
    np.testing.assert_allclose(res.fittedvalues, X @ b, rtol=1e-6)
    np.testing.assert_allclose(res.resid, y - X @ b, atol=1e-8)
    assert res.random_effects == {}


def test_text_term_rows_are_wald_tests_of_each_key(ssd_instance, parity_df):
    res = ssd_instance.fit_regression("y ~ cov + group*ssd()", data=parity_df, fixed_k=K)
    keys = res.keys()
    assert list(res.pvalues.index) == list(res.params.index) + keys
    names = list(res.model.names)
    for key in keys:
        prefix = key.removesuffix("ssd(text)")
        idx = [i for i, n in enumerate(names) if n.startswith(f"{prefix}_ssd_pca_")]
        assert len(idx) == K
        wald = res.model.contrast_test(np.eye(len(names))[idx])
        assert res.pvalues[key] == pytest.approx(float(wald["Pr(>F)"][0]), rel=1e-9)
    # Every level shows the same values: one glmm fit backs the whole regression.
    leaf = res[keys[1]]
    pd.testing.assert_series_equal(leaf.pvalues, res.pvalues)
    assert (leaf.fvalue, leaf.df_model, leaf.aic) == (res.fvalue, res.df_model, res.aic)


def test_f_pvalue_is_the_comparison_with_the_intercept_only_model(two, parity_df):
    essay, diary = two
    res = ssdiff.fit_regression("y ~ cov + ssd(essay) + ssd(diary)", parity_df,
                                essay=essay, diary=diary, fixed_k=K)
    keep = essay._vectors()[1] & diary._vectors()[1]
    cols = {"y": parity_df["y"].to_numpy()[keep], "cov": parity_df["cov"].to_numpy()[keep]}
    for name, text in (("e", essay), ("d", diary)):
        z = pca_fit_transform(standardize(text._x_rows(keep))[0], K)[0]
        cols.update({f"{name}{i}": z[:, i] for i in range(K)})
    rhs = " + ".join(c for c in cols if c != "y")
    cmp = glmm.compare(cols, "y ~ 1", f"y ~ {rhs}")
    assert res.f_pvalue == pytest.approx(float(cmp.table["Pr(>F)"][1]), rel=1e-6)
    assert res["essay"].f_pvalue == res.f_pvalue
    assert list(res.pvalues.index)[-2:] == ["ssd(essay)", "ssd(diary)"]


def test_llr_for_a_binomial_model(ssd_instance, parity_df):
    df = parity_df.assign(hi=(parity_df["y"] > parity_df["y"].median()).astype(float))
    res = ssd_instance.fit_regression("hi ~ cov + ssd()", data=df, fixed_k=K, family="binomial")
    keep = ssd_instance._vectors()[1]
    null = glmm.fit({"hi": df["hi"].to_numpy()[keep]}, "hi ~ 1", family="binomial")
    llr = 2 * (res.llf - null.loglik)
    assert res.llr == pytest.approx(llr, rel=1e-6)
    assert res.llr_pvalue == pytest.approx(chi2_sf(llr, res.df_model), rel=1e-6)
    assert np.isnan(res.fvalue) and np.isnan(res.rsquared)


def test_mixed_model_random_effects_and_resid_index(ssd_instance, parity_df):
    ids = np.arange(len(parity_df)) % 12
    df = parity_df.assign(id=ids.astype(str)).set_axis(np.arange(len(parity_df)) + 1000)
    res = ssd_instance.fit_regression("y ~ cov + ssd() + (1 | id)", data=df, fixed_k=K)
    re = res.random_effects
    assert list(re) == ["id"] and re["id"].shape == (12, 1)
    assert list(re["id"].columns) == ["(Intercept)"]
    assert list(res.resid.index) == list(df.index[ssd_instance._vectors()[1]])
    assert np.isfinite(res.llr) and np.isnan(res.fvalue)


def test_no_intercept_has_no_whole_model_test(ssd_instance, parity_df):
    res = ssd_instance.fit_regression("y ~ 0 + cov + ssd()", data=parity_df, fixed_k=K)
    assert all(np.isnan(v) for v in (res.fvalue, res.f_pvalue, res.llr, res.llr_pvalue))
    assert res.df_model == K + 1
