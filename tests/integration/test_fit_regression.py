"""Covariates: SSD.fit_regression (PCA regression, mixed models via glmm) and SSD.fit_pls(formula=...)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ssdiff.corpus import Corpus
from ssdiff.results.format import fmt_p, fmt_r
from ssdiff.ssd import SSD
from ssdiff.utils.math import f_sf, pca_fit_transform, standardize


def _ssd(kv, docs, y, lexicon, **kw):
    return SSD(kv, Corpus(docs, pretokenized=True, lang="pl"), y, lexicon, **kw)


@pytest.fixture
def frame(large_y):
    """One row per document of ``large_docs``: outcome, a covariate, 10 people × 5 texts."""
    rng = np.random.default_rng(7)
    n = len(large_y)
    return pd.DataFrame({
        "rating": large_y,
        "arousal": rng.normal(size=n),
        "age": rng.normal(40, 10, size=n),
        "ID": pd.Categorical(np.repeat([f"p{i}" for i in range(10)], 5)),
    })


def _reference_ols(ssd, k):
    """PCA + OLS on standardized y, computed directly: beta, r2, r2_adj, F-test p."""
    ys = standardize(ssd.y.reshape(-1, 1))[0].ravel()
    Xs, _, X_scale = standardize(ssd.x)
    z, comps, _ = pca_fit_transform(Xs, k)
    w = np.linalg.solve(z.T @ z, z.T @ ys)
    resid = ys - z @ w
    n = len(ys)
    r2 = 1 - resid @ resid / (ys @ ys)
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    f = (r2 / k) / ((1 - r2) / (n - k - 1))
    return comps.T @ w / X_scale, r2, r2_adj, f_sf(f, k, n - k - 1)


def test_no_formula_matches_pca_ols(tiny_kv, large_docs, large_y, lexicon):
    ssd = _ssd(tiny_kv, large_docs, large_y, lexicon)
    res = ssd.fit_regression(fixed_k=4)
    beta, r2, r2_adj, p = _reference_ols(ssd, 4)
    np.testing.assert_allclose(res.beta, beta, rtol=1e-8, atol=1e-12)
    assert res.stats.r2 == pytest.approx(r2, rel=1e-8)
    assert res.stats.r2_adj == pytest.approx(r2_adj, rel=1e-8)
    assert res.stats.pvalue == pytest.approx(p, rel=1e-6)
    assert res.test.name == "f_test"
    assert res.stats.backend == "PCA+OLS"
    assert res.model.nobs == ssd.n_kept


def test_sweep_k_used(tiny_kv, large_docs, large_y, lexicon):
    ssd = _ssd(tiny_kv, large_docs, large_y, lexicon)
    res = ssd.fit_regression(k_min=2, k_max=6, k_step=1)
    assert res.pca_k == res.sweep_result.best_k
    assert res.stats.r2 == pytest.approx(_reference_ols(ssd, res.pca_k)[1], rel=1e-8)


def test_fit_ols_warns_and_matches(tiny_kv, large_docs, large_y, lexicon):
    ssd = _ssd(tiny_kv, large_docs, large_y, lexicon)
    with pytest.warns(FutureWarning, match="fit_regression"):
        old = ssd.fit_ols(fixed_k=4)
    new = ssd.fit_regression(fixed_k=4)
    np.testing.assert_allclose(old.beta, new.beta)
    assert old.stats.pvalue == new.stats.pvalue


# --- y and data ------------------------------------------------------------

def test_y_column_name(tiny_kv, large_docs, large_y, lexicon, frame):
    ssd = _ssd(tiny_kv, large_docs, "rating", lexicon, data=frame)
    np.testing.assert_allclose(ssd.y, large_y)
    assert len(ssd.data) == ssd.n_kept


def test_y_column_list(tiny_kv, large_docs, lexicon, frame):
    ssd = _ssd(tiny_kv, large_docs, ["rating", "arousal"], lexicon, data=frame)
    assert ssd.y.shape == (len(frame), 2)
    assert ssd.outcome_names == ["rating", "arousal"]


def test_y_column_list_from_dict(tiny_kv, large_docs, lexicon, frame):
    data = {c: frame[c].to_numpy() for c in ("rating", "arousal")}
    ssd = _ssd(tiny_kv, large_docs, ["rating", "arousal"], lexicon, data=data)
    assert ssd.y.shape == (len(frame), 2)
    assert ssd.outcome_names == ["rating", "arousal"]


def test_y_name_without_data_raises(tiny_kv, large_docs, lexicon):
    with pytest.raises(ValueError, match="data"):
        _ssd(tiny_kv, large_docs, "rating", lexicon)


def test_data_length_mismatch_raises(tiny_kv, large_docs, large_y, lexicon, frame):
    with pytest.raises(ValueError, match="len"):
        _ssd(tiny_kv, large_docs, large_y, lexicon, data=frame.iloc[:-1])


def test_data_follows_dropped_documents(tiny_kv, large_docs, large_y, lexicon, frame):
    y = large_y.copy()
    y[[0, 7]] = np.nan
    data = {c: frame[c].to_numpy() for c in frame.columns}
    ssd = _ssd(tiny_kv, large_docs, y, lexicon, data=data)
    np.testing.assert_allclose(ssd.data["age"], np.delete(frame["age"].to_numpy(), [0, 7]))
    df_ssd = _ssd(tiny_kv, large_docs, y, lexicon, data=frame)
    assert df_ssd.data["ID"].dtype == frame["ID"].dtype
    assert list(df_ssd.data.index) == list(range(ssd.n_kept))


# --- formulas --------------------------------------------------------------

@pytest.mark.parametrize("formula", ["~ age", "y ~ age", "rating ~ age", "age"])
def test_formula_lhs_allowed(tiny_kv, large_docs, lexicon, frame, formula):
    ssd = _ssd(tiny_kv, large_docs, "rating", lexicon, data=frame)
    res = ssd.fit_regression(formula, fixed_k=3)
    assert "age" in res.model.names


def test_fixed_covariates_give_f_test(tiny_kv, large_docs, lexicon, frame):
    ssd = _ssd(tiny_kv, large_docs, "rating", lexicon, data=frame)
    res = ssd.fit_regression("~ age", fixed_k=3)
    assert res.test.name == "f_test"
    assert res.stats.backend == "PCA+GLMM"


@pytest.mark.parametrize("y", ["nope", ["rating", "nope"]])
def test_y_column_missing_raises(tiny_kv, large_docs, lexicon, frame, y):
    with pytest.raises(ValueError, match="nope"):
        _ssd(tiny_kv, large_docs, y, lexicon, data=frame)


def test_formula_other_lhs_raises(tiny_kv, large_docs, lexicon, frame):
    ssd = _ssd(tiny_kv, large_docs, "rating", lexicon, data=frame)
    with pytest.raises(ValueError, match="arousal"):
        ssd.fit_regression("arousal ~ age", fixed_k=3)


def test_formula_without_data_raises(tiny_kv, large_docs, large_y, lexicon):
    ssd = _ssd(tiny_kv, large_docs, large_y, lexicon)
    with pytest.raises(ValueError, match="data"):
        ssd.fit_regression("~ age", fixed_k=3)


def test_mixed_model(tiny_kv, large_docs, lexicon, frame):
    ssd = _ssd(tiny_kv, large_docs, "rating", lexicon, data=frame)
    res = ssd.fit_regression("~ age + (1 | ID)", fixed_k=3)
    assert res.test.name == "lrt"
    assert res.stats.backend == "PCA+GLMM"
    assert res.model.formula == "_ssd_y ~ age + (1 | ID) + _ssd_pca_1 + _ssd_pca_2 + _ssd_pca_3"
    assert res.model.re_groups
    assert 0 < res.stats.pvalue < 1
    assert np.isnan(res.stats.r2)
    assert res.fit_info.formula == "~ age + (1 | ID)"
    assert res.fit_info.family == "gaussian"
    assert res.test().name == "lrt"
    assert [r.term for r in res.covariates] == ["(Intercept)", "age"]
    text = res.summary()
    assert "Random effects:" in text
    assert "_ssd_" not in text
    header, row = _block(text, "SSD effects:")
    assert header.split() == ["p"]
    assert row.split()[:2] == ["text", fmt_p(res.test.pvalue)]


def test_missing_covariate_raises(tiny_kv, large_docs, lexicon, frame):
    data = frame.copy()
    data.loc[3, "age"] = np.nan
    ssd = _ssd(tiny_kv, large_docs, "rating", lexicon, data=data)
    with pytest.raises(ValueError, match="missing"):
        ssd.fit_regression("~ age", fixed_k=3)


def test_reserved_column_raises(tiny_kv, large_docs, lexicon, frame):
    ssd = _ssd(tiny_kv, large_docs, "rating", lexicon, data=frame.assign(_ssd_x=1.0))
    with pytest.raises(ValueError, match="_ssd_"):
        ssd.fit_regression("~ age", fixed_k=3)


def test_binomial(tiny_kv, large_docs, large_y, lexicon):
    y01 = (large_y > np.median(large_y)).astype(float)
    ssd = _ssd(tiny_kv, large_docs, y01, lexicon)
    res = ssd.fit_regression(family="binomial", fixed_k=3)
    assert res.model.family == "binomial"
    assert res.test.name == "lrt"
    assert res.stats.backend == "PCA+GLMM"
    names = list(res.model.names)
    w = np.array([res.model.beta[names.index(f"_ssd_pca_{i}")] for i in (1, 2, 3)])
    np.testing.assert_allclose(res.pca_weights, w)


@pytest.mark.parametrize("fit", [
    lambda ssd: ssd.fit_regression("~ age", fixed_k=3),
    lambda ssd: ssd.fit_pls(formula="~ age", k=1, test=False),
])
def test_predict_refused_after_formula(tiny_kv, large_docs, lexicon, frame, fit):
    ssd = _ssd(tiny_kv, large_docs, "rating", lexicon, data=frame)
    res = fit(ssd)
    with pytest.raises(NotImplementedError):
        res.predict(ssd.corpus)


# --- fit_pls with a formula ------------------------------------------------

def test_pls_formula_matches_residualized_pls(tiny_kv, large_docs, lexicon, frame):
    """PLS1 and its split_exact test on y and X with age and arousal regressed out."""
    import plskit

    ssd = _ssd(tiny_kv, large_docs, "rating", lexicon, data=frame)
    res = ssd.fit_pls(formula="~ age + arousal", k=1, n_splits=20)

    z = np.column_stack([np.ones(ssd.n_kept), frame["age"], frame["arousal"]])
    resid = lambda a: a - z @ np.linalg.lstsq(z, a, rcond=None)[0]  # noqa: E731
    x_res, y_res = resid(ssd.x), resid(ssd.y)
    Xs, _, X_scale = standardize(x_res)
    ys = standardize(y_res.reshape(-1, 1))[0].ravel()
    m = plskit.pls1_fit(Xs, ys, k=1, pre_standardized=True, seed=2137)
    beta = m.coef / X_scale
    if np.corrcoef(ys, x_res @ beta)[0, 1] < 0:
        beta = -beta
    p = plskit.pls1_confirmatory_test(
        Xs, ys, 1, test_method="split_exact", args={"n_splits": 20},
        pre_standardized=True, seed=2137,
    ).pvalue

    np.testing.assert_allclose(res.beta, beta, rtol=1e-6, atol=1e-10)
    assert res.stats.pvalue == pytest.approx(p)
    assert res.fit_info.formula == "~ age + arousal"
    # A rerun tests the residualized data too, not the raw y and X.
    assert res.test(n_splits=20, seed=2137).pvalue == pytest.approx(p)


@pytest.mark.parametrize("fit", [
    lambda ssd: ssd.fit_regression("~ age + arousal", fixed_k=3),
    lambda ssd: ssd.fit_pls(formula="~ age + arousal", k=1, test=False),
])
def test_formula_stats_and_residuals_describe_y(tiny_kv, large_docs, lexicon, frame, fit):
    """stats.y_std is y's own SD; docs.residual has the covariates taken out (OLS residuals are orthogonal to them)."""
    ssd = _ssd(tiny_kv, large_docs, "rating", lexicon, data=frame)
    res = fit(ssd)
    assert res.stats.y_mean == pytest.approx(np.mean(ssd.y))
    assert res.stats.y_std == pytest.approx(np.std(ssd.y))
    resid = res.docs.to_df(cols="all")["residual"].to_numpy()
    for c in ("age", "arousal"):
        assert abs(np.corrcoef(resid, frame[c])[0, 1]) < 1e-8


def test_pls_formula_random_effects_raise(tiny_kv, large_docs, lexicon, frame):
    ssd = _ssd(tiny_kv, large_docs, "rating", lexicon, data=frame)
    with pytest.raises(ValueError, match="random effects"):
        ssd.fit_pls(formula="~ age + (1 | ID)", k=1)


def test_pls_formula_covariate_explains_y_raises(tiny_kv, large_docs, lexicon, frame):
    ssd = _ssd(tiny_kv, large_docs, "rating", lexicon, data=frame.assign(copy=frame["rating"]))
    with pytest.raises(ValueError, match="explain"):
        ssd.fit_pls(formula="~ copy", k=1)


# --- covariate effects -----------------------------------------------------

def _ols(z, y):
    """OLS coefficients and standard errors."""
    coef, rss = np.linalg.lstsq(z, y, rcond=None)[:2]
    cov = rss[0] / (len(y) - z.shape[1]) * np.linalg.inv(z.T @ z)
    return coef, np.sqrt(np.diag(cov))


def test_regression_covariates_hold_components_fixed(tiny_kv, large_docs, lexicon, frame):
    ssd = _ssd(tiny_kv, large_docs, "rating", lexicon, data=frame)
    res = ssd.fit_regression("~ age", fixed_k=3)
    z_pca = pca_fit_transform(standardize(ssd.x)[0], 3)[0]
    coef, se = _ols(np.column_stack([np.ones(ssd.n_kept), frame["age"], z_pca]), ssd.y)
    cov = res.covariates
    assert [r.term for r in cov] == ["(Intercept)", "age"]
    np.testing.assert_allclose([r.estimate for r in cov], coef[:2], rtol=1e-6)
    np.testing.assert_allclose([r.se for r in cov], se[:2], rtol=1e-6)


@pytest.mark.parametrize("fit", [
    lambda ssd: ssd.fit_regression(fixed_k=3),
    lambda ssd: ssd.fit_pls(k=1, test=False),
    lambda ssd: ssd.fit_pls("~ age", k=1, test=False),
])
def test_no_covariates_without_model(tiny_kv, large_docs, lexicon, frame, fit):
    ssd = _ssd(tiny_kv, large_docs, "rating", lexicon, data=frame)
    assert len(fit(ssd).covariates) == 0


# --- compact summary -------------------------------------------------------

def _block(text, title):
    """The lines of one summary block: after ``title``, up to the next blank or ``---`` line."""
    lines = text.splitlines()
    start = lines.index(title) + 1
    end = next(i for i in range(start, len(lines)) if lines[i] in ("", "---"))
    return lines[start:end]


def test_summary_drops_text_terms(tiny_kv, large_docs, lexicon, frame):
    ssd = _ssd(tiny_kv, large_docs, "rating", lexicon, data=frame)
    res = ssd.fit_regression("~ age", fixed_k=3)
    text = res.summary()
    assert "_ssd_" not in text
    assert "Formula: rating ~ age + text" in text
    rows = [line.split() for line in _block(text, "Fixed effects:")[1:]]
    assert [r[0] for r in rows] == ["(Intercept)", "age"]
    np.testing.assert_allclose([float(r[1]) for r in rows],
                               [c.estimate for c in res.covariates], rtol=1e-3)
    corr = _block(text, "Correlation of Fixed Effects:")
    assert corr[0].split() == ["(Intercept)"]
    assert [line.split()[0] for line in corr[1:]] == ["age"]
    header, row = _block(text, "SSD effects:")
    assert fmt_p(res.test.pvalue) in row.split()
    # The p value ends where the Pr(>|t|) column of the fixed effects ends.
    pr_end = _block(text, "Fixed effects:")[0].index("Pr(>|t|)") + len("Pr(>|t|)")
    assert row[:pr_end].endswith(fmt_p(res.test.pvalue))


def test_summary_names_text_by_series_name(tiny_kv, large_docs, lexicon, frame):
    corpus = Corpus(pd.Series(large_docs, name="essay"), pretokenized=True, lang="pl")
    text = SSD(tiny_kv, corpus, "rating", lexicon, data=frame).fit_regression("~ age", fixed_k=3).summary()
    assert "Formula: rating ~ age + essay" in text
    assert _block(text, "SSD effects:")[1].split()[0] == "essay"


def test_regression_summary_f_test(tiny_kv, large_docs, lexicon, frame):
    """F statistic and ΔR² from the RSS of y ~ age with and without the components."""
    ssd = _ssd(tiny_kv, large_docs, "rating", lexicon, data=frame)
    res = ssd.fit_regression("~ age", fixed_k=3)
    n, k = ssd.n_kept, 3
    z0 = np.column_stack([np.ones(n), frame["age"]])
    z1 = np.column_stack([z0, pca_fit_transform(standardize(ssd.x)[0], k)[0]])
    rss0, rss1 = (np.linalg.lstsq(z, ssd.y, rcond=None)[1][0] for z in (z0, z1))
    df_resid = n - z1.shape[1]
    f = (rss0 - rss1) / k / (rss1 / df_resid)
    tss = np.sum((ssd.y - ssd.y.mean()) ** 2)
    res.test()
    params = res.test.params
    assert params["df"] == k and params["df_resid"] == df_resid
    assert params["statistic"] == pytest.approx(f, rel=1e-6)
    assert params["r2_delta"] == pytest.approx((rss0 - rss1) / tss, rel=1e-6)
    header, row = _block(res.summary(), "SSD effects:")
    assert header.split() == ["ΔR²", "p"]
    assert row.split()[:3] == ["text", fmt_r(params["r2_delta"], digits=3), fmt_p(res.test.pvalue)]


def test_pls_summary_without_formula(tiny_kv, large_docs, large_y, lexicon):
    res = _ssd(tiny_kv, large_docs, large_y, lexicon).fit_pls(k=1)
    text = res.summary()
    assert "Fixed effects" not in text
    assert "Text:  PLS, k = 1\n" in text
    assert f"(split_exact): p = {fmt_p(res.test.pvalue)}" in text
