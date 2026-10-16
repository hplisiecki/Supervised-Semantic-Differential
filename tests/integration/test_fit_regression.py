"""SSD.fit_regression: PCA regression with covariates and mixed models via glmm."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ssdiff.corpus import Corpus
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


def test_predict_refused_after_formula(tiny_kv, large_docs, lexicon, frame):
    ssd = _ssd(tiny_kv, large_docs, "rating", lexicon, data=frame)
    res = ssd.fit_regression("~ age", fixed_k=3)
    with pytest.raises(NotImplementedError):
        res.predict(ssd.corpus)
