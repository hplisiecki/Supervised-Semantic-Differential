"""fit_regression with ssd() text terms: interactions give one leaf per text term."""

from __future__ import annotations

import glmm
import numpy as np
import pytest

from ssdiff import RegressionResult
from ssdiff.results import PCAOLSResult
from ssdiff.results.format import fmt_r
from ssdiff.ssd import SSD
from ssdiff.utils.math import pca_fit_transform, standardize


@pytest.fixture(scope="module")
def essay(pack_emb, pack_df, pack_lexicon):
    return SSD(pack_df["text"].rename("essay"), embeddings=pack_emb, lexicon=pack_lexicon)


def test_default_formula_is_unchanged(essay, pack_df):
    a = essay.fit_regression("y ~ cov", data=pack_df, fixed_k=4)
    b = essay.fit_regression("y ~ cov + ssd()", data=pack_df, fixed_k=4)
    assert type(a) is PCAOLSResult
    np.testing.assert_allclose(a.beta, b.beta)


def test_factor_interaction_keys_and_sum(essay, pack_df):
    res = essay.fit_regression("y ~ cov + group*ssd()", data=pack_df, fixed_k=4)
    assert isinstance(res, RegressionResult)
    assert res.keys() == ["ssd(essay)", "grouplow:ssd(essay)", "groupmid:ssd(essay)"]
    w = dict(zip(res.model.names, res.model.beta))
    s = res["ssd(essay)"]._y_scale[0]
    np.testing.assert_allclose(res["grouplow:ssd(essay)"].pca_weights,
                               [w[f"grouplow:_ssd_pca_{i}"] / s for i in range(1, 5)])


def test_text_first_order(essay, pack_df):
    res = essay.fit_regression("y ~ ssd()*group", data=pack_df, fixed_k=4)
    assert res.keys()[1] == "ssd(essay):grouplow"


def test_leaves_share_the_fit(essay, pack_df):
    res = essay.fit_regression("y ~ group*ssd()", data=pack_df, fixed_k=4)
    a, b = res["ssd(essay)"], res["grouplow:ssd(essay)"]
    assert [d.y_hat for d in a.docs] == [d.y_hat for d in b.docs]
    assert a.stats.r2 == b.stats.r2 and a.beta_norm != b.beta_norm


def test_missing_level_after_drop(essay, pack_df):
    df = pack_df.copy(); df.loc[df.group == "mid", "y"] = np.nan
    res = essay.fit_regression("y ~ group*ssd()", data=df, fixed_k=4)
    assert res.keys() == ["ssd(essay)", "grouplow:ssd(essay)"]


def test_two_names_in_a_method_call_raise(essay, pack_df):
    with pytest.raises(ValueError, match="one text"):
        essay.fit_regression("y ~ ssd() + ssd(other)", data=pack_df)


def test_rows_and_dfs(essay, pack_df):
    res = essay.fit_regression("y ~ cov + group*ssd()", data=pack_df, fixed_k=4)
    rows = list(res.test)
    assert [r.label for r in rows] == [None, "ssd(essay)", "group:ssd(essay)"]
    assert [r.name for r in rows] == ["f_test", "wald_f", "wald_f"]
    assert [r.df for r in rows] == [4 * 3, 4, 4 * 2]
    assert res["ssd(essay)"].stats.pvalue == rows[0].pvalue


def test_wald_f_equals_refit_f_on_ols(essay, pack_df):
    res = essay.fit_regression("y ~ cov*ssd()", data=pack_df, fixed_k=4)
    assert res.keys() == ["ssd(essay)", "cov:ssd(essay)"]
    row = next(r for r in res.test if r.label == "cov:ssd(essay)")
    leaf = res["ssd(essay)"]
    z = pca_fit_transform(standardize(leaf.x)[0], 4)[0]   # the fit's own component scores
    df = pack_df[leaf._keep_mask].reset_index(drop=True).assign(
        _ssd_y=leaf.y, **{f"_ssd_pca_{i + 1}": z[:, i] for i in range(4)})
    pcs = " + ".join(f"_ssd_pca_{i}" for i in range(1, 5))
    cmp = glmm.compare(df, f"_ssd_y ~ cov + ({pcs})", f"_ssd_y ~ cov*({pcs})")
    assert row.statistic == pytest.approx(float(cmp.table["F"][1]), rel=1e-8)


def test_random_intercept(essay, pack_df):
    ids = np.arange(len(pack_df)) % 20
    effect = np.random.default_rng(0).normal(size=20)[ids]
    df = pack_df.assign(id=ids, y=pack_df["y"] + effect)
    res = essay.fit_regression("y ~ group*ssd() + (1 | id)", data=df, fixed_k=4)
    rows = list(res.test)
    assert rows[0].name == "lrt" and {r.name for r in rows[1:]} == {"wald_f"}
    assert all(np.isfinite(r.pvalue) for r in rows)


def test_summary_has_one_row_per_text_term_then_the_joint_tests(essay, pack_df):
    res = essay.fit_regression("y ~ cov + group*ssd()", data=pack_df, fixed_k=4)
    text = res.summary()
    terms = text.split("Text terms:")[1].split("\n\n")[0].strip().split("\n")[1:]
    assert [ln.split()[0] for ln in terms] == res.keys()
    joint = text.split("Joint tests:")[1].split("---")[0].strip().split("\n")
    assert joint[0].split()[:3] == ["ΔR²", "F", "value"]
    assert joint[1].split()[:4] == ["all", "text", "terms", fmt_r(res.test[0].r2_delta, digits=3)]


def test_poisson_gives_wald_chisq_rows(essay, pack_df):
    counts = np.round(pack_df["y"] - pack_df["y"].min()).astype(int)
    res = essay.fit_regression("n ~ group*ssd()", data=pack_df.assign(n=counts),
                               fixed_k=4, family="poisson")
    rows = list(res.test)
    assert [r.name for r in rows] == ["lrt", "wald_chisq", "wald_chisq"]
    assert all(r.df_resid is None and np.isfinite(r.pvalue) for r in rows)


def test_report_has_one_block_per_term_and_docs_once(essay, pack_df, tmp_path):
    res = essay.fit_regression("y ~ cov + group*ssd()", data=pack_df, fixed_k=4)
    rep = res.report()
    by_title = {s.title: s for s in rep.sections}
    assert [row[0] for row in by_title["Text terms"].rows] == res.keys()
    for key in res.keys():
        assert any(t.startswith(f"{key} — ") for t in by_title)
    assert sum(t.startswith("Docs — pos") for t in by_title) == 1
    for ext in ("md", "docx"):
        rep.save(tmp_path / f"report.{ext}")
        assert (tmp_path / f"report.{ext}").stat().st_size > 0
