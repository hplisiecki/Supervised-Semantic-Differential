"""Integration tests for the SSD class.

Fixtures come from conftest.py — all use tiny in-memory Embeddings,
no real embeddings, no network access.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from ssdiff.ssd import SSD
from ssdiff.utils.diagnostics import _get_hook, progress_hook

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D arrays."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# 1. Construction: vectors are built lazily; SSD has no fit_info
# ---------------------------------------------------------------------------

def test_construction_counts(ssd_instance):
    """n_kept <= len(ssd); SSD has no fit_info attribute pre-fit."""
    ssd = ssd_instance
    assert 0 < ssd.n_kept <= len(ssd)
    assert not hasattr(ssd, "fit_info")


def test_construction_shapes(ssd_instance):
    """The vectors are 2-D with one row per document that has one."""
    X, has_vector, _ = ssd_instance._vectors()
    assert X.ndim == 2
    assert has_vector.shape == (len(ssd_instance),)
    assert X.shape[0] == ssd_instance.n_kept


# ---------------------------------------------------------------------------
# 2. fit_pls basic invariants
# ---------------------------------------------------------------------------

def test_fit_pls_returns_plsresult(ssd_instance, parity_df):
    from ssdiff.results.continuous_result import PLSResult
    result = ssd_instance.fit_pls(parity_df["y"], k=2, n_splits=20)
    assert isinstance(result, PLSResult)


def test_fit_pls_r2_in_range(ssd_instance, parity_df):
    """PLSResult.stats.r2 is in [0, 1]."""
    result = ssd_instance.fit_pls(parity_df["y"], k=2, n_splits=20)
    assert 0.0 <= result.stats.r2 <= 1.0


def test_fit_pls_beta_shape(ssd_instance, parity_df, parity_emb):
    """beta shape is (D,) where D == embeddings.dim."""
    result = ssd_instance.fit_pls(parity_df["y"], k=2, n_splits=20)
    D = parity_emb.vectors.shape[1]
    assert result.beta.shape == (D,)


# ---------------------------------------------------------------------------
# 5. fit_regression with fixed_k
# ---------------------------------------------------------------------------

def test_fit_regression_fixed_k(ssd_instance, parity_df):
    """fit_regression(fixed_k=3) returns PCAOLSResult with pca_k == 3."""
    from ssdiff.results.continuous_result import PCAOLSResult
    result = ssd_instance.fit_regression(parity_df["y"], fixed_k=3)
    assert isinstance(result, PCAOLSResult)
    assert result.pca_k == 3


# ---------------------------------------------------------------------------
# 6. fit_regression sweep (fixed_k=None)
# ---------------------------------------------------------------------------

def test_fit_regression_sweep(tiny_kv, large_docs, large_y, lexicon):
    """fit_regression(fixed_k=None) sweep produces valid PCAOLSResult."""
    from ssdiff.results.continuous_result import PCAOLSResult
    ssd = SSD(tiny_kv, large_docs, lexicon)
    result = ssd.fit_regression(large_y, fixed_k=None, k_min=2, k_max=6, k_step=1)
    assert isinstance(result, PCAOLSResult)
    assert 0.0 <= result.stats.r2 <= 1.0
    assert result.pca_k >= 2


# ---------------------------------------------------------------------------
# 9. fit_pls on categorical y raises ValueError
# ---------------------------------------------------------------------------

def test_fit_pls_categorical_y_raises(tiny_kv, sample_docs, sample_groups, lexicon):
    """A categorical outcome raises ValueError on fit_pls()."""
    ssd = SSD(tiny_kv, sample_docs, lexicon)
    with pytest.raises(ValueError, match="categorical"):
        ssd.fit_pls(sample_groups)


# ---------------------------------------------------------------------------
# 10. fit_pls on y with only 1 unique value
# ---------------------------------------------------------------------------

def test_fit_pls_constant_y_rejected(tiny_kv, sample_docs, lexicon):
    """Constant y has zero variance — fitting PLS1 is undefined and must error."""
    constant_y = np.ones(len(sample_docs))
    ssd = SSD(tiny_kv, sample_docs, lexicon)
    with pytest.raises(ValueError, match="nonzero variance"):
        ssd.fit_pls(constant_y, k=1, n_splits=20)


# ---------------------------------------------------------------------------
# 11. use_full_doc=True produces different results from use_full_doc=False
# ---------------------------------------------------------------------------

def test_use_full_doc_differs(tiny_kv, sample_docs, sample_y, lexicon):
    """use_full_doc=True vs False: betas are not identical."""
    ssd_ctx = SSD(tiny_kv, sample_docs, lexicon, use_full_doc=False)
    ssd_full = SSD(tiny_kv, sample_docs, None, use_full_doc=True)

    res_ctx = ssd_ctx.fit_pls(sample_y, k=1, n_splits=20)
    res_full = ssd_full.fit_pls(sample_y, k=1, n_splits=20)

    # Results should differ — at minimum their betas are not identical
    assert not np.allclose(res_ctx.beta, res_full.beta), (
        "Expected use_full_doc=True and use_full_doc=False to produce different betas"
    )


# ---------------------------------------------------------------------------
# 12. progress_hook: fires per step; thread-local isolation
# ---------------------------------------------------------------------------

def test_progress_hook_does_not_break_fit_pls(ssd_instance, parity_df):
    """progress_hook around fit_pls is a no-op for the permutation, split and
    calibrated tests: they run inside plskit's Rust core, which reports no
    progress. This test checks that wrapping ``fit_pls`` in a
    ``progress_hook`` does not break it.
    """
    calls = []

    def cb(current, total, desc):
        calls.append((current, total, desc))

    with progress_hook(cb):
        result = ssd_instance.fit_pls(
            parity_df["y"], k=1, n_splits=20, random_state=1
        )

    # Hook may legitimately receive zero calls; what we care about is
    # that fit_pls produced a valid result.
    assert 0.0 <= result.stats.r2 <= 1.0
    # If the hook fires, each call must still have the right shape.
    for current, total, desc in calls:
        assert isinstance(current, int)
        assert isinstance(total, int)
        assert isinstance(desc, str)


def test_progress_hook_thread_local_isolation():
    """Two threads with different hooks do not cross-contaminate."""
    calls_a: list = []
    calls_b: list = []

    barrier = threading.Barrier(2)

    def thread_a():
        def cb_a(cur, tot, desc):
            calls_a.append(desc)

        with progress_hook(cb_a):
            barrier.wait()  # sync: both hooks set before either fires
            # Simulate a hooked progress call directly
            from ssdiff.utils.diagnostics import _progress
            list(_progress(range(3), total=3, desc="thread-A"))
        barrier.wait()  # sync: both done

    def thread_b():
        def cb_b(cur, tot, desc):
            calls_b.append(desc)

        with progress_hook(cb_b):
            barrier.wait()
            from ssdiff.utils.diagnostics import _progress
            list(_progress(range(2), total=2, desc="thread-B"))
        barrier.wait()

    ta = threading.Thread(target=thread_a)
    tb = threading.Thread(target=thread_b)
    ta.start()
    tb.start()
    ta.join(timeout=10)
    tb.join(timeout=10)

    # Each thread should only see its own hook's callbacks
    assert all(d == "thread-A" for d in calls_a), f"Thread A saw: {calls_a}"
    assert all(d == "thread-B" for d in calls_b), f"Thread B saw: {calls_b}"
    # No cross-contamination
    assert "thread-B" not in calls_a
    assert "thread-A" not in calls_b


# ---------------------------------------------------------------------------
# 13. progress_hook restores on exception
# ---------------------------------------------------------------------------

def test_progress_hook_restores_on_exception():
    """Hook is restored (to None or prior value) even if the block raises."""
    # Baseline: no hook before entering
    assert _get_hook() is None

    sentinel_calls = []

    def cb(cur, tot, desc):
        sentinel_calls.append(cur)

    with pytest.raises(RuntimeError, match="boom"), progress_hook(cb):
        # Hook is active inside
        assert _get_hook() is cb
        raise RuntimeError("boom")

    # After the exception, hook must be restored to None
    assert _get_hook() is None


def test_progress_hook_nested_restores():
    """Nested progress_hook: inner restores to outer, outer restores to None."""
    outer_calls = []
    inner_calls = []

    def outer_cb(cur, tot, desc):
        outer_calls.append(cur)

    def inner_cb(cur, tot, desc):
        inner_calls.append(cur)

    with progress_hook(outer_cb):
        assert _get_hook() is outer_cb
        with progress_hook(inner_cb):
            assert _get_hook() is inner_cb
        # After inner exits, outer is restored
        assert _get_hook() is outer_cb

    # After outer exits, None restored
    assert _get_hook() is None


# ---------------------------------------------------------------------------
# 15b. k="auto" routes through plskit.pls1_find_k_optimal
# ---------------------------------------------------------------------------

def test_fit_pls_k_auto(ssd_instance, parity_df):
    """fit_pls(k='auto') resolves to k>=1 via pls1_find_k_optimal."""
    result = ssd_instance.fit_pls(
        parity_df["y"], k="auto", k_max=3, n_splits=5, random_state=0,
    )
    assert result.fit_info["n_components"] >= 1


# ---------------------------------------------------------------------------
# 16. fit_regression raises ValueError on categorical y
# ---------------------------------------------------------------------------

def test_fit_regression_categorical_y_raises(tiny_kv, sample_docs, sample_groups, lexicon):
    """A categorical outcome raises ValueError on fit_regression()."""
    ssd = SSD(tiny_kv, sample_docs, lexicon)
    with pytest.raises(ValueError, match="categorical"):
        ssd.fit_regression(sample_groups)


# ---------------------------------------------------------------------------
# 17. A fit needs a lexicon unless use_full_doc=True
# ---------------------------------------------------------------------------

def test_fit_without_lexicon_raises(tiny_kv, sample_docs, sample_y):
    """No lexicon and use_full_doc=False raises ValueError at fit time."""
    ssd = SSD(tiny_kv, sample_docs, lexicon=None, use_full_doc=False)
    with pytest.raises(ValueError, match="lexicon"):
        ssd.fit_pls(sample_y)


def test_construction_full_doc_no_lexicon_ok(tiny_kv, sample_docs, sample_y):
    """SSD(use_full_doc=True) with lexicon=None does not raise."""
    ssd = SSD(tiny_kv, sample_docs, lexicon=None, use_full_doc=True)
    assert ssd.n_kept >= 0


# ---------------------------------------------------------------------------
# 2-D y (several outcomes)
# ---------------------------------------------------------------------------

def _y2(n, q=3, seed=0):
    return np.random.default_rng(seed).normal(size=(n, q))


def test_y_2d_array_accepted(tiny_kv, sample_docs, lexicon):
    y = _y2(len(sample_docs))
    res = SSD(tiny_kv, sample_docs, lexicon).fit_multipls(y, n_perm=20, n_splits=5)
    assert res.y.shape == (len(sample_docs), 3)
    assert res.outcome_names == ["y1", "y2", "y3"]


def test_y_dataframe_names(tiny_kv, sample_docs, lexicon):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(_y2(len(sample_docs)), columns=[10, "b", "c"])
    res = SSD(tiny_kv, sample_docs, lexicon).fit_multipls(df, n_perm=20, n_splits=5)
    assert res.outcome_names == ["10", "b", "c"]


def test_y_2d_nan_row_and_empty_doc_stay_aligned(tiny_kv, sample_docs, lexicon):
    docs = [list(d) for d in sample_docs] + [["zzz_not_in_vocab"]]
    y = _y2(len(docs))
    y[1, 2] = np.nan
    res = SSD(tiny_kv, docs, lexicon).fit_multipls(y, n_perm=20, n_splits=5)
    expected_rows = [i for i in range(len(docs)) if i != 1 and i != len(docs) - 1]
    np.testing.assert_array_equal(res.y, y[expected_rows])
    assert res.x.shape[0] == res.y.shape[0]


def test_y_2d_non_numeric_raises(tiny_kv, sample_docs, lexicon):
    y = np.array([["a", "b"]] * len(sample_docs), dtype=object)
    with pytest.raises(ValueError, match="numeric"):
        SSD(tiny_kv, sample_docs, lexicon).fit_multipls(y)


@pytest.mark.parametrize("method", ["fit_pls", "fit_regression", "fit_groups"])
def test_single_outcome_fits_reject_2d_y(tiny_kv, sample_docs, lexicon, method):
    ssd = SSD(tiny_kv, sample_docs, lexicon)
    with pytest.raises(ValueError, match="fit_multipls"):
        getattr(ssd, method)(_y2(len(sample_docs)))


@pytest.mark.parametrize("method", ["fit_pls", "fit_regression", "fit_groups"])
def test_single_column_2d_y_asks_for_1d(tiny_kv, sample_docs, lexicon, method):
    ssd = SSD(tiny_kv, sample_docs, lexicon)
    with pytest.raises(ValueError, match="1-D outcome"):
        getattr(ssd, method)(_y2(len(sample_docs), q=1))


# ---------------------------------------------------------------------------
# predict() and fitted values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fit", [
    lambda s, y: s.fit_pls(y, k=2, n_splits=20, random_state=42),
    lambda s, y: s.fit_regression(y, fixed_k=3),
], ids=["pls", "ols"])
def test_predict_training_text_reproduces_fit(ssd_instance, parity_df, fit):
    """predict() on the training text gives docs.y_hat, and its R² is stats.r2."""
    res = fit(ssd_instance, parity_df["y"])
    pred = res.predict(parity_df["text"])
    assert pred.shape == (len(ssd_instance),)

    y_hat = np.array([d.y_hat for d in res.docs])
    np.testing.assert_allclose(pred[res._keep_mask], y_hat, rtol=1e-10)
    # Fitted values of a centered least-squares-type fit average to mean(y).
    assert y_hat.mean() == pytest.approx(res.y.mean())

    sse = np.sum((res.y - y_hat) ** 2)
    sst = np.sum((res.y - res.y.mean()) ** 2)
    assert 1.0 - sse / sst == pytest.approx(res.stats.r2)


def test_predict_nan_for_documents_without_vector(tiny_kv, sample_docs, sample_y, lexicon):
    """No known word, or no seed in seed-context mode, gives NaN."""
    res = SSD(tiny_kv, sample_docs, lexicon).fit_pls(sample_y, k=1, n_splits=20)
    pred = res.predict(
        [["kraj", "piekny", "dom"], ["zzz_not_in_vocab"], ["piekny", "dom"], []]
    )
    assert np.isfinite(pred[0])
    assert np.isnan(pred[1:]).all()


def test_predict_full_doc_mode(tiny_kv, sample_docs, sample_y):
    """Full-document fits predict on documents without any seed word."""
    res = SSD(tiny_kv, sample_docs, use_full_doc=True).fit_pls(sample_y, k=1, n_splits=20)
    pred = res.predict([["piekny", "dom"]])
    assert np.isfinite(pred).all()


def test_predict_without_sif(tiny_kv, sample_docs, sample_y, lexicon):
    """sif_a=None: predict() on the training text rebuilds the fit's vectors."""
    res = SSD(tiny_kv, sample_docs, lexicon, sif_a=None).fit_pls(sample_y, k=1, n_splits=20)
    pred = res.predict(sample_docs)
    np.testing.assert_allclose(pred, [d.y_hat for d in res.docs], rtol=1e-10)


@pytest.mark.parametrize("k", [2, "auto"])
def test_fit_pls_test_false_skips_test(ssd_instance, parity_df, k):
    """test=False: NaN p-value, test left not run, same beta as test=True."""
    kw = dict(k=k, k_max=3, n_splits=5, random_state=0)
    tested = ssd_instance.fit_pls(parity_df["y"], **kw)
    untested = ssd_instance.fit_pls(parity_df["y"], test=False, **kw)

    np.testing.assert_array_equal(untested.beta, tested.beta)
    assert np.isnan(untested.stats.pvalue)
    assert untested.test.pvalue is None
    assert untested.fit_info.p_at_k == tested.fit_info.p_at_k

    untested.test(n_splits=5, seed=0)
    assert 0.0 <= untested.test.pvalue <= 1.0
    assert untested.stats.pvalue == untested.test.pvalue
