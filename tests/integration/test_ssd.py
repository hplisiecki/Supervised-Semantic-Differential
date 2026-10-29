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
# 9. A categorical outcome raises ValueError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["fit_pls", "fit_regression"])
def test_categorical_y_raises(tiny_kv, sample_docs, sample_groups, lexicon, method):
    ssd = SSD(sample_docs, embeddings=tiny_kv, lexicon=lexicon)
    with pytest.raises(ValueError, match="categorical"):
        getattr(ssd, method)(sample_groups)


# ---------------------------------------------------------------------------
# 10. fit_pls on y with only 1 unique value
# ---------------------------------------------------------------------------

def test_fit_pls_constant_y_rejected(tiny_kv, sample_docs, lexicon):
    """Constant y has zero variance — fitting PLS1 is undefined and must error."""
    constant_y = np.ones(len(sample_docs))
    ssd = SSD(sample_docs, embeddings=tiny_kv, lexicon=lexicon)
    with pytest.raises(ValueError, match="nonzero variance"):
        ssd.fit_pls(constant_y, k=1, n_splits=20)


# ---------------------------------------------------------------------------
# 11. use_full_doc=True produces different vectors from use_full_doc=False
# ---------------------------------------------------------------------------

def test_use_full_doc_differs(tiny_kv, sample_docs, lexicon):
    """Whole-document vectors differ from seed-window vectors for the same text and lexicon."""
    ctx = SSD(sample_docs, embeddings=tiny_kv, lexicon=lexicon, use_full_doc=False)
    full = SSD(sample_docs, embeddings=tiny_kv, lexicon=lexicon, use_full_doc=True)
    assert not np.allclose(ctx._vectors()[0], full._vectors()[0])


# ---------------------------------------------------------------------------
# 12. progress_hook: thread-local isolation
# ---------------------------------------------------------------------------

def test_progress_hook_thread_local_isolation():
    """Two threads with different hooks do not cross-contaminate."""
    calls_a: list = []
    calls_b: list = []

    barrier = threading.Barrier(2)

    def thread_a():
        def cb_a(cur, tot, desc):
            calls_a.append((cur, tot, desc))

        with progress_hook(cb_a):
            barrier.wait()  # sync: both hooks set before either fires
            # Simulate a hooked progress call directly
            from ssdiff.utils.diagnostics import _progress
            list(_progress(range(3), total=3, desc="thread-A"))
            # Wait inside the block: leaving it restores the previous hook,
            # which would hide a shared (non-thread-local) hook from the other thread.
            barrier.wait()

    def thread_b():
        def cb_b(cur, tot, desc):
            calls_b.append((cur, tot, desc))

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

    assert calls_a == [(1, 3, "thread-A"), (2, 3, "thread-A"), (3, 3, "thread-A")]
    assert calls_b == [(1, 2, "thread-B"), (2, 2, "thread-B")]


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
# 2-D y (several outcomes)
# ---------------------------------------------------------------------------

def _y2(n, q=3, seed=0):
    return np.random.default_rng(seed).normal(size=(n, q))


def test_y_2d_array_accepted(tiny_kv, sample_docs, lexicon):
    y = _y2(len(sample_docs))
    res = SSD(sample_docs, embeddings=tiny_kv, lexicon=lexicon).fit_multipls(y, n_perm=20, n_splits=5)
    assert res.y.shape == (len(sample_docs), 3)
    assert res.outcome_names == ["y1", "y2", "y3"]


def test_y_2d_nan_row_and_empty_doc_stay_aligned(tiny_kv, sample_docs, lexicon):
    docs = [list(d) for d in sample_docs] + [["zzz_not_in_vocab"]]
    y = _y2(len(docs))
    y[1, 2] = np.nan
    res = SSD(docs, embeddings=tiny_kv, lexicon=lexicon).fit_multipls(y, n_perm=20, n_splits=5)
    expected_rows = [i for i in range(len(docs)) if i != 1 and i != len(docs) - 1]
    np.testing.assert_array_equal(res.y, y[expected_rows])
    assert res.x.shape[0] == res.y.shape[0]
    assert res.stats.n == len(expected_rows)


def test_y_2d_non_numeric_raises(tiny_kv, sample_docs, lexicon):
    y = np.array([["a", "b"]] * len(sample_docs), dtype=object)
    with pytest.raises(ValueError, match="numeric"):
        SSD(sample_docs, embeddings=tiny_kv, lexicon=lexicon).fit_multipls(y)


@pytest.mark.parametrize("method", ["fit_pls", "fit_regression", "fit_groups"])
def test_single_outcome_fits_reject_2d_y(tiny_kv, sample_docs, lexicon, method):
    ssd = SSD(sample_docs, embeddings=tiny_kv, lexicon=lexicon)
    with pytest.raises(ValueError, match="fit_multipls"):
        getattr(ssd, method)(_y2(len(sample_docs)))


@pytest.mark.parametrize("method", ["fit_pls", "fit_regression", "fit_groups"])
def test_single_column_2d_y_asks_for_1d(tiny_kv, sample_docs, lexicon, method):
    ssd = SSD(sample_docs, embeddings=tiny_kv, lexicon=lexicon)
    with pytest.raises(ValueError, match="1-D outcome"):
        getattr(ssd, method)(_y2(len(sample_docs), q=1))


# ---------------------------------------------------------------------------
# predict() and fitted values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fit", [
    lambda s, y: s.fit_pls(y, k=2, n_splits=20, random_state=42),
    lambda s, y: s.fit_regression(y, fixed_k=3),
], ids=["pls", "ols"])
def test_predict_training_text_reproduces_fit(ssd_instance, pack_df, fit):
    """predict() on the training text gives docs.y_hat, and its R² is stats.r2."""
    res = fit(ssd_instance, pack_df["y"])
    pred = res.predict(pack_df["text"])
    assert pred.shape == (len(ssd_instance),)

    y_hat = np.array([d.y_hat for d in res.docs])
    np.testing.assert_allclose(pred[res._keep_mask], y_hat, rtol=1e-10)
    # A subset must reuse the training SIF counts, not recount on the new batch.
    sub = np.flatnonzero(res._keep_mask)[:4]
    np.testing.assert_allclose(
        res.predict(list(pack_df["text"].iloc[sub])), pred[sub], rtol=1e-10)
    # Fitted values of a centered least-squares-type fit average to mean(y).
    assert y_hat.mean() == pytest.approx(res.y.mean())

    sse = np.sum((res.y - y_hat) ** 2)
    sst = np.sum((res.y - res.y.mean()) ** 2)
    assert 1.0 - sse / sst == pytest.approx(res.stats.r2)


def test_predict_nan_for_documents_without_vector(tiny_kv, sample_docs, sample_y, lexicon):
    """No known word, or no seed in seed-context mode, gives NaN."""
    res = SSD(sample_docs, embeddings=tiny_kv, lexicon=lexicon).fit_pls(sample_y, k=1, n_splits=20)
    pred = res.predict(
        [["kraj", "piekny", "dom"], ["zzz_not_in_vocab"], ["piekny", "dom"], []]
    )
    assert np.isfinite(pred[0])
    assert np.isnan(pred[1:]).all()


def test_predict_full_doc_mode(tiny_kv, sample_docs, sample_y):
    """Full-document fits predict on documents without any seed word."""
    res = SSD(sample_docs, embeddings=tiny_kv, use_full_doc=True).fit_pls(sample_y, k=1, n_splits=20)
    pred = res.predict([["piekny", "dom"]])
    assert np.isfinite(pred).all()


def test_predict_without_sif(tiny_kv, sample_docs, sample_y, lexicon):
    """sif_a=None: predict() on the training text rebuilds the fit's vectors."""
    res = SSD(sample_docs, embeddings=tiny_kv, lexicon=lexicon, sif_a=None).fit_pls(sample_y, k=1, n_splits=20)
    pred = res.predict(sample_docs)
    np.testing.assert_allclose(pred, [d.y_hat for d in res.docs], rtol=1e-10)


@pytest.mark.parametrize("k", [2, "auto"])
def test_fit_pls_test_false_skips_test(ssd_instance, pack_df, k):
    """test=False: NaN p-value, test left not run, same beta as test=True."""
    kw = dict(k=k, k_max=3, n_splits=5, random_state=0)
    tested = ssd_instance.fit_pls(pack_df["y"], **kw)
    untested = ssd_instance.fit_pls(pack_df["y"], test=False, **kw)

    np.testing.assert_array_equal(untested.beta, tested.beta)
    assert np.isnan(untested.stats.pvalue)
    assert untested.test.pvalue is None
    assert untested.fit_info.p_at_k == tested.fit_info.p_at_k

    untested.test(n_splits=5, seed=0)
    assert 0.0 <= untested.test.pvalue <= 1.0
    assert untested.stats.pvalue == untested.test.pvalue
