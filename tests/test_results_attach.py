"""Pickle round-trip + detach/attach lifecycle (D14, acceptance #10, #11)."""

import pickle

import numpy as np
import pytest

from ssdiff.results.continuous_result import PLSResult


def _make_result():
    n, d = 50, 8
    rng = np.random.default_rng(0)
    beta = rng.standard_normal(d)
    x = rng.standard_normal((n, d))
    y = x @ beta + 0.1 * rng.standard_normal(n)
    return PLSResult(
        x=x, beta=beta, keep_mask=np.ones(n, dtype=bool),
        n_raw=n, n_kept=n, n_dropped=0,
        y=y, _y_mean=np.array([y.mean()]), _y_scale=np.array([y.std() + 1e-12]),
        r2=0.47, r2_adj=0.46, pvalue=1e-5,
    )


def test_pickle_round_trip_preserves_stats_and_docs():
    r = _make_result()
    r2 = pickle.loads(pickle.dumps(r))
    assert r2.stats.r2 == r.stats.r2
    assert r2.stats.n_kept == r.stats.n_kept
    assert len(r2.docs) == len(r.docs)


def test_words_before_attach_raises_with_hint():
    r = _make_result()
    with pytest.raises(RuntimeError, match="attach"):
        _ = r.words


def test_attach_embeddings_rewires_reference():
    r = _make_result()
    fake_emb = object()
    r.attach(embeddings=fake_emb)
    assert r.embeddings is fake_emb


def test_attach_corpus_rewires_reference():
    r = _make_result()
    fake_corpus = object()
    r.attach(corpus=fake_corpus)
    assert r.corpus is fake_corpus


def test_pickle_preserves_cache_entries():
    r = _make_result()
    # Fake a cache entry directly (no embeddings needed).
    key = ("clusters", (("side", "pos"), ("topn", 100)))
    r._cache[key] = "SENTINEL"
    r2 = pickle.loads(pickle.dumps(r))
    assert r2._cache[key] == "SENTINEL"
