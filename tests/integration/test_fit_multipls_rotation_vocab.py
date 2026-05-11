"""rotation_vocab parameter on SSD.fit_multipls.

Restricts the simple-structure target fed to varimax to the top-N
vocabulary rows (frequency-ranked by the word2vec/GloVe convention).
None ⇒ full vocab. Clamped silently to available rows; no-op for
``rotate="raw"``.
"""
from __future__ import annotations

import warnings

import numpy as np


def _info(res):
    return next(iter(res.pls_info))


def test_default_clamps_silently_on_small_vocab(ssd_instance):
    """Default rotation_vocab=50_000 clamps to actual vocab size, no warning."""
    n_vocab = int(ssd_instance.embeddings.vectors.shape[0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = ssd_instance.fit_multipls(k=2, n_splits=10, random_state=0)
    user = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user == [], [str(w.message) for w in user]
    assert _info(res)["rotation_vocab"] == n_vocab


def test_none_uses_full_vocab(ssd_instance):
    """rotation_vocab=None records None and feeds every row to varimax."""
    res = ssd_instance.fit_multipls(
        k=2, rotation_vocab=None, n_splits=10, random_state=0,
    )
    assert _info(res)["rotation_vocab"] is None


def test_explicit_int_recorded_as_effective(ssd_instance):
    """rotation_vocab smaller than vocab is recorded verbatim."""
    n_vocab = int(ssd_instance.embeddings.vectors.shape[0])
    sub = max(n_vocab // 2, 4)
    res = ssd_instance.fit_multipls(
        k=2, rotation_vocab=sub, n_splits=10, random_state=0,
    )
    assert _info(res)["rotation_vocab"] == sub


def test_huge_int_clamps_to_vocab_size(ssd_instance):
    """rotation_vocab > vocab is clamped silently to vocab size."""
    n_vocab = int(ssd_instance.embeddings.vectors.shape[0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = ssd_instance.fit_multipls(
            k=2, rotation_vocab=10_000_000, n_splits=10, random_state=0,
        )
    user = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user == [], [str(w.message) for w in user]
    assert _info(res)["rotation_vocab"] == n_vocab


def test_default_matches_explicit_full_clamp(ssd_instance):
    """Default and explicit huge-int both clamp to vocab → identical W_rot."""
    res_default = ssd_instance.fit_multipls(k=2, n_splits=10, random_state=0)
    res_huge = ssd_instance.fit_multipls(
        k=2, rotation_vocab=10_000_000, n_splits=10, random_state=0,
    )
    np.testing.assert_array_equal(res_default.W_rot, res_huge.W_rot)


def test_no_op_for_raw_rotation(ssd_instance):
    """rotate='raw' doesn't touch the embedding matrix; rotation_vocab is inert."""
    res_a = ssd_instance.fit_multipls(
        k=2, rotate="raw", rotation_vocab=3, n_splits=10, random_state=0,
    )
    res_b = ssd_instance.fit_multipls(
        k=2, rotate="raw", rotation_vocab=None, n_splits=10, random_state=0,
    )
    np.testing.assert_array_equal(res_a.W_rot, res_b.W_rot)
    np.testing.assert_array_equal(res_a.T_rot, res_b.T_rot)


def test_subset_runs_and_records_meta(tiny_kv_large, lexicon):
    """Subset and full rotation both succeed and record their respective values."""
    from ssdiff.corpus import Corpus
    from ssdiff.ssd import SSD

    rng = np.random.default_rng(7)
    docs = []
    for i in range(40):
        seed = ["kraj", "narod", "panstwo"][i % 3]
        extras = list(rng.choice(tiny_kv_large.index_to_key, size=3, replace=False))
        docs.append([seed, *extras])
    corpus = Corpus(docs, lang="pl", pretokenized=True)
    y = list(rng.normal(size=40))
    ssd = SSD(tiny_kv_large, corpus, y, lexicon)

    res_head = ssd.fit_multipls(
        k=2, rotation_vocab=10, n_splits=10, random_state=0,
    )
    res_full = ssd.fit_multipls(
        k=2, rotation_vocab=None, n_splits=10, random_state=0,
    )
    assert _info(res_head)["rotation_vocab"] == 10
    assert _info(res_full)["rotation_vocab"] is None
    assert res_head.W_rot.shape == res_full.W_rot.shape == (10, 2)
