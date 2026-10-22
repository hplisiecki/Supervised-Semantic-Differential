"""Tests for ssdiff/utils/vectors.py.

Covers:
  - compute_global_sif: counts, empty input
  - build_doc_vectors: seed mode, full mode, keep_mask
  - build_and_normalize_doc_vectors: unit norm, use_full_doc path, sif_a=None
"""

from __future__ import annotations

import numpy as np
import pytest

from ssdiff.utils.vectors import (
    build_and_normalize_doc_vectors,
    build_doc_vectors,
    compute_global_sif,
)

# ---------------------------------------------------------------------------
# Helper: replicate the SIF weight formula used in vectors.py
# ---------------------------------------------------------------------------

def _sif_weight(sif_a: float, count: int, total: int) -> float:
    return sif_a / (sif_a + count / max(total, 1))


# ---------------------------------------------------------------------------
# compute_global_sif
# ---------------------------------------------------------------------------


def test_compute_global_sif_counts_correctly():
    """Word counts are accumulated across sentences; empty input gives no counts."""
    sentences = [["a", "b", "a"], ["b", "c"]]
    wc, total = compute_global_sif(sentences)
    assert wc == {"a": 2, "b": 2, "c": 1}
    assert total == 5
    assert compute_global_sif([]) == ({}, 0)


# ---------------------------------------------------------------------------
# build_doc_vectors — seed mode
# ---------------------------------------------------------------------------


def test_build_doc_vectors_seed_mode_known_fixture(tiny_kv, lexicon):
    """Seed mode: vector equals mean of SIF-weighted context vectors around seed words.

    Doc: ['kraj', 'piekny', 'dom', 'silny']
    Seed: 'kraj' at index 0.  Window=2 → context indices in [max(0,0-2), min(4,0+3))
          = [0,1,2]; skip index 0 (seed itself) → j=1 ('piekny'), j=2 ('dom').
    Only one occurrence → doc vector = the single occ vector (no mean needed).
    """
    doc = ["kraj", "piekny", "dom", "silny"]
    docs = [doc]
    sif_a = 1e-3
    window = 2

    wc, total = compute_global_sif([doc])

    # Manually replicate _occ_vectors_in_doc for seed 'kraj'
    context_tokens = ["piekny", "dom"]  # j=1, j=2 within window
    sum_v = np.zeros(tiny_kv.vector_size, dtype=np.float64)
    w_sum = 0.0
    for c in context_tokens:
        a = _sif_weight(sif_a, wc[c], total)
        sum_v += a * tiny_kv[c]
        w_sum += a
    expected = (sum_v / w_sum).astype(np.float64)

    X, keep_mask = build_doc_vectors(
        docs, tiny_kv, lexicon, wc, total, window, sif_a, mode="seed"
    )

    assert keep_mask.tolist() == [True]
    assert X.shape == (1, tiny_kv.vector_size)
    assert X.dtype == np.float64
    np.testing.assert_allclose(X[0], expected, atol=1e-10)


# ---------------------------------------------------------------------------
# build_doc_vectors — full mode
# ---------------------------------------------------------------------------


def test_build_doc_vectors_full_mode_known_fixture(tiny_kv, lexicon):
    """Full mode: vector is SIF-weighted mean over ALL tokens in the document."""
    doc = ["kraj", "piekny", "dom", "silny"]
    docs = [doc]
    sif_a = 1e-3

    wc, total = compute_global_sif([doc])

    # Replicate _full_doc_vector
    sum_v = np.zeros(tiny_kv.vector_size, dtype=np.float64)
    w_sum = 0.0
    for c in doc:
        a = _sif_weight(sif_a, wc[c], total)
        sum_v += a * tiny_kv[c]
        w_sum += a
    expected = (sum_v / w_sum).astype(np.float64)

    X, keep_mask = build_doc_vectors(
        docs, tiny_kv, lexicon, wc, total, window=3, sif_a=sif_a, mode="full"
    )

    assert keep_mask.tolist() == [True]
    np.testing.assert_allclose(X[0], expected, atol=1e-10)


# ---------------------------------------------------------------------------
# build_doc_vectors — keep_mask
# ---------------------------------------------------------------------------


def test_build_doc_vectors_keep_mask_no_seeds(tiny_kv, lexicon, sample_docs_no_seeds):
    """Docs with no lexicon seeds in seed mode: all keep_mask entries are False."""
    wc, total = compute_global_sif(sample_docs_no_seeds)
    X, keep_mask = build_doc_vectors(
        sample_docs_no_seeds, tiny_kv, lexicon, wc, total,
        window=3, sif_a=1e-3, mode="seed"
    )
    assert not keep_mask.any(), "no-seed docs should all have keep_mask=False"
    assert X.shape[0] == 0


def test_build_doc_vectors_keep_mask_mixed(tiny_kv, lexicon):
    """Mixed docs: some with seeds, some without — keep_mask reflects per-doc outcome."""
    docs = [
        ["kraj", "piekny"],          # has seed 'kraj'
        ["dom", "miasto"],            # no seed
        ["narod", "silny"],           # has seed 'narod'
        ["rzeka", "gora"],            # no seed
    ]
    wc, total = compute_global_sif(docs)
    X, keep_mask = build_doc_vectors(
        docs, tiny_kv, lexicon, wc, total, window=2, sif_a=1e-3, mode="seed"
    )
    expected_mask = [True, False, True, False]
    assert keep_mask.tolist() == expected_mask
    assert X.shape[0] == 2  # two kept docs


# ---------------------------------------------------------------------------
# build_and_normalize_doc_vectors
# ---------------------------------------------------------------------------


def test_build_and_normalize_unit_norm(tiny_kv, lexicon, sample_docs):
    """All kept rows have L2 norm = 1 after normalization."""
    X, keep_mask = build_and_normalize_doc_vectors(
        sample_docs, tiny_kv, lexicon, window=3, sif_a=1e-3
    )
    assert keep_mask.all()
    norms = np.linalg.norm(X, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-6)


@pytest.mark.parametrize("use_full_doc, words", [
    (True, ["kraj", "piekny", "dom", "dom", "silny"]),
    (False, ["piekny", "dom"]),  # window=2 around 'kraj' at index 0
], ids=["full", "seed"])
def test_build_and_normalize_sif_none_is_plain_mean(tiny_kv, lexicon, use_full_doc, words):
    """sif_a=None: every word weighs 1, so the vector is the plain mean of its words.

    'dom' occurs twice, so SIF weights would differ between the words and
    move the vector off the plain mean.
    """
    doc = ["kraj", "piekny", "dom", "dom", "silny"]
    X, keep_mask = build_and_normalize_doc_vectors(
        [doc], tiny_kv, lexicon, window=2, sif_a=None,
        use_full_doc=use_full_doc, l2_normalize=False,
    )
    assert keep_mask.tolist() == [True]
    expected = np.mean([tiny_kv[w] for w in words], axis=0, dtype=np.float64)
    np.testing.assert_allclose(X[0], expected, atol=1e-12)
