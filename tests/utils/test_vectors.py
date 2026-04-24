"""Tests for ssdiff/utils/vectors.py.

Covers:
  - compute_global_sif: SIF weight ordering, empty input
  - build_doc_vectors: seed mode, full mode, profile mode, keep_mask
  - build_and_normalize_doc_vectors: unit norm, use_full_doc path
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


def test_compute_global_sif_empty_input():
    """Empty sentence list returns empty dict and zero total."""
    wc, total = compute_global_sif([])
    assert wc == {}
    assert total == 0


def test_compute_global_sif_sif_weight_ordering():
    """A rare token (count=1) has strictly higher SIF weight than a frequent one (count=10).

    compute_global_sif returns raw counts; the weight ordering follows from the
    formula  a / (a + count/total).  Larger count → smaller weight.
    """
    sif_a = 1e-3
    sentences = [["common"] * 10, ["rare"]]
    wc, total = compute_global_sif(sentences)

    assert wc["common"] == 10
    assert wc["rare"] == 1
    assert total == 11

    w_common = _sif_weight(sif_a, wc["common"], total)
    w_rare = _sif_weight(sif_a, wc["rare"], total)
    assert w_rare > w_common, (
        f"rare weight ({w_rare:.6f}) should exceed common weight ({w_common:.6f})"
    )


def test_compute_global_sif_counts_correctly():
    """Word counts are accumulated correctly across multiple sentences."""
    sentences = [["a", "b", "a"], ["b", "c"]]
    wc, total = compute_global_sif(sentences)
    assert wc == {"a": 2, "b": 2, "c": 1}
    assert total == 5


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
    np.testing.assert_allclose(X[0], expected, atol=1e-10)


def test_build_doc_vectors_seed_mode_shape(tiny_kv, lexicon, sample_docs):
    """Seed mode on standard fixtures: all 8 docs kept, shape is correct."""
    wc, total = compute_global_sif(sample_docs)
    X, keep_mask = build_doc_vectors(
        sample_docs, tiny_kv, lexicon, wc, total, window=3, sif_a=1e-3, mode="seed"
    )
    assert keep_mask.shape == (8,)
    assert keep_mask.all()
    assert X.shape == (8, tiny_kv.vector_size)
    assert X.dtype == np.float64


# ---------------------------------------------------------------------------
# build_doc_vectors — full mode vs seed mode differ
# ---------------------------------------------------------------------------


def test_build_doc_vectors_full_vs_seed_differ(tiny_kv, lexicon, sample_docs):
    """Full-doc mode produces different vectors than seed mode on the same input."""
    wc, total = compute_global_sif(sample_docs)

    X_seed, _ = build_doc_vectors(
        sample_docs, tiny_kv, lexicon, wc, total, window=3, sif_a=1e-3, mode="seed"
    )
    X_full, _ = build_doc_vectors(
        sample_docs, tiny_kv, lexicon, wc, total, window=3, sif_a=1e-3, mode="full"
    )

    # At least one row must differ (full uses all tokens; seed uses only context windows)
    assert not np.allclose(X_seed, X_full), (
        "full mode should produce different vectors than seed mode"
    )


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
# build_doc_vectors — profile mode (list[list[list[str]]])
# ---------------------------------------------------------------------------


def test_build_doc_vectors_profile_mode_known_fixture(tiny_kv, lexicon):
    """Profile-mode (grouped posts): doc vector equals mean of per-profile occ vectors.

    Two profiles for one 'document' (person).
    Profile 0: ['kraj', 'piekny', 'dom']   — seed 'kraj' at idx 0
    Profile 1: ['narod', 'silny', 'wielki'] — seed 'narod' at idx 0
    window=1 → context of 'kraj' = ['piekny'], context of 'narod' = ['silny'].
    """
    profiles = [
        [["kraj", "piekny", "dom"], ["narod", "silny", "wielki"]]
    ]
    all_flat = ["kraj", "piekny", "dom", "narod", "silny", "wielki"]
    sif_a = 1e-3
    window = 1

    wc, total = compute_global_sif([all_flat])  # one flat sentence for global counts

    def _occ(seed_idx, doc, context_tokens_at_window):
        sum_v = np.zeros(tiny_kv.vector_size, dtype=np.float64)
        w_sum = 0.0
        for c in context_tokens_at_window:
            a = _sif_weight(sif_a, wc.get(c, 0), total)
            sum_v += a * tiny_kv[c]
            w_sum += a
        return sum_v / w_sum if w_sum > 0 else None

    # Profile 0: 'kraj' at idx 0, window=1 → range [0,2), skip j=0 → j=1='piekny'
    occ0 = _occ(0, ["kraj", "piekny", "dom"], ["piekny"])
    # Profile 1: 'narod' at idx 0, window=1 → range [0,2), skip j=0 → j=1='silny'
    occ1 = _occ(0, ["narod", "silny", "wielki"], ["silny"])

    expected = np.mean([occ0, occ1], axis=0).astype(np.float64)

    X, keep_mask = build_doc_vectors(
        profiles, tiny_kv, lexicon, wc, total, window, sif_a, mode="seed"
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


def test_build_and_normalize_use_full_doc(tiny_kv, lexicon, sample_docs):
    """use_full_doc=True uses full-doc mode; result is still unit-normalized."""
    X, keep_mask = build_and_normalize_doc_vectors(
        sample_docs, tiny_kv, lexicon, window=3, sif_a=1e-3, use_full_doc=True
    )
    assert keep_mask.all()
    norms = np.linalg.norm(X, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-6)


def test_build_and_normalize_l2_false_no_unit_norm(tiny_kv, lexicon, sample_docs):
    """l2_normalize=False: vectors are NOT unit-normed (raw SIF-weighted means)."""
    X, keep_mask = build_and_normalize_doc_vectors(
        sample_docs, tiny_kv, lexicon, window=3, sif_a=1e-3, l2_normalize=False
    )
    norms = np.linalg.norm(X, axis=1)
    # Raw vectors from unit-normalized embeddings won't all be exactly 1.0
    # (they're weighted averages of different vectors); just confirm not all 1.0
    # This is a sanity check, not an exact value test.
    assert not np.allclose(norms, np.ones(len(norms))), (
        "without l2_normalize, rows should not be unit-normed"
    )
