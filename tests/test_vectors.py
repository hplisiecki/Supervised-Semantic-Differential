"""Tests for ssdlite.utils.vectors — SIF doc vector construction."""

import numpy as np

from ssdlite.utils.vectors import (
    compute_global_sif,
    build_doc_vectors,
    build_and_normalize_doc_vectors,
)


class TestComputeGlobalSIF:
    def test_basic(self):
        docs = [["a", "b", "a"], ["b", "c"]]
        wc, total = compute_global_sif(docs)
        assert wc == {"a": 2, "b": 2, "c": 1}
        assert total == 5


class TestBuildDocVectors:
    def test_seed_mode(self, tiny_kv, sample_docs, lexicon):
        wc, tot = compute_global_sif(sample_docs)
        X, keep = build_doc_vectors(
            sample_docs, tiny_kv, lexicon, wc, tot, window=3, sif_a=1e-3,
        )
        assert X.shape[0] == keep.sum()
        assert X.shape[1] == tiny_kv.vector_size
        assert keep.dtype == bool
        assert keep.shape[0] == len(sample_docs)

    def test_no_seeds_all_dropped(self, tiny_kv, sample_docs_no_seeds, lexicon):
        wc, tot = compute_global_sif(sample_docs_no_seeds)
        X, keep = build_doc_vectors(
            sample_docs_no_seeds, tiny_kv, lexicon, wc, tot, window=3, sif_a=1e-3,
        )
        assert X.shape[0] == 0
        assert keep.sum() == 0

    def test_full_mode(self, tiny_kv, sample_docs, lexicon):
        wc, tot = compute_global_sif(sample_docs)
        X, keep = build_doc_vectors(
            sample_docs, tiny_kv, lexicon, wc, tot, window=3, sif_a=1e-3, mode="full",
        )
        # Full mode ignores lexicon → all docs should be kept
        assert keep.sum() == len(sample_docs)


class TestBuildAndNormalize:
    def test_l2_normalized(self, tiny_kv, sample_docs, lexicon):
        X, keep = build_and_normalize_doc_vectors(
            sample_docs, tiny_kv, lexicon, window=3, sif_a=1e-3,
        )
        if X.shape[0] > 0:
            norms = np.linalg.norm(X, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-5)

    def test_no_normalize(self, tiny_kv, sample_docs, lexicon):
        X, keep = build_and_normalize_doc_vectors(
            sample_docs, tiny_kv, lexicon,
            window=3, sif_a=1e-3, l2_normalize=False,
        )
        if X.shape[0] > 0:
            norms = np.linalg.norm(X, axis=1)
            # Should NOT all be 1.0
            assert not np.allclose(norms, 1.0, atol=1e-3)
