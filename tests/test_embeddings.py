"""Tests for ssdlite.embeddings — Embeddings class."""

import os
import tempfile

import numpy as np
import pytest

from ssdlite.embeddings import Embeddings


class TestEmbeddingsConstruction:
    def test_basic(self, tiny_kv):
        assert len(tiny_kv) == 20
        assert tiny_kv.vector_size == 8
        assert "kraj" in tiny_kv
        assert "nonexistent" not in tiny_kv

    def test_getitem(self, tiny_kv):
        vec = tiny_kv["kraj"]
        assert vec.shape == (8,)
        assert vec.dtype == np.float32

    def test_get_vector_norm(self, tiny_kv):
        v_raw = tiny_kv.get_vector("kraj", norm=False)
        v_norm = tiny_kv.get_vector("kraj", norm=True)
        assert np.allclose(np.linalg.norm(v_norm), 1.0, atol=1e-5)
        # Raw should be the same since fixture is already unit-normed
        assert np.allclose(v_raw, v_norm, atol=1e-5)

    def test_repr(self, tiny_kv):
        r = repr(tiny_kv)
        assert "20" in r
        assert "8" in r


class TestEmbeddingsNormalize:
    def test_l2_normalize(self):
        keys = ["a", "b", "c"]
        vecs = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        emb = Embeddings(keys, vecs)
        result = emb.normalize(l2=True, abtt_m=0)
        assert result is emb  # in-place, returns self
        for i in range(3):
            assert np.allclose(np.linalg.norm(emb.vectors[i]), 1.0, atol=1e-5)

    def test_abtt(self):
        rng = np.random.default_rng(0)
        keys = [f"w{i}" for i in range(100)]
        vecs = rng.normal(size=(100, 10)).astype(np.float32)
        emb = Embeddings(keys, vecs)
        emb.normalize(l2=True, abtt_m=1, re_normalize=True)
        # After ABTT, top PC should have lower variance
        centered = emb.vectors - emb.vectors.mean(axis=0)
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        var_ratio = S[0] ** 2 / np.sum(S**2)
        assert var_ratio < 0.5  # top PC less dominant after removal


class TestParallelTxtRoundTrip:
    """Fast test for parallel txt loading — no real embedding files needed."""

    @pytest.fixture
    def medium_emb(self):
        """100 words, 10d — big enough to exercise multi-worker splitting."""
        keys = [f"word_{i:04d}" for i in range(100)]
        rng = np.random.default_rng(123)
        vecs = rng.normal(size=(100, 10)).astype(np.float32)
        return Embeddings(keys, vecs)

    def test_parallel_matches_sequential(self, medium_emb):
        with tempfile.TemporaryDirectory() as td:
            stem = os.path.join(td, "test")
            medium_emb.save(stem, fmt="txt")

            path = stem + ".txt"
            seq = Embeddings.load(path, parallel=False)
            par = Embeddings.load(path, parallel=True)

            assert len(seq) == len(par)
            assert seq.index_to_key == par.index_to_key
            assert np.allclose(seq.vectors, par.vectors, atol=1e-5)

    def test_parallel_preserves_all_words(self, medium_emb):
        with tempfile.TemporaryDirectory() as td:
            stem = os.path.join(td, "test")
            medium_emb.save(stem, fmt="txt")

            par = Embeddings.load(stem + ".txt", parallel=True)

            assert len(par) == 100
            assert par.index_to_key == medium_emb.index_to_key
            assert np.allclose(par.vectors, medium_emb.vectors, atol=1e-4)


class TestEmbeddingsPersistence:
    @pytest.fixture
    def small_emb(self):
        keys = ["hello", "world", "test"]
        vecs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
        return Embeddings(keys, vecs)

    def test_save_load_ssdembed(self, small_emb):
        with tempfile.TemporaryDirectory() as td:
            stem = os.path.join(td, "test")
            small_emb.save(stem)
            loaded = Embeddings.load(stem + ".ssdembed")
            assert len(loaded) == 3
            assert loaded.vector_size == 3
            assert np.allclose(small_emb.vectors, loaded.vectors)

    def test_save_load_text(self, small_emb):
        with tempfile.TemporaryDirectory() as td:
            stem = os.path.join(td, "test")
            small_emb.save(stem, fmt="txt")
            loaded = Embeddings.load(stem + ".txt")
            assert len(loaded) == 3
            assert np.allclose(small_emb.vectors, loaded.vectors, atol=1e-4)

    def test_save_load_bin(self, small_emb):
        with tempfile.TemporaryDirectory() as td:
            stem = os.path.join(td, "test")
            small_emb.save(stem, fmt="bin")
            loaded = Embeddings.load(stem + ".bin")
            assert len(loaded) == 3
            assert np.allclose(small_emb.vectors, loaded.vectors)

    def test_ssdembed_loads_writeable(self, small_emb):
        with tempfile.TemporaryDirectory() as td:
            stem = os.path.join(td, "test")
            small_emb.save(stem)
            loaded = Embeddings.load(stem + ".ssdembed")
            assert loaded.vectors.flags.writeable
            assert not isinstance(loaded.vectors, np.memmap)


class TestSimilarByVector:
    def test_self_is_top(self, tiny_kv):
        vec = tiny_kv["kraj"]
        nbrs = tiny_kv.similar_by_vector(vec, topn=3)
        assert len(nbrs) == 3
        assert nbrs[0][0] == "kraj"
        assert np.isclose(nbrs[0][1], 1.0, atol=1e-5)

    def test_restrict_vocab(self, tiny_kv):
        vec = tiny_kv["kraj"]
        nbrs = tiny_kv.similar_by_vector(vec, topn=5, restrict_vocab=5)
        for word, _ in nbrs:
            assert tiny_kv.key_to_index[word] < 5

    def test_zero_vector(self, tiny_kv):
        nbrs = tiny_kv.similar_by_vector(np.zeros(8), topn=3)
        assert nbrs == []
