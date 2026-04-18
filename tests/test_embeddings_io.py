"""Integration tests for Embeddings.load() with real embedding files.

All tests are marked @slow @local — they need files in Models/ and can
take seconds to minutes depending on format and file size.

Run with:  pytest tests/test_embeddings_io.py -v
Skip with: pytest -m "not slow"
"""

from __future__ import annotations

import os
import tempfile
import time

import numpy as np
import pytest

from ssdiff.embeddings import Embeddings

MODELS = "/home/plenartowicz/Projekty/SSD/Models"

# Paths — each test skips if the file doesn't exist.
# All non-.txt files on disk are L2-normalized; .txt files are raw.
SSDEMBED_NKJP = os.path.join(MODELS, "nkjp_300_normalized.ssdembed")
SSDEMBED_GLOVE = os.path.join(MODELS, "glove_800_normalized.ssdembed")
KV_NKJP = os.path.join(MODELS, "nkjp_300_normalized.kv")
KV_GLOVE = os.path.join(MODELS, "glove_800_normalized.kv")
BIN_NKJP = os.path.join(MODELS, "nkjp_300_normalized.bin")
TXT_NKJP = os.path.join(MODELS, "nkjp_300.txt")  # raw (un-normalized)

slow_local = pytest.mark.parametrize([], [])(lambda: None)  # unused, just for clarity
pytestmark = [pytest.mark.slow, pytest.mark.local]


def _skip_missing(path: str):
    if not os.path.exists(path):
        pytest.skip(f"File not found: {path}")


# Module-scoped fixtures for expensive-to-load formats.
# Loaded once and shared across every test in this file.

@pytest.fixture(scope="module")
def nkjp_txt_seq():
    _skip_missing(TXT_NKJP)
    t0 = time.perf_counter()
    emb = Embeddings.load(TXT_NKJP, parallel=False)
    print(f"\n  [fixture] nkjp.txt (seq): {len(emb)} words, loaded in {time.perf_counter()-t0:.2f}s")
    return emb


@pytest.fixture(scope="module")
def nkjp_txt_par():
    _skip_missing(TXT_NKJP)
    t0 = time.perf_counter()
    emb = Embeddings.load(TXT_NKJP, parallel=True)
    print(f"\n  [fixture] nkjp.txt (par): {len(emb)} words, loaded in {time.perf_counter()-t0:.2f}s")
    return emb


@pytest.fixture(scope="module")
def nkjp_ssdembed():
    _skip_missing(SSDEMBED_NKJP)
    return Embeddings.load(SSDEMBED_NKJP)


def _assert_valid_embeddings(emb: Embeddings, min_vocab: int = 1000, expected_dim: int | None = None):
    """Common assertions for loaded embeddings."""
    assert len(emb) >= min_vocab, f"Expected >= {min_vocab} words, got {len(emb)}"
    assert emb.vector_size > 0
    if expected_dim is not None:
        assert emb.vector_size == expected_dim, f"Expected dim={expected_dim}, got {emb.vector_size}"

    # Vectors should be finite
    assert np.all(np.isfinite(emb.vectors[:100])), "Vectors contain NaN/Inf"

    # Spot-check a word lookup
    word = emb.index_to_key[0]
    vec = emb[word]
    assert vec.shape == (emb.vector_size,)
    assert word in emb


# ==========================================================================
# .ssdembed format (pickle + .npy sidecar)
# ==========================================================================

class TestLoadSSDebed:
    def test_nkjp(self):
        _skip_missing(SSDEMBED_NKJP)
        t0 = time.perf_counter()
        emb = Embeddings.load(SSDEMBED_NKJP)
        dt = time.perf_counter() - t0
        print(f"\n  nkjp.ssdembed: {len(emb)} words, {emb.vector_size}d, loaded in {dt:.2f}s")
        _assert_valid_embeddings(emb, min_vocab=50_000, expected_dim=300)

    def test_glove(self):
        _skip_missing(SSDEMBED_GLOVE)
        t0 = time.perf_counter()
        emb = Embeddings.load(SSDEMBED_GLOVE)
        dt = time.perf_counter() - t0
        print(f"\n  glove.ssdembed: {len(emb)} words, {emb.vector_size}d, loaded in {dt:.2f}s")
        _assert_valid_embeddings(emb, min_vocab=100_000, expected_dim=800)


# ==========================================================================
# .kv format (gensim KeyedVectors pickle)
# ==========================================================================

class TestLoadKV:
    def test_nkjp(self):
        _skip_missing(KV_NKJP)
        t0 = time.perf_counter()
        emb = Embeddings.load(KV_NKJP)
        dt = time.perf_counter() - t0
        print(f"\n  nkjp.kv: {len(emb)} words, {emb.vector_size}d, loaded in {dt:.2f}s")
        _assert_valid_embeddings(emb, min_vocab=50_000, expected_dim=300)

    def test_glove(self):
        _skip_missing(KV_GLOVE)
        t0 = time.perf_counter()
        emb = Embeddings.load(KV_GLOVE)
        dt = time.perf_counter() - t0
        print(f"\n  glove.kv: {len(emb)} words, {emb.vector_size}d, loaded in {dt:.2f}s")
        _assert_valid_embeddings(emb, min_vocab=100_000, expected_dim=800)


# ==========================================================================
# .bin format (word2vec binary)
# ==========================================================================

class TestLoadBin:
    def test_nkjp(self):
        _skip_missing(BIN_NKJP)
        t0 = time.perf_counter()
        emb = Embeddings.load(BIN_NKJP)
        dt = time.perf_counter() - t0
        print(f"\n  nkjp.bin: {len(emb)} words, {emb.vector_size}d, loaded in {dt:.2f}s")
        _assert_valid_embeddings(emb, min_vocab=50_000, expected_dim=300)


# ==========================================================================
# .txt format (word2vec/GloVe text)
# ==========================================================================

class TestLoadTxt:
    def test_nkjp_sequential(self, nkjp_txt_seq):
        _assert_valid_embeddings(nkjp_txt_seq, min_vocab=50_000, expected_dim=300)

    def test_nkjp_parallel(self, nkjp_txt_par):
        _assert_valid_embeddings(nkjp_txt_par, min_vocab=50_000, expected_dim=300)

    def test_txt_sequential_vs_parallel_match(self, nkjp_txt_seq, nkjp_txt_par):
        """Parallel and sequential loading should produce identical results."""
        assert len(nkjp_txt_seq) == len(nkjp_txt_par), (
            f"Vocab size mismatch: seq={len(nkjp_txt_seq)} par={len(nkjp_txt_par)}"
        )
        assert nkjp_txt_seq.index_to_key == nkjp_txt_par.index_to_key, "Word order mismatch"
        assert np.allclose(nkjp_txt_seq.vectors, nkjp_txt_par.vectors, atol=1e-6), "Vector mismatch"


# ==========================================================================
# Cross-format consistency
# ==========================================================================

class TestCrossFormatConsistency:
    """Verify that the same embeddings loaded from different formats match."""

    def test_ssdembed_vs_kv_nkjp(self):
        """Vectors for common words should match across formats."""
        _skip_missing(SSDEMBED_NKJP)
        _skip_missing(KV_NKJP)
        emb_ss = Embeddings.load(SSDEMBED_NKJP)
        emb_kv = Embeddings.load(KV_NKJP)

        assert emb_ss.vector_size == emb_kv.vector_size
        # Compare common vocabulary (files may differ by 1-2 words)
        common = set(emb_ss.key_to_index) & set(emb_kv.key_to_index)
        assert len(common) >= min(len(emb_ss), len(emb_kv)) - 10, (
            f"Too few common words: {len(common)} of {len(emb_ss)}/{len(emb_kv)}"
        )
        # Spot-check 1000 random common words
        rng = np.random.default_rng(0)
        sample = rng.choice(list(common), size=min(1000, len(common)), replace=False)
        for w in sample:
            assert np.allclose(emb_ss[w], emb_kv[w], atol=1e-5), f"Mismatch for '{w}'"


# ==========================================================================
# Save round-trip with real data
# ==========================================================================

class TestSaveRoundTrip:
    """Load real embeddings, save to each format, reload, verify."""

    def test_roundtrip_ssdembed(self, nkjp_ssdembed):
        with tempfile.TemporaryDirectory() as td:
            stem = os.path.join(td, "rt")
            nkjp_ssdembed.save(stem)
            loaded = Embeddings.load(stem + ".ssdembed")
            assert len(loaded) == len(nkjp_ssdembed)
            assert np.allclose(nkjp_ssdembed.vectors, loaded.vectors, atol=1e-6)

    def test_roundtrip_bin(self, nkjp_ssdembed):
        with tempfile.TemporaryDirectory() as td:
            stem = os.path.join(td, "rt")
            with pytest.warns(UserWarning, match="normalization and ABTT metadata will be lost"):
                nkjp_ssdembed.save(stem, fmt="bin")
            loaded = Embeddings.load(stem + ".bin")
            assert len(loaded) == len(nkjp_ssdembed)
            assert np.allclose(nkjp_ssdembed.vectors, loaded.vectors, atol=1e-6)

    def test_roundtrip_txt(self, nkjp_ssdembed):
        # Text format is ~1 ms/word — sample 10k words to exercise encode/decode
        # without paying 4+ minutes to write the full 1.6M-vocab file.
        n = 10_000
        small = Embeddings(
            list(nkjp_ssdembed.index_to_key[:n]),
            nkjp_ssdembed.vectors[:n].copy(),
        )
        small.l2_normalized = nkjp_ssdembed.l2_normalized
        small.abtt_m = nkjp_ssdembed.abtt_m
        with tempfile.TemporaryDirectory() as td:
            stem = os.path.join(td, "rt")
            with pytest.warns(UserWarning, match="normalization and ABTT metadata will be lost"):
                small.save(stem, fmt="txt")
            loaded = Embeddings.load(stem + ".txt")
            assert len(loaded) == len(small)
            # Text format has limited precision (~6 digits)
            assert np.allclose(small.vectors, loaded.vectors, atol=1e-3)


# ==========================================================================
# Normalize with real embeddings
# ==========================================================================

class TestNormalizeReal:
    """Test normalize() on real embedding data."""

    def test_l2_normalize(self, nkjp_txt_seq):
        """L2 normalization should make all row norms ~1.0."""
        emb = Embeddings(list(nkjp_txt_seq.index_to_key), nkjp_txt_seq.vectors.copy())
        emb.normalize(l2=True, abtt_m=0, re_normalize=False)
        norms = np.linalg.norm(emb.vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), (
            f"Row norms not 1.0: min={norms.min():.6f} max={norms.max():.6f}"
        )

    def test_abtt_removes_dominant_component(self, nkjp_txt_seq):
        """ABTT(m=1) should reduce variance along top PC."""
        emb = Embeddings(list(nkjp_txt_seq.index_to_key), nkjp_txt_seq.vectors.copy())

        # Measure top PC variance before
        centered_before = emb.vectors - emb.vectors.mean(axis=0)
        _, S_before, _ = np.linalg.svd(centered_before[:5000], full_matrices=False)
        var_before = S_before[0] ** 2 / np.sum(S_before[:10] ** 2)

        emb.normalize(l2=True, abtt_m=1, re_normalize=True)

        # Measure top PC variance after
        centered_after = emb.vectors - emb.vectors.mean(axis=0)
        _, S_after, _ = np.linalg.svd(centered_after[:5000], full_matrices=False)
        var_after = S_after[0] ** 2 / np.sum(S_after[:10] ** 2)

        print(f"\n  Top PC variance: before={var_before:.4f} after={var_after:.4f}")
        assert var_after < var_before, "ABTT should reduce top PC dominance"

    def test_normalize_matches_pre_normalized(self, nkjp_txt_seq, nkjp_ssdembed):
        """Normalizing raw .txt should match the pre-normalized .ssdembed on common vocab."""
        emb = Embeddings(list(nkjp_txt_seq.index_to_key), nkjp_txt_seq.vectors.copy())
        emb.normalize(l2=True, abtt_m=1, re_normalize=True)

        # Compare on common vocab (may differ by 1-2 words)
        common = set(emb.key_to_index) & set(nkjp_ssdembed.key_to_index)
        assert len(common) >= min(len(emb), len(nkjp_ssdembed)) - 10

        # Spot-check 2000 words
        rng = np.random.default_rng(0)
        sample = rng.choice(list(common), size=min(2000, len(common)), replace=False)
        diffs = [np.max(np.abs(emb[w] - nkjp_ssdembed[w])) for w in sample]
        max_diff = max(diffs)
        mean_diff = np.mean(diffs)
        print(f"\n  Normalize match: max_diff={max_diff:.6f} mean_diff={mean_diff:.8f}")
        assert max_diff < 1e-3, f"Normalized vectors diverge: max_diff={max_diff:.6f}"

    def test_normalize_returns_self(self, nkjp_ssdembed):
        """normalize() should return self for chaining."""
        emb = Embeddings(
            list(nkjp_ssdembed.index_to_key),
            nkjp_ssdembed.vectors.copy(),
        )
        result = emb.normalize(l2=True)
        assert result is emb

    def test_normalize_preserves_vocab(self, nkjp_ssdembed):
        """normalize() should not alter vocabulary."""
        emb = Embeddings(
            list(nkjp_ssdembed.index_to_key),
            nkjp_ssdembed.vectors.copy(),
        )
        keys_before = list(emb.index_to_key)
        emb.normalize(l2=True, abtt_m=1)
        assert emb.index_to_key == keys_before
        assert len(emb) == len(keys_before)

    def test_similar_by_vector_after_normalize(self, nkjp_ssdembed):
        """Nearest-neighbor search should work on normalized embeddings."""
        word = "kraj"
        if word not in nkjp_ssdembed:
            pytest.skip(f"'{word}' not in vocabulary")
        vec = nkjp_ssdembed[word]
        nbrs = nkjp_ssdembed.similar_by_vector(vec, topn=5)
        assert len(nbrs) == 5
        assert nbrs[0][0] == word  # Self should be top neighbor
        assert np.isclose(nbrs[0][1], 1.0, atol=1e-4)
