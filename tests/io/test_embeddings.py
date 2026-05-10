"""Tests for ssdiff.embeddings.Embeddings: construction, lookup,
normalization, persistence round-trips, and similarity search.

All tests use tiny in-memory fixtures — no real embedding files, no network.
"""

from __future__ import annotations

import gzip
import warnings

import numpy as np
import pytest

from ssdiff.embeddings import Embeddings


# ---------------------------------------------------------------------------
# Helpers / local fixtures
# ---------------------------------------------------------------------------

def _make_emb(n: int = 6, dim: int = 4, seed: int = 0) -> Embeddings:
    """Build a tiny Embeddings with known non-unit vectors."""
    rng = np.random.default_rng(seed)
    words = [f"word{i}" for i in range(n)]
    mat = rng.normal(size=(n, dim)).astype(np.float32)
    return Embeddings(words, mat)


def _make_emb_named(words: list[str], dim: int = 4, seed: int = 7) -> Embeddings:
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(len(words), dim)).astype(np.float32)
    return Embeddings(words, mat)


# ---------------------------------------------------------------------------
# 1. Constructor validation
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_mismatched_keys_vectors_raises(self):
        """len(keys) != vectors.shape[0] must raise ValueError."""
        keys = ["a", "b", "c"]
        mat = np.ones((5, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="len\\(keys\\)"):
            Embeddings(keys, mat)

    def test_valid_construction(self):
        emb = _make_emb(n=4, dim=3)
        assert emb.vocab_size == 4
        assert emb.vector_size == 3
        assert emb.dim == 3

    def test_vectors_cast_to_float32(self):
        keys = ["x", "y"]
        mat = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        emb = Embeddings(keys, mat)
        assert emb.vectors.dtype == np.float32

    def test_source_path_initially_none(self):
        emb = _make_emb()
        assert emb._source_path is None


# ---------------------------------------------------------------------------
# 2. Basic lookup (__contains__, __len__, __getitem__)
# ---------------------------------------------------------------------------

class TestLookup:
    @pytest.fixture(scope="class")
    def emb(self) -> Embeddings:
        return _make_emb_named(["kraj", "narod", "panstwo"], dim=4)

    def test_len(self, emb):
        assert len(emb) == 3

    def test_contains_existing(self, emb):
        assert "kraj" in emb
        assert "narod" in emb

    def test_not_contains_missing(self, emb):
        assert "nieznanesowo" not in emb

    def test_getitem_returns_vector(self, emb):
        v = emb["kraj"]
        assert isinstance(v, np.ndarray)
        assert v.shape == (4,)

    def test_getitem_consistent_with_index(self, emb):
        idx = emb.key_to_index["narod"]
        np.testing.assert_array_equal(emb["narod"], emb.vectors[idx])

    def test_getitem_missing_raises(self, emb):
        with pytest.raises(KeyError):
            _ = emb["nonexistent_xyz"]


# ---------------------------------------------------------------------------
# 3. normalize(l2=True) — idempotent, every row norm == 1
# ---------------------------------------------------------------------------

class TestNormalizeL2:
    def test_l2_norms_are_unit(self):
        emb = _make_emb(n=8, dim=6)
        emb.normalize(l2=True, abtt=0)
        norms = np.linalg.norm(emb.vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_l2_normalize_twice_is_idempotent(self):
        emb = _make_emb(n=8, dim=6)
        emb.normalize(l2=True, abtt=0)
        v_after_first = emb.vectors.copy()
        emb.normalize(l2=True, abtt=0)  # second call — should be no-op
        np.testing.assert_array_equal(emb.vectors, v_after_first)

    def test_l2_normalized_flag_set(self):
        emb = _make_emb()
        assert not emb.l2_normalized
        emb.normalize(l2=True, abtt=0)
        assert emb.l2_normalized


# ---------------------------------------------------------------------------
# 4. normalize(abtt=N) — changes vectors; repeat same value warns; irreversible
# ---------------------------------------------------------------------------

class TestNormalizeABTT:
    def test_abtt_changes_vectors(self):
        emb = _make_emb(n=10, dim=6)
        original = emb.vectors.copy()
        emb.normalize(l2=True, abtt=2)
        assert not np.allclose(emb.vectors, original, atol=1e-6), (
            "ABTT should change the vectors"
        )

    def test_abtt_same_value_warns(self):
        emb = _make_emb(n=10, dim=6)
        emb.normalize(l2=True, abtt=2)
        vectors_after_first = emb.vectors.copy()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emb.normalize(l2=True, abtt=2)  # same m — should warn
        assert len(w) >= 1, "Expected a UserWarning for repeated ABTT"
        assert any("already applied" in str(warning.message).lower() for warning in w)

    def test_abtt_same_value_does_not_change_vectors(self):
        emb = _make_emb(n=10, dim=6)
        emb.normalize(l2=True, abtt=2)
        vectors_after_first = emb.vectors.copy()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            emb.normalize(l2=True, abtt=2)
        np.testing.assert_array_equal(emb.vectors, vectors_after_first)

    def test_abtt_abtt_attribute_set(self):
        emb = _make_emb(n=10, dim=6)
        emb.normalize(l2=True, abtt=2)
        assert emb.abtt == 2


# ---------------------------------------------------------------------------
# 5. Round-trips through .ssdembed, .bin, .txt, .vec
# ---------------------------------------------------------------------------

class TestRoundTrips:
    @pytest.fixture(scope="class")
    def source_emb(self) -> Embeddings:
        """A small Embeddings with distinct non-unit vectors."""
        words = ["alpha", "beta", "gamma", "delta"]
        rng = np.random.default_rng(123)
        mat = rng.normal(size=(4, 5)).astype(np.float32) * 2.0  # non-unit
        return Embeddings(words, mat)

    def test_round_trip_ssdembed(self, source_emb, tmp_path):
        stem = str(tmp_path / "model")
        source_emb.save(stem, fmt="ssdembed")
        loaded = Embeddings.load(stem + ".ssdembed")
        assert loaded.index_to_key == source_emb.index_to_key
        np.testing.assert_allclose(loaded.vectors, source_emb.vectors, atol=1e-6)

    def test_round_trip_bin(self, source_emb, tmp_path):
        stem = str(tmp_path / "model")
        source_emb.save(stem, fmt="bin")
        loaded = Embeddings.load(stem + ".bin")
        assert loaded.index_to_key == source_emb.index_to_key
        np.testing.assert_allclose(loaded.vectors, source_emb.vectors, atol=1e-6)

    def test_round_trip_txt(self, source_emb, tmp_path):
        stem = str(tmp_path / "model")
        source_emb.save(stem, fmt="txt")
        loaded = Embeddings.load(stem + ".txt")
        assert loaded.index_to_key == source_emb.index_to_key
        # Text format uses 6 significant digits — allow small atol
        np.testing.assert_allclose(loaded.vectors, source_emb.vectors, atol=1e-5)

    def test_round_trip_vec_extension(self, source_emb, tmp_path):
        """Save as .txt, rename to .vec, confirm .vec is loadable."""
        stem = str(tmp_path / "model")
        source_emb.save(stem, fmt="txt")
        import shutil
        txt_path = stem + ".txt"
        vec_path = stem + ".vec"
        shutil.copy(txt_path, vec_path)
        loaded = Embeddings.load(vec_path)
        assert loaded.index_to_key == source_emb.index_to_key
        np.testing.assert_allclose(loaded.vectors, source_emb.vectors, atol=1e-5)

    def test_round_trip_kv(self, source_emb, tmp_path):
        """Round-trip via .kv format; skip if gensim is not installed."""
        gensim = pytest.importorskip("gensim")
        stem = str(tmp_path / "model")
        source_emb.save(stem, fmt="kv")
        loaded = Embeddings.load(stem + ".kv")
        assert loaded.index_to_key == source_emb.index_to_key
        np.testing.assert_allclose(loaded.vectors, source_emb.vectors, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. Compressed variants (.txt.gz)
# ---------------------------------------------------------------------------

class TestCompressedFormats:
    @pytest.fixture(scope="class")
    def source_emb(self) -> Embeddings:
        words = ["lion", "tiger", "bear"]
        rng = np.random.default_rng(55)
        mat = rng.normal(size=(3, 4)).astype(np.float32)
        return Embeddings(words, mat)

    def test_round_trip_txt_gz(self, source_emb, tmp_path):
        """Write a .txt.gz manually and confirm Embeddings.load() reads it."""
        stem = str(tmp_path / "model")
        # First write plain text, then gzip it
        source_emb.save(stem, fmt="txt")
        txt_path = stem + ".txt"
        gz_path = stem + ".txt.gz"
        with open(txt_path, "rb") as fin, gzip.open(gz_path, "wb") as fout:
            fout.write(fin.read())
        loaded = Embeddings.load(gz_path)
        assert loaded.index_to_key == source_emb.index_to_key
        np.testing.assert_allclose(loaded.vectors, source_emb.vectors, atol=1e-5)

    def test_round_trip_bin_gz(self, source_emb, tmp_path):
        """Write a .bin.gz and confirm Embeddings.load() reads it."""
        stem = str(tmp_path / "model")
        source_emb.save(stem, fmt="bin")
        bin_path = stem + ".bin"
        gz_path = stem + ".bin.gz"
        with open(bin_path, "rb") as fin, gzip.open(gz_path, "wb") as fout:
            fout.write(fin.read())
        loaded = Embeddings.load(gz_path)
        assert loaded.index_to_key == source_emb.index_to_key
        np.testing.assert_allclose(loaded.vectors, source_emb.vectors, atol=1e-6)


# ---------------------------------------------------------------------------
# 7. similar_by_vector — "kraj" vector finds itself at rank 0 with cosine ≈ 1
# ---------------------------------------------------------------------------

class TestSimilarByVector:
    @pytest.fixture(scope="class")
    def emb(self) -> Embeddings:
        # Use the VOCAB_20 words with a unit-normalized tiny embedding
        words = [
            "kraj", "narod", "panstwo",
            "piekny", "silny", "zly", "dobry",
            "wielki", "maly", "stary",
        ]
        rng = np.random.default_rng(42)
        mat = rng.normal(size=(len(words), 8)).astype(np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        emb = Embeddings(words, mat)
        emb.l2_normalized = True  # rows are unit by construction
        return emb

    def test_self_similarity_rank0(self, emb):
        v = emb["kraj"]
        results = emb.similar_by_vector(v, topn=3)
        assert len(results) == 3
        top_word, top_sim = results[0]
        assert top_word == "kraj"
        assert abs(top_sim - 1.0) < 1e-6, f"Expected cosine≈1.0, got {top_sim}"

    def test_zero_vector_returns_empty(self, emb):
        zero = np.zeros(8, dtype=np.float32)
        results = emb.similar_by_vector(zero, topn=3)
        assert results == []

    def test_topn_respected(self, emb):
        v = emb["narod"]
        results = emb.similar_by_vector(v, topn=5)
        assert len(results) == 5

    def test_results_sorted_descending(self, emb):
        v = emb["panstwo"]
        results = emb.similar_by_vector(v, topn=4)
        sims = [s for _, s in results]
        assert sims == sorted(sims, reverse=True)


# ---------------------------------------------------------------------------
# 9. save(filename=None) without source path raises ValueError
# ---------------------------------------------------------------------------

class TestSaveNoSourcePath:
    def test_raises_valueerror_when_no_source_path(self):
        emb = _make_emb(n=3, dim=3)
        assert emb._source_path is None
        with pytest.raises(ValueError, match="filename is required"):
            emb.save(filename=None, fmt="ssdembed")

    def test_save_with_explicit_filename_works(self, tmp_path):
        emb = _make_emb(n=3, dim=3)
        stem = str(tmp_path / "explicit")
        emb.save(filename=stem, fmt="ssdembed")  # should not raise
        loaded = Embeddings.load(stem + ".ssdembed")
        assert len(loaded) == 3


# ---------------------------------------------------------------------------
# 10. load nonexistent path raises FileNotFoundError
# ---------------------------------------------------------------------------

class TestLoadErrors:
    def test_nonexistent_ssdembed_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            Embeddings.load("/tmp/nonexistent_ssdiff_test_xyz_12345.ssdembed")

    def test_nonexistent_txt_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            Embeddings.load("/tmp/nonexistent_ssdiff_test_xyz_12345.txt")

    def test_nonexistent_bin_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            Embeddings.load("/tmp/nonexistent_ssdiff_test_xyz_12345.bin")


# ---------------------------------------------------------------------------
# 11. load auto-detects unit rows → l2_normalized flag
# ---------------------------------------------------------------------------

def test_load_autodetects_unit_rows(tmp_path):
    """Embeddings.load auto-detects unit rows and sets l2_normalized=True."""
    import numpy as np
    from ssdiff.embeddings import Embeddings

    rng = np.random.default_rng(0)
    mat = rng.normal(size=(50, 8)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    emb = Embeddings([f"w{i}" for i in range(50)], mat)
    emb.l2_normalized = True
    p = tmp_path / "norm"
    emb.save(str(p), fmt="txt")

    reloaded = Embeddings.load(str(p) + ".txt")
    assert reloaded.l2_normalized is True


def test_load_does_not_falsely_set_unit_flag(tmp_path):
    """Embeddings.load leaves l2_normalized=False when rows are not unit."""
    import numpy as np
    from ssdiff.embeddings import Embeddings

    rng = np.random.default_rng(0)
    mat = rng.normal(size=(50, 8)).astype(np.float32) * 5.0  # not unit
    emb = Embeddings([f"w{i}" for i in range(50)], mat)
    p = tmp_path / "raw"
    emb.save(str(p), fmt="txt")

    reloaded = Embeddings.load(str(p) + ".txt")
    assert reloaded.l2_normalized is False


def test_ssd_init_raises_on_unnormed_embedding(tiny_kv, sample_docs, sample_y, lexicon):
    """SSD requires L2-normalised embeddings."""
    import pytest
    from ssdiff import SSD, Corpus
    # Force the flag off to simulate a raw (un-normalised) embedding.
    tiny_kv_raw = type(tiny_kv).__new__(type(tiny_kv))
    tiny_kv_raw.__dict__.update(tiny_kv.__dict__)
    tiny_kv_raw.l2_normalized = False
    corpus = Corpus(sample_docs, lang="pl", pretokenized=True)
    with pytest.raises(RuntimeError, match="L2-normalised embeddings"):
        SSD(tiny_kv_raw, corpus, sample_y, lexicon)


def test_get_vector_norm_true_raises_on_unnormed():
    """get_vector(word, norm=True) raises when the embedding is not unit-normed."""
    import numpy as np, pytest
    from ssdiff.embeddings import Embeddings
    raw = np.array([[3.0, 4.0]], dtype=np.float32)  # norm 5
    emb = Embeddings(["a"], raw)
    with pytest.raises(RuntimeError, match="L2-normalised"):
        emb.get_vector("a", norm=True)


def test_similar_by_vector_raises_on_unnormed():
    """similar_by_vector raises when the embedding is not unit-normed."""
    import numpy as np, pytest
    from ssdiff.embeddings import Embeddings
    raw = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    emb = Embeddings(["a", "b"], raw)
    with pytest.raises(RuntimeError, match="L2-normalised"):
        emb.similar_by_vector(np.array([1.0, 0.0], dtype=np.float32))


def test_lazy_normaliser_state_removed():
    """The lazy normaliser API and its hidden state are gone."""
    import numpy as np
    from ssdiff.embeddings import Embeddings
    assert not hasattr(Embeddings, "get_normed_vectors")
    fresh = Embeddings(["a"], np.ones((1, 2), dtype=np.float32))
    assert not hasattr(fresh, "_is_unit_normed")
