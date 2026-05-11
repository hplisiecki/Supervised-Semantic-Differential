"""Tests for Embeddings.load(ram_efficient=True) and attach_corpus."""

from __future__ import annotations

import os

import numpy as np
import pytest

from ssdiff import Corpus, SSD
from ssdiff.embeddings import Embeddings


def _make_ssdembed(tmp_path, n: int = 200, dim: int = 8, normalised: bool = True):
    """Build and save an .ssdembed fixture; return its path."""
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(n, dim)).astype(np.float32)
    if normalised:
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    emb = Embeddings([f"w{i}" for i in range(n)], mat)
    if normalised:
        emb.l2_normalized = True
    p = tmp_path / "fix"
    emb.save(str(p), fmt="ssdembed")
    return str(p) + ".ssdembed"


def test_ram_format_guard_txt(tmp_path):
    """ram_efficient=True only accepts .ssdembed (uncompressed)."""
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(10, 4)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    emb = Embeddings([f"w{i}" for i in range(10)], mat)
    emb.l2_normalized = True
    p = tmp_path / "raw"
    emb.save(str(p), fmt="txt")

    with pytest.raises(ValueError, match=r"\.ssdembed"):
        Embeddings.load(str(p) + ".txt", ram_efficient=True)


def test_ram_small_vocab_no_op(tmp_path):
    """When vocab <= _RAM_TOP_N, ram_efficient=True falls through to full load."""
    path = _make_ssdembed(tmp_path, n=200, dim=8)
    full = Embeddings.load(path)
    ram = Embeddings.load(path, ram_efficient=True)
    assert ram._partial is False
    assert ram.vectors.shape == full.vectors.shape
    np.testing.assert_array_equal(ram.vectors, full.vectors)


def test_ram_phase1_partial_load(tmp_path, monkeypatch):
    """With cap monkeypatched < V, Phase 1 yields partial state."""
    import ssdiff.embeddings as em
    monkeypatch.setattr(em, "_RAM_TOP_N", 10)

    path = _make_ssdembed(tmp_path, n=50, dim=4)
    ram = Embeddings.load(path, ram_efficient=True)
    assert ram._partial is True
    assert ram._corpus_attached is False
    assert ram.vectors.shape == (10, 4)
    assert len(ram.index_to_key) == 50  # full vocabulary still indexed
    assert ram._mmap is not None
    assert ram.l2_normalized is True


def test_ram_phase1_unnormed_file_raises(tmp_path, monkeypatch):
    """Phase 1 verifies the slice is unit; raises if not."""
    import ssdiff.embeddings as em
    monkeypatch.setattr(em, "_RAM_TOP_N", 10)

    path = _make_ssdembed(tmp_path, n=50, dim=4, normalised=False)
    with pytest.raises(RuntimeError, match="pre-normalised"):
        Embeddings.load(path, ram_efficient=True)


def _make_ssdembed_with_named_words(tmp_path, words, dim=4):
    """Build and save an .ssdembed fixture using explicit words."""
    n = len(words)
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(n, dim)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    emb = Embeddings(list(words), mat)
    emb.l2_normalized = True
    p = tmp_path / "fix"
    emb.save(str(p), fmt="ssdembed")
    return str(p) + ".ssdembed"


def test_attach_corpus_materialises_extras_flat(tmp_path, monkeypatch):
    """attach_corpus pulls extras from mmap; mmap is closed afterwards."""
    import ssdiff.embeddings as em
    monkeypatch.setattr(em, "_RAM_TOP_N", 5)

    words = [f"w{i}" for i in range(20)]
    path = _make_ssdembed_with_named_words(tmp_path, words)

    ram = Embeddings.load(path, ram_efficient=True)
    assert ram._partial is True
    assert ram.vectors.shape == (5, 4)

    # Corpus references words at ranks 7, 12, 18 — none in the prefix.
    docs = [["w7", "w12"], ["w18"]]
    corpus = Corpus(docs, lang="en", pretokenized=True)
    ram.attach_corpus(corpus)

    assert ram._corpus_attached is True
    assert ram._mmap is None
    # 5 prefix rows + 3 extras = 8 rows
    assert ram.vectors.shape == (8, 4)
    # Lookup returns the canonical row from the source file.
    full = Embeddings.load(path)
    np.testing.assert_array_equal(ram["w7"], full["w7"])
    np.testing.assert_array_equal(ram["w12"], full["w12"])
    np.testing.assert_array_equal(ram["w18"], full["w18"])
    # OOV stays OOV.
    assert "not_in_vocab" not in ram


def test_attach_corpus_materialises_extras_profile(tmp_path, monkeypatch):
    """Same as above but with a profile-style nested corpus."""
    import ssdiff.embeddings as em
    monkeypatch.setattr(em, "_RAM_TOP_N", 5)

    words = [f"w{i}" for i in range(20)]
    path = _make_ssdembed_with_named_words(tmp_path, words)

    ram = Embeddings.load(path, ram_efficient=True)
    docs = [[["w7"], ["w12"]], [["w18", "w7"]]]  # nested profile shape
    corpus = Corpus(docs, lang="en", pretokenized=True)
    ram.attach_corpus(corpus)

    assert ram._corpus_attached is True
    assert ram.vectors.shape == (8, 4)


def test_attach_corpus_no_op_in_full_mode(tiny_kv, sample_docs):
    """attach_corpus on a full-mode embedding is a no-op."""
    corpus = Corpus(sample_docs, lang="pl", pretokenized=True)
    ret = tiny_kv.attach_corpus(corpus)
    assert ret is tiny_kv
    assert tiny_kv._partial is False


def test_ssd_init_raises_if_attach_corpus_skipped(tmp_path, monkeypatch):
    """SSD raises a clear error when partial embeddings haven't been attached."""
    import ssdiff.embeddings as em
    monkeypatch.setattr(em, "_RAM_TOP_N", 5)

    words = [f"w{i}" for i in range(20)]
    path = _make_ssdembed_with_named_words(tmp_path, words)
    ram = Embeddings.load(path, ram_efficient=True)

    docs = [["w0", "w7"], ["w12"]]
    corpus = Corpus(docs, lang="en", pretokenized=True)
    y = [0.1, 0.2]
    with pytest.raises(RuntimeError, match="attach_corpus"):
        SSD(ram, corpus, y, lexicon={"w0"})


def test_ram_normalize_raises(tmp_path, monkeypatch):
    """normalize() raises in RAM mode."""
    import ssdiff.embeddings as em
    monkeypatch.setattr(em, "_RAM_TOP_N", 5)
    path = _make_ssdembed_with_named_words(tmp_path, [f"w{i}" for i in range(20)])
    ram = Embeddings.load(path, ram_efficient=True)
    with pytest.raises(RuntimeError, match="read-only"):
        ram.normalize(l2=True)
    with pytest.raises(RuntimeError, match="read-only"):
        ram.normalize(l2=False, abtt=1)


def test_ram_save_raises(tmp_path, monkeypatch):
    """save() raises in RAM mode for every supported format."""
    import ssdiff.embeddings as em
    monkeypatch.setattr(em, "_RAM_TOP_N", 5)
    path = _make_ssdembed_with_named_words(tmp_path, [f"w{i}" for i in range(20)])
    ram = Embeddings.load(path, ram_efficient=True)
    out = tmp_path / "out"
    for fmt in ("ssdembed", "txt", "bin", "kv"):
        with pytest.raises(RuntimeError, match="RAM-efficient"):
            ram.save(str(out), fmt=fmt)


def test_fit_multipls_no_warn_in_ram_mode_with_varimax(tmp_path, monkeypatch):
    """fit_multipls in RAM mode runs varimax cleanly: no partial-vocab warning.

    Rationale: rotation now explicitly targets the top ``rotation_vocab`` rows
    (default 50_000), which collapses the "partial vs full" distinction —
    ram_efficient + default rotation_vocab is identical to full-load +
    rotation_vocab=50_000.
    """
    import warnings as _warnings
    import ssdiff.embeddings as em
    monkeypatch.setattr(em, "_RAM_TOP_N", 5)

    words = [f"w{i}" for i in range(40)]
    path = _make_ssdembed_with_named_words(tmp_path, words, dim=4)
    ram = Embeddings.load(path, ram_efficient=True)

    docs = [[f"w{i % 40}" for i in range(j, j + 4)] for j in range(20)]
    corpus = Corpus(docs, lang="en", pretokenized=True)
    ram.attach_corpus(corpus)

    y = list(np.linspace(0.0, 1.0, 20))
    ssd = SSD(ram, corpus, y, lexicon={"w0", "w1", "w2"})

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        res = ssd.fit_multipls(k=1, n_splits=10, random_state=0)

    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert not any("materialized vocab subset" in m for m in msgs), msgs
    assert res is not None


def test_fit_multipls_raw_rotation_no_warn_in_ram_mode(tmp_path, monkeypatch):
    """rotate='raw' does not use the embedding matrix → no partial-vocab warning."""
    import warnings as _warnings
    import ssdiff.embeddings as em
    monkeypatch.setattr(em, "_RAM_TOP_N", 5)

    words = [f"w{i}" for i in range(40)]
    path = _make_ssdembed_with_named_words(tmp_path, words, dim=4)
    ram = Embeddings.load(path, ram_efficient=True)

    docs = [[f"w{i % 40}" for i in range(j, j + 4)] for j in range(20)]
    corpus = Corpus(docs, lang="en", pretokenized=True)
    ram.attach_corpus(corpus)

    y = list(np.linspace(0.0, 1.0, 20))
    ssd = SSD(ram, corpus, y, lexicon={"w0", "w1", "w2"})

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        res = ssd.fit_multipls(k=1, rotate="raw", n_splits=10, random_state=0)

    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert not any("materialized vocab subset" in m for m in msgs)
    assert res is not None


def test_similar_by_vector_clamps_in_ram(tmp_path, monkeypatch):
    """restrict_vocab > _RAM_TOP_N is clamped and emits one UserWarning."""
    import warnings
    import ssdiff.embeddings as em
    monkeypatch.setattr(em, "_RAM_TOP_N", 5)

    words = [f"w{i}" for i in range(50)]
    path = _make_ssdembed_with_named_words(tmp_path, words, dim=4)
    ram = Embeddings.load(path, ram_efficient=True)

    q = ram.vectors[0]
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        out_clamped = ram.similar_by_vector(q, topn=3, restrict_vocab=42)
        assert any("clamped to 5" in str(w.message) for w in ws)
        assert any(issubclass(w.category, UserWarning) for w in ws)

    out_at_cap = ram.similar_by_vector(q, topn=3, restrict_vocab=5)
    assert [w for w, _ in out_clamped] == [w for w, _ in out_at_cap]


def test_similar_by_vector_no_warning_when_within_cap(tmp_path, monkeypatch):
    """No warning for restrict_vocab <= _RAM_TOP_N."""
    import warnings
    import ssdiff.embeddings as em
    monkeypatch.setattr(em, "_RAM_TOP_N", 5)

    words = [f"w{i}" for i in range(50)]
    path = _make_ssdembed_with_named_words(tmp_path, words, dim=4)
    ram = Embeddings.load(path, ram_efficient=True)

    q = ram.vectors[0]
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        ram.similar_by_vector(q, topn=2, restrict_vocab=5)
        assert not any(issubclass(w.category, UserWarning) for w in ws)


def _build_corpus_y(words, n_docs=30):
    rng = np.random.default_rng(7)
    docs = []
    for i in range(n_docs):
        # Each doc mixes a seed with a few extras spread across the vocab.
        seed = words[i % 3]              # w0/w1/w2 are the lexicon
        extras = [words[(i * 5 + k) % len(words)] for k in range(3)]
        docs.append([seed, *extras])
    y_num = rng.normal(size=n_docs)
    y_grp = np.array(["A" if v > 0 else "B" for v in y_num])
    return docs, y_num, y_grp


@pytest.mark.parametrize("docs_kind", ["flat", "profile"])
def test_ram_vs_full_pls_equivalence(tmp_path, monkeypatch, docs_kind):
    import ssdiff.embeddings as em
    monkeypatch.setattr(em, "_RAM_TOP_N", 5)
    words = [f"w{i}" for i in range(40)]
    path = _make_ssdembed_with_named_words(tmp_path, words, dim=4)

    docs, y_num, _ = _build_corpus_y(words, n_docs=30)
    if docs_kind == "profile":
        docs = [[d] for d in docs]
    corpus = Corpus(docs, lang="en", pretokenized=True)
    lexicon = {"w0", "w1", "w2"}

    full = Embeddings.load(path)
    full_pls = SSD(full, corpus, y_num, lexicon).fit_pls()

    ram = Embeddings.load(path, ram_efficient=True)
    ram.attach_corpus(corpus)
    ram_pls = SSD(ram, corpus, y_num, lexicon).fit_pls()

    np.testing.assert_array_equal(full_pls.beta, ram_pls.beta)
    assert full_pls.stats.r2 == ram_pls.stats.r2


def test_ram_vs_full_ols_equivalence(tmp_path, monkeypatch):
    import ssdiff.embeddings as em
    monkeypatch.setattr(em, "_RAM_TOP_N", 5)
    words = [f"w{i}" for i in range(40)]
    path = _make_ssdembed_with_named_words(tmp_path, words, dim=4)

    docs, y_num, _ = _build_corpus_y(words, n_docs=30)
    corpus = Corpus(docs, lang="en", pretokenized=True)
    lexicon = {"w0", "w1", "w2"}

    full = Embeddings.load(path)
    full_ols = SSD(full, corpus, y_num, lexicon).fit_ols(fixed_k=2)

    ram = Embeddings.load(path, ram_efficient=True)
    ram.attach_corpus(corpus)
    ram_ols = SSD(ram, corpus, y_num, lexicon).fit_ols(fixed_k=2)

    np.testing.assert_array_equal(full_ols.beta, ram_ols.beta)
    assert full_ols.stats.r2 == ram_ols.stats.r2


def test_ram_vs_full_groups_equivalence(tmp_path, monkeypatch):
    import ssdiff.embeddings as em
    monkeypatch.setattr(em, "_RAM_TOP_N", 5)
    words = [f"w{i}" for i in range(40)]
    path = _make_ssdembed_with_named_words(tmp_path, words, dim=4)

    docs, _, _ = _build_corpus_y(words, n_docs=40)
    # Use alternating labels to guarantee exactly 20 docs per group
    # (MIN_GROUP_SIZE=20), making filter_small_groups a no-op.
    y_grp = np.array(["A" if i % 2 == 0 else "B" for i in range(40)])
    corpus = Corpus(docs, lang="en", pretokenized=True)
    lexicon = {"w0", "w1", "w2"}

    full = Embeddings.load(path)
    full_grp = SSD(full, corpus, y_grp, lexicon).fit_groups()

    ram = Embeddings.load(path, ram_efficient=True)
    ram.attach_corpus(corpus)
    ram_grp = SSD(ram, corpus, y_grp, lexicon).fit_groups()

    assert list(full_grp.pairs) == list(ram_grp.pairs)
    for pair in full_grp.pairs:
        np.testing.assert_array_equal(
            full_grp[(pair.g1, pair.g2)].beta,
            ram_grp[(pair.g1, pair.g2)].beta,
        )


# ---------------------------------------------------------------------------
# Issue 1: get_vector must use _local_row in partial mode
# ---------------------------------------------------------------------------

def test_get_vector_partial_mode_uses_local_row(tmp_path, monkeypatch):
    """get_vector must translate vocab index → local row in partial mode."""
    import ssdiff.embeddings as em
    monkeypatch.setattr(em, "_RAM_TOP_N", 5)

    words = [f"w{i}" for i in range(20)]
    path = _make_ssdembed_with_named_words(tmp_path, words)
    full = Embeddings.load(path)

    ram = Embeddings.load(path, ram_efficient=True)
    docs = [["w7", "w12", "w18"]]
    corpus = Corpus(docs, lang="en", pretokenized=True)
    ram.attach_corpus(corpus)

    # An extras token (vocab index 12, local row 5+k after attach_corpus).
    np.testing.assert_array_equal(ram.get_vector("w12"), full["w12"])
    np.testing.assert_array_equal(ram.get_vector("w12", norm=True), full["w12"])

    # OOV / not-materialised should still raise.
    # Use a word that is in vocab but was NOT in the corpus (rank 8 not in docs).
    with pytest.raises(KeyError, match="not materialised"):
        ram.get_vector("w8")


# ---------------------------------------------------------------------------
# Issue 2: similar_by_vector cap must not grow after attach_corpus
# ---------------------------------------------------------------------------

def test_similar_by_vector_after_attach_corpus_returns_correct_words(tmp_path, monkeypatch):
    """After attach_corpus, similar_by_vector must still pair words with their own vectors."""
    import ssdiff.embeddings as em
    monkeypatch.setattr(em, "_RAM_TOP_N", 5)

    words = [f"w{i}" for i in range(20)]
    path = _make_ssdembed_with_named_words(tmp_path, words)
    full = Embeddings.load(path)

    ram = Embeddings.load(path, ram_efficient=True)
    docs = [["w7", "w12", "w18"]]
    corpus = Corpus(docs, lang="en", pretokenized=True)
    ram.attach_corpus(corpus)
    # vectors.shape[0] is now 8 (5 prefix + 3 extras), but only the prefix
    # has rank-aligned (word, row) pairs.

    # Searching with the default (no restrict) must NOT scan extras as if they
    # were the next vocab slots.  Cap should stay at the prefix.
    q = full["w0"]
    out = ram.similar_by_vector(q, topn=3)
    # All returned words must be in the rank-aligned prefix (w0..w4).
    for word, _ in out:
        assert ram.key_to_index[word] < 5, (
            f"similar_by_vector returned {word} outside the prefix"
        )


# ---------------------------------------------------------------------------
# Issue 3: format guard covers .txt, .bin, .vec (and .kv if gensim available)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["txt", "bin", "vec"])
def test_ram_format_guard_text_binary(tmp_path, fmt):
    """ram_efficient=True only accepts .ssdembed (uncompressed)."""
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(10, 4)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    emb = Embeddings([f"w{i}" for i in range(10)], mat)
    emb.l2_normalized = True
    p = tmp_path / "raw"
    if fmt == "vec":
        # .vec is just .txt content with a different extension.
        emb.save(str(p), fmt="txt")
        src = str(p) + ".txt"
        new_path = str(p) + ".vec"
        os.rename(src, new_path)
        path = new_path
    else:
        emb.save(str(p), fmt=fmt)
        path = str(p) + "." + fmt

    with pytest.raises(ValueError, match=r"\.ssdembed"):
        Embeddings.load(path, ram_efficient=True)


def test_ram_format_guard_kv(tmp_path):
    """ram_efficient=True rejects .kv (gensim-style)."""
    pytest.importorskip("gensim")
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(10, 4)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    emb = Embeddings([f"w{i}" for i in range(10)], mat)
    emb.l2_normalized = True
    p = tmp_path / "raw"
    emb.save(str(p), fmt="kv")

    with pytest.raises(ValueError, match=r"\.ssdembed"):
        Embeddings.load(str(p) + ".kv", ram_efficient=True)
