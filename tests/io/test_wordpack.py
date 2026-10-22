"""Wordpack embeddings: loading, lookup, normalize, tokenizing, fitting, predict.

All tests use ``tests/fixtures/wordpack/fixture.wordpack``: a synthetic pack
of 48 keys (``KEYS``, in row order) at D = 16.
"""

from __future__ import annotations

import copy
import os
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ssdiff.embeddings as em
from ssdiff import SSD, Embeddings
from ssdiff.utils.text import PreprocessedDoc
from ssdiff.wordpack.reader import WordPack, WordPackError, read_header
from tests.conftest import PARITY_DIR, make_kv

def _tokens(docs):
    """Stand-in for a text: ``with_corpus`` reads only ``.docs``."""
    return SimpleNamespace(docs=docs)


KEYS = [
    "the", "i", "do", "n't", "'s", "'re", "'ve", "'ll", "'d", "'m",
    "ca", "wo", "it", "is", "a", "dog", "dogs", "cat",
    "well-known", "blue", "green", "new york", "café", "don",
    "covid-19", "covid", "4chan", "u.s.", "and/or", "and", "or",
    "it.", "don't", "e.g.", "shit", "dick", "duck", "never", "people",
    "like", "swim", "happens", "with", "in", "are", "they", "x", "b",
]

def test_fixture_pack_opens_and_passes_its_selftest(wordpack_path):
    with WordPack(wordpack_path) as pack:
        assert (pack.n, pack.dims) == (48, 16)
        assert list(pack.keys) == KEYS
        pack.selftest()


def _direct_rows(path, m: int) -> np.ndarray:
    """Every row of the pack at ABTT m, computed here from the stored int8 rows, mean and components."""
    with open(path, "rb") as f:
        base, tensors, _ = read_header(f, str(path))

        def tensor(name, dtype):
            start, end = tensors[name]["data_offsets"]
            f.seek(base + start)
            return np.frombuffer(f.read(end - start), dtype=dtype).reshape(tensors[name]["shape"])

        q = tensor("matrix", np.int8).astype(np.float64)
        abtt = tensor("abtt", "<f4").astype(np.float64)
    x = q / np.linalg.norm(q, axis=1, keepdims=True)
    if m:
        x = x - abtt[0]
        comps = abtt[1 : m + 1]
        x = x - (x @ comps.T) @ comps
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def test_wordpack_loads_at_abtt_1_with_its_metadata(wordpack_path):
    emb = Embeddings.load(wordpack_path)
    assert (emb.abtt, emb.l2_normalized) == (1, True)
    assert (emb.name, emb.version, emb.lang) == ("en-fixture", "1", "en")
    assert emb.stopwords == ["the", "a", "is", "i", "and", "or"]
    assert emb.neighbor_filter.pattern == r".*\d"
    assert emb.vectors.dtype == np.float32
    assert emb.vectors.shape == (48, 16)
    np.testing.assert_allclose(emb.vectors, _direct_rows(wordpack_path, 1), atol=1e-5)


def test_wordpack_lookup_uses_the_pack_table_not_a_dict(wordpack_path):
    emb = Embeddings.load(wordpack_path)
    assert emb.key_to_index is None
    assert len(emb) == 48
    assert emb.index_to_key[15] == "dog"
    assert "dogs" in emb and "new york" in emb
    np.testing.assert_array_equal(emb["dogs"], emb.vectors[16])
    np.testing.assert_array_equal(emb.get_vector("dogs", norm=True), emb.vectors[16])


def test_wordpack_absent_and_non_string_keys(wordpack_path):
    emb = Embeddings.load(wordpack_path)
    assert "zebra" not in emb
    assert 5 not in emb
    with pytest.raises(KeyError):
        emb["zebra"]


def test_failing_selftest_raises_naming_pack_and_reader_version(wordpack_path, tmp_path):
    data = bytearray(Path(wordpack_path).read_bytes())
    with open(wordpack_path, "rb") as f:
        base, tensors, _ = read_header(f, wordpack_path)
    off = base + tensors["matrix"]["data_offsets"][0]
    data[off] = (data[off] + 64) % 256  # one int8 component of row 0, which the self-test checks
    bad = tmp_path / "bad.wordpack"
    bad.write_bytes(bytes(data))
    with pytest.raises(WordPackError, match=r"en-fixture-1 .*self-test failed \(reader format version 1\)"):
        Embeddings.load(str(bad))


def test_a_file_that_is_not_a_wordpack_raises_naming_it(tmp_path):
    fake = tmp_path / "fake.wordpack"
    fake.write_bytes(b"not a wordpack at all")
    with pytest.raises(WordPackError, match=r"fake\.wordpack"):
        Embeddings.load(str(fake))


def test_pack_under_50k_rows_loads_fully_and_searches_without_clamping(wordpack_path):
    emb = Embeddings.load(wordpack_path)
    assert emb._partial is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        top = emb.similar_by_vector(emb["dog"], topn=3, restrict_vocab=10_000)
    assert top[0][0] == "dog"


def test_save_refuses_a_wordpack(wordpack_path, tmp_path):
    with pytest.raises(RuntimeError, match="wordpack"):
        Embeddings.load(wordpack_path).save(str(tmp_path / "out"))


def test_pickling_a_wordpack_raises(wordpack_path):
    with pytest.raises(RuntimeError, match="wordpack"):
        pickle.dumps(Embeddings.load(wordpack_path))


def test_word_beyond_the_prefix_is_read_on_request(wordpack_path, monkeypatch):
    full = Embeddings.load(wordpack_path)
    monkeypatch.setattr(em, "_RAM_TOP_N", 18)
    small = Embeddings.load(wordpack_path)
    assert small._partial
    assert small.vectors.shape == (18, 16)
    assert "swim" in small
    np.testing.assert_allclose(small["swim"], full["swim"], atol=1e-6)
    np.testing.assert_allclose(small.get_vector("they", norm=True), full["they"], atol=1e-6)
    assert KEYS.index("swim") not in small._local_row  # read, not cached on emb


def test_with_corpus_on_a_wordpack_and_a_view_of_a_view(wordpack_path, monkeypatch):
    full = Embeddings.load(wordpack_path)
    monkeypatch.setattr(em, "_RAM_TOP_N", 18)
    small = Embeddings.load(wordpack_path)
    view_a = small.with_corpus(_tokens([["swim", "dog", "zebra"]]))
    view_ab = view_a.with_corpus(_tokens([["they", "swim"]]))
    assert small._extras is None
    assert view_a._extras.shape == (1, 16)
    assert view_ab._extras.shape == (2, 16)
    for w in ("swim", "they", "dog"):
        np.testing.assert_allclose(view_ab[w], full[w], atol=1e-6)


def test_ssd_and_views_leave_the_pack_reusable_across_texts(wordpack_path, monkeypatch):
    full = Embeddings.load(wordpack_path)
    monkeypatch.setattr(em, "_RAM_TOP_N", 18)
    small = Embeddings.load(wordpack_path)
    ssd = SSD(small, ["dog swim", "they"], use_full_doc=True)
    assert ssd.embeddings is not small
    assert small._extras is None and small._local_row == {i: i for i in range(18)}
    view_a = small.with_corpus(_tokens([["swim"]]))
    view_b = small.with_corpus(_tokens([["they"]]))
    assert KEYS.index("they") not in view_a._local_row
    assert KEYS.index("swim") not in view_b._local_row
    for view in (view_a, view_b):
        for w in ("swim", "they"):
            np.testing.assert_allclose(view[w], full[w], atol=1e-6)


@pytest.mark.parametrize("entry", ["ssd", "attach"])
def test_new_text_rows_load_once_not_on_every_lookup(wordpack_path, monkeypatch, entry):
    """SSD on a view of another text, and result.attach(texts=...), read the
    new text's rows up front; lookups afterwards read nothing from the file."""
    full = Embeddings.load(wordpack_path)
    texts, y = _texts(40, seed=1)
    new_texts = texts[:38] + ["dog they", "happens they"]
    res = SSD(full, texts, use_full_doc=True).fit_regression(y, fixed_k=2)
    monkeypatch.setattr(em, "_RAM_TOP_N", 18)
    first = Embeddings.load(wordpack_path).with_corpus(_tokens([["swim"]]))
    if entry == "ssd":
        emb = SSD(first, new_texts, use_full_doc=True).embeddings
    else:
        r = copy.copy(res)
        r.embeddings = None
        emb = r.attach(texts=new_texts, embeddings=first).embeddings

    reads = []
    real = Embeddings._read_rows
    monkeypatch.setattr(Embeddings, "_read_rows", lambda self, idx: reads.append(list(idx)) or real(self, idx))
    for w in ("swim", "they", "happens"):
        np.testing.assert_allclose(emb[w], full[w], atol=1e-6)
    assert reads == []


def test_neighbour_search_stays_on_the_prefix_and_clamps_with_a_warning(wordpack_path, monkeypatch):
    monkeypatch.setattr(em, "_RAM_TOP_N", 18)
    view = Embeddings.load(wordpack_path).with_corpus(_tokens([["swim", "they"]]))
    q = view["they"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        default = view.similar_by_vector(q, topn=3)
    with pytest.warns(UserWarning, match="clamped to 18"):
        clamped = view.similar_by_vector(q, topn=3, restrict_vocab=40)
    assert clamped == default
    assert all(KEYS.index(w) < 18 for w, _ in default)


def test_threaded_reads_beyond_the_prefix_return_the_right_rows(wordpack_path, monkeypatch):
    full = Embeddings.load(wordpack_path)
    monkeypatch.setattr(em, "_RAM_TOP_N", 5)
    small = Embeddings.load(wordpack_path)
    words = KEYS[5:] * 20
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(small.__getitem__, words))
    for w, row in zip(words, rows):
        np.testing.assert_allclose(row, full[w], atol=1e-6)


def test_normalize_moves_abtt_down_and_up(wordpack_path):
    emb = Embeddings.load(wordpack_path)
    for m in (0, 3, 1):
        emb.normalize(abtt=m)
        assert emb.abtt == m and emb.l2_normalized
        np.testing.assert_allclose(emb.vectors, _direct_rows(wordpack_path, m), atol=1e-5)


def test_normalize_abtt_1_on_a_fresh_load_does_nothing(wordpack_path):
    emb = Embeddings.load(wordpack_path)
    before = emb.vectors
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        emb.normalize(l2=True, abtt=1)
    assert emb.vectors is before


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"abtt": 11}, "between 0 and 10"),
        ({"abtt": -1}, "between 0 and 10"),
        ({"l2": False}, "l2=False"),
    ],
)
def test_normalize_refuses_what_a_wordpack_cannot_do(wordpack_path, kwargs, match):
    emb = Embeddings.load(wordpack_path)
    before = emb.vectors
    with pytest.raises(ValueError, match=match):
        emb.normalize(**kwargs)
    assert emb.abtt == 1 and emb.vectors is before


def test_normalize_replaces_the_array_and_earlier_views_keep_the_old_m(wordpack_path, monkeypatch):
    monkeypatch.setattr(em, "_RAM_TOP_N", 18)
    small = Embeddings.load(wordpack_path)
    view = small.with_corpus(_tokens([["swim"]]))
    old = small.vectors
    small.normalize(abtt=0)
    at0, at1 = _direct_rows(wordpack_path, 0), _direct_rows(wordpack_path, 1)
    assert small.vectors is not old and view.vectors is old
    assert small._extras is None
    np.testing.assert_allclose(small["swim"], at0[KEYS.index("swim")], atol=1e-5)
    np.testing.assert_allclose(view["swim"], at1[KEYS.index("swim")], atol=1e-5)
    # A row the view reads on request also comes at the view's m.
    np.testing.assert_allclose(view["they"], at1[KEYS.index("they")], atol=1e-5)


@pytest.mark.parametrize("fmt", ["bin", "txt"])
def test_other_formats_warn_with_the_format_loaded(tmp_path, fmt):
    make_kv(["dog", "cat", "bird"], dim=4).save(str(tmp_path / "e"), fmt=fmt)
    with pytest.warns(UserWarning, match=rf"Loading a \.{fmt} embedding: .*token lists"):
        Embeddings.load(str(tmp_path / f"e.{fmt}"))


def test_wordpack_load_does_not_warn(wordpack_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Embeddings.load(wordpack_path)


def test_wordpack_text_fills_preprocessed_doc_consistently(wordpack_path):
    emb = Embeddings.load(wordpack_path)
    text = "I don't like the dog. Another cafe\u0301 with dogs!"  # decomposed é
    ssd = SSD(emb, [text], use_full_doc=True)
    d = ssd.pre_docs[0]
    t = emb._pack.tokenize(text)

    assert isinstance(d, PreprocessedDoc)
    assert d.raw == t.text == "I don't like the dog. Another café with dogs!"
    assert d.sent_char_spans == [(0, 21), (22, 45)]
    assert d.sents_surface == [d.raw[a:b] for a, b in d.sent_char_spans]
    assert d.sents_lemmas == [["do", "n't", "like", "dog"], ["another", "café", "with", "dogs"]]
    assert d.sents_kept_idx == [[1, 2, 3, 5], [0, 1, 2, 3]]
    assert d.doc_lemmas == [k for s in d.sents_lemmas for k in s]
    assert d.token_to_sent == [si for si, s in enumerate(d.sents_lemmas) for _ in s]
    for si, idx in enumerate(d.sents_kept_idx):
        sent_keys = [k for k, s in zip(t.keys, t.sent) if s == si]
        assert [sent_keys[j] for j in idx] == d.sents_lemmas[si]
    assert ssd.docs == [d.doc_lemmas]
    assert ssd.tokenizer == "wordpack:en-fixture-1"


def test_wordpack_text_drops_the_packs_stopwords(wordpack_path):
    emb = Embeddings.load(wordpack_path)
    assert SSD(emb, ["I like the dog."], use_full_doc=True).docs == [["like", "dog"]]


def test_wordpack_text_empty_none_and_nan_inputs(wordpack_path):
    emb = Embeddings.load(wordpack_path)
    flat = SSD(emb, ["", None, float("nan"), "   "], use_full_doc=True)
    assert flat.docs == [[], [], [], []]
    assert flat.pre_docs[0].sents_surface == []


WORDS = ["dog", "dogs", "cat", "blue", "green", "café", "never", "people", "like", "swim"]
# Unseen in training: happens, in, are, with, covid; "cats" is not in the pack at all.
NEW_TEXTS = ["People happens in the café.", "They are in green!", "Covid happens with cats."]


def _texts(n: int, seed: int) -> tuple[list[str], np.ndarray]:
    """n two-sentence texts over WORDS; y counts dog/dogs/people, plus noise."""
    rng = np.random.default_rng(seed)
    texts, y = [], []
    for _ in range(n):
        ws = [str(w) for w in rng.choice(WORDS, size=6)]
        texts.append("The " + " ".join(ws) + ". They swim!")
        y.append(sum(w in ("dog", "dogs", "people") for w in ws) + rng.normal(0, 0.3))
    return texts, np.asarray(y)


def test_prefix_plus_on_request_matches_a_full_load(wordpack_path, monkeypatch):
    texts, y = _texts(40, seed=0)
    full = Embeddings.load(wordpack_path)
    full_res = SSD(full, texts, use_full_doc=True).fit_regression(y, fixed_k=2)

    monkeypatch.setattr(em, "_RAM_TOP_N", 18)
    small = Embeddings.load(wordpack_path)
    assert small._partial
    small_res = SSD(small, texts, use_full_doc=True).fit_regression(y, fixed_k=2)

    np.testing.assert_allclose(small_res.beta, full_res.beta, rtol=1e-5, atol=1e-6)
    full_pred = full_res.predict(NEW_TEXTS)
    assert np.isfinite(full_pred).all()
    np.testing.assert_allclose(small_res.predict(NEW_TEXTS), full_pred, rtol=1e-5, atol=1e-6)


def test_normalize_after_a_fit_leaves_the_fit_unchanged(wordpack_path):
    texts, y = _texts(40, seed=1)
    emb = Embeddings.load(wordpack_path)
    res = SSD(emb, texts, use_full_doc=True).fit_regression(y, fixed_k=2)
    before = res.predict(NEW_TEXTS)
    emb.normalize(abtt=3)
    np.testing.assert_array_equal(res.predict(NEW_TEXTS), before)


def test_a_single_string_is_refused_not_split_into_characters(wordpack_path):
    texts, y = _texts(40, seed=1)
    emb = Embeddings.load(wordpack_path)
    res = SSD(emb, texts, use_full_doc=True).fit_regression(y, fixed_k=2)
    with pytest.raises(TypeError, match="single string"):
        SSD(emb, "dog swim")
    with pytest.raises(TypeError, match="single string"):
        res.predict("dog swim")


def test_predict_strings_on_a_token_list_fit_asks_for_token_lists(tiny_kv, sample_docs, sample_y, lexicon):
    res = SSD(tiny_kv, sample_docs, lexicon).fit_pls(sample_y, k=1, n_splits=5)
    with pytest.raises(TypeError, match="token lists"):
        res.predict(["piekny dom"])


def test_predict_token_lists_on_a_wordpack_fit_asks_for_strings(wordpack_path):
    texts, y = _texts(40, seed=2)
    emb = Embeddings.load(wordpack_path)
    res = SSD(emb, texts, use_full_doc=True).fit_regression(y, fixed_k=2)
    with pytest.raises(TypeError, match="strings"):
        res.predict([["dog", "swim"]])


def test_attach_refuses_a_wordpack_other_than_the_one_tokenized_for(wordpack_path, parity_emb):
    texts, y = _texts(40, seed=2)
    res = SSD(Embeddings.load(wordpack_path), texts, use_full_doc=True).fit_regression(y, fixed_k=2)
    assert parity_emb.abtt == res.embeddings.abtt == 1
    with pytest.raises(ValueError, match=r"wordpack:en-fixture-1"):
        res.attach(embeddings=parity_emb)


def test_attach_refuses_embeddings_that_are_not_a_wordpack(wordpack_path):
    texts, y = _texts(40, seed=2)
    res = SSD(Embeddings.load(wordpack_path), texts, use_full_doc=True).fit_regression(y, fixed_k=2)
    other = make_kv(WORDS + ["they"], dim=16)
    other.abtt = 1
    with pytest.raises(ValueError, match="not a wordpack"):
        res.attach(embeddings=other)


def test_a_token_list_text_is_not_checked(wordpack_path):
    emb = Embeddings.load(wordpack_path)
    texts, y = _texts(40, seed=3)
    docs = SSD(emb, texts, use_full_doc=True).docs
    res = SSD(emb, docs, use_full_doc=True).fit_regression(y, fixed_k=2)
    assert np.isfinite(res.predict([["dog", "swim"]])).all()


def test_snippets_run_on_a_wordpack_text(wordpack_path, monkeypatch):
    monkeypatch.setattr(em, "_RAM_TOP_N", 18)
    emb = Embeddings.load(wordpack_path)
    texts, y = _texts(40, seed=4)
    ssd = SSD(emb, texts, use_full_doc=True)
    res = ssd.fit_regression(y, fixed_k=2)
    rows = list(res.snippets)
    assert rows
    for sn in rows:
        assert sn.text_window in " ".join(ssd.pre_docs[sn.doc_id].sents_surface)


def _plain_report(res) -> str:
    return res.report(clusters=False, top_words=False, extreme_docs=False, misdiagnosed=False).to_text()


def test_report_names_the_wordpack(wordpack_path):
    emb = Embeddings.load(wordpack_path)
    texts, y = _texts(40, seed=5)
    res = SSD(emb, texts, use_full_doc=True).fit_regression(y, fixed_k=2)
    assert "en-fixture-1" in _plain_report(res)


def test_deepcopy_of_a_wordpack_result_keeps_its_embeddings(wordpack_path):
    texts, y = _texts(40, seed=1)
    emb = Embeddings.load(wordpack_path)
    res = SSD(emb, texts, use_full_doc=True).fit_regression(y, fixed_k=2)
    dup = copy.deepcopy(res)
    np.testing.assert_array_equal(dup.predict(NEW_TEXTS), res.predict(NEW_TEXTS))


def test_pickled_wordpack_result_takes_the_pack_back_by_path(wordpack_path):
    texts, y = _texts(40, seed=1)
    emb = Embeddings.load(wordpack_path).normalize(abtt=3)
    res = SSD(emb, texts, use_full_doc=True).fit_regression(y, fixed_k=2)
    back = pickle.loads(pickle.dumps(res))
    assert back.embeddings is None
    with pytest.raises(ValueError, match="abtt=3"):
        back.attach(embeddings=Embeddings.load(wordpack_path))
    back.attach(embeddings=wordpack_path)
    np.testing.assert_allclose(back.predict(NEW_TEXTS), res.predict(NEW_TEXTS), rtol=1e-6)


def test_attach_on_a_pickled_group_result_reaches_every_pair(wordpack_path):
    emb = Embeddings.load(wordpack_path)
    texts, y = _texts(40, seed=6)
    groups = np.where(y > np.median(y), "high", "low")
    gr = SSD(emb, texts, use_full_doc=True).fit_groups(groups, n_perm=49)
    back = pickle.loads(pickle.dumps(gr)).attach(embeddings=wordpack_path)
    for pair in gr.keys():
        assert list(back[pair].words) == list(gr[pair].words)


def test_report_names_the_file_for_other_formats(tmp_path, tiny_kv, sample_docs, sample_y, lexicon):
    tiny_kv.save(str(tmp_path / "tiny"))
    emb = Embeddings.load(str(tmp_path / "tiny.bin"))
    res = SSD(emb, sample_docs, lexicon).fit_regression(sample_y, fixed_k=2)
    assert "tiny.bin" in _plain_report(res)


def test_group_and_multipls_reports_name_the_wordpack(wordpack_path):
    emb = Embeddings.load(wordpack_path)
    texts, y = _texts(40, seed=6)
    ssd = SSD(emb, texts, use_full_doc=True)
    groups = np.where(y > np.median(y), "high", "low")
    gr = ssd.fit_groups(groups, n_perm=49)
    assert "en-fixture-1" in gr.report(clusters=False, top_words=False).to_text()
    y2 = np.column_stack([y, np.random.default_rng(0).normal(size=len(y))])
    mp = ssd.fit_multipls(y2, ssd_gradients=1, n_perm=19, n_splits=5)
    assert "en-fixture-1" in mp.report(top_words=False).to_text()


def test_pickle_round_trip_restores_snippets(parity_emb, parity_df, parity_lexicon):
    r = SSD(parity_emb, parity_df["text"], parity_lexicon).fit_pls("y", data=parity_df, k=1, n_splits=5)
    s = pickle.loads(pickle.dumps(r))
    assert s.embeddings is None and s.text.embeddings is None
    s.attach(embeddings=os.path.join(PARITY_DIR, "parity.wordpack"))
    assert len(s.words) and len(list(s.snippets(top_per_side=3)))
