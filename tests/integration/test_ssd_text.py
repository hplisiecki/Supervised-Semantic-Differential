"""SSD holds one text: tokens and lazily built document vectors."""

import pickle

import numpy as np
import pandas as pd
import pytest

from ssdiff.ssd import SSD
from ssdiff.utils.vectors import compute_global_sif


def test_token_lists(tiny_kv, sample_docs, lexicon):
    t = SSD(tiny_kv, sample_docs, lexicon)
    assert t.docs == sample_docs and t.pre_docs is None and t.tokenizer is None
    assert len(t) == len(sample_docs) and t.name is None


def test_strings_need_a_wordpack(tiny_kv):
    with pytest.raises(TypeError, match="token lists"):
        SSD(tiny_kv, ["kraj dobry"])


def test_wordpack_strings_and_name(parity_emb, parity_df, parity_lexicon):
    t = SSD(parity_emb, parity_df["text"], parity_lexicon)
    assert t.name == "text" and t.pre_docs is not None
    assert t.tokenizer == f"wordpack:{parity_emb.name}-{parity_emb.version}"


def test_a_list_among_strings_is_refused(parity_emb):
    with pytest.raises(TypeError, match="join each participant's posts"):
        SSD(parity_emb, ["people work.", ["people", "work"]])


def test_with_lexicon_shares_tokens(tiny_kv, sample_docs, lexicon):
    base = SSD(tiny_kv, sample_docs)
    t = base.with_lexicon(lexicon)
    assert t.docs is base.docs and t.lexicon == set(lexicon)
    np.testing.assert_array_equal(
        t._vectors()[0], SSD(tiny_kv, sample_docs, lexicon)._vectors()[0]
    )


def test_no_lexicon_raises_at_vectors(tiny_kv, sample_docs):
    with pytest.raises(ValueError, match="with_lexicon"):
        SSD(tiny_kv, sample_docs).n_kept


def test_sif_counts_use_all_documents(tiny_kv, sample_docs, lexicon):
    docs = sample_docs + [["piekny", "dom"]]   # no lexicon word: dropped from the fit
    counts = SSD(tiny_kv, docs, lexicon)._vectors()[2]
    assert counts == compute_global_sif(docs)
    assert counts != compute_global_sif(sample_docs)


def test_empty_cells_get_no_vector(parity_emb, parity_lexicon):
    texts = pd.Series(["", None, float("nan"), "people work every day."], name="t")
    t = SSD(parity_emb, texts, parity_lexicon)
    assert t._vectors()[1].tolist() == [False, False, False, True]


def test_pickle_drops_embeddings_and_vectors(tiny_kv, sample_docs, lexicon):
    t = SSD(tiny_kv, sample_docs, lexicon)
    t._vectors()
    s = pickle.loads(pickle.dumps(t))
    assert s.embeddings is None and s._cache is None and s.docs == sample_docs


@pytest.fixture
def text(tiny_kv, large_docs, lexicon):
    return SSD(tiny_kv, large_docs, lexicon)


def test_outcome_forms_agree(text, large_y):
    df = pd.DataFrame({"score": large_y})
    a = text.fit_regression(large_y, fixed_k=3)
    b = text.fit_regression("score", data=df, fixed_k=3)
    c = text.fit_regression("score ~ 1", data=df, fixed_k=3)
    np.testing.assert_allclose(a.beta, b.beta)
    np.testing.assert_allclose(a.beta, c.beta)


def test_series_index_is_ignored(text, large_y):
    s = pd.Series(large_y, index=np.arange(len(large_y))[::-1] + 100)
    np.testing.assert_allclose(
        text.fit_regression(s, fixed_k=3).beta,
        text.fit_regression(large_y, fixed_k=3).beta,
    )


def test_missing_outcome_drops_rows_not_sif(text, large_y):
    y = large_y.copy()
    y[:5] = np.nan
    r = text.fit_regression(y, fixed_k=3)
    assert r.stats.n_kept == text.n_kept - 5 and r._sif_counts == text._vectors()[2]


def test_formula_left_side_must_exist(text, large_y):
    with pytest.raises(ValueError, match="nope"):
        text.fit_regression("nope ~ 1", data=pd.DataFrame({"score": large_y}), fixed_k=3)


def test_formula_refused_where_not_taken(text, large_y):
    df = pd.DataFrame({"a": large_y, "b": large_y})
    with pytest.raises(ValueError, match="takes no formula"):
        text.fit_groups("a ~ 1", data=df)


def test_multipls_columns_and_groups_column(text, large_y, large_groups_2):
    df = pd.DataFrame({
        "a": large_y,
        "b": np.random.default_rng(1).normal(size=len(large_y)),
        "g": large_groups_2,
    })
    r = text.fit_multipls(["a", "b"], data=df, n_perm=19, n_splits=5)
    assert r.outcome_names == ["a", "b"]
    assert r.y.shape == (len(df), 2)
    assert text.fit_groups("g", data=df, n_perm=50).G == 2


def test_multipls_columns_from_a_dict(text, large_y):
    data = {"a": large_y, "b": np.random.default_rng(1).normal(size=len(large_y))}
    r = text.fit_multipls(["a", "b"], data=data, n_perm=19, n_splits=5)
    assert r.outcome_names == ["a", "b"]
    assert r.y.shape == (len(large_y), 2)


def test_n_no_vector(tiny_kv, large_docs, lexicon, large_y):
    docs = [["dom", "las"]] * 3 + large_docs[3:]   # no lexicon word in the first three
    r = SSD(tiny_kv, docs, lexicon).fit_regression(large_y, fixed_k=3)
    assert r.n_no_vector == 3
