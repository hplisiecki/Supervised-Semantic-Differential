"""ssdiff.fit_regression: several texts in one regression."""

from __future__ import annotations

import os

import pytest

import ssdiff
from ssdiff.results import PCAOLSResult
from ssdiff.ssd import SSD
from tests.conftest import PARITY_DIR


def test_two_texts(two, parity_df):
    essay, diary = two
    res = ssdiff.fit_regression("y ~ cov + ssd(essay) + group*ssd(diary)", parity_df,
                                essay=essay, diary=diary, fixed_k=3)
    assert res.keys() == ["essay", "diary"]
    assert type(res["essay"]) is PCAOLSResult and res["essay"].text is essay
    assert res["diary"].keys()[0] == "ssd(diary)"
    assert [r.label for r in res.test][:3] == [None, "essay", "diary"]
    assert any(n.startswith("_ssd_diary_pca_") for n in res.model.names)


def test_unused_or_missing_text_raises(two, parity_df):
    essay, diary = two
    with pytest.raises(ValueError, match="diary"):
        ssdiff.fit_regression("y ~ ssd(essay)", parity_df, essay=essay, diary=diary)
    with pytest.raises(ValueError, match="diary"):
        ssdiff.fit_regression("y ~ ssd(essay) + ssd(diary)", parity_df, essay=essay)


def test_no_text_term_raises(two, parity_df):
    with pytest.raises(ValueError, match="ssd"):
        ssdiff.fit_regression("y ~ cov", parity_df, essay=two[0], diary=two[1])


def test_text_named_like_a_column(two, parity_df):
    res = ssdiff.fit_regression("y ~ cov + ssd(cov)", parity_df, cov=two[0], fixed_k=3)
    assert "cov" in res.model.names and type(res) is PCAOLSResult


def test_rows_need_a_vector_in_every_text(parity_emb, parity_df, parity_lexicon, two):
    essay = two[0]
    short = SSD(parity_emb, [[] if i < 7 else d for i, d in enumerate(essay.docs)], parity_lexicon)
    res = ssdiff.fit_regression("y ~ ssd(essay) + ssd(short)", parity_df, essay=essay, short=short, fixed_k=3)
    has_e, has_s = essay._vectors()[1], short._vectors()[1]
    assert res["short"].n_no_vector == int((~has_s).sum()) >= 7
    assert res.stats.n_kept == int((has_e & has_s).sum())


def test_container_views_are_keyed_by_text(two, parity_df):
    essay, diary = two
    res = ssdiff.fit_regression("y ~ ssd(essay) + group*ssd(diary)", parity_df,
                                essay=essay, diary=diary, fixed_k=3)
    assert res.text is None and list(res.sweep) == ["essay", "diary"]
    assert [r.label for r in res.test] == [None, "essay", "diary", "ssd(diary)", "group:ssd(diary)"]
    assert len(res["diary"].test) == 3
    assert res.n_no_vector == {"essay": 0, "diary": 0}
    old = res["essay"].text
    assert res.attach(texts={"essay": list(parity_df["text"])}) is res
    assert res["essay"].text is not old
    with pytest.raises(TypeError, match="dict"):
        res.attach(texts=list(parity_df["text"]))


def test_two_texts_in_one_term_raises(two, parity_df):
    essay, diary = two
    with pytest.raises(ValueError, match="two texts"):
        ssdiff.fit_regression("y ~ ssd(essay)*ssd(diary)", parity_df,
                              essay=essay, diary=diary, fixed_k=3)


def test_outcome_checks(two, parity_df):
    essay, diary = two
    with pytest.raises(ValueError, match="numeric outcome"):
        ssdiff.fit_regression("group ~ ssd(essay) + ssd(diary)", parity_df,
                              essay=essay, diary=diary, fixed_k=3)
    with pytest.raises(ValueError, match="outcome column on its left side"):
        ssdiff.fit_regression("~ ssd(essay) + ssd(diary)", parity_df,
                              essay=essay, diary=diary, fixed_k=3)


def test_summary_labels_all_texts(two, parity_df, capsys):
    essay, diary = two
    res = ssdiff.fit_regression("y ~ ssd(essay) + ssd(diary)", parity_df,
                                essay=essay, diary=diary, fixed_k=3)
    assert "all texts" in res.summary()


def test_attach_texts_on_an_unpickled_one_text_regression(ssd_instance, parity_emb, parity_df):
    import pickle

    res = pickle.loads(pickle.dumps(ssd_instance.fit_regression("y ~ group*ssd()", data=parity_df, fixed_k=3)))
    res.attach(texts=list(parity_df["text"]), embeddings=parity_emb)
    assert len(res["ssd(text)"].snippets.pos)


def test_attach_texts_on_an_unpickled_two_text_regression(two, parity_emb, parity_df):
    import pickle

    essay, diary = two
    res = ssdiff.fit_regression("y ~ ssd(essay) + group*ssd(diary)", parity_df,
                                essay=essay, diary=diary, fixed_k=3)
    res = pickle.loads(pickle.dumps(res))
    res.attach(texts={"essay": list(parity_df["text"]), "diary": essay.docs[::-1]},
               embeddings={"essay": parity_emb, "diary": parity_emb})
    assert len(res["essay"].words.pos)


def test_predict_strings_on_a_wordpack_regression(ssd_instance, parity_df):
    res = ssd_instance.fit_regression(parity_df["y"], fixed_k=3)
    pred = res.predict(list(parity_df["text"][:5]))
    assert pred.shape == (5,)


def test_page_keeps_the_moderated_term_key_of_a_one_term_text(two, parity_df):
    essay, diary = two
    res = ssdiff.fit_regression("y ~ ssd(essay) + cov:ssd(diary)", parity_df,
                                essay=essay, diary=diary, fixed_k=3)
    assert [u["key"] for u in res.page_data()["units"]] == ["ssd(essay)", "cov:ssd(diary)"]


def test_attach_leaves_the_callers_ssd_alone(parity_emb, parity_df, parity_lexicon):
    essay = SSD(parity_emb, parity_df["text"], parity_lexicon)
    before = essay.embeddings
    res = essay.fit_regression(parity_df["y"], fixed_k=3)
    res.attach(embeddings=ssdiff.Embeddings.load(os.path.join(PARITY_DIR, "parity.wordpack")))
    assert essay.embeddings is before


def test_lone_moderated_term_predicts_from_the_whole_fit(ssd_instance, parity_df):
    import numpy as np

    res = ssd_instance.fit_regression("y ~ cov:ssd()", data=parity_df, fixed_k=3)
    y_hat = np.array([d.y_hat for d in res.docs])
    np.testing.assert_allclose(y_hat, np.asarray(res.model.fitted))


def test_text_named_y_is_refused(ssd_instance, parity_df):
    with pytest.raises(ValueError, match="reserved"):
        ssdiff.fit_regression("y ~ ssd(y)", parity_df, y=ssd_instance, fixed_k=3)


def test_fit_pls_refuses_a_text_term(ssd_instance, parity_df):
    with pytest.raises(ValueError, match="covariates only"):
        ssd_instance.fit_pls("y ~ cov + ssd()", data=parity_df)


def test_doc_detail_of_a_token_list_text_says_so(two, parity_df):
    diary = two[1]
    res = diary.fit_regression(parity_df["y"], fixed_k=3)
    assert "token lists" in res.docs.id(0).to_text()
