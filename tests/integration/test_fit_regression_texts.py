"""ssdiff.fit_regression: several texts in one regression."""

from __future__ import annotations

import pytest

import ssdiff
from ssdiff.embeddings import Embeddings
from ssdiff.results import PCAOLSResult
from ssdiff.ssd import SSD
from tests.conftest import TEST_PACK


def test_two_texts(two, pack_df):
    essay, diary = two
    res = ssdiff.fit_regression("y ~ cov + ssd(essay) + group*ssd(diary)", pack_df,
                                essay=essay, diary=diary, fixed_k=3)
    assert res.keys() == ["essay", "diary"]
    assert type(res["essay"]) is PCAOLSResult and res["essay"].text is essay
    assert res["diary"].keys()[0] == "ssd(diary)"
    assert [r.label for r in res.test][:3] == [None, "essay", "diary"]
    assert any(n.startswith("_ssd_diary_pca_") for n in res.model.names)


def test_unused_or_missing_text_raises(two, pack_df):
    essay, diary = two
    with pytest.raises(ValueError, match="diary"):
        ssdiff.fit_regression("y ~ ssd(essay)", pack_df, essay=essay, diary=diary)
    with pytest.raises(ValueError, match="diary"):
        ssdiff.fit_regression("y ~ ssd(essay) + ssd(diary)", pack_df, essay=essay)


@pytest.fixture
def default_pack(monkeypatch, pack_emb):
    """Embeddings.load() returns the test pack offline; the list records each call's arguments."""
    calls = []
    monkeypatch.setattr(Embeddings, "load", classmethod(lambda cls, *a, **k: calls.append(a) or pack_emb))
    return calls


def test_texts_built_from_columns_share_one_load(default_pack, pack_df):
    df = pack_df.assign(diary=pack_df["text"].to_numpy()[::-1])
    res = ssdiff.fit_regression("y ~ ssd(text) + ssd(diary)", df, fixed_k=3)
    assert default_pack == [()]
    assert res.keys() == ["text", "diary"]
    built = res["text"].text
    assert built.name == "text" and built.lexicon == set() and built.use_full_doc


def test_a_keyword_wins_over_a_column(default_pack, two, pack_df):
    res = ssdiff.fit_regression("y ~ ssd(text)", pack_df, text=two[0], fixed_k=3)
    assert res.text is two[0] and default_pack == []


def test_a_text_neither_passed_nor_a_column_raises(default_pack, pack_df):
    with pytest.raises(ValueError, match=r"neither columns of data nor passed.*essay=SSD"):
        ssdiff.fit_regression("y ~ ssd(essay)", pack_df)
    assert default_pack == []


def test_no_text_term_raises(two, pack_df):
    with pytest.raises(ValueError, match="ssd"):
        ssdiff.fit_regression("y ~ cov", pack_df, essay=two[0], diary=two[1])


def test_text_named_like_a_column(two, pack_df):
    res = ssdiff.fit_regression("y ~ cov + ssd(cov)", pack_df, cov=two[0], fixed_k=3)
    assert "cov" in res.model.names and type(res) is PCAOLSResult


def test_rows_need_a_vector_in_every_text(pack_emb, pack_df, pack_lexicon, two):
    essay = two[0]
    short = SSD([[] if i < 7 else d for i, d in enumerate(essay.docs)], embeddings=pack_emb, lexicon=pack_lexicon)
    res = ssdiff.fit_regression("y ~ ssd(essay) + ssd(short)", pack_df, essay=essay, short=short, fixed_k=3)
    has_e, has_s = essay._vectors()[1], short._vectors()[1]
    assert res["short"].n_no_vector == int((~has_s).sum()) >= 7
    assert res.stats.n_kept == int((has_e & has_s).sum())


def test_container_views_are_keyed_by_text(two, pack_df):
    essay, diary = two
    res = ssdiff.fit_regression("y ~ ssd(essay) + group*ssd(diary)", pack_df,
                                essay=essay, diary=diary, fixed_k=3)
    assert res.text is None and list(res.sweep) == ["essay", "diary"]
    assert [r.label for r in res.test] == [None, "essay", "diary", "ssd(diary)", "group:ssd(diary)"]
    assert len(res["diary"].test) == 3
    assert res.n_no_vector == {"essay": 0, "diary": 0}
    old = res["essay"].text
    assert res.attach(texts={"essay": list(pack_df["text"])}) is res
    assert res["essay"].text is not old
    with pytest.raises(TypeError, match="dict"):
        res.attach(texts=list(pack_df["text"]))


def test_two_texts_in_one_term_raises(two, pack_df):
    essay, diary = two
    with pytest.raises(ValueError, match="two texts"):
        ssdiff.fit_regression("y ~ ssd(essay)*ssd(diary)", pack_df,
                              essay=essay, diary=diary, fixed_k=3)


def test_outcome_checks(two, pack_df):
    essay, diary = two
    with pytest.raises(ValueError, match="numeric outcome"):
        ssdiff.fit_regression("group ~ ssd(essay) + ssd(diary)", pack_df,
                              essay=essay, diary=diary, fixed_k=3)
    with pytest.raises(ValueError, match="outcome column on its left side"):
        ssdiff.fit_regression("~ ssd(essay) + ssd(diary)", pack_df,
                              essay=essay, diary=diary, fixed_k=3)


def test_summary_labels_all_texts(two, pack_df):
    essay, diary = two
    res = ssdiff.fit_regression("y ~ ssd(essay) + ssd(diary)", pack_df,
                                essay=essay, diary=diary, fixed_k=3)
    block = res.summary().split("Joint tests:")[1].split("---")[0].strip().split("\n")[1:]
    assert [ln.split()[0] for ln in block] == ["all", "essay", "diary"]


def test_attach_texts_on_an_unpickled_one_text_regression(ssd_instance, pack_emb, pack_df):
    import pickle

    res = pickle.loads(pickle.dumps(ssd_instance.fit_regression("y ~ group*ssd()", data=pack_df, fixed_k=3)))
    res.attach(texts=list(pack_df["text"]), embeddings=pack_emb)
    assert len(res["ssd(text)"].snippets.pos)


def test_attach_texts_on_an_unpickled_two_text_regression(two, pack_emb, pack_df):
    import pickle

    essay, diary = two
    res = ssdiff.fit_regression("y ~ ssd(essay) + group*ssd(diary)", pack_df,
                                essay=essay, diary=diary, fixed_k=3)
    res = pickle.loads(pickle.dumps(res))
    res.attach(texts={"essay": list(pack_df["text"]), "diary": essay.docs[::-1]},
               embeddings={"essay": pack_emb, "diary": pack_emb})
    assert len(res["essay"].words.pos)


def test_page_keeps_the_moderated_term_key_of_a_one_term_text(two, pack_df):
    essay, diary = two
    res = ssdiff.fit_regression("y ~ ssd(essay) + cov:ssd(diary)", pack_df,
                                essay=essay, diary=diary, fixed_k=3)
    assert [u["key"] for u in res.page_data()["units"]] == ["ssd(essay)", "cov:ssd(diary)"]


def test_attach_leaves_the_callers_ssd_alone(pack_emb, pack_df, pack_lexicon):
    essay = SSD(pack_df["text"], embeddings=pack_emb, lexicon=pack_lexicon)
    before = essay.embeddings
    res = essay.fit_regression(pack_df["y"], fixed_k=3)
    res.attach(embeddings=ssdiff.Embeddings.load(TEST_PACK))
    assert essay.embeddings is before


def test_lone_moderated_term_predicts_from_the_whole_fit(ssd_instance, pack_df):
    import numpy as np

    res = ssd_instance.fit_regression("y ~ cov:ssd()", data=pack_df, fixed_k=3)
    y_hat = np.array([d.y_hat for d in res.docs])
    np.testing.assert_allclose(y_hat, np.asarray(res.model.fitted))


def test_text_named_y_is_refused(ssd_instance, pack_df):
    with pytest.raises(ValueError, match="reserved"):
        ssdiff.fit_regression("y ~ ssd(y)", pack_df, y=ssd_instance, fixed_k=3)


def test_fit_pls_refuses_a_text_term(ssd_instance, pack_df):
    with pytest.raises(ValueError, match="covariates only"):
        ssd_instance.fit_pls("y ~ cov + ssd()", data=pack_df)


def test_doc_detail_of_a_token_list_text_says_so(two, pack_df):
    diary = two[1]
    res = diary.fit_regression(pack_df["y"], fixed_k=3)
    assert "token lists" in res.docs.id(0).to_text()


def test_report_has_fit_info_per_text(two, pack_df):
    essay, diary = two
    res = ssdiff.fit_regression("y ~ cov + ssd(essay) + group*ssd(diary)", pack_df,
                                essay=essay, diary=diary, fixed_k=3)
    titles = [s.title for s in res.report().sections]
    assert [t for t in titles if t.startswith("Fit info")] == ["Fit info — essay", "Fit info — diary"]
