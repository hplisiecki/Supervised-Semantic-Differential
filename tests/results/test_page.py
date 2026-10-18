"""Results page: page_data() and save_page() on all four result kinds."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from ssdiff.corpus import Corpus
from ssdiff.results.continuous_result import SnippetsView
from ssdiff.results.page import REPORT_FORMAT, page_data, save_page
from tests.conftest import FakeNlp

SNIPPET_COLS = [c for c in SnippetsView._columns if c != "text_lemmas"]


def _cols(view):
    return list(view._columns)


@pytest.fixture(scope="module")
def text_ssd_groups(tiny_kv, large_docs_3x20, large_groups_3x20, lexicon):
    """Three-group SSD on a corpus with raw text, so snippets exist."""
    from ssdiff.ssd import SSD
    texts = [" ".join(d) for d in large_docs_3x20]
    corpus = Corpus(texts, nlp=FakeNlp(), lang="pl", stopwords=[])
    return SSD(tiny_kv, corpus, large_groups_3x20, lexicon)


@pytest.fixture(scope="module")
def groups_text_result(text_ssd_groups):
    return text_ssd_groups.fit_groups(n_perm=50, random_state=42)


@pytest.fixture(scope="module")
def multipls_result(tiny_kv, large_docs, lexicon):
    from ssdiff.ssd import SSD
    texts = [" ".join(d) for d in large_docs]
    corpus = Corpus(texts, nlp=FakeNlp(), lang="pl", stopwords=[])
    y = np.random.default_rng(3).normal(size=(len(texts), 3))
    ssd = SSD(tiny_kv, corpus, y, lexicon)
    return ssd.fit_multipls(ssd_gradients=2, n_perm=19, n_splits=5, random_state=0)


def test_fit_args_recorded(pls_result, pcaols_result, group_result_2g, multipls_result):
    assert pls_result.fit_args == {"formula": None, "k": 2, "k_max": 5, "ssd_gradients": 1,
                                   "rotate": "words", "n_splits": 20, "random_state": 42,
                                   "test": True}
    assert pcaols_result.fit_args["fixed_k"] == 3
    assert group_result_2g.fit_args == {"median_split": False, "n_perm": 50,
                                        "correction": "holm", "random_state": 42}
    assert multipls_result.fit_args["ssd_gradients"] == 2
    assert "verbose" not in pls_result.fit_args


def test_doc_ids_point_at_corpus_rows(groups_text_result, multipls_result):
    assert len(groups_text_result.doc_ids) == len(groups_text_result._x)
    assert len(multipls_result.doc_ids) == len(multipls_result.x)
    assert groups_text_result.doc_ids.dtype.kind == "i"


def test_pls_page_data(pls_result):
    d = page_data(pls_result)
    json.dumps(d, allow_nan=False)
    assert d["report_format"] == REPORT_FORMAT
    assert d["kind"] == "pls"
    assert set(d) >= {"report_format", "kind", "title", "created", "software",
                      "embeddings", "settings", "tables", "units"}
    assert d["software"]["package"] == "ssdiff" and d["software"]["language"] == "python"
    assert set(d["tables"]) == {"stats", "fit_info", "test"}
    assert d["tables"]["stats"]["columns"] == _cols(pls_result.stats)
    assert d["tables"]["fit_info"]["columns"] == _cols(pls_result.fit_info)
    assert d["tables"]["test"]["columns"] == _cols(pls_result.test)
    assert d["settings"]["fit"] == pls_result.fit_args
    assert d["settings"]["lexicon"] == sorted(pls_result.lexicon)
    [unit] = d["units"]
    assert (unit["key"], unit["label"], unit["pos_label"], unit["neg_label"]) == ("main", "", "+β", "−β")
    t = unit["tables"]
    assert set(t) == {"words", "clusters", "cluster_words", "snippets", "cluster_snippets", "docs"}
    assert t["words"]["columns"] == _cols(pls_result.words)
    assert t["clusters"]["columns"] == _cols(pls_result.clusters)
    assert t["cluster_words"]["columns"] == _cols(pls_result.clusters.words)
    assert t["snippets"]["columns"] == SNIPPET_COLS
    assert t["cluster_snippets"]["columns"] == SNIPPET_COLS
    assert t["docs"]["columns"] == _cols(pls_result.docs) + ["text"]
    assert len(t["docs"]["rows"]) == len(list(pls_result.docs))


def test_pls_gradients_page_data(ssd_instance):
    res = ssd_instance.fit_pls(k=2, ssd_gradients=2, n_splits=20, random_state=42)
    d = page_data(res)
    json.dumps(d, allow_nan=False)
    assert set(d["tables"]) == {"stats", "fit_info", "test", "pls_info"}
    assert d["tables"]["pls_info"]["columns"] == _cols(res.pls_info)
    assert [(u["key"], u["label"]) for u in d["units"]] == [("main", "β"), ("dim-1", "Dim 1"), ("dim-2", "Dim 2")]
    t = d["units"][1]["tables"]
    assert set(t) == {"stats", "words", "clusters", "cluster_words", "snippets", "cluster_snippets", "docs"}
    assert t["stats"]["columns"] == ["cov_share", "score_corr"]
    assert t["docs"]["columns"] == ["doc_id", "alignment_score", "text"]
    assert [r[0] for r in t["docs"]["rows"]] == [doc.doc_id for doc in res.docs]


@pytest.mark.parametrize("doc_texts, expect", [(0, "none"), (1, "some"), (None, "all")])
def test_doc_texts(pls_result, doc_texts, expect):
    rows = page_data(pls_result, doc_texts=doc_texts)["units"][0]["tables"]["docs"]["rows"]
    n_text = sum(r[-1] is not None for r in rows)
    if expect == "none":
        assert n_text == 0
    elif expect == "all":
        assert n_text == len(rows)
    else:
        # 1 highest + 1 lowest by alignment, 1 most over- + 1 most under-predicted
        assert 2 <= n_text <= 4


def test_covariates_table(tiny_kv, large_docs, large_y, lexicon):
    import pandas as pd

    from ssdiff.ssd import SSD
    data = pd.DataFrame({"age": np.random.default_rng(0).normal(size=len(large_y))})
    ssd = SSD(tiny_kv, Corpus(large_docs, pretokenized=True, lang="pl"), large_y, lexicon, data=data)
    res = ssd.fit_regression("~ age", fixed_k=3)
    t = page_data(res)["tables"]["covariates"]
    assert t["columns"] == _cols(res.covariates)
    assert [r[0] for r in t["rows"]] == ["(Intercept)", "age"]


def test_no_raw_text_leaves_out_snippets(pcaols_result_sweep):
    t = page_data(pcaols_result_sweep)["units"][0]["tables"]
    assert "snippets" not in t and "cluster_snippets" not in t
    assert all(r[-1] is None for r in t["docs"]["rows"])


def test_no_embeddings_leaves_out_word_tables(pls_result):
    r = copy.deepcopy(pls_result)
    r.embeddings = None
    r.clear_cache()
    t = page_data(r)["units"][0]["tables"]
    assert "words" not in t and "clusters" not in t and "cluster_words" not in t
    assert "docs" in t


def test_pcaols_page_data(pcaols_result, pcaols_result_sweep):
    d = page_data(pcaols_result)
    json.dumps(d, allow_nan=False)
    assert d["kind"] == "pca_ols"
    assert "sweep" not in d["tables"]  # fixed_k: no sweep was run
    assert d["tables"]["stats"]["columns"] == _cols(pcaols_result.stats)
    s = page_data(pcaols_result_sweep)
    assert s["tables"]["sweep"]["columns"] == _cols(pcaols_result_sweep.sweep)
    assert len(s["tables"]["sweep"]["rows"]) == len(pcaols_result_sweep.sweep)


def test_groups_page_data(groups_text_result):
    gr = groups_text_result
    d = page_data(gr)
    json.dumps(d, allow_nan=False)
    assert d["kind"] == "groups"
    assert set(d["tables"]) == {"stats", "test", "pairs"}
    assert d["tables"]["pairs"]["columns"] == _cols(gr.pairs)
    assert len(d["units"]) == len(gr.keys()) == 3
    u = d["units"][0]
    g1, g2 = gr.keys()[0]
    lab = lambda g: gr.group_labels.get(g, g)  # noqa: E731
    assert u["key"] == gr[(g1, g2)].contrast
    assert (u["label"], u["pos_label"], u["neg_label"]) == (f"{lab(g1)} vs {lab(g2)}", lab(g1), lab(g2))
    t = u["tables"]
    assert set(t) == {"words", "clusters", "cluster_words", "snippets", "cluster_snippets", "docs"}
    assert t["docs"]["columns"] == ["doc_id", "group", "alignment_score", "text"]
    assert len(t["docs"]["rows"]) == int(gr[(g1, g2)].groups_mask.sum())
    assert {r[1] for r in t["docs"]["rows"]} == {lab(g1), lab(g2)}


def test_multipls_page_data(multipls_result):
    mp = multipls_result
    d = page_data(mp)
    json.dumps(d, allow_nan=False)
    assert d["kind"] == "multipls"
    assert set(d["tables"]) == {"stats", "pls_info", "test"}
    assert d["tables"]["pls_info"]["columns"] == _cols(mp.pls_info)
    assert [u["label"] for u in d["units"]] == ["Dim 1", "Dim 2"]
    u = d["units"][0]
    assert (u["pos_label"], u["neg_label"]) == ("+", "−")
    t = u["tables"]
    assert t["stats"]["columns"] == ["cov_share", "score_corr"]
    assert t["outcomes"]["columns"] == ["outcome", "group", "loading"]
    assert t["docs"]["columns"] == ["doc_id", "alignment_score", "text"]
    assert len(t["docs"]["rows"]) == len(mp.doc_ids)


def test_old_results_without_new_attributes(pls_result, groups_text_result):
    r = copy.deepcopy(pls_result)
    del r.fit_args
    assert page_data(r)["settings"]["fit"] is None
    g = copy.deepcopy(groups_text_result)
    del g.doc_ids
    assert all("docs" not in u["tables"] for u in page_data(g)["units"])


def _read_data(path):
    html = path.read_text(encoding="utf-8")
    assert html.count('id="ssd-data"') == 1
    start = html.index('id="ssd-data">') + len('id="ssd-data">')
    end = html.index("</script>", start)
    return json.loads(html[start:end])


def test_save_page_round_trip(pls_result, tmp_path):
    p = tmp_path / "page.html"
    pls_result.save_page(p, title="Zażółć gęślą jaźń", dataset={"Plik": "dane.csv"})
    back = _read_data(p)
    expect = pls_result.page_data(title="Zażółć gęślą jaźń", dataset={"Plik": "dane.csv"})
    back.pop("created"), expect.pop("created")
    assert back == json.loads(json.dumps(expect))
    assert "cdn.jsdelivr.net" not in p.read_text(encoding="utf-8")


def test_save_page_survives_script_text(pls_result, tmp_path):
    p = tmp_path / "page.html"
    nasty = "a </script><script>alert(1)</script> <!--<script> b"
    save_page(pls_result, p, title=nasty)
    html = p.read_text(encoding="utf-8")
    assert "</script><script>alert" not in html
    assert _read_data(p)["title"] == nasty


def test_methods_exist_on_all_kinds(pls_result, pcaols_result, group_result_2g, multipls_result):
    for r in (pls_result, pcaols_result, group_result_2g, multipls_result):
        assert r.page_data()["report_format"] == 1


FIXTURES = Path(__file__).parent.parent / "fixtures" / "page"


def _shape(d):
    """Keys and table columns, not values: what R and the template rely on."""
    return {
        "top": sorted(d), "tables": {k: v["columns"] for k, v in d["tables"].items()},
        "units": [{k: v["columns"] for k, v in u["tables"].items()} for u in d["units"][:1]],
    }


@pytest.mark.parametrize("kind, fixture", [("pls", "pls_result"), ("pca_ols", "pcaols_result_sweep"),
                                           ("groups", "groups_text_result"), ("multipls", "multipls_result")])
def test_example_json_matches_current_shape(kind, fixture, request):
    saved = json.loads((FIXTURES / f"{kind}.json").read_text(encoding="utf-8"))
    now = page_data(request.getfixturevalue(fixture))
    assert _shape(saved)["top"] == _shape(now)["top"]
    assert _shape(saved)["tables"] == _shape(now)["tables"]
    # A pretokenized test fit has no snippet tables; compare the unit tables both sides have.
    [old_unit], [new_unit] = _shape(saved)["units"], _shape(now)["units"]
    assert {k: old_unit[k] for k in new_unit} == new_unit


def test_template_declares_utf8(pls_result, tmp_path):
    p = tmp_path / "page.html"
    save_page(pls_result, p)
    head = p.read_bytes()[:1024].lower()
    assert head.startswith(b"<!doctype html>")
    assert b'<meta charset="utf-8">' in head


def test_numpy_scalars_in_embeddings(pls_result, tmp_path):
    r = copy.deepcopy(pls_result)
    r.embeddings.abtt = np.int64(1)
    save_page(r, tmp_path / "page.html")
