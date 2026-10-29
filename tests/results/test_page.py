"""Results page: page_data() and save_page() on all four result kinds."""

import copy
import json

import numpy as np
import pytest

import ssdiff
from ssdiff.results.continuous_result import SnippetsView
from ssdiff.results.page import REPORT_FORMAT, page_data, save_page

SNIPPET_COLS = [c for c in SnippetsView._columns if c != "text_lemmas"]


@pytest.fixture(scope="module")
def groups_text_result(ssd_instance, pack_df):
    """Three-group fit on the test pack, so snippets exist."""
    return ssd_instance.fit_groups(pack_df["group"], n_perm=50, random_state=42)


@pytest.fixture(scope="module")
def multipls_result(ssd_instance, pack_df):
    return ssd_instance.fit_multipls(
        ["y", "y2"], data=pack_df, ssd_gradients=2, n_perm=19, n_splits=5, random_state=0
    )


def test_fit_args_recorded(pls_result, pcaols_result, group_result_2g, multipls_result):
    assert pls_result.fit_args == {"formula": None, "k": 2, "k_max": 5, "ssd_gradients": 1,
                                   "rotate": "words", "n_splits": 20, "random_state": 42,
                                   "test": True}
    assert pcaols_result.fit_args["fixed_k"] == 3
    assert group_result_2g.fit_args == {"median_split": False, "n_perm": 50,
                                        "correction": "holm", "random_state": 42}
    assert multipls_result.fit_args["ssd_gradients"] == 2
    assert "verbose" not in pls_result.fit_args


def test_doc_ids_point_at_text_rows(groups_text_result, multipls_result):
    assert len(groups_text_result.doc_ids) == len(groups_text_result._x)
    assert len(multipls_result.doc_ids) == len(multipls_result.x)
    assert groups_text_result.doc_ids.dtype.kind == "i"


def test_pls_page_data(pls_result):
    d = page_data(pls_result)
    json.dumps(d, allow_nan=False)
    assert d["report_format"] == REPORT_FORMAT
    assert d["kind"] == "pls"
    assert set(d) >= {"report_format", "kind", "title", "created", "software",
                      "texts", "settings", "tables", "units"}
    assert d["software"]["package"] == "ssdiff" and d["software"]["language"] == "python"
    assert set(d["tables"]) == {"stats", "fit_info", "test"}
    assert d["settings"]["fit"] == pls_result.fit_args
    assert d["texts"][0]["settings"]["lexicon"] == sorted(pls_result.lexicon)
    [unit] = d["units"]
    assert (unit["key"], unit["label"], unit["pos_label"], unit["neg_label"]) == ("main", "", "+β", "−β")
    t = unit["tables"]
    assert set(t) == {"words", "clusters", "cluster_words", "snippets", "cluster_snippets", "docs"}
    assert t["snippets"]["columns"] == SNIPPET_COLS
    assert t["cluster_snippets"]["columns"] == SNIPPET_COLS
    assert len(t["docs"]["rows"]) == len(list(pls_result.docs))


def test_pls_gradients_page_data(ssd_instance, pack_df):
    res = ssd_instance.fit_pls(pack_df["y"], k=2, ssd_gradients=2, n_splits=20, random_state=42)
    d = page_data(res)
    json.dumps(d, allow_nan=False)
    assert set(d["tables"]) == {"stats", "fit_info", "test", "pls_info"}
    assert d["tables"]["pls_info"]["columns"] == [
        "ssd_gradients", "rotate", "order", "signs", "kaiser_normalized", "sweeps", "V_converged", "random_state"]
    assert [(u["key"], u["label"]) for u in d["units"]] == [("main", "β"), ("dim-1", "Dim 1"), ("dim-2", "Dim 2")]
    t = d["units"][1]["tables"]
    assert set(t) == {"stats", "words", "clusters", "cluster_words", "snippets", "cluster_snippets", "docs"}
    assert t["stats"]["columns"] == ["cov_share", "score_corr"]
    assert t["docs"]["columns"] == ["doc_id", "alignment_score", "text"]
    assert [r[0] for r in t["docs"]["rows"]] == [doc.doc_id for doc in res.docs]


def test_regression_gradients_page_data(ssd_instance, pack_df):
    res = ssd_instance.fit_regression(pack_df["y"], fixed_k=4, ssd_gradients=2)
    d = page_data(res)
    json.dumps(d, allow_nan=False)
    assert d["kind"] == "pca_ols" and "pls_info" in d["tables"]
    assert [(u["key"], u["label"]) for u in d["units"]] == [("main", "β"), ("dim-1", "Dim 1"), ("dim-2", "Dim 2")]
    assert d["units"][1]["tables"]["stats"]["columns"] == ["cov_share", "score_corr"]


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


def test_covariates_table(ssd_instance, pack_df):
    res = ssd_instance.fit_regression("y ~ cov", data=pack_df, fixed_k=3)
    t = page_data(res)["tables"]["covariates"]
    assert t["columns"] == ["term", "estimate", "se", "statistic", "p_value", "ci_low", "ci_high"]
    assert [r[0] for r in t["rows"]] == ["(Intercept)", "cov"]


def test_no_raw_text_leaves_out_snippets(pcaols_result_sweep):
    t = page_data(pcaols_result_sweep)["units"][0]["tables"]
    assert "snippets" not in t and "cluster_snippets" not in t


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
    assert d["texts"][0]["sweep"] is None  # fixed_k: no sweep was run
    s = page_data(pcaols_result_sweep)
    sweep = s["texts"][0]["sweep"]
    assert len(sweep["rows"]) == len(pcaols_result_sweep.sweep)


def test_groups_page_data(groups_text_result):
    gr = groups_text_result
    d = page_data(gr)
    json.dumps(d, allow_nan=False)
    assert d["kind"] == "groups"
    assert set(d["tables"]) == {"stats", "test", "pairs"}
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
        assert r.page_data()["report_format"] == REPORT_FORMAT


def test_format_2_texts_entry(pls_result):
    d = page_data(pls_result)
    assert d["report_format"] == 2 and "embeddings" not in d
    assert set(d["settings"]) == {"fit", "clusters"}
    [t] = d["texts"]
    assert set(t) == {"name", "embeddings", "settings", "k", "sweep", "n_no_vector", "snippets"}
    assert t["snippets"] is True and t["settings"]["lexicon"] == sorted(pls_result.lexicon)
    assert all(u["text"] == 0 for u in d["units"])


def test_sweep_moves_to_the_text(pcaols_result_sweep):
    d = page_data(pcaols_result_sweep)
    assert "sweep" not in d["tables"] and d["texts"][0]["sweep"]["rows"]
    assert d["texts"][0]["k"] == pcaols_result_sweep.fit_info.n_components


def test_token_list_text_has_tokens_and_no_snippets(pcaols_result_sweep):
    d = page_data(pcaols_result_sweep, doc_texts=None)
    t = d["units"][0]["tables"]
    assert d["texts"][0]["snippets"] is False and "snippets" not in t
    first = t["docs"]["rows"][0]
    assert first[-1] == " ".join(pcaols_result_sweep.text.docs[first[0]])


def test_regression_page(two, pack_df):
    essay, diary = two
    res = ssdiff.fit_regression("y ~ ssd(essay) + group*ssd(diary)", pack_df, essay=essay, diary=diary, fixed_k=3)
    d = page_data(res)
    json.dumps(d, allow_nan=False)
    assert d["kind"] == "pca_ols" and [t["name"] for t in d["texts"]] == ["essay", "diary"]
    assert [u["key"] for u in d["units"]][:2] == ["ssd(essay)", "ssd(diary)"]
    assert [u["text"] for u in d["units"]] == [0] + [1] * (len(d["units"]) - 1)
    assert d["texts"][1]["snippets"] is False
    assert d["tables"]["test"]["rows"][0][0] is None
    by_key = {u["key"]: u for u in d["units"]}
    assert by_key["ssd(essay)"]["test"] is None   # one label: its test is the "essay" row
    assert by_key["ssd(diary)"]["test"] == "ssd(diary)"
    assert by_key["grouplow:ssd(diary)"]["test"] == "group:ssd(diary)"


def test_one_text_regression_page(two, pack_df):
    essay = two[0]
    d = page_data(essay.fit_regression("y ~ group*ssd()", data=pack_df, fixed_k=3))
    assert len(d["texts"]) == 1 and d["texts"][0]["name"] == essay.name
    assert {u["text"] for u in d["units"]} == {0}
    assert [u["key"] for u in d["units"]] == [f"ssd({essay.name})", f"grouplow:ssd({essay.name})",
                                              f"groupmid:ssd({essay.name})"]
    assert [u["test"] for u in d["units"]] == [f"ssd({essay.name})"] + [f"group:ssd({essay.name})"] * 2


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
    assert _read_data(tmp_path / "page.html")["texts"][0]["embeddings"]["abtt"] == 1
