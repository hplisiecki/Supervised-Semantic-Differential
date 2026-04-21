"""Flat-view iteration and save tests for ContinuousResult and GroupResult."""

import numpy as np
import pytest

from ssdiff.corpus import Corpus
from ssdiff.results import ClustersView
from ssdiff.results.schema import Cluster
from ssdiff.ssd import SSD


@pytest.fixture
def cr(tiny_kv, sample_preprocessed_docs, lexicon):
    """Fitted PLS result with pre_docs attached."""
    docs = [list(pd.doc_lemmas) for pd in sample_preprocessed_docs]
    y = np.array([1.0, 1.5, 0.7, 1.2])
    corpus = Corpus(docs, pretokenized=True, lang="pl")
    corpus.pre_docs = list(sample_preprocessed_docs)
    ssd = SSD(tiny_kv, corpus, y, lexicon)
    return ssd.fit_pls(n_components=1, p_method="perm", n_perm=50, random_state=42)


def test_continuous_clusters_is_clusters_view(cr):
    assert isinstance(cr.clusters, ClustersView)


def test_continuous_clusters_iterable_flat(cr):
    rows = list(cr.clusters)
    assert all(isinstance(r, Cluster) for r in rows)
    pos = [r for r in rows if r.side == "pos"]
    neg = [r for r in rows if r.side == "neg"]
    assert len(pos) > 0 and len(neg) > 0
    first_neg_idx = next(i for i, r in enumerate(rows) if r.side == "neg")
    assert all(r.side == "pos" for r in rows[:first_neg_idx])


def test_continuous_clusters_save_csv(tmp_path, cr):
    target = tmp_path / "clusters.csv"
    cr.clusters.save(str(target))
    assert target.exists()
    text = target.read_text()
    assert "side" in text.splitlines()[0]
    assert "pos" in text and "neg" in text


def test_continuous_clusters_pos_accessor_preserved(cr):
    pos = cr.clusters.pos
    assert all(c.side == "pos" for c in pos)


def test_continuous_clusters_words_accessor_preserved(cr):
    cw = cr.clusters.words
    assert len(list(cw)) > 0


def test_group_words_flat_iteration(group_result_3g):
    rows = list(group_result_3g.words)
    pairs = {r.contrast for r in rows}
    assert len(pairs) == 3


def test_group_words_len(group_result_3g):
    assert len(group_result_3g.words) == sum(
        len(group_result_3g.words[k]) for k in group_result_3g.words.keys()
    )


def test_group_words_dict_access_preserved(group_result_3g):
    keys = group_result_3g.words.keys()
    first_key = keys[0]
    single = group_result_3g.words[first_key]
    contrast = f"{first_key[0]}_{first_key[1]}"
    assert all(w.contrast == contrast for w in single)


def test_paired_save_csv_fanout_multi(tmp_path, group_result_3g):
    target = tmp_path / "words.csv"
    group_result_3g.words.save(str(target))
    out_dir = tmp_path / "words"
    assert out_dir.exists()
    files = sorted(p.name for p in out_dir.glob("*.csv"))
    assert len(files) == 3


def test_paired_save_xlsx_single_file(tmp_path, group_result_3g):
    pytest.importorskip("openpyxl")
    target = tmp_path / "words.xlsx"
    group_result_3g.words.save(str(target))
    assert target.exists()
    import openpyxl
    wb = openpyxl.load_workbook(str(target))
    assert len(wb.sheetnames) == 3


def test_paired_save_json_single_file(tmp_path, group_result_3g):
    import json
    target = tmp_path / "words.json"
    group_result_3g.words.save(str(target))
    data = json.loads(target.read_text())
    assert len(data.keys()) == 3
