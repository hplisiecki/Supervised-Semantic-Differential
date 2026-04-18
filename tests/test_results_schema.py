"""Frozen-dataclass contract for all domain rows."""

import pickle

import pytest

from ssdiff.results.schema import (
    Cluster,
    ClusterWord,
    Doc,
    FitInfo,
    Pair,
    Snippet,
    Stats,
    Suggestion,
    Summary,
    Word,
)

ALL = [Word, Cluster, ClusterWord, Snippet, Doc, Pair, Suggestion,
       Stats, FitInfo, Summary]


@pytest.mark.parametrize("cls", ALL)
def test_frozen(cls):
    assert cls.__dataclass_params__.frozen is True


@pytest.mark.parametrize("cls", ALL)
def test_slots(cls):
    assert hasattr(cls, "__slots__")


def test_word_fields():
    w = Word(side="pos", rank=1, word="excellent", cos_beta=0.42, contrast=None)
    assert w.side == "pos"
    assert w.word == "excellent"
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        w.side = "neg"  # type: ignore


def test_cluster_has_no_methods_or_backref():
    c = Cluster(cluster_id=0, side="pos", size=5, coherence=0.7,
                centroid_cos_beta=0.3, contrast=None)
    # D3: pure data, no `words` / `snippets` properties
    assert not hasattr(c, "words")
    assert not hasattr(c, "snippets")


def test_snippet_nullable_cluster_id():
    s = Snippet(
        snippet_id=1, side="pos", doc_id=42, cosine=0.42,
        seed="excellent", start_token_idx=3, end_token_idx=7,
        start_sent_idx=0, end_sent_idx=1,
        text_window="the service was excellent",
        text_surface="the service was excellent today",
        text_lemmas="the service be excellent today",
        cluster_id=None, contrast=None,
        post_id=None,
    )
    assert s.cluster_id is None


def test_pickle_round_trip():
    w = Word(side="pos", rank=1, word="x", cos_beta=0.1, contrast=None)
    assert pickle.loads(pickle.dumps(w)) == w
