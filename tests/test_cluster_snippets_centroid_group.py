"""Group-side cluster-snippets centroid fix regression tests.

Mirrors tests/test_cluster_snippets_centroid.py for GroupResult's
single-pair path. All five tests use a 2-group synthetic fixture.
Post-refactor: access clusters via gr[pair].clusters.pos (canonical path).
"""

import pickle

import numpy as np
import pytest

from ssdiff.corpus import Corpus
from ssdiff.ssd import SSD
from ssdiff.utils.text import PreprocessedDoc


def _make_pre_doc(tokens: list[str]) -> PreprocessedDoc:
    """Build a minimal PreprocessedDoc from a flat token list."""
    raw = " ".join(tokens)
    n = len(tokens)
    return PreprocessedDoc(
        raw=raw,
        sents_surface=[raw],
        sents_lemmas=[tokens],
        doc_lemmas=tokens,
        sent_char_spans=[(0, len(raw))],
        token_to_sent=list(range(n)),
        sents_kept_idx=[list(range(n))],
    )


@pytest.fixture
def gr(tiny_kv, large_docs, large_groups_2, lexicon):
    """2-group GroupResult with a live corpus for snippet extraction.

    Uses large_docs (50 docs, 25 per group) which satisfies the
    minimum-group-size filter, and synthesises PreprocessedDoc objects
    so snippet extraction can run without spaCy.
    """
    pre_docs = [_make_pre_doc(doc) for doc in large_docs]
    corpus = Corpus(large_docs, pretokenized=True, lang="pl")
    corpus.pre_docs = pre_docs
    ssd = SSD(tiny_kv, corpus, large_groups_2, lexicon)
    return ssd.fit_groups(n_perm=50, random_state=42)


def _leaf(gr):
    """Return the single PairResult leaf from a 2-group GroupResult."""
    pair = next(iter(gr.pairs))
    return gr[(pair.g1, pair.g2)]


def test_cluster_snippets_nonempty_per_cluster(gr):
    leaf = _leaf(gr)
    pos_rows = list(leaf.clusters.pos.snippets(k=None))
    pos_cluster_ids = {c.cluster_id for c in leaf.clusters.pos}
    assert pos_cluster_ids, "test fixture produced no positive clusters"
    seen = {s.cluster_id for s in pos_rows}
    assert pos_cluster_ids.issubset(seen), (
        f"clusters without snippets: {pos_cluster_ids - seen}"
    )
    neg_rows = list(leaf.clusters.neg.snippets(k=None))
    neg_cluster_ids = {c.cluster_id for c in leaf.clusters.neg}
    seen_neg = {s.cluster_id for s in neg_rows}
    assert neg_cluster_ids.issubset(seen_neg)


def test_centroid_label_maps_to_cluster_id(gr):
    leaf = _leaf(gr)
    first_cid = leaf.clusters.pos[0].cluster_id
    filtered = leaf.clusters.pos.snippets(cluster_id=first_cid)
    assert len(filtered) > 0
    assert all(s.cluster_id == first_cid for s in filtered)


def test_cache_key_respects_cluster_params(gr):
    leaf = _leaf(gr)
    _ = leaf.clusters.pos.snippets
    _ = leaf.cluster_snippets(side="pos", top_per_cluster=50)
    entries = [k for k in leaf._cache if k[0] == "cluster_snippets"]
    assert len(entries) == 2, f"expected 2 cache entries, got {entries}"


def test_top_level_accessor_matches_property(gr):
    leaf = _leaf(gr)
    a = list(leaf.clusters.pos.snippets)
    b = list(leaf.cluster_snippets(side="pos"))
    assert len(a) == len(b)
    assert all(x.snippet_id == y.snippet_id and x.cosine == y.cosine
               for x, y in zip(a, b))


def test_pickle_roundtrip_preserves_cluster_snippets(gr):
    leaf = _leaf(gr)
    _ = leaf.clusters.pos.snippets
    data = pickle.dumps(gr)
    gr2 = pickle.loads(data)
    gr2.attach(corpus=gr.corpus, embeddings=gr.embeddings)
    leaf2 = _leaf(gr2)
    rows = list(leaf2.clusters.pos.snippets)
    assert len(rows) > 0
