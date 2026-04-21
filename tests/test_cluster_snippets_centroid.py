"""Tests for centroid-based cluster snippets wired into ContinuousResult.

Covers:
- clusters.pos.snippets / .neg.snippets → SnippetsViewSided from centroid extraction
- Every populated cluster has ≥1 snippet (no silent empty tables)
- centroid_label rank maps back to Cluster.cluster_id in clusters._rows
- Cache key respects cluster params (re-clustered views get distinct entries)
- Top-level result.cluster_snippets(side=...) matches the property path
- β-snippets no longer carry cluster_id (seed_to_cluster enrichment removed)
- Pickle roundtrip preserves cached centroid snippets
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from ssdiff.corpus import Corpus
from ssdiff.results.continuous_result import SnippetsViewSided
from ssdiff.ssd import SSD


# ---------------------------------------------------------------------------
# Fixture: a fitted PLS result whose corpus has real pre_docs attached.
# The stock `pls_result` fixture uses pretokenized=True, which leaves
# corpus.pre_docs=None — no good for snippet extraction.
# ---------------------------------------------------------------------------

@pytest.fixture
def pls_result_with_pre_docs(tiny_kv, sample_preprocessed_docs, lexicon):
    docs = [list(pd.doc_lemmas) for pd in sample_preprocessed_docs]
    y = np.array([1.0, 1.5, 0.7, 1.2])
    corpus = Corpus(docs, pretokenized=True, lang="pl")
    corpus.pre_docs = list(sample_preprocessed_docs)
    ssd = SSD(tiny_kv, corpus, y, lexicon)
    return ssd.fit_pls(n_components=1, p_method="perm", n_perm=50,
                       random_state=42)


# ---------------------------------------------------------------------------
# 1. Every populated cluster has ≥1 snippet
# ---------------------------------------------------------------------------

def test_cluster_snippets_nonempty_per_cluster(pls_result_with_pre_docs):
    r = pls_result_with_pre_docs
    for side in ("pos", "neg"):
        sided_clusters = getattr(r.clusters, side)
        cluster_ids = {c.cluster_id for c in sided_clusters._rows}
        snippet_rows = list(sided_clusters.snippets(k=None))
        snippet_cluster_ids = {s.cluster_id for s in snippet_rows}
        missing = cluster_ids - snippet_cluster_ids
        assert not missing, (
            f"side={side}: clusters {missing} have no snippets "
            f"(present: {snippet_cluster_ids}, expected: {cluster_ids})"
        )


# ---------------------------------------------------------------------------
# 2. centroid_label rank maps back to Cluster.cluster_id
# ---------------------------------------------------------------------------

def test_centroid_label_maps_to_cluster_id(pls_result_with_pre_docs):
    r = pls_result_with_pre_docs
    sided = r.clusters.pos
    cluster_ids = [c.cluster_id for c in sided._rows]
    assert cluster_ids, "test needs at least one cluster"
    target_cid = cluster_ids[0]
    filtered = list(sided.snippets(k=None, cluster_id=target_cid))
    assert filtered, f"expected snippets for cluster_id={target_cid}"
    for s in filtered:
        assert s.cluster_id == target_cid
        assert s.side == "pos"


# ---------------------------------------------------------------------------
# 3. Cache key respects cluster params — distinct entries per cluster config
# ---------------------------------------------------------------------------

def test_cache_key_respects_cluster_params(pls_result_with_pre_docs):
    r = pls_result_with_pre_docs
    r.clear_cache()
    _ = r.clusters.pos.snippets
    _ = r.clusters.pos(topn=50).snippets
    entries = [k for k in r._cache if k[0] == "cluster_snippets"]
    topn_values = {dict(key).get("topn") for _, key in entries}
    assert len(entries) >= 2
    assert 100 in topn_values
    assert 50 in topn_values


# ---------------------------------------------------------------------------
# 4. Top-level accessor matches the property at default params
# ---------------------------------------------------------------------------

def test_top_level_accessor_matches_property(pls_result_with_pre_docs):
    r = pls_result_with_pre_docs
    r.clear_cache()
    via_property = list(r.clusters.pos.snippets(k=None))
    via_accessor = list(r.cluster_snippets(side="pos")(k=None))
    assert len(via_accessor) == len(via_property)
    for a, b in zip(via_accessor, via_property):
        assert a.snippet_id == b.snippet_id
        assert a.cluster_id == b.cluster_id
        assert a.cosine == b.cosine
        assert a.doc_id == b.doc_id
    entries = [k for k in r._cache if k[0] == "cluster_snippets"]
    assert len(entries) == 1


# ---------------------------------------------------------------------------
# 5. β-snippets no longer carry cluster_id
# ---------------------------------------------------------------------------

def test_beta_snippets_no_cluster_id(pls_result_with_pre_docs):
    r = pls_result_with_pre_docs
    _ = r.clusters.pos
    _ = r.clusters.neg
    beta_rows = list(r.snippets)
    assert beta_rows, "expected some β-snippets"
    assert all(s.cluster_id is None for s in beta_rows)


# ---------------------------------------------------------------------------
# 6. Pickle roundtrip preserves cached centroid snippets
# ---------------------------------------------------------------------------

def test_pickle_roundtrip_preserves_cluster_snippets(
    pls_result_with_pre_docs, tiny_kv, sample_preprocessed_docs,
):
    r = pls_result_with_pre_docs
    before = list(r.clusters.pos.snippets(k=None))

    blob = pickle.dumps(r)
    restored = pickle.loads(blob)

    docs = [list(pd.doc_lemmas) for pd in sample_preprocessed_docs]
    corpus = Corpus(docs, pretokenized=True, lang="pl")
    corpus.pre_docs = list(sample_preprocessed_docs)
    restored.attach(corpus=corpus, embeddings=tiny_kv)

    after = list(restored.clusters.pos.snippets(k=None))
    assert len(after) == len(before)
    for a, b in zip(after, before):
        assert a.snippet_id == b.snippet_id
        assert a.cluster_id == b.cluster_id
        assert a.cosine == pytest.approx(b.cosine)
        assert a.doc_id == b.doc_id
        assert a.text_window == b.text_window
