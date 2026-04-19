"""Tests for result caching: param-keyed cache behaviour and pickle round-trip.

The stale-cache regression (spec issue #2): two calls with different params
must each get their own cache entry; the first entry must NOT be overwritten.
This is the primary regression the redesign fixes.

Note: Session-scoped fixtures cannot be mutated (clear_cache / cache injection).
Tests that need a clean _cache use class-scoped or function-scoped copies,
constructing a fresh ClustersIndex pointing at the right parent.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from ssdiff.results.continuous_result import ClustersIndex


def _make_fresh_pls(ssd_instance):
    """Fit a new PLSResult from the shared SSD instance (not session-cached)."""
    return ssd_instance.fit_pls(n_components=2, p_method="perm", n_perm=50, random_state=42)


class TestRequireEmbeddings:
    """_require_resource() guard raises when embeddings is None."""

    def test_words_raises_without_embeddings(self, ssd_instance):
        """Accessing .words without embeddings raises RuntimeError."""
        r = _make_fresh_pls(ssd_instance)
        r.embeddings = None
        r._cache = {}
        with pytest.raises(RuntimeError, match="attach"):
            _ = r.words

    def test_clusters_raises_without_embeddings(self, ssd_instance):
        """Accessing .clusters.pos without embeddings raises RuntimeError."""
        r = _make_fresh_pls(ssd_instance)
        r.embeddings = None
        r._cache = {}
        # Re-wire ClustersIndex parent to point at this result so it reads
        # r.embeddings (None) rather than the old result.
        r.clusters = ClustersIndex(r)
        with pytest.raises(RuntimeError, match="attach"):
            _ = r.clusters.pos


class TestWordsCache:
    """words property is cached under key ('words', ())."""

    def test_words_populates_cache(self, ssd_instance):
        r = _make_fresh_pls(ssd_instance)
        r._cache = {}
        _ = r.words
        assert ("words", ()) in r._cache

    def test_words_returns_same_view_on_second_call(self, ssd_instance):
        r = _make_fresh_pls(ssd_instance)
        r._cache = {}
        view1 = r.words
        view2 = r.words
        assert view1 is view2

    def test_words_view_is_sliceable(self, ssd_instance):
        """list(result.words)[:5] gives 5 Word objects with side/word attrs."""
        r = _make_fresh_pls(ssd_instance)
        r._cache = {}
        top5 = list(r.words)[:5]
        assert len(top5) == 5
        assert all(hasattr(w, "side") and hasattr(w, "word") for w in top5)

    def test_clear_cache_words_drops_entry(self, ssd_instance):
        r = _make_fresh_pls(ssd_instance)
        r._cache = {}
        _ = r.words
        assert ("words", ()) in r._cache
        r.clear_cache("words")
        assert ("words", ()) not in r._cache

    def test_words_without_embeddings_raises_after_clear(self, ssd_instance):
        """After clear_cache and stripping embeddings, accessing .words raises."""
        r = _make_fresh_pls(ssd_instance)
        r._cache = {}
        _ = r.words          # populate
        r.clear_cache("words")
        r.embeddings = None
        with pytest.raises(RuntimeError, match="attach"):
            _ = r.words


class TestClustersCache:
    """Clusters are param-keyed: different topn → different cache entries.

    This is the regression test for spec issue #2: the first cache entry
    must NOT be overwritten when a second call uses different params.
    """

    def test_stale_cache_regression(self, ssd_instance, monkeypatch):
        """Two calls with different topn each get their own cache entry.

        Regression guard for spec issue #2: calling clusters.pos(topn=50)
        after clusters.pos (topn=100) must NOT overwrite the first entry.
        """
        from ssdiff.results.schema import Cluster

        r = _make_fresh_pls(ssd_instance)
        r._cache = {}

        call_count = {"n": 0}

        def _fake_clusters(self, *, side, topn, k, k_min, k_max,
                           random_state, min_cluster_size):
            call_count["n"] += 1
            return (
                [Cluster(cluster_id=0, side=side, size=topn,
                         coherence=0.5, centroid_cos_beta=0.3, contrast=None)],
                [],
            )

        monkeypatch.setattr(type(r), "_compute_clusters_for_side", _fake_clusters)
        r.clusters = ClustersIndex(r)

        v100 = r.clusters.pos            # topn=100 (default)
        v50 = r.clusters.pos(topn=50)   # new params → new entry
        v100_again = r.clusters.pos     # must be cache hit, same object

        assert v100 is v100_again, "First cache entry was overwritten (regression)"
        assert v100[0].size == 100
        assert v50[0].size == 50
        assert call_count["n"] == 2, f"Expected 2 compute calls, got {call_count['n']}"

    def test_clusters_pos_cached(self, ssd_instance):
        r = _make_fresh_pls(ssd_instance)
        r._cache = {}
        r.clusters = ClustersIndex(r)
        _ = r.clusters.pos
        cluster_keys = [k for k in r._cache if k[0] == "clusters"]
        assert len(cluster_keys) >= 1

    def test_clusters_neg_independent_from_pos(self, ssd_instance):
        r = _make_fresh_pls(ssd_instance)
        r._cache = {}
        r.clusters = ClustersIndex(r)
        cl_pos = r.clusters.pos
        cl_neg = r.clusters.neg
        for c in cl_pos:
            assert c.side == "pos"
        for c in cl_neg:
            assert c.side == "neg"

    def test_clear_cache_clusters_drops_all_cluster_entries(self, ssd_instance):
        r = _make_fresh_pls(ssd_instance)
        r._cache = {}
        r.clusters = ClustersIndex(r)
        _ = r.clusters.pos
        _ = r.clusters.neg
        _ = r.clusters.pos(topn=50)
        r.clear_cache("clusters")
        cluster_keys = [k for k in r._cache if k[0] == "clusters"]
        assert cluster_keys == []

    def test_two_different_topn_entries_both_exist(self, ssd_instance):
        """After two calls with different topn, both cache entries exist."""
        r = _make_fresh_pls(ssd_instance)
        r._cache = {}
        r.clusters = ClustersIndex(r)
        _ = r.clusters.pos            # topn=100 (default)
        _ = r.clusters.pos(topn=50)   # different params
        cluster_keys = [k for k in r._cache if k[0] == "clusters"]
        assert len(cluster_keys) == 2


class TestSnippetsCache:
    """Snippets are param-keyed: top_per_side is the key param."""

    def test_snippets_cached_after_access(self, ssd_instance):
        """Accessing .snippets populates the cache."""
        r = _make_fresh_pls(ssd_instance)
        r._cache = {}
        _ = r.snippets
        snip_keys = [k for k in r._cache if k[0] == "snippets"]
        assert len(snip_keys) >= 1

    def test_different_top_per_side_gets_own_entry(self, ssd_instance):
        """Two calls with different top_per_side each get their own cache entry."""
        r = _make_fresh_pls(ssd_instance)
        r._cache = {}
        s30 = r.snippets               # top_per_side=30 default
        r._snippets_for(top_per_side=10)
        # Must be different views (different params → different entries)
        snip_keys = [k for k in r._cache if k[0] == "snippets"]
        assert len(snip_keys) == 2, f"Expected 2 snippet entries, got {snip_keys}"
        # First entry not overwritten
        s30_again = r.snippets
        assert s30 is s30_again

    def test_snippets_without_corpus_raises(self, ssd_instance):
        """Accessing .snippets without corpus raises RuntimeError."""
        r = _make_fresh_pls(ssd_instance)
        r._cache = {}
        r.corpus = None
        with pytest.raises(RuntimeError, match="attach"):
            _ = r.snippets


class TestGroupResultCaching:
    """GroupResult stores flat rows; cache holds param-keyed views."""

    def test_group_result_has_cache(self, group_result_2g):
        """GroupResult has a _cache dict from the base class."""
        assert hasattr(group_result_2g, "_cache")
        assert isinstance(group_result_2g._cache, dict)

    def test_group_clear_cache_works(self, tiny_kv, large_docs, large_groups_2, lexicon):
        """clear_cache() on GroupResult clears the dict."""
        from ssdiff.corpus import Corpus
        from ssdiff.ssd import SSD
        corpus = Corpus(large_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, large_groups_2, lexicon)
        gr = ssd.fit_groups(n_perm=50, random_state=42)
        # Manually add a fake cache entry to verify clear works
        gr._cache[("fake", ())] = object()
        gr.clear_cache()
        assert gr._cache == {}


class TestPickleRoundTrip:
    """Pickle round-trip preserves the new _cache dict and critical attrs."""

    def test_pls_pickle_round_trip(self, ssd_instance, tmp_path):
        """PLSResult: pickle/unpickle preserves cache and core arrays."""
        r = _make_fresh_pls(ssd_instance)
        # Populate cache
        _ = r.words
        assert ("words", ()) in r._cache

        path = tmp_path / "result.pkl"
        with open(path, "wb") as f:
            pickle.dump(r, f)

        with open(path, "rb") as f:
            loaded = pickle.load(f)

        # Core arrays preserved
        np.testing.assert_array_equal(loaded.x, r.x)
        np.testing.assert_array_equal(loaded.y, r.y)
        np.testing.assert_array_equal(loaded.beta, r.beta)

        # Cache preserved
        assert ("words", ()) in loaded._cache

    def test_pls_pickle_embeddings_preserved(self, ssd_instance, tmp_path):
        """Pickle/unpickle preserves embeddings reference."""
        r = _make_fresh_pls(ssd_instance)
        assert r.embeddings is not None
        path = tmp_path / "result.pkl"
        with open(path, "wb") as f:
            pickle.dump(r, f)
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        # Embeddings are preserved by default (new design doesn't strip)
        assert loaded.embeddings is not None

    def test_pcaols_pickle_round_trip(self, pcaols_result, tmp_path):
        """PCAOLSResult: pickle/unpickle round-trip."""
        r = pcaols_result
        # words cache should work on the session-scoped fixture since we don't mutate it
        # We just pickle and unpickle and check stats
        path = tmp_path / "result.pkl"
        with open(path, "wb") as f:
            pickle.dump(r, f)
        with open(path, "rb") as f:
            loaded = pickle.load(f)

        np.testing.assert_array_equal(loaded.beta, r.beta)
        assert loaded.stats.r2 == pytest.approx(r.stats.r2)

    def test_group_pickle_round_trip(self, group_result_2g, tmp_path):
        """GroupResult: pickle/unpickle round-trip preserves rows."""
        r = group_result_2g
        path = tmp_path / "result.pkl"
        with open(path, "wb") as f:
            pickle.dump(r, f)
        with open(path, "rb") as f:
            loaded = pickle.load(f)

        assert loaded.G == r.G
        assert loaded.n_kept == r.n_kept
        assert len(loaded.pairs) == len(r.pairs)

    def test_pickle_does_not_mutate_original(self, ssd_instance, tmp_path):
        """Pickling does not change the original's embeddings or cache."""
        r = _make_fresh_pls(ssd_instance)
        emb_before = r.embeddings
        cache_before = dict(r._cache)
        path = tmp_path / "result.pkl"
        with open(path, "wb") as f:
            pickle.dump(r, f)
        assert r.embeddings is emb_before
        assert r._cache == cache_before

    def test_stripped_result_raises_on_words(self, ssd_instance, tmp_path):
        """Loaded result with stripped embeddings and no cache raises on .words."""
        r = _make_fresh_pls(ssd_instance)
        r.embeddings = None
        r._cache = {}
        path = tmp_path / "nocache.pkl"
        with open(path, "wb") as f:
            pickle.dump(r, f)
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        with pytest.raises(RuntimeError, match="attach"):
            _ = loaded.words

    def test_attach_after_pickle(self, ssd_instance, tiny_kv, tmp_path):
        """Re-attaching embeddings after unpickling lets words resolve."""
        r = _make_fresh_pls(ssd_instance)
        r._cache = {}
        r.embeddings = None
        path = tmp_path / "result.pkl"
        with open(path, "wb") as f:
            pickle.dump(r, f)
        with open(path, "rb") as f:
            loaded = pickle.load(f)

        assert loaded.embeddings is None
        # Re-attach and it works
        loaded.attach(embeddings=tiny_kv)
        loaded.clusters = ClustersIndex(loaded)  # rewire parent reference
        words = list(loaded.words)
        assert len(words) > 0
