"""Tests for interpretation caching and .save() round-trip."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from ssdiff.results import PLSResult, PCAOLSResult, GroupResult


class TestRequireEmbeddings:
    """_require_embeddings() guard raises when embeddings is None."""

    def test_top_words_raises_without_embeddings(self, pls_result):
        """Loaded result without embeddings cannot recompute."""
        import copy
        stripped = copy.copy(pls_result)
        stripped.embeddings = None
        stripped._cached_top_words = None
        with pytest.raises(RuntimeError, match="requires embeddings"):
            stripped.top_words()

    def test_neighbors_raises_without_embeddings(self, pls_result):
        import copy
        stripped = copy.copy(pls_result)
        stripped.embeddings = None
        stripped._cached_top_words = None
        with pytest.raises(RuntimeError, match="requires embeddings"):
            stripped.neighbors("pos")

    def test_cluster_neighbors_raises_without_embeddings(self, pls_result):
        import copy
        stripped = copy.copy(pls_result)
        stripped.embeddings = None
        stripped._cached_clusters_pos = None
        stripped._cached_clusters_neg = None
        with pytest.raises(RuntimeError, match="requires embeddings"):
            stripped.cluster_neighbors("pos")


class TestTopWordsCache:
    def test_caches_result(self, pls_result):
        """Calling top_words() populates the cache."""
        result = pls_result
        words = result.top_words(n=5)
        assert result._cached_top_words is not None
        assert result._cached_top_words_n == 5
        assert len([w for w in result._cached_top_words if w["side"] == "pos"]) == 5

    def test_serves_smaller_n_from_cache(self, pls_result):
        """top_words(n=3) returns from cache when n=5 was cached."""
        import copy
        result = copy.copy(pls_result)
        result._cached_top_words = pls_result.top_words(n=5)
        result._cached_top_words_n = 5
        result.embeddings = None
        words = result.top_words(n=3)
        pos = [w for w in words if w["side"] == "pos"]
        neg = [w for w in words if w["side"] == "neg"]
        assert len(pos) == 3
        assert len(neg) == 3
        assert all(w["rank"] <= 3 for w in words)

    def test_larger_n_without_embeddings_raises(self, pls_result):
        """Requesting more than cached without embeddings raises."""
        import copy
        result = copy.copy(pls_result)
        result._cached_top_words = pls_result.top_words(n=3)
        result._cached_top_words_n = 3
        result.embeddings = None
        with pytest.raises(RuntimeError, match="requires embeddings"):
            result.top_words(n=10)

    def test_recompute_flag(self, pls_result):
        """recompute=True forces fresh computation even with cache."""
        result = pls_result
        old_cache = result.top_words(n=5)
        new = result.top_words(n=5, recompute=True)
        assert len(new) == len(old_cache)


class TestNeighborsCache:
    def test_delegates_to_top_words(self, pls_result):
        """neighbors() uses top_words() internally — no separate cache."""
        import copy
        result = copy.copy(pls_result)
        result._cached_top_words = pls_result.top_words(n=10)
        result._cached_top_words_n = 10
        result.embeddings = None
        nbrs = result.neighbors("pos", n=5)
        assert len(nbrs) == 5
        assert all(isinstance(n, tuple) and len(n) == 2 for n in nbrs)


class TestClusterNeighborsCache:
    def test_caches_pos(self, pls_result):
        """cluster_neighbors('pos') caches to _cached_clusters_pos."""
        result = pls_result
        clusters = result.cluster_neighbors("pos", topn=10)
        assert result._cached_clusters_pos is not None
        assert result._cached_clusters_pos is clusters

    def test_serves_from_cache(self, pls_result):
        """Returns cached clusters without recomputing."""
        import copy
        result = copy.copy(pls_result)
        cached = [{"id": 0, "size": 2, "words": []}]
        result._cached_clusters_pos = cached
        result.embeddings = None
        got = result.cluster_neighbors("pos")
        assert got is cached

    def test_recompute_flag(self, pls_result):
        """recompute=True forces fresh computation."""
        result = pls_result
        old = result.cluster_neighbors("pos", topn=10)
        new = result.cluster_neighbors("pos", topn=10, recompute=True)
        assert new is not old


class TestSnippetsCache:
    def test_caches_result(self, pls_result, sample_preprocessed_docs):
        """Calling snippets() populates the cache."""
        result = pls_result
        snips = result.snippets(sample_preprocessed_docs, top_per_side=10)
        assert result._cached_snippets is not None
        assert result._cached_snippets_top == 10
        assert "pos" in snips and "neg" in snips

    def test_serves_from_cache_without_pre_docs(self, pls_result, sample_preprocessed_docs):
        """Cached snippets returned when no pre_docs provided."""
        import copy
        result = copy.copy(pls_result)
        cached = {"pos": [{"snippet": "test"}], "neg": []}
        result._cached_snippets = cached
        result._cached_snippets_top = 200
        result.embeddings = None
        got = result.snippets(top_per_side=100)
        assert got is cached

    def test_no_cache_no_pre_docs_raises(self, pls_result):
        """No cache and no pre_docs raises ValueError."""
        import copy
        result = copy.copy(pls_result)
        result._cached_snippets = None
        result.embeddings = None
        with pytest.raises(ValueError, match="No cached snippets"):
            result.snippets()

    def test_larger_top_recomputes(self, pls_result, sample_preprocessed_docs):
        """Requesting more snippets than cached triggers recompute."""
        import copy
        result = copy.copy(pls_result)
        result._cached_snippets = None
        result._cached_snippets_top = 0
        result.snippets(sample_preprocessed_docs, top_per_side=5)
        assert result._cached_snippets_top == 5
        result.snippets(sample_preprocessed_docs, top_per_side=10)
        assert result._cached_snippets_top == 10


class TestClusterSnippetsCache:
    def test_serves_from_cache_without_pre_docs(self, pls_result):
        """Cached cluster snippets returned when no pre_docs provided."""
        import copy
        result = copy.copy(pls_result)
        cached = {"pos": [], "neg": []}
        result._cached_cluster_snippets = cached
        result._cached_cluster_snippets_top = 100
        result.embeddings = None
        got = result.cluster_snippets(top_per_cluster=50)
        assert got is cached

    def test_no_cache_no_pre_docs_raises(self, pls_result):
        """No cache and no pre_docs raises ValueError."""
        import copy
        result = copy.copy(pls_result)
        result._cached_cluster_snippets = None
        result.embeddings = None
        with pytest.raises(ValueError, match="No cached cluster_snippets"):
            result.cluster_snippets()


class TestGroupResultCaching:
    def test_top_words_caches(self, group_result_2g):
        """GroupResult.top_words() populates cache with contrast key."""
        result = group_result_2g
        words = result.top_words(n=5)
        assert result._cached_top_words is not None
        assert all("contrast" in w for w in words)

    def test_top_words_serves_from_cache(self, group_result_2g):
        """Cached top_words returned without embeddings."""
        import copy
        result = copy.copy(group_result_2g)
        result._cached_top_words = group_result_2g.top_words(n=5)
        result._cached_top_words_n = 5
        result.embeddings = None
        words = result.top_words(n=3)
        pos = [w for w in words if w["side"] == "pos"]
        assert all(w["rank"] <= 3 for w in pos)

    def test_cluster_neighbors_caches(self, group_result_2g):
        """GroupResult.cluster_neighbors() caches all-contrasts list."""
        result = group_result_2g
        clusters = result.cluster_neighbors("pos", topn=10)
        assert result._cached_clusters_pos is not None
        assert all("contrast" in c for c in clusters)

    def test_cluster_neighbors_serves_from_cache(self, group_result_2g):
        """Cached cluster_neighbors returned without embeddings."""
        import copy
        result = copy.copy(group_result_2g)
        cached = [{"id": 0, "contrast": "A vs B", "words": []}]
        result._cached_clusters_pos = cached
        result.embeddings = None
        got = result.cluster_neighbors("pos")
        assert got is cached

    def test_snippets_caches(self, group_result_2g):
        """GroupResult.snippets() populates cache with contrast keys."""
        import copy
        result = copy.copy(group_result_2g)
        cached = {"pos": [{"contrast": "A vs B"}], "neg": []}
        result._cached_snippets = cached
        result._cached_snippets_top = 200
        result.embeddings = None
        got = result.snippets(top_per_side=100)
        assert got is cached


class TestSaveRoundTrip:
    def test_pls_save_load(self, pls_result, tmp_path):
        """PLSResult: save → load preserves cached data."""
        pls_result.top_words(n=10)
        pls_result.cluster_neighbors("pos", topn=10)
        pls_result.cluster_neighbors("neg", topn=10)

        path = tmp_path / "result.pkl"
        pls_result.save(path)

        with open(path, "rb") as f:
            loaded = pickle.load(f)

        # Stripped fields
        assert loaded.embeddings is None
        assert loaded.perm_null is None

        # Preserved caches
        assert loaded._cached_top_words is not None
        assert loaded._cached_clusters_pos is not None
        assert loaded._cached_clusters_neg is not None

        # Cached data matches
        cached_words = loaded.top_words(n=5)
        assert len(cached_words) > 0
        assert all(w["rank"] <= 5 for w in cached_words)

        # x preserved for doc_scores
        scores = loaded.doc_scores()
        assert "cos_align" in scores

    def test_pcaols_save_load(self, pcaols_result, tmp_path):
        """PCAOLSResult: save → load strips sweep_result."""
        pcaols_result.top_words(n=5)
        path = tmp_path / "result.pkl"
        pcaols_result.save(path)

        with open(path, "rb") as f:
            loaded = pickle.load(f)

        assert loaded.embeddings is None
        assert loaded.sweep_result is None
        assert loaded._cached_top_words is not None

    def test_group_save_load(self, group_result_2g, tmp_path):
        """GroupResult: save → load preserves cached data."""
        group_result_2g.top_words(n=5)
        group_result_2g.cluster_neighbors("pos", topn=10)

        path = tmp_path / "result.pkl"
        group_result_2g.save(path)

        with open(path, "rb") as f:
            loaded = pickle.load(f)

        assert loaded.embeddings is None
        assert loaded._cached_top_words is not None
        assert all("contrast" in w for w in loaded._cached_top_words)

    def test_save_does_not_modify_original(self, pls_result, tmp_path):
        """Saving does not strip embeddings from the original object."""
        assert pls_result.embeddings is not None
        pls_result.save(tmp_path / "result.pkl")
        assert pls_result.embeddings is not None

    def test_loaded_without_cache_raises(self, pls_result, tmp_path):
        """Loaded result with no cache raises RuntimeError on top_words()."""
        import copy
        fresh = copy.copy(pls_result)
        fresh._cached_top_words = None
        fresh._cached_top_words_n = 0
        stripped = copy.copy(fresh)
        stripped.embeddings = None
        path = tmp_path / "nocache.pkl"
        with open(path, "wb") as f:
            pickle.dump(stripped, f)

        with open(path, "rb") as f:
            loaded = pickle.load(f)

        with pytest.raises(RuntimeError, match="requires embeddings"):
            loaded.top_words()

    def test_recompute_after_reattach(self, pls_result, tiny_kv, tmp_path):
        """Re-attaching embeddings to loaded result enables recompute."""
        pls_result.top_words(n=5)
        path = tmp_path / "result.pkl"
        pls_result.save(path)

        with open(path, "rb") as f:
            loaded = pickle.load(f)

        loaded.embeddings = tiny_kv
        words = loaded.top_words(n=10, recompute=True)
        assert loaded._cached_top_words_n == 10
