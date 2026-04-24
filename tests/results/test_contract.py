"""Structural contract tests for ssdiff.results.

Covers core.py (cache API, View, ScalarView), schema.py (frozen dataclasses),
single_result.py (_SingleResult invariants), and lexicon_result.py.

One test per invariant — 15 tests total.
"""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest
from dataclasses import FrozenInstanceError

from ssdiff.results.core import Result, ScalarView, View
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
from ssdiff.results.lexicon_result import LexiconResult


# ---------------------------------------------------------------------------
# Helpers / tiny fixtures
# ---------------------------------------------------------------------------

def _make_simple_result() -> Result:
    """Minimal concrete Result with a working _cache for cache tests."""

    class _TinyView(View):
        _name = "dummy"
        _columns = ("x",)

        def __init__(self, value, **kw):
            super().__init__(**kw)
            self._value = value

        def __iter__(self):
            yield {"x": self._value}

        def __len__(self):
            return 1

        def __getitem__(self, i):
            if isinstance(i, slice):
                return self
            if i == 0:
                return {"x": self._value}
            raise IndexError(i)

    class _TinyResult(Result):
        def get_view(self, k: int) -> View:
            return self._cache_get("clusters", {"k": k}, lambda: _TinyView(k))

    r = _TinyResult()
    return r


def _make_scalar_view() -> ScalarView:
    """Minimal concrete ScalarView for invariant-5 testing."""

    class _MinScalar(ScalarView):
        _name = "mini"
        _columns = ("alpha", "beta")

        def __iter__(self):
            yield {"alpha": 1, "beta": 2}

    return _MinScalar()


def _make_lexicon_result(*, sort_descending=True) -> LexiconResult:
    """LexiconResult with known suggestions in a fixture-deterministic order."""
    # Three suggestions; corr acts as our effect size — in [0,1] when
    # var_type="categorical" (Cramér's V range).
    # rank lower=better (combined score): we set rank such that tok_a < tok_b < tok_c
    suggestions = [
        Suggestion(token="tok_a", freq=10, cov_all=0.5, cov_bal=0.4,
                   corr=0.8, pvalue=0.01, direction="positive", rank=0.1),
        Suggestion(token="tok_b", freq=8,  cov_all=0.4, cov_bal=0.3,
                   corr=0.5, pvalue=0.05, direction="positive", rank=0.3),
        Suggestion(token="tok_c", freq=5,  cov_all=0.3, cov_bal=0.2,
                   corr=0.2, pvalue=0.2,  direction="none",     rank=0.7),
    ]
    return LexiconResult(
        var_type="categorical",
        n_docs=100,
        n_tokens=3,
        suggestions=suggestions,
    )


# ---------------------------------------------------------------------------
# Invariant 1: _cache_get — first call computes, second call is identical object
# ---------------------------------------------------------------------------

def test_cache_get_identity_on_second_call():
    """Second _cache_get call with same (name, params) returns identical object."""
    r = _make_simple_result()
    first = r._cache_get("dummy", {"k": 5}, lambda: object())
    second = r._cache_get("dummy", {"k": 5}, lambda: object())
    assert first is second, "Cache must return identical object on second call"


# ---------------------------------------------------------------------------
# Invariant 2: Stale-cache regression — different params coexist independently
# ---------------------------------------------------------------------------

def test_cache_different_params_coexist():
    """Two distinct param sets produce two independent cache entries."""
    r = _make_simple_result()
    view5 = r.get_view(5)
    view10 = r.get_view(10)

    # Both entries exist independently — they are not the same object
    assert view5 is not view10

    # Re-fetching each returns the same original object (not recomputed)
    assert r.get_view(5) is view5
    assert r.get_view(10) is view10


# ---------------------------------------------------------------------------
# Invariant 3: clear_cache("words") removes only "words" entries
# ---------------------------------------------------------------------------

def test_clear_cache_named_view_selective():
    """clear_cache('words') removes 'words' entries; other entries survive."""
    r = _make_simple_result()
    sentinel_words = object()
    sentinel_clusters = object()

    r._cache[("words", ())] = sentinel_words
    r._cache[("clusters", (("k", 5),))] = sentinel_clusters

    r.clear_cache("words")

    assert ("words", ()) not in r._cache, "words entry should be gone"
    assert r._cache.get(("clusters", (("k", 5),))) is sentinel_clusters, \
        "clusters entry must survive"


# ---------------------------------------------------------------------------
# Invariant 4: View.__getitem__ and iteration work consistently
# ---------------------------------------------------------------------------

def test_view_getitem_and_iteration_consistent():
    """View[i] and iter(view) return matching data."""

    class _ListedView(View):
        _name = "listed"
        _columns = ("v",)

        def __init__(self, items):
            super().__init__()
            self._items = items

        def __iter__(self):
            return iter(self._items)

        def __len__(self):
            return len(self._items)

        def __getitem__(self, i):
            if isinstance(i, slice):
                return _ListedView(self._items[i])
            return self._items[i]

    data = [{"v": 10}, {"v": 20}, {"v": 30}]
    view = _ListedView(data)

    # iter
    as_list = list(view)
    assert as_list == data

    # __getitem__
    assert view[0] == {"v": 10}
    assert view[2] == {"v": 30}

    # slicing returns a new view and has correct length
    sliced = view[:2]
    assert len(sliced) == 2
    assert list(sliced) == data[:2]


# ---------------------------------------------------------------------------
# Invariant 5: ScalarView attr-access and dict-access return same values
# ---------------------------------------------------------------------------

def test_scalar_view_attr_and_dict_access_agree():
    """ScalarView attribute access and dict-style access return identical values."""
    sv = _make_scalar_view()

    # attr access
    alpha_attr = sv.alpha
    beta_attr = sv.beta

    # dict-style access
    alpha_dict = sv["alpha"]
    beta_dict = sv["beta"]

    assert alpha_attr == alpha_dict == 1
    assert beta_attr == beta_dict == 2


# ---------------------------------------------------------------------------
# Invariant 6: attach(embeddings=...) wires _embeddings attribute correctly
# ---------------------------------------------------------------------------

def test_attach_embeddings_wires_attribute(pls_result, tiny_kv):
    """result.attach(embeddings=new_emb) sets result.embeddings to new_emb."""
    # Use a different Embeddings-like object to verify identity
    from tests.conftest import make_kv
    new_emb = make_kv(["a", "b"], dim=4, seed=7)

    pls_result.attach(embeddings=new_emb)
    assert pls_result.embeddings is new_emb, \
        "embeddings must be wired by identity after attach()"

    # Restore original embeddings so other tests are not affected
    pls_result.attach(embeddings=tiny_kv)


# ---------------------------------------------------------------------------
# Invariant 7: result.words before attach raises RuntimeError mentioning "attach"
# ---------------------------------------------------------------------------

def test_words_without_embeddings_raises_runtime_error(ssd_instance):
    """Accessing .words when embeddings=None raises RuntimeError with 'attach'.

    Note: Result.attach(embeddings=None) is a no-op (guarded by `if embeddings is not None`).
    To simulate the un-attached state we directly set the attribute and clear the cache.
    """
    result = ssd_instance.fit_pls(n_components=1, p_method="perm", n_perm=20, random_state=0)

    # Save original embeddings so we can restore after the test
    original_emb = result.embeddings

    # Directly strip embeddings to simulate un-attached state
    result.embeddings = None
    result.clear_cache("words")

    try:
        with pytest.raises(RuntimeError, match=r"(?i)attach"):
            _ = result.words
    finally:
        # Restore so subsequent tests using this result are unaffected
        result.embeddings = original_emb


# ---------------------------------------------------------------------------
# Invariant 8: Pickle round-trip preserves stats and cache
# ---------------------------------------------------------------------------

def test_pickle_round_trip_preserves_state(pls_result):
    """pickle.loads(pickle.dumps(result)) preserves r2, n_kept, and cache contents."""
    # Pre-populate cache with a sentinel
    sentinel = object.__new__(object)
    pls_result._cache[("_pickle_test_key", ())] = sentinel

    data = pickle.dumps(pls_result)
    restored = pickle.loads(data)

    # Core stats preserved
    assert math.isfinite(restored.stats.r2)
    assert restored.stats.r2 == pls_result.stats.r2
    assert restored.stats.n_kept == pls_result.stats.n_kept

    # Cache entry survived the round-trip
    assert ("_pickle_test_key", ()) in restored._cache

    # Cleanup sentinel from live object
    del pls_result._cache[("_pickle_test_key", ())]


# ---------------------------------------------------------------------------
# Invariant 9: Every @dataclass in schema.py is frozen (mutation raises)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls,kwargs", [
    (Word,    dict(side="pos", rank=1, word="test", cos_beta=0.5)),
    (Cluster, dict(cluster_id=0, side="pos", size=3, coherence=0.7, centroid_cos_beta=0.6)),
    (ClusterWord, dict(cluster_id=0, side="pos", word="test", cos_centroid=0.8, cos_beta=0.6)),
    (Snippet, dict(snippet_id=0, side="pos", doc_id=0, cosine=0.5, seed="test",
                   start_token_idx=0, end_token_idx=2, start_sent_idx=0, end_sent_idx=0,
                   text_window="w", text_surface="s", text_lemmas="l")),
    (Doc,     dict(doc_id=0, y_true=1.0, y_hat=0.9, residual=0.1, alignment_score=0.5)),
    (FitInfo, dict()),
    (Stats,   dict(backend="PLS", r2=0.5, pvalue=0.05, n_raw=10, n_kept=9, n_dropped=1,
                   y_mean=1.0, y_std=0.5, beta_norm=1.0, delta=0.1,
                   iqr_effect=0.3, y_corr_pred=0.7)),
])
def test_schema_dataclasses_are_frozen(cls, kwargs):
    """Every schema dataclass is frozen — mutation raises FrozenInstanceError."""
    obj = cls(**kwargs)
    first_field = next(iter(obj.__dataclass_fields__))
    with pytest.raises(FrozenInstanceError):
        setattr(obj, first_field, None)


# ---------------------------------------------------------------------------
# Invariant 10: Snippet.cluster_id is Optional[int] — constructible with None
# ---------------------------------------------------------------------------

def test_snippet_cluster_id_optional():
    """Snippet can be constructed with cluster_id=None."""
    s = Snippet(
        snippet_id=0, side="neg", doc_id=1, cosine=0.3, seed="word",
        start_token_idx=0, end_token_idx=1, start_sent_idx=0, end_sent_idx=0,
        text_window="w", text_surface="s", text_lemmas="l",
        cluster_id=None,
    )
    assert s.cluster_id is None


# ---------------------------------------------------------------------------
# Invariant 11: alignment_scores[i] == cos(x[i], beta)
# ---------------------------------------------------------------------------

def test_alignment_scores_formula():
    """alignment_scores[i] == cosine(x[i], beta) computed from first principles."""
    from ssdiff.results.single_result import _SingleResult
    from ssdiff.utils.math import unit_vector

    rng = np.random.default_rng(123)
    dim = 4
    n = 5
    x = rng.normal(size=(n, dim)).astype(float)
    beta = rng.normal(size=(dim,)).astype(float)

    # Minimal _SingleResult subclass — no corpus/embeddings
    class _MinSingle(_SingleResult):
        pass

    result = _MinSingle(x=x, beta=beta)

    gradient = unit_vector(beta)
    x_norms = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    expected = ((x / x_norms) @ gradient).ravel()

    actual = result.alignment_scores
    np.testing.assert_allclose(actual, expected, rtol=1e-6,
                                err_msg="alignment_scores must equal cos(x_i, gradient)")


# ---------------------------------------------------------------------------
# Invariant 12: beta_norm == np.linalg.norm(beta)
# ---------------------------------------------------------------------------

def test_beta_norm_equals_norm_of_beta(pls_result):
    """pls_result.beta_norm equals np.linalg.norm(pls_result.beta) exactly."""
    expected = float(np.linalg.norm(pls_result.beta))
    assert pls_result.beta_norm == expected, \
        f"beta_norm {pls_result.beta_norm} != norm(beta) {expected}"


# ---------------------------------------------------------------------------
# Invariant 13: LexiconResult — effect-size field (corr) in [0, 1]
# ---------------------------------------------------------------------------

def test_lexicon_result_effect_size_in_range():
    """LexiconResult suggestions have corr (effect size) in [0, 1] and finite."""
    lr = _make_lexicon_result()
    for sug in lr.suggestions:
        assert math.isfinite(sug.corr), f"corr not finite for token {sug.token!r}"
        assert 0.0 <= sug.corr <= 1.0, f"corr={sug.corr} out of [0,1] for token {sug.token!r}"


# ---------------------------------------------------------------------------
# Invariant 14: LexiconResult.suggestions returns tokens in effect-size-descending order
# ---------------------------------------------------------------------------

def test_lexicon_suggestions_order_descending_by_rank():
    """suggestions iterates tokens in rank-ascending (effect-size-descending) order."""
    # The fixture sets rank: tok_a=0.1 < tok_b=0.3 < tok_c=0.7 (lower rank = better)
    lr = _make_lexicon_result()
    tokens = [sug.token for sug in lr.suggestions]
    assert tokens == ["tok_a", "tok_b", "tok_c"], \
        f"Suggestions not in rank-ascending order: {tokens}"

    # Also verify corr (effect size) is non-increasing
    corrs = [sug.corr for sug in lr.suggestions]
    for i in range(len(corrs) - 1):
        assert corrs[i] >= corrs[i + 1], \
            f"corr not descending: corrs[{i}]={corrs[i]} < corrs[{i+1}]={corrs[i+1]}"


# ---------------------------------------------------------------------------
# Invariant 15: LexiconResult.report().to_text() contains candidate token names
# ---------------------------------------------------------------------------

def test_lexicon_report_to_text_contains_token_names():
    """report().to_text() includes each suggestion token name from the fixture."""
    lr = _make_lexicon_result()
    text = lr.report().to_text()
    for token in ["tok_a", "tok_b", "tok_c"]:
        assert token in text, f"Token {token!r} not found in report text"
