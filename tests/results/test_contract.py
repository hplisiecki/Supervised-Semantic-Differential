"""Structural contract tests for ssdiff.results.

Covers core.py (cache API, View, ScalarView), schema.py (frozen dataclasses),
single_result.py (_SingleResult invariants), and lexicon_result.py.

One test per invariant.
"""

from __future__ import annotations

import dataclasses
import inspect
import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from ssdiff.results import schema
from ssdiff.results.core import Result, ScalarView, View
from ssdiff.results.lexicon_result import LexiconResult
from ssdiff.results.schema import Cluster, Suggestion
from ssdiff.ssd import SSD

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
    """Minimal concrete ScalarView for ScalarView tests."""

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
    # higher rank = better (combined score): rank is set so tok_a > tok_b > tok_c
    suggestions = [
        Suggestion(token="tok_a", freq=10, cov_all=0.5, cov_bal=0.4,
                   corr=0.8, pvalue=0.01, direction="positive", rank=0.7),
        Suggestion(token="tok_b", freq=8,  cov_all=0.4, cov_bal=0.3,
                   corr=0.5, pvalue=0.05, direction="positive", rank=0.3),
        Suggestion(token="tok_c", freq=5,  cov_all=0.3, cov_bal=0.2,
                   corr=0.2, pvalue=0.2,  direction="none",     rank=0.1),
    ]
    return LexiconResult(
        var_type="categorical",
        n_docs=100,
        n_tokens=3,
        suggestions=suggestions,
    )


# ---------------------------------------------------------------------------
# Stale-cache regression — different params coexist independently
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
# clear_cache("words") removes only "words" entries
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
# ScalarView attr-access and dict-access return same values
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
# attach(embeddings=...) wires _embeddings attribute correctly
# ---------------------------------------------------------------------------

def test_attach_embeddings_at_other_abtt_raises(pls_result):
    """Attaching Embeddings at a different ABTT than the fit's is refused."""
    from tests.conftest import make_kv
    other = make_kv(["a", "b"], dim=4, seed=7)
    other.abtt = pls_result.embeddings.abtt + 1
    before = pls_result.embeddings
    with pytest.raises(ValueError, match="abtt"):
        pls_result.attach(embeddings=other)
    assert pls_result.embeddings is before


def test_attach_texts_count_must_match(pls_result):
    with pytest.raises(ValueError, match="documents"):
        pls_result.attach(texts=["people work."])


# ---------------------------------------------------------------------------
# result.words before attach raises RuntimeError mentioning "attach"
# ---------------------------------------------------------------------------

def test_words_without_embeddings_raises_runtime_error(ssd_instance, pack_df):
    """Accessing .words when embeddings=None raises RuntimeError with 'attach'.

    Note: Result.attach(embeddings=None) is a no-op (guarded by `if embeddings is not None`).
    To simulate the un-attached state we directly set the attribute and clear the cache.
    """
    result = ssd_instance.fit_pls(pack_df["y"], k=1, n_splits=20, random_state=0)

    # Directly strip embeddings to simulate un-attached state
    result.embeddings = None
    result.clear_cache("words")

    with pytest.raises(RuntimeError, match=r"(?i)attach"):
        _ = result.words


# ---------------------------------------------------------------------------
# Pickle round-trip preserves stats and cache
# ---------------------------------------------------------------------------

def test_pickle_round_trip_preserves_state(pls_result):
    """pickle.loads(pickle.dumps(result)) preserves stats, beta, and the cached views."""
    words = list(pls_result.words)
    restored = pickle.loads(pickle.dumps(pls_result))

    assert restored.embeddings is None
    assert restored.stats.r2 == pls_result.stats.r2
    assert restored.stats.n_kept == pls_result.stats.n_kept
    np.testing.assert_array_equal(restored.beta, pls_result.beta)
    # embeddings are gone, so the words can only come from the pickled cache
    assert list(restored.words) == words


# ---------------------------------------------------------------------------
# Every @dataclass in schema.py is frozen (mutation raises)
# ---------------------------------------------------------------------------

def test_schema_dataclasses_are_frozen():
    """Every schema dataclass is frozen — mutation raises FrozenInstanceError."""
    classes = [
        c for _, c in inspect.getmembers(schema, inspect.isclass)
        if dataclasses.is_dataclass(c) and c.__module__ == schema.__name__
    ]
    assert len(classes) >= 10
    for cls in classes:
        with pytest.raises(FrozenInstanceError):
            setattr(cls.__new__(cls), dataclasses.fields(cls)[0].name, None)


# ---------------------------------------------------------------------------
# alignment_scores[i] == cos(x[i], beta)
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

    assert result.beta_norm == pytest.approx(np.linalg.norm(beta))

    actual = result.alignment_scores
    np.testing.assert_allclose(actual, expected, rtol=1e-6,
                                err_msg="alignment_scores must equal cos(x_i, gradient)")


# ---------------------------------------------------------------------------
# LexiconResult.report().to_text() contains candidate token names
# ---------------------------------------------------------------------------

def test_lexicon_report_to_text_contains_token_names():
    """report().to_text() includes each suggestion token name from the fixture."""
    lr = _make_lexicon_result()
    text = lr.report().to_text()
    for token in ["tok_a", "tok_b", "tok_c"]:
        assert token in text, f"Token {token!r} not found in report text"


# ---------------------------------------------------------------------------
# LexiconResult with seeds: report()
# ---------------------------------------------------------------------------

def _make_seeded_lexicon_result() -> LexiconResult:
    """Six tokens of two seeds in one table, highest cosine first."""
    rows = [
        ("one", "one_a", 1.0), ("two", "two_a", 1.0), ("one", "one_b", 0.8),
        ("two", "two_b", 0.7), ("one", "one_c", 0.6), ("two", "two_c", 0.5),
    ]
    suggestions = [
        Suggestion(token=token, freq=5, cov_all=0.3, cov_bal=0.2, corr=0.1,
                   pvalue=0.5, direction="none", rank=0.2, seed=seed, cos=cos)
        for seed, token, cos in rows
    ]
    return LexiconResult(
        var_type="continuous", n_docs=100, n_tokens=6, suggestions=suggestions,
        seeded=True,
    )


def test_suggestion_pickled_before_the_seed_fields_prints():
    """A Suggestion restored from a state without seed and cos has both as None."""
    old = Suggestion.__new__(Suggestion)
    old.__setstate__(["tok_a", 10, 0.5, 0.4, 0.8, 0.01, "positive", 0.1])
    assert (old.seed, old.cos) == (None, None)
    lr = LexiconResult(var_type="continuous", n_docs=100, n_tokens=1, suggestions=[old])
    assert "tok_a" in lr.suggestions.to_text()


def test_cluster_pickled_before_the_top_fields_has_empty_strings():
    """A Cluster restored from a state without top_words and top_snippet has both as ""."""
    old = Cluster.__new__(Cluster)
    old.__setstate__([0, "pos", 3, 0.5, 0.4, None])
    assert (old.top_words, old.top_snippet) == ("", "")


def test_seeded_lexicon_report_lists_the_first_n_rows():
    text = _make_seeded_lexicon_result().report(top={"n": 3}).to_text()
    for token in ["one_a", "two_a", "one_b"]:
        assert token in text, f"Token {token!r} not found in report text"
    for token in ["two_b", "one_c", "two_c"]:
        assert token not in text, f"Token {token!r} is past the first n rows"


def test_seeded_lexicon_report_has_seed_and_cos_columns():
    text = _make_seeded_lexicon_result().report().to_text()
    header = next(line for line in text.splitlines() if "cov_all" in line).split()
    assert header[:3] == ["seed", "token", "cos"]


# ---------------------------------------------------------------------------
# docs view and counts when documents are dropped
# ---------------------------------------------------------------------------

def _pack_text_with_empty_last(pack_emb, pack_df, pack_lexicon, n=12):
    """First ``n`` test-pack texts plus one whose words are all out of vocabulary."""
    texts = list(pack_df["text"][:n]) + ["zzzqqq"]
    return SSD(texts, embeddings=pack_emb, lexicon=pack_lexicon), texts


def test_docs_doc_id_and_text_match_text_row_with_nan_y(
    pack_emb, pack_df, pack_lexicon
):
    """A missing y and a document without a vector both leave later doc_ids on their text rows."""
    ssd, texts = _pack_text_with_empty_last(pack_emb, pack_df, pack_lexicon)
    y = np.append(pack_df["y"].to_numpy()[:12], 1.0)
    y[3] = np.nan
    res = ssd.fit_pls(y, k=2, n_splits=20, random_state=42)

    expected_rows = [i for i in range(len(texts)) if i not in (3, len(texts) - 1)]
    assert [d.doc_id for d in res.docs] == expected_rows
    for row in expected_rows:
        assert res.docs.id(row)._raw_text == texts[row]
    assert (res.stats.n_raw, res.stats.n_kept, res.stats.n_dropped) == (13, 11, 2)
    assert res._keep_mask.shape == (len(texts),)


def test_counts_span_text_with_missing_group_label(
    tiny_kv, large_docs, large_groups_2, lexicon
):
    """Documents with a None label are dropped from the counts."""
    groups = large_groups_2.copy()
    groups[2] = None
    res = SSD(large_docs, embeddings=tiny_kv, lexicon=lexicon).fit_groups(groups, n_perm=10, random_state=0)

    assert res.stats.n_kept == 49
