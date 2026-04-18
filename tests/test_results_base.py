"""Result ABC, View / ScalarView protocol, param-keyed cache."""

from dataclasses import dataclass

import pytest

from ssdiff.results.core import Result, ScalarView, View


@dataclass(frozen=True, slots=True)
class _Row:
    id: int
    x: float


class _ToyView(View[_Row]):
    """Smallest concrete View for unit tests."""

    _name = "toy"
    _columns = ("id", "x")

    def __init__(self, rows: list[_Row], params: dict | None = None):
        super().__init__()
        self._rows = rows
        self._params = dict(params or {})

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return self._rows[i]

    @property
    def params(self) -> dict:
        return dict(self._params)


class _ToyResult(Result):
    def __init__(self, seed: int = 0):
        super().__init__()
        self._seed = seed

    def _compute_toy(self, topn: int):
        return _ToyView([_Row(i, float(i * self._seed)) for i in range(topn)],
                        params={"topn": topn})


def test_paramless_view_is_not_cached_entry():
    """Stats/words live as fields, not in the cache dict."""
    r = _ToyResult()
    # paramless views should be set as attributes, not cache entries
    assert r._cache == {}


def test_param_view_cache_hit_on_same_params():
    r = _ToyResult(seed=1)
    v1 = r._cache_get("toy", {"topn": 100}, lambda: r._compute_toy(100))
    v2 = r._cache_get("toy", {"topn": 100}, lambda: r._compute_toy(999))  # lambda NOT called
    assert v1 is v2


def test_param_view_separate_entries_for_different_params():
    """Regression test for the stale-cache bug (spec issue #2)."""
    r = _ToyResult(seed=1)
    v100 = r._cache_get("toy", {"topn": 100}, lambda: r._compute_toy(100))
    v50 = r._cache_get("toy", {"topn": 50}, lambda: r._compute_toy(50))
    assert len(v100) == 100
    assert len(v50) == 50
    # Crucially: v100 was NOT overwritten.
    v100_again = r._cache_get("toy", {"topn": 100}, lambda: r._compute_toy(999))
    assert len(v100_again) == 100


def test_clear_cache_all():
    r = _ToyResult()
    r._cache_get("toy", {"topn": 5}, lambda: r._compute_toy(5))
    r._cache_get("other", {"k": 1}, lambda: _ToyView([]))
    r.clear_cache()
    assert r._cache == {}


def test_clear_cache_by_view():
    r = _ToyResult()
    r._cache_get("toy", {"topn": 5}, lambda: r._compute_toy(5))
    r._cache_get("other", {"k": 1}, lambda: _ToyView([]))
    r.clear_cache("toy")
    assert any(name == "other" for (name, _) in r._cache)
    assert not any(name == "toy" for (name, _) in r._cache)


def test_view_iteration_and_indexing():
    v = _ToyView([_Row(0, 0.0), _Row(1, 1.0)])
    assert len(v) == 2
    assert v[0].id == 0
    assert [row.id for row in v] == [0, 1]


def test_view_columns_attribute_is_ordered_tuple():
    v = _ToyView([])
    assert v.columns == ("id", "x")


class _ToyStats(ScalarView):
    _name = "stats"
    _columns = ("r2", "n")

    def __init__(self, r2: float, n: int):
        super().__init__()
        self._row = {"r2": r2, "n": n}

    def __iter__(self):
        yield self._row

    def to_dict(self) -> dict:
        return dict(self._row)


def test_scalarview_attr_and_dict_access():
    s = _ToyStats(0.47, 1240)
    assert s.r2 == 0.47
    assert s["n"] == 1240
    assert s.to_dict() == {"r2": 0.47, "n": 1240}


def test_scalarview_missing_attr_raises():
    s = _ToyStats(0.47, 1240)
    with pytest.raises(AttributeError):
        _ = s.does_not_exist


def test_attach_raises_before_call_when_resource_missing():
    r = _ToyResult()
    with pytest.raises(RuntimeError, match="attach"):
        r._require_resource("embeddings", "clusters")


def test_attach_sets_and_replaces_references():
    r = _ToyResult()
    obj1, obj2 = object(), object()
    r.attach(embeddings=obj1)
    assert r.embeddings is obj1
    r.attach(embeddings=obj2)
    assert r.embeddings is obj2
