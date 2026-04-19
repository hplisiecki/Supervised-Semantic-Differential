"""Default column & max-rows resolution (see docs/results_defaults_spec.md)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ssdiff.results import core as _core
from ssdiff.results.core import (
    DEFAULT_COLS,
    DEFAULT_MAX_ROWS,
    View,
)

# ---------------- fixtures ---------------------------------------------------

@dataclass(frozen=True, slots=True)
class _Row:
    id: int
    name: str
    x: float
    flag: bool


class _RegisteredView(View[_Row]):
    """Has a DEFAULT_COLS entry."""
    _name = "registered"
    _columns = ("id", "name", "x", "flag")

    def __init__(self, rows, *, _no_trunc: bool = False):
        super().__init__(_no_trunc=_no_trunc)
        self._rows = rows

    def __iter__(self): return iter(self._rows)
    def __len__(self): return len(self._rows)
    def __getitem__(self, i):
        if isinstance(i, slice):
            return type(self)(self._rows[i], _no_trunc=True)
        return self._rows[i]


class _UnregisteredView(_RegisteredView):
    """No DEFAULT_COLS entry — must fall through to full _columns."""
    _name = "unregistered"


@pytest.fixture(autouse=True)
def _register_default():
    DEFAULT_COLS["_RegisteredView"] = ("id", "name")
    yield
    DEFAULT_COLS.pop("_RegisteredView", None)


@pytest.fixture
def rows():
    return [_Row(i, f"r{i}", 0.1 * i, bool(i % 2)) for i in range(50)]


# ---------------- cols=None resolution ---------------------------------------

def test_default_cols_returns_registry_entry(rows):
    v = _RegisteredView(rows)
    assert v._default_cols() == ("id", "name")


def test_default_cols_falls_through_to_full_columns(rows):
    v = _UnregisteredView(rows)
    assert v._default_cols() == v._columns


def test_to_dict_uses_defaults(rows):
    v = _RegisteredView(rows[:2])
    assert v.to_dict() == [{"id": 0, "name": "r0"}, {"id": 1, "name": "r1"}]


def test_to_records_uses_defaults(rows):
    v = _RegisteredView(rows[:2])
    assert v.to_records() == [(0, "r0"), (1, "r1")]


# ---------------- cols="all" escape ------------------------------------------

def test_cols_all_returns_full_columns(rows):
    v = _RegisteredView(rows[:1])
    out = v.to_dict(cols="all")
    assert out == [{"id": 0, "name": "r0", "x": 0.0, "flag": False}]


def test_save_csv_uses_defaults(rows, tmp_path):
    import csv
    p = tmp_path / "x.csv"
    _RegisteredView(rows[:2]).save(str(p))
    read = list(csv.reader(p.read_text().splitlines()))
    assert read[0] == ["id", "name"]


def test_save_csv_cols_all_dumps_everything(rows, tmp_path):
    import csv
    p = tmp_path / "x.csv"
    _RegisteredView(rows[:2]).save(str(p), cols="all")
    read = list(csv.reader(p.read_text().splitlines()))
    assert read[0] == ["id", "name", "x", "flag"]


# ---------------- unknown cols still warn / fall through ---------------------

def test_unknown_cols_still_warns(rows):
    v = _RegisteredView(rows[:1])
    with pytest.warns(UserWarning, match="unknown cols ignored"):
        out = v.to_dict(cols=["name", "bogus"])
    assert out == [{"name": "r0"}]


def test_all_unknown_cols_falls_through_full(rows):
    v = _RegisteredView(rows[:1])
    with pytest.warns(UserWarning, match="all cols unknown"):
        out = v.to_dict(cols=["bogus"])
    # Falls back to full _columns (not narrowed defaults) per existing behaviour.
    assert list(out[0].keys()) == ["id", "name", "x", "flag"]


# ---------------- max_rows resolution ----------------------------------------

def test_to_text_max_rows_none_uses_module_default(rows):
    v = _RegisteredView(rows)  # 50 rows
    text = v.to_text()
    assert f"... {len(rows) - DEFAULT_MAX_ROWS} more rows" in text


def test_to_text_explicit_max_rows_overrides(rows):
    v = _RegisteredView(rows)
    text = v.to_text(max_rows=5)
    assert f"... {len(rows) - 5} more rows" in text


def test_module_level_default_max_rows_is_runtime(rows, monkeypatch):
    """Mutating DEFAULT_MAX_ROWS at module level affects the next call.

    Confirms runtime resolution (A-style) rather than def-time baking (B-style).
    """
    monkeypatch.setattr(_core, "DEFAULT_MAX_ROWS", 7)
    v = _RegisteredView(rows)
    text = v.to_text()
    assert f"... {len(rows) - 7} more rows" in text


# ---------------- real-view smoke tests --------------------------------------

def test_snippets_view_repr_narrows_from_15_to_default_cols():
    """Spec headline example: SnippetsView drops from 15 to 5 columns."""
    from ssdiff.results.continuous_result import SnippetsView
    from ssdiff.results.schema import Snippet
    rows = [
        Snippet(snippet_id=0, side="pos", doc_id=1, cosine=0.5, seed="kw",
                start_token_idx=0, end_token_idx=3,
                start_sent_idx=0, end_sent_idx=1,
                text_window="window here", text_surface="surface text",
                text_lemmas="lemma text",
                cluster_id=None, contrast=None, post_id=None),
    ]
    text = repr(SnippetsView(rows))
    body = text.split("Save:")[0]
    for k in ("side", "doc_id", "cosine", "seed", "text_window"):
        assert k in body
    # Dropped columns must not appear in the default body
    assert "text_surface" not in body
    assert "text_lemmas" not in body
    assert "start_token_idx" not in body


def test_words_view_drops_contrast_by_default():
    from ssdiff.results.continuous_result import WordsView
    from ssdiff.results.schema import Word
    rows = [Word(side="pos", rank=0, word="a", cos_beta=0.5, contrast=None),
            Word(side="neg", rank=0, word="b", cos_beta=-0.4, contrast=None)]
    v = WordsView(rows)
    assert v.to_dict() == [
        {"side": "pos", "rank": 0, "word": "a", "cos_beta": 0.5},
        {"side": "neg", "rank": 0, "word": "b", "cos_beta": -0.4},
    ]
    # cols="all" brings contrast back
    full = v.to_dict(cols="all")
    assert "contrast" in full[0]


def test_stats_and_ols_stats_diverge_by_class_name():
    """StatsView and OLSStatsView share _name='stats' but have different defaults."""
    assert DEFAULT_COLS["StatsView"] != DEFAULT_COLS["OLSStatsView"]
    assert "r2_adj" in DEFAULT_COLS["OLSStatsView"]
    assert "r2_adj" not in DEFAULT_COLS["StatsView"]
