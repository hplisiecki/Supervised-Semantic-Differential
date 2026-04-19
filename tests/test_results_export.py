"""Export family — D8 (one signature) and D9 (forgiving `cols`)."""

import csv
import json
import warnings
from dataclasses import dataclass

import pytest

from ssdiff.results.core import ScalarView, View, _validate_cols


@dataclass(frozen=True, slots=True)
class _Row:
    id: int
    name: str
    x: float


class _ToyView(View[_Row]):
    _name = "toy"
    _columns = ("id", "name", "x")

    def __init__(self, rows):
        super().__init__()
        self._rows = rows

    def __iter__(self): return iter(self._rows)
    def __len__(self): return len(self._rows)
    def __getitem__(self, i):
        if isinstance(i, slice):
            return type(self)(self._rows[i])
        return self._rows[i]


@pytest.fixture
def view():
    return _ToyView([_Row(0, "a", 0.1), _Row(1, "b", 0.2)])


# ---------------- cols validation (D9) ----------------
def test_validate_cols_none_returns_all(view):
    # _ToyView has no DEFAULT_COLS entry → falls through to full _columns.
    keep, warning = _validate_cols(None, view)
    assert keep == ("id", "name", "x")
    assert warning is None


def test_validate_cols_all_valid(view):
    keep, warning = _validate_cols(["name", "x"], view)
    assert keep == ("name", "x")
    assert warning is None


def test_validate_cols_partial_unknown(view):
    keep, warning = _validate_cols(["id", "missing", "x"], view)
    assert keep == ("id", "x")
    assert warning is not None
    assert "missing" in warning


def test_validate_cols_all_unknown(view):
    keep, warning = _validate_cols(["foo", "bar"], view)
    assert keep == ("id", "name", "x")
    assert warning is not None
    assert "all cols unknown" in warning.lower()


# ---------------- to_dict / to_records (no pandas) ----------------
def test_to_dict_returns_list_of_dicts(view):
    rows = view.to_dict()
    assert rows == [
        {"id": 0, "name": "a", "x": 0.1},
        {"id": 1, "name": "b", "x": 0.2},
    ]


def test_to_dict_honors_cols(view):
    rows = view.to_dict(cols=["name", "x"])
    assert rows == [{"name": "a", "x": 0.1}, {"name": "b", "x": 0.2}]


def test_to_records_returns_list_of_tuples(view):
    recs = view.to_records()
    assert recs == [(0, "a", 0.1), (1, "b", 0.2)]


# ---------------- save() → CSV ----------------
def test_save_csv_writes_file(view, tmp_path):
    p = tmp_path / "out.csv"
    result = view.save(str(p))
    assert result is None
    text = p.read_text(encoding="utf-8")
    rows = list(csv.reader(text.splitlines()))
    assert rows[0] == ["id", "name", "x"]
    assert rows[1] == ["0", "a", "0.1"]


# ---------------- save() → JSON ----------------
def test_save_json_writes_file(view, tmp_path):
    p = tmp_path / "out.json"
    view.save(str(p))
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data == [
        {"id": 0, "name": "a", "x": 0.1},
        {"id": 1, "name": "b", "x": 0.2},
    ]


# ---------------- cols partial-unknown warning behaviour ----------------
def test_save_csv_partial_unknown_warns(view, tmp_path):
    p = tmp_path / "out.csv"
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        view.save(str(p), cols=["name", "nope", "x"])
    assert any(issubclass(w.category, UserWarning) for w in ws)
    header = p.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == ["name", "x"]


def test_save_csv_all_unknown_warns_and_falls_back(view, tmp_path):
    p = tmp_path / "out.csv"
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        view.save(str(p), cols=["nope1", "nope2"])
    assert any(issubclass(w.category, UserWarning) for w in ws)
    header = p.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == ["id", "name", "x"]


# ---------------- pandas-gated formats ----------------
def test_to_df_raises_clear_error_without_pandas(monkeypatch, view):
    """If pandas import fails, we want a crisp ImportError, not a stacktrace."""
    import ssdiff.results.core as base
    monkeypatch.setattr(base, "_import_pandas", lambda: (_ for _ in ()).throw(
        ImportError("pandas required; install with: pip install ssdiff[results]")
    ))
    with pytest.raises(ImportError, match=r"ssdiff\[results\]"):
        view.to_df()


def test_to_df_happy_path(view):
    pytest.importorskip("pandas")
    df = view.to_df()
    assert list(df.columns) == ["id", "name", "x"]
    assert len(df) == 2


def test_to_df_cols_reorders(view):
    pytest.importorskip("pandas")
    df = view.to_df(cols=["x", "name"])
    assert list(df.columns) == ["x", "name"]


def test_to_html_returns_table(view):
    s = view.to_html()
    assert "<table" in s
    assert "<th>id</th>" in s


# ---------------- save() → extension dispatch ------------
def test_save_md_writes_pipe_table(view, tmp_path):
    p = tmp_path / "out.md"
    view.save(str(p))
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "| id | name | x |"
    assert lines[1] == "|---|---|---|"
    assert "| 0 | a | 0.1 |" in lines


def test_save_tex_writes_tabular(view, tmp_path):
    p = tmp_path / "out.tex"
    view.save(str(p))
    text = p.read_text(encoding="utf-8")
    assert r"\begin{tabular}" in text
    assert r"\toprule" in text
    assert r"\bottomrule" in text
    assert r"\end{tabular}" in text


def test_save_txt_writes_aligned_table(view, tmp_path):
    p = tmp_path / "out.txt"
    view.save(str(p))
    text = p.read_text(encoding="utf-8")
    # header columns appear; body rows appear
    for needle in ("id", "name", "x", "a", "b"):
        assert needle in text


def test_save_html_writes_table(view, tmp_path):
    p = tmp_path / "out.html"
    view.save(str(p))
    text = p.read_text(encoding="utf-8")
    assert "<table" in text
    assert "<th>id</th>" in text
    assert "<td>a</td>" in text


def test_save_xlsx_writes_workbook(view, tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    p = tmp_path / "out.xlsx"
    view.save(str(p))
    assert p.is_file() and p.stat().st_size > 0


def test_save_docx_writes_document(view, tmp_path):
    pytest.importorskip("docx")
    p = tmp_path / "out.docx"
    view.save(str(p))
    assert p.is_file() and p.stat().st_size > 0


# ---------------- save() → error paths -------------------
def test_save_path_without_extension_raises(view, tmp_path):
    with pytest.raises(ValueError, match="extension"):
        view.save(str(tmp_path / "out"))


def test_save_unknown_extension_raises(view, tmp_path):
    with pytest.raises(ValueError, match=r"[Uu]nsupported extension"):
        view.save(str(tmp_path / "out.xyz"))


# ---------------- save(k=...) row cap --------------------
def test_save_k_caps_rows_on_size_bearing_view(view, tmp_path):
    p = tmp_path / "out.csv"
    view.save(str(p), k=1)
    rows = list(csv.reader(p.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == ["id", "name", "x"]
    assert len(rows) == 2  # header + 1 body row
    assert rows[1] == ["0", "a", "0.1"]


def test_save_k_larger_than_length_is_noop(view, tmp_path):
    p = tmp_path / "out.csv"
    view.save(str(p), k=999)
    rows = list(csv.reader(p.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 3  # header + 2 body rows


# ---------------- ScalarView variants ----------------
class _ToyStats(ScalarView):
    _name = "stats"
    _columns = ("r2", "n")

    def __init__(self, r2: float, n: int):
        super().__init__()
        self._row = {"r2": r2, "n": n}

    def __iter__(self): yield self._row
    def to_dict(self, cols=None):
        keep, warning = _validate_cols(cols, self)
        if warning:
            warnings.warn(warning, UserWarning, stacklevel=2)
        return {k: self._row[k] for k in keep}


@pytest.fixture
def scalar():
    return _ToyStats(r2=0.47, n=100)


def test_scalar_save_csv_writes_single_row(scalar, tmp_path):
    p = tmp_path / "stats.csv"
    scalar.save(str(p))
    rows = list(csv.reader(p.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == ["r2", "n"]
    assert rows[1] == ["0.47", "100"]
    assert len(rows) == 2


def test_scalar_save_k_is_silently_ignored(scalar, tmp_path):
    """k= is a no-op on ScalarView (single row); must not raise or drop the row."""
    p = tmp_path / "stats.csv"
    scalar.save(str(p), k=0)
    rows = list(csv.reader(p.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 2  # header + the scalar's one row


