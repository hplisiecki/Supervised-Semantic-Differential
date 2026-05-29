"""Tests for View.save(), ScalarView.save(), and Report.save() / Report.to_text().

Consolidates 5 old IO test files. Tests all 8 tabular extensions (csv, json,
xlsx, md, txt, html, tex, docx) and 6 narrative Report extensions (md, txt,
html, tex, docx, json). Optional deps (openpyxl, python-docx) are
conditionally skipped via pytest.importorskip.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ssdiff.results.display import NARRATIVE_EXTS, TABULAR_EXTS
from ssdiff.results.report import Report, Section

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _words_view(pls_result):
    """Return the words tabular View from pls_result."""
    return pls_result.words


def _stats_view(pls_result):
    """Return the stats ScalarView from pls_result."""
    return pls_result.stats


def _minimal_report() -> Report:
    """Minimal Report with one kv section for format tests."""
    return Report(
        title="Test Report",
        sections=[
            Section(
                title="Model Summary",
                kind="kv",
                rows=[("r2", "0.42"), ("pvalue", "0.01")],
            )
        ],
    )


# ---------------------------------------------------------------------------
# Invariant 1 – parametrize over all supported tabular extensions
# ---------------------------------------------------------------------------

# Extensions that require optional deps; guarded with importorskip inside test
_OPTIONAL_TABULAR = {"xlsx": "openpyxl", "docx": "docx"}


@pytest.mark.parametrize("ext", TABULAR_EXTS)
def test_view_save_format_exists_nonempty(pls_result, tmp_path, ext):
    """Save a WordsView to each supported tabular extension.

    File must exist and be non-empty after save().
    """
    if ext in _OPTIONAL_TABULAR:
        pytest.importorskip(_OPTIONAL_TABULAR[ext])

    view = _words_view(pls_result)
    out = tmp_path / f"words.{ext}"
    view.save(str(out))

    assert out.exists(), f"save() did not create file for .{ext}"
    assert out.stat().st_size > 0, f"save() wrote empty file for .{ext}"


def test_view_save_csv_roundtrip(pls_result, tmp_path):
    """CSV save: read back and verify column names and at least one row value."""
    view = _words_view(pls_result)
    out = tmp_path / "words.csv"
    view.save(str(out))

    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) >= 1, "CSV has no data rows"
    # Default cols for WordsView: side, rank, word, cos_beta
    assert "word" in rows[0], "Expected 'word' column in CSV output"
    assert "side" in rows[0], "Expected 'side' column in CSV output"
    # At least one row should have a non-empty word value
    assert any(r["word"] for r in rows), "All 'word' values are empty"


# ---------------------------------------------------------------------------
# Invariant 2 – save(path=None) creates file in cwd using default name
# ---------------------------------------------------------------------------

def test_view_save_default_cwd(pls_result, tmp_path, monkeypatch):
    """save(path=None) creates a file in cwd with the default filename."""
    monkeypatch.chdir(tmp_path)
    view = _words_view(pls_result)
    view.save()  # path=None

    default_name = view._default_filename()
    out = tmp_path / default_name
    assert out.exists(), (
        f"save(None) did not create {default_name!r} in cwd"
    )
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Invariant 3 – save(Path("out.csv")) accepts a Path object
# ---------------------------------------------------------------------------

def test_view_save_accepts_path_object(pls_result, tmp_path):
    """save() must accept a pathlib.Path argument (not just str)."""
    view = _words_view(pls_result)
    out = tmp_path / "out.csv"
    view.save(out)  # passing a Path, not a str

    assert out.exists()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Invariant 4 – save("out") with no extension raises ValueError
# ---------------------------------------------------------------------------

def test_view_save_no_extension_raises(pls_result, tmp_path):
    """save(path_without_extension) must raise ValueError."""
    view = _words_view(pls_result)
    with pytest.raises(ValueError, match=r"extension"):
        view.save(str(tmp_path / "out"))


# ---------------------------------------------------------------------------
# Invariant 5 – save("out.xyz") with unknown extension raises ValueError
# ---------------------------------------------------------------------------

def test_view_save_unknown_extension_raises(pls_result, tmp_path):
    """save(path_with_unknown_extension) must raise ValueError."""
    view = _words_view(pls_result)
    with pytest.raises(ValueError, match=r"[Uu]nsupported"):
        view.save(str(tmp_path / "out.xyz"))


# ---------------------------------------------------------------------------
# Invariant 6 – save("out.csv", k=3) produces exactly 3 data rows
# ---------------------------------------------------------------------------

def test_view_save_k_caps_rows(pls_result, tmp_path):
    """save(path, k=3) must write exactly 3 data rows (not counting header)."""
    view = _words_view(pls_result)
    out = tmp_path / "words_k3.csv"
    view.save(str(out), k=3)

    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3, f"Expected 3 rows with k=3, got {len(rows)}"


# ---------------------------------------------------------------------------
# Invariant 6b – save(path, k=N) contract holds for EVERY row-bearing view
# ---------------------------------------------------------------------------
# Regression for an old gap: only WordsView was tested with k=N, so a broken
# __getitem__-on-slice in any sibling view (e.g. ClustersView returning a
# bare list) only surfaced when a downstream caller exercised that path.

_ROW_VIEW_GETTERS = [
    ("words",                 lambda r: r.words),
    ("words.pos",             lambda r: r.words.pos),
    ("words.neg",             lambda r: r.words.neg),
    ("clusters",              lambda r: r.clusters),
    ("clusters.pos",          lambda r: r.clusters.pos),
    ("clusters.neg",          lambda r: r.clusters.neg),
    ("clusters.words",        lambda r: r.clusters.words),
    ("clusters.pos.words",    lambda r: r.clusters.pos.words),
    ("clusters.neg.words",    lambda r: r.clusters.neg.words),
    ("snippets",              lambda r: r.snippets),
    ("snippets.pos",          lambda r: r.snippets.pos),
    ("snippets.neg",          lambda r: r.snippets.neg),
    ("docs",                  lambda r: r.docs),
    ("docs.pos()",            lambda r: r.docs.pos()),
    ("docs.neg()",            lambda r: r.docs.neg()),
]


@pytest.mark.parametrize("label,getter", _ROW_VIEW_GETTERS,
                         ids=[lbl for lbl, _ in _ROW_VIEW_GETTERS])
def test_view_save_k_caps_rows_all_views(pls_result, tmp_path, label, getter):
    """save(path, k=N) on any row-bearing view: no crash, ≤ N data rows.

    Contract is per the View base class, not per-call-site. Every concrete
    subclass must honor it — otherwise the bug only shows in whichever
    caller happens to pass ``k=N``.
    """
    view = getter(pls_result)
    if len(view) == 0:
        pytest.skip(f"{label}: view is empty on this fixture")
    k = 2
    out = tmp_path / f"{label.replace('.', '_').replace('(', '').replace(')', '')}.csv"
    view.save(str(out), k=k)

    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) <= k, (
        f"{label}: save(k={k}) wrote {len(rows)} data rows; expected ≤ {k}"
    )


@pytest.mark.parametrize("label,getter", _ROW_VIEW_GETTERS,
                         ids=[lbl for lbl, _ in _ROW_VIEW_GETTERS])
def test_view_slice_returns_view_all_views(pls_result, label, getter):
    """view[:k] must return something with ``_columns`` (i.e. a View, not list).

    ``View._resized(k)`` (used by ``save(k=N)``) delegates to ``self[:k]``,
    then ``_validate_cols`` reads ``view._columns``. A bare list slips through
    every type hint and only crashes at write time.
    """
    view = getter(pls_result)
    sliced = view[:2]
    assert hasattr(sliced, "_columns"), (
        f"{label}: view[:k] returned {type(sliced).__name__} "
        f"(must be a View with ``_columns``)"
    )


# ---------------------------------------------------------------------------
# Invariant 7 – ScalarView.save("out.csv", k=0) is a silent no-op or pins
# ---------------------------------------------------------------------------

def test_scalar_view_save_k_zero(pls_result, tmp_path):
    """ScalarView.save(..., k=0) should not raise; result is 0 or 1 data rows.

    The source uses _resized(k) which slices; for ScalarView the slice
    returns self (single row), so k=0 is treated as 'full single row'
    (the view always has exactly 1 row). We accept 0 or 1 rows — the
    important thing is no exception is raised.
    """
    view = _stats_view(pls_result)
    out = tmp_path / "stats_k0.csv"
    view.save(str(out), k=0)  # must not raise

    assert out.exists()
    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # ScalarView always iterates exactly 1 item regardless of k
    assert len(rows) in (0, 1), f"Unexpected row count: {len(rows)}"


# ---------------------------------------------------------------------------
# Invariant 9 – Report.save("out.md") starts with "# " (ATX header)
# ---------------------------------------------------------------------------

def test_report_save_md_starts_with_atx_header(tmp_path):
    """Markdown output must start with an ATX-style H1 header ('# ')."""
    report = _minimal_report()
    out = tmp_path / "out.md"
    report.save(str(out))

    content = out.read_text(encoding="utf-8")
    assert content.startswith("# "), (
        f"Markdown file does not start with '# '; starts with: {content[:30]!r}"
    )


# ---------------------------------------------------------------------------
# Invariant 10 – Report.save("out.xyz") raises ValueError
# ---------------------------------------------------------------------------

def test_report_save_unknown_extension_raises(tmp_path):
    """Report.save() with an unsupported extension must raise ValueError."""
    report = _minimal_report()
    with pytest.raises(ValueError, match=r"[Uu]nsupported"):
        report.save(str(tmp_path / "out.xyz"))


# ---------------------------------------------------------------------------
# Bonus: Report.save() round-trips for non-binary narrative formats
# ---------------------------------------------------------------------------

_OPTIONAL_NARRATIVE = {"docx": "docx"}

@pytest.mark.parametrize("ext", [e for e in NARRATIVE_EXTS if e != "docx"])
def test_report_save_narrative_exists_nonempty(tmp_path, ext):
    """Report.save() creates a non-empty file for each text narrative format."""
    report = _minimal_report()
    out = tmp_path / f"report.{ext}"
    report.save(str(out))

    assert out.exists(), f"Report.save() did not create file for .{ext}"
    assert out.stat().st_size > 0, f"Report.save() wrote empty file for .{ext}"


def test_report_save_docx_exists_nonempty(tmp_path):
    """Report.save() creates a non-empty .docx if python-docx is available."""
    pytest.importorskip("docx")
    report = _minimal_report()
    out = tmp_path / "report.docx"
    report.save(str(out))

    assert out.exists()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Bonus: JSON narrative round-trip preserves title
# ---------------------------------------------------------------------------

def test_report_save_json_roundtrip(tmp_path):
    """Report.save('out.json') must produce valid JSON with correct title."""
    report = _minimal_report()
    out = tmp_path / "report.json"
    report.save(str(out))

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["title"] == report.title
    assert "sections" in data
