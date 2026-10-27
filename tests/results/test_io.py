"""Tests for View.save(), ScalarView.save(), and Report.save() / Report.to_text().

Tests all 8 tabular extensions (csv, json,
xlsx, md, txt, html, tex, docx) and 6 narrative Report extensions (md, txt,
html, tex, docx, json). Optional deps (openpyxl, python-docx) are
conditionally skipped via pytest.importorskip.
"""

from __future__ import annotations

import csv
import json
import zipfile

import pytest

from ssdiff.results.display import NARRATIVE_EXTS, TABULAR_EXTS
from ssdiff.results.report import Report, Section

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# parametrize over all supported tabular extensions
# ---------------------------------------------------------------------------

# Extensions that require optional deps; guarded with importorskip inside test
_OPTIONAL_TABULAR = {"xlsx": "openpyxl", "docx": "docx"}


_TABULAR_MARKER = {
    "csv": "side,rank,word,cos_beta",
    "json": '"cos_beta":',
    "md": "| side | rank | word | cos_beta |",
    "txt": "side  rank",
    "html": "<table>",
    "tex": "\\begin{tabular}",
}


@pytest.mark.parametrize("ext", TABULAR_EXTS)
def test_view_save_writes_each_format(pls_result, tmp_path, ext):
    """Each tabular format holds the table in its own container or markup."""
    if ext in _OPTIONAL_TABULAR:
        pytest.importorskip(_OPTIONAL_TABULAR[ext])

    out = tmp_path / f"words.{ext}"
    pls_result.words.save(out)

    if ext in ("xlsx", "docx"):
        with zipfile.ZipFile(out) as z:
            member = "xl/workbook.xml" if ext == "xlsx" else "word/document.xml"
            assert member in z.namelist()
    else:
        text = out.read_text(encoding="utf-8")
        assert _TABULAR_MARKER[ext] in text
        assert pls_result.words[0].word in text


def test_view_save_csv_roundtrip(pls_result, tmp_path):
    """CSV save: read back and verify column names and at least one row value."""
    view = pls_result.words
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
# save(path=None) creates file in cwd using default name
# ---------------------------------------------------------------------------

def test_view_save_default_cwd(pls_result, tmp_path, monkeypatch):
    """save(path=None) creates a file in cwd with the default filename."""
    monkeypatch.chdir(tmp_path)
    view = pls_result.words
    view.save()  # path=None

    default_name = view._default_filename()
    out = tmp_path / default_name
    assert out.exists(), (
        f"save(None) did not create {default_name!r} in cwd"
    )
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# save(Path("out.csv")) accepts a Path object
# ---------------------------------------------------------------------------

def test_view_save_accepts_path_object(pls_result, tmp_path):
    """save() must accept a pathlib.Path argument (not just str)."""
    view = pls_result.words
    out = tmp_path / "out.csv"
    view.save(out)  # passing a Path, not a str

    assert out.exists()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# save("out") with no extension raises ValueError
# ---------------------------------------------------------------------------

def test_view_save_no_extension_raises(pls_result, tmp_path, monkeypatch):
    """save(path_without_extension) must raise ValueError."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="requires a path with an extension"):
        pls_result.words.save("out")


# ---------------------------------------------------------------------------
# save("out.xyz") with unknown extension raises ValueError
# ---------------------------------------------------------------------------

def test_view_save_unknown_extension_raises(pls_result, tmp_path):
    """save(path_with_unknown_extension) must raise ValueError."""
    view = pls_result.words
    with pytest.raises(ValueError, match=r"[Uu]nsupported"):
        view.save(str(tmp_path / "out.xyz"))


# ---------------------------------------------------------------------------
# save(path, k=N) contract holds for EVERY row-bearing view
# ---------------------------------------------------------------------------
# Every row-bearing view must honor save(path, k=N); a __getitem__ on a slice
# that returns a bare list would only crash at write time.

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
    """save(path, k=N) on any row-bearing view: no crash, exactly min(N, len) data rows.

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
    assert len(rows) == min(k, len(view)), (
        f"{label}: save(k={k}) wrote {len(rows)} data rows; expected {min(k, len(view))}"
    )


# ---------------------------------------------------------------------------
# ScalarView.save("out.csv", k=0) is a silent no-op or pins
# ---------------------------------------------------------------------------

def test_scalar_view_save_k_zero(pls_result, tmp_path):
    """ScalarView.save(..., k=0) still writes its single row."""
    view = pls_result.stats
    out = tmp_path / "stats_k0.csv"
    view.save(str(out), k=0)  # must not raise

    assert out.exists()
    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# Report.save("out.md") starts with "# " (ATX header)
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
# Report.save("out.xyz") raises ValueError
# ---------------------------------------------------------------------------

def test_report_save_unknown_extension_raises(tmp_path):
    """Report.save() with an unsupported extension must raise ValueError."""
    report = _minimal_report()
    with pytest.raises(ValueError, match=r"[Uu]nsupported"):
        report.save(str(tmp_path / "out.xyz"))


# ---------------------------------------------------------------------------
# Bonus: Report.save() round-trips for non-binary narrative formats
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ext", NARRATIVE_EXTS)
def test_report_save_narrative_exists_nonempty(tmp_path, ext):
    """Report.save() writes each narrative format with the report title in it."""
    if ext == "docx":
        pytest.importorskip("docx")
    out = tmp_path / f"report.{ext}"
    _minimal_report().save(out)

    if ext == "docx":
        assert zipfile.is_zipfile(out)
    else:
        assert "Test Report" in out.read_text(encoding="utf-8")


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


# ---------------------------------------------------------------------------
# .docx tables: three-rule layout, clusters in a paper's layout
# ---------------------------------------------------------------------------

def _rules(cell) -> dict:
    """``{edge: w:val}`` of one python-docx cell's borders."""
    from docx.oxml.ns import qn
    borders = cell._tc.tcPr.find(qn("w:tcBorders"))
    return {el.tag.split("}")[1]: el.get(qn("w:val")) for el in borders}


def test_docx_table_has_three_rules(pls_result, tmp_path):
    """Rules above and below the header and under the last row; none between rows or columns."""
    docx = pytest.importorskip("docx")
    out = tmp_path / "words.docx"
    pls_result.words.save(out)

    rows = docx.Document(out).tables[0].rows
    assert len(rows) > 2
    for i, row in enumerate(rows):
        top = "single" if i == 0 else "nil"
        bottom = "single" if i in (0, len(rows) - 1) else "nil"
        for cell in row.cells:
            assert _rules(cell) == {"top": top, "left": "nil", "bottom": bottom, "right": "nil"}


def _cluster_tables(result, path):
    """Save ``result.clusters`` as .docx; return ``(captions, tables as lists of row texts)``."""
    import docx
    result.clusters.save(path)
    doc = docx.Document(path)
    captions = [p.text for p in doc.paragraphs if p.text]
    tables = [[[c.text for c in row.cells] for row in t.rows] for t in doc.tables]
    return captions, tables


def test_clusters_docx_paper_layout(pls_result, tmp_path):
    """One table per side, numbered from 1, ten words per cluster, a snippet, the outcome in the caption."""
    pytest.importorskip("docx")
    captions, tables = _cluster_tables(pls_result, tmp_path / "clusters.docx")

    assert captions == ["Positive semantic clusters (+β̂ → higher y)",
                        "Negative semantic clusters (−β̂ → lower y)"]
    for side, table in zip(("pos", "neg"), tables):
        view = getattr(pls_result.clusters, side)
        assert table[0] == ["No.", "Size", "Top Words", "Representative Excerpt"]
        assert len(table) == 1 + len(view)
        for i, (line, c) in enumerate(zip(table[1:], view), start=1):
            words = sorted(view(c.cluster_id).words, key=lambda w: w.cos_centroid, reverse=True)
            assert line[:3] == [str(i), str(c.size), ", ".join(w.word for w in words[:10])]
            assert line[3]


def test_clusters_docx_group_pair_without_texts(group_result_2g, tmp_path):
    """A pair's poles point toward its groups; token-list texts give no excerpt column."""
    pytest.importorskip("docx")
    pair = group_result_2g[group_result_2g.keys()[0]]
    captions, tables = _cluster_tables(pair, tmp_path / "clusters.docx")

    assert captions == [f"Positive semantic clusters (+β̂ → toward {pair.g1})",
                        f"Negative semantic clusters (−β̂ → toward {pair.g2})"]
    assert all(t[0] == ["No.", "Size", "Top Words"] for t in tables)
