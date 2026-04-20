"""Integration tests for gr.words.save() across all supported formats.

Uses the real ``group_result_3g`` session-scoped fixture (no hand-constructed
views).  One test per format; dispatch minutiae are covered exhaustively in
``test_paired_view_save.py``.

Canonical pairs for the 3-group fixture: g_1_vs_g_2, g_1_vs_g_3, g_2_vs_g_3.
Section titles: "g_1 vs g_2", "g_1 vs g_3", "g_2 vs g_3".
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expected_pair_strings(gr):
    """Return the set of canonical pair key strings, e.g. {'g_1_vs_g_2', ...}."""
    return {f"{p.g1}_vs_{p.g2}" for p in gr.pairs}


def _expected_pair_titles(gr):
    """Return the set of per-pair heading titles, e.g. {'g_1 vs g_2', ...}."""
    return {f"{p.g1} vs {p.g2}" for p in gr.pairs}


# ---------------------------------------------------------------------------
# csv — fan-out, one file per pair + UserWarning
# ---------------------------------------------------------------------------


def test_save_csv_fanout(group_result_3g, tmp_path):
    """gr.words.save(path.csv) for N=3 fans out to tmp/words/g_i_vs_g_j.csv."""
    target = tmp_path / "words.csv"
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        group_result_3g.words.save(target)

    # Original path NOT written
    assert not target.exists(), "original csv path should not be created"

    # Subfolder name is hardcoded to 'words', not derived from filename stem
    folder = tmp_path / "words"
    assert folder.is_dir(), "fan-out subfolder 'words' should exist"

    for pair in group_result_3g.pairs:
        expected_file = folder / f"{pair.g1}_vs_{pair.g2}.csv"
        assert expected_file.is_file(), f"expected {expected_file}"

    # UserWarning must be emitted
    assert any(issubclass(w.category, UserWarning) for w in recorded), \
        "expected a UserWarning on csv fan-out"


def test_save_csv_fanout_subfolder_hardcoded(group_result_3g, tmp_path):
    """Subfolder name is always 'words', regardless of the caller's filename stem."""
    target = tmp_path / "anything.csv"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        group_result_3g.words.save(target)

    assert not target.exists()
    # Subfolder must be 'words', NOT 'anything'
    assert (tmp_path / "words").is_dir(), "subfolder must be 'words', not 'anything'"
    assert not (tmp_path / "anything").exists(), "unexpected 'anything' subfolder"


# ---------------------------------------------------------------------------
# xlsx — single file, one sheet per pair
# ---------------------------------------------------------------------------


def test_save_xlsx_single_file(group_result_3g, tmp_path):
    """gr.words.save(words.xlsx) produces one file; sheets match canonical pair strings."""
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "words.xlsx"
    group_result_3g.words.save(path)

    assert path.is_file(), "expected single xlsx file"

    wb = openpyxl.load_workbook(path)
    assert set(wb.sheetnames) == _expected_pair_strings(group_result_3g)


# ---------------------------------------------------------------------------
# json — single file, keys match canonical pair strings
# ---------------------------------------------------------------------------


def test_save_json_single_file(group_result_3g, tmp_path):
    """gr.words.save(words.json) produces one file; keys are canonical pair strings."""
    path = tmp_path / "words.json"
    group_result_3g.words.save(path)

    assert path.is_file(), "expected single json file"

    data = json.loads(path.read_text())
    assert set(data.keys()) == _expected_pair_strings(group_result_3g)


# ---------------------------------------------------------------------------
# md — single file, each pair title present
# ---------------------------------------------------------------------------


def test_save_md_sectioned(group_result_3g, tmp_path):
    """gr.words.save(words.md) produces one file with per-pair headings."""
    path = tmp_path / "words.md"
    group_result_3g.words.save(path)

    assert path.is_file()
    body = path.read_text()
    for title in _expected_pair_titles(group_result_3g):
        assert title in body, f"expected section '{title}' in md output"


# ---------------------------------------------------------------------------
# html — single file, each pair title present
# ---------------------------------------------------------------------------


def test_save_html_sectioned(group_result_3g, tmp_path):
    """gr.words.save(words.html) produces one file with per-pair headings."""
    path = tmp_path / "words.html"
    group_result_3g.words.save(path)

    assert path.is_file()
    body = path.read_text()
    for title in _expected_pair_titles(group_result_3g):
        assert title in body, f"expected section '{title}' in html output"


# ---------------------------------------------------------------------------
# tex — single file, each pair title present
# ---------------------------------------------------------------------------


def test_save_tex_sectioned(group_result_3g, tmp_path):
    """gr.words.save(words.tex) produces one file with per-pair subsections."""
    path = tmp_path / "words.tex"
    group_result_3g.words.save(path)

    assert path.is_file()
    body = path.read_text()
    for title in _expected_pair_titles(group_result_3g):
        assert title in body, f"expected section '{title}' in tex output"


# ---------------------------------------------------------------------------
# docx — single file, one heading per pair
# ---------------------------------------------------------------------------


def test_save_docx_sectioned(group_result_3g, tmp_path):
    """gr.words.save(words.docx) produces one file with one heading per pair."""
    docx = pytest.importorskip("docx")
    path = tmp_path / "words.docx"
    group_result_3g.words.save(path)

    assert path.is_file()
    doc = docx.Document(str(path))
    headings = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading")]
    for title in _expected_pair_titles(group_result_3g):
        assert title in headings, f"expected heading '{title}' in docx"
