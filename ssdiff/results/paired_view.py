"""Save helper for multi-key views in GroupResult.

``_paired_save`` is the unified dispatch used by :class:`~ssdiff.results.multi_container._ShimView`
to write multi-key result views to disk.  It handles all supported formats:

- **csv** (N≥2): fan-out into ``<parent>/<view_name>/key.csv`` subdirectory.
- **xlsx** (N≥2): single file, one sheet per key.
- **json**: single file, one top-level key per pair.
- **md / html / tex / docx / txt**: single file with one section per key.

For N=1 it delegates directly to the single child view's ``.save()``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from ssdiff.results.core import (
    _check_ext,
    _get_ext,
    _project,
    _render_csv,
    _render_md_table,
    _render_tex_table,
    _row_to_dict,
    _validate_cols,
    _warn,
    _json_default,
    _require,
)


# ---------------------------------------------------------------------------
# Multi-sheet xlsx helper
# ---------------------------------------------------------------------------

def _write_xlsx_multisheet(
    sheets: dict[str, tuple[list[dict], Sequence[str]]],
    path: str | Path,
) -> None:
    """Write multiple sheets to a single Excel file.

    Parameters
    ----------
    sheets : dict mapping sheet_name → (rows, cols)
    path : output file path
    """
    pd = _require("pandas")
    _require("openpyxl")
    with pd.ExcelWriter(str(path), engine="openpyxl") as writer:
        for sheet_name, (rows, cols) in sheets.items():
            df = pd.DataFrame(rows, columns=list(cols))
            df.to_excel(writer, sheet_name=sheet_name, index=False)


# ---------------------------------------------------------------------------
# Multi-section docx helper
# ---------------------------------------------------------------------------

def _write_docx_sectioned(
    sections: list[tuple[str, list[dict], Sequence[str]]],
    path: str | Path,
) -> None:
    """Write a .docx file with one heading + table per section.

    Parameters
    ----------
    sections : list of (title, rows, cols)
    path : output file path
    """
    from ssdiff.results.format import fmt_cell
    docx = _require("docx", extra="results")
    doc = docx.Document()
    for title, rows, cols in sections:
        doc.add_heading(title, level=1)
        table = doc.add_table(rows=1 + len(rows), cols=len(cols))
        for j, c in enumerate(cols):
            table.rows[0].cells[j].text = str(c)
        for i, r in enumerate(rows, start=1):
            for j, c in enumerate(cols):
                table.rows[i].cells[j].text = fmt_cell(r.get(c), c)
    doc.save(str(path))


# ---------------------------------------------------------------------------
# Shared save dispatch
# ---------------------------------------------------------------------------

def _paired_save(
    view_name: str,
    views: dict,
    path: str | Path | None,
    *,
    cols=None,
    k: int | None = None,
    key_to_str=None,
    key_heading=None,
) -> None:
    """Unified save dispatch for all multi-key views.

    Parameters
    ----------
    view_name : str
        Logical name used for csv subfolder and sheet prefix.
    views : dict
        Mapping from key → single-key view.
    path : str, Path, or None
        Output path. None → ``<cwd>/<view_name>.csv``.
    cols : sequence of str, optional
    k : int, optional
    key_to_str : callable, optional
        Maps a dict key to a filename/sheet/JSON-key-safe string.
        Defaults to pair-tuple form: ``('g1', 'g2') → 'g1_g2'``.
    key_heading : callable, optional
        Maps a dict key to a human-readable section heading string.
        Defaults to pair-tuple form: ``('g1', 'g2') → 'g1 vs g2'``.
    """
    if key_to_str is None:
        def key_to_str(kk):  # pair-tuple default
            return f"{kk[0]}_{kk[1]}"
    if key_heading is None:
        def key_heading(kk):
            return f"{kk[0]} vs {kk[1]}"

    if path is None:
        path = Path(f"{view_name}.csv")
    path = Path(path)
    ext = _get_ext(str(path))
    _check_ext(ext)

    n = len(views)

    # N=1: delegate to the single child view
    if n == 1:
        (_, child), = views.items()
        child.save(str(path), cols=cols, k=k)  # type: ignore[union-attr]
        return

    # N≥2 dispatch by extension
    if ext == "csv":
        _warn(
            f"multi-pair csv save fans out; writing "
            f"{path.parent}/{view_name}/*.csv instead of {path}",
            stacklevel=4,
        )
        subfolder = path.parent / view_name
        subfolder.mkdir(parents=True, exist_ok=True)
        for pair_key, child in views.items():
            filename = key_to_str(pair_key) + ".csv"
            child_view = child._resized(k) if k is not None else child  # type: ignore[union-attr]
            keep, warning = _validate_cols(cols, child_view)
            if warning:
                _warn(warning, stacklevel=4)
            rows = [_project(_row_to_dict(r), keep) for r in child_view]
            out_path = subfolder / filename
            with open(out_path, "w", encoding="utf-8", newline="") as f:
                f.write(_render_csv(rows, keep))

    elif ext == "xlsx":
        sheets: dict[str, tuple[list[dict], Sequence[str]]] = {}
        for pair_key, child in views.items():
            sheet_name = key_to_str(pair_key)
            child_view = child._resized(k) if k is not None else child  # type: ignore[union-attr]
            keep, warning = _validate_cols(cols, child_view)
            if warning:
                _warn(warning, stacklevel=4)
            rows = [_project(_row_to_dict(r), keep) for r in child_view]
            sheets[sheet_name] = (rows, keep)
        _write_xlsx_multisheet(sheets, path)

    elif ext == "json":
        result: dict[str, list[dict]] = {}
        for pair_key, child in views.items():
            key_str = key_to_str(pair_key)
            child_view = child._resized(k) if k is not None else child  # type: ignore[union-attr]
            keep, warning = _validate_cols(cols, child_view)
            if warning:
                _warn(warning, stacklevel=4)
            rows = [_project(_row_to_dict(r), keep) for r in child_view]
            result[key_str] = rows
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, default=_json_default, indent=2)

    elif ext == "md":
        parts: list[str] = []
        for pair_key, child in views.items():
            heading = f"## {key_heading(pair_key)}"
            child_view = child._resized(k) if k is not None else child  # type: ignore[union-attr]
            keep, warning = _validate_cols(cols, child_view)
            if warning:
                _warn(warning, stacklevel=4)
            rows = [_project(_row_to_dict(r), keep) for r in child_view]
            parts.append(heading)
            parts.append(_render_md_table(rows, keep))
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(parts))

    elif ext == "html":
        parts: list[str] = []
        for pair_key, child in views.items():
            heading = f"<h2>{key_heading(pair_key)}</h2>"
            child_view = child._resized(k) if k is not None else child  # type: ignore[union-attr]
            keep, warning = _validate_cols(cols, child_view)
            if warning:
                _warn(warning, stacklevel=4)
            # Use the child view's to_html for formatted output
            html_body = child_view.to_html(cols=list(keep) if keep else None)  # type: ignore[union-attr]
            parts.append(heading)
            parts.append(html_body)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))

    elif ext == "tex":
        parts: list[str] = []
        for pair_key, child in views.items():
            child_view = child._resized(k) if k is not None else child  # type: ignore[union-attr]
            keep, warning = _validate_cols(cols, child_view)
            if warning:
                _warn(warning, stacklevel=4)
            rows = [_project(_row_to_dict(r), keep) for r in child_view]
            parts.append(f"\\subsection*{{{key_heading(pair_key)}}}")
            parts.append(_render_tex_table(rows, keep))
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(parts))

    elif ext == "txt":
        parts: list[str] = []
        for pair_key, child in views.items():
            heading = key_heading(pair_key)
            child_view = child._resized(k) if k is not None else child  # type: ignore[union-attr]
            keep, warning = _validate_cols(cols, child_view)
            if warning:
                _warn(warning, stacklevel=4)
            text_body = child_view.to_text(cols=list(keep) if keep else None)  # type: ignore[union-attr]
            parts.append(heading)
            parts.append(text_body)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(parts))

    elif ext == "docx":
        sections: list[tuple[str, list[dict], Sequence[str]]] = []
        for pair_key, child in views.items():
            title = key_heading(pair_key)
            child_view = child._resized(k) if k is not None else child  # type: ignore[union-attr]
            keep, warning = _validate_cols(cols, child_view)
            if warning:
                _warn(warning, stacklevel=4)
            rows = [_project(_row_to_dict(r), keep) for r in child_view]
            sections.append((title, rows, keep))
        _write_docx_sectioned(sections, path)
