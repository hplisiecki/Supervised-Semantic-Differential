"""Paired-view collection classes for multi-group GroupResult.

Each class wraps a ``dict[tuple[str, str], <single-pair view>]`` keyed on
canonical pair tuples ``(g_i, g_j)`` with numeric order (g_1 < g_2 < …).

Public interface (same on every paired view):

    pv[(g1, g2)]                         → single-pair view (canonical order only)
    pv.keys()                            → list[tuple[str, str]]
    len(pv)                              → int
    iter(pv)                             → yields (pair_tuple, single_view) tuples
    pv.save(path, *, cols=None, k=None)  → None
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ssdiff.results.core import (
    SUPPORTED_EXTS,
    _check_ext,
    _get_ext,
    _project,
    _render_csv,
    _render_json,
    _render_md_table,
    _render_tex_table,
    _row_to_dict,
    _validate_cols,
    _warn,
    _json_default,
    _require,
)

if TYPE_CHECKING:
    from ssdiff.results.continuous_result import (
        ClustersViewSided,
        SnippetsView,
        WordsView,
    )


# ---------------------------------------------------------------------------
# Internal helpers: numeric-key sort for canonical pair ordering
# ---------------------------------------------------------------------------

def _numeric_key(label: str) -> float:
    """Sort key for canonical labels like ``g_1``, ``g_2``, …

    Falls back to ``float('inf')`` for non-canonical labels so they sort
    after canonical ones.
    """
    if label.startswith("g_"):
        try:
            return float(label.split("_", 1)[1])
        except (ValueError, IndexError):
            pass
    return float("inf")


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    """Return ``(a, b)`` sorted in canonical (numeric) order."""
    return tuple(sorted((a, b), key=_numeric_key))  # type: ignore[return-value]


def _pair_str(key: tuple[str, str]) -> str:
    """Convert a canonical pair tuple to a string key, e.g. ``g_1_vs_g_2``."""
    return f"{key[0]}_vs_{key[1]}"


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
    views: dict[tuple[str, str], object],
    path: str | Path | None,
    *,
    cols=None,
    k: int | None = None,
) -> None:
    """Unified save dispatch for all paired views.

    Parameters
    ----------
    view_name : str
        Hardcoded view name used for csv subfolder and sheet prefix.
    views : dict
        Mapping from canonical pair tuple → single-pair view.
    path : str, Path, or None
        Output path. None → ``<cwd>/<view_name>.csv``.
    cols : sequence of str, optional
    k : int, optional
    """
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
            filename = _pair_str(pair_key) + ".csv"
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
            sheet_name = _pair_str(pair_key)
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
            key_str = _pair_str(pair_key)
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
            heading = f"## {pair_key[0]} vs {pair_key[1]}"
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
            heading = f"<h2>{pair_key[0]} vs {pair_key[1]}</h2>"
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
            heading = f"% {pair_key[0]} vs {pair_key[1]}"
            child_view = child._resized(k) if k is not None else child  # type: ignore[union-attr]
            keep, warning = _validate_cols(cols, child_view)
            if warning:
                _warn(warning, stacklevel=4)
            rows = [_project(_row_to_dict(r), keep) for r in child_view]
            parts.append(f"\\subsection*{{{pair_key[0]} vs {pair_key[1]}}}")
            parts.append(_render_tex_table(rows, keep))
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(parts))

    elif ext == "txt":
        parts: list[str] = []
        for pair_key, child in views.items():
            heading = f"{pair_key[0]} vs {pair_key[1]}"
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
            title = f"{pair_key[0]} vs {pair_key[1]}"
            child_view = child._resized(k) if k is not None else child  # type: ignore[union-attr]
            keep, warning = _validate_cols(cols, child_view)
            if warning:
                _warn(warning, stacklevel=4)
            rows = [_project(_row_to_dict(r), keep) for r in child_view]
            sections.append((title, rows, keep))
        _write_docx_sectioned(sections, path)


# ---------------------------------------------------------------------------
# Base classes for paired views
# ---------------------------------------------------------------------------

class _PairedMappingBase:
    """Dict-access infrastructure shared by all paired-view and paired-index classes.

    Holds ``_views``, ``_view_name``, and the four mapping methods
    (``__getitem__``, ``keys``, ``__len__``, ``__iter__``).
    """

    def __init__(
        self,
        views: dict[tuple[str, str], object],
        view_name: str,
    ) -> None:
        self._views = dict(views)
        self._view_name = view_name

    # ------------------------------------------------------------------
    # Core mapping interface
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError(key)
        a, b = key
        canonical = _canonical_pair(a, b)
        if key != canonical:
            raise KeyError(
                f"pair must be accessed in canonical order {canonical!r}, got {key!r}"
            )
        if key not in self._views:
            raise KeyError(key)
        return self._views[key]

    def keys(self) -> list[tuple[str, str]]:
        """Return list of canonical pair tuples."""
        return list(self._views.keys())

    def __len__(self) -> int:
        return len(self._views)

    def __iter__(self):
        """Yields ``(pair_tuple, single_view)`` 2-tuples."""
        return iter(self._views.items())

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        pairs_str = ", ".join(f"({a!r}, {b!r})" for a, b in self._views)
        return (
            f"{type(self).__name__}(view_name={self._view_name!r}, "
            f"n={len(self._views)}, pairs=[{pairs_str}])"
        )


class _PairedViewBase(_PairedMappingBase):
    """Paired view base that adds ``.save()`` on top of the mapping interface."""

    # ------------------------------------------------------------------
    # Unified save
    # ------------------------------------------------------------------

    def save(
        self,
        path: str | Path | None = None,
        *,
        cols=None,
        k: int | None = None,
    ) -> None:
        """Write the paired view to ``path``; format inferred from extension.

        Dispatch:

        - N=1: delegate to the single underlying view's ``.save()``.
        - N≥2, csv: fan-out → ``<parent>/<view_name>/g_i_vs_g_j.csv``
          (emits ``UserWarning``; original path NOT written).
        - N≥2, xlsx: single file, one sheet per pair.
        - N≥2, json: single file, keys ``g_i_vs_g_j``.
        - N≥2, md / html / tex / docx / txt: single file with per-pair sections.

        Default path (if omitted): ``<cwd>/<view_name>.csv``.
        """
        _paired_save(
            view_name=self._view_name,
            views=self._views,
            path=path,
            cols=cols,
            k=k,
        )


# ---------------------------------------------------------------------------
# Concrete paired-view classes
# ---------------------------------------------------------------------------

class WordsViewPaired(_PairedViewBase):
    """Paired collection of :class:`~ssdiff.results.continuous_result.WordsView` instances.

    Keys are canonical pair tuples ``(g_i, g_j)``; access in non-canonical
    order raises :exc:`KeyError`.

    Examples
    --------
    >>> pv = gr.words                   # WordsViewPaired when N≥2
    >>> pv[("g_1", "g_2")]              # → WordsView
    >>> pv.save("words.xlsx")           # one sheet per pair
    >>> pv.save("words.csv")            # fan-out: words/g_1_vs_g_2.csv …
    """

    def __init__(
        self,
        views: dict[tuple[str, str], "WordsView"],
        view_name: str = "words",
    ) -> None:
        super().__init__(views=views, view_name=view_name)


class ClustersViewSidedPaired(_PairedViewBase):
    """Paired collection of :class:`~ssdiff.results.continuous_result.ClustersViewSided` instances.

    Returned by :attr:`ClustersIndexPaired.pos` and
    :attr:`ClustersIndexPaired.neg`.

    Examples
    --------
    >>> pv = gr.clusters.pos            # ClustersViewSidedPaired when N≥2
    >>> pv[("g_1", "g_2")]              # → ClustersViewSided
    >>> pv.save("clusters_pos.csv")     # fan-out: clusters_pos/g_1_vs_g_2.csv …
    """

    def __init__(
        self,
        views: dict[tuple[str, str], "ClustersViewSided"],
        view_name: str,
    ) -> None:
        super().__init__(views=views, view_name=view_name)


class ClustersIndexPaired(_PairedMappingBase):
    """Paired collection wrapper for cluster index access.

    Wraps a ``dict[tuple[str, str], <clusters-index-like>]`` (each value
    has ``.pos`` / ``.neg`` returning a :class:`ClustersViewSided`).

    Access `.pos` / `.neg` to get :class:`ClustersViewSidedPaired` instances
    whose save subfolder names are hardcoded to ``"clusters_pos"`` /
    ``"clusters_neg"``.

    Examples
    --------
    >>> pv = gr.clusters                # ClustersIndexPaired when N≥2
    >>> pv.pos.save("clusters_pos.csv") # fan-out: clusters_pos/g_1_vs_g_2.csv …
    >>> pv[("g_1", "g_2")]              # → underlying index for that pair
    """

    def __init__(
        self,
        views: dict[tuple[str, str], object],
    ) -> None:
        super().__init__(views=views, view_name="clusters")

    @property
    def pos(self) -> ClustersViewSidedPaired:
        """Return :class:`ClustersViewSidedPaired` for the positive pole across all pairs."""
        return ClustersViewSidedPaired(
            views={k: v.pos for k, v in self._views.items()},  # type: ignore[union-attr]
            view_name="clusters_pos",
        )

    @property
    def neg(self) -> ClustersViewSidedPaired:
        """Return :class:`ClustersViewSidedPaired` for the negative pole across all pairs."""
        return ClustersViewSidedPaired(
            views={k: v.neg for k, v in self._views.items()},  # type: ignore[union-attr]
            view_name="clusters_neg",
        )

    def __repr__(self) -> str:
        pairs_str = ", ".join(f"({a!r}, {b!r})" for a, b in self._views)
        return (
            f"ClustersIndexPaired(n={len(self._views)}, pairs=[{pairs_str}])\n"
            f"  .pos → ClustersViewSidedPaired (view_name='clusters_pos')\n"
            f"  .neg → ClustersViewSidedPaired (view_name='clusters_neg')"
        )


class SnippetsViewPaired(_PairedViewBase):
    """Paired collection of :class:`~ssdiff.results.continuous_result.SnippetsView` instances.

    Keys are canonical pair tuples ``(g_i, g_j)``; access in non-canonical
    order raises :exc:`KeyError`.

    Examples
    --------
    >>> pv = gr.snippets                # SnippetsViewPaired when N≥2
    >>> pv[("g_1", "g_2")]             # → SnippetsView
    >>> pv.save("snippets.json")        # one file, keys g_1_vs_g_2 …
    """

    def __init__(
        self,
        views: dict[tuple[str, str], "SnippetsView"],
        view_name: str = "snippets",
    ) -> None:
        super().__init__(views=views, view_name=view_name)
