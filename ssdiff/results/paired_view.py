"""Paired-view collection classes for multi-group GroupResult.

Each class wraps a ``dict[tuple[str, str], <single-pair view>]`` keyed on
canonical pair tuples ``(g_i, g_j)`` with numeric order (g1 < g2 < …).

Public interface (same on every paired view):

    pv[(g1, g2)]                         → single-pair view (canonical order only)
    pv.keys()                            → list[tuple[str, str]]
    len(pv)                              → total row count across all pairs
    iter(pv)                             → yields flat rows across all pairs
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
from ssdiff.results.display import (
    _save_hint_enabled,
    build_paired_view_save_hint,
)

if TYPE_CHECKING:
    from ssdiff.results.continuous_result import (
        ClusterWordsView,
        ClustersViewSided,
        SnippetsView,
        SnippetsViewSided,
        WordsView,
    )


# ---------------------------------------------------------------------------
# Internal helpers: numeric-key sort for canonical pair ordering
# ---------------------------------------------------------------------------

def _numeric_key(label: str) -> float:
    """Sort key for canonical labels like ``g1``, ``g2``, …

    Falls back to ``float('inf')`` for non-canonical labels so they sort
    after canonical ones.
    """
    if label.startswith("g"):
        tail = label[1:]
        if tail.isdigit():
            return float(tail)
    return float("inf")


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    """Return ``(a, b)`` sorted in canonical (numeric) order."""
    return tuple(sorted((a, b), key=_numeric_key))  # type: ignore[return-value]


def _pair_str(key: tuple[str, str]) -> str:
    """Convert a canonical pair tuple to a string key, e.g. ``g1_g2``."""
    return f"{key[0]}_{key[1]}"


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
        header = (
            f"{type(self).__name__}(view_name={self._view_name!r}, "
            f"n={len(self._views)}, pairs=[{pairs_str}])"
        )
        if _save_hint_enabled() and self._views:
            return f"{header}\n\n{build_paired_view_save_hint(self)}"
        return header


class _PairedViewBase(_PairedMappingBase):
    """Paired view base: mapping + ``.save()`` + flat iteration across all pairs."""

    # ------------------------------------------------------------------
    # Flat iteration (rows across all pairs, canonical pair order)
    # ------------------------------------------------------------------

    def __iter__(self):
        """Iterate rows across all pairs in canonical pair order."""
        for pair_key in sorted(self._views.keys(), key=lambda k: (_numeric_key(k[0]), _numeric_key(k[1]))):
            yield from iter(self._views[pair_key])

    def __len__(self) -> int:
        return sum(len(v) for v in self._views.values())

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
        - N≥2, csv: fan-out → ``<parent>/<view_name>/gi_gj.csv``
          (emits ``UserWarning``; original path NOT written).
        - N≥2, xlsx: single file, one sheet per pair.
        - N≥2, json: single file, keys ``gi_gj``.
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
    >>> pv[("g1", "g2")]               # → WordsView
    >>> pv.save("words.xlsx")           # one sheet per pair
    >>> pv.save("words.csv")            # fan-out: words/g1_g2.csv …
    """

    def __init__(
        self,
        views: dict[tuple[str, str], "WordsView"],
        view_name: str = "words",
    ) -> None:
        super().__init__(views=views, view_name=view_name)


class ClusterWordsViewPaired(_PairedViewBase):
    """Paired collection of :class:`~ssdiff.results.continuous_result.ClusterWordsView`
    (or :class:`~ssdiff.results.continuous_result.ClusterWordsViewSided`) per contrast.

    Examples
    --------
    >>> pv = gr.clusters.pos.words      # ClusterWordsViewPaired when N≥2
    >>> pv[("g1", "g2")]               # → ClusterWordsViewSided
    >>> pv.save("cluster_words_pos.csv")  # fan-out: cluster_words_pos/g1_g2.csv …
    """

    def __init__(
        self,
        views: dict[tuple[str, str], "ClusterWordsView"],
        view_name: str = "cluster_words",
    ) -> None:
        super().__init__(views=views, view_name=view_name)


class ClustersViewSidedPaired(_PairedViewBase):
    """Paired collection of :class:`~ssdiff.results.continuous_result.ClustersViewSided` instances.

    Returned by :attr:`ClustersViewPaired.pos` and
    :attr:`ClustersViewPaired.neg`.

    Examples
    --------
    >>> pv = gr.clusters.pos            # ClustersViewSidedPaired when N≥2
    >>> pv[("g1", "g2")]               # → ClustersViewSided
    >>> pv.save("clusters_pos.csv")     # fan-out: clusters_pos/g1_g2.csv …
    """

    def __init__(
        self,
        views: dict[tuple[str, str], "ClustersViewSided"],
        view_name: str,
    ) -> None:
        super().__init__(views=views, view_name=view_name)

    @property
    def words(self) -> ClusterWordsViewPaired:
        """Paired cluster-words for this side across all pairs.

        Each pair's member is a ``ClusterWordsViewSided`` carrying its ``_side``,
        so per-pair defaults still resolve to ``cluster_words_{side}.csv``.
        view_name swaps the ``clusters_`` prefix for ``cluster_words_`` so
        multi-pair csv fan-out writes into ``cluster_words_{side}/…``.
        """
        new_view_name = self._view_name.replace("clusters_", "cluster_words_", 1)
        members: dict[tuple[str, str], "ClusterWordsView"] = {}
        for pair_key, child in self._views.items():
            members[pair_key] = child.words  # type: ignore[attr-defined]
        return ClusterWordsViewPaired(members, view_name=new_view_name)


class ClustersViewPaired(_PairedViewBase):
    """Paired collection wrapper for cluster access — flat iterable over all Cluster rows.

    Iterates rows across all pairs in canonical pair order; within each pair, pos first then neg.
    Accessors preserved: ``.pos`` / ``.neg`` (return ``ClustersViewSidedPaired``), ``.words``,
    dict lookup ``pv[(g1,g2)]``.

    Examples
    --------
    >>> pv = gr.clusters                  # ClustersViewPaired when N≥2
    >>> pv.pos.save("clusters_pos.csv")   # fan-out: clusters_pos/g1_g2.csv …
    >>> pv[("g1", "g2")]                  # → underlying index for that pair
    """

    _name = "clusters"
    _columns = ("cluster_id", "side", "size", "coherence", "centroid_cos_beta", "contrast")

    def __init__(
        self,
        views: dict[tuple[str, str], object],
    ) -> None:
        super().__init__(views=views, view_name="clusters")

    def __iter__(self):
        """Iterate all Cluster rows across pairs: canonical pair order, pos then neg within each."""
        for pair_key in sorted(self._views.keys(), key=lambda k: (_numeric_key(k[0]), _numeric_key(k[1]))):
            idx = self._views[pair_key]
            yield from iter(idx.pos)  # type: ignore[union-attr]
            yield from iter(idx.neg)  # type: ignore[union-attr]

    def __len__(self) -> int:
        total = 0
        for idx in self._views.values():
            total += len(idx.pos) + len(idx.neg)  # type: ignore[union-attr]
        return total

    def __getitem__(self, key):
        """Tuple key → single-pair index; integer key is not supported."""
        if isinstance(key, tuple):
            return super().__getitem__(key)
        raise KeyError(f"ClustersViewPaired does not support integer indexing; use a pair tuple")

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

    @property
    def words(self) -> ClusterWordsViewPaired:
        """Combined cluster-words (both sides) across all pairs.

        Each member is a plain :class:`ClusterWordsView` concatenating
        pos + neg rows, so per-pair saves default to ``cluster_words.csv``.
        """
        from ssdiff.results.continuous_result import ClusterWordsView
        members: dict[tuple[str, str], "ClusterWordsView"] = {}
        for pair_key, idx in self._views.items():
            pos_rows = idx.pos._words_rows  # type: ignore[attr-defined]
            neg_rows = idx.neg._words_rows  # type: ignore[attr-defined]
            members[pair_key] = ClusterWordsView(list(pos_rows) + list(neg_rows))
        return ClusterWordsViewPaired(members, view_name="cluster_words")

    def __repr__(self) -> str:
        pairs_str = ", ".join(f"({a!r}, {b!r})" for a, b in self._views)
        header = (
            f"ClustersViewPaired(n={len(self._views)}, pairs=[{pairs_str}])\n"
            f"  .pos   → ClustersViewSidedPaired (view_name='clusters_pos')\n"
            f"  .neg   → ClustersViewSidedPaired (view_name='clusters_neg')\n"
            f"  .words → ClusterWordsViewPaired  (view_name='cluster_words')"
        )
        if _save_hint_enabled() and self._views:
            return f"{header}\n\n{build_paired_view_save_hint(self)}"
        return header


class SnippetsViewSidedPaired(_PairedViewBase):
    """Paired collection of :class:`~ssdiff.results.continuous_result.SnippetsViewSided` instances.

    Returned by :attr:`SnippetsViewPaired.pos` and :attr:`SnippetsViewPaired.neg`.

    Examples
    --------
    >>> pv = gr.snippets.pos           # SnippetsViewSidedPaired when N>=2
    >>> pv[("g1", "g2")]               # → SnippetsViewSided
    >>> pv.save("snippets_pos.csv")    # fan-out: snippets_pos/g1_g2.csv …
    """

    def __init__(
        self,
        views: dict[tuple[str, str], "SnippetsViewSided"],
        view_name: str,
    ) -> None:
        super().__init__(views=views, view_name=view_name)


class SnippetsViewPaired(_PairedViewBase):
    """Paired collection of :class:`~ssdiff.results.continuous_result.SnippetsView` instances.

    Keys are canonical pair tuples ``(g_i, g_j)``; access in non-canonical
    order raises :exc:`KeyError`.

    Examples
    --------
    >>> pv = gr.snippets                # SnippetsViewPaired when N≥2
    >>> pv[("g1", "g2")]               # → SnippetsView
    >>> pv.save("snippets.json")        # one file, keys g1_g2 …
    """

    def __init__(
        self,
        views: dict[tuple[str, str], "SnippetsView"],
        view_name: str = "snippets",
    ) -> None:
        super().__init__(views=views, view_name=view_name)

    @property
    def pos(self) -> SnippetsViewSidedPaired:
        """Paired positive-side snippets across all pairs."""
        from ssdiff.results.continuous_result import SnippetsViewSided
        members = {
            key: SnippetsViewSided(side="pos", all_rows=list(child._rows))
            for key, child in self._views.items()
        }
        return SnippetsViewSidedPaired(members, view_name="snippets_pos")

    @property
    def neg(self) -> SnippetsViewSidedPaired:
        """Paired negative-side snippets across all pairs."""
        from ssdiff.results.continuous_result import SnippetsViewSided
        members = {
            key: SnippetsViewSided(side="neg", all_rows=list(child._rows))
            for key, child in self._views.items()
        }
        return SnippetsViewSidedPaired(members, view_name="snippets_neg")
