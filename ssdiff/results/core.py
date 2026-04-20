"""Result ABC, View / ScalarView protocols, parameter-keyed cache.

The cache is a dict keyed on (view_name, frozen_params).
Each distinct parameter set gets its own entry — this is how we
fix the stale-cache bug documented in the spec (issue #2).
"""

from __future__ import annotations

import csv as _csv
import html as _html
import io
import json
import warnings
from abc import ABC
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Generic, TypeVar

from ssdiff.results.display import (
    DEFAULT_COLS,
    DEFAULT_MAX_ROWS,
    _save_hint_enabled,
)

# ``DEFAULT_COLS`` / ``DEFAULT_MAX_ROWS`` live in ``results.display`` as part
# of the display-layer scaffolding. Re-exported here so existing internal
# imports (``from ssdiff.results.core import DEFAULT_MAX_ROWS``) keep working.

T = TypeVar("T")


def _freeze_params(params: dict) -> tuple:
    """Convert params dict to a hashable tuple of sorted (k, v) pairs."""
    return tuple(sorted(params.items()))


# ---------------- optional-dep gating ----------------------------------------

def _require(mod: str, *, extra: str = "results"):
    """Import `mod` or raise ImportError with an install hint.

    Kept as a single helper so all error messages match exactly.
    """
    try:
        return __import__(mod)
    except ImportError as exc:
        raise ImportError(
            f"{mod} is required for this method; install with: "
            f"pip install ssdiff[{extra}]"
        ) from exc


def _import_pandas():
    """Import pandas or raise ImportError with an install hint."""
    return _require("pandas")


# ---------------- cols validation --------------------------------------------

def _validate_cols(cols, view: View) -> tuple[tuple[str, ...], str | None]:
    """Enforce D9 rules on a `cols` argument.

    Returns (keep, warning_message_or_None).

    Resolution:
      * ``cols=None``       → the view's default columns (``view._default_cols()``)
      * ``cols="all"``      → full ``view._columns`` (escape sentinel)
      * explicit sequence   → validated against ``view._columns``
    """
    allowed = view._columns
    if cols is None:
        return view._default_cols(), None
    if cols == "all":
        return tuple(allowed), None
    allowed_set = set(allowed)
    asked = list(cols)
    unknown = [c for c in asked if c not in allowed_set]
    known = tuple(c for c in asked if c in allowed_set)
    if not known:
        return tuple(allowed), f"all cols unknown; writing all columns: {unknown!r}"
    if unknown:
        return known, f"unknown cols ignored: {unknown!r}"
    return known, None


def _warn(msg: str, *, stacklevel: int = 3) -> None:
    """Emit a UserWarning with a stacklevel that points to caller code."""
    warnings.warn(msg, UserWarning, stacklevel=stacklevel)


# ---------------- row / rendering helpers ------------------------------------

def _row_to_dict(row) -> dict:
    """Extract a dict from any frozen dataclass row."""
    if hasattr(row, "__dataclass_fields__"):
        return {f: getattr(row, f) for f in row.__dataclass_fields__}
    if isinstance(row, dict):
        return dict(row)
    raise TypeError(f"unsupported row type: {type(row).__name__}")


def _project(row: dict, cols: Sequence[str]) -> dict:
    """Return a new dict containing only the requested columns, in order."""
    return {c: row.get(c) for c in cols}


def _json_default(obj):
    """``json.dumps`` default handler that converts numpy scalars and arrays."""
    import numpy as _np
    if isinstance(obj, _np.integer):
        return int(obj)
    if isinstance(obj, _np.floating):
        return float(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")


# ---------------- save() extension helpers -----------------------------------

SUPPORTED_EXTS: tuple[str, ...] = (
    "csv", "json", "xlsx", "md", "txt", "html", "tex", "docx",
)


def _get_ext(path: str) -> str:
    """Extract and lowercase the file extension from ``path``, or raise ValueError."""
    if "." not in path:
        raise ValueError(
            f"save() requires a path with an extension; got {path!r}. "
            f"Supported: {SUPPORTED_EXTS}"
        )
    return path.rsplit(".", 1)[-1].lower()


def _check_ext(ext: str) -> None:
    """Raise ValueError if ``ext`` is not in SUPPORTED_EXTS."""
    if ext not in SUPPORTED_EXTS:
        raise ValueError(
            f"Unsupported extension .{ext} for save(). "
            f"Supported: {SUPPORTED_EXTS}"
        )


def _render_csv(rows: list[dict], cols: Sequence[str]) -> str:
    """Render rows as a CSV string with a header row."""
    buf = io.StringIO()
    w = _csv.DictWriter(buf, fieldnames=list(cols))
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def _render_json(rows: list[dict]) -> str:
    """Render rows as a pretty-printed JSON string (handles numpy types)."""
    return json.dumps(rows, ensure_ascii=False, default=_json_default, indent=2)


def _render_md_table(rows: list[dict], cols: Sequence[str]) -> str:
    """Render rows as a GitHub-flavoured Markdown table string."""
    from ssdiff.results.format import fmt_cell
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = "\n".join(
        "| " + " | ".join(fmt_cell(r.get(c), c) for c in cols) + " |"
        for r in rows
    )
    return "\n".join([header, sep, body]) if rows else "\n".join([header, sep])


def _render_tex_table(rows: list[dict], cols: Sequence[str]) -> str:
    """Render rows as a LaTeX tabular environment string (booktabs style)."""
    from ssdiff.results.format import fmt_cell
    col_spec = "l" * len(cols)
    head = " & ".join(cols) + r" \\"
    body = "\n".join(
        " & ".join(fmt_cell(r.get(c), c) for c in cols) + r" \\"
        for r in rows
    )
    if rows:
        return (f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n{head}\n"
                f"\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}")
    return (f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n{head}\n"
            f"\\bottomrule\n\\end{{tabular}}")


def _write_xlsx(rows: list[dict], cols: Sequence[str], path: str) -> None:
    """Write rows to an Excel file using pandas + openpyxl."""
    pd = _require("pandas")
    _require("openpyxl")
    df = pd.DataFrame(rows, columns=list(cols))
    df.to_excel(path, index=False)


def _write_docx_table(rows: list[dict], cols: Sequence[str], path: str,
                      *, title: str | None = None) -> None:
    """Write rows as a .docx table, with an optional heading, using python-docx."""
    from ssdiff.results.format import fmt_cell
    docx = _require("docx", extra="results")
    doc = docx.Document()
    if title:
        doc.add_heading(title, level=1)
    table = doc.add_table(rows=1 + len(rows), cols=len(cols))
    for j, c in enumerate(cols):
        table.rows[0].cells[j].text = str(c)
    for i, r in enumerate(rows, start=1):
        for j, c in enumerate(cols):
            table.rows[i].cells[j].text = fmt_cell(r.get(c), c)
    doc.save(path)



class View(Generic[T]):
    """Base class for all tabular views.

    Subclasses define:
      - `_name`           : cache key name
      - `_columns`        : ordered tuple of column names (for `cols` validation)
      - `__iter__`        : yields frozen domain dataclasses
      - `__len__`, `__getitem__`
      - `params` property (for parameterized views; `{}` for paramless)

    Output methods (to_df, to_csv, …) are defined further down in this module
    (Task 4) directly on `View` / `ScalarView`.
    """

    _name: str = ""
    _columns: tuple[str, ...] = ()

    def __init__(self, *, _no_trunc: bool = False):
        self._no_trunc: bool = _no_trunc

    def __iter__(self) -> Iterator[T]:           # pragma: no cover - abstract
        raise NotImplementedError

    def __len__(self) -> int:                    # pragma: no cover - abstract
        raise NotImplementedError

    def __getitem__(self, i: int) -> T:          # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def columns(self) -> tuple[str, ...]:
        """Ordered tuple of all column names available on this view."""
        return self._columns

    @classmethod
    def _default_cols(cls) -> tuple[str, ...]:
        """Default column subset for ``cols=None``. Looked up by class name
        so that sibling views sharing ``_name`` can still diverge."""
        return DEFAULT_COLS.get(cls.__name__, cls._columns)

    @property
    def params(self) -> dict:
        """Parameters used to compute this view; empty dict for parameterless views."""
        return {}

    def _filename_stem(self) -> str:
        """Return the default filename stem (no extension) for ``save()``.

        The default returns ``self._name``.  Sided subclasses override this to
        append the side, e.g. ``"clusters_pos"`` / ``"clusters_neg"``.
        """
        return self._name

    def _save_hint(self) -> str:
        """Return the save-hint footer text shown below repr output."""
        cols_preview = "   ".join(repr(c) for c in self._columns[:4])
        more = " …" if len(self._columns) > 4 else ""
        stem = self._filename_stem()
        return (f"Save:  .save()  .save('file.ext')  .save('file.ext', cols=[...])"
                f"   # default: {stem}.csv\n"
                f"       extensions: csv json xlsx md txt html tex docx\n"
                f"       .to_df()  .to_records()  .to_dict()\n"
                f"       cols=[{cols_preview}{more}] → select & order columns")

    def _save_hint_html(self) -> str:
        """Return the save-hint footer as an HTML ``<pre>`` block."""
        return f"<pre class='ssd-save-hint'>{_html.escape(self._save_hint())}</pre>"

    def __repr__(self) -> str:
        body = self.to_text()
        if _save_hint_enabled():
            return body + "\n\n" + self._save_hint()
        return body

    def _repr_html_(self) -> str:
        body = self.to_html()
        if _save_hint_enabled():
            return body + "\n" + self._save_hint_html()
        return body

    # -------- D8/D9 tabular output family --------------------------------
    def to_dict(self, cols=None) -> list[dict]:
        """Return a list of row dicts, optionally restricted to ``cols``.

        Parameters
        ----------
        cols : sequence of str, ``"all"``, or None
            Column subset and order. ``None`` uses the view default;
            ``"all"`` returns every available column. Unknown names warn
            and are skipped.

        Returns
        -------
        list of dict
            One dict per row, keys are column names.
        """
        keep, warning = _validate_cols(cols, self)
        if warning:
            _warn(warning)
        return [_project(_row_to_dict(r), keep) for r in self]

    def to_records(self, cols=None) -> list[tuple]:
        """Return a list of row tuples, optionally restricted to ``cols``.

        Parameters
        ----------
        cols : sequence of str, ``"all"``, or None
            Column subset and order. See :meth:`to_dict` for resolution rules.

        Returns
        -------
        list of tuple
            One tuple per row, values ordered as ``cols``.
        """
        keep, warning = _validate_cols(cols, self)
        if warning:
            _warn(warning)
        return [tuple(_row_to_dict(r).get(c) for c in keep) for r in self]

    def to_df(self, cols=None):
        """Return a pandas DataFrame, optionally restricted to ``cols``.

        Parameters
        ----------
        cols : sequence of str, ``"all"``, or None
            Column subset and order. See :meth:`to_dict` for resolution rules.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        ImportError
            If pandas is not installed (``pip install ssdiff[results]``).
        """
        pd = _import_pandas()
        keep, warning = _validate_cols(cols, self)
        if warning:
            _warn(warning)
        rows = [_project(_row_to_dict(r), keep) for r in self]
        return pd.DataFrame(rows, columns=list(keep))

    def to_html(self, cols=None) -> str:
        """HTML table string — used by ``_repr_html_``. For file output, use ``save('x.html')``."""
        from ssdiff.results.format import fmt_cell
        keep, warning = _validate_cols(cols, self)
        if warning:
            _warn(warning)
        rows = [_project(_row_to_dict(r), keep) for r in self]
        parts = ["<table>", "<thead><tr>"]
        parts += [f"<th>{_html.escape(c)}</th>" for c in keep]
        parts += ["</tr></thead>", "<tbody>"]
        for r in rows:
            parts.append("<tr>" + "".join(
                f"<td>{_html.escape(fmt_cell(r.get(c), c))}</td>" for c in keep
            ) + "</tr>")
        parts += ["</tbody>", "</table>"]
        return "\n".join(parts)

    # -------- unified save() ---------------------------------------------
    def save(self, path: str | Path | None = None, *, cols=None, k: int | None = None) -> None:
        """Write the view to ``path``; format is inferred from the extension.

        Supported extensions: ``.csv .json .xlsx .md .txt .html .tex .docx``.

        Parameters
        ----------
        path : str, Path, or None
            Output file. Extension picks the format. When omitted, defaults to
            ``<cwd>/<view_name>.csv`` where ``<view_name>`` is the per-view
            hardcoded filename stem (e.g. ``words``, ``clusters_pos``).
        cols : sequence of str, optional
            Column subset (and order). Unknown names warn and are skipped.
        k : int, optional
            Row cap for size-bearing views (words / docs / clusters / snippets).
            Silently ignored on single-row views.
        """
        if path is None:
            path = Path(f"{self._filename_stem()}.csv")
        ext = _get_ext(str(path))
        _check_ext(ext)

        view = self._resized(k) if k is not None else self
        keep, warning = _validate_cols(cols, view)
        if warning:
            _warn(warning)
        rows = [_project(_row_to_dict(r), keep) for r in view]

        if ext == "csv":
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(_render_csv(rows, keep))
        elif ext == "json":
            with open(path, "w", encoding="utf-8") as f:
                f.write(_render_json(rows))
        elif ext == "md":
            with open(path, "w", encoding="utf-8") as f:
                f.write(_render_md_table(rows, keep))
        elif ext == "tex":
            with open(path, "w", encoding="utf-8") as f:
                f.write(_render_tex_table(rows, keep))
        elif ext == "txt":
            with open(path, "w", encoding="utf-8") as f:
                f.write(view.to_text(cols=cols))
        elif ext == "html":
            with open(path, "w", encoding="utf-8") as f:
                f.write(view.to_html(cols=cols))
        elif ext == "xlsx":
            _write_xlsx(rows, keep, path)
        elif ext == "docx":
            _write_docx_table(rows, keep, path, title=self._name or None)

    def _resized(self, k: int):
        """Return a view trimmed to the first ``k`` rows. Default uses slicing."""
        try:
            return self[:k]
        except (TypeError, NotImplementedError):
            return self

    _text_truncate: int | None = None

    def to_text(self, max_rows: int | None = None, cols=None) -> str:
        """Aligned text rendering for console display.

        Caps at ``max_rows`` with a ``... N more rows`` footer unless this
        view was produced by slicing (``_no_trunc=True``), in which case the
        full content renders without truncation or footer.

        ``max_rows=None`` resolves to the module-level ``DEFAULT_MAX_ROWS``.

        Subclasses with long text columns set ``_text_truncate`` (chars) to
        clip cells in the terminal. Data exports stay full-width.
        """
        from ssdiff.results.format import default_alignment, fmt_cell, fmt_table
        if max_rows is None:
            max_rows = DEFAULT_MAX_ROWS
        keep, warning = _validate_cols(cols, self)
        if warning:
            _warn(warning)
        all_rows = [_project(_row_to_dict(r), keep) for r in self]
        n = len(all_rows)
        if self._no_trunc or n <= max_rows:
            shown = all_rows
            footer = None
        else:
            shown = all_rows[:max_rows]
            footer = f"... {n - max_rows} more rows"
        rows_seq = [[fmt_cell(r.get(c), c) for c in keep] for r in shown]
        out = fmt_table(rows_seq, headers=list(keep),
                        numeric=default_alignment(len(keep)),
                        text_truncate=self._text_truncate)
        if footer:
            out = out + "\n" + footer
        return out


class ScalarView(View):
    """Single-row view (stats, omnibus, summary).

    Narrative formats render 2-column K/V; data formats render 1-row tabular.
    Attribute and dict-style access read the single row directly.
    """

    def __len__(self) -> int:
        return 1

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self
        if isinstance(key, int):
            if key != 0:
                raise IndexError(key)
            return next(iter(self))
        # dict-style lookup — read the full row, not the (possibly narrowed) default subset.
        d = dict(next(iter(self)))
        if key not in d:
            raise KeyError(key)
        return d[key]

    def __getattr__(self, item: str):
        # Only called if normal lookup failed.
        if item.startswith("_"):
            raise AttributeError(item)
        try:
            d = dict(next(iter(self)))
        except Exception as exc:                 # pragma: no cover - defensive
            raise AttributeError(item) from exc
        if item in d:
            return d[item]
        raise AttributeError(item)

    # -------- ScalarView overrides (K/V narrative layout) ---------------
    def to_dict(self, cols=None) -> dict:
        row = dict(next(iter(self)))
        keep, warning = _validate_cols(cols, self)
        if warning:
            _warn(warning)
        return {k: row.get(k) for k in keep}

    def to_df(self, cols=None):
        pd = _import_pandas()
        row = self.to_dict(cols=cols)
        return pd.DataFrame([row])

    def to_html(self, cols=None) -> str:
        """KV-layout HTML — used by ``_repr_html_``. For file output, use ``save('x.html')``."""
        from ssdiff.results.format import fmt_cell
        row = self.to_dict(cols=cols)
        parts = ['<table class="kv">',
                 "<thead><tr><th>Metric</th><th>Value</th></tr></thead>", "<tbody>"]
        for k, v in row.items():
            parts.append(
                f"<tr><th>{_html.escape(k)}</th>"
                f"<td>{_html.escape(fmt_cell(v, k))}</td></tr>"
            )
        parts += ["</tbody>", "</table>"]
        return "\n".join(parts)

    def to_text(self, max_rows: int | None = None, cols=None) -> str:
        """K/V layout: first column is keys (left-aligned), second is values (right-aligned)."""
        from ssdiff.results.format import fmt_cell, fmt_table
        if max_rows is None:
            max_rows = DEFAULT_MAX_ROWS
        row = self.to_dict(cols=cols)
        rows_seq = [[k, fmt_cell(v, k)] for k, v in row.items()]
        return fmt_table(rows_seq, headers=["", ""],
                         numeric=[False, True], gutter=2)

    def _save_hint(self) -> str:
        cols_str = "   ".join(self._columns[:4])
        more = " …" if len(self._columns) > 4 else ""
        stem = self._filename_stem()
        return (f"Save:  .save()  .save('file.ext')  .save('file.ext', cols=[...])"
                f"   # default: {stem}.csv\n"
                f"       extensions: csv json xlsx md txt html tex docx\n"
                f"       .to_dict()  .to_df()\n"
                f"       cols=[{cols_str}{more}] → select & order keys")

    # save() uses ScalarView's single-row iter, producing one-row csv/xlsx/etc.


class TestView(ScalarView):
    """Unified runnable-test view — every Result type exposes `.test`.

    Attribute access shows the current test name + results.
    Calling reruns (optionally switching test) and updates parent state.
    Subclasses implement `_run(name, params)` returning (name, info_dict).
    `info_dict` must contain a ``pvalue`` key; other keys are test-specific.
    """

    _name = "test"
    _columns: tuple[str, ...] = ("name", "pvalue")
    _default_name: str | None = None

    def __init__(self, parent=None, *, name: str | None = None,
                 info: dict | None = None):
        """Initialise the view with an optional already-run test result.

        Parameters
        ----------
        parent : Result or None
            The owning result object; required for ``__call__`` reruns.
        name : str or None
            Test name from the last run (e.g. ``"split"``).
        info : dict or None
            Key/value result from the last run, must contain ``"pvalue"``.
        """
        super().__init__()
        self._parent = parent
        self._test_name = name
        self._info = dict(info or {})

    @property
    def name(self) -> str | None:
        """Name of the last test run (e.g. ``"split"``), or None if not yet run."""
        return self._test_name

    @property
    def pvalue(self):
        """P-value from the last test run, or None if not yet run."""
        return self._info.get("pvalue")

    @property
    def params(self) -> dict:
        """Non-pvalue entries from the last test run's info dict."""
        return {k: v for k, v in self._info.items() if k != "pvalue"}

    def __iter__(self):
        row = {"name": self._test_name}
        for c in self._columns[1:]:
            row[c] = self._info.get(c)
        yield row

    def __repr__(self) -> str:
        if not self._info:
            return (f"<{type(self).__name__} — no test run; "
                    f"call .test(...) to run>")
        return super().__repr__()

    def __call__(self, name: str | None = None, **params) -> TestView:
        """Run (or rerun) the statistical test, updating the view in-place.

        Parameters
        ----------
        name : str or None
            Test variant to run (e.g. ``"perm"``, ``"split"``).  Defaults to
            the previously used test name, then to ``_default_name``.
        **params
            Override test-specific parameters (e.g. ``n_perm=5000``).

        Returns
        -------
        TestView
            ``self`` — updated with the new result so chaining works.

        Raises
        ------
        RuntimeError
            If the view has been detached from its parent result.
        """
        if self._parent is None:
            raise RuntimeError(
                f"{type(self).__name__}() requires a parent result — "
                "view was detached."
            )
        resolved = name or self._test_name or self._default_name
        new_name, new_info = self._run(resolved, params)
        self._test_name = new_name
        self._info = new_info
        self._on_rerun()
        return self

    def _run(self, name: str | None, params: dict) -> tuple[str, dict]:
        """Execute the named test variant and return (name, info_dict). Subclass must implement."""
        raise NotImplementedError

    def _on_rerun(self) -> None:
        """Hook for updating parent state (stats, pairs, ...) after a rerun."""
        pass

    def _rerun_hint(self) -> str | None:
        """Return a ``'Rerun: .test(...)'`` hint line, or None if not applicable."""
        return None

    def _save_hint(self) -> str:
        base = super()._save_hint()
        rerun = self._rerun_hint()
        if rerun:
            return base + "\n" + rerun
        return base


class Result(ABC):  # noqa: B024 — marker base class; subclasses share infra, not abstract methods
    """Base class for all ssdiff results.

    Subclasses materialize paramless views as attributes in __init__
    and compute param views on demand through `_cache_get`.
    """

    embeddings: Any = None
    corpus: Any = None
    _access: tuple[str, ...] = ()
    _arrays: tuple[str, ...] = ()

    def __init__(self):
        self._cache: dict[tuple[str, tuple], View] = {}

    # -------- display hooks ----------------------------------------------
    def _summary(self) -> str:
        """Compact text header — subclass override."""
        return type(self).__name__

    def _summary_html(self) -> str:
        """Compact HTML header used by ``to_html()``."""
        return f"<p>{type(self).__name__}</p>"

    def _save_hint(self) -> str:
        """Return the save-hint footer text; subclasses override with something helpful."""
        return ""

    def _save_hint_html(self) -> str:
        """Return the save-hint footer as an HTML snippet; subclasses override."""
        return ""

    def _access_hint(self) -> str:
        """Render the arrays / views / methods discoverability block.

        Every entry is prefixed with ``.`` so the repr doubles as a
        copy-paste prompt (``result.beta``, ``result.test()``).
        ``_access`` entries with a ``(...)`` suffix render as methods;
        bare names render as views. Both ``test`` (a TestView) and
        ``test(...)`` (its callable rerun form) can co-exist in the list.
        """
        if not self._access and not self._arrays:
            return ""
        views: list[str] = []
        methods: list[str] = []
        for name in self._access:
            if "(" in name:
                methods.append("." + name)
            else:
                views.append("." + name)
        lines = []
        if self._arrays:
            lines.append("  arrays:  " + "  ".join("." + a for a in self._arrays))
        if views:
            lines.append("  views:   " + "  ".join(views))
        if methods:
            lines.append("  methods: " + "  ".join(methods))
        return "\n".join(lines)

    def to_text(self) -> str:
        """Return a compact plain-text summary (used by ``__repr__``, no citation block)."""
        parts = [self._summary()]
        access = self._access_hint()
        if access:
            parts.append(access)
        return "\n".join(parts)

    def to_html(self) -> str:
        """Return a compact HTML summary (used by ``_repr_html_``, no citation block)."""
        return self._summary_html()

    def __repr__(self) -> str:
        body = self.to_text()
        if _save_hint_enabled():
            hint = self._save_hint()
            if hint:
                return body + "\n\n" + hint
        return body

    def _repr_html_(self) -> str:
        body = self.to_html()
        if _save_hint_enabled():
            hint = self._save_hint_html()
            if hint:
                return body + "\n" + hint
        return body

    # -------- cache ------------------------------------------------------
    def _cache_get(self, name: str, params: dict,
                   compute: Callable[[], View]) -> View:
        """Return cached view for (name, params), computing and storing it on first call."""
        key = (name, _freeze_params(params))
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    def clear_cache(self, view: str | None = None) -> None:
        """Two-form cache clearing.

        `clear_cache()` drops everything; `clear_cache("clusters")` drops
        all entries whose view_name matches. No by-params form (D5).
        """
        if view is None:
            self._cache = {}
            return
        self._cache = {k: v for k, v in self._cache.items() if k[0] != view}

    # -------- attach -----------------------------------------------------
    def attach(self, corpus=None, embeddings=None) -> Result:
        """Re-wire corpus / embeddings references after un-pickling (D14)."""
        if corpus is not None:
            self.corpus = corpus
        if embeddings is not None:
            self.embeddings = embeddings
        return self

    def _require_resource(self, resource: str, view_name: str) -> None:
        """Raise with an actionable hint when an accessor needs a missing resource."""
        if getattr(self, resource, None) is None:
            raise RuntimeError(
                f"{view_name} needs {resource} but none is attached. "
                f"Call result.attach({resource}=...) first."
            )
