"""Report builder + renderers.

A ``Report`` is a multi-section builder. Public surface:

- ``repr(report)`` / ``report._repr_html_()`` — console rendering
- ``report.to_text()`` / ``report.to_html()`` — return strings
- ``report.save(path)`` — file output, extension dispatch
  (``.md .txt .html .tex .docx .json``)
"""

from __future__ import annotations

import html as _html
import warnings
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from ssdiff.results.core import _require
from ssdiff.results.display import (
    DEFAULT_EXT,
    DEFAULT_FILENAMES,
    NARRATIVE_EXTS,
    build_report_save_hint,
)
from ssdiff.results.format import fmt_table

SectionKind = Literal["kv", "table", "list"]


def _md_cell(value) -> str:
    """Stringify and escape a value for use inside a Markdown table cell."""
    text = str(value).replace("|", r"\|")
    return text.replace("\n", "<br>")


def _build_cluster_section(
    title: str,
    clusters_view,
    *,
    n_clusters: int,
    n_words: int,
    n_snippets: int,
    snippet_provider,
) -> "Section":
    """Build a Section with No./Size/Top Words/Representative Excerpt columns.

    ``clusters_view`` is a ``ClustersViewSided`` exposing ``_words_rows``.
    ``snippet_provider`` is a no-arg callable returning a side-scoped snippets
    view (must expose ``_all_side_rows``) or ``None`` when snippets are
    unavailable (no corpus, or extraction failed). When ``n_snippets <= 0`` or
    the provider returns ``None``, the "Representative Excerpt" column is
    omitted entirely.
    """
    words_by_cid: dict[int, list] = {}
    for cw in clusters_view._words_rows:
        words_by_cid.setdefault(cw.cluster_id, []).append(cw)

    include_excerpt = False
    snips_by_cid: dict[int, list] = {}
    if n_snippets > 0:
        sn_view = snippet_provider()
        if sn_view is not None:
            for sn in sn_view._all_side_rows:
                if sn.cluster_id is None:
                    continue
                snips_by_cid.setdefault(sn.cluster_id, []).append(sn)
            include_excerpt = True

    rows = []
    for i, c in enumerate(list(clusters_view)[:n_clusters], start=1):
        words_sorted = sorted(
            words_by_cid.get(c.cluster_id, []),
            key=lambda w: w.cos_centroid, reverse=True,
        )
        top_words_str = ", ".join(w.word for w in words_sorted[:n_words])
        row = [i, c.size, top_words_str]
        if include_excerpt:
            cluster_snips = sorted(
                snips_by_cid.get(c.cluster_id, []),
                key=lambda s: s.cosine, reverse=True,
            )[:n_snippets]
            row.append(" / ".join(s.text_window for s in cluster_snips))
        rows.append(row)

    headers = ["No.", "Size", "Top Words"]
    numeric = [True, True, False]
    if include_excerpt:
        headers.append("Representative Excerpt")
        numeric.append(False)

    return Section(
        title=title, kind="table",
        headers=headers, rows=rows, numeric=numeric,
    )


def _resolve_section(value, defaults: dict, *, name: str) -> dict | None:
    """Normalize a ``report()`` section argument.

    Accepts ``True``/``False``/``None``/``dict``. ``True`` returns
    ``defaults``; a dict is merged onto ``defaults``; ``False``/``None``
    returns ``None`` (skip section). ``int`` and any other type raise
    ``TypeError`` — use ``True`` or a dict like ``{"n": 10}``.
    """
    if value is None or value is False:
        return None
    if value is True:
        return dict(defaults)
    if isinstance(value, dict):
        unknown = set(value) - set(defaults)
        if unknown:
            raise TypeError(
                f"{name}: unknown keys {sorted(unknown)}; "
                f"valid keys are {sorted(defaults)}"
            )
        return {**defaults, **value}
    raise TypeError(
        f"{name} must be True, False, None, or a dict like "
        f"{{'n': 10}} — got {type(value).__name__} ({value!r})"
    )


@dataclass
class Section:
    """One logical section within a Report.

    Fields
    ------
    title : str
        Section heading rendered as ``<h2>`` in HTML / ``## `` in Markdown.
    kind : SectionKind
        Layout type: ``"kv"`` for key-value pairs, ``"table"`` for a
        multi-column table, ``"list"`` for a bullet list.
    rows : list
        Row data.  For ``"kv"``: list of ``(key, value)`` tuples.
        For ``"table"``: list of lists (one per row).
        For ``"list"``: list of strings.
    headers : list of str or None
        Column headers for ``"table"`` sections; unused otherwise.
    numeric : list of bool or None
        Column alignment for ``"table"`` sections — ``True`` right-aligns.
    """
    title: str
    kind: SectionKind
    rows: list = field(default_factory=list)
    headers: list[str] | None = None
    numeric: list[bool] | None = None


@dataclass
class Report:
    """Multi-section report builder with format-agnostic rendering.

    Use the ``save(path)`` method to write to disk; format is inferred from
    the file extension.  ``to_text()`` / ``to_html()`` return strings directly.

    Fields
    ------
    title : str
        Main report heading.
    sections : list of Section
        Ordered content sections.
    subtitle : str or None
        Optional subtitle rendered below the title.
    """
    title: str
    sections: list[Section]
    subtitle: str | None = None

    _SUPPORTED_EXTS: ClassVar[tuple[str, ...]] = NARRATIVE_EXTS

    # -------- repr dunders ----------------------------------------------
    def _default_filename(self) -> str:
        """Return the default ``'stem.ext'`` filename for ``save(path=None)``."""
        return DEFAULT_FILENAMES.get(type(self).__name__, f"report.{DEFAULT_EXT}")

    def _save_hint(self) -> str:
        return build_report_save_hint(self)

    def _save_hint_html(self) -> str:
        return f"<pre class='ssd-save-hint'>{_html.escape(self._save_hint())}</pre>"

    def __repr__(self) -> str:
        from ssdiff.results.display import _save_hint_enabled
        body = self.to_text()
        if _save_hint_enabled():
            return body + "\n\n" + self._save_hint()
        return body

    def _repr_html_(self) -> str:
        from ssdiff.results.display import _save_hint_enabled
        body = self.to_html()
        if _save_hint_enabled():
            return body + "\n" + self._save_hint_html()
        return body

    # -------- string renderers (used by repr + save) --------------------
    def to_text(self) -> str:
        """Render the report as a fixed-width text string."""
        parts = [self._text_header()]
        for s in self.sections:
            parts.append(self._text_section(s))
        return "\n\n".join(parts)

    def to_html(self) -> str:
        """Render the report as an HTML string."""
        parts = ['<div class="ssd-report">']
        parts.append(f"<h1>{_html.escape(self.title)}</h1>")
        if self.subtitle:
            parts.append(f"<p class='subtitle'>{_html.escape(self.subtitle)}</p>")
        for s in self.sections:
            parts.append(f"<h2>{_html.escape(s.title)}</h2>")
            parts.append(self._html_section_body(s))
        parts.append("</div>")
        return "\n".join(parts)

    def _to_markdown(self) -> str:
        """Render the report as a GitHub-flavoured Markdown string."""
        lines = [f"# {self.title}"]
        if self.subtitle:
            lines.append(self.subtitle)
        for s in self.sections:
            lines.append(f"## {s.title}")
            lines.append(self._md_section_body(s))
        return "\n\n".join(lines)

    def _to_latex(self) -> str:
        """Render the report as a LaTeX fragment (booktabs-style tables)."""
        parts = [f"\\section*{{{self.title}}}"]
        if self.subtitle:
            parts.append(self.subtitle)
        for s in self.sections:
            parts.append(f"\\subsection*{{{s.title}}}")
            parts.append(self._latex_section_body(s))
        return "\n\n".join(parts)

    def _to_json(self) -> str:
        """Render the report as a JSON string (title + sections array)."""
        import json
        payload = {
            "title": self.title,
            "subtitle": self.subtitle,
            "sections": [
                {"title": s.title, "kind": s.kind,
                 "headers": s.headers, "rows": s.rows}
                for s in self.sections
            ],
        }
        return json.dumps(payload, ensure_ascii=False, default=str, indent=2)

    def _write_docx(self, path: str) -> None:
        """Write the report as a .docx file using python-docx."""
        docx = _require("docx", extra="results")
        doc = docx.Document()
        doc.add_heading(self.title, level=1)
        if self.subtitle:
            doc.add_paragraph(self.subtitle)
        for s in self.sections:
            doc.add_heading(s.title, level=2)
            self._docx_section_body(doc, s)
        doc.save(path)

    # -------- unified save() --------------------------------------------
    def save(self, path: str | None = None) -> None:
        """Write the report to ``path``; format inferred from the extension.

        Supported: ``.md .txt .html .tex .docx .json``. When ``path`` is
        omitted, defaults to ``<cwd>/<default-filename>`` from
        ``DEFAULT_FILENAMES`` (``"report.md"`` for ``Report``).
        """
        if path is None:
            path = self._default_filename()
        path = str(path)
        ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
        if ext not in self._SUPPORTED_EXTS:
            raise ValueError(
                f"Unsupported extension .{ext} for Report.save(). "
                f"Supported: {self._SUPPORTED_EXTS}"
            )

        if ext == "docx":
            self._write_docx(path)
            return

        renderer = {
            "txt": self.to_text,
            "md": self._to_markdown,
            "html": self.to_html,
            "tex": self._to_latex,
            "json": self._to_json,
        }[ext]
        with open(path, "w", encoding="utf-8") as f:
            f.write(renderer())

    # -------- section helpers -------------------------------------------
    def _text_header(self) -> str:
        """Return the title/subtitle/separator header for text output."""
        h = self.title
        if self.subtitle:
            h += f"\n{self.subtitle}"
        h += "\n" + "─" * 80
        return h

    def _text_section(self, s: Section) -> str:
        """Render a single Section as plain text (kv / table / list)."""
        lines = [s.title]
        if s.kind == "kv":
            kv = [[k, v] for k, v in s.rows]
            lines.append(fmt_table(kv, headers=["", ""],
                                   numeric=[False, True], gutter=2))
        elif s.kind == "table":
            lines.append(fmt_table(s.rows,
                                   headers=s.headers or [],
                                   numeric=s.numeric or [False] * len(s.headers or []),
                                   max_width=80, text_truncate=70))
        elif s.kind == "list":
            for item in s.rows:
                lines.append(f"  {item}")
        return "\n".join(lines)

    def _md_section_body(self, s: Section) -> str:
        """Render the body of one Section as Markdown."""
        if s.kind == "kv":
            rows = "\n".join(f"| {_md_cell(k)} | {_md_cell(v)} |" for k, v in s.rows)
            return f"| Metric | Value |\n|:-------|------:|\n{rows}"
        if s.kind == "table":
            header = "| " + " | ".join(s.headers or []) + " |"
            sep = "|" + "|".join(["---"] * len(s.headers or [])) + "|"
            body = "\n".join(
                "| " + " | ".join(_md_cell(c) for c in r) + " |"
                for r in s.rows
            )
            return "\n".join([header, sep, body])
        if s.kind == "list":
            return "\n".join(f"- {item}" for item in s.rows)
        return ""

    def _html_section_body(self, s: Section) -> str:
        """Render the body of one Section as an HTML fragment."""
        if s.kind == "kv":
            rows = "".join(
                f"<tr><th>{_html.escape(str(k))}</th>"
                f"<td>{_html.escape(str(v))}</td></tr>"
                for k, v in s.rows
            )
            return f"<table class='kv'>{rows}</table>"
        if s.kind == "table":
            thead = "".join(f"<th>{_html.escape(h)}</th>" for h in (s.headers or []))
            rows = "".join(
                "<tr>" + "".join(f"<td>{_html.escape(str(c))}</td>" for c in r) + "</tr>"
                for r in s.rows
            )
            return f"<table><thead><tr>{thead}</tr></thead><tbody>{rows}</tbody></table>"
        if s.kind == "list":
            items = "".join(f"<li>{_html.escape(str(i))}</li>" for i in s.rows)
            return f"<ul>{items}</ul>"
        return ""

    def _latex_section_body(self, s: Section) -> str:
        """Render the body of one Section as a LaTeX tabular / list fragment."""
        if s.kind == "kv":
            body = "\n".join(f"{k} & {v} \\\\" for k, v in s.rows)
            return ("\\begin{tabular}{ll}\n\\toprule\nMetric & Value \\\\\n"
                    f"\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}")
        if s.kind == "table":
            cols = "l" * len(s.headers or [])
            head = " & ".join(s.headers or []) + r" \\"
            body = "\n".join(" & ".join(str(c) for c in r) + r" \\" for r in s.rows)
            return (f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head}\n"
                    f"\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}")
        if s.kind == "list":
            items = "\n".join(f"\\item {i}" for i in s.rows)
            return f"\\begin{{itemize}}\n{items}\n\\end{{itemize}}"
        return ""

    def _docx_section_body(self, doc, s: Section) -> None:
        """Append the body of one Section to a python-docx Document object."""
        if s.kind == "kv":
            t = doc.add_table(rows=len(s.rows), cols=2)
            for i, (k, v) in enumerate(s.rows):
                t.rows[i].cells[0].text = str(k)
                t.rows[i].cells[1].text = str(v)
        elif s.kind == "table":
            headers = s.headers or []
            t = doc.add_table(rows=1 + len(s.rows), cols=len(headers))
            for j, h in enumerate(headers):
                t.rows[0].cells[j].text = h
            for i, r in enumerate(s.rows, start=1):
                for j, c in enumerate(r):
                    t.rows[i].cells[j].text = str(c)
        elif s.kind == "list":
            for item in s.rows:
                doc.add_paragraph(str(item), style="List Bullet")

    @staticmethod
    def _warn_if_cols(cols) -> None:
        """Emit a UserWarning if ``cols`` is not None (Report renderers ignore it)."""
        if cols is not None:
            warnings.warn(
                "Report renderers ignore `cols`; rendering all content.",
                UserWarning, stacklevel=3,
            )
