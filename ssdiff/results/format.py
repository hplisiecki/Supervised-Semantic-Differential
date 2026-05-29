"""APA-inspired formatting primitives (en_US locale).

D11 contract — numeric formatting rules applied uniformly across all
text/markdown/html/latex/docx renderers. Unit-tested in
tests/test_results_format.py against the exact strings in the spec.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

MINUS = "\u2212"        # U+2212 MINUS SIGN
ELLIPSIS = "\u2026"     # U+2026 HORIZONTAL ELLIPSIS


def _no_leading_zero(x: float, digits: int) -> str:
    """`.42`, `−.18`, `1.00` — strip leading zero when |x| < 1."""
    if not math.isfinite(x):
        return "nan"
    s = f"{abs(x):.{digits}f}"
    if abs(x) < 1:
        s = s.lstrip("0")
    if x < 0:
        return f"{MINUS}{s}"
    return s


def fmt_p(p: float) -> str:
    """P-value: `<.001` if p<.001, else 3 decimals, no leading zero."""
    if not math.isfinite(p):
        return "nan"
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}".lstrip("0")


def fmt_r(r: float, *, digits: int = 2, signed: bool = False) -> str:
    """Bounded coefficient (|r| ≤ 1): 2–3 decimals, no leading zero.

    Parameters
    ----------
    r : float
        Correlation or cosine value (expected in ``[−1, 1]``).
    digits : int
        Decimal places (default 2).
    signed : bool
        If ``True``, prefix a ``+`` sign for non-negative values.
    """
    s = _no_leading_zero(r, digits)
    if signed and not s.startswith(MINUS):
        s = f"+{s}"
    return s


def fmt_d(x: float, *, digits: int = 2) -> str:
    """Unbounded scalar (Cohen's d, ‖β‖): leading zero kept."""
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


PVALUE_COLUMNS: frozenset[str] = frozenset({
    "pvalue", "p_value", "p", "p_raw", "p_corrected",
})


def fmt_cell(value, column: str | None = None, digits: int = 3) -> str:
    """Column-aware display formatter.

    Pvalue columns (see PVALUE_COLUMNS) render through fmt_p so tiny values
    collapse to `<.001` instead of leaking as `2.36e-07`. Everything else
    falls through to fmt_sig.
    """
    if (
        column in PVALUE_COLUMNS
        and value is not None
        and not isinstance(value, str)
    ):
        try:
            return fmt_p(float(value))
        except (TypeError, ValueError):
            pass
    return fmt_sig(value, digits=digits)


def fmt_sig(x, digits: int = 3) -> str:
    """Display-side formatter: floats → `digits` significant figures.

    Integers render as ints; strings pass through; None → "".
    Used by repr/html paths so raw floats don't leak as `0.3014621812…`.
    Data exports (to_df/to_csv/to_json) never go through this.
    """
    if x is None:
        return ""
    if isinstance(x, bool):
        return str(x)
    if hasattr(x, "__index__"):
        return str(int(x))
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(xf):
        if math.isnan(xf):
            return "nan"
        return "inf" if xf > 0 else "-inf"
    return f"{xf:.{digits}g}"


def fmt_count(n: int) -> str:
    """Integer with thousands separator."""
    return f"{int(n):,}"


def fmt_pct(x: float, *, digits: int = 1) -> str:
    """Percent (1 decimal)."""
    return f"{x * 100:.{digits}f}%"


def truncate(s: str, max_chars: int) -> str:
    """Clip text to `max_chars` characters, appending `…`."""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + ELLIPSIS


def fmt_table(
    rows: Sequence[Sequence],
    *,
    headers: Sequence[str],
    numeric: Sequence[bool],
    gutter: int = 2,
    max_width: int | None = None,
    text_truncate: int | None = None,
) -> str:
    """Render a fixed-width table with D11 alignment rules.

    Parameters
    ----------
    rows : sequence of sequences
        Row data; each inner sequence must have the same length as ``headers``.
    headers : sequence of str
        Column headers.
    numeric : sequence of bool
        ``True`` right-aligns the corresponding column; ``False`` left-aligns.
    gutter : int
        Number of spaces between columns (default 2).
    max_width : int or None
        Advisory total width — currently unused in enforcement but signals intent.
    text_truncate : int or None
        If set, clip every cell to this many characters (appending ``…``).

    Returns
    -------
    str
        Multi-line string with a header row followed by data rows.
    """
    if text_truncate is not None:
        rows = [
            [truncate(str(c), text_truncate) for c in r]
            for r in rows
        ]

    str_rows = [[str(c) for c in r] for r in rows]
    widths = [
        max(len(headers[i]),
            max((len(r[i]) for r in str_rows), default=0))
        for i in range(len(headers))
    ]
    gap = " " * gutter

    def _fmt_cell(cell: str, width: int, right: bool) -> str:
        """Pad ``cell`` to ``width`` characters, right- or left-aligned."""
        return cell.rjust(width) if right else cell.ljust(width)

    out_lines = [gap.join(_fmt_cell(headers[i], widths[i], numeric[i]) for i in range(len(headers)))]
    for r in str_rows:
        out_lines.append(gap.join(_fmt_cell(r[i], widths[i], numeric[i]) for i in range(len(headers))))
    if not str_rows:
        out_lines.append("(empty)")
    return "\n".join(out_lines)


def default_alignment(n_cols: int) -> tuple[bool, ...]:
    """First column left-aligned, all subsequent columns right-aligned.

    Returns a tuple suitable for `fmt_table(numeric=...)`. Per
    spec_console_repr.md: labels live in column 0 and read better
    flush-left; remaining columns are numeric or short codes that line
    up better right-aligned.
    """
    if n_cols <= 0:
        return ()
    return (False,) + (True,) * (n_cols - 1)
