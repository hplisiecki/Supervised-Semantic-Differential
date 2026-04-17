"""LexiconResult — container for lexicon suggestion / evaluation output."""

from __future__ import annotations


class LexiconResult:
    """Per-token statistics with optional aggregate coverage summary.

    Returned by :meth:`Corpus.suggest_lexicon` and
    :meth:`Corpus.evaluate_lexicon`.

    Attributes
    ----------
    rows : list[dict]
        One dict per token with keys ``token``, ``freq``, ``cov_all``,
        ``cov_bal``, ``corr``, ``rank``, ``pvalue``, ``direction``.
    summary : dict or None
        Aggregate coverage (present when created via
        ``evaluate_lexicon``).  Keys: ``docs_any``, ``cov_all``,
        ``q1``, ``q4``, ``corr_any``, ``hits_mean``, ``hits_median``,
        ``types_mean``, ``types_median`` [, ``group_cov``].
    var_type : str
        ``"continuous"`` or ``"categorical"``.
    n_docs : int
        Number of documents after NaN filtering.
    """

    __slots__ = ("rows", "summary", "var_type", "n_docs")

    def __init__(
        self,
        rows: list[dict],
        *,
        summary: dict | None = None,
        var_type: str = "continuous",
        n_docs: int = 0,
    ) -> None:
        self.rows = rows
        self.summary = summary
        self.var_type = var_type
        self.n_docs = n_docs

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def tokens(self) -> list[str]:
        """Token strings in rank order."""
        return [r["token"] for r in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.summary is not None:
            return (
                f"LexiconResult({len(self.rows)} tokens, "
                f"cov={self.summary['cov_all']:.1%}, "
                f"n_docs={self.n_docs})"
            )
        return f"LexiconResult({len(self.rows)} tokens, n_docs={self.n_docs})"

    def report(self) -> str:
        """Print a formatted report and return it as a string."""
        text = self._build_report()
        print(text)
        return text

    def _build_report(self) -> str:
        parts: list[str] = []
        is_cat = self.var_type == "categorical"
        corr_label = "V" if is_cat else "r"

        # ── Per-token table ──────────────────────────────────────
        header = (
            f"{'Token':<20} {'Freq':>5} {'Cov':>6} "
            f"{'CovBal':>6} {corr_label:>7} {'Rank':>6} "
            f"{'p':>9} {'Dir'}"
        )
        sep = "─" * len(header)
        parts.append(sep)
        parts.append(header)
        parts.append(sep)

        for row in self.rows:
            parts.append(
                f"{row['token']:<20} {row['freq']:>5} "
                f"{row['cov_all']:>5.1%} "
                f"{row['cov_bal']:>5.1%} "
                f"{row['corr']:>+7.4f} "
                f"{row['rank']:>6.4f} "
                f"{row['pvalue']:>9.2e} "
                f"{row['direction']}"
            )

        parts.append(sep)

        # ── Coverage summary (when present) ──────────────────────
        if self.summary is not None:
            s = self.summary
            parts.append("")
            parts.append(
                f"  Docs with any hit : {s['docs_any']:,} / "
                f"{self.n_docs:,} ({s['cov_all']:.1%})"
            )

            if is_cat:
                parts.append(
                    f"  Min group cov     : {s['q1']:.1%}  |  "
                    f"Max group cov : {s['q4']:.1%}"
                )
                parts.append(
                    f"  Cramér's V (any)  : {s['corr_any']:.4f}"
                )
                gcov = s.get("group_cov", {})
                if gcov:
                    items = "  ".join(
                        f"{g}={v:.1%}" for g, v in gcov.items()
                    )
                    parts.append(f"  Per-group cov     : {items}")
            else:
                parts.append(
                    f"  Q1 coverage       : {s['q1']:.1%}  |  "
                    f"Q4 coverage : {s['q4']:.1%}"
                )
                parts.append(
                    f"  Corr (any hit)    : {s['corr_any']:+.4f}"
                )

            parts.append(
                f"  Hits/doc          : mean={s['hits_mean']:.2f}, "
                f"median={s['hits_median']:.1f}"
            )
            parts.append(
                f"  Types/doc         : mean={s['types_mean']:.2f}, "
                f"median={s['types_median']:.1f}"
            )
            parts.append(sep)

        return "\n".join(parts)
