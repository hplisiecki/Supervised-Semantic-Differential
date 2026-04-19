"""LexiconResult — view wrapper for suggest_lexicon / evaluate_lexicon output."""

from __future__ import annotations

from collections.abc import Iterator

from ssdiff.results.core import Result, ScalarView, View
from ssdiff.results.format import fmt_count, fmt_d, fmt_p, fmt_pct, fmt_r
from ssdiff.results.report import Report, Section
from ssdiff.results.schema import Suggestion, Summary

# ---------- LexiconStatsView (ScalarView) ----------

class LexiconStatsView(ScalarView):
    """ScalarView exposing basic dataset metadata for a LexiconResult."""

    _name = "stats"
    _columns = ("var_type", "n_docs", "n_tokens")

    def __init__(self, *, var_type: str, n_docs: int, n_tokens: int):
        super().__init__()
        self._var_type = var_type
        self._n_docs = n_docs
        self._n_tokens = n_tokens

    def __iter__(self):
        yield {
            "var_type": self._var_type,
            "n_docs": self._n_docs,
            "n_tokens": self._n_tokens,
        }


# ---------- SummaryView (ScalarView) ----------

class SummaryView(ScalarView):
    """ScalarView exposing aggregate coverage statistics from ``evaluate_lexicon``."""

    _name = "summary"
    _columns = (
        "docs_any", "cov_all", "q1", "q4", "corr_any",
        "hits_mean", "hits_median", "types_mean", "types_median", "group_cov",
    )

    def __init__(self, summary: Summary):
        super().__init__()
        self._summary = summary

    def __iter__(self):
        s = self._summary
        yield {f: getattr(s, f) for f in self._columns}


# ---------- SuggestionsView ----------

class SuggestionsView(View[Suggestion]):
    """Tabular view of candidate lexicon tokens sorted by the combined ranking score."""

    _name = "suggestions"
    _columns = ("token", "freq", "cov_all", "cov_bal", "corr", "pvalue", "direction", "rank")

    def __init__(self, rows: list[Suggestion], *, _no_trunc: bool = False):
        super().__init__(_no_trunc=_no_trunc)
        self._rows = rows

    def __iter__(self) -> Iterator[Suggestion]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return SuggestionsView(self._rows[i], _no_trunc=True)
        return self._rows[i]


# ---------- LexiconResult ----------

class LexiconResult(Result):
    """Result from Corpus.suggest_lexicon() or Corpus.evaluate_lexicon().

    Attributes
    ----------
    stats : LexiconStatsView
        ScalarView exposing var_type, n_docs, n_tokens.
    suggestions : SuggestionsView
        View of Suggestion rows in rank order.
    summary : SummaryView or None
        Aggregate coverage (present only after evaluate_lexicon).
    tokens : list[str]
        Token strings in rank order.
    """

    def __init__(
        self,
        *,
        var_type: str,
        n_docs: int,
        n_tokens: int,
        suggestions: list[Suggestion],
        summary: Summary | None = None,
        corpus=None,
    ):
        super().__init__()
        self.corpus = corpus

        self.stats = LexiconStatsView(
            var_type=var_type, n_docs=n_docs, n_tokens=n_tokens,
        )
        self.suggestions = SuggestionsView(suggestions)
        self.summary = SummaryView(summary) if summary is not None else None

    # -------- convenience property ----------------------------------------

    @property
    def tokens(self) -> list[str]:
        """Token strings in rank order."""
        return [s.token for s in self.suggestions]

    _access = (
        "stats", "suggestions", "tokens", "summary", "report()",
    )

    def _summary(self) -> str:
        return (f"LexiconResult  n_tokens={self.stats.n_tokens}  "
                f"n_docs={fmt_count(self.stats.n_docs)}")

    def _summary_html(self) -> str:
        return (f"<p><b>LexiconResult</b> n_tokens={self.stats.n_tokens} · "
                f"n_docs={fmt_count(self.stats.n_docs)}</p>")

    def _save_hint(self) -> str:
        return (
            "Save:  lex.report().save('lexicon.md')            # narrative\n"
            "       lex.suggestions.save('suggestions.csv')    # data"
        )

    def _save_hint_html(self) -> str:
        return f"<pre class='ssd-save-hint'>{self._save_hint()}</pre>"

    # -------- report ------------------------------------------------------

    def report(self, top: int = 20) -> Report:
        """Build a narrative Report for this lexicon result.

        Parameters
        ----------
        top : int
            Maximum number of suggestions to include in the table.

        Returns
        -------
        Report
            A ``Report`` with a stats section, a suggestions table, and (when
            available) a coverage-summary section from ``evaluate_lexicon``.
        """
        sections = []

        # Stats section
        s = self.stats
        sections.append(Section(title="Stats", kind="kv", rows=[
            ("var_type", s.var_type),
            ("n_docs", fmt_count(s.n_docs)),
            ("n_tokens", fmt_count(s.n_tokens)),
        ]))

        # Suggestions table
        rows = []
        for sug in list(self.suggestions)[:top]:
            rows.append([
                sug.token,
                fmt_count(sug.freq),
                fmt_pct(sug.cov_all),
                fmt_pct(sug.cov_bal),
                fmt_r(sug.corr, signed=True),
                fmt_p(sug.pvalue),
                sug.direction,
                fmt_r(sug.rank),
            ])
        if rows:
            sections.append(Section(
                title=f"Suggestions (top {min(top, len(rows))})",
                kind="table",
                headers=["token", "freq", "cov_all", "cov_bal", "corr", "pvalue", "dir", "rank"],
                rows=rows,
                numeric=[False, True, True, True, True, True, False, True],
            ))

        # Summary section (only when evaluate_lexicon built the result)
        if self.summary is not None:
            sm = self.summary
            kv_rows = [
                ("docs_any", fmt_count(sm.docs_any)),
                ("cov_all", fmt_pct(sm.cov_all)),
                ("q1", fmt_pct(sm.q1)),
                ("q4", fmt_pct(sm.q4)),
                ("corr_any", fmt_r(sm.corr_any, signed=True)),
                ("hits_mean", fmt_d(sm.hits_mean)),
                ("hits_median", fmt_d(sm.hits_median)),
                ("types_mean", fmt_d(sm.types_mean)),
                ("types_median", fmt_d(sm.types_median)),
            ]
            if sm.group_cov is not None:
                kv_rows.append(("group_cov", str(sm.group_cov)))
            sections.append(Section(title="Coverage summary", kind="kv", rows=kv_rows))

        return Report(
            title="LexiconResult",
            subtitle=f"(n_docs = {fmt_count(self.stats.n_docs)})",
            sections=sections,
            cite=False,
        )
