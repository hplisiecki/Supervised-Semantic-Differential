"""Display-layer scaffolding for ssdiff results.

Holds the module-level ``set_repr_hints`` flag, default-column / default-
row-cap registries used for truncation and column selection, and the
``_save_hint_enabled`` helper used by ``__repr__`` / ``_repr_html_``
across ``Result``, ``View``, ``ScalarView``, and ``Report``.
"""

from __future__ import annotations

_REPR_HINTS_ENABLED: bool = True


def set_repr_hints(enabled: bool) -> None:
    """Toggle save-hint footers in __repr__ / _repr_html_ output.

    Default: True. Disable for log-stream consumers (SSD_APP).
    """
    if not isinstance(enabled, bool):
        raise TypeError(f"enabled must be bool, got {type(enabled).__name__}")
    global _REPR_HINTS_ENABLED
    _REPR_HINTS_ENABLED = enabled


def _save_hint_enabled() -> bool:
    """Return True if save-hint footers should appear in repr output."""
    return _REPR_HINTS_ENABLED


# ---------------- defaults registry ------------------------------------------
# Keyed by view class name (``__name__``, not ``_name``) so that sibling views
# that share a ``_name`` — e.g. StatsView vs OLSStatsView — can diverge. Views
# without an entry fall through to full ``_columns`` via ``View._default_cols``.
# Values per docs/results_tables.md.
DEFAULT_COLS: dict[str, tuple[str, ...]] = {
    # continuous
    "StatsView":           ("backend", "r2", "pvalue", "n_kept", "iqr_effect"),
    "OLSStatsView":        ("backend", "r2", "r2_adj", "pvalue", "n_kept", "iqr_effect"),
    "FitInfoView":         ("n_components", "pca_k", "p_method", "random_state"),
    "WordsView":           ("side", "rank", "word", "cos_beta"),
    "WordsViewSided":      ("side", "rank", "word", "cos_beta"),
    "ClustersViewSided":   ("cluster_id", "size", "coherence", "centroid_cos_beta"),
    "ClusterWordsView":    ("cluster_id", "word", "cos_centroid", "cos_beta"),
    "SnippetsView":        ("side", "doc_id", "cosine", "seed", "text_window"),
    "SnippetsViewSided":   ("side", "doc_id", "cosine", "seed", "text_window"),
    "PLSTestView":         ("name", "pvalue", "split_r2"),
    "SweepView":           ("k", "var_explained", "interp_auck",
                            "stab_auck_raw", "joint_score"),
    # group
    "GroupStatsView":      ("G", "n_kept", "pvalue"),
    "GroupTestView":       ("name", "pvalue", "omnibus_T"),
    "PairsListView":       ("contrast", "T", "p_corrected", "cohens_d"),
    "PairStatsView":       ("T", "p_corrected", "cohens_d", "n_g1", "n_g2"),
    # lexicon
    "SuggestionsView":     ("rank", "token", "freq", "corr", "pvalue", "direction"),
    "SummaryView":         ("docs_any", "cov_all", "corr_any", "hits_median"),
}

DEFAULT_MAX_ROWS: int = 20

# Per-class overrides for ``to_text`` row cap; unlisted views fall back to the
# module-level ``DEFAULT_MAX_ROWS``.  Useful when a view's rows are expensive
# to read (long text, many columns) or naturally wider than average.
DEFAULT_MAX_ROWS_BY_CLASS: dict[str, int] = {
    "SnippetsView":       10,
    "SnippetsViewSided":  10,
    "DocsView":           50,
}

# Per-class terminal cell truncation (chars) — clips long text columns in
# ``to_text``/repr only; data exports stay full-width.  ``None`` / missing = no
# truncation.  Keyed by class ``__name__``, like the other registries.
DEFAULT_TEXT_TRUNCATE: dict[str, int] = {
    "SnippetsView":       40,
    "SnippetsViewSided":  40,
}


# ---------------- supported save() extensions -------------------------------
# ``TABULAR_EXTS`` → extensions accepted by tabular ``View.save``; ``NARRATIVE_EXTS``
# → extensions accepted by ``Report.save``.  Exposed here so save-hint helpers,
# ``_check_ext`` in ``core.py``, and ``Report._SUPPORTED_EXTS`` all share one source.
TABULAR_EXTS: tuple[str, ...] = (
    "csv", "json", "xlsx", "md", "txt", "html", "tex", "docx",
)
NARRATIVE_EXTS: tuple[str, ...] = (
    "md", "txt", "html", "tex", "docx", "json",
)


# ---------------- default filename / extension ------------------------------
# Centralises ``save()`` defaults, mirroring ``DEFAULT_COLS``. Each entry is a
# full ``'stem.ext'`` string so narrative outputs (scalar k/v views and
# ``Report`` → ``.md``) and tabular views (words / clusters / … → ``.csv``)
# diverge per class. Sided variants use a ``{side}`` placeholder, filled from
# the instance's ``_side`` / ``_side_key`` attribute at save time. Classes
# without an entry fall through to ``f'{self._name}.{DEFAULT_EXT}'``.
DEFAULT_EXT: str = "csv"

DEFAULT_FILENAMES: dict[str, str] = {
    # narrative (Report + single-row scalar views)
    "Report":              "report.md",
    "StatsView":           "stats.md",
    "OLSStatsView":        "stats.md",
    "FitInfoView":         "fit_info.md",
    "DocDetailView":       "doc.md",
    "PLSTestView":         "pls_test.md",
    "PCAOLSTestView":      "pcaols_test.md",
    "GroupStatsView":      "group_stats.md",
    "PairStatsView":       "pair_stats.md",
    "GroupTestView":       "group_test.md",
    "LexiconStatsView":    "lexicon_stats.md",
    "SummaryView":         "lexicon_summary.md",
    # tabular — continuous
    "WordsView":           "words.csv",
    "WordsViewSided":      "words_{side}.csv",
    "ClusterWordsView":    "cluster_words.csv",
    "ClustersViewSided":   "clusters_{side}.csv",
    "SnippetsView":        "snippets.csv",
    "SnippetsViewSided":   "snippets_{side}.csv",
    "DocsView":            "docs.csv",
    "SweepView":           "sweep.csv",
    # tabular — group
    "PairsListView":       "pairs.csv",
    # tabular — lexicon
    "SuggestionsView":     "lexicon_suggestions.csv",
}


# ---------------- save-hint helpers -----------------------------------------
# Centralised footer builders used by ``View.__repr__`` / ``_repr_html_`` /
# ``Report.__repr__``.  Helpers live here (next to the registries they read
# from) so every class shares one phrasing and the list of supported
# extensions / default filename are never duplicated.

def build_view_save_hint(view) -> str:
    """Return the save-hint footer for a tabular ``View``."""
    cols_preview = "   ".join(repr(c) for c in view._columns)
    return (f"Save:  .save() or with args  .save('{view._default_filename()}', cols=[...])\n"
            f"       extensions: {' '.join(TABULAR_EXTS)}\n"
            f"       .to_df()  .to_records()  .to_dict()\n"
            f"       cols=[{cols_preview}] → select & order columns")


def build_scalar_save_hint(view) -> str:
    """Return the save-hint footer for a single-row ``ScalarView``."""
    cols_str = "   ".join(view._columns)
    return (f"Save:  .save() or with args: .save('{view._default_filename()}', cols=[...])\n"
            f"       extensions: {' '.join(TABULAR_EXTS)}\n"
            f"       .to_dict()  .to_df()\n"
            f"       cols=[{cols_str}] → select & order keys")


def build_report_save_hint(report) -> str:
    """Return the save-hint footer for a ``Report``."""
    return (f"Save:  .save()  .save('{report._default_filename()}')   "
            f"# default: {report._default_filename()}; "
            f"extensions: {' '.join(NARRATIVE_EXTS)}")
