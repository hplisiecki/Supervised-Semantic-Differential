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
    "SidedWordsView":      ("side", "rank", "word", "cos_beta"),
    "SidedClustersView":   ("cluster_id", "size", "coherence", "centroid_cos_beta"),
    "ClusterWordsView":    ("cluster_id", "word", "cos_centroid", "cos_beta"),
    "SnippetsView":        ("side", "doc_id", "cosine", "seed", "text_window"),
    "SidedSnippetsView":   ("side", "doc_id", "cosine", "seed", "text_window"),
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
