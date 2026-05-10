"""ContinuousResult + PLSResult + PCAOLSResult."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ssdiff.results.core import ScalarView, TestView, View
from ssdiff.results.display import _save_hint_enabled
from ssdiff.results.single_result import _SingleResult
from ssdiff.results.format import (
    fmt_count,
    fmt_d,
    fmt_p,
    fmt_r,
)
from ssdiff.results.report import Report, Section
from ssdiff.results.schema import (
    Cluster,
    ClusterWord,
    Doc,
    FitInfo,
    Snippet,
    Stats,
    Word,
)


def _rolling_median(x: np.ndarray, window: int = 7) -> np.ndarray:
    """Rolling median with NaN-awareness. Used for sweep plot smoothing."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        w = x[lo:hi]
        w = w[np.isfinite(w)]
        if len(w):
            out[i] = float(np.median(w))
    return out


# ---------- FitInfoView (ScalarView) ----------
class FitInfoView(ScalarView):
    """ScalarView exposing backend fit configuration from ``FitInfo``."""

    _name = "fit_info"
    _columns = (
        "n_components", "pca_k", "p_at_k", "n_splits",
        "split_mean_r", "random_state",
        "k_min", "k_max", "k_step", "best_k", "pca_k_source",
    )

    def __init__(self, info: FitInfo):
        super().__init__()
        self._info = info

    def __iter__(self):
        yield {f: getattr(self._info, f) for f in self._columns}


# ---------- StatsView (ScalarView) ----------
class StatsView(ScalarView):
    """ScalarView exposing model-quality statistics from ``Stats`` (PLS backend)."""

    _name = "stats"
    _columns = (
        "backend", "r2", "pvalue",
        "n_raw", "n_kept", "n_dropped",
        "y_mean", "y_std", "beta_norm", "delta", "iqr_effect", "y_corr_pred",
    )

    def __init__(self, row: Stats):
        super().__init__()
        self._stats = row

    def __iter__(self):
        yield {f: getattr(self._stats, f) for f in self._columns}


class OLSStatsView(StatsView):
    """Stats view for PCA+OLS — adds r2_adj after r2."""
    _columns = (
        "backend", "r2", "r2_adj", "pvalue",
        "n_raw", "n_kept", "n_dropped",
        "y_mean", "y_std", "beta_norm", "delta", "iqr_effect", "y_corr_pred",
    )


# ---------- WordsView ----------
class WordsView(View[Word]):
    """Tabular view of nearest-neighbor words for both β poles.

    By default, display truncates to ``DEFAULT_MAX_ROWS`` split evenly between
    ``pos`` and ``neg`` sides.  Use ``.pos`` / ``.neg`` for one-sided access.
    """

    _name = "words"
    _columns = ("side", "rank", "word", "cos_beta", "contrast")

    def __init__(self, rows: list[Word], *, _no_trunc: bool = False):
        super().__init__(_no_trunc=_no_trunc)
        self._rows = rows

    def __iter__(self): return iter(self._rows)
    def __len__(self): return len(self._rows)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return WordsView(self._rows[i], _no_trunc=True)
        return self._rows[i]

    @property
    def pos(self) -> WordsViewSided:
        return WordsViewSided("pos", self._rows)

    @property
    def neg(self) -> WordsViewSided:
        return WordsViewSided("neg", self._rows)

    def to_text(self, max_rows: int | None = None, cols=None) -> str:
        from ssdiff.results.core import DEFAULT_MAX_ROWS
        if max_rows is None:
            max_rows = DEFAULT_MAX_ROWS
        if self._no_trunc:
            return super().to_text(max_rows=max_rows, cols=cols)
        pos_rows = [w for w in self._rows if w.side == "pos"]
        neg_rows = [w for w in self._rows if w.side == "neg"]
        if not pos_rows or not neg_rows:
            return super().to_text(max_rows=max_rows, cols=cols)
        per_side = max(1, max_rows // 2)
        shown = WordsView(pos_rows[:per_side] + neg_rows[:per_side], _no_trunc=True)
        out = View.to_text(shown, cols=cols)
        hidden = (len(pos_rows) - per_side) + (len(neg_rows) - per_side)
        if hidden > 0:
            out += f"\n... {hidden} more rows (use .pos / .neg for one side)"
        return out

    def _save_hint(self) -> str:
        return (super()._save_hint()
                + "\nSides: .pos   .neg → WordsViewSided (top 20; .pos(50) for more)")


class WordsViewSided(WordsView):
    """Words filtered to one β side.

    Defaults to 20 rows (by rank) for display AND export — iterating,
    saving to CSV, or rendering in the terminal all stop at 20. Call with
    a different ``k`` to resize:

        words.pos          → WordsViewSided, 20 rows
        words.pos(50)      → WordsViewSided, 50 rows (or fewer, if fewer available)
        words.pos(None)    → all available rows on this side
    """

    def __init__(self, side: str, all_rows: list[Word], k: int | None = 20,
                 *, _no_trunc: bool = False):
        side_rows = [w for w in all_rows if w.side == side]
        limited = side_rows if k is None else side_rows[:k]
        super().__init__(limited, _no_trunc=True)
        self._side_key = side
        self._all_side_rows = side_rows
        self._k = k

    def __call__(self, k: int | None = 20) -> WordsViewSided:
        return WordsViewSided(self._side_key, self._all_side_rows, k=k)

    def _save_hint(self) -> str:
        max_k = len(self._all_side_rows)
        current = "all" if self._k is None else str(self._k)
        return (View._save_hint(self)
                + f"\nRows:  (k) → first k "
                f"(current {current}, max {max_k}; k=None for all)")


# ---------- Clusters ----------
class ClusterWordsView(View[ClusterWord]):
    """Tabular view of word members for one cluster (obtained via ``clusters.pos(id).words``)."""

    _name = "cluster_words"
    _columns = ("cluster_id", "side", "word", "cos_centroid", "cos_beta", "contrast")

    def __init__(self, rows, *, _no_trunc: bool = False):
        super().__init__(_no_trunc=_no_trunc)
        self._rows = rows

    def __iter__(self): return iter(self._rows)
    def __len__(self): return len(self._rows)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return ClusterWordsView(self._rows[i], _no_trunc=True)
        return self._rows[i]

    def __call__(self, n: int | None = None) -> ClusterWordsView:
        """Return the first ``n`` rows (``None`` = all)."""
        if n is None:
            return ClusterWordsView(list(self._rows), _no_trunc=True)
        return ClusterWordsView(self._rows[:n], _no_trunc=True)


class ClusterWordsViewSided(ClusterWordsView):
    """Cluster-word rows for one β side.

    ``clusters.pos.words`` returns a ``ClusterWordsViewSided`` with all
    positive-side cluster word rows. Positional ``(n)`` slices to the first
    ``n`` rows (type-preserving). To drill to one cluster, use
    ``clusters.pos(cluster_id).words`` — that yields a plain
    ``ClusterWordsView`` pre-filtered to that cluster.
    """

    def __init__(self, side: str, rows, *, _no_trunc: bool = False):
        super().__init__(rows, _no_trunc=_no_trunc)
        self._side = side

    def __call__(self, n: int | None = None) -> ClusterWordsViewSided:
        """Return the first ``n`` rows (``None`` = all). Type-preserving slice."""
        rows = list(self._rows) if n is None else self._rows[:n]
        return ClusterWordsViewSided(self._side, rows, _no_trunc=True)


class ClustersViewSided(View[Cluster]):
    """Cluster summary view for one β pole.

    Callable with a positional ``cluster_id`` to drill into one cluster;
    callable with recompute kwargs (``topn=``, ``k=``, …) to re-cluster.
    After drilling, ``.words`` / ``.snippets`` are pre-filtered to that
    cluster and their ``(n)`` call becomes a first-n row slice.
    """

    _name = "clusters"
    _columns = ("cluster_id", "side", "size", "coherence", "centroid_cos_beta", "contrast")

    def __init__(self, parent: ContinuousResult, side: str,
                 rows: list[Cluster], words_rows: list[ClusterWord],
                 params: dict, *, cluster_id: int | None = None,
                 _no_trunc: bool = False):
        super().__init__(_no_trunc=_no_trunc)
        self._parent = parent
        self._side = side
        self._rows = rows
        self._words_rows = words_rows
        self._params = dict(params)
        self._cluster_id = cluster_id

    def __iter__(self): return iter(self._rows)
    def __len__(self): return len(self._rows)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return ClustersViewSided(
                parent=self._parent, side=self._side,
                rows=self._rows[i], words_rows=self._words_rows,
                params=self._params, cluster_id=self._cluster_id,
                _no_trunc=True,
            )
        return self._rows[i]

    @property
    def params(self): return dict(self._params)

    def __call__(self, cluster_id=None, **params) -> ClustersViewSided:
        """Drill to one cluster (positional) or recompute (kwargs).

        * ``clusters.pos(N)`` → filter to cluster ``cluster_id=N``
        * ``clusters.pos(topn=50, k=3)`` → recompute with new params
        * Cannot mix positional drill and recompute kwargs in one call.
        """
        if cluster_id is not None and params:
            raise TypeError(
                "pass a positional cluster_id OR recompute kwargs, not both"
            )
        if params:
            merged = {**self._params, **params}
            merged.pop("side", None)
            return self._parent._clusters_for(self._side, **merged)
        if cluster_id is None:
            return self
        if self._cluster_id is not None:
            raise TypeError(
                f"already filtered to cluster_id={self._cluster_id}; "
                f"chain .words(n) / .snippets(n) to slice rows"
            )
        filtered_rows = [c for c in self._rows if c.cluster_id == cluster_id]
        if not filtered_rows:
            raise KeyError(
                f"no cluster with cluster_id={cluster_id} on side={self._side!r}"
            )
        filtered_words = [
            w for w in self._words_rows if w.cluster_id == cluster_id
        ]
        return ClustersViewSided(
            parent=self._parent, side=self._side,
            rows=filtered_rows, words_rows=filtered_words,
            params=self._params, cluster_id=cluster_id, _no_trunc=True,
        )

    @property
    def words(self):
        """Per-side cluster-words view.

        Undrilled → ``ClusterWordsViewSided`` (call with cluster_id to drill).
        Drilled   → ``ClusterWordsView`` pre-filtered to that cluster
                    (call with ``n`` for first-n slice).
        """
        if self._cluster_id is not None:
            return ClusterWordsView(list(self._words_rows))
        return ClusterWordsViewSided(side=self._side, rows=self._words_rows)

    @property
    def snippets(self) -> SnippetsViewSided:
        """Centroid-based snippets on this side.

        Undrilled → ``SnippetsViewSided`` (call with cluster_id to drill).
        Drilled   → pre-filtered to this cluster (call with ``n`` for first-n slice).
        """
        all_side = self._parent._cluster_snippets_for(
            side=self._side,
            **{k: v for k, v in self._params.items() if k != "side"},
        )
        if self._cluster_id is None:
            return all_side
        return all_side(cluster_id=self._cluster_id)

    def _save_hint(self) -> str:
        base = super()._save_hint()
        if self._cluster_id is not None:
            return base + (
                f"\nCluster {self._cluster_id} ({self._side}):\n"
                f"  .words(n)    → first n words in this cluster (ClusterWordsView)\n"
                f"  .snippets(n) → first n snippets in this cluster"
            )
        ids = sorted({c.cluster_id for c in self._rows})
        if ids:
            examples = ", ".join(f"({i}).words" for i in ids[:3])
            drill_line = (
                f"\nDrill: (cluster_id) → zoom to one cluster  "
                f"(e.g. {ids[0]}; available: "
                f"{', '.join(str(i) for i in ids[:6])}"
                f"{', …' if len(ids) > 6 else ''})"
            )
            words_line = (
                f"\nWords: .words → ClusterWordsViewSided (all on this side);"
                f" {examples} → one ClusterWordsView"
            )
        else:
            drill_line = "\nDrill: (cluster_id) → zoom to one cluster"
            words_line = (
                "\nWords: .words → ClusterWordsViewSided   "
                "(cluster_id).words → ClusterWordsView"
            )
        topn = self._params.get("topn", 100)
        return base + drill_line + words_line + (
            f"\nSnippets: .snippets → side snippets; "
            f"(cluster_id).snippets → one cluster"
            f"\nRecompute: (topn=50, k=3, …) (current topn={topn})"
        )


class ClustersView(View[Cluster]):
    """Flat-iterable cluster view — pos rows, then neg rows.

    Exposes ``.pos`` / ``.neg`` for per-side access and ``.words`` for the
    combined cluster-words table. Supports ``.save()`` (inherited from
    ``View``) which writes all rows (pos then neg) to a single file.
    """

    _name = "clusters"
    _columns = ("cluster_id", "side", "size", "coherence", "centroid_cos_beta", "contrast")

    def __init__(self, parent: ContinuousResult):
        super().__init__()
        self._parent = parent

    @property
    def _rows(self) -> list[Cluster]:
        """Materialize pos + neg rows in canonical order (pos first)."""
        pos = list(self._parent._clusters_for("pos")._rows)
        neg = list(self._parent._clusters_for("neg")._rows)
        return pos + neg

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return self._rows[i]

    @property
    def pos(self) -> ClustersViewSided:
        return self._parent._clusters_for("pos")

    @property
    def neg(self) -> ClustersViewSided:
        return self._parent._clusters_for("neg")

    @property
    def words(self) -> ClusterWordsView:
        """Combined cluster-words across both sides."""
        pos_rows = self._parent._clusters_for("pos")._words_rows
        neg_rows = self._parent._clusters_for("neg")._words_rows
        return ClusterWordsView(list(pos_rows) + list(neg_rows))

    @property
    def snippets(self):
        raise AttributeError(
            "cluster snippets are per-side; use .pos.snippets or .neg.snippets, "
            "or result.cluster_snippets(side='pos')"
        )

    def _cached_count(self, side: str) -> int | None:
        """Return cached len for `side`, or None if not yet computed."""
        for (name, key), view in self._parent._cache.items():
            if name != "clusters":
                continue
            if dict(key).get("side") == side:
                return len(view)
        return None

    def _save_hint(self) -> str:
        return ("Save:  .save('clusters.csv')               # all rows (pos then neg)\n"
                "       .pos.save('clusters_pos.csv')\n"
                "       .neg.save('clusters_neg.csv')")

    def to_text(self) -> str:
        lines = ["ClustersView"]
        for side, label in (("pos", "positive"), ("neg", "negative")):
            n = self._cached_count(side)
            if n is None:
                lines.append(f"  .{side}  → {label} clusters (call to compute)")
            else:
                lines.append(f"  .{side}  → {n} {label} clusters")
        return "\n".join(lines)

    def to_html(self) -> str:
        return f"<pre>{self.to_text()}</pre>"

    def __repr__(self) -> str:
        body = self.to_text()
        if _save_hint_enabled():
            return body + "\n\n" + self._save_hint()
        return body

    def _repr_html_(self) -> str:
        body = self.to_html()
        if _save_hint_enabled():
            return (body
                    + f"\n<pre class='ssd-save-hint'>{self._save_hint()}</pre>")
        return body

# ---------- SnippetsView ----------
_SNIPPET_EXTRACTION_KWARGS = frozenset({"top_per_side", "min_cosine", "n_jobs"})


class SnippetsView(View[Snippet]):
    """Tabular view of text snippets extracted near seed words along the gradient.

    Use ``.pos`` / ``.neg`` for side filtering, and call the sided view with
    ``k`` / ``cluster_id=`` to resize or filter. The flat view's ``__call__``
    is reserved for **recomputing** with different extraction parameters
    (``top_per_side``, ``min_cosine``, ``n_jobs``). Text columns are truncated
    to 40 characters in terminal display; full text is preserved in exports.
    """

    _name = "snippets"
    _columns = (
        "snippet_id", "side", "doc_id", "cosine", "seed",
        "start_token_idx", "end_token_idx", "start_sent_idx", "end_sent_idx",
        "text_window", "text_surface", "text_lemmas",
        "cluster_id", "post_id", "contrast",
    )

    def __init__(self, rows: list[Snippet], params: dict | None = None,
                 parent: ContinuousResult | None = None,
                 *, _no_trunc: bool = False):
        super().__init__(_no_trunc=_no_trunc)
        self._rows = rows
        self._params = dict(params or {})
        self._parent = parent

    def __iter__(self): return iter(self._rows)
    def __len__(self): return len(self._rows)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return SnippetsView(self._rows[i], params=self._params,
                                parent=self._parent, _no_trunc=True)
        return self._rows[i]

    @property
    def params(self): return dict(self._params)

    @property
    def pos(self) -> SnippetsViewSided:
        """Snippets filtered to the β-positive side (default k=30)."""
        return SnippetsViewSided("pos", self._rows)

    @property
    def neg(self) -> SnippetsViewSided:
        """Snippets filtered to the β-negative side (default k=30)."""
        return SnippetsViewSided("neg", self._rows)

    def __call__(self, **params) -> SnippetsView:
        """Recompute snippets with updated extraction parameters.

        Accepts only extraction kwargs (``top_per_side``, ``min_cosine``,
        ``n_jobs``). Use ``.pos`` / ``.neg`` for side filtering, and
        ``.pos(k, cluster_id=...)`` for per-cluster filtering.
        """
        if self._parent is None:
            return self
        unknown = set(params) - _SNIPPET_EXTRACTION_KWARGS
        if unknown:
            raise TypeError(
                f"SnippetsView() got unexpected kwargs {sorted(unknown)!r}. "
                f"Accepted extraction kwargs: {sorted(_SNIPPET_EXTRACTION_KWARGS)}. "
                f"Use .pos / .neg for side filtering, or "
                f".pos(cluster_id=...) for cluster filtering."
            )
        merged = {**self._params, **params}
        return self._parent._snippets_for(**merged)

    def _save_hint(self) -> str:
        top_per_side = self._params.get("top_per_side", 30)
        return (super()._save_hint()
                + "\nSides: .pos   .neg → SnippetsViewSided "
                  "(top 30; .pos(50) / .pos(cluster_id=0) for more)"
                + f"\nSize:  (top_per_side=100) → recompute "
                  f"(current {top_per_side})")


_UNSET = object()


class SnippetsViewSided(SnippetsView):
    """Snippets filtered to one β side.

    Positional ``__call__`` is always **row count**::

        snippets.pos(50)     → first 50 rows
        snippets.pos(None)   → all rows on this side

    To filter by cluster, drill at the clusters view — use
    ``clusters.pos(cluster_id).snippets`` — or pass the explicit
    ``cluster_id=`` keyword.
    """

    def __init__(self, side: str, all_rows: list[Snippet],
                 k: int | None = 30, cluster_id: int | None = None,
                 *, _no_trunc: bool = False):
        side_rows = [s for s in all_rows if s.side == side]
        if cluster_id is not None:
            side_rows = [s for s in side_rows if s.cluster_id == cluster_id]
        limited = side_rows if k is None else side_rows[:k]
        super().__init__(limited, params={"side": side}, _no_trunc=True)
        self._side_key = side
        self._all_side_rows = side_rows
        self._all_rows = all_rows
        self._k = k
        self._cluster_id = cluster_id

    def __call__(self, k: int | None = _UNSET, *,
                 cluster_id=_UNSET) -> SnippetsViewSided:
        """Resize rows (positional ``k``) and/or filter by cluster (``cluster_id=``)."""
        new_k = k if k is not _UNSET else self._k
        new_cid = cluster_id if cluster_id is not _UNSET else self._cluster_id
        return SnippetsViewSided(
            self._side_key, self._all_rows, k=new_k, cluster_id=new_cid,
        )

    def _save_hint(self) -> str:
        max_k = len(self._all_side_rows)
        current_k = "all" if self._k is None else str(self._k)
        rows_line = (
            f"\nRows:  (k) → first k "
            f"(current {current_k}, max {max_k}; k=None for all)"
        )
        base = View._save_hint(self)
        if self._cluster_id is not None:
            return (base
                    + f"\n[filtered to cluster_id={self._cluster_id}]"
                    + rows_line)
        return base + rows_line


# ---------- DocsView ----------
class DocsView(View[Doc]):
    """Tabular view of per-document predictions and alignment scores.

    Supports sorting accessors (``.pos()``, ``.neg()``, ``.misdiagnosed()``)
    and single-doc detail lookup via ``.id(doc_id)``.
    """

    _name = "docs"
    _columns = ("doc_id", "y_true", "y_hat", "residual", "alignment_score")

    def __init__(self, rows: list[Doc], *, parent: ContinuousResult | None = None,
                 _no_trunc: bool = False, _preview: bool = False):
        super().__init__(_no_trunc=_no_trunc)
        self._rows = rows
        self._parent = parent
        self._preview = _preview

    def __iter__(self): return iter(self._rows)
    def __len__(self): return len(self._rows)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return DocsView(self._rows[i], parent=self._parent, _no_trunc=True)
        return self._rows[i]

    def pos(self, k: int = 20) -> DocsView:
        """Docs most aligned with β-pos (highest y_hat)."""
        return DocsView(
            sorted(self._rows, key=lambda d: -d.y_hat)[:k],
            parent=self._parent, _no_trunc=True,
        )

    def neg(self, k: int = 20) -> DocsView:
        """Docs most aligned with β-neg (lowest y_hat)."""
        return DocsView(
            sorted(self._rows, key=lambda d: d.y_hat)[:k],
            parent=self._parent, _no_trunc=True,
        )

    def misdiagnosed(self, k: int = 20, direction: str = "both") -> DocsView:
        """Docs with largest prediction error.

        ``direction`` selects the residual sign:
          * ``"both"``  — largest |residual| (default)
          * ``"over"``  — model over-predicts (y_hat > y_true, residual < 0)
          * ``"under"`` — model under-predicts (y_hat < y_true, residual > 0)
        """
        if direction == "both":
            rows = sorted(self._rows, key=lambda d: -abs(d.residual))[:k]
        elif direction == "over":
            rows = sorted(self._rows, key=lambda d: d.residual)[:k]
        elif direction == "under":
            rows = sorted(self._rows, key=lambda d: -d.residual)[:k]
        else:
            raise ValueError(
                f"direction must be one of 'both'/'over'/'under', got {direction!r}"
            )
        return DocsView(rows, parent=self._parent, _no_trunc=True)

    def id(self, doc_id: int) -> DocDetailView:
        """Return a DocDetailView for the document with the given ``doc_id``.

        Parameters
        ----------
        doc_id : int
            Document index (aligns with the corpus row index).

        Raises
        ------
        KeyError
            If no document with this ``doc_id`` exists in the view.
        """
        matches = [d for d in self._rows if d.doc_id == doc_id]
        if not matches:
            raise KeyError(f"No doc with doc_id={doc_id}")
        raw_text = _lookup_raw_text(self._parent, doc_id)
        return DocDetailView(matches[0], raw_text)

    def to_text(self, max_rows: int | None = None, cols=None) -> str:
        from ssdiff.results.core import DEFAULT_MAX_ROWS
        if max_rows is None:
            max_rows = DEFAULT_MAX_ROWS
        if self._preview and len(self._rows) > 10:
            sorted_rows = sorted(self._rows, key=lambda d: d.y_hat)
            neg = DocsView(sorted_rows[:5], _no_trunc=True)
            pos = DocsView(list(reversed(sorted_rows[-5:])), _no_trunc=True)
            header = f"Docs — preview of {len(self._rows)} (β-pos 5 / β-neg 5 by y_hat)"
            return (f"{header}\n\n"
                    f"pos 5:\n{pos.to_text(cols=cols)}\n\n"
                    f"neg 5:\n{neg.to_text(cols=cols)}")
        return super().to_text(max_rows=max_rows, cols=cols)

    def _save_hint(self) -> str:
        return (super()._save_hint()
                + "\nSlice: .pos()   .neg()   .pos(10)   .id(5)   .misdiagnosed(10)")


def _lookup_raw_text(parent, doc_id: int):
    """Pull pre-lemmatization text for `doc_id` from parent.corpus, if available."""
    if parent is None:
        return None
    corpus = getattr(parent, "corpus", None)
    if corpus is None:
        return None
    pre = getattr(corpus, "pre_docs", None)
    if not pre or doc_id < 0 or doc_id >= len(pre):
        return None
    entry = pre[doc_id]
    if hasattr(entry, "raw"):
        return entry.raw
    if hasattr(entry, "raw_posts"):
        return list(entry.raw_posts)
    return None


class DocDetailView(ScalarView):
    """Single-doc view: stats + original pre-lemma text (when corpus attached)."""

    _name = "doc"
    _columns = ("doc_id", "y_true", "y_hat", "residual", "alignment_score")

    def __init__(self, doc: Doc, raw_text):
        super().__init__()
        self._doc = doc
        self._raw_text = raw_text

    def __iter__(self):
        yield {f: getattr(self._doc, f) for f in self._columns}

    def to_text(self, max_rows: int | None = None, cols=None) -> str:
        body = super().to_text(max_rows=max_rows, cols=cols)
        if self._raw_text is None:
            return body + "\n\nText:  (attach corpus to see original text)"
        if isinstance(self._raw_text, list):
            text_block = "\n".join(
                f"  [post {i}] {p}" for i, p in enumerate(self._raw_text)
            )
        else:
            text_block = "  " + str(self._raw_text).replace("\n", "\n  ")
        return body + "\n\nText:\n" + text_block

    def to_html(self, cols=None) -> str:
        base = super().to_html(cols=cols)
        if self._raw_text is None:
            tail = "<p><i>(attach corpus to see original text)</i></p>"
        elif isinstance(self._raw_text, list):
            import html as _h
            items = "".join(f"<li>{_h.escape(p)}</li>" for p in self._raw_text)
            tail = f"<h4>Text</h4><ol>{items}</ol>"
        else:
            import html as _h
            tail = f"<h4>Text</h4><pre>{_h.escape(str(self._raw_text))}</pre>"
        return base + tail

    def __repr__(self) -> str:
        return self.to_text()

    def _repr_html_(self) -> str:
        return self.to_html()


# ---------- Sweep / SplitTest ----------
@dataclass(frozen=True, slots=True)
class SweepRow:
    k: int
    var_explained: float
    mean_coherence: float
    mean_abs_cosb: float
    aggregate: float
    n_clusters: int
    total_size: int
    beta_delta_1_minus_cos: float
    interp_hat: float
    interp_resid: float
    interp_resid_z: float
    interp_auck: float
    stab_good_raw: float
    stab_z_raw: float
    stab_auck_raw: float
    joint_score: float


class SweepView(View[SweepRow]):
    _name = "sweep"
    _columns = (
        "k", "var_explained", "mean_coherence", "mean_abs_cosb",
        "aggregate", "n_clusters", "total_size", "beta_delta_1_minus_cos",
        "interp_hat", "interp_resid", "interp_resid_z", "interp_auck",
        "stab_good_raw", "stab_z_raw", "stab_auck_raw", "joint_score",
    )

    def __init__(self, rows, *, _no_trunc: bool = False):
        super().__init__(_no_trunc=_no_trunc)
        self._rows = rows

    def __iter__(self): return iter(self._rows)
    def __len__(self): return len(self._rows)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return SweepView(self._rows[i], _no_trunc=True)
        return self._rows[i]


class PLSTestView(TestView):
    """`.test` for PLSResult — split_nb confirmatory test at the fitted k."""

    _columns = ("name", "pvalue", "split_r2", "n_splits", "random_state")
    _default_name = "split_nb"

    _DEFAULTS = {
        "split_nb": dict(n_splits=50, seed=None, verbose=False),
    }

    def _run(self, name, params):
        """Dispatch to plskit and return ('split_nb', info_dict)."""
        import plskit

        if name is not None and name != "split_nb":
            raise TypeError(
                f"PLSTestView only supports method='split_nb' "
                f"(got {name!r}). Pass kwargs only: "
                f".test(n_splits=..., seed=...)."
            )
        merged = {**self._DEFAULTS["split_nb"], **params}
        parent = self._parent
        n_comp = parent.fit_info.p_at_k or parent.fit_info.n_components or 1
        r = plskit.pls1_confirmatory_test(
            parent.x.astype(float, copy=False),
            parent.y.astype(float, copy=False),
            int(n_comp),
            method="split_nb",
            args={"n_splits": int(merged["n_splits"])},
            pre_standardized=False,
            seed=merged["seed"],
            verbose=merged["verbose"],
        )
        info = {
            "pvalue": float(r.pvalue),
            "statistic": float(r.statistic),
            "split_r2": float(r.statistic),
            "n_splits": r.n_splits,
            "random_state": r.seed,
        }
        return "split_nb", info

    def _on_rerun(self):
        """Propagate the updated p-value back to parent stats after a rerun."""
        self._parent._refresh_stats_pvalue(self.pvalue)

    def _rerun_hint(self) -> str:
        return "Rerun: .test(n_splits=..., seed=...)"


class PCAOLSTestView(TestView):
    """`.test` for PCAOLSResult — currently F-test only (extensible)."""

    _columns = ("name", "pvalue")
    _default_name = "f_test"

    def _run(self, name, params):
        """Return (name, info_dict) for the F-test; rerun is a no-op (analytic test)."""
        if name != "f_test":
            raise ValueError(
                f"Unknown PCA+OLS test {name!r}. Available: ('f_test',)"
            )
        # F-test is analytic and computed at fit. Rerun is a no-op;
        # we return the current stored value so repeated calls stay consistent.
        return name, {"pvalue": self.pvalue}

    def _on_rerun(self):
        # No state to refresh — pvalue is unchanged.
        pass

    def _rerun_hint(self) -> str:
        return "Rerun: .test('f_test')   # only test currently available"


# ---------- ContinuousResult ----------
class ContinuousResult(_SingleResult):
    """Shared base for continuous-outcome results (PLS / PCA+OLS).

    Attributes
    ----------
    beta : ndarray of shape (D,)
        Raw regression direction in embedding space (β_K in Plisiecki et
        al., 2026). Carries magnitude — use this when scale matters:
        prediction (``d_i · beta`` ≈ outcome on standardized scale),
        effect size via ``beta_norm``, regression-style reasoning.
    gradient : ndarray of shape (D,)
        Unit-length version of ``beta`` — the **semantic gradient** in
        Plisiecki et al. (2026). Direction most strongly associated with
        higher outcome. Use this when only direction matters: cosine
        similarity, nearest-neighbor lookup, clustering of semantic poles.
    beta_norm : float
        Magnitude ‖β_K‖ of the regression direction. Effect-size summary:
        how much the standardized outcome changes per unit move along
        ``gradient``.
    alignment_scores : ndarray of shape (n,)
        Per-document **SSD alignment score** (s_i = d_i · gradient); cosine
        similarity between each **personal concept vector (PCV)** and the
        semantic gradient. Cached on first access.
    """

    _stats_view_cls: type[StatsView] = StatsView

    def __init__(
        self,
        *,
        backend: str,
        x: np.ndarray,
        beta: np.ndarray,
        keep_mask: np.ndarray,
        n_raw: int, n_kept: int, n_dropped: int,
        y: np.ndarray,
        _y_mean: np.ndarray, _y_scale: np.ndarray,
        r2: float, pvalue: float, r2_adj: float | None = None,
        embeddings=None, corpus=None,
        lexicon: set | None = None, window: int = 3, sif_a: float = 1e-3, lang: str = "pl",
        fit_info: FitInfo | dict | None = None,
        raw_diagnostics: dict | None = None,
    ):
        """Construct a continuous result from backend outputs.

        Parameters
        ----------
        backend : str
            Backend label (``"PLS"`` or ``"PCA+OLS"``).
        x : ndarray of shape (n_kept, D)
            Per-document embedding vectors (personal concept vectors, PCVs) after
            filtering.
        beta : ndarray of shape (D,)
            Raw regression direction in embedding space.
        keep_mask : ndarray of shape (n_raw,) of bool
            Boolean mask indicating which of the original ``n_raw`` documents
            were retained.  Used to align ``doc_id`` with corpus row indices.
        n_raw, n_kept, n_dropped : int
            Document counts before and after filtering.
        y : ndarray of shape (n_kept,)
            Outcome values on their original scale (after any inverse transform).
        _y_mean : ndarray of shape (1,)
            Mean used to standardize ``y`` at fit time (for back-transform).
        _y_scale : ndarray of shape (1,)
            Standard deviation used to standardize ``y`` at fit time.
        r2 : float
            In-sample R² reported by the backend.
        pvalue : float
            P-value from the initial significance test (may be updated later via
            ``result.test(...)``).
        r2_adj : float or None
            Adjusted R² (PCA+OLS only; ``None`` for PLS).
        embeddings : Embeddings or None
            Word-embedding model for nearest-neighbor and cluster views.
        corpus : Corpus or None
            Text corpus for snippet extraction.
        lexicon : set or None
            Optional seed tokens that constrain snippet extraction.
        window : int
            Token window size passed to the snippet extractor.
        sif_a : float
            SIF smoothing parameter for snippet vector computation.
        lang : str
            Language code used for token filtering in neighbor search.
        fit_info : FitInfo, dict, or None
            Backend hyperparameters (n_components, pca_k, etc.).  A dict is
            converted to FitInfo automatically.
        raw_diagnostics : dict or None
            Backend-specific extra outputs (PLS X-scores, PCA components, etc.)
            forwarded to the subclass.
        """
        super().__init__(
            x=x, beta=beta,
            embeddings=embeddings, corpus=corpus,
            lexicon=lexicon, window=window, sif_a=sif_a, lang=lang,
        )
        self._keep_mask = keep_mask
        self.y = y
        self._y_mean = _y_mean
        self._y_scale = _y_scale

        if fit_info is None:
            fit_info = FitInfo()
        elif isinstance(fit_info, dict):
            fit_info = FitInfo(**fit_info)
        self.fit_info = FitInfoView(fit_info)
        self._raw_diagnostics = dict(raw_diagnostics or {})

        y_mean = float(_y_mean[0])
        y_std = float(_y_scale[0])
        x_norms = np.sqrt(np.einsum("ij,ij->i", x, x))[:, None]
        x_norms = np.maximum(x_norms, 1e-12)
        cos_align = ((x / x_norms) @ self.gradient).ravel()
        yhat_std = (x @ self.beta).ravel()
        yhat_raw = y_mean + y_std * yhat_std

        denom = float(np.std(y) * np.std(yhat_std))
        corr = float(np.corrcoef(y, yhat_std)[0, 1]) if denom > 0 else 0.0
        if not np.isfinite(corr):
            corr = 0.0
        delta = 0.10 * self.beta_norm * y_std
        q75, q25 = np.percentile(cos_align, [75, 25])
        iqr_effect = float(q75 - q25) * self.beta_norm * y_std

        stats_row = Stats(
            backend=backend, r2=r2, r2_adj=r2_adj, pvalue=pvalue,
            n_raw=n_raw, n_kept=n_kept, n_dropped=n_dropped,
            y_mean=y_mean, y_std=y_std, beta_norm=self.beta_norm,
            delta=delta, iqr_effect=iqr_effect, y_corr_pred=abs(corr),
        )
        self.stats = self._stats_view_cls(stats_row)

        keep_idx = np.where(keep_mask)[0] if keep_mask is not None else np.arange(len(y))
        residuals = y - yhat_raw
        self.docs = DocsView(
            [
                Doc(doc_id=int(keep_idx[i]),
                    y_true=float(y[i]),
                    y_hat=float(yhat_raw[i]),
                    residual=float(residuals[i]),
                    alignment_score=float(cos_align[i]))
                for i in range(len(y))
            ],
            parent=self, _preview=True,
        )

        # Subclasses set self.test in their own __init__.
        self.test: TestView | None = None

    def _refresh_stats_pvalue(self, new_pvalue: float | None) -> None:
        """Rebuild self.stats with an updated pvalue.

        Called by TestView after a rerun so that stats.pvalue and test.pvalue
        always agree.
        """
        from dataclasses import replace as _replace
        if new_pvalue is None:
            return
        new_stats = _replace(self.stats._stats, pvalue=float(new_pvalue))
        self.stats = StatsView(new_stats)

    _access = (
        "stats", "fit_info", "words", "clusters", "snippets", "docs", "test",
        "report()", "test(...)", "attach(...)",
    )
    _arrays = ("x", "y", "beta", "gradient", "alignment_scores")

    def _summary(self) -> str:
        s = self.stats
        return (f"{type(self).__name__}  r²={fmt_r(s.r2)}  "
                f"p={fmt_p(s.pvalue)}  "
                f"n={fmt_count(s.n_kept)}  backend={s.backend}")

    def _summary_html(self) -> str:
        s = self.stats
        return (f"<p><b>{type(self).__name__}</b> "
                f"r²={fmt_r(s.r2)} · p={fmt_p(s.pvalue)} · "
                f"n={fmt_count(s.n_kept)} · "
                f"backend={s.backend}</p>")

    def _save_hint(self) -> str:
        cls = type(self).__name__.lower().replace("result", "")
        return (
            f"Save:  {cls}.report().save('report.md')    # narrative report\n"
            f"       {cls}.words.save('words.csv')       # per-view file output\n"
            f"       {cls}.docs.to_df()                  # pandas DataFrame"
        )

    def _save_hint_html(self) -> str:
        return f"<pre class='ssd-save-hint'>{self._save_hint()}</pre>"

    # -------- report --------------------------------------------------
    def report(self, *, top_words: int | None = 5, clusters: int | None = None,
               snippets_per_cluster: int | None = None,
               extreme_docs: int | None = None,
               misdiagnosed: int | None = None) -> Report:
        """Build a multi-section narrative Report for this result.

        Parameters
        ----------
        top_words : int or None
            Number of words to include per β pole in the words section.
            ``None`` skips the words section.
        clusters : int or None
            ``topn`` value passed to the cluster extractor; controls how many
            neighbors are clustered per side.  ``None`` skips clusters.
        snippets_per_cluster : int or None
            Currently unused — reserved for a future per-cluster snippet table.
        extreme_docs : int or None
            Number of most-aligned (pos) and least-aligned (neg) documents to
            include.  ``None`` skips the extreme-docs section.
        misdiagnosed : int or None
            Number of over-predicted and under-predicted documents to include.
            ``None`` skips the misdiagnosed section.

        Returns
        -------
        Report
            A ``Report`` object that can be rendered with ``.to_text()``,
            ``.to_html()``, or saved with ``.save('report.md')``.
        """
        sections = []
        s = self.stats
        stat_rows = [
            ("backend", s.backend),
            ("r²", fmt_r(s.r2)),
        ]
        if "r2_adj" in s.columns:
            stat_rows.append(("r²_adj", fmt_r(s.r2_adj)))
        stat_rows.extend([
            ("p-value", fmt_p(s.pvalue)),
            ("n_raw", fmt_count(s.n_raw)),
            ("n_kept", fmt_count(s.n_kept)),
            ("β norm", fmt_d(s.beta_norm)),
            ("Δ (Q4−Q1)", fmt_d(s.iqr_effect)),
        ])
        sections.append(Section(title="Stats", kind="kv", rows=stat_rows))

        fi = self.fit_info
        fit_rows = []
        if fi.n_components is not None:
            fit_rows.append(("n_components", fi.n_components))
        if fi.p_at_k is not None:
            fit_rows.append(("p_at_k", fi.p_at_k))
        if fi.pca_k is not None:
            fit_rows.append(("pca_k", fi.pca_k))
        if fi.pca_k_source is not None:
            fit_rows.append(("pca_k_source", fi.pca_k_source))
        if fi.random_state is not None:
            fit_rows.append(("random_state", fi.random_state))
        if fit_rows:
            sections.append(Section(title="Fit info", kind="kv", rows=fit_rows))

        if top_words and self.embeddings is not None:
            pos_words = [w for w in self.words if w.side == "pos"][:top_words]
            neg_words = [w for w in self.words if w.side == "neg"][:top_words]
            rows = []
            for w in pos_words + neg_words:
                rows.append([w.side, w.rank, w.word, fmt_r(w.cos_beta, signed=True)])
            sections.append(Section(title=f"Top words (n={top_words} per side)",
                                    kind="table",
                                    headers=["Side", "Rank", "Word", "cos_β"],
                                    rows=rows,
                                    numeric=[False, True, False, True]))

        if clusters and self.embeddings is not None:
            for side in ("pos", "neg"):
                cl = getattr(self.clusters, side)(topn=clusters)
                rows = []
                for c in cl:
                    rows.append([c.cluster_id, c.size, fmt_r(c.coherence),
                                 fmt_r(c.centroid_cos_beta, signed=True)])
                sections.append(Section(
                    title=f"Clusters ({side}, topn={clusters})",
                    kind="table",
                    headers=["cluster", "size", "coherence", "centroid cos_β"],
                    rows=rows,
                    numeric=[True, True, True, True],
                ))

        if extreme_docs:
            for side_name, picker in (("pos", self.docs.pos), ("neg", self.docs.neg)):
                rows = [[d.doc_id, fmt_d(d.y_true), fmt_d(d.y_hat), fmt_d(d.residual)]
                        for d in picker(extreme_docs)]
                sections.append(Section(
                    title=f"Docs — {side_name} {extreme_docs}",
                    kind="table",
                    headers=["doc_id", "y_true", "y_hat", "residual"],
                    rows=rows,
                    numeric=[True, True, True, True],
                ))

        if misdiagnosed:
            for direction in ("over", "under"):
                rows = [[d.doc_id, fmt_d(d.y_true), fmt_d(d.y_hat), fmt_d(d.residual)]
                        for d in self.docs.misdiagnosed(misdiagnosed, direction=direction)]
                sections.append(Section(
                    title=f"Misdiagnosed — {direction}-predicted {misdiagnosed}",
                    kind="table",
                    headers=["doc_id", "y_true", "y_hat", "residual"],
                    rows=rows,
                    numeric=[True, True, True, True],
                ))

        return Report(
            title=f"{type(self).__name__} — r² = {fmt_r(s.r2)}",
            subtitle=f"(n = {fmt_count(s.n_kept)})",
            sections=sections,
        )


# ---------- PLSResult ----------
class PLSResult(ContinuousResult):
    """PLS1 result.

    Attributes
    ----------
    n_components : int
        Number of PLS latent components fit.
    component_scores : ndarray of shape (n, A)
        PLS1 X-scores T — per-document projection onto each of the A
        latent components.
    component_weights : ndarray of shape (D, A)
        PLS1 X-weights W in embedding space (unit-normed).
    find_k_result : plskit.FindKOptimalResult | None
        Full ``plskit.pls1_find_k_optimal`` output when ``k="auto"``;
        ``None`` otherwise. Inspect ``find_k_result.k_star``,
        ``find_k_result.cv_scores``, ``find_k_result.pvalues``.
    cv_scores : dict | None
        Flat ``{k: cv_score}`` dict — convenience copy of
        ``find_k_result.cv_scores`` (None when ``k`` was an int).
    """

    _arrays = (
        "x", "y", "beta", "gradient", "alignment_scores",
        "component_scores", "component_weights",
    )

    def __init__(self, *, test_name: str | None = None,
                 test_info: dict | None = None, **kw):
        """Construct a PLS result, pulling extra PLS diagnostics from ``raw_diagnostics``.

        Parameters
        ----------
        test_name : str or None
            Name of the initial test (if one was run at fit time).
        test_info : dict or None
            Info dict from the initial test run (must contain ``"pvalue"``).
        **kw
            All remaining arguments forwarded to ``ContinuousResult.__init__``.
        """
        kw.setdefault("backend", "PLS")
        super().__init__(**kw)
        self.test = PLSTestView(
            parent=self, name=test_name, info=test_info,
        )
        raw = self._raw_diagnostics
        self.find_k_result = raw.get("find_k_result")
        self.cv_scores = (
            dict(self.find_k_result.cv_scores)
            if self.find_k_result is not None
            and self.find_k_result.cv_scores is not None
            else None
        )
        self.n_components = int(self.fit_info.n_components or 0)
        self.component_scores = raw.get("component_scores")
        self.component_weights = raw.get("component_weights")


# ---------- PCAOLSResult ----------
class PCAOLSResult(ContinuousResult):
    """PCA + OLS result.

    Attributes
    ----------
    sweep_result : PCAKSelectionResult | None
        Full PCA sweep diagnostics. The **PCA sweep** procedure selects K
        by joint interpretability+stability score (see Plisiecki, Leniarska
        et al., 2026). ``None`` when ``fit_ols(fixed_k=...)`` was used.
    pca_k : int
        Number of PCA components used.
    pca_components : ndarray of shape (K, D)
        PCA loadings V_K — the first K principal directions of the
        standardized PCV matrix.
    pca_weights : ndarray of shape (K,)
        OLS regression weights w_K estimated in PCA space;
        ``β = V_K w_K / x_scale``.
    """

    _stats_view_cls = OLSStatsView
    _access = (
        "stats", "fit_info", "words", "clusters", "snippets", "docs", "sweep",
        "test", "report()", "test(...)", "attach(...)",
    )
    _arrays = (
        "x", "y", "beta", "gradient", "alignment_scores",
        "pca_components", "pca_weights",
    )

    def __init__(self, *, sweep: list | None = None,
                 test_name: str | None = None,
                 test_info: dict | None = None, **kw):
        """Construct a PCA+OLS result, pulling sweep data and PCA components from diagnostics.

        Parameters
        ----------
        sweep : list of dict or None
            Per-K sweep rows (keys matching :class:`SweepRow` fields) used to
            populate the ``.sweep`` SweepView.
        test_name : str or None
            Name of the initial test; defaults to ``"f_test"``.
        test_info : dict or None
            Info dict from the initial test run (must contain ``"pvalue"``).
        **kw
            All remaining arguments forwarded to ``ContinuousResult.__init__``.
        """
        kw.setdefault("backend", "PCA+OLS")
        super().__init__(**kw)
        self.sweep = SweepView([
            SweepRow(
                k=int(r["k"]),
                var_explained=float(r["var_explained"]),
                mean_coherence=float(r["mean_coherence"]),
                mean_abs_cosb=float(r["mean_abs_cosb"]),
                aggregate=float(r["aggregate"]),
                n_clusters=int(r["n_clusters"]),
                total_size=int(r["total_size"]),
                beta_delta_1_minus_cos=float(r["beta_delta_1_minus_cos"]),
                interp_hat=float(r["interp_hat"]),
                interp_resid=float(r["interp_resid"]),
                interp_resid_z=float(r["interp_resid_z"]),
                interp_auck=float(r["interp_auck"]),
                stab_good_raw=float(r["stab_good_raw"]),
                stab_z_raw=float(r["stab_z_raw"]),
                stab_auck_raw=float(r["stab_auck_raw"]),
                joint_score=float(r["joint_score"]),
            )
            for r in (sweep or [])
        ])
        self.sweep_result = self._raw_diagnostics.get("sweep_result")
        self.pca_k = int(self.fit_info.n_components or 0)
        self.pca_components = self._raw_diagnostics.get("pca_components")
        self.pca_weights = self._raw_diagnostics.get("pca_weights")
        # Default test = F-test; pvalue is whatever came from fit.
        default_info = test_info or {"pvalue": kw.get("pvalue")}
        self.test = PCAOLSTestView(
            parent=self, name=test_name or "f_test", info=default_info,
        )

    def _save_hint(self) -> str:
        return (
            "Save:  ols.report().save('report.md')        # narrative\n"
            "       ols.words.save('words.csv')           # data\n"
            "       ols.plot_sweep('sweep.png')           # chart"
        )

    def plot_sweep(self, path: str | None = None, *, dpi: int = 300) -> bytes:
        """Dual-axis PCA-K sweep chart. Unchanged from v0.x behaviour.

        Raises RuntimeError when `sweep_result is None`.
        """
        if self.sweep_result is None:
            raise RuntimeError(
                "No sweep data — fit_ols() was called with an explicit "
                "fixed_k. Re-run with fixed_k=None to enable "
                "the PCA-K sweep."
            )

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot_sweep(). "
                "Install it with: pip install ssdiff[results]"
            ) from None

        import io

        rows = self.sweep_result.df_joined
        x = [r["PCA_K"] for r in rows]
        y_left = np.array([r.get("interp_resid_z", np.nan) for r in rows])
        y_right = np.array([r.get("beta_delta_1_minus_cos", np.nan) for r in rows])

        y_right_smooth = _rolling_median(y_right, window=7)

        fig, ax1 = plt.subplots()

        ax1.plot(x, y_left, marker="o", color="tab:blue",
                 label="detrended interpretability (z)")
        ax1.axhline(0.0, linewidth=1, color="0.6")
        ax1.set_xlabel("PCA_K")
        ax1.set_ylabel("Detrended interpretability (z)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(x, y_right_smooth, linewidth=2, color="tab:orange",
                 label="beta change (smoothed 1-cos)")
        ax2.set_ylabel("Beta change (smoothed 1 - cosine)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        ax1.axvline(self.sweep_result.best_k, color="red", linewidth=2,
                     label=f"best K = {self.sweep_result.best_k}")

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        png = buf.getvalue()

        if path is not None:
            with open(path, "wb") as f:
                f.write(png)
        else:
            plt.show()

        plt.close(fig)
        return png
