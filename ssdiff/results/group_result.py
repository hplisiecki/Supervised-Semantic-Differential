"""GroupResult + PairView — group comparison results."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ssdiff.results.core import Result, ScalarView, TestView, View
from ssdiff.results.format import fmt_count, fmt_d, fmt_p
from ssdiff.results.report import Report, Section
from ssdiff.results.schema import (
    Cluster,
    ClusterWord,
    Pair,
    Snippet,
    Word,
)

# ---------- GroupStatsView (ScalarView) ----------

class GroupStatsView(ScalarView):
    """ScalarView exposing omnibus-test metadata for a GroupResult."""

    _name = "stats"
    _columns = ("G", "n_kept", "n_perm", "correction", "random_state", "pvalue")

    def __init__(self, *, G: int, n_kept: int, n_perm: int,
                 correction: str, random_state, pvalue: float):
        super().__init__()
        self._G = G
        self._n_kept = n_kept
        self._n_perm = n_perm
        self._correction = correction
        self._random_state = random_state
        self._pvalue = pvalue

    def __iter__(self):
        yield {
            "G": self._G,
            "n_kept": self._n_kept,
            "n_perm": self._n_perm,
            "correction": self._correction,
            "random_state": self._random_state,
            "pvalue": self._pvalue,
        }


# ---------- PairStatsView (ScalarView) ----------

class PairStatsView(ScalarView):
    """ScalarView exposing per-pair contrast statistics from a ``Pair`` dataclass."""

    _name = "stats"
    _columns = ("T", "p_raw", "p_corrected", "cohens_d", "n_g1", "n_g2", "contrast_norm")

    def __init__(self, pair: Pair):
        super().__init__()
        self._pair = pair

    def __iter__(self):
        p = self._pair
        yield {
            "T": p.T,
            "p_raw": p.p_raw,
            "p_corrected": p.p_corrected,
            "cohens_d": p.cohens_d,
            "n_g1": p.n_g1,
            "n_g2": p.n_g2,
            "contrast_norm": p.contrast_norm,
        }


# ---------- GroupTestView (TestView) ----------

class GroupTestView(TestView):
    """`.test` for GroupResult — permutation omnibus + pairwise."""

    _columns = ("name", "pvalue", "omnibus_T", "omnibus_p",
                "G", "n_kept", "n_perm", "correction", "random_state")
    _default_name = "permutation"

    _DEFAULTS = {
        "permutation": dict(n_perm=5000, correction="holm",
                            random_state=2137, verbose=False),
    }

    def _run(self, name, params):
        """Run the unified permutation test and return (name, info_dict) with rebuilt pairs."""
        if name not in self._DEFAULTS:
            raise ValueError(
                f"Unknown group test {name!r}. "
                f"Available: {tuple(self._DEFAULTS)}"
            )
        merged = {**self._DEFAULTS[name], **params}
        parent = self._parent
        if parent.x is None or parent.groups is None:
            raise RuntimeError(
                "gr.test() requires the original x / groups arrays; "
                "were they discarded by un-pickling or detach()?"
            )

        from ssdiff.backends.group import unified_permutation_test

        group_labels = sorted(set(parent.groups), key=str)
        test_result = unified_permutation_test(
            parent.x, parent.groups, group_labels,
            n_perm=merged["n_perm"],
            correction=merged["correction"],
            random_state=merged["random_state"],
            verbose=merged["verbose"],
        )

        # Rebuild pair rows (see SSD.fit_groups for the canonical build path).
        new_pairs = []
        for (g1, g2), pw in test_result["pairwise"].items():
            new_pairs.append(Pair(
                contrast=f"{g1}_vs_{g2}",
                g1=str(g1), g2=str(g2),
                T=float(pw["T"]),
                p_raw=float(pw["p_raw"]),
                p_corrected=float(pw["p_corrected"]),
                cohens_d=float(pw["cohens_d"]),
                n_g1=int(pw["n_g1"]),
                n_g2=int(pw["n_g2"]),
                contrast_norm=float(pw["contrast_norm"]),
            ))

        parent._update_pairs(new_pairs, n_perm=merged["n_perm"],
                             correction=merged["correction"],
                             random_state=merged["random_state"])

        info = {
            "pvalue": float(test_result["omnibus_p"]),
            "omnibus_T": float(test_result["omnibus_T"]),
            "omnibus_p": float(test_result["omnibus_p"]),
            "G": int(test_result["G"]),
            "n_kept": len(parent.x),
            "n_perm": merged["n_perm"],
            "correction": merged["correction"],
            "random_state": merged["random_state"],
        }
        return name, info

    def _on_rerun(self):
        """No-op: GroupStatsView is refreshed inside ``_run`` via ``parent._update_pairs``."""
        # GroupStatsView pvalue is already refreshed via parent._update_pairs.
        pass

    def _rerun_hint(self) -> str:
        return "Rerun: .test(n_perm=..., correction=...)"

    def to_text(self, max_rows: int | None = None, cols=None) -> str:
        """Render the omnibus row plus a pairwise summary block below it."""
        base = super().to_text(max_rows=max_rows, cols=cols)
        if not self._info or self._parent is None:
            return base
        lines = [base, "", "pairwise:"]
        for p in self._parent.pairs:
            lines.append(
                f"  {p.contrast}   T={p.T:+.3f}   "
                f"p={fmt_p(p.p_corrected)}   d={p.cohens_d:+.3f}"
            )
        return "\n".join(lines)


# ---------- PairsListView ----------

class PairsListView(View[Pair]):
    """Iterates canonical Pair rows; supports tuple-key lookup returning PairView."""

    _name = "pairs"
    _columns = (
        "contrast", "g1", "g2", "T", "p_raw", "p_corrected",
        "cohens_d", "n_g1", "n_g2", "contrast_norm",
    )

    def __init__(self, pairs: list[Pair], *, _no_trunc: bool = False):
        super().__init__(_no_trunc=_no_trunc)
        self._pairs = pairs
        # canonical key: tuple(sorted((g1, g2))) -> Pair
        self._index: dict[tuple[str, str], Pair] = {}
        for p in pairs:
            key = tuple(sorted((p.g1, p.g2)))
            self._index[key] = p

    def __iter__(self) -> Iterator[Pair]:
        return iter(self._pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return PairsListView(self._pairs[key], _no_trunc=True)
        if isinstance(key, int):
            return self._pairs[key]
        # tuple-key lookup
        if isinstance(key, tuple) and len(key) == 2:
            g1, g2 = key
            canonical = tuple(sorted((g1, g2)))
            if canonical not in self._index:
                raise KeyError(f"no pair ({g1!r}, {g2!r})")
            pair = self._index[canonical]
            reversed_order = (g1, g2) != (pair.g1, pair.g2)
            return PairView(pair=pair, reversed=reversed_order,
                            parent=getattr(self, "_parent", None))
        raise KeyError(key)

    def _save_hint(self) -> str:
        if self._pairs:
            p = self._pairs[0]
            example = f"view[{p.g1!r}, {p.g2!r}]"
        else:
            example = "view['<group1>', '<group2>']"
        return (super()._save_hint()
                + f"\nLookup: {example} → PairView")


# ---------- PairView ----------

class PairView:
    """Lightweight view for a single contrast.

    Exposes `.contrast` and `.pair` (with sign-flip when accessed in reverse
    order). Sub-views (.words, .clusters, .snippets) filter the parent
    GroupResult's flat rows by contrast (lazy, empty if rows not computed).
    """

    def __init__(self, *, pair: Pair, reversed: bool,
                 words_rows: list[Word] | None = None,
                 cluster_rows: list[Cluster] | None = None,
                 cluster_words_rows: list[ClusterWord] | None = None,
                 snippets_rows: list[Snippet] | None = None,
                 parent: GroupResult | None = None):
        self._canonical_pair = pair
        self._reversed = reversed
        self._words_rows = words_rows or []
        self._cluster_rows = cluster_rows or []
        self._cluster_words_rows = cluster_words_rows or []
        self._snippets_rows = snippets_rows or []
        self._parent = parent
        self._canonical_cache: tuple[np.ndarray, np.ndarray, float] | None = None

    @property
    def pair(self) -> Pair:
        """Returns the Pair dataclass, sign-flipped if accessed in reverse order."""
        p = self._canonical_pair
        if not self._reversed:
            return p
        # Reverse: swap g1/g2, swap n_g1/n_g2, flip T/cohens_d/contrast_norm,
        # invert contrast string, preserve p-values.
        contrast_flipped = f"{p.g2}_vs_{p.g1}"
        return Pair(
            contrast=contrast_flipped,
            g1=p.g2,
            g2=p.g1,
            T=-p.T,
            p_raw=p.p_raw,
            p_corrected=p.p_corrected,
            cohens_d=-p.cohens_d,
            n_g1=p.n_g2,
            n_g2=p.n_g1,
            contrast_norm=-p.contrast_norm,
        )

    @property
    def stats(self) -> PairStatsView:
        """ScalarView exposing per-pair stats (sign-flipped when reversed)."""
        return PairStatsView(self.pair)

    def _compute_canonical(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute (beta, gradient, beta_norm) for the canonical g1 → g2 order.

        Cached on first access. Callers apply ``_reversed`` sign flip at the
        public property boundary.
        """
        if self._canonical_cache is not None:
            return self._canonical_cache
        parent = self._parent
        if parent is None or parent.x is None or parent.groups is None:
            raise RuntimeError(
                "PairView requires the original x / groups arrays on its "
                "parent; were they discarded by un-pickling or detach()?"
            )
        p = self._canonical_pair
        x = parent.x
        g = parent.groups
        beta = x[g == p.g1].mean(axis=0) - x[g == p.g2].mean(axis=0)
        beta = np.asarray(beta, dtype=float).copy()
        beta.setflags(write=False)
        beta_norm = float(np.linalg.norm(beta))
        gradient = (beta / max(beta_norm, 1e-12)).copy()
        gradient.setflags(write=False)
        self._canonical_cache = (beta, gradient, beta_norm)
        return self._canonical_cache

    @property
    def beta(self) -> np.ndarray:
        """Pair contrast vector ``c_g1 − c_g2`` in embedding space.

        Carries magnitude — parallels ``ContinuousResult.beta``. Sign-flipped
        when accessed in reverse pair order.
        """
        beta, _, _ = self._compute_canonical()
        return -beta if self._reversed else beta

    @property
    def gradient(self) -> np.ndarray:
        """Unit-length pair direction ``beta / ‖beta‖``.

        Parallels ``ContinuousResult.gradient``. Sign-flipped on reverse access.
        """
        _, gradient, _ = self._compute_canonical()
        return -gradient if self._reversed else gradient

    @property
    def beta_norm(self) -> float:
        """Magnitude ``‖beta‖`` of the pair contrast direction.

        Scalar magnitude — invariant under reverse access.
        """
        _, _, beta_norm = self._compute_canonical()
        return beta_norm

    @property
    def alignment_scores(self) -> np.ndarray:
        """Per-document projection onto this pair's unit contrast direction.

        Computed as ``x @ gradient``, parallel to
        ``ContinuousResult.alignment_scores``. Sign flips when the pair is
        accessed in reverse order (via ``self.gradient``).
        """
        parent = self._parent
        if parent is None or parent.x is None or parent.groups is None:
            raise RuntimeError(
                "pair.alignment_scores requires the original x / groups arrays; "
                "were they discarded by un-pickling or detach()?"
            )
        return (parent.x @ self.gradient).ravel()

    @property
    def contrast(self) -> str:
        """Contrast label (e.g. ``"A_vs_B"``), sign-aware on reverse access."""
        return self.pair.contrast

    @property
    def words(self):
        """WordsView for this contrast, with side and sign flipped when reversed."""
        from ssdiff.results.continuous_result import WordsView
        contrast = self._canonical_pair.contrast
        rows = [w for w in self._words_rows if w.contrast == contrast]
        if self._reversed:
            rows = [
                Word(side="neg" if w.side == "pos" else "pos",
                     rank=w.rank, word=w.word,
                     cos_beta=-w.cos_beta, contrast=self.contrast)
                for w in rows
            ]
        return WordsView(rows)

    @property
    def clusters(self):
        """ClustersIndex-like object backed by pre-computed rows for this contrast."""
        # Return a lightweight index backed by filtered rows.
        return _PairClustersIndex(
            pair_view=self,
            cluster_rows=self._cluster_rows,
            cluster_words_rows=self._cluster_words_rows,
        )

    @property
    def snippets(self):
        """SnippetsView for this contrast, with side flipped when reversed."""
        from ssdiff.results.continuous_result import SnippetsView
        contrast = self._canonical_pair.contrast
        rows = [s for s in self._snippets_rows if s.contrast == contrast]
        if self._reversed:
            rows = [
                Snippet(
                    snippet_id=s.snippet_id,
                    side="neg" if s.side == "pos" else "pos",
                    doc_id=s.doc_id, cosine=s.cosine, seed=s.seed,
                    start_token_idx=s.start_token_idx,
                    end_token_idx=s.end_token_idx,
                    start_sent_idx=s.start_sent_idx,
                    end_sent_idx=s.end_sent_idx,
                    text_window=s.text_window,
                    text_surface=s.text_surface,
                    text_lemmas=s.text_lemmas,
                    cluster_id=s.cluster_id,
                    contrast=self.contrast,
                    post_id=s.post_id,
                )
                for s in rows
            ]
        return SnippetsView(rows)

    def _save_hint(self) -> str:
        return ("Save:  .stats.to_dict()\n"
                "       .words.save('pair_words.csv')\n"
                "       .snippets.save('pair_snippets.csv')")

    def to_text(self) -> str:
        from ssdiff.results.format import fmt_p, fmt_r
        p = self.pair
        return (
            f"PairView  {p.g1} \u2192 {p.g2}  "
            f"T={fmt_r(p.T, signed=True)}  "
            f"p_corr={fmt_p(p.p_corrected)}  "
            f"d={fmt_r(p.cohens_d, signed=True)}  "
            f"n={p.n_g1}/{p.n_g2}\n"
            f"  arrays:  .beta  .gradient  .alignment_scores\n"
            f"  views:   .stats  .words  .clusters  .snippets"
        )

    def to_html(self) -> str:
        return f"<pre>{self.to_text()}</pre>"

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
            return (body
                    + f"\n<pre class='ssd-save-hint'>{self._save_hint()}</pre>")
        return body


class _PairClustersIndex:
    """Minimal ClustersIndex-like object backed by flat rows for a PairView."""

    def __init__(self, pair_view: PairView,
                 cluster_rows: list[Cluster],
                 cluster_words_rows: list[ClusterWord]):
        self._pair_view = pair_view
        self._cluster_rows = cluster_rows
        self._cluster_words_rows = cluster_words_rows

    def _sided(self, side: str):
        """Build a SidedClustersView for ``side``, applying sign-flip when the pair is reversed."""
        from ssdiff.results.continuous_result import SidedClustersView
        contrast = self._pair_view._canonical_pair.contrast
        # If reversed, flip side label used for filtering canonical rows
        lookup_side = side
        if self._pair_view._reversed:
            lookup_side = "neg" if side == "pos" else "pos"
        rows = [c for c in self._cluster_rows
                if c.contrast == contrast and c.side == lookup_side]
        words_rows = [w for w in self._cluster_words_rows
                      if w.contrast == contrast and w.side == lookup_side]
        if self._pair_view._reversed:
            rows = [
                Cluster(cluster_id=c.cluster_id, side=side,
                        size=c.size, coherence=c.coherence,
                        centroid_cos_beta=-c.centroid_cos_beta,
                        contrast=self._pair_view.contrast)
                for c in rows
            ]
            words_rows = [
                ClusterWord(cluster_id=w.cluster_id, side=side,
                            word=w.word, cos_centroid=w.cos_centroid,
                            cos_beta=-w.cos_beta,
                            contrast=self._pair_view.contrast)
                for w in words_rows
            ]
        return SidedClustersView(
            parent=None, side=side, rows=rows, words_rows=words_rows,
            snippets_rows=None, params={},
        )

    @property
    def pos(self):
        """SidedClustersView for the positive pole of this contrast."""
        return self._sided("pos")

    @property
    def neg(self):
        """SidedClustersView for the negative pole of this contrast."""
        return self._sided("neg")


# ---------- GroupResult ----------

class GroupResult(Result):
    """Result from ``SSD.fit_groups()``.

    Each group centroid is computed from the **personal concept vectors
    (PCVs)** of the documents in that group; pairwise contrasts
    β = c_{g1} − c_{g2} are tested by permuting group labels.

    Attributes
    ----------
    test : GroupTestView
        Omnibus test view; ``gr.test.omnibus_T`` / ``.omnibus_p`` / ``.pvalue``.
        Call ``gr.test(n_perm=...)`` to rerun the permutation test.
    pairs : PairsListView
        Iterable list of Pair rows; supports ``gr.pairs["A", "B"]`` returning
        a PairView with sign-flip semantics on reverse-order access.
    stats : GroupStatsView
        Metadata (G, n_kept, n_perm, correction, random_state, pvalue).
    G : int
        Number of groups.
    n_kept : int
        Number of documents retained after filtering.
    n_perm : int
        Number of permutations used.
    correction : str
        P-value correction method applied.
    random_state : int
        Random seed used.
    x : ndarray of shape (n_kept, D)
        Per-document vectors retained after ``filter_small_groups``.
        Load-bearing for ``gr.test(...)`` reruns and contrast recomputation.
    groups : ndarray of shape (n_kept,)
        Group labels aligned with ``x``. For continuous ``y`` passed to
        ``fit_groups(median_split=True)`` these are the median-split bins;
        otherwise they are the raw categorical labels cast to ``object``.
    """

    def __init__(
        self,
        *,
        G: int,
        n_kept: int,
        n_perm: int,
        correction: str,
        random_state: int,
        omnibus_T: float,
        omnibus_p: float,
        pairs: list[Pair],
        words_rows: list[Word],
        cluster_rows: list[Cluster],
        cluster_words_rows: list[ClusterWord],
        snippets_rows: list[Snippet],
        embeddings=None,
        corpus=None,
        x: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ):
        """Construct a GroupResult from backend outputs.

        Parameters
        ----------
        G : int
            Number of groups.
        n_kept : int
            Documents retained after ``filter_small_groups``.
        n_perm : int
            Permutations used in the omnibus + pairwise tests.
        correction : str
            P-value correction method (``"holm"``, ``"bonferroni"``,
            ``"fdr_bh"``, or ``"none"``).
        random_state : int
            Random seed used for permutation shuffling.
        omnibus_T : float
            Observed omnibus test statistic (mean pairwise cosine distance).
        omnibus_p : float
            Omnibus permutation p-value.
        pairs : list of Pair
            Per-contrast statistics from the permutation test.
        words_rows : list of Word
            Flat list of neighbor words for all contrasts (filtered by
            ``contrast`` field in PairView).
        cluster_rows : list of Cluster
            Flat list of cluster summaries for all contrasts.
        cluster_words_rows : list of ClusterWord
            Flat list of cluster-word memberships for all contrasts.
        snippets_rows : list of Snippet
            Flat list of extracted text snippets for all contrasts.
        embeddings : Embeddings or None
            Word-embedding model (forwarded to sub-views).
        corpus : Corpus or None
            Text corpus (forwarded to sub-views).
        x : ndarray of shape (n_kept, D) or None
            Document vectors retained after filtering; required for test reruns.
        groups : ndarray of shape (n_kept,) or None
            Group labels aligned with ``x``; required for test reruns.
        """
        super().__init__()
        self.embeddings = embeddings
        self.corpus = corpus

        self.G = G
        self.n_kept = n_kept
        self.n_perm = n_perm
        self.correction = correction
        self.random_state = random_state

        # Stored for gr.test(...) rerun. May be None when reconstructed
        # without the source arrays (e.g. synthetic test fixtures).
        self.x = x
        self.groups = groups

        self.stats = GroupStatsView(
            G=G, n_kept=n_kept, n_perm=n_perm,
            correction=correction, random_state=random_state,
            pvalue=float(omnibus_p),
        )

        self.test = GroupTestView(
            parent=self,
            name="permutation",
            info={
                "pvalue": float(omnibus_p),
                "omnibus_T": float(omnibus_T),
                "omnibus_p": float(omnibus_p),
                "G": G,
                "n_kept": n_kept,
                "n_perm": n_perm,
                "correction": correction,
                "random_state": random_state,
            },
        )

        # Build PairsListView — pairs carry per-pair sub-view rows lazily
        # (words_rows etc. are stored flat; PairView filters by contrast on access).
        self._words_rows = list(words_rows)
        self._cluster_rows = list(cluster_rows)
        self._cluster_words_rows = list(cluster_words_rows)
        self._snippets_rows = list(snippets_rows)

        # Inject sub-view rows into each PairView via a closure-friendly wrapper.
        # PairsListView stores canonical Pair dataclasses; PairView is ephemeral.
        self.pairs = _GroupPairsListView(
            pairs=pairs,
            words_rows=self._words_rows,
            cluster_rows=self._cluster_rows,
            cluster_words_rows=self._cluster_words_rows,
            snippets_rows=self._snippets_rows,
            parent=self,
        )

    def _update_pairs(self, new_pairs: list[Pair], *, n_perm: int,
                      correction: str, random_state) -> None:
        """Swap in new pair rows + refresh stats after a gr.test() rerun."""
        self.n_perm = n_perm
        self.correction = correction
        self.random_state = random_state
        self.pairs = _GroupPairsListView(
            pairs=new_pairs,
            words_rows=self._words_rows,
            cluster_rows=self._cluster_rows,
            cluster_words_rows=self._cluster_words_rows,
            snippets_rows=self._snippets_rows,
            parent=self,
        )
        self.stats = GroupStatsView(
            G=self.G, n_kept=self.n_kept, n_perm=n_perm,
            correction=correction, random_state=random_state,
            pvalue=float(self.test.pvalue) if self.test is not None else float("nan"),
        )

    _access = (
        "stats", "test", "pairs",
        "report()", "test(...)", "attach(...)",
    )
    _arrays = ("x", "groups")

    def _summary(self) -> str:
        return (f"GroupResult  G={self.G}  n={fmt_count(self.n_kept)}  "
                f"omnibus p={fmt_p(self.test.omnibus_p)}")

    def _summary_html(self) -> str:
        return (f"<p><b>GroupResult</b> G={self.G} · "
                f"n={fmt_count(self.n_kept)} · "
                f"omnibus p={fmt_p(self.test.omnibus_p)}</p>")

    def _save_hint(self) -> str:
        return (
            "Save:  gr.report().save('report.md')         # narrative\n"
            "       gr.pairs.save('pairs.csv')            # data"
        )

    def _save_hint_html(self) -> str:
        return f"<pre class='ssd-save-hint'>{self._save_hint()}</pre>"

    # -------- report -------------------------------------------------------

    def report(self, *, top_words: int | None = 5,
               clusters: int | None = None,
               snippets_per_cluster: int | None = None) -> Report:
        """Build a multi-section narrative Report for this group result.

        Parameters
        ----------
        top_words : int or None
            Words per pole per contrast to include. ``None`` skips the words
            section. Currently unused — words are not included in the group
            report by default.
        clusters : int or None
            Reserved; not yet implemented in the group report.
        snippets_per_cluster : int or None
            Reserved; not yet implemented in the group report.

        Returns
        -------
        Report
            A ``Report`` with an omnibus section and a pairwise-contrasts table.
        """
        sections = []

        # Omnibus section
        sections.append(Section(title="Omnibus", kind="kv", rows=[
            ("G", self.G),
            ("n_kept", fmt_count(self.n_kept)),
            ("n_perm", fmt_count(self.n_perm)),
            ("correction", self.correction),
            ("T", fmt_d(self.test.omnibus_T)),
            ("p", fmt_p(self.test.omnibus_p)),
        ]))

        # Pairwise table
        rows = []
        for p in self.pairs:
            rows.append([
                p.contrast, fmt_d(p.T), fmt_p(p.p_raw),
                fmt_p(p.p_corrected), fmt_d(p.cohens_d),
                fmt_count(p.n_g1), fmt_count(p.n_g2),
            ])
        if rows:
            sections.append(Section(
                title="Pairwise contrasts",
                kind="table",
                headers=["contrast", "T", "p_raw", "p_corrected",
                         "Cohen's d", "n_g1", "n_g2"],
                rows=rows,
                numeric=[False, True, True, True, True, True, True],
            ))

        return Report(
            title=f"GroupResult — G={self.G}",
            subtitle=f"(n = {fmt_count(self.n_kept)}, omnibus p = {fmt_p(self.test.omnibus_p)})",
            sections=sections,
        )


# ---------- _GroupPairsListView (adds sub-view row injection to PairView) ----------

class _GroupPairsListView(PairsListView):
    """PairsListView variant that injects sub-view rows into returned PairViews."""

    def __init__(self, *, pairs: list[Pair],
                 words_rows: list[Word],
                 cluster_rows: list[Cluster],
                 cluster_words_rows: list[ClusterWord],
                 snippets_rows: list[Snippet],
                 parent: GroupResult):
        super().__init__(pairs)
        self._words_rows = words_rows
        self._cluster_rows = cluster_rows
        self._cluster_words_rows = cluster_words_rows
        self._snippets_rows = snippets_rows
        self._parent = parent

    def __getitem__(self, key):
        """Return a PairView (or sliced _GroupPairsListView) with sub-view rows injected."""
        if isinstance(key, slice):
            return _GroupPairsListView(
                pairs=self._pairs[key],
                words_rows=self._words_rows,
                cluster_rows=self._cluster_rows,
                cluster_words_rows=self._cluster_words_rows,
                snippets_rows=self._snippets_rows,
                parent=self._parent,
            )
        result = super().__getitem__(key)
        if isinstance(result, PairView):
            # Re-wrap with sub-view rows injected
            return PairView(
                pair=result._canonical_pair,
                reversed=result._reversed,
                words_rows=self._words_rows,
                cluster_rows=self._cluster_rows,
                cluster_words_rows=self._cluster_words_rows,
                snippets_rows=self._snippets_rows,
                parent=self._parent,
            )
        return result
