"""GroupResult — group-comparison results with view-type-first accessors.

``GroupResult`` exposes tables as top-level attributes (``gr.words``,
``gr.clusters``, ``gr.snippets``) mirroring :class:`ContinuousResult`.
Dispatch is by pair count: a 2-group fit yields the same single-view
types as the continuous case; a multi-group fit yields the paired
collection views from :mod:`ssdiff.results.paired_view`.

Per-pair arrays (``gr.beta``, ``gr.gradient``, ``gr.beta_norm``,
``gr.alignment_scores``) mirror the same shape: plain arrays for the
single-pair case, dicts keyed on canonical pair tuples for multi-pair.

Canonical labels: all groups are renamed ``g_1, g_2, …`` in
``sorted(set(groups), key=str)`` order. Originals survive in
``gr.group_labels``. Canonical pair keys use the numeric trailing index
so that ``g_2 < g_10``. Reverse-order tuple lookup raises ``KeyError`` —
no sign-flip anywhere.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from functools import cached_property

import numpy as np

from ssdiff.results.core import Result, ScalarView, TestView, View
from ssdiff.results.format import fmt_count, fmt_d, fmt_p, fmt_r
from ssdiff.results.report import Report, Section
from ssdiff.results.schema import (
    Cluster,
    ClusterWord,
    Pair,
    Snippet,
    Word,
)


# ---------- Canonical pair helpers ----------

def _canonical_pair_key(g1: str, g2: str) -> tuple[str, str]:
    """Return ``(g1, g2)`` sorted by the numeric trailing index of canonical labels.

    Falls back to string order for non-canonical labels.
    """
    def _sort_key(s: str) -> tuple[int, int | str]:
        if s.startswith("g_"):
            try:
                return (0, int(s.split("_", 1)[1]))
            except (ValueError, IndexError):
                pass
        return (1, s)
    return tuple(sorted((g1, g2), key=_sort_key))  # type: ignore[return-value]


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

        # Rebuild pair rows. ``parent.groups`` is already canonical (relabeled in
        # ``__init__``), so pair keys returned by the backend are canonical too.
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
    """Iterates canonical Pair rows; supports tuple-key lookup returning the Pair itself.

    Canonical order only: ``view[(g2, g1)]`` raises ``KeyError``. No sign-flip.
    """

    _name = "pairs"
    _columns = (
        "contrast", "g1", "g2", "T", "p_raw", "p_corrected",
        "cohens_d", "n_g1", "n_g2", "contrast_norm",
    )

    def __init__(self, pairs: list[Pair], *, _no_trunc: bool = False):
        super().__init__(_no_trunc=_no_trunc)
        self._pairs = list(pairs)
        # canonical key: tuple(sorted((g1, g2))) -> Pair
        self._index: dict[tuple[str, str], Pair] = {}
        for p in self._pairs:
            key = _canonical_pair_key(p.g1, p.g2)
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
        # tuple-key lookup — canonical order only; returns the Pair dataclass directly.
        if isinstance(key, tuple) and len(key) == 2:
            g1, g2 = key
            canonical = _canonical_pair_key(g1, g2)
            if canonical not in self._index:
                raise KeyError(f"no pair ({g1!r}, {g2!r})")
            pair = self._index[canonical]
            if (g1, g2) != (pair.g1, pair.g2):
                raise KeyError(
                    f"pair must be accessed in canonical order "
                    f"{(pair.g1, pair.g2)!r}, got {(g1, g2)!r}"
                )
            return pair
        raise KeyError(key)

    def _save_hint(self) -> str:
        if self._pairs:
            p = self._pairs[0]
            example = f"view[{p.g1!r}, {p.g2!r}]"
        else:
            example = "view['<group1>', '<group2>']"
        return (super()._save_hint()
                + f"\nLookup: {example} → Pair")


# ---------- helper: single-pair ClustersIndex-like ----------

class _SinglePairClustersIndex:
    """Lightweight ClustersIndex-like backed by flat cluster rows for one contrast.

    Mirrors the ``ClustersIndex`` public contract (``.pos`` / ``.neg`` → ``ClustersViewSided``)
    without involving a parent ``ContinuousResult``. Used both in the single-pair case
    (directly as ``gr.clusters``) and in the multi-pair case (as the per-pair children
    wrapped by ``ClustersIndexPaired``).
    """

    def __init__(self, *, contrast: str,
                 cluster_rows: list[Cluster],
                 cluster_words_rows: list[ClusterWord]):
        self._contrast = contrast
        self._cluster_rows = cluster_rows
        self._cluster_words_rows = cluster_words_rows

    def _sided(self, side: str):
        from ssdiff.results.continuous_result import ClustersViewSided
        rows = [c for c in self._cluster_rows
                if c.contrast == self._contrast and c.side == side]
        words_rows = [w for w in self._cluster_words_rows
                      if w.contrast == self._contrast and w.side == side]
        return ClustersViewSided(
            parent=None, side=side, rows=rows, words_rows=words_rows,
            snippets_rows=None, params={},
        )

    @property
    def pos(self):
        """ClustersViewSided for the positive pole of this contrast."""
        return self._sided("pos")

    @property
    def neg(self):
        """ClustersViewSided for the negative pole of this contrast."""
        return self._sided("neg")


# ---------- GroupResult ----------

class GroupResult(Result):
    """Result from ``SSD.fit_groups()``.

    Each group centroid is computed from the **personal concept vectors
    (PCVs)** of the documents in that group; pairwise contrasts
    β = c_{g1} − c_{g2} are tested by permuting group labels.

    Top-level view dispatch mirrors :class:`ContinuousResult`. When
    ``len(pairs) == 1`` (2-group fit), ``gr.words`` / ``gr.clusters`` /
    ``gr.snippets`` and ``gr.beta`` / ``gr.gradient`` / ``gr.beta_norm`` /
    ``gr.alignment_scores`` return the same types as the continuous
    single-view case. When ``len(pairs) >= 2`` they return paired views
    and dicts keyed by canonical pair tuples.

    Attributes
    ----------
    test : GroupTestView
        Omnibus test view; ``gr.test.omnibus_T`` / ``.omnibus_p`` / ``.pvalue``.
        Call ``gr.test(n_perm=...)`` to rerun the permutation test.
    pairs : PairsListView
        Iterable of canonical ``Pair`` rows; ``gr.pairs[(g1, g2)]`` returns
        the ``Pair`` dataclass directly (canonical order only).
    stats : GroupStatsView
        Metadata (G, n_kept, n_perm, correction, random_state, pvalue).
    group_labels : dict[str, str]
        Canonical → original label mapping (``{"g_1": "Warsaw", …}``).
        Empty when ``groups`` was not provided at construction.
    words, clusters, snippets
        Top-level view accessors — single-pair → continuous-style views;
        multi-pair → paired collection views.
    beta, gradient, beta_norm, alignment_scores
        Top-level per-pair arrays — single-pair → plain array/float;
        multi-pair → ``dict[tuple[str, str], …]``.
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
        Canonical group labels (``g_1``, ``g_2``, …) aligned with ``x``.
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

        Group labels are rewritten to canonical ``g_1, g_2, …`` form
        (in ``sorted(set(groups), key=str)`` order); the originals land in
        ``self.group_labels``. When ``groups`` is ``None`` the relabeling
        is skipped and ``group_labels`` is empty — this supports synthetic
        fixtures that feed ``pairs`` directly.

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
            Per-contrast statistics from the permutation test. Labels are
            rewritten to canonical form.
        words_rows : list of Word
            Flat list of neighbor words for all contrasts.
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
            Document vectors retained after filtering; required for test reruns
            and per-pair array computation.
        groups : ndarray of shape (n_kept,) or None
            Group labels aligned with ``x``; rewritten to canonical form here.
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

        # Canonical group label relabeling. When groups is None (pre-built
        # fixture path), skip the rewrite and leave pairs unchanged.
        if groups is None:
            self.groups = groups
            self.group_labels: dict[str, str] = {}
        else:
            self.groups, self.group_labels, pairs = self._canonicalize(groups, pairs)

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

        # Store flat row lists — top-level view accessors filter by contrast.
        self._words_rows = list(words_rows)
        self._cluster_rows = list(cluster_rows)
        self._cluster_words_rows = list(cluster_words_rows)
        self._snippets_rows = list(snippets_rows)

        self.pairs = PairsListView(pairs)

    # -------- canonicalization helper ---------------------------------

    @staticmethod
    def _canonicalize(
        groups: np.ndarray, pairs: list[Pair],
    ) -> tuple[np.ndarray, dict[str, str], list[Pair]]:
        """Rewrite ``groups`` and ``pairs`` to canonical ``g_1, g_2, …`` labels.

        Returns ``(relabeled_groups, group_labels_dict, canonical_pairs)``.
        """
        sorted_originals = sorted(set(groups), key=str)
        group_labels = {
            f"g_{i + 1}": str(orig) for i, orig in enumerate(sorted_originals)
        }
        orig_to_canonical = {orig: f"g_{i + 1}" for i, orig in enumerate(sorted_originals)}
        relabeled = np.array([orig_to_canonical[g] for g in groups], dtype=object)

        canonical_pairs: list[Pair] = []
        for p in pairs:
            cg1 = orig_to_canonical.get(p.g1, p.g1)
            cg2 = orig_to_canonical.get(p.g2, p.g2)
            can_g1, can_g2 = _canonical_pair_key(cg1, cg2)
            canonical_pairs.append(replace(
                p,
                g1=can_g1,
                g2=can_g2,
                contrast=f"{can_g1}_vs_{can_g2}",
            ))
        return relabeled, group_labels, canonical_pairs

    # -------- _update_pairs ------------------------------------------

    def _update_pairs(self, new_pairs: list[Pair], *, n_perm: int,
                      correction: str, random_state) -> None:
        """Swap in new pair rows + refresh stats after a gr.test() rerun.

        New pairs from the backend already carry canonical labels (because
        ``self.groups`` is canonical), but canonical ordering is re-applied
        defensively so the stored :class:`Pair` objects always have
        ``(g1, g2) == _canonical_pair_key(g1, g2)``.
        """
        self.n_perm = n_perm
        self.correction = correction
        self.random_state = random_state

        ordered_pairs: list[Pair] = []
        for p in new_pairs:
            can_g1, can_g2 = _canonical_pair_key(p.g1, p.g2)
            ordered_pairs.append(replace(
                p,
                g1=can_g1,
                g2=can_g2,
                contrast=f"{can_g1}_vs_{can_g2}",
            ))
        self.pairs = PairsListView(ordered_pairs)
        self.stats = GroupStatsView(
            G=self.G, n_kept=self.n_kept, n_perm=n_perm,
            correction=correction, random_state=random_state,
            pvalue=float(self.test.pvalue) if self.test is not None else float("nan"),
        )

        # Invalidate cached top-level array/dict accessors that depend on
        # the pair rows (beta/gradient/etc depend on Pair identities).
        for name in ("_pair_arrays", "beta", "gradient", "beta_norm", "alignment_scores"):
            self.__dict__.pop(name, None)
        # words / clusters / snippets come from _words_rows / _cluster_rows /
        # _snippets_rows which are NOT rebuilt here — they can stay cached.

    # -------- per-pair array computation -----------------------------

    def _compute_pair_arrays(
        self, canonical_pair: tuple[str, str],
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Compute ``(beta, gradient, beta_norm, alignment_scores)`` for one canonical pair.

        ``beta = c_{g1} − c_{g2}`` where centroids are computed from
        ``self.x`` masked by ``self.groups``. ``gradient = beta / ‖beta‖``.
        ``alignment_scores = self.x @ gradient``.
        """
        if self.x is None or self.groups is None:
            raise RuntimeError(
                "per-pair arrays require the original x / groups arrays; "
                "were they discarded by un-pickling or detach()?"
            )
        g1, g2 = canonical_pair
        x = self.x
        g = self.groups
        beta = x[g == g1].mean(axis=0) - x[g == g2].mean(axis=0)
        beta = np.asarray(beta, dtype=float).copy()
        beta.setflags(write=False)
        beta_norm = float(np.linalg.norm(beta))
        gradient = (beta / max(beta_norm, 1e-12)).copy()
        gradient.setflags(write=False)
        alignment = (x @ gradient).ravel()
        alignment.setflags(write=False)
        return beta, gradient, beta_norm, alignment

    def _pair_keys(self) -> list[tuple[str, str]]:
        """Return the list of canonical pair tuples in the pair-row order."""
        return [(p.g1, p.g2) for p in self.pairs]

    # -------- top-level array accessors -------------------------------

    @cached_property
    def _pair_arrays(self) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
        """One computation of (beta, gradient, beta_norm, alignment_scores) per canonical pair.

        Backs the four public per-pair-array accessors so each pair is processed
        exactly once, even when all four properties are accessed.
        """
        return {k: self._compute_pair_arrays(k) for k in self._pair_keys()}

    @cached_property
    def beta(self):
        """Pair contrast vector(s).

        Single-pair (``len(pairs) == 1``): ``ndarray`` of shape ``(D,)`` —
        ``c_{g1} − c_{g2}``. Multi-pair: ``dict[tuple, ndarray]`` keyed on
        canonical pair tuples.
        """
        d = self._pair_arrays
        if len(d) == 1:
            return next(iter(d.values()))[0]
        return {k: v[0] for k, v in d.items()}

    @cached_property
    def gradient(self):
        """Unit-length pair direction(s) ``beta / ‖beta‖``.

        Single-pair: ``ndarray(D,)``. Multi-pair: ``dict[tuple, ndarray]``.
        """
        d = self._pair_arrays
        if len(d) == 1:
            return next(iter(d.values()))[1]
        return {k: v[1] for k, v in d.items()}

    @cached_property
    def beta_norm(self):
        """Magnitude ‖β‖ of the pair contrast direction(s).

        Single-pair: ``float``. Multi-pair: ``dict[tuple, float]``.
        """
        d = self._pair_arrays
        if len(d) == 1:
            return next(iter(d.values()))[2]
        return {k: v[2] for k, v in d.items()}

    @cached_property
    def alignment_scores(self):
        """Per-document projection(s) onto the pair gradient — ``x @ gradient``.

        Single-pair: ``ndarray(n_kept,)``. Multi-pair: ``dict[tuple, ndarray]``.
        """
        d = self._pair_arrays
        if len(d) == 1:
            return next(iter(d.values()))[3]
        return {k: v[3] for k, v in d.items()}

    # -------- top-level view accessors --------------------------------

    @cached_property
    def words(self):
        """Top-level words view.

        Single-pair: :class:`~ssdiff.results.continuous_result.WordsView`
        built from all ``words_rows`` (contrast filter is a no-op since
        rows all carry the same canonical contrast).
        Multi-pair: :class:`~ssdiff.results.paired_view.WordsViewPaired`
        whose children are per-contrast ``WordsView`` instances.
        """
        from ssdiff.results.continuous_result import WordsView
        from ssdiff.results.paired_view import WordsViewPaired

        keys = self._pair_keys()
        if len(keys) == 1:
            return WordsView(list(self._words_rows))

        children: dict[tuple[str, str], WordsView] = {}
        for key in keys:
            contrast = f"{key[0]}_vs_{key[1]}"
            rows = [w for w in self._words_rows if w.contrast == contrast]
            children[key] = WordsView(rows)
        return WordsViewPaired(children)

    @cached_property
    def clusters(self):
        """Top-level clusters view.

        Single-pair: a ``ClustersIndex``-like object (``.pos`` / ``.neg`` →
        :class:`~ssdiff.results.continuous_result.ClustersViewSided`).
        Multi-pair: :class:`~ssdiff.results.paired_view.ClustersIndexPaired`.
        """
        from ssdiff.results.paired_view import ClustersIndexPaired

        keys = self._pair_keys()
        if len(keys) == 1:
            contrast = f"{keys[0][0]}_vs_{keys[0][1]}"
            return _SinglePairClustersIndex(
                contrast=contrast,
                cluster_rows=self._cluster_rows,
                cluster_words_rows=self._cluster_words_rows,
            )

        children = {}
        for key in keys:
            contrast = f"{key[0]}_vs_{key[1]}"
            children[key] = _SinglePairClustersIndex(
                contrast=contrast,
                cluster_rows=self._cluster_rows,
                cluster_words_rows=self._cluster_words_rows,
            )
        return ClustersIndexPaired(children)

    @cached_property
    def snippets(self):
        """Top-level snippets view.

        Single-pair: :class:`~ssdiff.results.continuous_result.SnippetsView`.
        Multi-pair: :class:`~ssdiff.results.paired_view.SnippetsViewPaired`.
        """
        from ssdiff.results.continuous_result import SnippetsView
        from ssdiff.results.paired_view import SnippetsViewPaired

        keys = self._pair_keys()
        if len(keys) == 1:
            return SnippetsView(list(self._snippets_rows))

        children: dict[tuple[str, str], SnippetsView] = {}
        for key in keys:
            contrast = f"{key[0]}_vs_{key[1]}"
            rows = [s for s in self._snippets_rows if s.contrast == contrast]
            children[key] = SnippetsView(rows)
        return SnippetsViewPaired(children)

    # -------- Result machinery ---------------------------------------

    _access = (
        "stats", "test", "pairs", "words", "clusters", "snippets",
        "beta", "gradient", "beta_norm", "alignment_scores",
        "group_labels",
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

    def _iter_pair_views(self):
        """Yield ``(pair, words_view, clusters_index, snippets_view)`` for each Pair.

        Works for both single-pair and multi-pair dispatch.
        """
        pairs = list(self.pairs)
        if len(pairs) == 1:
            p = pairs[0]
            yield p, self.words, self.clusters, self.snippets
        else:
            for p in pairs:
                key = (p.g1, p.g2)
                yield p, self.words[key], self.clusters[key], self.snippets[key]

    def report(self, *, top_words: int | None = 5,
               clusters: int | None = None,
               snippets_per_cluster: int | None = None) -> Report:
        """Build a multi-section narrative Report for this group result.

        Parameters
        ----------
        top_words : int or None
            Words per pole per contrast to include. ``None`` skips the words
            section.
        clusters : int or None
            Number of clusters per side per contrast to include.
            ``None`` skips the clusters section.
        snippets_per_cluster : int or None
            Number of snippets per side per contrast to include.
            ``None`` skips the snippets section.

        Returns
        -------
        Report
            A ``Report`` with omnibus, group-labels, pairwise-contrasts, and
            optional per-pair top-words, clusters, and snippets sections.
        """
        sections = []

        # Omnibus section (includes random_state)
        omnibus_rows = [
            ("G", self.G),
            ("n_kept", fmt_count(self.n_kept)),
            ("n_perm", fmt_count(self.n_perm)),
            ("correction", self.correction),
            ("T", fmt_d(self.test.omnibus_T)),
            ("p", fmt_p(self.test.omnibus_p)),
        ]
        if self.random_state is not None:
            omnibus_rows.append(("random_state", self.random_state))
        sections.append(Section(title="Omnibus", kind="kv", rows=omnibus_rows))

        # Group labels section
        if self.group_labels:
            label_rows = [(k, v) for k, v in sorted(self.group_labels.items())]
            sections.append(Section(title="Group labels", kind="kv", rows=label_rows))

        # Pairwise contrasts table
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

        # Top words — one table per pair
        if top_words and self.embeddings is not None:
            for p, words_view, _cl, _sn in self._iter_pair_views():
                pair_title = f"{p.g1} vs {p.g2}"
                pos_words = [w for w in words_view if w.side == "pos"][:top_words]
                neg_words = [w for w in words_view if w.side == "neg"][:top_words]
                word_rows = []
                for w in pos_words + neg_words:
                    word_rows.append([w.side, w.rank, w.word, fmt_r(w.cos_beta, signed=True)])
                sections.append(Section(
                    title=pair_title,
                    kind="table",
                    headers=["side", "rank", "word", "cos_β"],
                    rows=word_rows,
                    numeric=[False, True, False, True],
                ))

        # Clusters — one table per pair per side (pos + neg)
        if clusters and self.embeddings is not None:
            for p, _wv, clusters_index, _sn in self._iter_pair_views():
                pair_title = f"{p.g1} vs {p.g2}"
                for side in ("pos", "neg"):
                    cl_view = getattr(clusters_index, side)
                    cl_rows = []
                    for c in list(cl_view)[:clusters]:
                        cl_rows.append([
                            c.cluster_id, c.size,
                            fmt_r(c.coherence),
                            fmt_r(c.centroid_cos_beta, signed=True),
                        ])
                    sections.append(Section(
                        title=f"{pair_title} — {side}",
                        kind="table",
                        headers=["cluster", "size", "coherence", "centroid cos_β"],
                        rows=cl_rows,
                        numeric=[True, True, True, True],
                    ))

        # Snippets — one table per pair (snippets are pre-stored; no embeddings needed)
        if snippets_per_cluster:
            for p, _wv, _cl, snippets_view in self._iter_pair_views():
                pair_title = f"{p.g1} vs {p.g2}"
                pos_snips = [s for s in snippets_view if s.side == "pos"][:snippets_per_cluster]
                neg_snips = [s for s in snippets_view if s.side == "neg"][:snippets_per_cluster]
                snip_rows = []
                for s in pos_snips + neg_snips:
                    snip_rows.append([s.side, s.cluster_id, fmt_r(s.cosine, signed=True), s.text_window])
                sections.append(Section(
                    title=f"Snippets — {pair_title}",
                    kind="table",
                    headers=["side", "cluster_id", "cos", "text_window"],
                    rows=snip_rows,
                    numeric=[False, True, True, False],
                ))

        return Report(
            title=f"GroupResult — G={self.G}",
            subtitle=f"(n = {fmt_count(self.n_kept)}, omnibus p = {fmt_p(self.test.omnibus_p)})",
            sections=sections,
        )
