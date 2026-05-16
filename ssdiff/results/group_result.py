"""GroupResult — group-comparison results backed by ``_MultiContainer``.

``GroupResult`` holds ``dict[pair_tuple, PairResult]`` leaves and inherits
aggregate accessors from :class:`~ssdiff.results.multi_container._MultiContainer`.

- ``gr[(g1, g2)]`` → ``PairResult`` for one canonical pair
- ``gr.words`` / ``.clusters`` / ``.snippets`` → :class:`~ssdiff.results.multi_container._ShimView` dicts
- ``gr.beta`` / ``.gradient`` / ``.beta_norm`` / ``.alignment_scores`` → plain dicts
- ``gr.stats`` / ``.test`` → omnibus metadata + rerunnable permutation test
- ``gr.pairs`` → tabular per-pair stats
- ``gr.group_labels`` → canonical → original label mapping

Canonical labels: all groups are renamed ``g1, g2, …`` in
``sorted(set(groups), key=str)`` order.  Originals survive in
``gr.group_labels``.  Canonical pair keys use the numeric trailing index
so that ``g2 < g10``.  Reverse-order tuple lookup raises ``KeyError`` —
no sign-flip anywhere.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import numpy as np

from ssdiff.results.core import Result, ScalarView, TestView, View
from ssdiff.results.format import fmt_count, fmt_d, fmt_p, fmt_r
from ssdiff.results.multi_container import _MultiContainer
from ssdiff.results.report import (
    Report,
    Section,
    _build_cluster_section,
    _resolve_section,
)
from ssdiff.results.schema import (
    Cluster,
    ClusterWord,
    Pair,
    Snippet,
    Word,
)
from ssdiff.results.single_result import _SingleResult


# ---------- Canonical pair helpers ----------

def _canonical_pair_key(g1: str, g2: str) -> tuple[str, str]:
    """Return ``(g1, g2)`` sorted by the numeric trailing index of canonical labels.

    Falls back to string order for non-canonical labels.
    """
    def _sort_key(s: str) -> tuple[int, int | str]:
        if s.startswith("g"):
            tail = s[1:]
            if tail.isdigit():
                return (0, int(tail))
        return (1, s)
    return tuple(sorted((g1, g2), key=_sort_key))  # type: ignore[return-value]


# ---------- PairResult ----------

class PairResult(_SingleResult):
    """One group-vs-group contrast. Leaf inside a ``GroupResult`` container.

    Holds ``g1`` / ``g2`` labels and a ``groups_mask`` pointing into the
    container's ``x``. ``beta = c_{g1} − c_{g2}`` is computed once at
    construction; ``.x`` is a cached slice that shares memory with the
    container (no per-pair duplication).

    Has no independent ``.test`` — call ``container.test(...)`` to rerun.
    """

    def __init__(
        self,
        *,
        container,
        g1: str,
        g2: str,
        embeddings=None,
        corpus=None,
        lexicon=None,
        window: int = 3,
        sif_a: float = 1e-3,
        lang: str | None = None,
    ):
        self._container = container
        self.g1 = str(g1)
        self.g2 = str(g2)
        self.contrast = f"{self.g1}_{self.g2}"

        cx = container._x
        cg = container._groups
        if cx is None or cg is None:
            raise RuntimeError(
                "PairResult requires the container to hold _x and _groups; "
                "build GroupResult with x= and groups= to enable per-pair vectors."
            )
        self.groups_mask = (cg == self.g1) | (cg == self.g2)

        # Compute beta = mean(g1) - mean(g2). Copied to ensure independence.
        beta = cx[cg == self.g1].mean(axis=0) - cx[cg == self.g2].mean(axis=0)
        beta = np.asarray(beta, dtype=float).copy()

        # Call base __init__ with x=None; the .x property below supersedes
        # by slicing container._x on demand (cached).
        super().__init__(
            x=None,
            beta=beta,
            embeddings=embeddings, corpus=corpus,
            lexicon=lexicon, window=window, sif_a=sif_a, lang=lang,
        )
        # After super().__init__ sets self._x = None, reset both sentinels.
        self._x = None
        self._x_cache = None

    @property
    def x(self) -> np.ndarray:
        """Slice of the container's x belonging to this pair's two groups."""
        if self._x_cache is not None:
            return self._x_cache
        sliced = self._container._x[self.groups_mask]
        sliced.setflags(write=False)
        self._x_cache = sliced
        return sliced

    @property
    def words(self):
        """Words view with contrast tag set to ``'g1_g2'``."""
        from ssdiff.results.continuous_result import WordsView

        key = ("words", ())
        if key in self._cache:
            return self._cache[key]
        self._require_resource("embeddings", "words")
        rows = self._compute_words_rows(contrast=self.contrast)
        view = WordsView(rows)
        self._cache[key] = view
        return view

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_container"] = None
        state["_x_cache"] = None
        return state


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
                contrast=f"{g1}_{g2}",
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


# ---------- GroupResult ----------

class GroupResult(_MultiContainer):
    """Result from ``SSD.fit_groups()``. Container of ``PairResult`` leaves.

    Access patterns:
    - ``gr[(g1, g2)]`` → ``PairResult`` for one canonical pair
    - ``gr.words`` / ``.clusters`` / ``.snippets`` → shim dicts of per-pair views
    - ``gr.beta`` / ``.gradient`` / ``.beta_norm`` / ``.alignment_scores`` → plain dicts
    - ``gr.stats`` / ``.test`` → omnibus metadata + rerunnable permutation test
    - ``gr.pairs`` → tabular per-pair stats
    - ``gr.group_labels`` → canonical → original label mapping
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
        pairs: list,  # list[Pair]
        embeddings=None,
        corpus=None,
        x=None,
        groups=None,
        lang: str | None = None,
        lexicon=None,
        window: int = 3,
        sif_a: float = 1e-3,
    ):
        super().__init__()  # Result.__init__ sets self._cache
        self.embeddings = embeddings
        self.corpus = corpus
        self.G = G
        self.n_kept = n_kept
        self.n_perm = n_perm
        self.correction = correction
        self.random_state = random_state

        if groups is None:
            self._x = x
            self._groups = None
            self.group_labels: dict[str, str] = {}
        else:
            self._groups, self.group_labels, pairs = self._canonicalize(groups, pairs)
            self._x = x

        self.lang = lang if lang is not None else getattr(corpus, "lang", None)
        self.lexicon = set(lexicon) if lexicon is not None else set()
        self.window = int(window)
        self.sif_a = float(sif_a)

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
                "G": G, "n_kept": n_kept, "n_perm": n_perm,
                "correction": correction, "random_state": random_state,
            },
        )
        self.pairs = PairsListView(pairs)
        self._leaves = self._build_leaves()

    @property
    def x(self):
        return self._x

    @property
    def groups(self):
        return self._groups

    def _build_leaves(self) -> dict:
        leaves = {}
        if self._x is None or self._groups is None:
            return leaves
        for p in self.pairs:
            leaves[(p.g1, p.g2)] = PairResult(
                container=self, g1=p.g1, g2=p.g2,
                embeddings=self.embeddings, corpus=self.corpus,
                lexicon=self.lexicon, window=self.window,
                sif_a=self.sif_a, lang=self.lang,
            )
        return leaves

    def _key_to_str(self, key) -> str:
        return f"{key[0]}_{key[1]}"

    def _key_repr(self, key) -> str:
        return f"{key[0]} vs {key[1]}"

    # -------- canonicalization helper ---------------------------------

    @staticmethod
    def _canonicalize(
        groups: np.ndarray, pairs: list[Pair],
    ) -> tuple[np.ndarray, dict[str, str], list[Pair]]:
        """Rewrite ``groups`` and ``pairs`` to canonical ``g1, g2, …`` labels.

        Returns ``(relabeled_groups, group_labels_dict, canonical_pairs)``.
        """
        sorted_originals = sorted(set(groups), key=str)
        group_labels = {
            f"g{i + 1}": str(orig) for i, orig in enumerate(sorted_originals)
        }
        orig_to_canonical = {orig: f"g{i + 1}" for i, orig in enumerate(sorted_originals)}
        # Pair.g1/g2 come in as ``str(orig)`` from fit_groups — key a mirror of
        # the map by ``str(orig)`` so pair-label lookup works for non-string
        # originals (e.g. floats, ints) without changing ``groups`` dtype.
        str_to_canonical = {str(orig): canon for orig, canon in orig_to_canonical.items()}
        relabeled = np.array([orig_to_canonical[g] for g in groups], dtype=object)

        canonical_pairs: list[Pair] = []
        for p in pairs:
            cg1 = str_to_canonical.get(p.g1, p.g1)
            cg2 = str_to_canonical.get(p.g2, p.g2)
            can_g1, can_g2 = _canonical_pair_key(cg1, cg2)
            canonical_pairs.append(replace(
                p,
                g1=can_g1,
                g2=can_g2,
                contrast=f"{can_g1}_{can_g2}",
            ))
        return relabeled, group_labels, canonical_pairs

    # -------- _update_pairs ------------------------------------------

    def _update_pairs(self, new_pairs: list[Pair], *, n_perm: int,
                      correction: str, random_state) -> None:
        """Swap in new pair rows + refresh stats + leaves after a gr.test() rerun."""
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
                contrast=f"{can_g1}_{can_g2}",
            ))
        self.pairs = PairsListView(ordered_pairs)
        self.stats = GroupStatsView(
            G=self.G, n_kept=self.n_kept, n_perm=n_perm,
            correction=correction, random_state=random_state,
            pvalue=float(self.test.pvalue) if self.test is not None else float("nan"),
        )
        self._leaves = self._build_leaves()
        self._cache = {}

    # -------- dict access + canonical-order enforcement ---------------

    def __getitem__(self, key):
        """Zoom to one canonical pair. Only single-tuple form accepted.

        Accepts: ``gr[('g1', 'g2')]`` (canonical order only).
        Reverse order raises KeyError with canonical-hint message.
        """
        if not (isinstance(key, tuple) and len(key) == 2
                and all(isinstance(k, str) for k in key)):
            raise KeyError(
                f"GroupResult key must be a (g1, g2) tuple of two strings; got {key!r}"
            )
        if key not in self._leaves:
            canonical = _canonical_pair_key(*key)
            if canonical != key and canonical in self._leaves:
                raise KeyError(
                    f"pair must be accessed in canonical order {canonical!r}, got {key!r}"
                )
            raise KeyError(f"unknown pair {key!r}; known: {list(self._leaves.keys())!r}")
        return self._leaves[key]

    def __setstate__(self, state):
        self.__dict__.update(state)
        for leaf in self._leaves.values():
            leaf._container = self

    # -------- Result machinery ---------------------------------------

    _access = (
        "stats", "test", "pairs", "words", "clusters", "snippets",
        "beta", "gradient", "beta_norm", "alignment_scores",
        "group_labels", "report()", "test(...)", "attach(...)",
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

    def report(self, *,
               clusters: bool | dict | None = True,
               top_words: bool | dict | None = True) -> Report:
        """Build a multi-section narrative Report for this group result.

        ``Omnibus``, ``Group labels`` and ``Pairwise contrasts`` are always
        included. The remaining sections accept ``True`` / ``False`` /
        ``None`` / ``dict``:

        - ``False`` or ``None`` skips the section.
        - ``True`` (the default for every section) renders with defaults.
        - ``dict`` overrides defaults, e.g. ``clusters={"n": 20}``.

        Section defaults and dict keys
        ------------------------------
        - ``clusters`` — ``{"n": 10, "n_words": 5, "n_snippets": 1}`` per side
          per pair. Words appear inside each cluster row; snippets fill the
          "Representative Excerpt" column (set ``n_snippets=0`` to drop it).
        - ``top_words`` — ``{"n": 5}`` words per pole per pair.

        Returns
        -------
        Report
            A ``Report`` with omnibus, group-labels, pairwise-contrasts, and
            per-pair top-words and clusters sections.
        """
        tw = _resolve_section(top_words, {"n": 5}, name="top_words")
        cl = _resolve_section(
            clusters,
            {"n": 10, "n_words": 5, "n_snippets": 1},
            name="clusters",
        )

        sections = []

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

        if self.group_labels:
            label_rows = [(k, v) for k, v in sorted(self.group_labels.items())]
            sections.append(
                Section(title="Group labels", kind="kv", rows=label_rows)
            )

        pair_rows = []
        for p in self.pairs:
            pair_rows.append([
                p.contrast, fmt_d(p.T), fmt_p(p.p_raw),
                fmt_p(p.p_corrected), fmt_d(p.cohens_d),
                fmt_count(p.n_g1), fmt_count(p.n_g2),
            ])
        if pair_rows:
            sections.append(Section(
                title="Pairwise contrasts",
                kind="table",
                headers=["contrast", "T", "p_raw", "p_corrected",
                         "Cohen's d", "n_g1", "n_g2"],
                rows=pair_rows,
                numeric=[False, True, True, True, True, True, True],
            ))

        # Clusters — one table per pair per side (pos + neg)
        if cl and self.embeddings is not None:
            n_cl = cl["n"]
            n_words = cl["n_words"]
            n_snippets = cl["n_snippets"]
            for (g1, g2), leaf in self._leaves.items():
                pair_title = f"{g1} vs {g2}"
                clusters_view = leaf.clusters
                for side in ("pos", "neg"):
                    cl_view = getattr(clusters_view, side)

                    def _snippet_provider(_leaf=leaf, _side=side):
                        if getattr(_leaf, "corpus", None) is None:
                            return None
                        try:
                            return _leaf._cluster_snippets_for(_side)
                        except Exception:
                            return None

                    sections.append(_build_cluster_section(
                        title=f"{pair_title} — {side}",
                        clusters_view=cl_view,
                        n_clusters=n_cl,
                        n_words=n_words,
                        n_snippets=n_snippets,
                        snippet_provider=_snippet_provider,
                    ))

        # Top words — one table per pair
        if tw and self.embeddings is not None:
            n_tw = tw["n"]
            for (g1, g2), leaf in self._leaves.items():
                pair_title = f"{g1} vs {g2}"
                words_view = leaf.words
                pos_words = [w for w in words_view if w.side == "pos"][:n_tw]
                neg_words = [w for w in words_view if w.side == "neg"][:n_tw]
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

        return Report(
            title=f"GroupResult — G={self.G}",
            subtitle=f"(n = {fmt_count(self.n_kept)}, omnibus p = {fmt_p(self.test.omnibus_p)})",
            sections=sections,
        )
