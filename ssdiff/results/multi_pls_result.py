"""MultiPLSResult — rotated PLS with per-dim interpretable axes.

Container of ``_PLSComponentResult`` leaves keyed by ``"dim-1"``,
``"dim-2"``, …, ``"combined"``. Mirrors the ``GroupResult`` /
``PairResult`` pattern: one shared ``.test`` on the container, one
``.beta`` / ``.words`` / etc. per leaf.

Access patterns (canonical):
- ``res["dim-1"]`` → ``_PLSComponentResult`` for one rotated axis
- ``res["combined"]`` → leaf holding the unrotated PLS prediction β
- ``res.test(...)`` → rerun raw_perm / split_nb / split_perm at the container level
- ``res.pls_info`` → ``{k, rotate, order, signs, rotation_meta}``

Power-user shortcuts (same data, different angle):
- ``res.words["dim-1"]`` → same as ``res["dim-1"].words``; returned by
  the ``_ShimView`` inherited from ``_MultiContainer``.
"""

from __future__ import annotations

import numpy as np

from ssdiff.results.core import ScalarView, TestView
from ssdiff.results.multi_container import _MultiContainer
from ssdiff.results.single_result import _SingleResult


class _PLSComponentResult(_SingleResult):
    """One rotated PLS axis (or the combined β). Leaf inside ``MultiPLSResult``.

    ``key`` identifies the leaf: ``"dim-1"`` / ``"dim-2"`` / … /
    ``"combined"``. For dim leaves, ``dim_index`` is the 0-based column in
    the container's ``_W_rot``; for the combined leaf, ``dim_index`` is
    ``None`` and ``beta`` comes from ``_beta_combined``.

    Has no independent ``.test`` — use ``container.test(...)``.
    """

    def __init__(
        self,
        *,
        container,
        key: str,
        dim_index: int | None,
    ):
        self._container = container
        self._key = key
        self._dim_index = dim_index

        if dim_index is None:
            beta = np.asarray(container._beta_combined, dtype=float).copy()
        else:
            beta = np.asarray(container._W_rot[:, dim_index], dtype=float).copy()

        super().__init__(
            x=None,
            beta=beta,
            embeddings=container.embeddings,
            corpus=container.corpus,
            lexicon=container.lexicon,
            window=container.window,
            sif_a=container.sif_a,
            lang=container.lang,
        )
        # _x is served lazily through the .x property from the container.
        self._x = None

    @property
    def x(self) -> np.ndarray:
        """Container's x — shared read-only view, no per-leaf copy."""
        return self._container._x

    @property
    def words(self):
        """Words view tagged with this leaf's key as the contrast."""
        from ssdiff.results.continuous_result import WordsView

        cache_key = ("words", ())
        if cache_key in self._cache:
            return self._cache[cache_key]
        self._require_resource("embeddings", "words")

        rows = self._compute_words_rows(contrast=self._key)

        view = WordsView(rows)
        self._cache[cache_key] = view
        return view

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_container"] = None
        return state


# ---------- PLSInfoView ----------

class PLSInfoView(ScalarView):
    """Scalar view exposing MultiPLS fit diagnostics (n_components, rotation, order)."""

    _name = "pls_info"
    _columns = (
        "n_components", "rotate", "order", "signs",
        "kaiser_normalized", "sweeps", "V_converged",
        "pvalue_source", "random_state",
    )

    def __init__(
        self, *,
        n_components: int,
        rotate: str,
        order: np.ndarray,
        signs: np.ndarray,
        rotation_meta: dict,
        pvalue_source: str | None,
        random_state: int | None,
    ):
        super().__init__()
        self._row = {
            "n_components": int(n_components),
            "rotate": rotate,
            "order": tuple(int(i) for i in order) if order is not None else None,
            "signs": tuple(float(s) for s in signs) if signs is not None else None,
            "kaiser_normalized": rotation_meta.get("kaiser_normalized"),
            "sweeps": rotation_meta.get("sweeps"),
            "V_converged": rotation_meta.get("V_converged"),
            "pvalue_source": pvalue_source,
            "random_state": random_state,
        }

    def __iter__(self):
        yield dict(self._row)


# ---------- MultiPLSStatsView ----------

class MultiPLSStatsView(ScalarView):
    """Scalar view exposing model-level stats for MultiPLSResult."""

    _name = "stats"
    _columns = ("r2", "pvalue", "n", "n_components", "rotate")

    def __init__(self, *, r2: float, pvalue: float, n: int,
                 n_components: int, rotate: str):
        super().__init__()
        self._row = {
            "r2": float(r2),
            "pvalue": float(pvalue) if pvalue is not None else float("nan"),
            "n": int(n),
            "n_components": int(n_components),
            "rotate": rotate,
        }

    def __iter__(self):
        yield dict(self._row)


# ---------- MultiPLSTestView ----------

class MultiPLSTestView(TestView):
    """`.test` for MultiPLSResult — confirmatory test at the fitted k."""

    _columns = ("name", "pvalue", "split_r2", "n_splits",
                "n_perm", "random_state")
    _default_name = "split_nb"

    _DEFAULTS = {
        "raw_perm":   dict(n_perm=2000, seed=None, verbose=False),
        "split_nb":   dict(n_splits=50, seed=None, verbose=False),
        "split_perm": dict(n_splits=50, n_perm=2000, seed=None, verbose=False),
        "score":      dict(seed=None, verbose=False),
        "e":          dict(seed=None, verbose=False),
    }

    def _run(self, name, params):
        from ssdiff.backends.pls import confirmatory_test

        if name not in self._DEFAULTS:
            raise ValueError(
                f"Unknown PLS test {name!r}. "
                f"Available: {tuple(self._DEFAULTS)}"
            )
        merged = {**self._DEFAULTS[name], **params}
        parent = self._parent
        return confirmatory_test(
            parent._x_raw, parent._y_raw, parent.n_components,
            method=name,
            n_perm=merged.get("n_perm", 2000),
            n_splits=merged.get("n_splits", 50),
            seed=merged["seed"],
            verbose=merged["verbose"],
        )

    def _on_rerun(self):
        self._parent._refresh_pvalue(self.pvalue)

    def _rerun_hint(self) -> str:
        return (
            "Rerun: .test('raw_perm'|'split_nb'|'split_perm'|'score'|'e', "
            "n_perm=..., n_splits=...)"
        )


# ---------- MultiPLSResult ----------

class MultiPLSResult(_MultiContainer):
    """Container of ``_PLSComponentResult`` leaves — rotated PLS axes + combined β.

    Typical use::

        res = ssd.fit_multipls(n_components=2, rotate="varimax")
        res["dim-1"].words         # rotated-axis top words
        res["combined"].words      # unrotated prediction β top words
        res.test("split_perm")     # rerun test on the whole model
        res.pls_info               # rotation diagnostics
    """

    _access = (
        "stats", "pls_info", "test",
        "words", "clusters", "snippets",
        "beta", "gradient", "beta_norm", "alignment_scores",
        "report()", "test(...)", "attach(...)",
    )
    _arrays = ("x", "y", "W", "P", "Q", "W_rot", "T_rot", "beta_combined")

    def __init__(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        W: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        W_rot: np.ndarray,
        T_rot: np.ndarray,
        beta_combined: np.ndarray,
        n_components: int,
        rotation_meta: dict,
        r2: float,
        test_name: str | None,
        test_info: dict | None,
        # The following echo SSD state for reconstruction & re-running tests.
        embeddings=None, corpus=None, lexicon=None,
        window: int = 3, sif_a: float = 1e-3, lang: str | None = None,
        random_state: int | None = None,
    ):
        super().__init__()
        self._x = np.asarray(x, dtype=np.float64)
        self._x_raw = self._x  # kept under a second name for clarity in tests
        self._y_raw = np.asarray(y, dtype=np.float64)
        self.W = np.asarray(W, dtype=np.float64)
        self.P = np.asarray(P, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self._W_rot = np.asarray(W_rot, dtype=np.float64)
        self.W_rot = self._W_rot
        self.T_rot = np.asarray(T_rot, dtype=np.float64)
        self._beta_combined = np.asarray(beta_combined, dtype=np.float64)
        self.beta_combined = self._beta_combined

        self.n_components = int(n_components)
        self._rotation_meta = dict(rotation_meta)
        self.random_state = random_state

        self.embeddings = embeddings
        self.corpus = corpus
        self.lexicon = set(lexicon) if lexicon else set()
        self.window = int(window)
        self.sif_a = float(sif_a)
        self.lang = lang if lang is not None else getattr(corpus, "lang", None)

        pvalue = (test_info or {}).get("pvalue", float("nan"))
        self.stats = MultiPLSStatsView(
            r2=r2, pvalue=pvalue, n=self._x.shape[0],
            n_components=self.n_components, rotate=self._rotation_meta["rotate"],
        )

        # Build pls_info. `order` / `signs` are not strictly needed on the
        # container after construction — we echo them via rotation_meta
        # (they're the identity by the time mpls_fit returns, because
        # reordering has already been applied to W_rot).
        self.pls_info = PLSInfoView(
            n_components=self.n_components,
            rotate=self._rotation_meta["rotate"],
            order=self._rotation_meta.get("order", np.arange(self.n_components)),
            signs=self._rotation_meta.get("signs", np.ones(self.n_components)),
            rotation_meta=self._rotation_meta,
            pvalue_source=test_name,
            random_state=random_state,
        )

        self.test = MultiPLSTestView(
            parent=self,
            name=test_name,
            info=test_info,
        )

        self._leaves = self._build_leaves()

    # ------ property for x/y shims consistent with GroupResult ---------
    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def y(self) -> np.ndarray:
        return self._y_raw

    # ------ leaf construction & key formatting ---------------------------
    def _build_leaves(self) -> dict:
        leaves: dict = {}
        for i in range(self.n_components):
            key = f"dim-{i+1}"
            leaves[key] = _PLSComponentResult(
                container=self, key=key, dim_index=i,
            )
        leaves["combined"] = _PLSComponentResult(
            container=self, key="combined", dim_index=None,
        )
        return leaves

    def _key_to_str(self, key: str) -> str:
        return key.replace("-", "_")

    def _key_repr(self, key: str) -> str:
        if key == "combined":
            return "Combined"
        if key.startswith("dim-"):
            return f"Dim {key.split('-', 1)[1]}"
        return str(key)

    def __setstate__(self, state):
        self.__dict__.update(state)
        for leaf in self._leaves.values():
            leaf._container = self

    # ------ attach fan-out -----------------------------------------------
    def attach(self, corpus=None, embeddings=None):
        """Re-wire ``corpus`` / ``embeddings`` on the container AND every leaf.

        The default ``Result.attach`` only updates ``self``; a
        ``MultiPLSResult`` also needs its leaves updated so their
        ``.words`` views pick up the new resources.
        """
        super().attach(corpus=corpus, embeddings=embeddings)
        for leaf in self._leaves.values():
            if corpus is not None:
                leaf.corpus = corpus
            if embeddings is not None:
                leaf.embeddings = embeddings
        return self

    # ------ p-value refresh (called from MultiPLSTestView._on_rerun) -----
    def _refresh_pvalue(self, new_pvalue: float | None) -> None:
        if new_pvalue is None:
            return
        r = self.stats._row
        self.stats = MultiPLSStatsView(
            r2=r["r2"], pvalue=float(new_pvalue), n=r["n"],
            n_components=r["n_components"], rotate=r["rotate"],
        )

    # ------ display --------------------------------------------------------
    def _summary(self) -> str:
        from ssdiff.results.format import fmt_count, fmt_p, fmt_r
        s = self.stats._row
        return (
            f"MultiPLSResult  k={s['n_components']}  rotate={s['rotate']}  "
            f"r²={fmt_r(s['r2'])}  p={fmt_p(s['pvalue'])}  n={fmt_count(s['n'])}"
        )

    def _summary_html(self) -> str:
        return f"<p>{self._summary()}</p>"

    def _save_hint(self) -> str:
        return (
            "Save:  res['dim-1'].words.save('dim1_words.csv')\n"
            "       res.words.save('words.csv')             # per-leaf fan-out\n"
            "       res.report().save('report.md')          # narrative"
        )

    def _save_hint_html(self) -> str:
        return f"<pre class='ssd-save-hint'>{self._save_hint()}</pre>"

    # ------ minimal report (v1 scope) -------------------------------------
    def report(self, *, top_words: int | None = 5):
        """Minimal v1 report: fit summary + per-leaf words tables.

        ``clusters`` / ``snippets`` / ``misdiagnosed`` sections are
        reserved for Milestone 2 (per-leaf feature-parity with
        ``PairResult``); passing them here would silently no-op.
        """
        from ssdiff.results.format import fmt_count, fmt_p, fmt_r
        from ssdiff.results.report import Report, Section

        s = self.stats._row
        pi = next(iter(self.pls_info))
        sections = [
            Section(
                title="Stats",
                kind="kv",
                rows=[
                    ("r²", fmt_r(s["r2"])),
                    ("p-value", fmt_p(s["pvalue"])),
                    ("n", fmt_count(s["n"])),
                    ("n_components", s["n_components"]),
                    ("rotate", s["rotate"]),
                ],
            ),
            Section(
                title="Rotation",
                kind="kv",
                rows=[
                    ("kaiser_normalized", pi["kaiser_normalized"]),
                    ("sweeps", pi["sweeps"]),
                    ("V_converged",
                     f"{pi['V_converged']:.4f}"
                     if pi["V_converged"] is not None else "-"),
                ],
            ),
        ]

        if top_words and self.embeddings is not None:
            for key, leaf in self._leaves.items():
                heading = self._key_repr(key)
                pos = [w for w in leaf.words if w.side == "pos"][:top_words]
                neg = [w for w in leaf.words if w.side == "neg"][:top_words]
                rows = []
                for w in pos + neg:
                    rows.append([
                        w.side, w.rank, w.word,
                        fmt_r(w.cos_beta, signed=True),
                    ])
                sections.append(Section(
                    title=heading,
                    kind="table",
                    headers=["side", "rank", "word", "cos_β"],
                    rows=rows,
                    numeric=[False, True, False, True],
                ))

        return Report(
            title=f"MultiPLSResult — k={self.n_components}, rotate={self._rotation_meta['rotate']}",
            subtitle=f"(n = {fmt_count(s['n'])}, r² = {fmt_r(s['r2'])})",
            sections=sections,
        )
