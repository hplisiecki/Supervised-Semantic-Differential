"""MultiPLSResult — PLS3 fit of several outcomes, one leaf per dimension.

Container of ``_PLSComponentResult`` leaves keyed by ``"dim-1"``,
``"dim-2"``, …. Mirrors the ``GroupResult`` / ``PairResult`` pattern: one
shared ``.test`` on the container, one ``.beta`` / ``.words`` /
``.outcomes`` / etc. per leaf.

Access patterns (canonical):
- ``res["dim-1"]`` → ``_PLSComponentResult`` for one (rotated) dimension
- ``res["dim-1"].outcomes`` → outcome loadings on that dimension
- ``res.test(...)`` → rerun the split_exact LV1 test at the container level
- ``res.pls_info`` → ``{k, rotate, order, signs, rotation diagnostics}``

Power-user shortcuts (same data, different angle):
- ``res.words["dim-1"]`` → same as ``res["dim-1"].words``; returned by
  the ``_ShimView`` inherited from ``_MultiContainer``.
"""

from __future__ import annotations

import numpy as np

from ssdiff.results.core import ScalarView, TestView, View
from ssdiff.results.multi_container import _MultiContainer
from ssdiff.results.schema import Outcome
from ssdiff.results.single_result import _SingleResult


# ---------- OutcomesView ----------

class OutcomesView(View[Outcome]):
    """Tabular view of outcome loadings (``V_rot`` column) on one dimension."""

    _name = "outcomes"
    _columns = ("outcome", "group", "loading")

    def __init__(self, rows: list[Outcome], *, _no_trunc: bool = False):
        super().__init__(_no_trunc=_no_trunc)
        self._rows = rows

    def __iter__(self): return iter(self._rows)
    def __len__(self): return len(self._rows)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return OutcomesView(self._rows[i], _no_trunc=True)
        return self._rows[i]


# ---------- per-dimension stats ----------

class PLSComponentStatsView(ScalarView):
    """Scalar view exposing one dimension's share of cross-covariance and score correlation."""

    _name = "stats"
    _columns = ("cov_share", "score_corr")

    def __init__(self, *, cov_share: float, score_corr: float):
        super().__init__()
        self._row = {"cov_share": float(cov_share), "score_corr": float(score_corr)}

    def __iter__(self):
        yield dict(self._row)


class _PLSComponentResult(_SingleResult):
    """One PLS3 dimension. Leaf inside ``MultiPLSResult``.

    ``key`` is ``"dim-1"`` / ``"dim-2"`` / …; ``dim_index`` is the 0-based
    column in the container's ``U_rot`` / ``V_rot``. ``beta`` is that
    ``U_rot`` column divided by ``X_scale``: the raw doc-vector space that
    ``fit_pls``'s ``coef / scale`` and the ``"words"`` rotation target use.

    Has no independent ``.test`` and no p-value — use ``container.test(...)``.
    """

    _access = ("stats", "outcomes", "words", "clusters", "snippets")
    _arrays = ("x", "beta", "gradient", "alignment_scores")

    def __init__(self, *, container, key: str, dim_index: int):
        self._container = container
        self._key = key
        self._dim_index = dim_index

        beta = container._U_rot[:, dim_index] / container._X_scale

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

        sigma2 = container.singular_values ** 2
        # c_i = (R' Σ² R)_ii — dimension i's squared cross-covariance.
        c_i = float(container.R[:, dim_index] ** 2 @ sigma2)
        score_corr = float(np.corrcoef(
            container.x_scores[:, dim_index], container.y_scores[:, dim_index],
        )[0, 1])
        self.stats = PLSComponentStatsView(
            cov_share=c_i / float(sigma2.sum()), score_corr=score_corr,
        )

        loadings = container.V_rot[:, dim_index]
        groups = container.outcome_groups or {}
        group_of = {name: g for g, members in groups.items() for name in members}
        rows = [
            Outcome(outcome=name, group=group_of.get(name), loading=float(v))
            for name, v in zip(container.outcome_names, loadings)
        ]
        by_name = {r.outcome: r.loading for r in rows}
        rows += [
            Outcome(outcome=g, group=g,
                    loading=float(np.mean([by_name[n] for n in members])))
            for g, members in groups.items()
        ]
        self.outcomes = OutcomesView(rows)

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
    """Scalar view exposing PLS3 fit diagnostics (k, rotation, order, signs)."""

    _name = "pls_info"
    _columns = (
        "k", "rotate", "order", "signs",
        "kaiser_normalized", "sweeps", "V_converged", "random_state",
    )

    def __init__(
        self, *,
        k: int,
        rotate: str | None,
        order: np.ndarray,
        signs: np.ndarray,
        rotation_meta: dict,
        random_state: int | None,
    ):
        super().__init__()
        self._row = {
            "k": int(k),
            "rotate": rotate,
            "order": tuple(int(i) for i in order),
            "signs": tuple(float(s) for s in signs),
            "kaiser_normalized": rotation_meta["kaiser_normalized"],
            "sweeps": rotation_meta["sweeps"],
            "V_converged": rotation_meta["V_converged"],
            "random_state": random_state,
        }

    def __iter__(self):
        yield dict(self._row)


# ---------- MultiPLSStatsView ----------

class MultiPLSStatsView(ScalarView):
    """Scalar view exposing model-level stats for MultiPLSResult."""

    _name = "stats"
    _columns = ("n", "q", "k", "rotate", "cov_explained", "pvalue")

    def __init__(self, *, n: int, q: int, k: int, rotate: str | None,
                 cov_explained: float, pvalue: float | None):
        super().__init__()
        self._row = {
            "n": int(n),
            "q": int(q),
            "k": int(k),
            "rotate": rotate,
            "cov_explained": float(cov_explained),
            "pvalue": float(pvalue) if pvalue is not None else float("nan"),
        }

    def __iter__(self):
        yield dict(self._row)


# ---------- MultiPLSTestView ----------

class MultiPLSTestView(TestView):
    """`.test` for MultiPLSResult — split_exact PLS3 test at LV1.

    Statistic is the held-out LV1 correlation (``tanh`` of the mean
    Fisher-z over splits); the p-value is rotation-independent. Leaves
    carry no p-value.
    """

    _columns = ("name", "pvalue", "statistic", "n_splits", "n_perm", "random_state")
    _default_name = "split_exact"

    _DEFAULTS = {
        "split_exact": dict(n_splits=50, n_perm=1000, seed=None, verbose=False),
    }

    def _run(self, name, params):
        import plskit

        if name is not None and name != "split_exact":
            raise TypeError(
                f"MultiPLSTestView only supports method='split_exact' "
                f"(got {name!r}). Pass kwargs only: "
                f".test(n_splits=..., n_perm=..., seed=...)."
            )
        merged = {**self._DEFAULTS["split_exact"], **params}
        parent = self._parent
        r = plskit.pls3_confirmatory_test(
            parent._x, parent._y_raw, 1,
            method="split_exact",
            args={"n_perm": int(merged["n_perm"]),
                  "n_splits": int(merged["n_splits"])},
            pre_standardized_X=False,
            pre_standardized_Y=False,
            seed=merged["seed"],
            verbose=merged["verbose"],
        )
        info = {
            "pvalue": float(r.pvalue),
            "statistic": float(r.statistic),
            "n_splits": r.n_splits,
            "n_perm": r.n_perm,
            "random_state": r.seed,
        }
        return "split_exact", info

    def _on_rerun(self):
        self._parent._refresh_pvalue(self.pvalue)

    def _rerun_hint(self) -> str:
        return "Rerun: .test(n_splits=..., n_perm=..., seed=...)"


# ---------- MultiPLSResult ----------

class MultiPLSResult(_MultiContainer):
    """Container of ``_PLSComponentResult`` leaves — PLS3 dimensions.

    Typical use::

        res = ssd.fit_multipls(k=2, rotate="outcomes")
        res["dim-1"].outcomes      # outcome loadings on dimension 1
        res["dim-1"].words         # top words along dimension 1
        res.test(n_splits=100, n_perm=2000, seed=42)  # rerun the LV1 test
        res.pls_info               # rotation diagnostics
    """

    _access = (
        "stats", "pls_info", "test",
        "words", "clusters", "snippets",
        "beta", "gradient", "beta_norm", "alignment_scores",
        "outcome_names", "outcome_groups",
        "report()", "test(...)", "attach(...)",
    )
    _arrays = (
        "x", "y", "U", "V", "U_rot", "V_rot", "singular_values",
        "x_scores", "y_scores", "R",
    )

    def __init__(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
        U_rot: np.ndarray,
        V_rot: np.ndarray,
        singular_values: np.ndarray,
        x_scores: np.ndarray,
        y_scores: np.ndarray,
        R: np.ndarray,
        order: np.ndarray,
        signs: np.ndarray,
        rotation_meta: dict,
        rotate: str | None,
        cov_explained: float,
        outcome_names: list[str],
        outcome_groups: dict[str, list[str]] | None,
        X_scale: np.ndarray,
        test_name: str | None,
        test_info: dict | None,
        # The following echo SSD state for reconstruction & re-running tests.
        embeddings=None, corpus=None, lexicon=None,
        window: int = 3, sif_a: float = 1e-3, lang: str | None = None,
        random_state: int | None = None,
    ):
        super().__init__()
        self._x = np.asarray(x, dtype=np.float64)
        self._y_raw = np.asarray(y, dtype=np.float64)
        self.U = np.asarray(U, dtype=np.float64)
        self.V = np.asarray(V, dtype=np.float64)
        self._U_rot = np.asarray(U_rot, dtype=np.float64)
        self.U_rot = self._U_rot
        self.V_rot = np.asarray(V_rot, dtype=np.float64)
        self.singular_values = np.asarray(singular_values, dtype=np.float64)
        self.x_scores = np.asarray(x_scores, dtype=np.float64)
        self.y_scores = np.asarray(y_scores, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self._X_scale = np.asarray(X_scale, dtype=np.float64)

        self.k = int(self.U_rot.shape[1])
        self.rotate = rotate
        self.outcome_names = list(outcome_names)
        self.outcome_groups = (
            {g: list(m) for g, m in outcome_groups.items()}
            if outcome_groups is not None else None
        )
        self.random_state = random_state

        self.embeddings = embeddings
        self.corpus = corpus
        self.lexicon = set(lexicon) if lexicon else set()
        self.window = int(window)
        self.sif_a = float(sif_a)
        self.lang = lang if lang is not None else getattr(corpus, "lang", None)

        pvalue = (test_info or {}).get("pvalue", float("nan"))
        self.stats = MultiPLSStatsView(
            n=self._x.shape[0], q=self._y_raw.shape[1], k=self.k,
            rotate=rotate, cov_explained=cov_explained, pvalue=pvalue,
        )

        self.pls_info = PLSInfoView(
            k=self.k,
            rotate=rotate,
            order=order,
            signs=signs,
            rotation_meta=rotation_meta,
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
        for i in range(self.k):
            key = f"dim-{i+1}"
            leaves[key] = _PLSComponentResult(container=self, key=key, dim_index=i)
        return leaves

    def _key_to_str(self, key: str) -> str:
        return key.replace("-", "_")

    def _key_repr(self, key: str) -> str:
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
        self.stats = MultiPLSStatsView(**{**self.stats._row, "pvalue": new_pvalue})

    # ------ display --------------------------------------------------------
    def _summary(self) -> str:
        from ssdiff.results.format import fmt_count, fmt_p, fmt_r
        s = self.stats._row
        return (
            f"MultiPLSResult  k={s['k']}  q={s['q']}  rotate={s['rotate']}  "
            f"cov_explained={fmt_r(s['cov_explained'])}  "
            f"p={fmt_p(s['pvalue'])}  n={fmt_count(s['n'])}"
        )

    def _summary_html(self) -> str:
        return f"<p>{self._summary()}</p>"

    def _save_hint(self) -> str:
        return (
            "Save:  res['dim-1'].outcomes.save('dim1_outcomes.csv')\n"
            "       res['dim-1'].words.save('dim1_words.csv')\n"
            "       res.words.save('words.csv')             # per-leaf fan-out\n"
            "       res.report().save('report.md')          # narrative"
        )

    def _save_hint_html(self) -> str:
        return f"<pre class='ssd-save-hint'>{self._save_hint()}</pre>"

    # ------ report --------------------------------------------------------
    def report(self, *, top_words: bool | dict | None = True):
        """Report: fit summary, per-dimension outcome loadings and words.

        ``top_words`` accepts ``True`` / ``False`` / ``None`` / ``dict``:

        - ``False`` or ``None`` skips the words tables.
        - ``True`` includes them with defaults.
        - ``dict`` overrides defaults, e.g. ``top_words={"n": 10}``.

        Default: ``top_words={"n": 5}`` words per pole per leaf.
        ``clusters``, ``snippets`` and ``misdiagnosed`` are not accepted.
        """
        from ssdiff.results.format import fmt_count, fmt_p, fmt_r
        from ssdiff.results.report import Report, Section, _resolve_section

        tw = _resolve_section(top_words, {"n": 5}, name="top_words")

        s = self.stats._row
        pi = next(iter(self.pls_info))
        sections = [
            Section(
                title="Stats",
                kind="kv",
                rows=[
                    ("cov_explained", fmt_r(s["cov_explained"])),
                    ("p-value", fmt_p(s["pvalue"])),
                    ("n", fmt_count(s["n"])),
                    ("q", s["q"]),
                    ("k", s["k"]),
                    ("rotate", s["rotate"]),
                ],
            ),
            Section(
                title="Rotation",
                kind="kv",
                rows=[
                    ("kaiser_normalized", pi["kaiser_normalized"]),
                    ("sweeps", pi["sweeps"]),
                    ("V_converged", f"{pi['V_converged']:.4f}"),
                ],
            ),
            Section(
                title="Dimensions",
                kind="table",
                headers=["dim", "cov_share", "score_corr"],
                rows=[
                    [self._key_repr(key), fmt_r(leaf.stats.cov_share),
                     fmt_r(leaf.stats.score_corr, signed=True)]
                    for key, leaf in self._leaves.items()
                ],
                numeric=[False, True, True],
            ),
        ]

        for key, leaf in self._leaves.items():
            heading = self._key_repr(key)
            sections.append(Section(
                title=f"{heading} — outcomes",
                kind="table",
                headers=["outcome", "group", "loading"],
                rows=[
                    [o.outcome, o.group if o.group is not None else "-",
                     fmt_r(o.loading, signed=True)]
                    for o in leaf.outcomes
                ],
                numeric=[False, False, True],
            ))
            if tw and self.embeddings is not None:
                n_tw = tw["n"]
                pos = [w for w in leaf.words if w.side == "pos"][:n_tw]
                neg = [w for w in leaf.words if w.side == "neg"][:n_tw]
                sections.append(Section(
                    title=f"{heading} — words",
                    kind="table",
                    headers=["side", "rank", "word", "cos_β"],
                    rows=[
                        [w.side, w.rank, w.word, fmt_r(w.cos_beta, signed=True)]
                        for w in pos + neg
                    ],
                    numeric=[False, True, False, True],
                ))

        return Report(
            title=f"MultiPLSResult — k={self.k}, rotate={self.rotate}",
            subtitle=(
                f"(n = {fmt_count(s['n'])}, q = {s['q']}, "
                f"cov_explained = {fmt_r(s['cov_explained'])})"
            ),
            sections=sections,
        )
