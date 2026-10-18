"""SSD: Supervised Semantic Differential — continuous outcome analysis."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np

from ssdiff.corpus import Corpus, _check_tokenizer
from ssdiff.embeddings import Embeddings
from ssdiff.utils.math import _categorical_mask, f_sf, pca_fit_transform, standardize
from ssdiff.utils.vectors import build_and_normalize_doc_vectors, compute_global_sif


class SSD:
    """Supervised Semantic Differential — continuous outcome.

    Attributes
    ----------
    x : ndarray of shape (n, D)
        Matrix of per-document vectors. In the standard SSD pipeline each
        row is a **personal concept vector (PCV)**; in the lexical-norm mode
        each row is a single word embedding.
    y : ndarray of shape (n,) or (n, q)
        Outcome values, numeric (continuous case) or object (group case),
        with NaN / empty entries dropped. 2-D for several numeric outcomes
        (:meth:`fit_multipls`); a row is dropped if any outcome is NaN.
    outcome_names : list of str or None
        Names of the ``q`` outcome columns for 2-D ``y`` (DataFrame column
        names as strings, else ``"y1"``, …, ``"yq"``); ``None`` for 1-D ``y``.
    data : DataFrame, dict of arrays, or None
        The ``data`` given to the constructor, one row per row of ``x``.

    Builds document vectors from corpus + lexicon, then fit with a backend:

    >>> emb = Embeddings.load("model.bin")
    >>> corpus = Corpus(texts, lang="pl")
    >>> ssd = SSD(emb, corpus, y, lexicon)
    >>> pls = ssd.fit_pls()
    >>> pls.r2, pls.pvalue
    >>> pls.top_words(20)
    >>>
    >>> reg = ssd.fit_regression()
    >>> reg.r2
    >>>
    >>> groups = ssd.fit_groups()          # y as categorical labels
    >>> groups = ssd.fit_groups(median_split=True)  # median split
    """

    def __init__(
        self,
        embeddings: Embeddings,
        corpus: Corpus,
        y,
        lexicon: Sequence[str] | set[str] | None = None,
        *,
        window: int = 3,
        sif_a: float | None = 1e-3,
        use_full_doc: bool = False,
        data=None,
    ) -> None:
        """Build a **personal concept vector (PCV)** for each document via
        SIF-weighted aggregation of local contexts around lexicon occurrences,
        preparing data for PLS or PCA+OLS fitting.

        Parameters
        ----------
        embeddings : Embeddings
            Word embeddings instance providing the vector space.
        corpus : Corpus
            Tokenized corpus (``Corpus`` instance) aligned with ``y``.
        y : array-like of shape (n,) or (n, q), DataFrame, str or list of str
            Outcome variable: 1-D numeric or categorical labels, or a 2-D
            numeric matrix / DataFrame of several outcomes. Entries (rows,
            for 2-D) with NaN are silently dropped together with the
            corresponding documents. With ``data``, a column name, or a list
            of column names for several outcomes.
        lexicon : sequence or set of str, optional
            Seed words used for context-window extraction. Required in the
            default seed-context mode; ignored when ``use_full_doc=True``.
        window : int, default 3
            Context window size (tokens) around each seed word.
        sif_a : float or None, default 1e-3
            SIF smoothing parameter for document-vector weighting. ``None``
            turns SIF off: every word weighs 1 (plain mean) and no word
            counts are computed.
        use_full_doc : bool, default False
            If True, use full-document vectors instead of seed-windowed
            contexts.
        data : DataFrame or dict of 1-D arrays, optional
            One row per corpus document: the outcome columns named by ``y``
            and the covariates and grouping factors named in
            :meth:`fit_regression` formulas. Kept as ``self.data``, restricted
            to the documents that stay in the fit.

        Raises
        ------
        ValueError
            If ``len(y) != len(corpus)`` or ``len(data) != len(corpus)``, if
            2-D ``y`` is not numeric, if ``y`` names a column without
            ``data``, or if ``lexicon`` is empty and ``use_full_doc=False``.
        """
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if sif_a is not None and sif_a <= 0:
            raise ValueError(f"sif_a must be > 0, got {sif_a}")

        if not embeddings.l2_normalized:
            raise RuntimeError(
                "SSD requires L2-normalised embeddings. Call "
                "emb.normalize(l2=True, abtt=N) before constructing SSD."
            )

        _check_tokenizer(embeddings, corpus)

        if getattr(embeddings, "_pack", None) is not None:
            # A wordpack always gets a view, so a later emb.normalize() leaves
            # this fit on the rows it was made with. A partial pack also loads
            # the corpus tail into the view, never into the caller's embedding,
            # so one pack stays reusable across corpora. A view already
            # attached to another corpus gets one more view, so rows of this
            # corpus are read in one batch, not on every lookup.
            embeddings = embeddings.with_corpus(corpus)

        self.embeddings = embeddings
        self.corpus = corpus
        self.lexicon = set(lexicon) if lexicon is not None else set()
        if not use_full_doc and not self.lexicon:
            raise ValueError(
                "lexicon is required when use_full_doc=False "
                "(seed-context mode needs seed words)."
            )
        self.window = window
        self.sif_a = sif_a
        self.lang = getattr(corpus, "lang", None)

        # Build doc vectors
        docs = corpus.docs

        if data is not None:
            lengths = (
                {len(data)} if hasattr(data, "iloc")
                else {len(np.asarray(v)) for v in data.values()}
            )
            if lengths != {len(docs)}:
                shown = lengths.pop() if len(lengths) == 1 else sorted(lengths)
                raise ValueError(
                    f"len(data)={shown} != len(docs)={len(docs)}. "
                    "data must have one row per corpus document."
                )

        # The column name y was given as; fit_regression accepts it as a formula's left side.
        self._y_name = y if isinstance(y, str) else None
        outcome_names = None
        # A list of strings without data stays what it was: categorical labels.
        is_list_of_names = (
            data is not None and isinstance(y, list) and y
            and all(isinstance(c, str) for c in y)
        )
        if isinstance(y, str) or is_list_of_names:
            if data is None:
                raise ValueError(
                    f"y={y!r} names a column, but no data was given. "
                    f"Pass data=df, or pass the values, e.g. y=df[{y!r}]."
                )
            names = [y] if isinstance(y, str) else list(y)
            missing = [c for c in names if c not in data]
            if missing:
                raise ValueError(f"y names columns {missing} that data does not have.")
            if isinstance(y, str):
                y = data[y]
            else:
                outcome_names = names
                y = np.column_stack([np.asarray(data[c]) for c in names])
        elif hasattr(y, "columns"):
            outcome_names = [str(c) for c in y.columns]

        # Accept both numeric and categorical y
        y_raw = np.asarray(y)
        try:
            y_num = np.asarray(y, dtype=float)
            is_numeric = True
        except (ValueError, TypeError):
            y_num = None
            is_numeric = False

        if y_raw.ndim == 2:
            if not is_numeric:
                raise ValueError("2-D y (several outcomes) must be numeric.")
            if outcome_names is None:
                outcome_names = [f"y{j + 1}" for j in range(y_raw.shape[1])]

        if len(y_raw) != len(docs):
            raise ValueError(
                f"len(y)={len(y_raw)} != len(docs)={len(docs)}. "
                "y and corpus must have the same length."
            )

        # Filter invalid y entries; `valid` has one entry per corpus document.
        n_corpus = len(docs)
        if is_numeric:
            finite = np.isfinite(y_num)
            if finite.ndim == 2:
                finite = finite.all(axis=1)
            valid = finite
            if not finite.all():
                docs = [d for d, m in zip(docs, finite) if m]
                y_num = y_num[finite]
            y_clean = y_num
        else:
            # For categorical: drop None, "", NaN
            valid = _categorical_mask(y_raw)
            if not valid.all():
                docs = [d for d, v in zip(docs, valid) if v]
                y_raw = y_raw[valid]
            y_clean = y_raw

        # Kept for predict(): new documents are weighted with these counts.
        self._sif_counts = compute_global_sif(docs) if sif_a is not None else None
        self.use_full_doc = use_full_doc
        X, keep = build_and_normalize_doc_vectors(
            docs, embeddings, self.lexicon,
            window=window, sif_a=sif_a, use_full_doc=use_full_doc,
            sif_counts=self._sif_counts,
        )

        # One entry per corpus document: False for an invalid y or no vector.
        keep_full = np.zeros(n_corpus, dtype=bool)
        keep_full[np.flatnonzero(valid)[keep]] = True
        self._keep_mask = keep_full
        self.n_raw = n_corpus
        self.n_kept = int(keep.sum())
        self.n_dropped = self.n_raw - self.n_kept

        self.x = np.asarray(X, dtype=np.float64)
        self.y = y_clean[keep]
        self.is_numeric = is_numeric
        self.outcome_names = outcome_names

        # Rows of data line up with rows of x. A DataFrame stays a DataFrame
        # so categorical columns keep their dtype and level order for glmm.
        if data is not None:
            idx = np.flatnonzero(keep_full)
            if hasattr(data, "iloc"):
                data = data.iloc[idx].reset_index(drop=True)
            else:
                data = {k: np.asarray(v)[idx] for k, v in data.items()}
        self.data = data

    # ── Shared helpers ─────────────────────────────────────────

    def _require_1d_y(self, method: str) -> None:
        if self.y.ndim == 2 and self.y.shape[1] == 1:
            raise ValueError(
                f"{method}() takes a 1-D y, but y has shape {self.y.shape}. "
                "Pass a single column as 1-D, e.g. df['col']."
            )
        if self.y.ndim == 2:
            raise ValueError(
                f"{method}() takes one outcome, but y has "
                f"{self.y.shape[1]} columns. Use fit_multipls() for several outcomes."
            )

    def _compute_fit_stats(
        self, ys: np.ndarray, y_pred: np.ndarray, p: int,
    ) -> dict:
        """Compute R², R²_adj, and F p-value from predictions."""
        n = len(ys)
        resid = ys - y_pred
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))

        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r2_adj = (
            1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
            if n - p - 1 > 0 else float("nan")
        )

        msr = (ss_tot - ss_res) / max(p, 1)
        mse = ss_res / (n - p - 1) if n - p - 1 > 0 else float("inf")
        f_stat_val = msr / mse if np.isfinite(mse) and mse > 0 else 0.0
        f_pvalue = (
            f_sf(f_stat_val, p, n - p - 1)
            if np.isfinite(mse) and n - p - 1 > 0
            else float("nan")
        )

        return {
            "r2": r2,
            "r2_adj": r2_adj,
            "f_pvalue": f_pvalue,
        }

    def _formula_rhs(self, formula: str | None, method: str) -> str:
        """The right side of *formula* (``"1"`` for None), after the checks every formula fit shares."""
        rhs = "1"
        if formula is not None:
            if self.data is None:
                raise ValueError(
                    f"{method}() got a formula but this SSD has no data. "
                    "Construct it with SSD(..., data=df)."
                )
            lhs, _, rhs = formula.rpartition("~")
            lhs, rhs = lhs.strip(), rhs.strip() or "1"
            if lhs not in ("", "y", self._y_name):
                raise ValueError(
                    f"The formula's left side is {lhs!r}, but the outcome is y"
                    + (f" ({self._y_name!r})" if self._y_name else "")
                    + ". Give the right side only, e.g. '~ age + gender'."
                )
        if self.data is not None:
            columns = self.data.columns if hasattr(self.data, "iloc") else self.data.keys()
            clash = sorted(str(c) for c in columns if str(c).startswith("_ssd_"))
            if clash:
                raise ValueError(
                    f"data columns {clash} start with '_ssd_', which {method}() "
                    "reserves for its own columns. Rename them."
                )
        return rhs

    def _data_with(self, **cols):
        """``self.data`` with *cols* added; only *cols* when there is no data."""
        if hasattr(self.data, "iloc"):
            return self.data.assign(**cols)
        if self.data is not None:
            return {**self.data, **cols}
        return cols

    def _check_nobs(self, fit) -> None:
        """Raise if glmm dropped kept documents for missing values in the formula's columns."""
        if fit.nobs != self.n_kept:
            raise ValueError(
                f"The formula's columns have missing values in "
                f"{self.n_kept - fit.nobs} of the {self.n_kept} kept documents. "
                "Fill or drop those rows before building the SSD."
            )

    def _residualize(self, rhs: str) -> tuple[np.ndarray, np.ndarray]:
        """The document vectors and y with the formula's fixed effects regressed out by OLS.

        One glmm fit per column of x: glmm builds the design from the formula
        (factors, interactions) but exposes no design matrix to project with.
        """
        import glmm

        # Every random-effect term, (1 | g) or (x || g), has a bar.
        if "|" in rhs:
            raise ValueError(
                "fit_pls() takes fixed covariates only, but the formula has random "
                "effects. Use fit_regression() for random effects."
            )
        fit = glmm.fit(self._data_with(_ssd_y=self.y), f"_ssd_y ~ {rhs}")
        self._check_nobs(fit)
        y_res = np.asarray(fit.residuals(), dtype=float)
        x_res = np.column_stack([
            np.asarray(
                glmm.fit(self._data_with(_ssd_x=self.x[:, j]), f"_ssd_x ~ {rhs}").residuals(),
                dtype=float,
            )
            for j in range(self.x.shape[1])
        ])
        return x_res, y_res

    def _orient_beta(
        self, beta: np.ndarray, ys: np.ndarray, x: np.ndarray | None = None,
    ) -> np.ndarray:
        """Orient beta so higher alignment → higher outcome (alignment on *x*, default ``self.x``)."""
        yhat_std = ((self.x if x is None else x) @ beta).ravel()
        if float(np.std(yhat_std)) > 0:
            c = float(np.corrcoef(ys, yhat_std)[0, 1])
            corr = c if np.isfinite(c) else 0.0
        else:
            corr = 0.0
        if corr < 0:
            beta = -beta
        return beta

    def _base_result_kwargs(self) -> dict:
        """Common kwargs for result object construction.

        Does not include _y_mean / _y_scale — callers add those
        after computing standardization in their fit method.
        """
        return {
            "embeddings": self.embeddings,
            "corpus": self.corpus,
            "lexicon": self.lexicon,
            "window": self.window,
            "sif_a": self.sif_a,
            "lang": self.lang,
            "x": self.x,
            "keep_mask": self._keep_mask,
            "n_raw": self.n_raw,
            "n_kept": self.n_kept,
            "n_dropped": self.n_dropped,
            "y": self.y,
            "sif_counts": self._sif_counts,
            "use_full_doc": self.use_full_doc,
        }

    # ── PLS backend ────────────────────────────────────────────

    def fit_pls(
        self,
        formula: str | None = None,
        *,
        k: int | Literal["auto"] = "auto",
        k_max: int = 5,
        ssd_gradients: int = 1,
        rotate: Literal["words"] | None = "words",
        n_splits: int = 50,
        random_state: int = 2137,
        test: bool = True,
        verbose: bool = False,
    ):
        """Fit PLS1 NIPALS and return PLSResult.

        The standard fit (k selection, the test, ``beta``, ``r2``, ``docs``,
        ``predict()``) is the same for every ``ssd_gradients``. With
        ``ssd_gradients=N > 1`` the result also holds N rotated gradients,
        ``result["dim-1"]`` … ``result["dim-N"]``, from a separate N-component
        PLS1 fit on the same standardized data.

        Significance is always tested with the split_exact method
        (permutation-calibrated split-half test; Lenartowicz, 2026). The
        ``find_k_optimal`` diagnostic chain stays on the cheaper split_nb.
        For ``k="auto"`` the reported p-value is the **same sample k=1
        confirmatory** test — independent of the ``find_k_optimal``
        selection that picks ``k_star`` for the actual fit.

        With a formula, y and every column of the document vectors are first
        regressed on the formula's fixed effects by OLS, and PLS runs on the
        residuals (Frisch–Waugh–Lovell). The test permutes the residualized
        y against the residualized vectors (Kennedy, 1995); it can run
        slightly liberal when the covariates use many degrees of freedom
        relative to n.

        Parameters
        ----------
        formula : str or None
            Fixed covariates over columns of ``data``, e.g.
            ``"~ age + gender"``; a left side, if given, must name y. Random
            effects such as ``(1 | ID)`` raise ``ValueError``: use
            :meth:`fit_regression` for them. With a formula, ``r2`` is the R²
            of the residualized y, ``docs.residual`` has the covariates
            taken out, and ``predict()`` is not available. The covariates'
            own coefficients are not reported: use :meth:`fit_regression`
            for them.
        k : int or ``"auto"``, default ``"auto"``
            Number of PLS components.

            - ``int`` — fit at exactly this k; p-value comes from
              ``plskit.pls1_confirmatory_test`` at this k.
            - ``"auto"`` — let ``plskit.pls1_find_k_optimal`` pick
              ``k_star`` (selector ``"r2_se"``); p-value is the same sample
              k=1 confirmatory split_exact statistic.
        k_max : int, default 5
            Cap for ``k="auto"``. Further clamped to ``min(k_max, n-1, D)``.
            Ignored when ``k`` is an int.
        ssd_gradients : int, default 1
            Number of gradients returned, in ``[1, min(n - 1, D)]``. At 1
            the result is the standard fit alone and ``rotate`` is ignored.
            At N > 1, ``plskit.pls1_fit(k=N)`` gives N orthonormal weight
            vectors W; each leaf ``"dim-i"`` has ``beta = W_rot[:, i] /
            X_scale``, its ``stats`` (``score_corr``, ``cov_share``),
            ``words``, ``clusters`` and ``snippets``, and no p-value.
            ``fit_info.n_components`` stays the standard fit's k.
        rotate : {"words", None}, default "words"
            Rotation of the N gradients.

            - ``"words"`` — varimax against the projection of the leading
              vocabulary rows onto W, then dimensions ordered by
              ``|corr(score, y)|``, descending, each flipped so that
              correlation is positive.
            - ``None`` — W in NIPALS order and signs. Dimensions 2…N then
              have zero covariance with y (NIPALS gives ``w_1 ∝ X'y``), so
              their ``score_corr`` and ``cov_share`` are 0.
        n_splits : int, default 50
            Random splits for the split-half tests.
        random_state : int, default 2137
            Random seed.
        test : bool, default True
            ``False`` skips the confirmatory test: ``pvalue`` is ``NaN`` and
            ``result.test`` is left not run. Choosing k for ``k="auto"``
            still runs. ``result.test(n_splits=..., seed=...)`` runs the
            skipped test later; with no arguments it uses 50 splits and no
            seed, not this fit's ``n_splits`` and ``random_state``.
        verbose : bool

        Returns
        -------
        PLSResult

        Raises
        ------
        ValueError
            On 2-D or categorical y, constant y, an invalid ``k`` or
            ``ssd_gradients``, a ``rotate`` other than ``"words"`` or
            ``None`` with ``ssd_gradients > 1``, or when plskit returns
            fewer than ``ssd_gradients`` weight vectors.
        RuntimeError
            ``rotate="words"`` with ``ssd_gradients > 1`` and no embeddings
            attached.
        """
        fit_args = {"formula": formula, "k": k, "k_max": k_max,
                    "ssd_gradients": ssd_gradients, "rotate": rotate,
                    "n_splits": n_splits, "random_state": random_state, "test": test}
        import plskit

        from ssdiff.backends.pls import pls1_rotate, word_target
        from ssdiff.results.continuous_result import PLSResult

        self._require_1d_y("fit_pls")
        if not self.is_numeric:
            raise ValueError(
                "fit_pls() requires numeric y. This SSD was constructed with "
                "categorical labels — use fit_groups() instead."
            )
        n_kept, D = self.x.shape
        g_max = min(n_kept - 1, D)
        if not isinstance(ssd_gradients, int) or not 1 <= ssd_gradients <= g_max:
            raise ValueError(
                f"ssd_gradients must be an int in [1, {g_max}] (min(n - 1, D) = "
                f"min({n_kept - 1}, {D})); got {ssd_gradients!r}"
            )
        if ssd_gradients > 1:
            if rotate not in (None, "words"):
                raise ValueError(
                    f"fit_pls() rotate must be 'words' or None; got {rotate!r}. "
                    "One outcome gives no outcome loadings to rotate against."
                )
            if rotate == "words" and self.embeddings is None:
                raise RuntimeError(
                    "fit_pls(rotate='words') requires Embeddings (the rotation "
                    "target is the vocabulary); none attached. Re-construct SSD "
                    "with embeddings, or pass rotate=None."
                )

        y_std = float(np.std(self.y))
        if y_std <= 1e-12:
            raise ValueError(
                f"fit_pls() requires y with nonzero variance (std={y_std:.3g}). "
                "PLS1 is undefined when y is constant."
            )

        x_fit, y_fit = self.x, self.y
        if formula is not None:
            rhs = self._formula_rhs(formula, "fit_pls")
            x_fit, y_fit = self._residualize(rhs)
            if float(np.std(y_fit)) <= 1e-8 * y_std:
                raise ValueError(
                    "The formula's covariates explain y fully, so no variance "
                    "is left for the text."
                )

        ys_2d, _y_mean, _y_scale = standardize(y_fit.reshape(-1, 1))
        ys = ys_2d.ravel()
        Xs, X_mean, X_scale = standardize(x_fit)
        if formula is not None:
            # Residuals have mean 0. docs.y_hat centers on y's own mean and on
            # the raw vectors' mean; _y_scale stays the residual sd, the unit
            # beta was fit in.
            _y_mean, X_mean = np.atleast_1d(np.mean(self.y)), self.x.mean(axis=0)

        find_k_raw = None
        test_name = "split_exact"
        pvalue = float("nan")
        if k == "auto":
            eff_k_max = max(min(int(k_max), n_kept - 1, D), 1)
            if verbose:
                print(
                    f"[fit_pls] auto-selecting k via find_k_optimal "
                    f"(selector=r2_se, k_max={eff_k_max})"
                )
            find_k_raw = plskit.pls1_find_k_optimal(
                Xs, ys, eff_k_max,
                selector="r2_se",
                diagnostic="split_nb",
                args={"n_folds": 5, "n_splits": int(n_splits)},
                pre_standardized=True,
                seed=random_state,
                verbose=verbose,
            )
            k_star = int(find_k_raw.k_star)
            if k_star < 1:
                raise ValueError(
                    "fit_pls(k='auto'): plskit found no component "
                    "(k_star = 0); y shows no signal in X."
                )
            if verbose:
                print(f"[fit_pls] k_star = {k_star}")
                if k_star > 1:
                    chain = ", ".join(
                        f"{p:.3g}" for p in find_k_raw.pvalues[:k_star]
                    )
                    path_max = float(np.nanmax(find_k_raw.pvalues[:k_star]))
                    print(
                        f"[fit_pls] diagnostic chain (split_nb, k=1..{k_star}): "
                        f"{chain}"
                    )
                    print(
                        f"        path-max: {path_max:.3g}  "
                        f"[same-sample]"
                    )
            if test:
                if verbose:
                    print(
                        f"[fit_pls] running k=1 confirmatory test "
                        f"(split_exact, n_splits={n_splits})"
                    )
                r = plskit.pls1_confirmatory_test(
                    Xs, ys, 1,
                    test_method="split_exact",
                    args={"n_splits": int(n_splits)},
                    pre_standardized=True,
                    seed=random_state,
                    verbose=verbose,
                )
                pvalue = float(r.pvalue)
                if verbose:
                    print(f"[fit_pls] p-value (k=1) = {pvalue:.6g}")
            n_comp = k_star
            p_at_k = 1
        elif isinstance(k, int):
            if k < 1:
                raise ValueError(f"k must be >= 1, got {k}")
            n_comp = min(int(k), n_kept - 1, D)
            if n_comp < 1:
                raise ValueError(
                    f"Cannot fit PLS: need at least 2 samples and 1 feature, "
                    f"got n={n_kept}, features={D}, requested k={k}"
                )
            if test:
                r = plskit.pls1_confirmatory_test(
                    Xs, ys, n_comp,
                    test_method="split_exact",
                    args={"n_splits": int(n_splits)},
                    pre_standardized=True,
                    seed=random_state,
                    verbose=verbose,
                )
                pvalue = float(r.pvalue)
            p_at_k = n_comp
        else:
            raise ValueError(f"k must be int or 'auto', got {k!r}")

        m = plskit.pls1_fit(
            Xs, ys, k=n_comp, pre_standardized=True, seed=random_state,
        )
        T, P, W, Q, coef = m.T, m.P, m.W, m.Q, m.coef

        y_pred = Xs @ coef
        stats = self._compute_fit_stats(ys, y_pred, W.shape[1])

        scale = np.where(X_scale > 1e-12, X_scale, 1.0)
        beta = coef / scale
        beta = self._orient_beta(beta, ys, x_fit)

        dims = None
        if ssd_gradients > 1:
            # NIPALS weights don't depend on how many components are fitted,
            # so W's leading columns are reused when the main fit has enough.
            W_g = (W[:, :ssd_gradients] if ssd_gradients <= W.shape[1] else
                   plskit.pls1_fit(Xs, ys, k=ssd_gradients, pre_standardized=True,
                                   seed=random_state).W)
            if W_g.shape[1] < ssd_gradients:
                raise ValueError(
                    f"plskit.pls1_fit returned {W_g.shape[1]} weight vectors but "
                    f"ssd_gradients={ssd_gradients} was requested (NIPALS deflation "
                    f"collapsed). Reduce ssd_gradients."
                )
            L = (word_target(self.embeddings.vectors, W_g, X_mean, X_scale)
                 if rotate == "words" else None)
            dims = pls1_rotate(W_g, Xs, ys, rotate=rotate, L=L)
            dims["beta"] = dims["W_rot"] / X_scale[:, None]
            dims["rotate"] = rotate

        kwargs = self._base_result_kwargs()
        kwargs["_y_mean"] = _y_mean
        kwargs["_y_scale"] = _y_scale
        kwargs["_x_mean"] = X_mean

        raw_diagnostics = {
            "find_k_result": find_k_raw,
            "component_scores": T,
            "component_weights": W,
        }
        if formula is not None:
            # .test() reruns on these, not on the raw x and y.
            raw_diagnostics["x_resid"] = x_fit
            raw_diagnostics["y_resid"] = y_fit
            # The PLS fit's own residual on the residualized y, so the
            # covariates are taken out of docs.residual too. No glmm fit of
            # the covariates with the text score: beta was fitted to the same
            # y, so that fit's SEs and p-values come out too small at D ≈ n.
            kwargs["residuals"] = (ys - y_pred) * float(_y_scale[0])

        if test:
            test_info = {
                "pvalue": pvalue,
                "n_splits": int(n_splits),
                "random_state": random_state,
            }
        else:
            test_name, test_info = None, None

        result = PLSResult(
            beta=beta,
            pvalue=pvalue,
            r2=stats["r2"],
            fit_info={
                "n_components": n_comp,
                "p_at_k": p_at_k,
                "n_splits": int(n_splits),
                "random_state": random_state,
                "formula": formula,
                "ssd_gradients": ssd_gradients,
                "rotate": rotate if ssd_gradients > 1 else None,
            },
            raw_diagnostics=raw_diagnostics,
            test_name=test_name,
            test_info=test_info,
            dims=dims,
            **kwargs,
        )
        result._y_name = self._y_name
        result._text_name = getattr(self.corpus, "name", None)
        result.fit_args = fit_args
        return result

    # ── PLS3 backend (several outcomes) ────────────────────────

    def fit_multipls(
        self,
        *,
        ssd_gradients: int | None = None,
        rotate: Literal["words", "outcomes"] | None = None,
        outcome_groups: dict[str, list[str]] | None = None,
        n_splits: int = 50,
        n_perm: int = 1000,
        random_state: int = 2137,
        verbose: bool = False,
    ):
        """Fit PLS3 (PLSC) of the doc vectors against several outcomes.

        PLS3 takes the SVD of the standardized cross-covariance ``Xs' Ys``:
        dimension i pairs a word-space direction ``U[:, i]`` with an
        outcome-loading vector ``V[:, i]``. Needs 2-D ``y`` with at least
        two columns; use :meth:`fit_pls` for one outcome.

        Significance is the split_exact PLS3 test at LV1
        (``plskit.pls3_confirmatory_test``): a container-level p-value that
        does not depend on ``rotate``. Leaves carry no p-value.

        Parameters
        ----------
        ssd_gradients : int or None, default None
            Number of dimensions, in ``[1, min(q, n - 1, D)]``. ``None``
            means ``q``, or with ``outcome_groups`` the number of rotation
            target rows. At ``q`` every component is kept and
            ``cov_explained`` is 1 by construction.
        rotate : {None, "words", "outcomes"}, default None
            One orthogonal ``R`` applied to both sides (``U_rot = U @ R``,
            ``V_rot = V @ R``).

            - ``None`` — SVD dimensions as they are.
            - ``"words"`` — varimax against the projection of the leading
              vocabulary rows onto ``U``.
            - ``"outcomes"`` — varimax against the outcome loadings ``V``
              (or the group rows from ``outcome_groups``). At
              ``ssd_gradients = q`` without groups, ``R = V'`` directly:
              each dimension loads 1 on one outcome.

            Rotated modes reorder dimensions by squared cross-covariance.
            Every mode flips signs so the largest-magnitude outcome loading
            of each dimension is positive.
        outcome_groups : dict of str to list of str, optional
            Only with ``rotate="outcomes"``. Each group becomes one rotation
            target row, the mean of its outcomes' loadings; outcomes in no
            group keep their own row. Names refer to ``outcome_names``.
            ``ssd_gradients`` must not exceed the number of target rows.
        n_splits : int, default 50
            Random splits for the split_exact test.
        n_perm : int, default 1000
            Permutations for the split_exact test.
        random_state : int, default 2137
            Random seed for the test.
        verbose : bool

        Returns
        -------
        MultiPLSResult

        Raises
        ------
        ValueError
            On 1-D ``y``, fewer than two outcomes, a constant outcome, an
            invalid ``rotate`` / ``outcome_groups`` / ``ssd_gradients``, or
            when plskit keeps fewer than ``ssd_gradients`` components.
        RuntimeError
            If no embeddings are attached.
        """
        fit_args = {"ssd_gradients": ssd_gradients, "rotate": rotate,
                    "outcome_groups": outcome_groups, "n_splits": n_splits,
                    "n_perm": n_perm, "random_state": random_state}
        import plskit

        from ssdiff.backends.pls import pls3_rotate, word_target
        from ssdiff.results.multi_pls_result import MultiPLSResult

        if self.y.ndim != 2 or self.y.shape[1] < 2:
            raise ValueError(
                "fit_multipls() needs several outcomes (2-D y with at least "
                "two columns). For one outcome pass a 1-D y and use fit_pls()."
            )
        names = self.outcome_names
        n_kept, D = self.x.shape
        q = self.y.shape[1]

        y_std = self.y.std(axis=0)
        constant = [name for name, s in zip(names, y_std) if s <= 1e-12]
        if constant:
            raise ValueError(
                f"fit_multipls() requires outcomes with nonzero variance; "
                f"constant: {constant}."
            )
        if self.embeddings is None:
            raise RuntimeError(
                "fit_multipls() requires Embeddings (the per-dimension views "
                "and rotate='words' use the vocabulary); none attached. "
                "Re-construct SSD with embeddings."
            )
        if rotate not in (None, "words", "outcomes"):
            raise ValueError(
                f"rotate must be None, 'words' or 'outcomes'; got {rotate!r}"
            )

        grouped: dict[str, str] = {}
        if outcome_groups is not None:
            if rotate != "outcomes":
                raise ValueError(
                    f"outcome_groups requires rotate='outcomes'; got rotate={rotate!r}"
                )
            for g, members in outcome_groups.items():
                if len(members) == 0:
                    raise ValueError(f"outcome_groups[{g!r}] is empty.")
                for name in members:
                    if name not in names:
                        raise ValueError(
                            f"outcome_groups[{g!r}]: unknown outcome {name!r}; "
                            f"outcomes are {names}."
                        )
                    if name in grouped:
                        raise ValueError(
                            f"outcome {name!r} is in two groups: "
                            f"{grouped[name]!r} and {g!r}."
                        )
                    grouped[name] = g

        # One rotation target row per group plus one per ungrouped outcome.
        n_rows = len(outcome_groups) + q - len(grouped) if outcome_groups else q
        default_note = (" (the default: one per outcome, or per group with "
                        "outcome_groups)" if ssd_gradients is None else "")
        if ssd_gradients is None:
            ssd_gradients = n_rows
        k_max = min(q, n_kept - 1, D)
        if not isinstance(ssd_gradients, int) or not 1 <= ssd_gradients <= k_max:
            raise ValueError(
                f"ssd_gradients must be an int in [1, {k_max}] (min(q, n - 1, D) = "
                f"min({q}, {n_kept - 1}, {D})); got {ssd_gradients!r}{default_note}"
            )
        if ssd_gradients > n_rows:
            raise ValueError(
                f"rotate='outcomes' needs ssd_gradients <= the number of rotation "
                f"target rows ({n_rows}: one per group plus one per ungrouped "
                f"outcome); got ssd_gradients={ssd_gradients}"
            )

        Xs, X_mean, X_scale = standardize(self.x)
        Ys, _, _ = standardize(self.y)
        try:
            m = plskit.pls3_fit(
                Xs, Ys, k=ssd_gradients, pre_standardized_X=True, pre_standardized_Y=True,
            )
        except plskit.PlsKitError as exc:
            raise ValueError(
                f"plskit.pls3_fit rejected ssd_gradients={ssd_gradients}: {exc}"
            ) from exc
        if m.k_used < ssd_gradients:
            raise ValueError(
                f"plskit.pls3_fit kept k_used={m.k_used} components of "
                f"ssd_gradients={ssd_gradients}{default_note}; the rest fall below its truncation floor (e.g. collinear "
                f"outcomes). Refit with ssd_gradients <= {m.k_used}."
            )
        U, V, sigma = m.U, m.V, m.singular_values  # (D, k), (q, k), (k,)

        # rotate="outcomes" without groups leaves L=None: pls3_rotate targets V itself.
        L = None
        if rotate == "words":
            L = word_target(self.embeddings.vectors, U, X_mean, X_scale)
        elif rotate == "outcomes" and outcome_groups is not None:
            col = {name: j for j, name in enumerate(names)}
            L = np.vstack(
                [V[[col[n] for n in members]].mean(axis=0)
                 for members in outcome_groups.values()]
                + [V[j] for j, name in enumerate(names) if name not in grouped]
            )

        rot = pls3_rotate(U, V, sigma, rotate=rotate, L=L)
        U_rot, V_rot = rot["U_rot"], rot["V_rot"]

        cov_explained = float(np.sum(sigma ** 2) / np.linalg.norm(Xs.T @ Ys) ** 2)

        if verbose:
            print(
                f"[fit_multipls] running LV1 test "
                f"(split_exact, n_splits={n_splits}, n_perm={n_perm})"
            )
        r = plskit.pls3_confirmatory_test(
            Xs, Ys, 1,
            test_method="split_exact",
            args={"n_perm": int(n_perm), "n_splits": int(n_splits)},
            seed=random_state,
            verbose=verbose,
        )
        test_info = {
            "pvalue": float(r.pvalue),
            "statistic": float(r.statistic),
            "n_splits": int(n_splits),
            "n_perm": int(n_perm),
            "random_state": random_state,
        }

        result = MultiPLSResult(
            x=self.x, y=self.y,
            U=U, V=V, U_rot=U_rot, V_rot=V_rot,
            singular_values=sigma,
            x_scores=Xs @ U_rot, y_scores=Ys @ V_rot,
            R=rot["R"], order=rot["order"], signs=rot["signs"],
            rotation_meta=rot["rotation_meta"],
            rotate=rotate,
            cov_explained=cov_explained,
            outcome_names=names,
            outcome_groups=outcome_groups,
            X_scale=X_scale,
            test_name="split_exact",
            test_info=test_info,
            embeddings=self.embeddings,
            corpus=self.corpus,
            lexicon=self.lexicon,
            window=self.window,
            sif_a=self.sif_a,
            lang=self.lang,
            random_state=random_state,
        )
        result.fit_args = fit_args
        # Corpus row of each row of ``x``: the results page links documents back to their text.
        result.doc_ids = np.flatnonzero(self._keep_mask)
        return result

    # ── PCA regression backend ────────────────────────────────

    def fit_ols(self, **kwargs):
        """Old name of :meth:`fit_regression`; takes the same keyword arguments."""
        warnings.warn(
            "fit_ols() is renamed fit_regression() and will be removed; "
            "call ssd.fit_regression(...) with the same arguments.",
            FutureWarning, stacklevel=2,
        )
        return self.fit_regression(**kwargs)

    def fit_regression(
        self,
        formula: str | None = None,
        *,
        family: str = "gaussian",
        link: str | None = None,
        fixed_k: int | None = None,
        k_min: int = 2,
        k_max: int = 120,
        k_step: int = 2,
        verbose: bool = False,
        pca_sweep_args: dict | None = None,
    ):
        """Regress y on the first K PCA components of the document vectors.

        K is chosen by the PCA-K sweep (OLS on standardized y), unless
        ``fixed_k`` is given. The final model at that K is fit by ``glmm``:
        the K component scores are added to ``formula``, so covariates and
        random effects (``"~ age + (1 | ID)"``) are controlled for. The
        p-value tests the K components together: the model with the formula
        alone against the model with the formula and the components, both
        fit by ML on the same rows. An OLS model (Gaussian, no random
        effects) gets the F-test, and so do Gamma and inverse-Gaussian GLMs
        (Wald F); every other model gets the likelihood-ratio test.

        Parameters
        ----------
        formula : str or None
            Right-hand side over columns of ``data``, e.g.
            ``"~ age + gender + (1 | ID)"``. A left-hand side is allowed only
            as ``y`` or the column name ``y`` was given as. ``None`` fits
            intercept plus components.
        family : str, default "gaussian"
            Any ``glmm`` family, e.g. ``"binomial"`` for a 0/1 y.
        link : str or None
            Link function; ``None`` takes glmm's default for the family.
        fixed_k : int or None
            Fixed number of PCA components. None = auto-select via sweep.
        k_min, k_max, k_step : int
            Range for PCA-K sweep when fixed_k is None.
        verbose : bool
            Print progress.
        pca_sweep_args : dict, optional
            Extra keyword arguments forwarded to
            :func:`ssdiff.backends.pca_sweep.pca_sweep` when ``fixed_k is None``.
            Useful for tuning the sweep's clustering / scoring (e.g.
            ``cluster_topn``, ``cluster_k_min``, ``cluster_k_max``,
            ``weight_by_size``, ``auck_radius``) or for enabling table output
            (``save_tables``, ``out_dir``, ``prefix``).
            Keys that ``fit_regression`` derives internally (``Xs``,
            ``X_scale``, ``x``, ``ys``, ``embeddings``, ``pca_k_values``,
            ``verbose``, ``lang``) cannot be overridden and raise
            ``ValueError`` if passed. Ignored when ``fixed_k`` is set.

        Returns
        -------
        PCAOLSResult
            ``.model`` is the glmm fit with the components: covariate
            coefficients, random effects, ``.summary()``. ``.covariates``
            lists its covariate rows, with the components held fixed.
            ``beta`` is on the
            standardized-y scale for a Gaussian identity-link fit, else on
            the link scale (e.g. log-odds). With a formula, ``r2`` is the
            whole model's R², covariates included; it is NaN for GLMs and
            mixed models.

        Raises
        ------
        ValueError
            If y is categorical or 2-D, if a formula is given without
            ``data``, if its left side names another column, if a ``data``
            column name starts with ``_ssd_``, or if a column the formula
            uses has missing values in kept documents.
        """
        fit_args = {"formula": formula, "family": family, "link": link,
                    "fixed_k": fixed_k, "k_min": k_min, "k_max": k_max, "k_step": k_step,
                    "pca_sweep_args": pca_sweep_args}
        import glmm

        from ssdiff.results.continuous_result import PCAOLSResult

        self._require_1d_y("fit_regression")
        if not self.is_numeric:
            raise ValueError(
                "fit_regression() requires numeric y. This SSD was constructed with "
                "categorical labels — use fit_groups() instead."
            )

        rhs = self._formula_rhs(formula, "fit_regression")

        ys = standardize(self.y.reshape(-1, 1))[0].ravel()

        # Standardize X
        Xs, X_mean, X_scale = standardize(self.x)

        if fixed_k is None:
            from ssdiff.backends.pca_sweep import pca_sweep
            reserved = {
                "Xs", "X_scale", "x", "ys", "embeddings",
                "pca_k_values", "verbose", "lang",
            }
            extra = dict(pca_sweep_args) if pca_sweep_args else {}
            collisions = reserved & extra.keys()
            if collisions:
                raise ValueError(
                    f"pca_sweep_args cannot override fit_regression-managed keys: "
                    f"{sorted(collisions)}. Use the corresponding fit_regression "
                    f"parameter (k_min/k_max/k_step, verbose) instead."
                )
            sweep_result = pca_sweep(
                Xs=Xs,
                X_scale=X_scale,
                x=self.x,
                ys=ys,
                embeddings=self.embeddings,
                pca_k_values=list(range(k_min, k_max + 1, k_step)),
                verbose=verbose,
                lang=self.lang,
                **extra,
            )
            n_pca = sweep_result.best_k
        else:
            n_pca = int(fixed_k)
            sweep_result = None

        max_comp = min(n_pca, Xs.shape[0], Xs.shape[1])

        # PCA via SVD
        z, components, _ = pca_fit_transform(Xs, max_comp)

        pcs = [f"_ssd_pca_{i}" for i in range(1, max_comp + 1)]
        cols = self._data_with(_ssd_y=self.y, **{c: z[:, i] for i, c in enumerate(pcs)})
        h0 = f"_ssd_y ~ {rhs}"
        h1 = h0 + " + " + " + ".join(pcs)
        cmp = glmm.compare(cols, h0, h1, family=family, link=link)
        fit = cmp.fit1
        self._check_nobs(fit)

        p_col = "Pr(>F)" if "Pr(>F)" in cmp.table else "Pr(>Chisq)"
        pvalue = float(cmp.table[p_col][1])
        test_name = "f_test" if p_col == "Pr(>F)" else "lrt"
        f_test = test_name == "f_test"
        test_info = {
            "pvalue": pvalue,
            "statistic": float(cmp.table["F" if f_test else "Chisq"][1]),
            "df": int(cmp.table["Df"][1]),
            "df_resid": int(cmp.table["Res.Df"][1]) if f_test else None,
            # glmm reports r_squared for OLS only; NaN otherwise.
            "r2_delta": float(cmp.fit1.r_squared - cmp.fit0.r_squared),
        }

        names = list(fit.names)
        w_reg = np.array([fit.beta[names.index(c)] for c in pcs])
        # A Gaussian identity-link fit reports w per raw-y unit; dividing by
        # sd(y) puts beta on the standardized-y scale of the other backends.
        # Other models keep the link scale, which has no y unit to divide by.
        linear = fit.family == "gaussian" and fit.link == "identity"
        if linear:
            _y_mean, _y_scale = standardize(self.y.reshape(-1, 1))[1:]
            w_reg = w_reg / _y_scale[0]
        else:
            _y_mean, _y_scale = np.zeros(1), np.ones(1)

        # Back-project to doc space. No sign flip (_orient_beta): beta must
        # keep the sign of the model's coefficients. Without covariates an
        # OLS fit already correlates positively with y.
        scale = np.where(X_scale > 1e-12, X_scale, 1.0)
        beta = (components.T @ w_reg) / scale

        kwargs = self._base_result_kwargs()
        kwargs["_y_mean"] = _y_mean
        kwargs["_y_scale"] = _y_scale
        kwargs["_x_mean"] = X_mean
        kwargs["model"] = fit

        _sweep_rows = None
        if sweep_result is not None:
            _sweep_rows = [
                {**r, "k": r["PCA_K"]} for r in sweep_result.df_joined
            ]

        result = PCAOLSResult(
            backend="PCA+OLS" if formula is None and linear else "PCA+GLMM",
            beta=beta,
            pvalue=pvalue,
            r2=float(fit.r_squared),
            r2_adj=float(fit.adj_r_squared),
            fit_info={
                "n_components": n_pca,
                "k_min": k_min if sweep_result is not None else None,
                "k_max": k_max if sweep_result is not None else None,
                "k_step": k_step if sweep_result is not None else None,
                "best_k": sweep_result.best_k if sweep_result is not None else None,
                "pca_k_source": "sweep" if fixed_k is None else "fixed",
                "formula": formula,
                "family": fit.family,
            },
            raw_diagnostics={
                "sweep_result": sweep_result,
                "pca_components": components,
                "pca_weights": w_reg,
            },
            sweep=_sweep_rows,
            test_name=test_name,
            test_info=test_info,
            **kwargs,
        )
        result._y_name = self._y_name
        result._text_name = getattr(self.corpus, "name", None)
        result.fit_args = fit_args
        return result

    # ── Group comparison backend ──────────────────────────────

    def fit_groups(
        self,
        *,
        median_split: bool = False,
        n_perm: int = 5000,
        correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "holm",
        random_state: int = 2137,
        verbose: bool = False,
    ):
        """Fit group comparison using y as group labels.

        Parameters
        ----------
        median_split : bool, default False
            If True, split y into "low"/"high" at the median.
            If False, treat y values as categorical group labels.
        n_perm : int, default 5000
            Number of permutations for significance tests.
        correction : str, default "holm"
            P-value correction method: "holm", "bonferroni", "fdr_bh",
            or "none".
        random_state : int, default 2137
            Random seed for reproducibility.
        verbose : bool
            Print progress and diagnostics.

        Returns
        -------
        GroupResult

        Raises
        ------
        ValueError
            If fewer than 2 groups remain after filtering.
        """
        from ssdiff.backends.group import (
            filter_small_groups,
            unified_permutation_test,
        )
        from ssdiff.backends.group import (
            median_split as _median_split,
        )
        from ssdiff.results.group_result import GroupResult
        from ssdiff.results.schema import Pair

        self._require_1d_y("fit_groups")
        fit_args = {"median_split": median_split, "n_perm": n_perm,
                    "correction": correction, "random_state": random_state}
        # Corpus row of each row of ``x``: the results page links documents back to their text.
        doc_ids = np.flatnonzero(self._keep_mask)

        # Local copies — never mutate self.x or self.y
        x_local = self.x.copy()

        if median_split:
            groups_local = _median_split(self.y, random_state=random_state)
        else:
            groups_local = np.asarray(self.y, dtype=object)

        # Small-group filtering (only when not median_split)
        n_group_dropped = 0
        if not median_split:
            x_local, groups_local, n_group_dropped = filter_small_groups(
                x_local, groups_local,
            )
            doc_ids = doc_ids[np.isin(np.asarray(self.y, dtype=object), np.unique(groups_local))]

        group_labels = sorted(set(groups_local), key=str)
        G = len(group_labels)
        if G < 2:
            raise ValueError(
                f"Need at least 2 groups after filtering, got {G}"
            )

        # Run unified permutation test
        test_result = unified_permutation_test(
            x_local, groups_local, group_labels,
            n_perm=n_perm,
            correction=correction,
            random_state=random_state,
            verbose=verbose,
        )

        # Build list[Pair] from backend's pairwise dict
        pairs = []
        for (g1, g2), pw in test_result["pairwise"].items():
            pairs.append(Pair(
                contrast=f"{g1}_{g2}",
                g1=str(g1),
                g2=str(g2),
                T=float(pw["T"]),
                p_raw=float(pw["p_raw"]),
                p_corrected=float(pw["p_corrected"]),
                cohens_d=float(pw["cohens_d"]),
                n_g1=int(pw["n_g1"]),
                n_g2=int(pw["n_g2"]),
                contrast_norm=float(pw["contrast_norm"]),
            ))

        # words / clusters / snippets are computed lazily on access by
        # GroupResult, using (embeddings, x, groups) + per-pair gradient.
        result = GroupResult(
            G=G,
            n_kept=len(x_local),
            n_perm=n_perm,
            correction=correction,
            random_state=random_state,
            omnibus_T=float(test_result["omnibus_T"]),
            omnibus_p=float(test_result["omnibus_p"]),
            pairs=pairs,
            embeddings=self.embeddings,
            corpus=self.corpus,
            x=x_local,
            groups=groups_local,
            lang=self.lang,
            lexicon=self.lexicon,
            window=self.window,
            sif_a=self.sif_a,
        )
        result.fit_args = fit_args
        result.doc_ids = doc_ids
        return result

    def __repr__(self) -> str:
        dim = self.x.shape[1] if self.x.ndim == 2 else 0
        header = (
            f"SSD  n={self.n_kept:,}/{self.n_raw:,}  "
            f"D={dim}  |L|={len(self.lexicon)}  lang={self.lang}"
        )
        arrays = "  arrays:  .x  .y"
        methods = "  methods: .fit_pls()  .fit_multipls()  .fit_regression()  .fit_groups()"
        return "\n".join([header, arrays, methods])

    def _repr_html_(self) -> str:
        """Render a plain-text repr inside a ``<pre>`` block for Jupyter."""
        import html as _html
        return f"<pre>{_html.escape(repr(self))}</pre>"
