"""SSD: Supervised Semantic Differential — one text, many fits."""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np

from ssdiff.backends.regression import (
    check_text_terms,
    expand,
    label_of,
    null_formula,
    placeholder,
    term_groups,
    text_labels,
    text_names,
    to_placeholders,
    with_text_term,
)
from ssdiff.embeddings import Embeddings
from ssdiff.results.schema import TermTest
from ssdiff.utils.math import _categorical_mask, f_sf, pca_fit_transform, standardize
from ssdiff.utils.text import build_docs_from_preprocessed, preprocess_texts_wordpack
from ssdiff.utils.vectors import build_and_normalize_doc_vectors, compute_global_sif


def _check_tokenizer(embeddings, text) -> None:
    """Raise when *text* was not tokenized for *embeddings*.

    A wordpack is keyed by surface form, so a text tokenized with one pack
    fits silently on whatever words another embedding happens to share. A
    text of token lists (no recorded tokenizer) is not checked.
    """
    tokenizer = getattr(text, "tokenizer", None)
    if tokenizer is None:
        return
    if getattr(embeddings, "_pack", None) is None:
        raise ValueError(
            f"The text was tokenized with the wordpack {tokenizer[len('wordpack:'):]}, "
            "but these embeddings are not a wordpack. Load that wordpack with "
            "Embeddings.load(...), or build the text from token lists: SSD(emb, token_lists)."
        )
    expected = f"wordpack:{embeddings.name}-{embeddings.version}"
    if tokenizer != expected:
        raise ValueError(
            f"The text was tokenized with {tokenizer}, but the embeddings are "
            f"{expected}. Build the text with SSD(emb, texts)."
        )


def _check_window_sif(window: int, sif_a: float | None) -> None:
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if sif_a is not None and sif_a <= 0:
        raise ValueError(f"sif_a must be > 0, got {sif_a}")


def _outcome_values(
    outcome, data, n: int, method: str, *, multi: bool = False, keep_labels: bool = False,
):
    """Resolve an outcome to ``(values, name, outcome_names)``.

    *outcome* is a column name of *data*, a list of column names (only with
    ``multi=True``), or the values themselves (array, list, Series,
    DataFrame). Values are read by position, never by a pandas index.
    ``values`` is float64 when the outcome converts to numbers, else an object
    array of labels; with ``keep_labels=True`` it is always the labels as
    given (whole-number labels stay ints). ``name`` is the column name, else ``"y"``;
    ``outcome_names`` is set for 2-D outcomes only.
    """
    name, names = "y", None
    is_names = (
        multi and data is not None and isinstance(outcome, list) and len(outcome) > 0
        and all(isinstance(c, str) for c in outcome)
    )
    if isinstance(outcome, str) or is_names:
        if data is None:
            raise ValueError(
                f"{method}() got the column name {outcome!r} but no data. "
                f"Pass data=df, or pass the values, e.g. df[{outcome!r}]."
            )
        cols = [outcome] if isinstance(outcome, str) else list(outcome)
        missing = [c for c in cols if c not in data]
        if missing:
            raise ValueError(f"{method}(): data has no column {missing}.")
        if isinstance(outcome, str):
            values, name = data[outcome], outcome
        else:
            values, names = np.column_stack([np.asarray(data[c]) for c in cols]), cols
    else:
        values = outcome
        if hasattr(outcome, "columns"):
            names = [str(c) for c in outcome.columns]
        elif getattr(outcome, "name", None) is not None:
            name = str(outcome.name)

    raw = np.asarray(values)
    try:
        num = np.asarray(values, dtype=float)
        numeric = True
    except (ValueError, TypeError):
        numeric = False
    if raw.ndim == 2:
        if not numeric:
            raise ValueError("2-D outcome (several outcomes) must be numeric.")
        if names is None:
            names = [f"y{j + 1}" for j in range(raw.shape[1])]
    if len(raw) != n:
        raise ValueError(
            f"{method}(): the outcome has {len(raw)} values but the text has {n} documents."
        )
    return (num if numeric and not keep_labels else raw), name, names


def _check_data(data, n: int, method: str) -> None:
    """Raise unless *data* (if given) has *n* rows and no column starting with ``_ssd_``."""
    if data is None:
        return
    lengths = (
        {len(data)} if hasattr(data, "iloc")
        else {len(np.asarray(v)) for v in data.values()}
    )
    if lengths != {n}:
        shown = lengths.pop() if len(lengths) == 1 else sorted(lengths)
        raise ValueError(
            f"len(data)={shown} != len(text)={n}. "
            "data must have one row per document."
        )
    columns = data.columns if hasattr(data, "iloc") else data.keys()
    clash = sorted(str(c) for c in columns if str(c).startswith("_ssd_"))
    if clash:
        raise ValueError(
            f"data columns {clash} start with '_ssd_', which {method}() "
            "reserves for its own columns. Rename them."
        )


def _check_outcome_side(formula: str, outcome: str) -> None:
    """Raise when the left side of *formula*, *outcome*, is empty."""
    if not outcome:
        raise ValueError(
            f"The formula {formula!r} needs the outcome column on its "
            "left side, e.g. 'score ~ age'."
        )


def _require_numeric(is_numeric: bool, method: str) -> None:
    if not is_numeric:
        raise ValueError(
            f"{method}() requires a numeric outcome, but it holds "
            "categorical labels — use fit_groups() instead."
        )


def _require_1d_y(y: np.ndarray, method: str) -> None:
    if y.ndim == 2 and y.shape[1] == 1:
        raise ValueError(
            f"{method}() takes a 1-D outcome, but it has shape {y.shape}. "
            "Pass a single column as 1-D, e.g. df['col']."
        )
    if y.ndim == 2:
        raise ValueError(
            f"{method}() takes one outcome, but it has "
            f"{y.shape[1]} columns. Use fit_multipls() for several outcomes."
        )


def _regression_outcome(outcome, data, n: int):
    """Check a fit_regression outcome: ``(formula or None, y over all n documents, y name)``."""
    formula = None
    if isinstance(outcome, str) and "~" in outcome:
        formula, outcome = outcome, outcome.rpartition("~")[0].strip()
        _check_outcome_side(formula, outcome)
    _check_data(data, n, "fit_regression")
    y, y_name, _ = _outcome_values(outcome, data, n, "fit_regression")
    _require_1d_y(y, "fit_regression")
    _require_numeric(y.dtype.kind == "f", "fit_regression")
    return formula, y, y_name


def _take_rows(data, idx: np.ndarray):
    """Rows *idx* of *data*, renumbered from 0. A DataFrame stays a DataFrame, so
    categorical columns keep their dtype and level order for glmm; ``None`` stays ``None``."""
    if data is None:
        return None
    if hasattr(data, "iloc"):
        return data.iloc[idx].reset_index(drop=True)
    return {k: np.asarray(v)[idx] for k, v in data.items()}


@dataclass
class _Rows:
    """The rows one fit uses: documents with a valid outcome and a vector.

    ``x``, ``y`` and ``data`` have one row per kept document, in text order.
    ``keep_mask`` has one entry per document of the text. ``rhs`` is the
    formula's right side, ``"1"`` without a formula. ``n_no_vector`` counts
    documents with a valid outcome and no vector.
    """

    x: np.ndarray
    y: np.ndarray
    is_numeric: bool
    y_name: str
    outcome_names: list[str] | None
    data: Any
    formula: str | None
    rhs: str
    keep_mask: np.ndarray
    n_raw: int
    n_kept: int
    n_dropped: int
    n_no_vector: int


def _wald_row(fit, label: str, L: np.ndarray) -> TermTest:
    """One Wald test of ``L @ beta = 0`` as a ``test`` row labelled *label*."""
    t = fit.contrast_test(L)
    if "F value" in t.columns:
        name, cols = "wald_f", ("F value", "NumDF", "DenDF", "Pr(>F)")
    else:
        name, cols = "wald_chisq", ("Chisq", "Df", None, "Pr(>Chisq)")
    stat, df, df_resid, p = (None if c is None else float(t[c][0]) for c in cols)
    return TermTest(label, name, p, stat, df, df_resid, None)


def _compare_row(cmp) -> tuple[str, dict]:
    """The test name and statistics of a ``glmm.compare`` table's second row."""
    p_col = "Pr(>F)" if "Pr(>F)" in cmp.table else "Pr(>Chisq)"
    f_test = p_col == "Pr(>F)"
    info = {
        "pvalue": float(cmp.table[p_col][1]),
        "statistic": float(cmp.table["F" if f_test else "Chisq"][1]),
        "df": int(cmp.table["Df"][1]),
        "df_resid": int(cmp.table["Res.Df"][1]) if f_test else None,
        # glmm reports r_squared for OLS only; NaN otherwise.
        "r2_delta": float(cmp.fit1.r_squared - cmp.fit0.r_squared),
    }
    return ("f_test" if f_test else "lrt"), info


def _fit_regression(
    y: np.ndarray,
    y_name: str,
    formula: str,
    data,
    texts: dict[str | None, SSD],
    *,
    family: str,
    link: str | None,
    fixed_k: int | None,
    k_min: int,
    k_max: int,
    k_step: int,
    verbose: bool,
    pca_sweep_args: dict | None,
    fit_args: dict,
):
    """Fit *formula*, whose ``ssd(...)`` terms stand for the *texts*, with glmm.

    *texts* maps the name inside ``ssd(...)`` (``None`` for ``ssd()``) to its
    text. *y* and *data* have one entry per document. Rows with a missing *y*
    or a missing vector in any text are dropped. Each text's PCA components
    replace its placeholder in the formula; the result is one leaf per text
    term (the term's coefficients back-projected to document space), or the
    leaf alone when there is one term. With several texts the result is keyed
    by text, then by term.
    """
    import glmm

    from ssdiff.results.continuous_result import PCAOLSResult, RegressionResult

    check_text_terms(formula)
    if "y" in texts:
        raise ValueError(
            "fit_regression(): the text name 'y' is reserved for the outcome; "
            "pick another name."
        )
    several = len(texts) > 1

    valid = np.isfinite(y)
    keep = valid.copy()
    for text in texts.values():
        keep &= text._vectors()[1]
    n_kept = int(keep.sum())
    data_kept = _take_rows(data, np.flatnonzero(keep))

    # Per text: its rows, standardization, K and PCA components.
    parts = {}
    for name, text in texts.items():
        has_vector = text._vectors()[1]
        rows = _Rows(
            x=text._x_rows(keep), y=y[keep], is_numeric=True, y_name=y_name,
            outcome_names=None, data=data_kept,
            formula=formula, rhs=formula.partition("~")[2].strip(),
            keep_mask=keep, n_raw=len(y), n_kept=n_kept,
            n_dropped=len(y) - n_kept, n_no_vector=int((valid & ~has_vector).sum()),
        )
        ys = standardize(rows.y.reshape(-1, 1))[0].ravel()
        Xs, X_mean, X_scale = standardize(rows.x)

        if fixed_k is None:
            from ssdiff.backends.pca_sweep import pca_sweep
            reserved = {
                "Xs", "X_scale", "x", "ys", "embeddings",
                "pca_k_values", "verbose",
            }
            extra = dict(pca_sweep_args) if pca_sweep_args else {}
            collisions = reserved & extra.keys()
            if collisions:
                raise ValueError(
                    f"pca_sweep_args cannot override fit_regression-managed keys: "
                    f"{sorted(collisions)}. Use the corresponding fit_regression "
                    f"parameter (k_min/k_max/k_step, verbose) instead."
                )
            if several:
                # One sweep table per text, not one file each text overwrites.
                extra["prefix"] = f"{extra['prefix']}_{name}" if "prefix" in extra else name
            sweep_result = pca_sweep(
                Xs=Xs,
                X_scale=X_scale,
                x=rows.x,
                ys=ys,
                embeddings=text.embeddings,
                pca_k_values=list(range(k_min, k_max + 1, k_step)),
                verbose=verbose,
                **extra,
            )
            n_pca = sweep_result.best_k
        else:
            n_pca = int(fixed_k)
            sweep_result = None

        max_comp = min(n_pca, Xs.shape[0], Xs.shape[1])
        z, components, _ = pca_fit_transform(Xs, max_comp)
        prefix = f"_ssd_{name}_pca_" if several else "_ssd_pca_"
        parts[name] = SimpleNamespace(
            rows=rows, X_mean=X_mean, X_scale=X_scale, sweep_result=sweep_result,
            z=z, components=components,
            pcs=[f"{prefix}{i}" for i in range(1, max_comp + 1)],
            ph=placeholder(name),
            # A bare ssd() is shown with its text's name, as in the leaf keys.
            shown=f"ssd({name if name is not None else text.name or ''})",
        )

    rows0 = next(iter(parts.values())).rows
    cols = SSD._data_with(
        rows0, _ssd_y=rows0.y,
        **{c: p.z[:, i] for p in parts.values() for i, c in enumerate(p.pcs)},
    )
    f_ph = "_ssd_y ~" + to_placeholders(formula).partition("~")[2]
    labels, _ = glmm.term_labels(f_ph)
    own_labels = {name: text_labels(labels, p.ph) for name, p in parts.items()}
    for lab in labels:
        if sum(p.ph in lab.split(":") for p in parts.values()) > 1:
            raise ValueError(
                f"The term '{lab}' has two texts; two texts cannot interact in one term."
            )
    h0 = null_formula(f_ph, {lab for v in own_labels.values() for lab in v})
    h1 = expand(f_ph, {p.ph: p.pcs for p in parts.values()})
    cmp = glmm.compare(cols, h0, h1, family=family, link=link)
    fit = cmp.fit1
    SSD._check_nobs(rows0, fit)

    test_name, test_info = _compare_row(cmp)
    pvalue = test_info["pvalue"]

    # A Gaussian identity-link fit reports w per raw-y unit; dividing by
    # sd(y) puts beta on the standardized-y scale of the other backends.
    # Other models keep the link scale, which has no y unit to divide by.
    linear = fit.family == "gaussian" and fit.link == "identity"
    if linear:
        _y_mean, _y_scale = standardize(rows0.y.reshape(-1, 1))[1:]
    else:
        _y_mean, _y_scale = np.zeros(1), np.ones(1)

    for p in parts.values():
        p.groups = term_groups(
            list(fit.names), {c: (p.shown, i) for i, c in enumerate(p.pcs, 1)}
        )
    n_leaves = sum(len(p.groups) for p in parts.values())

    # The all-text row, one row per text, then one Wald row per label of a
    # text that has several.
    tests = [TermTest(None, test_name, **test_info)]
    text_tests = {name: [] for name in parts}
    wald_tests = {name: [] for name in parts}
    for name, p in parts.items():
        if several:
            others = {q.ph: q.pcs for k, q in parts.items() if k != name}
            sub = glmm.compare(
                cols, expand(null_formula(f_ph, set(own_labels[name])), others), h1,
                family=family, link=link,
            )
            sub_name, sub_info = _compare_row(sub)
            text_tests[name].append(TermTest(name, sub_name, **sub_info))
        shown_labels = {
            lab: ":".join(p.shown if v == p.ph else v for v in lab.split(":"))
            for lab in own_labels[name]
        }
        if len(shown_labels) > 1:
            labels_shown = list(shown_labels.values())
            for lab_shown in labels_shown:
                idx = [
                    pos for key, members in p.groups.items()
                    if label_of(key, labels_shown) == lab_shown
                    for _, pos in members
                ]
                wald_tests[name].append(_wald_row(fit, lab_shown, np.eye(len(fit.names))[idx]))
    tests += [t for name in parts for t in text_tests[name]]
    tests += [t for name in parts for t in wald_tests[name]]

    results = {}
    for name, p in parts.items():
        text, rows = texts[name], p.rows
        sweep_result = p.sweep_result
        formula_shown = (
            re.sub(r"\bssd\(\s*\)", p.shown, formula) if name is None else formula
        )
        kwargs = text._base_result_kwargs(rows)
        kwargs["_y_mean"] = _y_mean
        kwargs["_y_scale"] = _y_scale
        kwargs["_x_mean"] = p.X_mean
        kwargs["model"] = fit
        if n_leaves > 1 or any(":" in key for key in p.groups):
            # Each leaf holds one term of the fit; the fit's predictions are
            # shared. A lone moderated term's beta alone is not the model.
            kwargs["y_hat"] = np.asarray(fit.fitted, dtype=float)

        _sweep_rows = None
        if sweep_result is not None:
            _sweep_rows = [
                {**r, "k": r["PCA_K"]} for r in sweep_result.df_joined
            ]

        scale = np.where(p.X_scale > 1e-12, p.X_scale, 1.0)
        leaves = {}
        for key, members in p.groups.items():
            w_reg = np.array([fit.beta[pos] for _, pos in members])
            if linear:
                w_reg = w_reg / _y_scale[0]
            # Back-project to doc space. No sign flip (_orient_beta): beta must
            # keep the sign of the model's coefficients. Without covariates an
            # OLS fit already correlates positively with y.
            beta = (p.components.T @ w_reg) / scale
            leaf = PCAOLSResult(
                backend="PCA+OLS" if fit_args["formula"] is None and linear else "PCA+GLMM",
                beta=beta,
                pvalue=pvalue,
                r2=float(fit.r_squared),
                r2_adj=float(fit.adj_r_squared),
                fit_info={
                    "n_components": len(p.pcs),
                    "k_min": k_min if sweep_result is not None else None,
                    "k_max": k_max if sweep_result is not None else None,
                    "k_step": k_step if sweep_result is not None else None,
                    "best_k": sweep_result.best_k if sweep_result is not None else None,
                    "pca_k_source": "sweep" if fixed_k is None else "fixed",
                    "formula": fit_args["formula"],
                    "family": fit.family,
                },
                raw_diagnostics={
                    "sweep_result": sweep_result,
                    "pca_components": p.components,
                    "pca_weights": w_reg,
                },
                sweep=_sweep_rows,
                test_name=test_name,
                test_info=test_info,
                **kwargs,
            )
            leaf._formula_shown = formula_shown
            leaf.fit_args = fit_args
            leaf.n_no_vector = rows.n_no_vector
            leaf._term = key
            leaves[key] = leaf
        results[name] = (
            next(iter(leaves.values())) if len(leaves) == 1
            else RegressionResult(
                leaves, text_tests[name] + wald_tests[name] if several else tests
            )
        )
    if several:
        return RegressionResult(results, tests, by_text=True)
    return next(iter(results.values()))


class SSD:
    """Supervised Semantic Differential: one text, tokenized once.

    An ``SSD`` holds the tokens of one text and builds its document vectors
    on first use. It knows nothing about an outcome.

    Attributes
    ----------
    embeddings : Embeddings
        The embeddings the vectors are built in (a view for a wordpack).
    name : str or None
        The text's column name when ``texts`` is a pandas Series.
    docs : list of list of str
        Tokens of each document.
    pre_docs : list of PreprocessedDoc or None
        Sentence structure for snippets; ``None`` for token-list texts.
    tokenizer : str or None
        ``"wordpack:<name>-<version>"`` for strings, ``None`` for token lists.
    lexicon : set of str
    window : int
    sif_a : float or None
    use_full_doc : bool

    >>> emb = Embeddings.load("english.wordpack")
    >>> ssd = SSD(emb, df["text"], lexicon)
    >>> ssd.with_lexicon(other_lexicon)    # same tokens, another lexicon
    """

    def __init__(
        self,
        embeddings: Embeddings,
        texts: Sequence[str] | Sequence[Sequence[str]],
        lexicon: Sequence[str] | set[str] | None = None,
        *,
        window: int = 3,
        sif_a: float | None = 1e-3,
        use_full_doc: bool = False,
    ) -> None:
        """Tokenize the texts; document vectors are built on first use.

        Parameters
        ----------
        embeddings : Embeddings
            L2-normalised word embeddings. Strings need a wordpack.
        texts : sequence of str, or sequence of token lists
            One entry per document. Strings are tokenized with the
            wordpack's tokenizer and stopwords; an empty, ``None`` or NaN
            cell is an empty document. Token lists are used as given.
        lexicon : sequence or set of str, optional
            Seed words for context-window extraction. Needed for vectors
            unless ``use_full_doc=True``; ``with_lexicon`` adds one later.
        window : int, default 3
            Context window size (tokens) around each seed word.
        sif_a : float or None, default 1e-3
            SIF smoothing parameter. ``None`` turns SIF off: every word
            weighs 1 (plain mean).
        use_full_doc : bool, default False
            If True, use full-document vectors instead of seed-windowed
            contexts.

        Raises
        ------
        TypeError
            If ``texts`` is a single string, or holds strings and the
            embeddings are not a wordpack.
        ValueError
            If ``window < 1`` or ``sif_a <= 0``.
        RuntimeError
            If the embeddings are not L2-normalised.
        """
        _check_window_sif(window, sif_a)
        if not embeddings.l2_normalized:
            raise RuntimeError(
                "SSD requires L2-normalised embeddings. Call "
                "emb.normalize(l2=True, abtt=N) before constructing SSD."
            )
        if isinstance(texts, str):
            # Iterating a str would make one document per character.
            raise TypeError("texts is a single string; pass a list of texts, e.g. [text].")

        name = getattr(texts, "name", None)
        self.name: str | None = None if name is None else str(name)
        texts = list(texts)

        if texts and isinstance(texts[0], (list, tuple)):
            bad = next((i for i, d in enumerate(texts) if not isinstance(d, (list, tuple))), None)
            if bad is not None:
                raise TypeError(
                    f"texts mixes token lists with other values: texts[{bad}] is a "
                    f"{type(texts[bad]).__name__}. Pass one list of tokens per document, "
                    "or one string per document."
                )
            self.docs: list[list[str]] = [list(d) for d in texts]
            self.pre_docs = None
            self.tokenizer: str | None = None
        else:
            pack = getattr(embeddings, "_pack", None)
            if pack is None:
                raise TypeError(
                    "SSD tokenizes strings with a wordpack; these embeddings are not one. "
                    "Pass token lists, one list of tokens per document."
                )
            self.pre_docs = preprocess_texts_wordpack(texts, pack, embeddings.stopwords)
            self.docs = build_docs_from_preprocessed(self.pre_docs)
            self.tokenizer = f"wordpack:{embeddings.name}-{embeddings.version}"
            # A wordpack always gets a view, so a later emb.normalize() leaves
            # this text on the rows it was made with. A partial pack also loads
            # the text's tail into the view, never into the caller's embedding,
            # so one pack stays reusable across texts.
            embeddings = embeddings.with_corpus(self)

        self.embeddings = embeddings
        self.lexicon = set(lexicon) if lexicon is not None else set()
        self.window = window
        self.sif_a = sif_a
        self.use_full_doc = use_full_doc
        self._cache = None

    def with_lexicon(
        self,
        lexicon: Sequence[str] | set[str],
        *,
        window: int = 3,
        sif_a: float | None = 1e-3,
        use_full_doc: bool = False,
    ) -> SSD:
        """A new ``SSD`` on the same tokens with another lexicon.

        Shares ``embeddings``, ``name``, ``docs``, ``pre_docs`` and
        ``tokenizer``; builds its own document vectors.
        """
        _check_window_sif(window, sif_a)
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new.lexicon = set(lexicon)
        new.window = window
        new.sif_a = sif_a
        new.use_full_doc = use_full_doc
        new._cache = None
        return new

    def _vectors(self) -> tuple[np.ndarray, np.ndarray, tuple | None]:
        """``(X, has_vector, sif_counts)``, built on first call.

        ``X`` has one row per document with a vector; ``has_vector`` has one
        entry per document. ``sif_counts`` are the SIF word counts of all
        documents, or ``None`` when ``sif_a`` is None.
        """
        if self.embeddings is None:
            raise RuntimeError(
                "This SSD has no embeddings (they are not pickled). "
                "Build it again with SSD(embeddings, texts, lexicon)."
            )
        if not self.use_full_doc and not self.lexicon:
            raise ValueError(
                "This SSD has no lexicon. Build it with SSD(emb, texts, lexicon), "
                "use ssd.with_lexicon(lexicon), or pass use_full_doc=True."
            )
        if self._cache is None:
            sif_counts = compute_global_sif(self.docs) if self.sif_a is not None else None
            X, has_vector = build_and_normalize_doc_vectors(
                self.docs, self.embeddings, self.lexicon,
                window=self.window, sif_a=self.sif_a, use_full_doc=self.use_full_doc,
                sif_counts=sif_counts,
            )
            self._cache = (np.asarray(X, dtype=np.float64), has_vector, sif_counts)
        return self._cache

    @property
    def n_kept(self) -> int:
        """Number of documents that have a vector."""
        return int(self._vectors()[1].sum())

    def __len__(self) -> int:
        return len(self.docs)

    def __getstate__(self) -> dict:
        # Embeddings are re-attached after loading; vectors are rebuilt on demand.
        state = self.__dict__.copy()
        state["embeddings"] = None
        state["_cache"] = None
        return state

    # ── Shared helpers ─────────────────────────────────────────

    def _x_rows(self, keep_mask: np.ndarray) -> np.ndarray:
        """Vectors of the documents where *keep_mask* is true; each must have a vector."""
        X, has_vector, _ = self._vectors()
        return X[keep_mask[has_vector]]

    def _rows(self, outcome, data, method: str, *, formula_ok: bool, multi: bool = False) -> _Rows:
        """Resolve *outcome* (values, column name or formula) to the rows of one fit.

        A string with ``~`` is a formula whose left side names the outcome
        column of *data*. A row is kept when its outcome is valid (finite
        for numbers, non-empty for labels) and its document has a vector.
        """
        n = len(self)
        formula, rhs = None, "1"
        if isinstance(outcome, str) and "~" in outcome:
            if not formula_ok:
                raise ValueError(f"{method}() takes no formula")
            formula = outcome
            lhs, _, rhs = outcome.rpartition("~")
            outcome, rhs = lhs.strip(), rhs.strip() or "1"
            _check_outcome_side(formula, outcome)

        _check_data(data, n, method)

        values, y_name, outcome_names = _outcome_values(outcome, data, n, method, multi=multi)
        is_numeric = values.dtype.kind == "f"
        if is_numeric:
            valid = np.isfinite(values)
            if valid.ndim == 2:
                valid = valid.all(axis=1)
        else:
            valid = _categorical_mask(values)

        has_vector = self._vectors()[1]
        keep = valid & has_vector
        data = _take_rows(data, np.flatnonzero(keep))
        n_kept = int(keep.sum())
        return _Rows(
            x=self._x_rows(keep), y=values[keep], is_numeric=is_numeric,
            y_name=y_name, outcome_names=outcome_names, data=data,
            formula=formula, rhs=rhs, keep_mask=keep, n_raw=n, n_kept=n_kept,
            n_dropped=n - n_kept, n_no_vector=int((valid & ~has_vector).sum()),
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

    @staticmethod
    def _data_with(rows: _Rows, **cols):
        """``rows.data`` with *cols* added; only *cols* when there is no data."""
        if hasattr(rows.data, "iloc"):
            return rows.data.assign(**cols)
        if rows.data is not None:
            return {**rows.data, **cols}
        return cols

    @staticmethod
    def _check_nobs(rows: _Rows, fit) -> None:
        """Raise if glmm dropped kept documents for missing values in the formula's columns."""
        if fit.nobs != rows.n_kept:
            raise ValueError(
                f"The formula's columns have missing values in "
                f"{rows.n_kept - fit.nobs} of the {rows.n_kept} kept documents. "
                "Fill or drop those rows before fitting."
            )

    def _residualize(self, rows: _Rows) -> tuple[np.ndarray, np.ndarray]:
        """The document vectors and y with the formula's fixed effects regressed out by OLS.

        One glmm fit per column of x: glmm builds the design from the formula
        (factors, interactions) but exposes no design matrix to project with.
        """
        import glmm

        rhs = rows.rhs
        # Every random-effect term, (1 | g) or (x || g), has a bar.
        if "|" in rhs:
            raise ValueError(
                "fit_pls() takes fixed covariates only, but the formula has random "
                "effects. Use fit_regression() for random effects."
            )
        fit = glmm.fit(self._data_with(rows, _ssd_y=rows.y), f"_ssd_y ~ {rhs}")
        self._check_nobs(rows, fit)
        y_res = np.asarray(fit.residuals(), dtype=float)
        x_res = np.column_stack([
            np.asarray(
                glmm.fit(self._data_with(rows, _ssd_x=rows.x[:, j]), f"_ssd_x ~ {rhs}").residuals(),
                dtype=float,
            )
            for j in range(rows.x.shape[1])
        ])
        return x_res, y_res

    @staticmethod
    def _orient_beta(beta: np.ndarray, ys: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Orient beta so higher alignment → higher outcome (alignment on *x*)."""
        yhat_std = (x @ beta).ravel()
        if float(np.std(yhat_std)) > 0:
            c = float(np.corrcoef(ys, yhat_std)[0, 1])
            corr = c if np.isfinite(c) else 0.0
        else:
            corr = 0.0
        if corr < 0:
            beta = -beta
        return beta

    def _base_result_kwargs(self, rows: _Rows) -> dict:
        """Common kwargs for result object construction.

        Does not include _y_mean / _y_scale — callers add those
        after computing standardization in their fit method.
        """
        return {
            "embeddings": self.embeddings,
            "text": self,
            "lexicon": self.lexicon,
            "window": self.window,
            "sif_a": self.sif_a,
            "x": rows.x,
            "keep_mask": rows.keep_mask,
            "n_raw": rows.n_raw,
            "n_kept": rows.n_kept,
            "n_dropped": rows.n_dropped,
            "y": rows.y,
            "sif_counts": self._vectors()[2],
            "use_full_doc": self.use_full_doc,
        }

    # ── PLS backend ────────────────────────────────────────────

    def fit_pls(
        self,
        outcome,
        *,
        data=None,
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

        Significance is tested with plskit's default ``test_method="auto"``
        (Lenartowicz & Plisiecki, 2026, "Cheap and Powerful Tests for
        Supervised Subspaces: Per-Component Inference for PLS", NeurIPS 2026,
        arXiv:2609.36307). It runs split_exact (permutation-calibrated
        split-half test) or split_nb (the same statistic with an asymptotic
        correction), chosen from X alone; ``result.test.name`` says which
        ran. The ``find_k_optimal`` diagnostic chain stays on split_nb.
        For ``k="auto"`` the reported p-value is the **same sample k=1
        confirmatory** test — independent of the ``find_k_optimal``
        selection that picks ``k_star`` for the actual fit.

        With a formula (``"y ~ age + gender"``), y and every column of the
        document vectors are first regressed on the formula's fixed effects by
        OLS, and PLS runs on the residuals (Frisch–Waugh–Lovell). split_exact permutes the residualized
        y against the residualized vectors (Kennedy, 1995); it can run
        slightly liberal when the covariates use many degrees of freedom
        relative to n.

        Parameters
        ----------
        outcome : array-like, str
            The numeric outcome, one value per document of the text: values,
            a column name of ``data``, or a formula such as
            ``"score ~ age + gender"`` whose left side names a column of
            ``data`` and whose right side holds the fixed covariates. Random
            effects such as ``(1 | ID)`` raise ``ValueError``: use
            :meth:`fit_regression` for them. With a formula, ``r2`` is the R²
            of the residualized y, ``docs.residual`` has the covariates
            taken out, and ``predict()`` is not available. The covariates'
            own coefficients are not reported: use :meth:`fit_regression`
            for them. Missing values drop the document.
        data : DataFrame or dict of 1-D arrays, optional
            One row per document: the outcome column and the formula's
            covariates.
        k : int or ``"auto"``, default ``"auto"``
            Number of PLS components.

            - ``int`` — fit at exactly this k; p-value comes from
              ``plskit.pls1_confirmatory_test`` at this k.
            - ``"auto"`` — let ``plskit.pls1_find_k_optimal`` pick
              ``k_star`` (selector ``"r2_se"``); p-value is the same sample
              k=1 confirmatory test.
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
        """
        import plskit

        from ssdiff.backends.pls import pls1_rotate, word_target
        from ssdiff.results.continuous_result import PLSResult

        if isinstance(outcome, str) and text_names(outcome):
            raise ValueError(
                "fit_pls() takes covariates only in its formula, no ssd() term; "
                "the text is the SSD itself."
            )
        rows = self._rows(outcome, data, "fit_pls", formula_ok=True)
        formula = rows.formula
        fit_args = {"formula": formula, "k": k, "k_max": k_max,
                    "ssd_gradients": ssd_gradients, "rotate": rotate,
                    "n_splits": n_splits, "random_state": random_state, "test": test}
        _require_1d_y(rows.y, "fit_pls")
        _require_numeric(rows.is_numeric, "fit_pls")
        n_kept, D = rows.x.shape
        g_max = min(n_kept - 1, D)
        if not isinstance(ssd_gradients, int) or not 1 <= ssd_gradients <= g_max:
            raise ValueError(
                f"ssd_gradients must be an int in [1, {g_max}] (min(n - 1, D) = "
                f"min({n_kept - 1}, {D})); got {ssd_gradients!r}"
            )
        if ssd_gradients > 1 and rotate not in (None, "words"):
            raise ValueError(
                f"fit_pls() rotate must be 'words' or None; got {rotate!r}. "
                "One outcome gives no outcome loadings to rotate against."
            )

        y_std = float(np.std(rows.y))
        if y_std <= 1e-12:
            raise ValueError(
                f"fit_pls() requires y with nonzero variance (std={y_std:.3g}). "
                "PLS1 is undefined when y is constant."
            )

        x_fit, y_fit = rows.x, rows.y
        if formula is not None:
            x_fit, y_fit = self._residualize(rows)
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
            _y_mean, X_mean = np.atleast_1d(np.mean(rows.y)), rows.x.mean(axis=0)

        find_k_raw = None
        test_name = None
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
                        f"(test_method='auto', n_splits={n_splits})"
                    )
                r = plskit.pls1_confirmatory_test(
                    Xs, ys, 1,
                    test_method="auto",
                    args={"n_splits": int(n_splits)},
                    pre_standardized=True,
                    seed=random_state,
                    verbose=verbose,
                )
                pvalue = float(r.pvalue)
                test_name = r.test_method
                if verbose:
                    print(f"[fit_pls] p-value (k=1, {test_name}) = {pvalue:.6g}")
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
                    test_method="auto",
                    args={"n_splits": int(n_splits)},
                    pre_standardized=True,
                    seed=random_state,
                    verbose=verbose,
                )
                pvalue = float(r.pvalue)
                test_name = r.test_method
            p_at_k = n_comp
        else:
            raise ValueError(f"k must be int or 'auto', got {k!r}")

        m = plskit.pls1_fit(
            Xs, ys, k=n_comp, pre_standardized=True, seed=random_state,
        )
        T, W, coef = m.T, m.W, m.coef

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

        kwargs = self._base_result_kwargs(rows)
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
        result.fit_args = fit_args
        result.n_no_vector = rows.n_no_vector
        return result

    # ── PLS3 backend (several outcomes) ────────────────────────

    def fit_multipls(
        self,
        outcome,
        *,
        data=None,
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
        outcome-loading vector ``V[:, i]``. Needs at least
        two outcome columns; use :meth:`fit_pls` for one outcome.

        Significance is the PLS3 test at LV1
        (``plskit.pls3_confirmatory_test`` with its default
        ``test_method="auto"``, which runs split_exact or split_nb, chosen
        from X alone): a container-level p-value that does not depend on
        ``rotate``. Leaves carry no p-value.

        Parameters
        ----------
        outcome : list of str, array-like
            The outcomes, numeric: a list of column names of ``data``, or a
            2-D array or DataFrame with one row per document. A row with a
            missing value in any outcome drops the document.
        data : DataFrame or dict of 1-D arrays, optional
            One row per document; holds the outcome columns.
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
            Random splits for the test.
        n_perm : int, default 1000
            Permutations for the test; unused when split_nb runs.
        random_state : int, default 2137
            Random seed for the test.
        verbose : bool

        Returns
        -------
        MultiPLSResult

        Raises
        ------
        ValueError
            On a 1-D outcome, fewer than two outcomes, a constant outcome, an
            invalid ``rotate`` / ``outcome_groups`` / ``ssd_gradients``, or
            when plskit keeps fewer than ``ssd_gradients`` components.
        """
        fit_args = {"ssd_gradients": ssd_gradients, "rotate": rotate,
                    "outcome_groups": outcome_groups, "n_splits": n_splits,
                    "n_perm": n_perm, "random_state": random_state}
        import plskit

        from ssdiff.backends.pls import pls3_rotate, word_target
        from ssdiff.results.multi_pls_result import MultiPLSResult

        rows = self._rows(outcome, data, "fit_multipls", formula_ok=False, multi=True)
        if rows.y.ndim != 2 or rows.y.shape[1] < 2:
            raise ValueError(
                "fit_multipls() needs several outcomes (at least two columns). "
                "For one outcome use fit_pls()."
            )
        names = rows.outcome_names
        n_kept, D = rows.x.shape
        q = rows.y.shape[1]

        y_std = rows.y.std(axis=0)
        constant = [name for name, s in zip(names, y_std) if s <= 1e-12]
        if constant:
            raise ValueError(
                f"fit_multipls() requires outcomes with nonzero variance; "
                f"constant: {constant}."
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

        Xs, X_mean, X_scale = standardize(rows.x)
        Ys, _, _ = standardize(rows.y)
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
                f"(test_method='auto', n_splits={n_splits}, n_perm={n_perm})"
            )
        r = plskit.pls3_confirmatory_test(
            Xs, Ys, 1,
            test_method="auto",
            args={"n_perm": int(n_perm), "n_splits": int(n_splits)},
            seed=random_state,
            verbose=verbose,
        )
        test_info = {
            "pvalue": float(r.pvalue),
            "statistic": float(r.statistic),
            "n_splits": int(n_splits),
            "n_perm": r.n_perm,
            "random_state": random_state,
        }

        result = MultiPLSResult(
            x=rows.x, y=rows.y,
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
            test_name=r.test_method,
            test_info=test_info,
            embeddings=self.embeddings,
            text=self,
            lexicon=self.lexicon,
            window=self.window,
            sif_a=self.sif_a,
            random_state=random_state,
        )
        result.fit_args = fit_args
        result.n_no_vector = rows.n_no_vector
        # Text row of each row of ``x``: the results page links documents back to their text.
        result.doc_ids = np.flatnonzero(rows.keep_mask)
        return result

    # ── PCA regression backend ────────────────────────────────

    def fit_ols(self, *args, **kwargs):
        """Deprecated alias of :meth:`fit_regression`; same arguments, emits ``FutureWarning``."""
        warnings.warn(
            "fit_ols() is renamed fit_regression() and will be removed; "
            "call ssd.fit_regression(...) with the same arguments.",
            FutureWarning, stacklevel=2,
        )
        return self.fit_regression(*args, **kwargs)

    def fit_regression(
        self,
        outcome,
        *,
        data=None,
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

        The text is a term of the formula, ``ssd()``: ``"score ~ age + ssd()"``.
        A formula without ``ssd()`` gets ``+ ssd()`` appended. The text can
        interact with covariates, ``"score ~ age + group*ssd()"``; the result
        then holds one leaf per text term (see Returns).

        K is chosen by the PCA-K sweep (OLS on standardized y), unless
        ``fixed_k`` is given. The final model at that K is fit by ``glmm``:
        the K component scores replace each ``ssd()`` term in the formula, so
        covariates and random effects (``"score ~ age + sex*ssd() + (1 | ID)"``)
        are controlled for. The p-value tests all the text terms together: the
        full formula against the formula without them (the null model is built
        from the term labels), both fit by ML on the same rows. A text with
        several terms also gets one Wald row per term. An OLS model (Gaussian, no random
        effects) gets the F-test, and so do Gamma and inverse-Gaussian GLMs
        (Wald F); every other model gets the likelihood-ratio test.

        Parameters
        ----------
        outcome : array-like, str
            The outcome, one value per document of the text: values, a column
            name of ``data``, or a formula such as
            ``"score ~ age + gender + (1 | ID)"`` whose left side names a
            column of ``data`` and whose right side holds covariates and
            random effects. Without a formula the model is intercept plus
            components. Missing values drop the document.
        data : DataFrame or dict of 1-D arrays, optional
            One row per document: the outcome column and the formula's
            covariates and grouping factors.
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
            ``verbose``) cannot be overridden and raise
            ``ValueError`` if passed. Ignored when ``fixed_k`` is set.

        Returns
        -------
        PCAOLSResult or RegressionResult
            A formula with one text term gives a ``PCAOLSResult``; one with
            several (``group*ssd()``) gives a ``RegressionResult`` with one
            ``PCAOLSResult`` leaf per text term, keyed by term
            (``res["grouplow:ssd(essay)"]``).

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
            If the outcome is categorical or 2-D, if a column name or formula
            is given without ``data`` or names a column ``data`` does not
            have, if a ``data`` column name starts with ``_ssd_``, if the
            formula has ``ssd(...)`` terms for more than one text, or if a
            column the formula uses has missing values in kept documents.
        """
        formula, y, y_name = _regression_outcome(outcome, data, len(self))
        full = (
            with_text_term(formula) if formula is not None else f"{y_name} ~ ssd()"
        )
        names = text_names(full)
        if len(names) > 1:
            shown = ", ".join("ssd()" if n is None else f"ssd({n})" for n in names)
            raise ValueError(
                f"A method call fits one text, but the formula has {shown}. "
                "Use ssdiff.fit_regression(formula, data, **texts) for several texts."
            )
        fit_args = {"formula": formula, "family": family, "link": link,
                    "fixed_k": fixed_k, "k_min": k_min, "k_max": k_max, "k_step": k_step,
                    "pca_sweep_args": pca_sweep_args}
        return _fit_regression(
            y, y_name, full, data, {names[0]: self},
            family=family, link=link, fixed_k=fixed_k, k_min=k_min, k_max=k_max,
            k_step=k_step, verbose=verbose, pca_sweep_args=pca_sweep_args,
            fit_args=fit_args,
        )

    # ── Group comparison backend ──────────────────────────────

    def fit_groups(
        self,
        outcome,
        *,
        data=None,
        median_split: bool = False,
        n_perm: int = 5000,
        correction: Literal["holm", "bonferroni", "fdr_bh", "none"] = "holm",
        random_state: int = 2137,
        verbose: bool = False,
    ):
        """Fit group comparison using the outcome as group labels.

        Parameters
        ----------
        outcome : array-like, str
            One label per document of the text: values, or a column name of
            ``data``. Documents with an empty or missing label drop out.
        data : DataFrame or dict of 1-D arrays, optional
            One row per document; holds the label column.
        median_split : bool, default False
            If True, split the outcome into "low"/"high" at the median.
            If False, treat its values as categorical group labels.
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

        rows = self._rows(outcome, data, "fit_groups", formula_ok=False)
        _require_1d_y(rows.y, "fit_groups")
        fit_args = {"median_split": median_split, "n_perm": n_perm,
                    "correction": correction, "random_state": random_state}
        # Text row of each row of ``x``: the results page links documents back to their text.
        doc_ids = np.flatnonzero(rows.keep_mask)

        # Local copy — never mutate rows.x
        x_local = rows.x.copy()

        if median_split:
            groups_local = _median_split(rows.y, random_state=random_state)
        else:
            # A numeric outcome holds floats; a whole number labels its group as an int ("3", not "3.0").
            groups_local = np.asarray(
                [int(g) if rows.is_numeric and float(g).is_integer() else g for g in rows.y], dtype=object)

        # Small-group filtering (only when not median_split)
        n_group_dropped = 0
        if not median_split:
            x_local, groups_local, n_group_dropped = filter_small_groups(
                x_local, groups_local,
            )
            doc_ids = doc_ids[np.isin(np.asarray(rows.y, dtype=object), np.unique(groups_local))]

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
            text=self,
            x=x_local,
            groups=groups_local,
            lexicon=self.lexicon,
            window=self.window,
            sif_a=self.sif_a,
        )
        result.fit_args = fit_args
        result.doc_ids = doc_ids
        result.n_no_vector = rows.n_no_vector
        return result

    def _lexicon_outcome(self, outcome, data, method: str, var_type: str):
        """The outcome values for a lexicon helper; ``None`` stays ``None``.

        Categorical labels are kept as given, so group keys read 1, 2, not 1.0, 2.0.
        """
        if outcome is None:
            return None
        return _outcome_values(
            outcome, data, len(self), method, keep_labels=var_type == "categorical",
        )[0]

    def suggest_lexicon(
        self,
        outcome=None,
        *,
        data=None,
        seeds: Sequence[str] | None = None,
        top_k: int = 30,
        min_docs: int = 5,
        n_bins: int = 4,
        corr_cap: float = 0.30,
        var_type: str = "continuous",
    ):
        """Suggest seed words: ranked by balanced coverage, or close to *seeds*.

        Uses the tokens of this text, as the fits do.

        With *seeds*, lists the text's tokens closest to the seeds by cosine
        in ``self.embeddings``, so that the forms and related words that occur
        in the text can be picked for the lexicon. A lexicon matches exact
        tokens, so a plural or other form counts only when it is listed.

        Parameters
        ----------
        outcome : array-like, column name, or None
            Outcome variable (numeric for continuous, labels for
            categorical), or the name of a column of *data*. Must have one
            value per document; invalid entries (NaN / None) are filtered
            automatically. ``None`` is allowed with *seeds*: no rows are dropped, the
            outcome columns (``cov_bal``, ``corr``, ``pvalue``,
            ``direction``, ``rank``) are ``None``, and *n_bins*,
            *corr_cap* and *var_type* are ignored.
        data : DataFrame, optional
            Holds the column *outcome* names.
        seeds : sequence of str or None
            Seed words; each must be a key of ``self.embeddings``. A seed
            need not occur in the text.
        top_k : int, default 30
            Maximum number of candidate tokens to return. With *seeds*, the
            number of neighbours listed per seed.
        min_docs : int, default 5
            Minimum document frequency for a token to be considered.
        n_bins : int, default 4
            Number of quantile bins for continuous outcomes.
        corr_cap : float, default 0.30
            Association cap for rank penalty.
        var_type : str, default "continuous"
            ``"continuous"`` or ``"categorical"``.

        Returns
        -------
        LexiconResult
            Per-token rows sorted by descending rank. With *seeds*, the rows
            are each seed, its *top_k* closest tokens, and the seed with an
            English ending (``-s``, ``-ed``, ``-ing``, ...), in one table
            sorted by descending cosine. A token listed for several seeds has
            one row, with the seed it is closest to. Only tokens in at least
            *min_docs* documents and in ``self.embeddings`` are listed.

        Raises
        ------
        TypeError
            If *seeds* is a single string rather than a sequence of seeds.
        ValueError
            If *outcome* is ``None`` without *seeds* (ranking needs an
            outcome); if *seeds* is empty; if a seed is not a key of
            ``self.embeddings``; if *top_k* is None or negative.
        """
        from ssdiff.results.lexicon_result import LexiconResult
        from ssdiff.results.schema import Suggestion
        from ssdiff.utils.lexicon import _filter_y, _rank_tokens, _seed_neighbours, _token_sets

        if top_k is None or top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k!r}")
        if seeds is None:
            if outcome is None:
                raise ValueError(
                    "Ranking needs an outcome. To list neighbours of words "
                    "without one, pass seeds=."
                )
        else:
            if isinstance(seeds, str):
                # Iterating a str would make one seed per character.
                raise TypeError("seeds is a single string; pass a list of seeds, e.g. [seed].")
            seeds = list(seeds)
            if not seeds:
                raise ValueError("seeds is empty; pass at least one seed word.")
            missing = [s for s in seeds if s not in self.embeddings]
            if missing:
                raise ValueError(f"These seeds are not in the embeddings: {missing}")

        y = self._lexicon_outcome(outcome, data, "suggest_lexicon", var_type)
        docs, y_clean = _filter_y(self.docs, y, var_type=var_type)
        token_sets = _token_sets(docs)
        if seeds is None:
            raw_rows = _rank_tokens(
                token_sets, y_clean,
                top_k=top_k, min_docs=min_docs, n_bins=n_bins,
                corr_cap=corr_cap, var_type=var_type,
            )
            suggestions = [
                Suggestion(
                    rank=r["rank"], token=r["token"], freq=r["freq"],
                    cov_all=r["cov_all"], cov_bal=r["cov_bal"],
                    corr=r["corr"], pvalue=r["pvalue"], direction=r["direction"],
                )
                for r in raw_rows
            ]
        else:
            listed = _seed_neighbours(
                token_sets, self.embeddings, seeds,
                top_k=top_k, min_docs=min_docs,
            )
            stats = {
                r["token"]: r
                for r in self.token_stats(
                    y, [token for _, token, _ in listed],
                    n_bins=n_bins, corr_cap=corr_cap, var_type=var_type,
                )
            }
            suggestions = [
                Suggestion(**stats[token], seed=seed, cos=cos)
                for seed, token, cos in listed
            ]
        return LexiconResult(
            var_type=var_type, n_docs=len(docs),
            n_tokens=len(suggestions), suggestions=suggestions,
            summary=None, seeded=seeds is not None, text=self,
        )

    def token_stats(
        self,
        outcome,
        lexicon,
        *,
        data=None,
        n_bins: int = 4,
        corr_cap: float = 0.30,
        var_type: str = "continuous",
    ) -> list[dict]:
        """Compute per-token statistics for a specific lexicon.

        Parameters
        ----------
        outcome : array-like, column name, or None
            Outcome variable, or the name of a column of *data*. Must have
            one value per document; invalid entries are filtered
            automatically. ``None`` drops no rows and leaves ``cov_bal``, ``corr``,
            ``rank``, ``pvalue`` and ``direction`` as ``None``; rows are
            then sorted by descending ``freq``. *n_bins*, *corr_cap* and
            *var_type* are ignored.
        lexicon : iterable of str
            Tokens to compute stats for.
        data : DataFrame, optional
            Holds the column *outcome* names.
        n_bins : int, default 4
            Number of quantile bins (continuous only).
        corr_cap : float, default 0.30
            Association cap for rank penalty.
        var_type : str, default "continuous"
            ``"continuous"`` or ``"categorical"``.

        Returns
        -------
        list[dict]
            One dict per token with keys ``token``, ``freq``,
            ``cov_all``, ``cov_bal``, ``corr``, ``rank``, ``pvalue``,
            ``direction``, sorted by descending rank.
        """
        from ssdiff.utils.lexicon import (
            _chi2_pvalue,
            _effect_direction,
            _filter_y,
            _pointbiserial_pvalue,
            _rank_for_token_stats,
            _token_sets,
        )

        y = self._lexicon_outcome(outcome, data, "token_stats", var_type)
        docs, y_clean = _filter_y(self.docs, y, var_type=var_type)
        token_sets = _token_sets(docs)
        is_categorical = var_type == "categorical"
        lex = [str(w) for w in lexicon]

        rows: list[dict] = []
        for t in lex:
            pres = np.fromiter(
                (1 if t in ts else 0 for ts in token_sets),
                dtype=np.int8,
                count=len(token_sets),
            )
            if y_clean is None:
                rows.append(dict(
                    token=t, freq=int(pres.sum()), cov_all=float(pres.mean()),
                    cov_bal=None, corr=None, rank=None, pvalue=None, direction=None,
                ))
                continue
            cov_all, cov_bal, corr, rank = _rank_for_token_stats(
                pres, y_clean,
                n_bins=n_bins, corr_cap=corr_cap,
                categorical=is_categorical,
            )
            if is_categorical:
                pval = _chi2_pvalue(pres.astype(int), y_clean)
                direction = _effect_direction(
                    pres, y_clean, categorical=True,
                )
            else:
                pval = _pointbiserial_pvalue(pres.astype(float), y_clean)
                direction = _effect_direction(
                    pres, y_clean, categorical=False,
                )
            rows.append(dict(
                token=t,
                freq=int(pres.sum()),
                cov_all=cov_all,
                cov_bal=cov_bal,
                corr=corr,
                rank=rank,
                pvalue=pval,
                direction=direction,
            ))

        if y_clean is None:
            rows.sort(key=lambda r: (-r["freq"], r["token"]))
        else:
            rows.sort(key=lambda r: (-r["rank"], -r["cov_bal"], -r["freq"], r["token"]))
        return rows

    def coverage_summary(
        self,
        outcome,
        lexicon,
        *,
        data=None,
        n_bins: int = 4,
        var_type: str = "continuous",
    ) -> dict:
        """Aggregate coverage statistics for a lexicon.

        Parameters
        ----------
        outcome : array-like, column name, or None
            Outcome variable, or the name of a column of *data*. Must have
            one value per document; invalid entries are filtered
            automatically. ``None`` drops no rows and leaves ``q1``, ``q4`` and
            ``corr_any`` as ``None``, without ``group_cov``. *n_bins* and
            *var_type* are ignored.
        lexicon : iterable of str
            Tokens to measure coverage for.
        data : DataFrame, optional
            Holds the column *outcome* names.
        n_bins : int, default 4
            Number of quantile bins (continuous only).
        var_type : str, default "continuous"
            ``"continuous"`` or ``"categorical"``.

        Returns
        -------
        dict
            Keys: ``docs_any``, ``cov_all``, ``q1``, ``q4``,
            ``corr_any``, ``hits_mean``, ``hits_median``,
            ``types_mean``, ``types_median``
            [, ``group_cov`` for categorical].
        """
        from ssdiff.utils.lexicon import (
            _cramers_v,
            _filter_y,
            _quantile_bins,
            _texts_to_token_lists,
            _token_sets,
            _z,
        )

        y = self._lexicon_outcome(outcome, data, "coverage_summary", var_type)
        docs, y_clean = _filter_y(self.docs, y, var_type=var_type)
        is_categorical = y is not None and var_type == "categorical"

        if y is not None and (len(docs) == 0 or len(y_clean) == 0):
            summary = dict(
                docs_any=0, cov_all=0.0, q1=0.0, q4=0.0,
                corr_any=0.0, hits_mean=0.0, hits_median=0.0,
                types_mean=0.0, types_median=0.0,
            )
            if is_categorical:
                summary["group_cov"] = {}
            return summary

        lex = [str(w) for w in lexicon]
        token_sets = _token_sets(docs)

        pres_any = np.fromiter(
            (1 if any(w in ts for w in lex) else 0 for ts in token_sets),
            dtype=np.int8,
            count=len(token_sets),
        )

        docs_any = int(pres_any.sum())
        overall = float(pres_any.mean()) if len(pres_any) else 0.0

        if y_clean is None:
            q1 = q4 = corr_any = None
        elif is_categorical:
            groups = y_clean
            group_labels = sorted(set(groups))
            group_cov_any: dict = {}
            for g in group_labels:
                idx = np.where(groups == g)[0]
                group_cov_any[g] = (
                    float(pres_any[idx].mean()) if len(idx) else 0.0
                )
            q1 = min(group_cov_any.values()) if group_cov_any else 0.0
            q4 = max(group_cov_any.values()) if group_cov_any else 0.0
            corr_any = _cramers_v(pres_any.astype(int), groups)
        else:
            bins = _quantile_bins(y_clean, n_bins=n_bins)
            low_idx = np.where(bins == bins.min())[0]
            high_idx = np.where(bins == bins.max())[0]
            q1 = float(pres_any[low_idx].mean()) if len(low_idx) else 0.0
            q4 = float(pres_any[high_idx].mean()) if len(high_idx) else 0.0
            y_std = _z(y_clean)
            if pres_any.std() < 1e-12:
                corr_any = 0.0
            else:
                c = float(np.corrcoef(pres_any, y_std)[0, 1])
                corr_any = c if np.isfinite(c) else 0.0

        texts = _texts_to_token_lists(docs)
        lex_set = set(lex)
        hits_per_unit = np.array(
            [sum(1 for t in toks if t in lex_set) for toks in texts],
            dtype=np.int32,
        )
        types_per_unit = np.array(
            [len(set(toks) & lex_set) for toks in texts],
            dtype=np.int32,
        )

        summary = dict(
            docs_any=docs_any,
            cov_all=overall,
            q1=q1,
            q4=q4,
            corr_any=corr_any,
            hits_mean=float(hits_per_unit.mean()) if len(hits_per_unit) else 0.0,
            hits_median=float(np.median(hits_per_unit)) if len(hits_per_unit) else 0.0,
            types_mean=float(types_per_unit.mean()) if len(types_per_unit) else 0.0,
            types_median=float(np.median(types_per_unit)) if len(types_per_unit) else 0.0,
        )
        if is_categorical:
            summary["group_cov"] = group_cov_any

        return summary

    def evaluate_lexicon(
        self,
        outcome,
        lexicon,
        *,
        data=None,
        n_bins: int = 4,
        corr_cap: float = 0.30,
        var_type: str = "continuous",
    ):
        """Per-token stats and aggregate coverage for a lexicon.

        Convenience method combining :meth:`token_stats` and
        :meth:`coverage_summary` into a single
        :class:`~ssdiff.LexiconResult`.

        Parameters
        ----------
        outcome : array-like, column name, or None
            Outcome variable (one value per document), or the name of a
            column of *data*. ``None`` returns only the statistics that need
            no outcome; the others are ``None``.
        lexicon : iterable of str
            Tokens to evaluate; a repeated token is evaluated once.
        data : DataFrame, optional
            Holds the column *outcome* names.
        n_bins, corr_cap, var_type :
            Passed to both ``token_stats`` and ``coverage_summary``.

        Returns
        -------
        LexiconResult
            Per-token rows plus aggregate coverage summary.
        """
        from ssdiff.results.lexicon_result import LexiconResult
        from ssdiff.results.schema import Suggestion, Summary
        from ssdiff.utils.lexicon import _filter_y

        lexicon = list(dict.fromkeys(lexicon))
        y = self._lexicon_outcome(outcome, data, "evaluate_lexicon", var_type)
        docs, _ = _filter_y(self.docs, y, var_type=var_type)
        raw_rows = self.token_stats(
            y, lexicon, n_bins=n_bins, corr_cap=corr_cap,
            var_type=var_type,
        )
        s = self.coverage_summary(
            y, lexicon, n_bins=n_bins, var_type=var_type,
        )
        suggestions = [
            Suggestion(
                rank=r["rank"], token=r["token"], freq=r["freq"],
                cov_all=r["cov_all"], cov_bal=r["cov_bal"],
                corr=r["corr"], pvalue=r["pvalue"], direction=r["direction"],
            )
            for r in raw_rows
        ]
        summary = Summary(
            docs_any=s["docs_any"],
            cov_all=s["cov_all"],
            q1=s["q1"],
            q4=s["q4"],
            corr_any=s["corr_any"],
            hits_mean=s["hits_mean"],
            hits_median=s["hits_median"],
            types_mean=s["types_mean"],
            types_median=s["types_median"],
            group_cov=s.get("group_cov"),
        )
        return LexiconResult(
            var_type=var_type, n_docs=len(docs),
            n_tokens=len(suggestions), suggestions=suggestions,
            summary=summary, text=self,
        )

    def __repr__(self) -> str:
        header = f"SSD  name={self.name}  docs={len(self):,}  |L|={len(self.lexicon)}"
        methods = "  methods: .fit_pls()  .fit_multipls()  .fit_regression()  .fit_groups()"
        return "\n".join([header, methods])

    def _repr_html_(self) -> str:
        """Render a plain-text repr inside a ``<pre>`` block for Jupyter."""
        import html as _html
        return f"<pre>{_html.escape(repr(self))}</pre>"


def fit_regression(
    formula: str,
    data,
    *,
    family: str = "gaussian",
    link: str | None = None,
    fixed_k: int | None = None,
    k_min: int = 2,
    k_max: int = 120,
    k_step: int = 2,
    verbose: bool = False,
    pca_sweep_args: dict | None = None,
    **texts: SSD,
):
    """Regress an outcome on several texts in one model.

    ``ssdiff.fit_regression("score ~ age + ssd(essay) + sex*ssd(diary)", df,
    essay=essay, diary=diary)``. The left side is the outcome column of
    *data*; each keyword names a text in the formula as ``ssd(name)``. Every
    text and *data* have one row per document, in the same order. A row stays
    when it has a valid outcome and a vector in every text. Each text gets its
    own standardization, PCA and K; ``fixed_k``, ``k_min``, ``k_max``,
    ``k_step`` and ``pca_sweep_args`` apply to every text; with
    ``save_tables``, each text's sweep table file name has the text name in its
    prefix. The other
    parameters are as in :meth:`SSD.fit_regression`.

    Returns
    -------
    RegressionResult
        Keyed by text name. An entry is a ``RegressionResult`` keyed by term,
        or a ``PCAOLSResult`` for a text with one term. ``test`` holds the
        all-text row, one row per text, then the per-term rows; ``sweep`` is
        keyed by text name. With one text the result has no text level, as
        from ``SSD.fit_regression``.

    Raises
    ------
    ValueError
        If the formula has no ``ssd(name)`` term, has bare ``ssd()``, names a
        text that was not passed, or a text was passed that the formula does
        not name; or if the texts and *data* differ in length.
    """
    if "~" not in formula:
        raise ValueError(f"fit_regression(): the formula {formula!r} needs an outcome and a '~'.")
    names = text_names(formula)
    if not names:
        raise ValueError(
            f"fit_regression(): the formula {formula!r} has no ssd(name) term; "
            "write one for each text, e.g. 'y ~ ssd(essay) + ssd(diary)'."
        )
    if None in names:
        raise ValueError(
            "fit_regression(): bare ssd() is for SSD.fit_regression; "
            "write ssd(name) and pass name=text."
        )
    unused = [n for n in texts if n not in names]
    missing = [n for n in names if n not in texts]
    if unused or missing:
        raise ValueError(
            f"fit_regression(): the formula names {names} but the texts passed are "
            f"{list(texts)} (not in the formula: {unused}; not passed: {missing})."
        )
    if data is None:
        raise ValueError("fit_regression() needs data=df.")
    sizes = {n: len(t) for n, t in texts.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"fit_regression(): the texts differ in length: {sizes}.")
    n = next(iter(sizes.values()))
    _, y, y_name = _regression_outcome(formula, data, n)
    fit_args = {"formula": formula, "family": family, "link": link,
                "fixed_k": fixed_k, "k_min": k_min, "k_max": k_max, "k_step": k_step,
                "pca_sweep_args": pca_sweep_args}
    return _fit_regression(
        y, y_name, formula, data, {n: texts[n] for n in names},
        family=family, link=link, fixed_k=fixed_k, k_min=k_min, k_max=k_max,
        k_step=k_step, verbose=verbose, pca_sweep_args=pca_sweep_args,
        fit_args=fit_args,
    )
