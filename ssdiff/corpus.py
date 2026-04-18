"""Corpus: tokenize and lemmatize texts via spaCy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as _np

from ssdiff.utils.text import (
    build_docs_from_preprocessed,
    lang_to_model,
    load_spacy,
    load_stopwords,
    preprocess_texts,
)


class Corpus:
    """Tokenize and lemmatize texts via spaCy.

    >>> corpus = Corpus(texts, lang="pl")
    >>> corpus = Corpus(texts, model="pl_core_news_lg")
    >>> corpus.docs        # list[list[str]] — lemmatized tokens
    >>> corpus.pre_docs    # list[PreprocessedDoc] — for snippets
    """

    def __init__(
        self,
        texts: Sequence[Union[str, Sequence[str]]],
        *,
        lang: str | None = None,
        model: str | None = None,
        nlp=None,
        stopwords: Sequence[str] | None = None,
        pretokenized: bool = False,
    ) -> None:
        """Tokenize and lemmatize texts using spaCy.

        Parameters
        ----------
        texts : sequence of str or sequence of sequence of str
            Raw text strings, or pre-tokenized token lists when
            ``pretokenized=True``. For profile mode pass
            ``list[list[str]]`` (multiple posts per participant).
        lang : str or None
            Language code (e.g. ``"pl"``, ``"en"``). Resolves to the
            default spaCy model for that language.
        model : str or None
            Explicit spaCy model name (e.g. ``"pl_core_news_lg"``).
            Overrides *lang*.
        nlp : spacy.Language or None
            Pre-loaded spaCy ``Language`` object. Overrides *lang* and
            *model*.
        stopwords : sequence of str or None
            Custom stopword list. If ``None``, loads bundled / spaCy
            defaults for the resolved language.
        pretokenized : bool, default False
            If ``True``, skip spaCy processing -- *texts* are already
            token lists.

        Raises
        ------
        ValueError
            If none of *lang*, *model*, or *nlp* is provided (and
            ``pretokenized`` is ``False``).
        """
        if pretokenized:
            self.docs: list = list(texts)  # type: ignore
            self.pre_docs: list | None = None
            self.lang = lang
            return

        if nlp is None:
            if model is not None:
                nlp = load_spacy(model)
            elif lang is not None:
                spacy_model = lang_to_model(lang)
                nlp = load_spacy(spacy_model)
            else:
                raise ValueError("Provide lang=, model=, or nlp=.")

        # Resolve lang for stopwords
        resolved_lang = lang
        if resolved_lang is None and model is not None:
            resolved_lang = model.split("_")[0]

        if stopwords is None and resolved_lang is not None:
            stopwords = load_stopwords(resolved_lang)

        self.lang = resolved_lang
        self.pre_docs = preprocess_texts(texts, nlp, stopwords or [])
        self.docs = build_docs_from_preprocessed(self.pre_docs)

    def suggest_lexicon(
        self,
        y,
        *,
        top_k: int = 30,
        min_docs: int = 5,
        n_bins: int = 4,
        corr_cap: float = 0.30,
        var_type: str = "continuous",
    ):
        """Suggest seed words ranked by balanced coverage.

        Uses the lemmatized tokens on this Corpus instance — no
        re-tokenization, consistent with what SSD will consume.

        Parameters
        ----------
        y : array-like
            Outcome variable (numeric for continuous, labels for
            categorical).  Must have the same length as ``self.docs``;
            invalid entries (NaN / None) are filtered automatically.
        top_k : int, default 30
            Maximum number of candidate tokens to return.
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
            Per-token rows sorted by descending rank.
        """
        from ssdiff.results.lexicon_result import LexiconResult
        from ssdiff.results.schema import Suggestion
        from .utils.lexicon import _filter_y, _rank_tokens, _token_sets

        docs, y_clean = _filter_y(self.docs, y, var_type=var_type)
        token_sets = _token_sets(docs)
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
        return LexiconResult(
            var_type=var_type, n_docs=len(docs),
            n_tokens=len(suggestions), suggestions=suggestions,
            summary=None, corpus=self,
        )

    def token_stats(
        self,
        y,
        lexicon,
        *,
        n_bins: int = 4,
        corr_cap: float = 0.30,
        var_type: str = "continuous",
    ) -> list[dict]:
        """Compute per-token statistics for a specific lexicon.

        Parameters
        ----------
        y : array-like
            Outcome variable.  Must have the same length as
            ``self.docs``; invalid entries are filtered automatically.
        lexicon : iterable of str
            Tokens to compute stats for.
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
        from .utils.lexicon import (
            _chi2_pvalue,
            _effect_direction,
            _filter_y,
            _pointbiserial_pvalue,
            _rank_for_token_stats,
            _token_sets,
        )

        docs, y_clean = _filter_y(self.docs, y, var_type=var_type)
        token_sets = _token_sets(docs)
        is_categorical = var_type == "categorical"
        lex = [str(w) for w in lexicon]

        rows: list[dict] = []
        for t in lex:
            pres = _np.fromiter(
                (1 if t in ts else 0 for ts in token_sets),
                dtype=_np.int8,
                count=len(token_sets),
            )
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

        rows.sort(key=lambda r: (-r["rank"], -r["cov_bal"], -r["freq"]))
        return rows

    def coverage_summary(
        self,
        y,
        lexicon,
        *,
        n_bins: int = 4,
        var_type: str = "continuous",
    ) -> dict:
        """Aggregate coverage statistics for a lexicon.

        Parameters
        ----------
        y : array-like
            Outcome variable.  Must have the same length as
            ``self.docs``; invalid entries are filtered automatically.
        lexicon : iterable of str
            Tokens to measure coverage for.
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
        from .utils.lexicon import (
            _cramers_v,
            _filter_y,
            _quantile_bins,
            _texts_to_token_lists,
            _token_sets,
            _z,
        )

        docs, y_clean = _filter_y(self.docs, y, var_type=var_type)
        is_categorical = var_type == "categorical"

        if len(docs) == 0 or len(y_clean) == 0:
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

        pres_any = _np.fromiter(
            (1 if any(w in ts for w in lex) else 0 for ts in token_sets),
            dtype=_np.int8,
            count=len(token_sets),
        )

        docs_any = int(pres_any.sum())
        overall = float(pres_any.mean()) if len(pres_any) else 0.0

        if is_categorical:
            groups = y_clean
            group_labels = sorted(set(groups))
            group_cov_any: dict = {}
            for g in group_labels:
                idx = _np.where(groups == g)[0]
                group_cov_any[g] = (
                    float(pres_any[idx].mean()) if len(idx) else 0.0
                )
            q1 = min(group_cov_any.values()) if group_cov_any else 0.0
            q4 = max(group_cov_any.values()) if group_cov_any else 0.0
            corr_any = _cramers_v(pres_any.astype(int), groups)
        else:
            bins = _quantile_bins(y_clean, n_bins=n_bins)
            low_idx = _np.where(bins == bins.min())[0]
            high_idx = _np.where(bins == bins.max())[0]
            q1 = float(pres_any[low_idx].mean()) if len(low_idx) else 0.0
            q4 = float(pres_any[high_idx].mean()) if len(high_idx) else 0.0
            y_std = _z(y_clean)
            if pres_any.std() < 1e-12:
                corr_any = 0.0
            else:
                c = float(_np.corrcoef(pres_any, y_std)[0, 1])
                corr_any = c if _np.isfinite(c) else 0.0

        texts = _texts_to_token_lists(docs)
        lex_set = set(lex)
        hits_per_unit = _np.array(
            [sum(1 for t in toks if t in lex_set) for toks in texts],
            dtype=_np.int32,
        )
        types_per_unit = _np.array(
            [len(set(toks) & lex_set) for toks in texts],
            dtype=_np.int32,
        )

        summary = dict(
            docs_any=docs_any,
            cov_all=overall,
            q1=q1,
            q4=q4,
            corr_any=corr_any,
            hits_mean=float(hits_per_unit.mean()) if len(hits_per_unit) else 0.0,
            hits_median=float(_np.median(hits_per_unit)) if len(hits_per_unit) else 0.0,
            types_mean=float(types_per_unit.mean()) if len(types_per_unit) else 0.0,
            types_median=float(_np.median(types_per_unit)) if len(types_per_unit) else 0.0,
        )
        if is_categorical:
            summary["group_cov"] = group_cov_any

        return summary

    def evaluate_lexicon(
        self,
        y,
        lexicon,
        *,
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
        y : array-like
            Outcome variable (same length as ``self.docs``).
        lexicon : iterable of str
            Tokens to evaluate.
        n_bins, corr_cap, var_type :
            Passed to both ``token_stats`` and ``coverage_summary``.

        Returns
        -------
        LexiconResult
            Per-token rows plus aggregate coverage summary.
        """
        from ssdiff.results.lexicon_result import LexiconResult
        from ssdiff.results.schema import Suggestion, Summary
        from .utils.lexicon import _filter_y

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
            summary=summary, corpus=self,
        )

    @property
    def n_texts(self) -> int:
        return len(self.docs)

    def __len__(self) -> int:
        return len(self.docs)

    def __repr__(self) -> str:
        return f"Corpus({len(self.docs)} docs)"
