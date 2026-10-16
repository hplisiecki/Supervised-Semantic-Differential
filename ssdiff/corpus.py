"""Corpus: tokenize texts with spaCy (lemmas) or with a wordpack (surface keys)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as _np

from ssdiff.utils.text import (
    SpacyModelNotInstalledError,
    build_docs_from_preprocessed,
    download_spacy_model,
    lang_to_model,
    load_spacy,
    load_stopwords,
    preprocess_texts,
    preprocess_texts_wordpack,
)


def _resolve_spacy_model(model: str, auto_download: bool | None) -> "object":
    """Load *model*, optionally downloading if missing.

    ``auto_download=True`` downloads without prompting; ``False`` re-raises
    immediately; ``None`` prompts interactively, declining if stdin is not a
    TTY.
    """
    try:
        return load_spacy(model)
    except SpacyModelNotInstalledError:
        if auto_download is False:
            raise
        if auto_download is None:
            import sys
            if not sys.stdin.isatty():
                raise
            try:
                ans = input(
                    f"spaCy model '{model}' is not installed. "
                    f"Download now? [y/N]: "
                ).strip().lower()
            except EOFError:
                raise
            if ans not in ("y", "yes"):
                raise
        download_spacy_model(model)
        return load_spacy(model)


def _check_tokenizer(embeddings, corpus) -> None:
    """Raise when *corpus* was not tokenized for *embeddings*.

    A wordpack is keyed by surface form and a spaCy corpus holds lemmas, so
    a mismatched pair would fit silently on whatever words they happen to
    share. A corpus without a recorded tokenizer (``pretokenized=True``) is
    not checked.
    """
    tokenizer = getattr(corpus, "tokenizer", None)
    if tokenizer is None:
        return
    if getattr(embeddings, "_pack", None) is None:
        if tokenizer.startswith("wordpack:"):
            raise ValueError(
                f"The corpus was tokenized with the wordpack {tokenizer[len('wordpack:'):]}, "
                "but these embeddings are not a wordpack. Load that wordpack with "
                "Embeddings.load(...), or build the corpus with Corpus(texts, lang=...)."
            )
        return
    expected = f"wordpack:{embeddings.name}-{embeddings.version}"
    if tokenizer != expected:
        raise ValueError(
            f"The corpus was tokenized with {tokenizer}, but the embeddings are "
            f"{expected}. Build the corpus with Corpus(texts, embeddings=emb)."
        )


class Corpus:
    """Tokenize texts via spaCy (lemmas) or a wordpack (the pack's keys).

    >>> corpus = Corpus(texts, lang="pl")
    >>> corpus = Corpus(texts, model="pl_core_news_lg")
    >>> corpus = Corpus(texts, embeddings=emb)   # emb loaded from a .wordpack
    >>> corpus.docs        # list[list[str]] — tokens per document
    >>> corpus.pre_docs    # list[PreprocessedDoc] — for snippets
    >>> corpus.tokenizer   # "spacy:<model>", "wordpack:<name>-<version>" or None
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
        auto_download: bool | None = None,
        embeddings=None,
    ) -> None:
        """Tokenize texts with spaCy or with a wordpack.

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
            defaults for the resolved language, or the wordpack's list.
        pretokenized : bool, default False
            If ``True``, skip tokenization -- *texts* are already
            token lists.
        auto_download : bool or None, default None
            Behavior when the required spaCy model is not installed.
            ``True`` downloads it silently; ``False`` raises
            :class:`~ssdiff.utils.text.SpacyModelNotInstalledError`;
            ``None`` (default) prompts interactively when running in a
            TTY and raises otherwise.
        embeddings : Embeddings or None
            Embeddings loaded from a ``.wordpack``. Tokenizes with the
            pack's own tokenizer and stopwords; needs no spaCy. Tokens are
            the pack's keys (surface forms for English), not lemmas. Cannot
            be combined with *lang*, *model*, *nlp* or *pretokenized*.

        Raises
        ------
        TypeError
            If *texts* is a single string rather than a sequence of texts.
        ValueError
            If none of *lang*, *model*, *nlp* or *embeddings* is provided
            (and ``pretokenized`` is ``False``); if *embeddings* is combined
            with another tokenizer or is not a wordpack.
        SpacyModelNotInstalledError
            If the required spaCy model is not installed and
            auto-download was declined or unavailable.
        """
        if isinstance(texts, str):
            # Iterating a str would make one document per character.
            raise TypeError("texts is a single string; pass a list of texts, e.g. [text].")
        if embeddings is not None:
            if lang is not None or model is not None or nlp is not None or pretokenized:
                raise ValueError(
                    "embeddings= tokenizes with the wordpack; do not combine it "
                    "with lang=, model=, nlp= or pretokenized=True."
                )
            pack = getattr(embeddings, "_pack", None)
            if pack is None:
                raise ValueError(
                    "Corpus(texts, embeddings=emb) needs embeddings loaded from a "
                    ".wordpack file. For other formats use Corpus(texts, lang=...)."
                )
            self.lang = embeddings.lang
            self.tokenizer: str | None = f"wordpack:{embeddings.name}-{embeddings.version}"
            self.pre_docs = preprocess_texts_wordpack(
                texts, pack, embeddings.stopwords if stopwords is None else stopwords
            )
            self.docs = build_docs_from_preprocessed(self.pre_docs)
            return

        if pretokenized:
            self.docs: list = list(texts)  # type: ignore
            self.pre_docs: list | None = None
            self.lang = lang
            self.tokenizer = None
            return

        if nlp is None:
            if model is None and lang is None:
                raise ValueError("Provide lang=, model=, nlp= or embeddings=.")
            spacy_name = model if model is not None else lang_to_model(lang)
            nlp = _resolve_spacy_model(spacy_name, auto_download)
        else:
            meta = getattr(nlp, "meta", None) or {}
            spacy_name = f"{meta.get('lang')}_{meta.get('name')}"
        self.tokenizer = f"spacy:{spacy_name}"

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
        seeds: Sequence[str] | None = None,
        embeddings=None,
        top_k: int = 30,
        min_docs: int = 5,
        n_bins: int = 4,
        corr_cap: float = 0.30,
        var_type: str = "continuous",
    ):
        """Suggest seed words: ranked by balanced coverage, or close to *seeds*.

        Uses the lemmatized tokens on this Corpus instance — no
        re-tokenization, consistent with what SSD will consume.

        With *seeds*, lists the corpus tokens closest to the seeds by cosine
        in *embeddings*, so that the forms and related words that occur in
        the text can be picked for the lexicon. A lexicon matches exact
        tokens, so a plural or other form counts only when it is listed.

        Parameters
        ----------
        y : array-like or None
            Outcome variable (numeric for continuous, labels for
            categorical).  Must have the same length as ``self.docs``;
            invalid entries (NaN / None) are filtered automatically.
            ``None`` is allowed with *seeds*: no rows are dropped, the
            outcome columns (``cov_bal``, ``corr``, ``pvalue``,
            ``direction``, ``rank``) are ``None``, and *n_bins*,
            *corr_cap* and *var_type* are ignored.
        seeds : sequence of str or None
            Seed words; each must be a key of *embeddings*. A seed need not
            occur in the corpus.
        embeddings : Embeddings or None
            Embeddings the cosines are taken in. Required with *seeds*.
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
            *min_docs* documents and in *embeddings* are listed.

        Raises
        ------
        TypeError
            If *seeds* is a single string rather than a sequence of seeds.
        ValueError
            If *y* is ``None`` without *seeds* (ranking needs an outcome);
            if *seeds* is empty; if only one of *seeds* and *embeddings* is
            given; if a seed is not a key of *embeddings*; if the corpus was
            not tokenized for *embeddings*.
        """
        from ssdiff.results.lexicon_result import LexiconResult
        from ssdiff.results.schema import Suggestion

        from .utils.lexicon import _filter_y, _rank_tokens, _seed_neighbours, _token_sets

        if seeds is None:
            if embeddings is not None:
                raise ValueError("embeddings= is only used with seeds=.")
            if y is None:
                raise ValueError(
                    "Ranking needs an outcome y. To list neighbours of words "
                    "without one, pass seeds= and embeddings=."
                )
        else:
            if isinstance(seeds, str):
                # Iterating a str would make one seed per character.
                raise TypeError("seeds is a single string; pass a list of seeds, e.g. [seed].")
            seeds = list(seeds)
            if not seeds:
                raise ValueError("seeds is empty; pass at least one seed word.")
            if embeddings is None:
                raise ValueError(
                    "seeds= needs embeddings=: the tokens listed for a seed are "
                    "its neighbours in the embedding."
                )
            _check_tokenizer(embeddings, self)
            missing = [s for s in seeds if s not in embeddings]
            if missing:
                raise ValueError(f"These seeds are not in the embeddings: {missing}")

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
                token_sets, embeddings.with_corpus(self), seeds,
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
            summary=None, seeded=seeds is not None, corpus=self,
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
        y : array-like or None
            Outcome variable.  Must have the same length as
            ``self.docs``; invalid entries are filtered automatically.
            ``None`` drops no rows and leaves ``cov_bal``, ``corr``,
            ``rank``, ``pvalue`` and ``direction`` as ``None``; rows are
            then sorted by descending ``freq``. *n_bins*, *corr_cap* and
            *var_type* are ignored.
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
            rows.sort(key=lambda r: -r["freq"])
        else:
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
        y : array-like or None
            Outcome variable.  Must have the same length as
            ``self.docs``; invalid entries are filtered automatically.
            ``None`` drops no rows and leaves ``q1``, ``q4`` and
            ``corr_any`` as ``None``, without ``group_cov``. *n_bins* and
            *var_type* are ignored.
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

        pres_any = _np.fromiter(
            (1 if any(w in ts for w in lex) else 0 for ts in token_sets),
            dtype=_np.int8,
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
        y : array-like or None
            Outcome variable (same length as ``self.docs``). ``None``
            returns only the statistics that need no outcome; the others are
            ``None``.
        lexicon : iterable of str
            Tokens to evaluate; a repeated token is evaluated once.
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

        lexicon = list(dict.fromkeys(lexicon))
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
        """Number of documents in the corpus (alias for ``len(self)``)."""
        return len(self.docs)

    def __len__(self) -> int:
        return len(self.docs)

    def __repr__(self) -> str:
        header = f"Corpus  n={len(self.docs):,}  lang={self.lang}"
        arrays = "  arrays:  .docs  .pre_docs"
        methods = (
            "  methods: .suggest_lexicon(...)  .evaluate_lexicon(...)  "
            ".token_stats(...)  .coverage_summary(...)"
        )
        return "\n".join([header, arrays, methods])

    def _repr_html_(self) -> str:
        """Render a plain-text repr inside a ``<pre>`` block for Jupyter."""
        import html as _html
        return f"<pre>{_html.escape(repr(self))}</pre>"
