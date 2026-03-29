"""Corpus: tokenize and lemmatize texts via spaCy."""

from __future__ import annotations

from typing import Sequence, Union

from ssdlite.utils.text import (
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

    @property
    def n_texts(self) -> int:
        return len(self.docs)

    def __len__(self) -> int:
        return len(self.docs)

    def __repr__(self) -> str:
        return f"Corpus({len(self.docs)} docs)"
