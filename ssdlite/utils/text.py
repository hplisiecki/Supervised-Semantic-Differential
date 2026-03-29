"""spaCy-based text preprocessing: tokenization, lemmatization, stopwords."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Optional, Sequence, Union

import spacy

from ssdlite.lang_config import lang_to_model, get_config  # noqa: F401 — re-export


# ---------- Stopwords ----------


@lru_cache(maxsize=16)
def load_stopwords(lang: str = "pl", *, lowercase: bool = True) -> list[str]:
    """Load stopwords.

    If the language has a bundled stopword file (configured in
    ``lang_config.LANGUAGES``), loads from that file. Otherwise falls
    back to spaCy's built-in stopwords.
    """
    lang = (lang or "pl").strip().lower()

    cfg = get_config(lang)
    if cfg.stopwords_file is not None:
        ref = resources.files("ssdlite.utils").joinpath(cfg.stopwords_file)
        text = ref.read_text(encoding="utf-8")
        words = [s.strip() for s in text.splitlines() if s.strip()]
        return [w.lower() for w in words] if lowercase else words

    nlp_blank = spacy.blank(lang)
    sw = getattr(nlp_blank.Defaults, "stop_words", None)
    if not sw:
        raise LookupError(f"No stopwords available in spaCy for language '{lang}'.")
    words = list(sw)
    return [w.lower() for w in words] if lowercase else words


# ---------- spaCy loader ----------

def load_spacy(
    model: str,
    *,
    disable: Sequence[str] = ("ner",),
) -> spacy.language.Language:
    """Load a spaCy model with optional pipeline component disabling.

    Parameters
    ----------
    model : str
        spaCy model name (e.g. ``"pl_core_news_lg"``).
    disable : sequence of str, optional
        Pipeline components to disable, by default ``("ner",)``.

    Returns
    -------
    spacy.language.Language
        Loaded model with a sentencizer added if no parser is present.

    Raises
    ------
    RuntimeError
        If the model cannot be loaded (e.g. not installed).
    """
    if not model or not isinstance(model, str) or not model.strip():
        raise ValueError("Provide a spaCy model name (e.g. 'pl_core_news_lg').")
    try:
        nlp = spacy.load(model, disable=list(disable))
        if "parser" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp
    except Exception as e:
        raise RuntimeError(
            f"Could not load spaCy model '{model}': {e}. See https://spacy.io/models"
        ) from e


# ---------- Token filter ----------

_URL = re.compile(r"https?://\S+")
_AT = re.compile(r"@\S+")


def _keep_token(tok, stopset: set[str]) -> bool:
    if tok.is_space or tok.is_punct or tok.is_quote or tok.is_currency:
        return False
    if _URL.match(tok.text) or _AT.match(tok.text):
        return False
    if tok.is_digit:
        return False
    lem = tok.lemma_.lower()
    if not lem or lem in stopset:
        return False
    return True


# ---------- Data structures ----------

@dataclass
class PreprocessedDoc:
    """Preprocessed single document with sentence-level structure.

    Attributes
    ----------
    raw : str
        Original text string.
    sents_surface : list[str]
        Surface-form sentences.
    sents_lemmas : list[list[str]]
        Lemmatized tokens per sentence (stopwords removed).
    doc_lemmas : list[str]
        Flat list of all kept lemmas across sentences.
    sent_char_spans : list[tuple[int, int]]
        Character-level (start, end) spans for each sentence.
    token_to_sent : list[int]
        Maps each token index in doc_lemmas to its sentence index.
    sents_kept_idx : list[list[int]]
        Per-sentence indices of kept tokens within the spaCy sentence.
    """

    raw: str
    sents_surface: list[str]
    sents_lemmas: list[list[str]]
    doc_lemmas: list[str]
    sent_char_spans: list[tuple[int, int]]
    token_to_sent: list[int]
    sents_kept_idx: list[list[int]]


@dataclass
class PreprocessedProfile:
    """Preprocessed multi-post profile (e.g. social media user with multiple posts).

    Each field is a list indexed by post position.

    Attributes
    ----------
    raw_posts : list[str]
        Original text of each post.
    post_sents_surface : list[list[str]]
        Surface-form sentences per post.
    post_sents_lemmas : list[list[list[str]]]
        Lemmatized tokens per sentence per post.
    post_doc_lemmas : list[list[str]]
        Flat lemma list per post.
    post_sent_char_spans : list[list[tuple[int, int]]]
        Character spans per sentence per post.
    post_token_to_sent : list[list[int]]
        Token-to-sentence mapping per post.
    post_sents_kept_idx : list[list[list[int]]]
        Kept token indices per sentence per post.
    """

    raw_posts: list[str]
    post_sents_surface: list[list[str]]
    post_sents_lemmas: list[list[list[str]]]
    post_doc_lemmas: list[list[str]]
    post_sent_char_spans: list[list[tuple[int, int]]]
    post_token_to_sent: list[list[int]]
    post_sents_kept_idx: list[list[list[int]]]


# ---------- Helpers ----------

def _pipe(nlp, texts: Sequence[str], batch_size: int, n_process: int):
    if "parser" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp.pipe(texts, batch_size=batch_size, n_process=n_process)


def _extract_from_doc(doc, stopset: set[str]):
    s_surface, s_lemmas, s_spans, s_kept_idx = [], [], [], []
    doc_lemmas, token_to_sent = [], []
    for si, sent in enumerate(doc.sents):
        s_surface.append(sent.text)
        s_spans.append((sent.start_char, sent.end_char))
        kept_lemmas, kept_idx = [], []
        for j, tok in enumerate(sent):
            if _keep_token(tok, stopset):
                kept_lemmas.append(tok.lemma_.lower())
                kept_idx.append(j)
        s_lemmas.append(kept_lemmas)
        s_kept_idx.append(kept_idx)
        start_flat = len(doc_lemmas)
        doc_lemmas.extend(kept_lemmas)
        token_to_sent.extend([si] * (len(doc_lemmas) - start_flat))
    return s_surface, s_lemmas, doc_lemmas, s_spans, token_to_sent, s_kept_idx


def _is_profile_input(texts) -> bool:
    for x in texts:
        if x is None:
            continue
        return not isinstance(x, (str, bytes))
    return False


def _sanitize_posts(posts) -> list[str]:
    out: list[str] = []
    for p in posts or []:
        if isinstance(p, bytes):
            p = p.decode(errors="ignore")
        if isinstance(p, str):
            s = p.strip()
            if s:
                out.append(s)
    return out


# ---------- Public API ----------

def preprocess_texts(
    texts: Sequence[Union[str, Sequence[str]]],
    nlp,
    stopwords: Optional[Sequence[str]] = None,
    batch_size: int = 64,
    n_process: int = 1,
) -> list[Union[PreprocessedDoc, PreprocessedProfile]]:
    """Preprocess texts into lemmatized tokens with sentence boundaries.

    Handles both flat texts (``list[str]``) and profile mode
    (``list[list[str]]``).  The mode is auto-detected from the input.

    Parameters
    ----------
    texts : sequence of str or sequence of sequence of str
        Input documents.  Plain strings for flat mode; nested lists of
        strings (one inner list per post) for profile mode.
    nlp : spacy.language.Language
        Loaded spaCy model (see :func:`load_spacy`).
    stopwords : sequence of str, optional
        Stopword list; tokens whose lemma appears here are dropped.
    batch_size : int, optional
        Batch size forwarded to ``nlp.pipe()``, by default 64.
    n_process : int, optional
        Number of processes forwarded to ``nlp.pipe()``, by default 1.

    Returns
    -------
    list[PreprocessedDoc] or list[PreprocessedProfile]
        One entry per input element.  Returns ``PreprocessedDoc`` objects
        in flat mode and ``PreprocessedProfile`` objects in profile mode.
    """
    if nlp is None:
        raise ValueError("nlp is None. Call load_spacy(model) first.")

    stopset = set(stopwords or [])
    out: list = []

    if not _is_profile_input(texts):
        texts_str = []
        for t in texts:
            if t is None or (isinstance(t, float) and t != t):
                texts_str.append("")
                continue
            if isinstance(t, bytes):
                t = t.decode(errors="ignore")
            texts_str.append(t if isinstance(t, str) else str(t))

        for doc in _pipe(nlp, texts_str, batch_size, n_process):
            s_surface, s_lemmas, doc_lemmas, s_spans, token_to_sent, s_kept_idx = (
                _extract_from_doc(doc, stopset)
            )
            out.append(
                PreprocessedDoc(
                    raw=doc.text,
                    sents_surface=s_surface,
                    sents_lemmas=s_lemmas,
                    doc_lemmas=doc_lemmas,
                    sent_char_spans=s_spans,
                    token_to_sent=token_to_sent,
                    sents_kept_idx=s_kept_idx,
                )
            )
        return out

    # Profile mode
    posts_per_profile = [_sanitize_posts(p) for p in texts]
    lengths = [len(p) for p in posts_per_profile]
    flat_posts = [pp for plist in posts_per_profile for pp in plist]

    out_profiles: list[PreprocessedProfile] = []
    prof_idx = 0
    remaining = lengths[0] if lengths else 0

    def _empty_profile():
        return PreprocessedProfile(
            raw_posts=[], post_sents_surface=[], post_sents_lemmas=[],
            post_doc_lemmas=[], post_sent_char_spans=[], post_token_to_sent=[],
            post_sents_kept_idx=[],
        )

    while prof_idx < len(lengths) and remaining == 0:
        out_profiles.append(_empty_profile())
        prof_idx += 1
        remaining = lengths[prof_idx] if prof_idx < len(lengths) else 0

    cur_surface, cur_lemmas, cur_docs = [], [], []
    cur_spans, cur_tok2sent, cur_kept_idx = [], [], []

    for doc in _pipe(nlp, flat_posts, batch_size, n_process):
        s_surface, s_lemmas, doc_lemmas, s_spans, token_to_sent, s_kept_idx = (
            _extract_from_doc(doc, stopset)
        )
        cur_surface.append(s_surface)
        cur_lemmas.append(s_lemmas)
        cur_docs.append(doc_lemmas)
        cur_spans.append(s_spans)
        cur_tok2sent.append(token_to_sent)
        cur_kept_idx.append(s_kept_idx)
        remaining -= 1

        if remaining == 0:
            out_profiles.append(
                PreprocessedProfile(
                    raw_posts=posts_per_profile[prof_idx],
                    post_sents_surface=cur_surface,
                    post_sents_lemmas=cur_lemmas,
                    post_doc_lemmas=cur_docs,
                    post_sent_char_spans=cur_spans,
                    post_token_to_sent=cur_tok2sent,
                    post_sents_kept_idx=cur_kept_idx,
                )
            )
            prof_idx += 1
            cur_surface, cur_lemmas, cur_docs = [], [], []
            cur_spans, cur_tok2sent, cur_kept_idx = [], [], []

            if prof_idx < len(lengths):
                remaining = lengths[prof_idx]
                while prof_idx < len(lengths) and remaining == 0:
                    out_profiles.append(_empty_profile())
                    prof_idx += 1
                    remaining = lengths[prof_idx] if prof_idx < len(lengths) else 0

    while prof_idx < len(lengths):
        out_profiles.append(_empty_profile())
        prof_idx += 1

    return out_profiles


def build_docs_from_preprocessed(
    pre_docs: list[Union[PreprocessedDoc, PreprocessedProfile]],
) -> list:
    """Extract flat token lists from preprocessed docs/profiles."""
    if not pre_docs:
        return []
    first = pre_docs[0]
    if isinstance(first, PreprocessedDoc):
        return [p.doc_lemmas for p in pre_docs]
    return [[lemmas for lemmas in p.post_doc_lemmas] for p in pre_docs]
