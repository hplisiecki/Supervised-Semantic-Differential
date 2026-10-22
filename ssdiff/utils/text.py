"""Text preprocessing: a wordpack's tokenizer (surface keys) and stopwords."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
        Per-sentence indices of kept tokens within the sentence, counting stopwords.
    """

    raw: str
    sents_surface: list[str]
    sents_lemmas: list[list[str]]
    doc_lemmas: list[str]
    sent_char_spans: list[tuple[int, int]]
    token_to_sent: list[int]
    sents_kept_idx: list[list[int]]


# ---------- Helpers ----------

def _extract_from_tokens(t, stopset: set[str], lower):
    """Sentence-level and flat token data from a wordpack reader's ``Tokens``.

    Returns ``(sents_surface, sents_lemmas, doc_lemmas, sent_char_spans,
    token_to_sent, sents_kept_idx)``.

    *lower* is the pack's own lowercasing (``pack.chars.lower``), so a cased
    key matches a lowercase stopword the same way in every reader.
    ``sents_kept_idx`` counts every token of the sentence, stopwords included.
    """
    s_surface = [t.text[a:b] for a, b in t.sents]
    s_lemmas: list[list[str]] = [[] for _ in t.sents]
    s_kept_idx: list[list[int]] = [[] for _ in t.sents]
    seen = [0] * len(t.sents)
    for key, si in zip(t.keys, t.sent):
        j = seen[si]
        seen[si] += 1
        if lower(key) not in stopset:
            s_lemmas[si].append(key)
            s_kept_idx[si].append(j)
    doc_lemmas = [k for sent in s_lemmas for k in sent]
    token_to_sent = [si for si, sent in enumerate(s_lemmas) for _ in sent]
    return s_surface, s_lemmas, doc_lemmas, list(t.sents), token_to_sent, s_kept_idx


# ---------- Public API ----------

def _assemble_preprocessed(texts, extract) -> list[PreprocessedDoc]:
    """Shape per-text extraction results into ``PreprocessedDoc`` objects.

    *extract* takes a list of strings and yields, for each string in order,
    ``(raw, fields)``, where *fields* is the 6-tuple returned by
    :func:`_extract_from_tokens`.
    """
    texts_str = []
    for i, t in enumerate(texts):
        if isinstance(t, (list, tuple)):
            raise TypeError(
                f"texts[{i}] is a {type(t).__name__}; each element of texts must be "
                "one document's text. For several posts per participant, join each "
                "participant's posts into one text, or keep one row per post and add "
                "a random effect for the participant in fit_regression()."
            )
        if t is None or (isinstance(t, float) and t != t):
            texts_str.append("")
            continue
        if isinstance(t, bytes):
            t = t.decode(errors="ignore")
        texts_str.append(t if isinstance(t, str) else str(t))

    out: list[PreprocessedDoc] = []
    for raw, fields in extract(texts_str):
        s_surface, s_lemmas, doc_lemmas, s_spans, token_to_sent, s_kept_idx = fields
        out.append(
            PreprocessedDoc(
                raw=raw,
                sents_surface=s_surface,
                sents_lemmas=s_lemmas,
                doc_lemmas=doc_lemmas,
                sent_char_spans=s_spans,
                token_to_sent=token_to_sent,
                sents_kept_idx=s_kept_idx,
            )
        )
    return out


def preprocess_texts_wordpack(
    texts: Sequence[str],
    pack,
    stopwords: Sequence[str] | None = None,
) -> list[PreprocessedDoc]:
    """Preprocess texts with a wordpack's own tokenizer.

    Tokens are the pack's lookup keys (surface forms for English, clitics
    split), not lemmas.
    ``raw`` is the text after the reader's NFC normalisation and character
    fold, and every offset in ``sent_char_spans`` indexes it.

    Parameters
    ----------
    texts : sequence of str
        Input documents, one string per document.
    pack : WordPack
        Open pack (``ssdiff.wordpack.reader.WordPack``).
    stopwords : sequence of str, optional
        Lowercase stopwords; a token is dropped when its key, lowercased by
        the pack's rule, is in this list.
    """
    stopset = set(stopwords or [])

    def extract(strings):
        for s in strings:
            t = pack.tokenize(s)
            yield t.text, _extract_from_tokens(t, stopset, pack.chars.lower)

    return _assemble_preprocessed(texts, extract)


def build_docs_from_preprocessed(
    pre_docs: list[PreprocessedDoc],
) -> list[list[str]]:
    """Extract flat token lists from preprocessed docs.

    Bridges the gap between :func:`preprocess_texts_wordpack` output and
    downstream consumers that expect plain ``list[list[str]]`` token lists.

    Parameters
    ----------
    pre_docs : list[PreprocessedDoc]
        Output of :func:`preprocess_texts_wordpack`.

    Returns
    -------
    list[list[str]]
        ``doc_lemmas`` of each document.
    """
    return [p.doc_lemmas for p in pre_docs]
