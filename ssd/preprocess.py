# ===== words_apart_orig/preprocess.py =====
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence
import re
import requests
from typing import Optional
import spacy

# --- stopwords (same list/source as your app) ---
def load_stopwords(lang: str = "pl", *, lowercase: bool = True, timeout: float = 5.0) -> List[str]:
    """
    Load stopwords.
    - For Polish ("pl"): fetch the original list from GitHub (bieli/stopwords).
    - For other languages: use spaCy's built-in stopwords (spacy.blank(lang).Defaults.stop_words).
    - Raises on failure with clear messages. No prints unless an exception is raised.
    """
    lang = (lang or "pl").strip().lower()

    if lang == "pl":
        url = "https://raw.githubusercontent.com/bieli/stopwords/master/polish.stopwords.txt"
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            raise OSError(
                "Failed to fetch Polish stopwords from GitHub "
                "(bieli/stopwords). Check your connection or the URL."
            ) from e
        words = [s.strip() for s in r.text.splitlines() if s.strip()]
        if not words:
            raise ValueError("Fetched Polish stopword list is empty.")
        return [w.lower() for w in words] if lowercase else words

    # Non-Polish: use spaCy
    try:
        import spacy
    except ImportError as e:
        raise ImportError(
            "spaCy is required to load stopwords for languages other than Polish. "
            "Install it: pip install spacy"
        ) from e

    try:
        nlp = spacy.blank(lang)
    except Exception as e:
        raise LookupError(
            f"spaCy does not recognize language code '{lang}'. "
            "Use an ISO code spaCy supports (e.g., 'en', 'de', 'es', 'fr', 'it', ...)."
        ) from e

    sw = getattr(nlp.Defaults, "stop_words", None)
    if not sw:
        raise LookupError(
            f"No stopwords available in spaCy for language '{lang}'."
        )

    words = list(sw)
    return [w.lower() for w in words] if lowercase else words



def load_spacy(model: Optional[str]) -> Optional["spacy.language.Language"]:
    """Load a spaCy model with friendly feedback.
    - Prints status messages.
    - Tries to download once if loading fails.
    - Returns the nlp object on success, else None.
    """
    if not model or not isinstance(model, str) or not model.strip():
        print("✖ Please provide a non-empty spaCy model name (e.g., 'pl_core_news_lg').")
        return None

    # 1) Try to load directly
    try:
        nlp = spacy.load(model, disable=["ner"])
        return nlp
    except Exception as e:
        print(f"… Could not load '{model}' directly ({e}). Attempting download…")

    # 2) Try to download via cli (may print its own messages)
    try:
        from spacy.cli import download
        try:
            download(model)   # may print ✘/✓ lines; may raise SystemExit
        except SystemExit:
            # spaCy’s CLI sometimes exits instead of raising a normal error
            print(f"✖ No compatible package found for '{model}'. Check the name and spaCy version.")
            return None
    except Exception as e:
        print(f"✖ Failed to download '{model}': {e}")
        return None

    # 3) Try loading again after (claimed) download
    try:
        nlp = spacy.load(model, disable=["ner"])
        return nlp
    except Exception as e:
        print(f"✖ Downloaded (or tried) but still cannot load '{model}': {e}")
        print("   Make sure the model name is valid and matches your spaCy version.")
        print("   See: https://spacy.io/models")
        return None


_URL = re.compile(r"https?://\S+")
_AT  = re.compile(r"@\S+")
def _keep_token(tok, stopset: set[str]) -> bool:
    if tok.is_space or tok.is_punct or tok.is_quote or tok.is_currency:
        return False
    if _URL.match(tok.text) or _AT.match(tok.text):
        return False
    if tok.is_digit:
        return False
    # keep letters; drop if lemma is in stopwords
    lem = tok.lemma_.lower()
    if not lem:
        return False
    if lem in stopset:
        return False
    return True

@dataclass
class PreprocessedDoc:
    raw: str
    sents_surface: List[str]            # original sentence texts
    sents_lemmas: List[List[str]]       # lemma tokens per sentence
    doc_lemmas: List[str]               # flattened lemma tokens (for ssd)
    sent_char_spans: List[tuple[int,int]]  # (start,end) in raw
    token_to_sent: List[int]            # map from lemma index in doc_lemmas -> sent index

def preprocess_texts(
    texts: Sequence[str],
    nlp=None,
    stopwords: Optional[Sequence[str]] = None,
    batch_size: int = 64
) -> List[PreprocessedDoc]:
    if nlp is None:
        nlp = load_spacy()
    stopset = set(stopwords)
    out: List[PreprocessedDoc] = []

    for doc in nlp.pipe(texts, batch_size=batch_size):
        raw = doc.text
        s_surface, s_lemmas, s_spans = [], [], []
        doc_lemmas, token_to_sent = [], []

        for sent in doc.sents:
            stext = sent.text
            s_surface.append(stext)
            s_spans.append((sent.start_char, sent.end_char))

            lemmas = []
            for tok in sent:
                if _keep_token(tok, stopset):
                    lemmas.append(tok.lemma_.lower())

            s_lemmas.append(lemmas)
            # extend flattened
            start_flat_index = len(doc_lemmas)
            doc_lemmas.extend(lemmas)
            token_to_sent.extend([len(s_surface)-1] * (len(doc_lemmas) - start_flat_index))

        out.append(PreprocessedDoc(
            raw=raw,
            sents_surface=s_surface,
            sents_lemmas=s_lemmas,
            doc_lemmas=doc_lemmas,
            sent_char_spans=s_spans,
            token_to_sent=token_to_sent
        ))
    return out


def build_docs_from_preprocessed(pre_docs: List[PreprocessedDoc]) -> List[List[str]]:
    """Return lemmatized token lists (one per doc) for ssd."""
    return [P.doc_lemmas for P in pre_docs]