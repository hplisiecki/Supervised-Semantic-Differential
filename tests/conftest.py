"""Shared fixtures for ssdiff test suite.

All fixtures use tiny in-memory Embeddings — no real embeddings,
network access, or spaCy downloads needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from ssdiff.corpus import Corpus
from ssdiff.embeddings import Embeddings
from ssdiff.utils.text import PreprocessedDoc

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

VOCAB_20 = [
    "kraj", "narod", "panstwo",                     # seeds (lexicon)
    "piekny", "silny", "zly", "dobry",              # context words
    "wielki", "maly", "stary", "nowy",
    "dom", "szkola", "praca", "miasto",
    "rzeka", "gora", "las",
    "ABC123", "Warszawa",                            # "bad" tokens for filter
]

VOCAB_50 = VOCAB_20 + [
    "ludzie", "czas", "swiat", "dzien", "noc",
    "woda", "ogien", "ziemia", "niebo", "slonce",
    "droga", "pole", "morze", "kwiat", "drzewo",
    "kamien", "wiatr", "deszcz", "snieg", "chmura",
    "ptak", "ryba", "kon", "pies", "kot",
    "serce", "reka", "glowa", "oko", "usta",
]


# ---------------------------------------------------------------------------
# Embeddings helpers
# ---------------------------------------------------------------------------

def make_kv(words: list[str], dim: int, seed: int = 42) -> Embeddings:
    """Build a tiny Embeddings with unit-normalized random vectors."""
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(len(words), dim)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return Embeddings(words, mat)


@pytest.fixture(scope="session")
def tiny_kv() -> Embeddings:
    """20-word, 8-dimensional Embeddings (unit-normalized)."""
    return make_kv(VOCAB_20, dim=8, seed=42)


@pytest.fixture(scope="session")
def tiny_kv_large() -> Embeddings:
    """50-word, 10-dimensional Embeddings for clustering tests."""
    return make_kv(VOCAB_50, dim=10, seed=99)


# ---------------------------------------------------------------------------
# Lexicon / docs / outcome
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def lexicon() -> set[str]:
    return {"kraj", "narod", "panstwo"}


@pytest.fixture(scope="session")
def sample_docs() -> list[list[str]]:
    """8 documents, each containing >=1 seed from lexicon."""
    return [
        ["kraj", "piekny", "dom", "silny"],
        ["narod", "wielki", "miasto", "dobry"],
        ["panstwo", "silny", "szkola", "nowy"],
        ["kraj", "zly", "praca", "maly"],
        ["narod", "piekny", "rzeka", "stary"],
        ["panstwo", "dobry", "gora", "wielki"],
        ["kraj", "nowy", "las", "silny"],
        ["narod", "maly", "dom", "zly"],
    ]


@pytest.fixture(scope="session")
def sample_docs_no_seeds() -> list[list[str]]:
    """4 documents with NO lexicon seeds."""
    return [
        ["piekny", "dom", "silny"],
        ["wielki", "miasto", "dobry"],
        ["szkola", "nowy", "maly"],
        ["rzeka", "stary", "gora"],
    ]


@pytest.fixture(scope="session")
def sample_y() -> np.ndarray:
    return np.array([1.0, 1.2, 0.9, 0.8, 1.5, 1.1, 0.7, 1.3])


@pytest.fixture(scope="session")
def sample_y_with_nan() -> np.ndarray:
    return np.array([1.0, 1.2, 0.9, np.nan, 1.5, 1.1, 0.7, 1.3])


@pytest.fixture(scope="session")
def sample_groups() -> np.ndarray:
    return np.array(["A", "A", "A", "A", "B", "B", "B", "B"], dtype=object)


@pytest.fixture(scope="session")
def sample_groups_3() -> np.ndarray:
    return np.array(["X", "X", "X", "Y", "Y", "Z", "Z", "Z"], dtype=object)


# --- Larger fixtures for group tests (need >=20 docs per group) ---

def _make_large_docs(n: int, rng: np.random.Generator) -> list[list[str]]:
    """Generate n docs with random tokens, each containing >=1 seed."""
    seeds = ["kraj", "narod", "panstwo"]
    context = ["piekny", "silny", "zly", "dobry", "wielki", "maly",
               "stary", "nowy", "dom", "szkola", "praca", "miasto"]
    docs = []
    for i in range(n):
        seed = seeds[i % len(seeds)]
        ctx = list(rng.choice(context, size=3, replace=False))
        docs.append([seed] + ctx)
    return docs


@pytest.fixture(scope="session")
def large_docs() -> list[list[str]]:
    """50 docs for group tests."""
    rng = np.random.default_rng(42)
    return _make_large_docs(50, rng)


@pytest.fixture(scope="session")
def large_groups_2() -> np.ndarray:
    """2 groups, 25 each."""
    return np.array(["A"] * 25 + ["B"] * 25, dtype=object)


@pytest.fixture(scope="session")
def large_groups_3() -> np.ndarray:
    """3 groups: 20, 15, 15."""
    return np.array(["X"] * 20 + ["Y"] * 15 + ["Z"] * 15, dtype=object)


@pytest.fixture(scope="session")
def large_docs_3x20() -> list[list[str]]:
    """60 docs (3 groups x 20) for 3-group tests."""
    rng = np.random.default_rng(99)
    return _make_large_docs(60, rng)


@pytest.fixture(scope="session")
def large_groups_3x20() -> np.ndarray:
    """3 groups, 20 each."""
    return np.array(["X"] * 20 + ["Y"] * 20 + ["Z"] * 20, dtype=object)


@pytest.fixture(scope="session")
def large_y() -> np.ndarray:
    """50 continuous y values."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=5, scale=2, size=50)


# ---------------------------------------------------------------------------
# Preprocessed docs (manually built, no spaCy needed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_preprocessed_docs() -> list[PreprocessedDoc]:
    return [
        PreprocessedDoc(
            raw="Kraj jest piekny i silny.",
            sents_surface=["Kraj jest piekny i silny."],
            sents_lemmas=[["kraj", "piekny", "silny"]],
            doc_lemmas=["kraj", "piekny", "silny"],
            sent_char_spans=[(0, 26)],
            token_to_sent=[0, 0, 0],
            sents_kept_idx=[[0, 2, 4]],
        ),
        PreprocessedDoc(
            raw="Narod jest wielki. Miasto jest duze.",
            sents_surface=["Narod jest wielki.", "Miasto jest duze."],
            sents_lemmas=[["narod", "wielki"], ["miasto"]],
            doc_lemmas=["narod", "wielki", "miasto"],
            sent_char_spans=[(0, 18), (19, 35)],
            token_to_sent=[0, 0, 1],
            sents_kept_idx=[[0, 2], [0]],
        ),
        PreprocessedDoc(
            raw="Panstwo i szkola sa nowe.",
            sents_surface=["Panstwo i szkola sa nowe."],
            sents_lemmas=[["panstwo", "szkola", "nowy"]],
            doc_lemmas=["panstwo", "szkola", "nowy"],
            sent_char_spans=[(0, 24)],
            token_to_sent=[0, 0, 0],
            sents_kept_idx=[[0, 2, 4]],
        ),
        PreprocessedDoc(
            raw="Dom i praca w miescie.",
            sents_surface=["Dom i praca w miescie."],
            sents_lemmas=[["dom", "praca", "miasto"]],
            doc_lemmas=["dom", "praca", "miasto"],
            sent_char_spans=[(0, 22)],
            token_to_sent=[0, 0, 0],
            sents_kept_idx=[[0, 2, 4]],
        ),
    ]


# ---------------------------------------------------------------------------
# Mock spaCy objects (for preprocess tests without spaCy)
# ---------------------------------------------------------------------------

@dataclass
class MockToken:
    text: str
    lemma_: str
    is_space: bool = False
    is_punct: bool = False
    is_quote: bool = False
    is_currency: bool = False
    is_digit: bool = False


class MockSent:
    def __init__(self, tokens, text, start_char, end_char):
        self._tokens = tokens
        self.text = text
        self.start_char = start_char
        self.end_char = end_char

    def __iter__(self):
        return iter(self._tokens)


class MockDoc:
    def __init__(self, sents, text):
        self._sents = sents
        self.text = text

    @property
    def sents(self):
        return iter(self._sents)


def _fake_nlp_pipe(texts, batch_size=64, n_process=1):
    for t in texts:
        words = t.split()
        tokens = [MockToken(text=w, lemma_=w.lower()) for w in words]
        sent = MockSent(tokens, text=t, start_char=0, end_char=len(t))
        yield MockDoc(sents=[sent], text=t)


class FakeNlp:
    pipe_names = ["sentencizer"]

    def pipe(self, texts, batch_size=64, n_process=1):
        return _fake_nlp_pipe(texts, batch_size, n_process)


@pytest.fixture
def fake_nlp():
    return FakeNlp()


# ---------------------------------------------------------------------------
# Fitted result fixtures (for test_results.py)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ssd_instance(tiny_kv, sample_docs, sample_y, lexicon):
    """SSD data container (no fit yet)."""
    from ssdiff.ssd import SSD
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    return SSD(tiny_kv, corpus, sample_y, lexicon)


@pytest.fixture(scope="session")
def pls_result(ssd_instance):
    """Fitted PLSResult from SSD.fit_pls()."""
    return ssd_instance.fit_pls(n_components=2, p_method="perm", n_perm=50, random_state=42)


@pytest.fixture(scope="session")
def pcaols_result(ssd_instance):
    """Fitted PCAOLSResult from SSD.fit_ols()."""
    return ssd_instance.fit_ols(n_components=3)


@pytest.fixture(scope="session")
def pcaols_result_sweep(tiny_kv, large_docs, large_y, lexicon):
    """Fitted PCAOLSResult with sweep (n_components=None)."""
    from ssdiff.ssd import SSD
    corpus = Corpus(large_docs, pretokenized=True, lang="pl")
    ssd = SSD(tiny_kv, corpus, large_y, lexicon)
    return ssd.fit_ols(n_components=None, k_min=2, k_max=6, k_step=1)


@pytest.fixture(scope="session")
def group_result_2g(tiny_kv, large_docs, large_groups_2, lexicon):
    """Fitted GroupResult with 2 groups."""
    from ssdiff.ssd import SSD
    corpus = Corpus(large_docs, pretokenized=True, lang="pl")
    ssd = SSD(tiny_kv, corpus, large_groups_2, lexicon)
    return ssd.fit_groups(n_perm=50, random_state=42)


@pytest.fixture(scope="session")
def group_result_3g(tiny_kv, large_docs_3x20, large_groups_3x20, lexicon):
    """Fitted GroupResult with 3 groups."""
    from ssdiff.ssd import SSD
    corpus = Corpus(large_docs_3x20, pretokenized=True, lang="pl")
    ssd = SSD(tiny_kv, corpus, large_groups_3x20, lexicon)
    return ssd.fit_groups(n_perm=50, random_state=42)
