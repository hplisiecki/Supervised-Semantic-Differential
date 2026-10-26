"""Shared fixtures for ssdiff test suite.

All fixtures use tiny in-memory Embeddings — no real embeddings,
network access, or spaCy downloads needed.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

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
    emb = Embeddings(words, mat)
    emb.l2_normalized = True  # rows are unit by construction
    return emb


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
# Fitted result fixtures (for test_results.py)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ssd_instance(parity_emb, parity_df, parity_lexicon):
    """SSD text on the parity wordpack: strings, so ``pre_docs`` (snippets) exist."""
    from ssdiff.ssd import SSD
    return SSD(parity_df["text"], embeddings=parity_emb, lexicon=parity_lexicon)


@pytest.fixture
def two(parity_emb, parity_df, parity_lexicon):
    """Two texts over the parity rows: ``essay`` from strings, ``diary`` from token lists."""
    from ssdiff.ssd import SSD
    essay = SSD(parity_df["text"], embeddings=parity_emb, lexicon=parity_lexicon)
    diary = SSD(essay.docs[::-1], embeddings=parity_emb, lexicon=parity_lexicon)
    return essay, diary


@pytest.fixture(scope="session")
def pls_result(ssd_instance, parity_df):
    """Fitted PLSResult from SSD.fit_pls()."""
    return ssd_instance.fit_pls(parity_df["y"], k=2, n_splits=20, random_state=42)


@pytest.fixture(scope="session")
def pcaols_result(ssd_instance, parity_df):
    """Fitted PCAOLSResult from SSD.fit_regression()."""
    return ssd_instance.fit_regression(parity_df["y"], fixed_k=3)


@pytest.fixture(scope="session")
def pcaols_result_sweep(tiny_kv, large_docs, large_y, lexicon):
    """Fitted PCAOLSResult with sweep (fixed_k=None)."""
    from ssdiff.ssd import SSD
    ssd = SSD(large_docs, embeddings=tiny_kv, lexicon=lexicon)
    return ssd.fit_regression(large_y, fixed_k=None, k_min=2, k_max=6, k_step=1)


@pytest.fixture(scope="session")
def group_result_2g(tiny_kv, large_docs, large_groups_2, lexicon):
    """Fitted GroupResult with 2 groups."""
    from ssdiff.ssd import SSD
    ssd = SSD(large_docs, embeddings=tiny_kv, lexicon=lexicon)
    return ssd.fit_groups(large_groups_2, n_perm=50, random_state=42)


@pytest.fixture(scope="session")
def group_result_3g(tiny_kv, large_docs_3x20, large_groups_3x20, lexicon):
    """Fitted GroupResult with 3 groups."""
    from ssdiff.ssd import SSD
    ssd = SSD(large_docs_3x20, embeddings=tiny_kv, lexicon=lexicon)
    return ssd.fit_groups(large_groups_3x20, n_perm=50, random_state=42)


# ---------------------------------------------------------------------------
# Wordpack fixture
# ---------------------------------------------------------------------------

PARITY_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "r_parity")

WORDPACK_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "wordpack", "fixture.wordpack")


@pytest.fixture(scope="session")
def wordpack_path() -> str:
    """Synthetic wordpack: 48 keys, D = 16, name ``en-fixture`` version ``1``."""
    return WORDPACK_FIXTURE


@pytest.fixture(scope="session")
def parity_emb() -> Embeddings:
    """English wordpack (10,000 words, 100 dims) shared with the R parity fixtures."""
    return Embeddings.load(os.path.join(PARITY_DIR, "parity.wordpack"))


def _parity_texts() -> dict:
    with open(os.path.join(PARITY_DIR, "texts.json")) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def parity_df() -> pd.DataFrame:
    """240 texts with columns ``text``, ``y``, ``y2``, ``group``, ``cov``."""
    return pd.DataFrame({k: v for k, v in _parity_texts().items() if k != "lexicon"})


@pytest.fixture(scope="session")
def parity_lexicon() -> list[str]:
    return _parity_texts()["lexicon"]
