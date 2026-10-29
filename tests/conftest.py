"""Shared fixtures for ssdiff test suite.

All fixtures use tiny in-memory Embeddings — no real embeddings,
network access, or spaCy downloads needed.
"""

from __future__ import annotations

import os
import re

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
def ssd_instance(pack_emb, pack_df, pack_lexicon):
    """SSD text on the test pack: strings, so ``pre_docs`` (snippets) exist."""
    from ssdiff.ssd import SSD
    return SSD(pack_df["text"], embeddings=pack_emb, lexicon=pack_lexicon)


@pytest.fixture
def two(pack_emb, pack_df, pack_lexicon):
    """Two texts over the same rows: ``essay`` from strings, ``diary`` from token lists."""
    from ssdiff.ssd import SSD
    essay = SSD(pack_df["text"], embeddings=pack_emb, lexicon=pack_lexicon)
    diary = SSD(essay.docs[::-1], embeddings=pack_emb, lexicon=pack_lexicon)
    return essay, diary


@pytest.fixture(scope="session")
def pls_result(ssd_instance, pack_df):
    """Fitted PLSResult from SSD.fit_pls()."""
    return ssd_instance.fit_pls(pack_df["y"], k=2, n_splits=20, random_state=42)


@pytest.fixture(scope="session")
def pcaols_result(ssd_instance, pack_df):
    """Fitted PCAOLSResult from SSD.fit_regression()."""
    return ssd_instance.fit_regression(pack_df["y"], fixed_k=3)


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

TEST_PACK = os.path.join(os.path.dirname(__file__), "fixtures", "wordpack", "test.wordpack")

PACK_LEXICON = ["dog", "cat", "people"]


@pytest.fixture(scope="session")
def wordpack_path() -> str:
    """Synthetic wordpack: 48 keys, D = 16, name ``test`` version ``1``."""
    return TEST_PACK


@pytest.fixture(scope="session")
def pack_emb() -> Embeddings:
    return Embeddings.load(TEST_PACK)


@pytest.fixture(scope="session")
def pack_df(pack_emb) -> pd.DataFrame:
    """240 texts over the test pack's words, with columns ``text``, ``y``, ``y2``, ``group``, ``cov``.

    ``y`` and ``y2`` tilt each text's word draws along the pack's ``blue - green``
    and ``swim - like`` directions, so the fits have a signal to find.
    """
    rng = np.random.default_rng(2137)
    stop = set(pack_emb.stopwords)
    vocab = [w for w in pack_emb.index_to_key
             if re.fullmatch(r"[a-z]+", w) and w not in stop and w not in PACK_LEXICON]
    V = np.vstack([pack_emb[w] for w in vocab]).astype(np.float64)

    def projection(a, b):
        p = V @ (pack_emb[a] - pack_emb[b])
        return (p - p.mean()) / p.std()

    p = np.column_stack([projection("blue", "green"), projection("swim", "like")])
    z = rng.standard_normal((240, 2))
    texts = []
    for zi in z:
        logits = 0.3 * (p @ zi)
        prob = np.exp(logits - logits.max())
        prob /= prob.sum()
        sentences = []
        for _ in range(rng.integers(3, 7)):
            sent = list(rng.choice(vocab, size=rng.integers(6, 11), p=prob))
            if rng.random() < 0.8:
                sent.insert(int(rng.integers(0, len(sent) + 1)), str(rng.choice(PACK_LEXICON)))
            sentences.append(" ".join(sent).capitalize() + ".")
        texts.append(" ".join(sentences))
    y, y2 = z[:, 0], z[:, 1]
    cut = np.quantile(y + 0.5 * rng.standard_normal(len(y)), [1 / 3, 2 / 3])
    group = np.where(y < cut[0], "low", np.where(y < cut[1], "mid", "high"))
    cov = 0.3 * y + rng.standard_normal(len(y))
    return pd.DataFrame({"text": texts, "y": y, "y2": y2, "group": group, "cov": cov})


@pytest.fixture(scope="session")
def pack_lexicon() -> list[str]:
    return list(PACK_LEXICON)
