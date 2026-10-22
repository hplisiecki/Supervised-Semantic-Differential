"""Regenerate the reference files that the R package (ssdiff) checks itself against.

Offline, run-once script — NOT collected by pytest. R copies the files into its
own tests and compares its results with them.

What it writes into this directory:
  - ``parity.wordpack`` — a small English pack built from
                          the GloVe slice ``examples/data/glove_300_en_top20k_ratings.bin``
                          (licence PDDL-1.0): its ``N_WORDS`` most frequent
                          lowercase alphabetic words, PCA'd to ``DIMS`` dims.
  - ``texts.json``      — synthetic texts drawn from the pack's vocabulary with a
                          planted signal, the columns ``y``, ``y2`` (``Y`` is
                          both), ``group`` (three labels) and ``cov``, and the lexicon.
  - ``parity/<case>.json`` — one file per fit or lexicon call (``CASES``).

Planted signal: two latent outcomes z1, z2 ~ N(0, 1) per document. Each
context word ``w`` is drawn with probability proportional to
``exp(TILT * (z1 * p1(w) + z2 * p2(w)))``, where p1 and p2 are the word's
standardized projections on the pack's ``good - bad`` and ``money - love``
directions. ``y = z1``, ``y2 = z2``.

Needs plskit and an editable install of the ``wordpack_compiler`` package: the
English language files are read from its source checkout. The committed pack is
the reference; the script is for maintainers.

Run from the ssdiff folder:
    python tests/fixtures/r_parity/generate_fixtures.py
"""

from __future__ import annotations

import json
import re
import tempfile
import warnings
import zipfile
from pathlib import Path

import numpy as np
import wordpack_compiler
from wordpack_compiler.build import build

from ssdiff import SSD, Embeddings
from ssdiff.results.page import _clean, _view_table
from ssdiff.utils.math import pca_fit_transform

HERE = Path(__file__).parent
SSDIFF_ROOT = HERE.parents[2]
SOURCE = SSDIFF_ROOT / "examples" / "data" / "glove_300_en_top20k_ratings.bin"
LANGUAGE_DIR = Path(wordpack_compiler.__file__).parents[1] / "languages" / "en"
PACK = HERE / "parity.wordpack"
TEXTS = HERE / "texts.json"
OUT = HERE / "parity"

# --- Pack ------------------------------------------------------------------
N_WORDS = 10_000
DIMS = 100
MAX_PACK_BYTES = 1_500_000

# --- Texts -----------------------------------------------------------------
SEED = 2137
N_DOCS = 240
LEXICON = ["people", "life", "work", "time", "family"]
CONTENT_RANKS = (100, 4000)  # context words come from these frequency ranks of the pack
# Weak on purpose: a strong signal puts every permutation p-value at its floor
# 1 / (n_perm + 1), which any permutation sequence reproduces, so R's generator
# would go untested.
TILT = 0.1
N_TOP_WORDS = 20
N_VECTOR_ROWS = 5
N_TOKEN_TEXTS = 3

# Each case: (name, ssd y, call, positional args, keyword args). Every fit kind
# runs with default arguments plus one non-default case.
CASES = [
    ("pls_default", "y", "fit_pls", [], {}),
    ("pls_k2", "y", "fit_pls", [], {"k": 2}),
    ("pls_gradients3", "y", "fit_pls", [], {"ssd_gradients": 3}),
    ("pls_formula", "y", "fit_pls", ["~ cov"], {}),
    ("multipls_default", ["y", "y2"], "fit_multipls", [], {}),
    ("multipls_rotate_outcomes", ["y", "y2"], "fit_multipls", [], {"rotate": "outcomes"}),
    ("regression_default", "y", "fit_regression", [], {}),
    ("regression_fixed_k10", "y", "fit_regression", [], {"fixed_k": 10}),
    ("regression_formula", "y", "fit_regression", ["~ cov"], {}),
    ("groups_default", "group", "fit_groups", [], {}),
    ("groups_median_split", "y", "fit_groups", [], {"median_split": True}),
]


def build_pack() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the .bin-without-spaCy notice
        src = Embeddings.load(str(SOURCE))
    words = [w for w in src.index_to_key if re.fullmatch(r"[a-z]+", w)][:N_WORDS]
    rows, _, _ = pca_fit_transform(np.asarray(src.vectors[[src.key_to_index[w] for w in words]],
                                              dtype=np.float64), DIMS)
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        zip_path = work / "source.zip"
        lines = [w + " " + " ".join(f"{v:.6f}" for v in r) for w, r in zip(words, rows)]
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("vectors.txt", "\n".join(lines) + "\n")
        recipe = work / "recipe.toml"
        recipe.write_text(
            'name = "en-parity"\n'
            'description = "Test pack for the R package (ssdiff) parity tests: GloVe slice, PCA-reduced."\n'
            'url = "examples/data/glove_300_en_top20k_ratings.bin"\n'
            f"dims = {DIMS}\n"
            'license = "PDDL-1.0"\n'
            'citation = "GloVe word vectors (Stanford NLP), English 300d slice"\n',
            encoding="utf-8",
        )
        build(recipe, LANGUAGE_DIR, zip_path, PACK, 1)
    size = PACK.stat().st_size
    if size > MAX_PACK_BYTES:
        raise RuntimeError(f"{PACK.name} is {size:,} bytes, over {MAX_PACK_BYTES:,}; lower N_WORDS or DIMS")
    print(f"{PACK.name}: {len(words):,} words, {DIMS} dims, {size:,} bytes")


def make_texts(emb) -> dict:
    rng = np.random.default_rng(SEED)
    stop = set(emb.stopwords)
    lo, hi = CONTENT_RANKS
    vocab = [w for w in emb.index_to_key[lo:hi] if w not in stop and w not in LEXICON]
    V = np.vstack([emb[w] for w in vocab]).astype(np.float64)

    def projection(a, b):
        p = V @ (emb[a] - emb[b])
        return (p - p.mean()) / p.std()

    p1, p2 = projection("good", "bad"), projection("money", "love")
    z = rng.standard_normal((N_DOCS, 2))
    texts = []
    for z1, z2 in z:
        logits = TILT * (z1 * p1 + z2 * p2)
        prob = np.exp(logits - logits.max())
        prob /= prob.sum()
        sentences = []
        for _ in range(rng.integers(3, 7)):
            sent = list(rng.choice(vocab, size=rng.integers(6, 11), p=prob))
            if rng.random() < 0.8:
                sent.insert(int(rng.integers(0, len(sent) + 1)), str(rng.choice(LEXICON)))
            sentences.append(" ".join(sent).capitalize() + ".")
        texts.append(" ".join(sentences))
    y, y2 = z[:, 0], z[:, 1]
    cut = np.quantile(y + 0.5 * rng.standard_normal(N_DOCS), [1 / 3, 2 / 3])
    group = np.where(y < cut[0], "low", np.where(y < cut[1], "mid", "high"))
    cov = 0.3 * y + rng.standard_normal(N_DOCS)
    return {"lexicon": LEXICON, "text": texts, "y": y.tolist(), "y2": y2.tolist(),
            "group": group.tolist(), "cov": cov.tolist()}


def _table_or_none(view):
    return None if view is None else _view_table(view)


def _unit(leaf) -> dict:
    return {
        "beta": _clean(np.asarray(leaf.beta)),
        "words": {"pos": _view_table(leaf.words.pos(N_TOP_WORDS)),
                  "neg": _view_table(leaf.words.neg(N_TOP_WORDS))},
        "cluster_words": _view_table(leaf.clusters.words),
    }


def _units(result) -> dict:
    """The main fit (when ``beta`` is one vector) and every leaf: gradients, dimensions or pairs."""
    units = {} if isinstance(result.beta, dict) else {"main": _unit(result)}
    for key in result.keys() if hasattr(result, "keys") else []:
        units[key if isinstance(key, str) else "_".join(map(str, key))] = _unit(result[key])
    return units


def fit_case(ssd, df, name, y, call, args, kwargs) -> dict:
    outcome = f"{y} {args[0]}" if args else y
    result = getattr(ssd, call)(outcome, data=df, **kwargs)
    x, has_vector, _ = ssd._vectors()
    kept = np.flatnonzero(has_vector)
    out = {
        "case": name,
        "ssd": {"y": y, "lexicon": LEXICON},
        "call": call,
        "args": list(args),
        "kwargs": kwargs,
        "tokens": ssd.docs[:N_TOKEN_TEXTS],
        "n_kept": ssd.n_kept,
        "doc_vectors": {"doc_id": kept[:N_VECTOR_ROWS].tolist(), "x": _clean(x[:N_VECTOR_ROWS])},
    }
    for table in ("stats", "test", "fit_info", "pairs", "pls_info", "sweep"):
        view = getattr(result, table, None)
        if view is not None:
            out[table] = _view_table(view)
    out["units"] = _units(result)
    return out


def lexicon_cases(ssd, df) -> list[dict]:
    y = np.asarray(df["y"])
    suggest = ssd.suggest_lexicon(y)
    evaluate = ssd.evaluate_lexicon(y, LEXICON)
    return [
        {"case": "suggest_lexicon", "call": "suggest_lexicon", "args": ["y"], "kwargs": {},
         "stats": _view_table(suggest.stats), "suggestions": _view_table(suggest.suggestions)},
        {"case": "evaluate_lexicon", "call": "evaluate_lexicon", "args": ["y", LEXICON], "kwargs": {},
         "stats": _view_table(evaluate.stats), "suggestions": _view_table(evaluate.suggestions),
         "summary": _table_or_none(evaluate.summary)},
    ]


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    import pandas as pd

    build_pack()
    emb = Embeddings.load(str(PACK))
    data = make_texts(emb)
    _write(TEXTS, data)
    df = pd.DataFrame({k: v for k, v in data.items() if k != "lexicon"})
    ssd = SSD(emb, df["text"], LEXICON)

    OUT.mkdir(exist_ok=True)
    for name, y, call, args, kwargs in CASES:
        _write(OUT / f"{name}.json", fit_case(ssd, df, name, y, call, args, kwargs))
        print(f"parity/{name}.json")
    for case in lexicon_cases(ssd, df):
        _write(OUT / f"{case['case']}.json", case)
        print(f"parity/{case['case']}.json")


if __name__ == "__main__":
    main()
