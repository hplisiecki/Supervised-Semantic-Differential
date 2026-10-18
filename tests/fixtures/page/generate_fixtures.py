"""Regenerate the example results-page JSON files, one per result kind.

Offline, run-once script — NOT collected by pytest. Each file is
``page_data(...)`` of a small synthetic fit (the test suite's tiny embedding
and FakeNlp corpora, so snippets and document texts exist), with ``created``
fixed so reruns only differ where the output changed.

``tests/results/test_page.py`` checks that the files still have the shape
``page_data`` writes now; the R package checks its own output against them.

Run from the ssdiff folder:
    python -m tests.fixtures.page.generate_fixtures
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ssdiff.corpus import Corpus
from ssdiff.results.page import page_data
from ssdiff.ssd import SSD
from tests.conftest import VOCAB_20, FakeNlp, _make_large_docs, make_kv

OUT = Path(__file__).parent
LEXICON = {"kraj", "narod", "panstwo"}


def _ssd(docs, y):
    corpus = Corpus([" ".join(d) for d in docs], nlp=FakeNlp(), lang="pl", stopwords=[])
    return SSD(make_kv(VOCAB_20, dim=8, seed=42), corpus, y, LEXICON)


def main() -> None:
    docs50 = _make_large_docs(50, np.random.default_rng(42))
    y50 = np.random.default_rng(42).normal(loc=5, scale=2, size=50)
    docs60 = _make_large_docs(60, np.random.default_rng(99))
    groups60 = np.array(["X"] * 20 + ["Y"] * 20 + ["Z"] * 20, dtype=object)
    y3 = np.random.default_rng(3).normal(size=(50, 3))

    fits = {
        "pls": _ssd(docs50, y50).fit_pls(k=2, n_splits=20, random_state=42),
        "pca_ols": _ssd(docs50, y50).fit_regression(fixed_k=None, k_min=2, k_max=6, k_step=1),
        "groups": _ssd(docs60, groups60).fit_groups(n_perm=50, random_state=42),
        "multipls": _ssd(docs50, y3).fit_multipls(ssd_gradients=2, n_perm=19, n_splits=5, random_state=0),
    }
    for kind, result in fits.items():
        # Small limits keep the committed files small; the shape is the same at any limit.
        data = page_data(result, title=f"Example ({kind})", snippets_per_side=10,
                         snippets_per_cluster=5, doc_texts=3)
        data["created"] = "2026-01-01T00:00:00Z"
        text = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=1)
        (OUT / f"{kind}.json").write_text(text + "\n", encoding="utf-8")
        print(f"{kind}.json: {len(text):,} chars")


if __name__ == "__main__":
    main()
