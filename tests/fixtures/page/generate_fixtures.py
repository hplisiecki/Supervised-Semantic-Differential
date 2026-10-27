"""Regenerate the example results-page JSON files, one per result kind (and the two-text regression and the contrast).

Offline, run-once script — NOT collected by pytest. Each file is
``page_data(...)`` of a fit on the parity wordpack and texts (so snippets and
document texts exist), with ``created``
fixed so reruns only differ where the output changed.

``tests/results/test_page.py`` checks that the files still have the shape
``page_data`` writes now; the R package (ssdiff) checks its own output against them.

Run from the ssdiff folder:
    python -m tests.fixtures.page.generate_fixtures
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import ssdiff
from ssdiff import SSD, Embeddings
from ssdiff.results.page import page_data

PARITY = Path(__file__).parents[1] / "r_parity"
OUT = Path(__file__).parent


def main() -> None:
    texts = json.loads((PARITY / "texts.json").read_text(encoding="utf-8"))
    df = pd.DataFrame({k: v for k, v in texts.items() if k != "lexicon"})
    ssd = SSD(df["text"], embeddings=Embeddings.load(str(PARITY / "parity.wordpack")), lexicon=texts["lexicon"])

    diary = SSD(ssd.docs[::-1], embeddings=ssd.embeddings, lexicon=texts["lexicon"])  # token lists: no snippets
    fits = {
        "pls": ssd.fit_pls(df["y"], k=2, n_splits=20, random_state=42),
        "pca_ols": ssd.fit_regression(df["y"], fixed_k=None, k_min=2, k_max=6, k_step=1),
        "groups": ssd.fit_groups(df["group"], n_perm=50, random_state=42),
        "regression": ssdiff.fit_regression(
            "y ~ ssd(essay) + group*ssd(diary)", df, essay=ssd, diary=diary,
            k_min=2, k_max=6, k_step=1),
        "multipls": ssd.fit_multipls(["y", "y2"], data=df, ssd_gradients=2, n_perm=19, n_splits=5,
                                     random_state=0),
    }
    fits["contrast"] = ssdiff.contrast(fits["pls"].gradient, fits["pca_ols"].gradient, ssd.embeddings,
                                       labels=("pls", "pca_ols"))
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
