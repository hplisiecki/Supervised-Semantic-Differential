"""Regression tests: SSDLite PLS against known GloVe benchmark results.

Loads real GloVe embeddings + Kalibra datasets and verifies that SSD.fit_pls()
reproduces the R², adj_R², coverage, and top-20 words from the stored
benchmark results in Benchmarks/results_glove.json.

Requires local files:
  - Models/glove_800_normalized.kv
  - Corpuses/Kalibra/kalibra_*.csv
  - Benchmarks/results_glove.json

Run with:  pytest tests/test_regression_glove.py -v
Skip with: pytest -m "not slow"
"""

from __future__ import annotations

import json
import os

import pytest

pd = pytest.importorskip("pandas")

from ssdlite import SSD, Corpus, Embeddings  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SSD_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, ".."))

EMBEDDINGS_PATH = os.path.join(SSD_ROOT, "Models", "glove_800_normalized.kv")
DATA_DIR = os.path.join(SSD_ROOT, "Corpuses", "Kalibra")
RESULTS_PATH = os.path.join(SSD_ROOT, "Benchmarks", "results_glove.json")

WINDOW = 3
SIF_A = 1e-3

DATASETS = {
    "polityka":    ("polityka_closed", "polityka_open", ["polityka", "wpływ", "wybory"]),
    "klimat":      ("klimat_closed", "klimat_open", ["zmiana", "klimatyczny", "klimat"]),
    "szczepienie": ("szczepienie_closed", "szczepienie_open", ["szczepienie", "szczepić", "szczepionka"]),
    "zaufanie":    ("zaufanie_closed", "zaufanie_open", ["człowiek", "osoba", "obcy", "nieznajomy"]),
    "naukowcy":    ("naukowcy_closed", "naukowcy_open", ["nauka", "naukowiec", "badanie"]),
    "zdrowie":     ("zdrowie_closed", "zdrowie_open", ["zdrowie", "czuć", "samopoczucie"]),
    "imigrant":    ("imigrant_closed", "imigrant_open", ["imigrant", "imigracja", "cudzoziemiec", "migrant"]),
}

pytestmark = [pytest.mark.slow, pytest.mark.local]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _skip_missing(*paths):
    for p in paths:
        if not os.path.exists(p):
            pytest.skip(f"File not found: {p}")


@pytest.fixture(scope="module")
def glove_embeddings():
    _skip_missing(EMBEDDINGS_PATH)
    return Embeddings.load(EMBEDDINGS_PATH)


@pytest.fixture(scope="module")
def benchmark_pls():
    """Load PLS entries from the GloVe benchmark results."""
    _skip_missing(RESULTS_PATH)
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        all_rows = json.load(f)
    return {r["dataset"]: r for r in all_rows if r["version"] == "pls"}


def _load_dataset(theme):
    """Load a Kalibra dataset, returning (texts, y)."""
    y_col, text_col, _ = DATASETS[theme]
    csv_path = os.path.join(DATA_DIR, f"kalibra_{theme}.csv")
    _skip_missing(csv_path)
    df = pd.read_csv(csv_path, encoding="utf-8")
    y_series = pd.to_numeric(df[y_col], errors="coerce")
    mask = ~y_series.isna()
    texts = df.loc[mask, text_col].fillna("").astype(str).tolist()
    y = y_series[mask].to_numpy()
    return texts, y


def _run_pls(glove_embeddings, theme):
    """Run SSD.fit_pls() for a given dataset, return (result, expected_row)."""
    _, _, lexicon = DATASETS[theme]
    texts, y = _load_dataset(theme)
    corpus = Corpus(texts, lang="pl")
    ssd_obj = SSD(glove_embeddings, corpus, y, lexicon,
                  window=WINDOW, sif_a=SIF_A)
    return ssd_obj.fit_pls()


# ---------------------------------------------------------------------------
# Parametrized regression tests
# ---------------------------------------------------------------------------

@pytest.fixture(
    scope="module",
    params=list(DATASETS.keys()),
)
def pls_run(request, glove_embeddings, benchmark_pls):
    """Run PLS for one dataset and return (result, expected)."""
    theme = request.param
    if theme not in benchmark_pls:
        pytest.skip(f"No benchmark for {theme}")
    result = _run_pls(glove_embeddings, theme)
    expected = benchmark_pls[theme]
    return result, expected, theme


class TestPLSRegression:
    """Verify SSD.fit_pls() exactly reproduces benchmark numeric results."""

    def test_r2(self, pls_run):
        result, expected, theme = pls_run
        assert result.r2 == pytest.approx(expected["R2"], abs=1e-10), (
            f"{theme}: R2 {result.r2:.10f} != expected {expected['R2']:.10f}"
        )

    def test_adj_r2(self, pls_run):
        result, expected, theme = pls_run
        assert result.r2_adj == pytest.approx(expected["adj_R2"], abs=1e-10), (
            f"{theme}: adj_R2 {result.r2_adj:.10f} != expected {expected['adj_R2']:.10f}"
        )

    def test_n_observations(self, pls_run):
        result, expected, theme = pls_run
        assert result.n_raw == expected["N"], (
            f"{theme}: N {result.n_raw} != expected {expected['N']}"
        )

    def test_coverage(self, pls_run):
        result, expected, theme = pls_run
        actual_cov = round(result.n_kept / result.n_raw, 4) if result.n_raw > 0 else 0
        assert actual_cov == expected["coverage"], (
            f"{theme}: coverage {actual_cov} != expected {expected['coverage']}"
        )

    def test_n_components(self, pls_run):
        result, expected, theme = pls_run
        assert result.n_components == expected["K"], (
            f"{theme}: K {result.n_components} != expected {expected['K']}"
        )


class TestTopWordsRegression:
    """Verify top-20 words exactly match the benchmark."""

    def test_pos_words_exact(self, pls_run):
        result, expected, theme = pls_run
        tw = result.top_words(n=20)
        actual_pos = [w["word"] for w in tw if w["side"] == "pos"]
        expected_pos = expected["pos_words"]
        assert actual_pos == expected_pos, (
            f"{theme} pos words differ.\n"
            f"  actual:   {actual_pos}\n"
            f"  expected: {expected_pos}"
        )

    def test_neg_words_exact(self, pls_run):
        result, expected, theme = pls_run
        tw = result.top_words(n=20)
        actual_neg = [w["word"] for w in tw if w["side"] == "neg"]
        expected_neg = expected["neg_words"]
        assert actual_neg == expected_neg, (
            f"{theme} neg words differ.\n"
            f"  actual:   {actual_neg}\n"
            f"  expected: {expected_neg}"
        )
