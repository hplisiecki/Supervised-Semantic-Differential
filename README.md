# Supervised Semantic Differential (SSD)

[![Tests](https://github.com/hplisiecki/Supervised-Semantic-Differential/workflows/Tests/badge.svg)](https://github.com/hplisiecki/Supervised-Semantic-Differential/actions)
[![PyPI](https://img.shields.io/pypi/v/ssdiff)](https://pypi.org/project/ssdiff/)
[![Python](https://img.shields.io/pypi/pyversions/ssdiff)](https://pypi.org/project/ssdiff/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://img.shields.io/badge/DOI-10.31234%2Fosf.io%2Fgvrsb__v1-blue)](https://doi.org/10.31234/osf.io/gvrsb_v1)

**SSD** lets you recover **interpretable semantic directions** related to specific concepts directly from open-ended text and relate them to **numeric outcomes**
(e.g., psychometric scales, judgments) or **categorical groups** (e.g., clinical diagnosis, experimental condition). It builds per-document concept vectors from **local contexts around seed words**,
learns a **semantic gradient (beta)** that best predicts the outcome, and then provides multiple interpretability layers:

- **Nearest neighbors** of each pole (+beta / -beta)
- **Clustering** of neighbors into themes
- **Text snippets**: top sentences whose local contexts align with each cluster centroid or the beta axis
- **Per-document scores** (cosine alignments) for further analysis
- **Cross-group comparisons** with permutation inference

The method has been presented in the following preprint:
https://doi.org/10.31234/osf.io/gvrsb_v1

> **No-code option:** a GUI desktop application for SSD is available at [hplisiecki/SSD_APP](https://github.com/hplisiecki/SSD_APP). It wraps this package into a point-and-click interface with a guided three-stage workflow, interactive lexicon builder, and APA-formatted export — pre-built binaries for Windows, Linux, and macOS are available with no Python installation required.

---

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Core Concepts](#core-concepts)
- [Word Embeddings](#word-embeddings)
- [Preprocessing (Corpus)](#preprocessing-corpus)
- [Lexicon Utilities](#lexicon-utilities)
- [Fitting SSD](#fitting-ssd)
  - [PCA + OLS](#pca--ols)
  - [PLS](#pls)
- [Choosing PCA Dimensionality (PCA Sweep)](#choosing-pca-dimensionality-pca-sweep)
- [Neighbors & Clustering](#neighbors--clustering)
- [Interpreting with Snippets](#interpreting-with-snippets)
- [Per-Document SSD Scores](#per-document-ssd-scores)
- [Cross-Group Comparison](#cross-group-comparison)
- [API Summary](#api-summary)
- [Citing & License](#citing--license)

---

## Installation

```bash
pip install ssdiff
```

Dependencies (installed automatically): `numpy`, `spacy`, `matplotlib`.

Optional:
- `gensim` — needed only for saving embeddings in `.kv` format: `pip install ssdiff[gensim]`

> Loading `.kv` files works without gensim (handled by an internal unpickler shim).

---

## Quickstart

Below is an end-to-end minimal example. Adjust paths and column names to your data.

```python
from ssdiff import Embeddings, Corpus, SSD
import numpy as np

# 1) Load and normalize embeddings
emb = Embeddings.load("path/to/embeddings.txt", verbose=True)
emb.normalize(l2=True, abtt_m=1)

# 2) Load your data
texts = [...]                          # list of raw text strings
scores = np.array([...])               # numeric outcome

# 3) Tokenize texts
corpus = Corpus(texts, lang="en")      # spaCy tokenization + lemmatization

# 4) Define a lexicon (tokens must match lemmatized forms)
lexicon = ["happy", "sad", "joy", "anger"]

# 5) Build SSD and fit
ssd = SSD(emb, corpus, y=scores, lexicon=lexicon)
result = ssd.fit_pls()                 # or ssd.fit_ols() for PCA+OLS

# 6) Inspect
print(result.summary())
result.top_words(n=20)
result.cluster_neighbors("pos", topn=100)
```

---

## Core Concepts

- **Seed lexicon**: a small set of tokens (lemmas) indicating the concept of interest (e.g., {climate, warming, change}).
- **Per-document vector**: SIF-weighted average of context vectors around each seed occurrence (+-3 tokens), then averaged across occurrences.
- **SSD fitting**: Learn a semantic gradient (beta) that best predicts the outcome y. Two backends are available:
  - **PLS**: Partial Least Squares regression directly in embedding space.
  - **PCA+OLS**: PCA dimensionality reduction followed by OLS regression (matches original SSD paper).
- **Interpretation**: nearest neighbors to +beta/-beta, clustering neighbors into themes, and showing original sentences whose local context aligns with centroids or beta.

---

## Word Embeddings

The method requires pre-trained word embeddings in one of the supported formats:

| Format | Extension | Notes |
|--------|-----------|-------|
| SSD native | `.ssdembed` | Fastest to load (pickle + `.vectors.npy` sidecar) |
| gensim KeyedVectors | `.kv` | Loads without gensim via internal shim |
| word2vec binary | `.bin` | Standard binary format |
| Text | `.txt`, `.vec` | One word per line + floats |
| Compressed | `.txt.gz`, `.vec.gz`, `.bin.gz` | Gzip-compressed versions of the above |

To capture semantic information without frequency-based artifacts, apply L2 normalization
and All-But-The-Top (ABTT) transformation:

```python
from ssdiff import Embeddings

emb = Embeddings.load("path/to/model.bin", verbose=True)
emb.normalize(l2=True, abtt_m=1)   # L2 + ABTT (remove top-1 PC)
```

Calling `normalize()` with no arguments applies both L2 and ABTT (m=1) by default.
Processing state is tracked — calling it again safely skips already-applied steps.

> **Tip:** Save normalized embeddings as `.ssdembed` to preserve both vectors and processing metadata (L2, ABTT state). Other formats (`.kv`, `.bin`, `.txt`) only store raw vectors.

The model is not included in the package and will differ depending on your language and domain.
Look for pre-trained static word embeddings in your language with good vocabulary coverage for your domain. GloVe and word2vec trained on large general corpora are a reliable starting point.

For Polish, the `nkjp+wiki-lemmas-all-300-cbow-hs.txt.gz` (no. 25) from the [Polish Word2Vec model list](https://dsmodels.nlp.ipipan.waw.pl) was found to work well.

---

## Preprocessing (Corpus)

The `Corpus` class encapsulates the full spaCy preprocessing pipeline — tokenization, lemmatization, and stopword removal.

```python
from ssdiff import Corpus

corpus = Corpus(texts, lang="en")      # auto-downloads spaCy model if needed
corpus.docs       # list[list[str]] — lemmatized tokens per document
corpus.pre_docs   # list[PreprocessedDoc] — for snippet extraction
corpus.n_texts    # number of documents
```

You can also pass a pre-loaded spaCy pipeline or pre-tokenized data:

```python
# Custom spaCy pipeline
import spacy
nlp = spacy.load("en_core_web_lg", disable=["ner"])
corpus = Corpus(texts, nlp=nlp)

# Pre-tokenized input
docs = [["happy", "day", "sunshine"], ["sad", "rain", "cold"], ...]
corpus = Corpus(docs, pretokenized=True, lang="en")
```

**Supported languages (20)**: ca, da, de, el, en, es, fr, hr, it, lt, mk, nb, nl, pl, pt, ro, ru, sl, sv, uk.

> CJK languages (Chinese, Japanese, Korean) are not included due to fundamental differences in tokenization and lemmatization. If you need CJK support, you can pass a custom spaCy pipeline via `nlp=` and pre-trained embeddings with matching vocabulary.

spaCy models for various languages can be found [here](https://spacy.io/models). To install a model manually:

```bash
python -m spacy download en_core_web_sm
```

---

## Lexicon Utilities

These helpers make lexicon selection transparent and data-driven (you can also hand-pick tokens).

### `suggest_lexicon(...)`

Rank tokens by balanced coverage with a mild penalty for strong association with the outcome.
The recommended way is via `Corpus.suggest_lexicon()`, which uses the already-lemmatized tokens:

```python
corpus = Corpus(texts, lang="en")
candidates = corpus.suggest_lexicon(y, top_k=30)
ssd = SSD(emb, corpus, y, lexicon=candidates)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `y` | `array-like` | -- | Outcome variable (numeric or categorical) |
| `top_k` | `int` | `30` | Maximum number of words to return |
| `min_docs` | `int` | `5` | Minimum document frequency |
| `n_bins` | `int` | `4` | Quantile bins for balanced coverage |
| `corr_cap` | `float` | `0.30` | Penalty threshold for outcome association |
| `var_type` | `str` | `"continuous"` | `"continuous"` or `"categorical"` |

The standalone function is also available for pre-tokenized data:

```python
from ssdiff.utils.lexicon import suggest_lexicon

words = suggest_lexicon(
    (token_lists, y),    # tuple of (list[list[str]], array-like)
    top_k=150,
    min_docs=5,
    var_type="continuous",
)
```

### `token_presence_stats(...)`

Per-token coverage and association diagnostics:

```python
from ssdiff.utils.lexicon import token_presence_stats

stats = token_presence_stats(
    texts, y, token="keyword",
    n_bins=4, corr_cap=0.30, verbose=False, var_type="continuous",
)
```

### `coverage_by_lexicon(...)`

Summary statistics for your chosen lexicon:

```python
from ssdiff.utils.lexicon import coverage_by_lexicon

summary, per_token = coverage_by_lexicon(
    (corpus.docs, y),
    lexicon=["word1", "word2", "word3"],
    n_bins=4, verbose=True, var_type="continuous",
)
```

Returns `(summary_dict, per_token_list)`:
- `summary`: `docs_any`, `cov_all`, `q1`, `q4`, `corr_any`, `hits_mean`, `hits_median`, `types_mean`, `types_median`
- `per_token`: list of dicts with `token`, `frequency`, `association`, `pvalue`, `effect_direction`

---

## Fitting SSD

Create an SSD instance with embeddings, corpus, outcome, and lexicon.
The constructor builds document vectors but does **not** fit a model — call `fit_pls()`, `fit_ols()`, or `fit_groups()` explicitly.

```python
from ssdiff import Embeddings, Corpus, SSD

emb = Embeddings.load("model.ssdembed")
emb.normalize(l2=True, abtt_m=1)
corpus = Corpus(texts, lang="en")

ssd = SSD(
    emb, corpus, y=scores,
    lexicon=["word1", "word2", "word3"],
    window=3,           # context window +/-3 tokens around lexicon hits
    sif_a=1e-3,         # SIF weighting parameter
    use_full_doc=False,  # False = seed context windows (default)
)
```

### PCA + OLS

Original SSD algorithm from the paper.

```python
result = ssd.fit_ols(
    n_components=None,    # None = auto-select via interpretability+stability sweep
    k_min=20,
    k_max=120,
    k_step=2,
    verbose=False,
)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `n_components` | `int \| None` | `None` | PCA components. `None` = auto-select via sweep |
| `k_min` | `int` | `20` | Minimum PCA-K for sweep |
| `k_max` | `int` | `120` | Maximum PCA-K for sweep |
| `k_step` | `int` | `2` | Step size |
| `verbose` | `bool` | `False` | Print progress |

### PLS

New proposed algorithm. PLS regression operates directly in the full embedding space, finding latent directions that maximize covariance between document vectors and the outcome without a separate dimensionality-reduction step. With the default single component it recovers one semantic gradient in a single pass, sidestepping the researcher degree of freedom in choosing PCA dimensionality. When more than one component is needed, automatic selection via cross-validation is available. Several significance testing methods are provided, including a split-half replication test and a permutation-calibrated variant with exact false-positive-rate control.

```python
result = ssd.fit_pls(
    n_components=1,       # or "auto" for CV-based selection
    p_method="auto",      # significance test (see below)
    verbose=False,
)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `n_components` | `int \| "auto"` | `1` | Number of PLS components. `"auto"` selects via 10-fold CV |
| `cv_folds` | `int` | `10` | CV folds for component selection |
| `use_1se` | `bool` | `True` | 1-SE rule for parsimonious selection |
| `pca_preprocess` | `int \| str \| None` | `None` | Optional PCA preprocessing (e.g. `50` or `"var95"`) |
| `p_method` | `str \| None` | `"auto"` | Significance test method |
| `n_perm` | `int` | `1000` | Permutation iterations |
| `n_splits` | `int` | `50` | Split-half iterations |
| `split_ratio` | `float` | `0.5` | Training fraction for split-based tests |
| `random_state` | `int` | `2137` | Random seed |
| `verbose` | `bool` | `False` | Print progress |

**`p_method` options**:

| Value | Description |
|-------|-------------|
| `"auto"` | `"split"` when `n_components=1`, `"perm"` otherwise |
| `"perm"` | Permutation test on cross-validated R-squared |
| `"split"` | Split-half test with overlap-corrected t-test |
| `"split_cal"` | Permutation-calibrated split-half (exact FPR control, slower) |
| `None` | Skip significance testing (p-value = NaN) |

### Inspecting results

Both `PLSResult` and `PCAOLSResult` share the same interpretation API:

```python
print(result.summary())
# SSD Model Summary (PLS)
# -----------------------
# Backend: PLS (3 components)
# Docs:  487 kept / 512 total (25 dropped)
# R-squared = 0.2341   R-squared_adj = 0.2293
# ...

result.r2, result.r2_adj        # model fit
result.pvalue                   # significance (NaN if skipped)
result.beta_norm                # ||beta|| in SD(y) per +1.0 cosine
result.delta                    # raw y change per +0.10 cosine
result.iqr_effect               # raw y change across IQR of cosine
result.y_corr_pred              # |corr(y, y_hat)|
result.cos_align                # per-doc cosine to beta (array)
```

For a comprehensive text report:

```python
print(result.report(top_words=10, clusters=100, extreme_docs=30, misdiagnosed=20))
```

---

## Choosing PCA Dimensionality (PCA Sweep)

When using `fit_ols()`, selecting the number of PCA components (`n_components = K`) can be a researcher degree of freedom. Pass `n_components=None` (the default) to run an automatic **PCA sweep** that evaluates a range of K values and selects the most robust solution.

### What PCA Sweep optimizes

For each candidate PCA dimensionality K, the sweep fits SSD and tracks:

1. **Interpretability quality** — based on clustering the nearest neighbors at each pole of the semantic gradient and computing aggregate cluster coherence and alignment with beta.

2. **Stability of the semantic gradient** — measured as the cosine change between consecutive gradients: `beta_delta = 1 - cos(beta_unit(K-1), beta_unit(K))`. Smaller values mean more stable gradients.

These signals are smoothed using an AUCK window.

### Example

```python
result = ssd.fit_ols(n_components=None, k_min=20, k_max=120, verbose=True)
print(f"Selected K = {result.n_components}")
print(result.summary())

result.plot_sweep("sweep.png")   # save sweep plot
result.plot_sweep()              # display interactively
```

The **blue curve** shows **detrended interpretability** as a function of K. The **orange curve** shows **solution stability**. The **red vertical line** marks the selected K.

---

## Neighbors & Clustering

### Nearest neighbors

Get the top N nearest neighbors of +beta/-beta:

```python
result.top_words(n=20)
# list[dict] with keys: side, rank, word, cos

result.neighbors("pos", n=20)
# list[tuple[str, float]] — (word, cosine) pairs
```

### Clustering neighbors into themes

Group the top N neighbors into k clusters (k-means; auto-selected via silhouette):

```python
clusters = result.cluster_neighbors(
    side="pos",
    topn=100,
    k=None,          # None = auto-select via silhouette
    k_min=2,
    k_max=10,
)
# list[dict] with keys: id, size, centroid_cos_beta, coherence, words

for cl in clusters:
    print(f"Cluster {cl['id']}: {', '.join(cl['words'][:5])} (coherence={cl['coherence']:.3f})")
```

---

## Interpreting with Snippets

After fitting, SSD lets you link the abstract directions in embedding space back to actual language by inspecting text snippets.

### Snippets along beta

```python
snips = result.snippets(corpus.pre_docs, top_per_side=200)
# dict with keys "pos" and "neg", each containing snippet metadata
```

### Snippets from extreme documents

```python
snips = result.snippets_extreme(corpus.pre_docs, k=50, by="predicted")
```

The snippet extraction:
1. Locates each occurrence of a seed word in the corpus.
2. Extracts a small window of surrounding context.
3. Represents that window as a SIF-weighted context vector.
4. Computes cosine similarity between the context vector and beta, ranking snippets by alignment.

---

## Per-Document SSD Scores

The SSD score for each document quantifies how closely its meaning aligns with the semantic gradient (beta).

```python
scores = result.doc_scores()
# dict with keys:
#   keep_mask   — bool array (n_raw,), which docs were kept
#   cos_align   — float array (n_kept,), cosine alignment to beta
#   score_std   — float array (n_kept,), standardized predicted scores
#   yhat_raw    — float array (n_kept,), predicted outcome in original scale
```

### Extreme documents

Top/bottom documents by predicted or observed outcome:

```python
extremes = result.extreme_docs(k=50, by="predicted")
# list[dict] with keys: idx, y_true, yhat, cos, side ("top"/"bottom")
```

### Misdiagnosed documents

Documents where the model is most wrong:

```python
result.misdiagnosed(k=20)                  # both over- and under-predicted
result.misdiagnosed(k=20, side="over")     # model over-predicts
result.misdiagnosed(k=20, side="under")    # model under-predicts
# list[dict] with keys: idx, y_true, yhat, cos, residual, side
```

---

## Cross-Group Comparison

When your research question involves **categorical groups** rather than a continuous outcome, use `ssd.fit_groups()`.

### When to use fit_groups vs fit_pls/fit_ols

| Scenario | Use |
|---|---|
| Continuous outcome (scale score, rating) | `fit_pls()` or `fit_ols()` |
| Categorical groups (diagnosis, condition) | `fit_groups()` |
| Continuous outcome AND group labels | Both — `fit_pls()` for the continuous analysis, `fit_groups()` for the group comparison |

### Fitting groups

```python
# Categorical groups
ssd = SSD(emb, corpus, y=group_labels, lexicon=lexicon)
result = ssd.fit_groups(n_perm=5000, correction="holm")

# Or: median split on continuous y
ssd = SSD(emb, corpus, y=scores, lexicon=lexicon)
result = ssd.fit_groups(median_split=True, n_perm=5000)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `median_split` | `bool` | `False` | Split continuous y into "low"/"high" at median |
| `n_perm` | `int` | `5000` | Permutation iterations |
| `correction` | `str` | `"holm"` | P-value correction: `"holm"`, `"bonferroni"`, `"fdr_bh"`, `"none"` |
| `random_state` | `int` | `2137` | Random seed |

Groups with fewer than 20 documents are automatically dropped.

### Interpreting group results

```python
print(result.summary())

# Pairwise results
result.results_table()     # list[dict] with T, p_raw, p_corrected, cohens_d, ...

# Interpretation (works across all contrasts, adds "contrast" key)
result.top_words(20)
result.cluster_neighbors("pos", topn=100)
result.snippets(corpus.pre_docs)

# Filter to specific groups (returns new GroupResult, no recomputation)
r = result.filter_groups("A", "B")
```

Key attributes:
- `result.omnibus_T` — mean pairwise cosine distance between centroids
- `result.omnibus_p` — permutation p-value for omnibus test
- `result.pairwise` — dict mapping `(g1, g2)` to result dicts with `T`, `p_raw`, `p_corrected`, `beta_unit`, `cohens_d`, etc.

---

## API Summary

The `ssdiff` top-level package exports three classes:

```python
from ssdiff import Embeddings, Corpus, SSD
```

### `Embeddings`

- `Embeddings.load(path, verbose=False, parallel=False)` — load `.ssdembed`, `.kv`, `.bin`, `.txt`, `.vec` (and `.gz` variants)
- `.normalize(l2=True, abtt_m=1, re_normalize=True)` — in-place L2 + ABTT; tracks state, safe to call repeatedly
- `.save(filename=None, fmt="ssdembed")` — save to native, text, binary, or gensim format
- `emb["word"]` — vector lookup
- `"word" in emb` — membership check
- `len(emb)` — vocabulary size
- `.vector_size` — embedding dimensionality
- `.similar_by_vector(vec, topn=10)` — nearest neighbor search

### `Corpus`

- `Corpus(texts, lang=None, model=None, nlp=None, stopwords=None, pretokenized=False)`
- `.docs` — lemmatized tokens per document
- `.pre_docs` — sentence-level structure for snippet extraction
- `.suggest_lexicon(y, top_k=30, ...)` — data-driven seed word selection

### `SSD`

- `SSD(embeddings, corpus, y, lexicon, *, window=3, sif_a=1e-3, use_full_doc=False)`
- `.fit_pls(n_components=1, p_method="auto", ...)` -> `PLSResult`
- `.fit_ols(n_components=None, ...)` -> `PCAOLSResult`
- `.fit_groups(median_split=False, n_perm=5000, correction="holm", ...)` -> `GroupResult`

### `PLSResult` / `PCAOLSResult`

Attributes: `r2`, `r2_adj`, `pvalue`, `n_components`, `beta`, `beta_unit`, `beta_norm`, `delta`, `iqr_effect`, `y_corr_pred`, `cos_align`, `y_mean`, `y_std`.

Methods:
- `.summary()` — human-readable model summary
- `.report(top_words=5, clusters=None, extreme_docs=None, misdiagnosed=None)` — comprehensive text report
- `.top_words(n=20)` -> `list[dict]`
- `.neighbors(side="pos", n=20)` -> `list[tuple[str, float]]`
- `.cluster_neighbors(side="pos", topn=100, ...)` -> `list[dict]`
- `.doc_scores()` -> `dict`
- `.extreme_docs(k=50, by="predicted")` -> `list[dict]`
- `.misdiagnosed(k=20, side="both")` -> `list[dict]`
- `.snippets(pre_docs, top_per_side=200)` -> `dict`
- `.snippets_extreme(pre_docs, k=50, by="predicted")` -> `dict`
- `.split_test(n_splits=50, method="split")` -> `dict` (PLSResult only)
- `.plot_sweep(path=None)` — PCA-K sweep plot (PCAOLSResult only)

### `GroupResult`

Attributes: `omnibus_T`, `omnibus_p`, `group_labels`, `G`, `n_kept`, `pairwise`.

Methods:
- `.summary()` — group comparison summary
- `.report(top_words=5, clusters=None)` — comprehensive text report
- `.results_table()` -> `list[dict]`
- `.top_words(n=20)` -> `list[dict]` (with `"contrast"` key)
- `.cluster_neighbors(side="pos", ...)` -> `list[dict]`
- `.snippets(pre_docs)` -> `dict`
- `.filter_groups(*labels)` -> `GroupResult`

### Lexicon utilities

```python
from ssdiff.utils.lexicon import suggest_lexicon, token_presence_stats, coverage_by_lexicon
```

All return plain Python types (`list[dict]`, `dict`). No pandas dependency — convert with `pd.DataFrame(result.top_words())` if needed.

---

## Citing & License

- License: GPL v3 (see LICENSE).
- If you use SSD in published work, please cite the associated paper.
- A suggested citation:

Plisiecki, H., Lenartowicz, P., Pokropek, A., Malyska, K., & Flakus, M. (2025). Measuring Individual Differences in Meaning: The Supervised Semantic Differential. PsyArXiv. https://doi.org/10.31234/osf.io/gvrsb_v1

---

## Questions / Contributions

- File issues and feature requests on the repo's Issues page.
- Pull requests welcome — especially for:
  - Robustness diagnostics and visualization helpers
  - Documentation improvements

Contact: hplisiecki@gmail.com

Project was funded by the National Science Centre, Poland (grant no. 2020/38/E/HS6/00302).
