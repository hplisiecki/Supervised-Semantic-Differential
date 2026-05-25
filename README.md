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
  - [PLS](#pls)
  - [Multi-component PLS (in development)](#multi-component-pls-in-development)
  - [PCA + OLS](#pca--ols)
  - [Cross-Group Comparison](#cross-group-comparison)
  - [Inspecting results](#inspecting-results)
- [Neighbors & Clustering](#neighbors--clustering)
- [Interpreting with Snippets](#interpreting-with-snippets)
- [Per-Document SSD Scores](#per-document-ssd-scores)
- [API Summary](#api-summary)
- [Citing & License](#citing--license)

---

## Installation

```bash
pip install ssdiff
```

**Python**: 3.10 – 3.14.

Core dependencies (installed automatically): `numpy`, `spacy`.

Optional extras:
- `ssdiff[results]` — pandas / openpyxl / python-docx / matplotlib for `to_df()`, `.xlsx`/`.docx` export, and `plot_sweep()`.
- `ssdiff[gensim]` — only needed to *save* embeddings in `.kv` format.

> Loading `.kv` files works without gensim (handled by an internal unpickler shim).

---

## Quickstart

Below is an end-to-end minimal example. Adjust paths and column names to your data.

```python
from ssdiff import Embeddings, Corpus, SSD
import numpy as np

# 1) Load and normalize embeddings
emb = Embeddings.load("path/to/embeddings.txt", verbose=True)
emb.normalize(l2=True, abtt=1)

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
print(result.stats)            # r², p-value, n_kept, β-norm, IQR effect
print(result.words.pos)        # top β-positive neighbours
print(result.words.neg)        # top β-negative neighbours
print(result.clusters.pos(topn=100))   # cluster the 100 nearest +β neighbours
result.report().save("report.md")
```

Every result attribute is a view: print it, slice with `(n)`, dispatch
to one side with `.pos` / `.neg`, or export with `.to_df()` / `.save(...)`.

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
emb.normalize(l2=True, abtt=1)   # L2 + ABTT (remove top-1 PC)
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

These helpers make lexicon selection transparent and data-driven (you can also hand-pick tokens). They are **methods on `Corpus`** — they operate on the already-lemmatized tokens, so what they score is exactly what `SSD` will consume.

### `corpus.suggest_lexicon(y, ...)`

Rank tokens by balanced coverage with a mild penalty for strong association with the outcome. Returns a `LexiconResult` view (printable, exportable, sliceable).

```python
corpus = Corpus(texts, lang="en")
result = corpus.suggest_lexicon(y, top_k=30)
print(result)                      # tabular view
ssd = SSD(emb, corpus, y, lexicon=result.tokens)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `y` | `array-like` | — | Outcome variable (numeric or categorical) |
| `top_k` | `int` | `30` | Maximum number of words to return |
| `min_docs` | `int` | `5` | Minimum document frequency |
| `n_bins` | `int` | `4` | Quantile bins for balanced coverage |
| `corr_cap` | `float` | `0.30` | Penalty threshold for outcome association |
| `var_type` | `str` | `"continuous"` | `"continuous"` or `"categorical"` |

### `corpus.evaluate_lexicon(y, lexicon, ...)`

Score an existing lexicon against an outcome. Returns a `LexiconResult` bundling per-token diagnostics (`.suggestions`) and an aggregate coverage summary (`.summary`) — both saveable, with `.report()` producing a narrative markdown overview.

```python
corpus = Corpus(texts, lang="en")
lex = corpus.evaluate_lexicon(y, lexicon=["happy", "sad", "anger"])

print(lex)                              # tabular view
lex.suggestions.save("tokens.csv")      # per-token rows
lex.summary.save("coverage.csv")        # aggregate stats
lex.report().save("lexicon.md")         # narrative overview
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `y` | `array-like` | — | Outcome variable (numeric or categorical) |
| `lexicon` | `iterable[str]` | — | Tokens to evaluate (matched against lemmatized corpus) |
| `n_bins` | `int` | `4` | Quantile bins for balanced coverage |
| `corr_cap` | `float` | `0.30` | Penalty threshold for outcome association |
| `var_type` | `str` | `"continuous"` | `"continuous"` or `"categorical"` |

`.suggestions` columns: `token, freq, cov_all, cov_bal, corr, pvalue, direction, rank`.
`.summary` fields: `docs_any, cov_all, q1, q4, corr_any, hits_mean, hits_median, types_mean, types_median` (plus `group_cov` for categorical `y`).

---

## Fitting SSD

Create an SSD instance with embeddings, corpus, outcome, and lexicon.
The constructor builds document vectors but does **not** fit a model — call `fit_pls()`, `fit_multipls()`, `fit_ols()`, or `fit_groups()` explicitly.

```python
from ssdiff import Embeddings, Corpus, SSD

emb = Embeddings.load("model.ssdembed")
emb.normalize(l2=True, abtt=1)
corpus = Corpus(texts, lang="en")

ssd = SSD(
    emb, corpus, y=scores,
    lexicon=["word1", "word2", "word3"],
    window=3,           # context window +/-3 tokens around lexicon hits
    sif_a=1e-3,         # SIF weighting parameter
    use_full_doc=False,  # False = seed context windows (default)
)
```

### PLS

New proposed algorithm. PLS regression operates directly in the full embedding space, finding latent directions that maximize covariance between document vectors and the outcome without a separate dimensionality-reduction step. With a single component it recovers one semantic gradient in a single pass, sidestepping the researcher degree of freedom in choosing PCA dimensionality. With `k="auto"` (default) the number of components is selected via `plskit.pls1_find_k_optimal` (selector `r2_se`); the reported p-value is always the **honest k=1 confirmatory** `split_nb` (Lenartowicz, 2026) split-half statistic, independent of selection.

```python
result = ssd.fit_pls(
    k="auto",             # int, or "auto" for find_k_optimal
    k_max=5,              # cap for "auto"
    n_splits=50,          # split_nb iterations
    random_state=2137,
    verbose=False,
)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `k` | `int \| "auto"` | `"auto"` | Number of PLS components. `int` fits at exactly that k. `"auto"` calls `plskit.pls1_find_k_optimal` (selector `r2_se`, diagnostic `split_nb`); p-value is the honest k=1 confirmatory `split_nb` statistic. |
| `k_max` | `int` | `5` | Cap for `k="auto"`, clamped to `min(k_max, n-1, D)`. Ignored when `k` is an int. |
| `n_splits` | `int` | `50` | Random splits for the `split_nb` test. |
| `random_state` | `int` | `2137` | Random seed. |
| `verbose` | `bool` | `False` | Print K-selection chain and confirmatory test progress. |

To re-run the test with different settings, call `result.test(n_splits=200)` — it overwrites `result.stats.pvalue` and `result.test.pvalue` in place.

### Multi-component PLS (in development)

When you expect more than one interpretable semantic axis related to the outcome, `fit_multipls()` fits `k` PLS components and rotates the W-subspace (`"varimax"` or `"raw"`). The returned `MultiPLSResult` is a container of per-dim leaves keyed by `"dim-1"`, `"dim-2"`, … (one per rotated axis).

```python
result = ssd.fit_multipls(
    k="auto",             # or an int
    k_max=5,
    rotate="varimax",     # or "raw"
    rotation_vocab=50_000,
    n_splits=50,
    random_state=2137,
    verbose=False,
)

print(result.stats)             # container-level r², pvalue, n_components
print(result.test)              # honest k=1 confirmatory split_nb
result.words                    # pivoted top-words view across rotated dims
result["dim-1"].words           # zoom into rotated axis 1
result["dim-1"].clusters.pos    # cluster +β neighbours on that axis
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `k` | `int \| "auto"` | `"auto"` | Number of PLS components. Same semantics as `fit_pls`. |
| `k_max` | `int` | `5` | Cap for `k="auto"`, clamped to `min(k_max, n-1, D)`. |
| `rotate` | `"raw" \| "varimax"` | `"varimax"` | Rotation applied to the W-subspace. |
| `rotation_vocab` | `int \| None` | `50_000` | Leading vocabulary rows fed to varimax as the simple-structure target. Assumes frequency-ranked vocab. `None` uses the full matrix. No-op for `rotate="raw"`. |
| `n_splits`, `random_state`, `verbose` | — | — | Same meaning and defaults as `fit_pls`. |

Container-level p-value follows `fit_pls` semantics (honest k=1 confirmatory). Each rotated leaf carries a diagnostic per-dim p-value remapped via the `mpls_fit` rotation `order`.

> **Status.** API is stable for research use; feature parity with `PLSResult` (per-leaf docs, snippets, misdiagnosed) is still being rolled out. See [`examples/demo_multipls.py`](examples/demo_multipls.py) and [`docs/api_reference.md`](docs/api_reference.md). RAM-efficient embeddings (`Embeddings.load(ram_efficient=True)`) are not supported by `fit_multipls` — it needs the full vocabulary as a rotation target.

### PCA + OLS

Original SSD algorithm from the paper.

```python
result = ssd.fit_ols(
    fixed_k=None,         # None = auto-select via interpretability+stability sweep
    k_min=2,
    k_max=120,
    k_step=2,
    verbose=False,
)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `fixed_k` | `int \| None` | `None` | Fixed PCA components. `None` = auto-select via sweep |
| `k_min` | `int` | `2` | Minimum PCA-K for sweep |
| `k_max` | `int` | `120` | Maximum PCA-K for sweep |
| `k_step` | `int` | `2` | Step size |
| `verbose` | `bool` | `False` | Print progress |

#### Automatic K selection (PCA sweep)

Selecting the number of PCA components (`fixed_k = K`) can be a researcher degree of freedom. Pass `fixed_k=None` (the default) to run an automatic **PCA sweep** that evaluates a range of K values and selects the most robust solution.

For each candidate PCA dimensionality K, the sweep fits SSD and tracks:

1. **Interpretability quality** — based on clustering the nearest neighbors at each pole of the semantic gradient and computing aggregate cluster coherence and alignment with beta.

2. **Stability of the semantic gradient** — measured as the cosine change between consecutive gradients: `beta_delta = 1 - cos(gradient(K-1), gradient(K))`. Smaller values mean more stable gradients.

These signals are smoothed using an AUCK window.

```python
result = ssd.fit_ols(fixed_k=None, k_min=2, k_max=120, verbose=True)
print(f"Selected K = {result.n_components}")
print(result.stats)

result.plot_sweep("sweep.png")   # save sweep plot
result.plot_sweep()              # display interactively
```

The **blue curve** shows **detrended interpretability** as a function of K. The **orange curve** shows **solution stability**. The **red vertical line** marks the selected K.

### Cross-Group Comparison

When your research question involves **categorical groups** rather than a continuous outcome, use `ssd.fit_groups()`.

| Scenario | Use |
|---|---|
| Continuous outcome (scale score, rating) | `fit_pls()` or `fit_ols()` |
| Categorical groups (diagnosis, condition) | `fit_groups()` |
| Continuous outcome AND group labels | Both — `fit_pls()` for the continuous analysis, `fit_groups()` for the group comparison |

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

Groups are canonicalised internally — original labels are remapped to `"g1"`, `"g2"`, … (in sorted order). The original-label mapping is exposed on `result.group_labels`.

#### Interpreting group results

```python
print(result)                  # header + view directory
print(result.stats)            # G, n_kept, n_perm, correction, omnibus pvalue
print(result.test)             # omnibus pvalue + pairwise contrasts block

# Pairwise rows (T, p_raw, p_corrected, cohens_d, n_g1, n_g2 per contrast)
result.pairs                   # PairsView — exports via .to_df() / .save()

# Pivoted interpretation across all contrasts (adds a "contrast" column)
result.words.pos
result.clusters.pos(topn=100)
result.snippets.pos

# Zoom into one pair → PairResult (canonical keys: "g1", "g2", ...)
pair = result[("g1", "g2")]
pair.words.pos
pair.clusters.pos
pair.snippets

# Re-run the permutation test with different settings
result.test(n_perm=10_000, correction="fdr_bh")
```

Key attributes:
- `result.G` — number of retained groups (after the 20-doc minimum filter)
- `result.n_kept` — total documents across retained groups
- `result.group_labels` — `dict` mapping canonical keys (`"g1"`, …) to original labels
- `result.test.omnibus_T`, `result.test.omnibus_p` — omnibus statistic and permutation p
- `result.pairs` — list-like view of `Pair` rows with per-contrast `T`, `p_raw`, `p_corrected`, `cohens_d`, `n_g1`, `n_g2`, `contrast_norm`

### Inspecting results

Both `PLSResult` and `PCAOLSResult` share the same interpretation API — everything is a printable, exportable view:

```python
print(result)                   # header + view/array directory
print(result.stats)             # backend, r², r²_adj (OLS only), p, n_kept,
                                # β-norm, Δ (per +0.10 cosine), IQR effect,
                                # |corr(y, ŷ)|, y_mean, y_std
print(result.fit_info)          # n_components, p_at_k, random_state,
                                # plus PCA-K sweep info for OLS

# Direct array attributes (numpy ndarrays)
result.beta                     # raw direction in embedding space
result.gradient                 # unit-length version of beta
result.beta_norm                # ||beta|| (effect-size summary)
result.alignment_scores         # per-doc cosine to gradient
result.n_components             # number of PLS / PCA components

# Comprehensive narrative report — every section is on by default; pass
# section=False to drop one.
print(result.report(clusters={"n": 100, "n_words": 10, "n_snippets": 2},
                    extreme_docs={"n": 30}, misdiagnosed={"n": 20}))
result.report().save("report.md")    # also .html / .docx / .tex

# Re-run the significance test in place
result.test(n_splits=200)            # PLSResult — overwrites stats.pvalue
```

For `MultiPLSResult` and `GroupResult`, see the "Multi-component PLS" and "Cross-Group Comparison" sections above.

---

## Neighbors & Clustering

### Nearest neighbors

`result.words` is a tabular view with columns `side`, `rank`, `word`, `cos_beta`:

```python
result.words            # default: top 20 per pole
result.words.pos        # one-sided, default 20 rows
result.words.pos(50)    # resize to 50 rows
result.words.neg(None)  # all available rows on this side

# Standard view exports
result.words.to_df()              # pandas DataFrame
result.words.save("words.csv")    # csv / json / md / xlsx / docx / tex
```

### Clustering neighbors into themes

`result.clusters` k-means clusters the top neighbours per pole (k auto-selected via silhouette unless pinned):

```python
result.clusters.pos              # default topn=100
result.clusters.pos(topn=200, k=4)         # recompute with different params
result.clusters.pos(cluster_id=0).words    # zoom into one cluster
result.clusters.pos(cluster_id=0).snippets # snippets aligned with that centroid
result.clusters.words            # flat per-side cluster-words table

# Columns: cluster_id, side, size, coherence, centroid_cos_beta
result.clusters.pos.to_df()
result.clusters.save("clusters.csv")
```

---

## Interpreting with Snippets

After fitting, SSD lets you link the abstract directions in embedding space back to actual language by inspecting text snippets near seed-word occurrences. Snippets are pulled from the `Corpus` attached at fit time — no need to pass `pre_docs` manually.

```python
result.snippets                       # default: top 30 per pole
result.snippets.pos                   # SnippetsViewSided, top 30
result.snippets.pos(50)               # resize
result.snippets(top_per_side=200, min_cosine=0.1)   # recompute extraction

# Snippets aligned with a specific cluster centroid
result.clusters.pos(cluster_id=0).snippets

# Columns: snippet_id, side, doc_id, cosine, seed, start/end indices,
# text_window, text_surface, text_lemmas, cluster_id, contrast
result.snippets.to_df()
result.snippets.save("snippets.xlsx")
```

The snippet extraction:
1. Locates each occurrence of a seed word in the corpus.
2. Extracts a small window of surrounding context.
3. Represents that window as a SIF-weighted context vector.
4. Computes cosine similarity between the context vector and β, ranking snippets by alignment.

---

## Per-Document SSD Scores

`result.docs` exposes per-document predictions and the cosine alignment score (the SSD score, ⟨d_i, gradient⟩):

```python
result.docs                          # all rows; columns: doc_id, y_true,
                                     # y_hat, residual, alignment_score
result.docs.pos(20)                  # 20 most β-positive (highest y_hat)
result.docs.neg(20)                  # 20 most β-negative
result.docs.id(42)                   # single-doc detail (incl. raw text)

# Misdiagnosed — largest |residual|
result.docs.misdiagnosed(20)                     # both over and under
result.docs.misdiagnosed(20, direction="over")   # y_hat > y_true
result.docs.misdiagnosed(20, direction="under")  # y_hat < y_true

result.docs.to_df()
result.docs.save("docs.csv")
```

The full per-document alignment vector is also available directly:

```python
result.alignment_scores              # ndarray of shape (n_kept,)
```

---

## API Summary

The `ssdiff` top-level package exports three primary classes plus result and view classes:

```python
from ssdiff import Embeddings, Corpus, SSD
# Result / view classes (re-exported for type hints, isinstance checks, pickling):
from ssdiff import (
    PLSResult, PCAOLSResult, GroupResult, LexiconResult,
    WordsView, WordsViewSided, ClustersView, ClustersViewSided,
    ClusterWordsView, ClusterWordsViewSided, SnippetsView, SnippetsViewSided,
)
# In-development; not exported at top level:
from ssdiff.results.multi_pls_result import MultiPLSResult
```

### `Embeddings`

- `Embeddings.load(path, *, verbose=False, parallel=False, ram_efficient=False)` — load `.ssdembed`, `.kv`, `.bin`, `.txt`, `.vec` (and `.gz` variants)
- `.normalize(l2=True, abtt=1, re_normalize=True)` — in-place L2 + ABTT; tracks state, safe to call repeatedly
- `.save(filename=None, fmt="ssdembed")` — save to native, text, binary, or gensim format
- `emb["word"]` / `emb.get_vector("word", norm=False)` — vector lookup
- `"word" in emb` — membership check
- `len(emb)` / `.vocab_size` — vocabulary size
- `.vector_size` (alias `.dim`) — embedding dimensionality
- `.similar_by_vector(vec, topn=10, restrict_vocab=None)` — nearest neighbor search

### `Corpus`

- `Corpus(texts, *, lang=None, model=None, nlp=None, stopwords=None, pretokenized=False, auto_download=None)`
- `.docs` — lemmatized tokens per document
- `.pre_docs` — sentence-level structure for snippet extraction
- `.n_texts` — number of documents
- `.suggest_lexicon(y, *, top_k=30, ...)` -> `LexiconResult` — data-driven seed word selection
- `.evaluate_lexicon(y, lexicon, ...)` -> `LexiconResult` — score an existing lexicon (per-token + aggregate)

### `SSD`

- `SSD(embeddings, corpus, y, lexicon, *, window=3, sif_a=1e-3, use_full_doc=False)`
- `.fit_pls(*, k="auto", k_max=5, n_splits=50, random_state=2137, verbose=False)` -> `PLSResult`
- `.fit_multipls(*, k="auto", k_max=5, rotate="varimax", rotation_vocab=50_000, n_splits=50, ...)` -> `MultiPLSResult` *(in development)*
- `.fit_ols(*, fixed_k=None, k_min=2, k_max=120, k_step=2, verbose=False)` -> `PCAOLSResult`
- `.fit_groups(*, median_split=False, n_perm=5000, correction="holm", random_state=2137, verbose=False)` -> `GroupResult`

### `PLSResult` / `PCAOLSResult`

**Direct array attributes**: `beta`, `gradient`, `beta_norm`, `alignment_scores`, `n_components`, `x`, `y`. PLS adds `component_scores`, `component_weights`, `find_k_result`, `cv_scores`. PCA+OLS adds `pca_components`, `pca_weights`, `pca_k`, `sweep_result`.

**Scalar views** (all expose `.r2`, `.pvalue`, … as attributes; print to read, export with `.to_df()` / `.save(...)`):
- `.stats` — `backend`, `r2`, `r2_adj` (OLS only), `pvalue`, `n_raw`, `n_kept`, `n_dropped`, `y_mean`, `y_std`, `beta_norm`, `delta`, `iqr_effect`, `y_corr_pred`
- `.fit_info` — `n_components`, `p_at_k`, `n_splits`, `random_state`, plus PCA-K sweep info for OLS

**Tabular views**:
- `.words` → `WordsView` (with `.pos` / `.neg` → `WordsViewSided`, callable `(n)` to resize)
- `.clusters` → `ClustersView` (with `.pos` / `.neg` → `ClustersViewSided`, callable `(topn=…, k=…)` to recompute or `(cluster_id)` to zoom)
- `.snippets` → `SnippetsView` (with `.pos` / `.neg`, callable `(top_per_side=…)` to recompute)
- `.docs` → `DocsView` with `.pos(k)`, `.neg(k)`, `.misdiagnosed(k, direction=…)`, `.id(doc_id)`
- `.sweep` → `SweepView` (PCA+OLS only) — per-K interpretability/stability rows
- `.test` → `TestView` — callable to **re-run** the test in place (`result.test(n_splits=200)` overwrites `stats.pvalue` and `test.pvalue`)

**Methods**:
- `.report(clusters=True, top_words=True, extreme_docs=True, misdiagnosed=True)` -> `Report` — every section is on by default; pass `section=False` to drop one. Each section toggle is `True` / `False` / `None` / `dict` (e.g. `clusters={"n": 20, "n_words": 5, "n_snippets": 1}`). Stats + Fit info are always included. Use `.to_text()`, `.to_html()`, `.save("report.md")`.
- `.attach(corpus=None, embeddings=None)` — re-attach after un-pickling
- `.plot_sweep(path=None)` — PCA-K sweep chart (`PCAOLSResult` only)

### `GroupResult`

**Direct attributes**: `G`, `n_kept`, `n_perm`, `correction`, `random_state`, `group_labels` (canonical → original label dict), `x`, `groups`, `beta`, `gradient`, `beta_norm`, `alignment_scores`.

**Views**: `.stats`, `.test` (omnibus `pvalue`, `omnibus_T`, `omnibus_p`), `.pairs` (per-contrast `T`, `p_raw`, `p_corrected`, `cohens_d`, `n_g1`, `n_g2`), `.words`, `.clusters`, `.snippets` (all pivoted across contrasts, add a `contrast` column).

**Pair access**: `result[("g1", "g2")]` → `PairResult` (canonical keys only) with its own `.words`, `.clusters`, `.snippets`, `.gradient`, `.beta`. Use `result.keys()` to list available pair keys; `result.group_labels` to map canonical → original.

**Methods**: `.report(clusters=True, top_words=True)` — both sections on by default; pass `section=False` to drop one. Each toggle is `True` / `False` / `None` / `dict` (e.g. `clusters={"n": 20, "n_words": 5, "n_snippets": 1}`). Omnibus + Group labels + Pairwise contrasts are always included. `.test(n_perm=…, correction=…)` (re-runs in place); `.attach(...)`.

### Lexicon utilities

The lexicon helpers are **methods on `Corpus`**, not standalone imports:

```python
corpus = Corpus(texts, lang="en")
suggestions = corpus.suggest_lexicon(y, top_k=30)           # → LexiconResult
lex = corpus.evaluate_lexicon(y, lexicon=["happy", "sad"])  # → LexiconResult
```

`LexiconResult` views (`.suggestions`, `.summary`) and `.report()` support `.to_df()` (requires `ssdiff[results]`), `.to_dict()`, `.to_records()`, and `.save("file.{csv,json,md,xlsx,docx,tex,html}")`.

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
