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
https://doi.org/10.31234/osf.io/gvrsb_v3

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
  - [Several outcomes (PLS3)](#several-outcomes-pls3)
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
pip install ssdiff            # wordpacks: Corpus(texts, embeddings=emb)
pip install ssdiff[spacy]     # spaCy: Corpus(texts, lang=...)
```

**Python**: 3.10 – 3.14.

Core dependencies (installed automatically): `numpy`, `plskit`.

Optional extras:
- `ssdiff[spacy]` — spaCy, needed to tokenize with `Corpus(texts, lang=...)`. Wordpacks need no spaCy (see [Wordpacks](#wordpacks)).
- `ssdiff[results]` — pandas / openpyxl / python-docx / matplotlib for `to_df()`, `.xlsx`/`.docx` export, and `plot_sweep()`.
- `ssdiff[gensim]` — only needed to *save* embeddings in `.kv` format.

> Loading `.kv` files works without gensim (handled by an internal unpickler shim).

---

## Quickstart

Below is an end-to-end minimal example for English text. For another language, or your own embedding file, see [Word Embeddings](#word-embeddings).

```python
from ssdiff import Embeddings, Corpus, SSD
import numpy as np

# 1) Load embeddings: the default English wordpack, downloaded once (380 MB)
emb = Embeddings.load()

# 2) Load your data
texts = [...]                          # list of raw text strings
scores = np.array([...])               # numeric outcome

# 3) Tokenize texts
corpus = Corpus(texts, embeddings=emb) # the pack's tokenizer and stopwords; no spaCy

# 4) Define a lexicon (the forms as they occur in the texts)
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

For English, `Embeddings.load()` downloads a ready wordpack. For another language, load your own embedding file. The full guide is [`docs/embeddings.md`](docs/embeddings.md).

### Wordpacks

A wordpack carries its own tokenizer and stopword list, so it needs no spaCy. Load one by short tag or by full name; it is downloaded once into a cache directory and loaded from there. With no argument, `Embeddings.load()` loads `en`:

```python
from ssdiff import Embeddings, Corpus, SSD

emb = Embeddings.load()                                # the default: same as Embeddings.load("en")
# emb = Embeddings.load("en-wiki")                     # short tag: the current default for that tag
# emb = Embeddings.load("en-glove2024-dolma-1")        # full name: always this file
# emb = Embeddings.load("path/to/file.wordpack")       # a file you built yourself
corpus = Corpus(texts, embeddings=emb)                 # the pack's tokenizer and stopwords
res = SSD(emb, corpus, y, use_full_doc=True).fit_pls()
pred = res.predict(new_texts)
```

| short tag | full name | use it for |
|---|---|---|
| `en` | `en-glove2024-dolma-1` | informal or recent text: comments, social media, open survey answers, forums |
| `en-wiki` | `en-glove2024-wikigiga-1` | formal text: news, essays, reports, older press |

A full name always means the same file, and it is what `report()` and `corpus.tokenizer` record, so use it in scripts whose results must be reproducible. A short tag follows the published default and can move to a newer pack.

Packs are kept in `~/.cache/wordpack` on Linux (`$XDG_CACHE_HOME/wordpack` when that variable is set), `~/Library/Caches/wordpack` on macOS and `%LOCALAPPDATA%\wordpack` on Windows; set `WORDPACK_CACHE` to use another directory. Nothing removes them: delete files there to free space.

A pack loads with ABTT (m = 1) already applied and keeps its 50,000 most frequent words in RAM; other words are read from the file when needed, so keep the file in place while you use `emb`.

`emb.normalize(abtt=m)` sets any m from 0 to 10, up or down.

English packs are keyed by surface form, not by lemma. A seed lexicon lists the forms that occur in the texts: `dog` and `dogs`, not the lemma alone. ssdiff does no lemma expansion.

### Your own embedding file

For another language, or your own model, load an embedding file in one of these formats:

| Format | Extension | Notes |
|--------|-----------|-------|
| Wordpack | `.wordpack` | Tokenizer, stopwords and ABTT included; needs no spaCy |
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

Look for pre-trained static word embeddings in your language with good vocabulary coverage for your domain. GloVe and word2vec trained on large general corpora are a reliable starting point.

For Polish, the `nkjp+wiki-lemmas-all-300-cbow-hs.txt.gz` (no. 25) from the [Polish Word2Vec model list](https://dsmodels.nlp.ipipan.waw.pl) was found to work well.

---

## Preprocessing (Corpus)

With a wordpack, `Corpus(texts, embeddings=emb)` uses the pack's own tokenizer and stopwords. With any other embedding, the `Corpus` class runs the full spaCy preprocessing pipeline — tokenization, lemmatization, and stopword removal. That needs `pip install ssdiff[spacy]`.

```python
from ssdiff import Corpus

corpus = Corpus(texts, lang="en")      # auto-downloads spaCy model if needed
corpus.docs       # list[list[str]] — tokens per document (lemmas for spaCy, the pack's keys for a wordpack)
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
The constructor builds document vectors but does **not** fit a model — call `fit_pls()`, `fit_multipls()`, `fit_ols()`, or `fit_groups()` explicitly. Pass a 1-D `y` for `fit_pls`, `fit_ols` and `fit_groups`. Pass several numeric outcomes (a DataFrame or a 2-D array) for `fit_multipls`.

```python
from ssdiff import Embeddings, Corpus, SSD

emb = Embeddings.load("model.bin")
emb.normalize(l2=True, abtt=1)
corpus = Corpus(texts, lang="en")

ssd = SSD(
    emb, corpus, y=scores,
    lexicon=["word1", "word2", "word3"],
    window=3,           # context window +/-3 tokens around lexicon hits
    sif_a=1e-3,         # SIF weighting parameter; None turns SIF off
    use_full_doc=False,  # False = seed context windows (default)
)
```

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
print(f"Selected K = {result.pca_k}")   # PCAOLSResult exposes .pca_k (PLSResult uses .n_components)
print(result.stats)

result.plot_sweep("sweep.png")   # save sweep plot
result.plot_sweep()              # display interactively
```

The **blue curve** shows **detrended interpretability** as a function of K. The **orange curve** shows **solution stability**. The **red vertical line** marks the selected K.


### PLS

PLS regression operates directly in the full embedding space, finding latent directions that maximize covariance between document vectors and the outcome without a separate dimensionality-reduction step. With a single component it recovers one semantic gradient in a single pass. With `k="auto"` (default) the number of components is selected via selector `r2_se`); the p-value comes from the `split_exact` split-half test, at the fitted `k` (for `k="auto"`, at k=1, independent of selection).

```python
result = ssd.fit_pls(
    k="auto",             # int, or "auto" for find_k_optimal
    k_max=5,              # cap for "auto"
    keep=None,            # int | "auto" -> sparse PLS (needs an int k)
    n_splits=50,          # split-half iterations
    random_state=2137,
    verbose=False,
)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `k` | `int \| "auto"` | `"auto"` | Number of PLS components. `int` fits at exactly that k. `"auto"` calls `plskit.pls1_find_k_optimal` (selector `r2_se`, diagnostic `split_nb`). The p-value is the `split_exact` test (permutation-calibrated split-half) at the fitted `k`; with `k="auto"` it is the k=1 `split_exact` test. |
| `k_max` | `int` | `5` | Cap for `k="auto"`, clamped to `min(k_max, n-1, D)`. Ignored when `k` is an int. |
| `keep` | `int \| "auto" \| None` | `None` | Sparse-PLS knob. `None` is dense PLS. An `int` in `[1, n_features]` gives each component exactly that many non-zero weights. `"auto"` cross-validates the keep-count. Sparse mode needs an explicit integer `k`; `keep` with `k="auto"` raises. |
| `n_splits` | `int` | `50` | Random splits for the split-half tests. |
| `random_state` | `int` | `2137` | Random seed. |
| `verbose` | `bool` | `False` | Print K-selection chain and confirmatory test progress. |

To re-run the test with different settings, call `result.test(n_splits=200)` — it overwrites `result.stats.pvalue` and `result.test.pvalue` in place.

With `keep` set, the p-value is the largest p-value in the sequential chain of `k` tests (`result.test.name` is `"split_nb_seq"`). That is a different test from the dense p-value, so do not compare the two directly. The chain is computed at fit time, so `result.test()` raises `RuntimeError` on a sparse result. To change `n_splits` or the seed, refit. `result.fit_info.keep` holds the keep-count that was used (`None` for dense fits).

### Several outcomes (PLS3)

When you have several numeric outcomes, `fit_multipls()` fits them together. Pass `y` as a DataFrame (column names become outcome names) or a 2-D array (names are `"y1"`, `"y2"`, …). Rows with a missing value in any outcome are dropped.

PLS3 takes the SVD of the cross-covariance between document vectors and outcomes. Dimension `i` pairs one direction in word space with one vector of outcome loadings. The returned `MultiPLSResult` is a container of per-dimension leaves keyed by `"dim-1"`, `"dim-2"`, …

```python
result = ssd.fit_multipls(
    k=2,                  # required, 1 <= k <= min(q, n-1, D)
    rotate=None,          # None, "words" or "outcomes"
    outcome_groups=None,  # dict of group name -> outcome names; needs rotate="outcomes"
    n_splits=50,
    n_perm=1000,
    random_state=2137,
    verbose=False,
)

print(result.stats)             # n, q, k, rotate, cov_explained, pvalue
print(result.test)              # LV1 test (split_exact)
result["dim-1"].outcomes        # outcome loadings on dimension 1
result["dim-1"].words           # words along dimension 1
result["dim-1"].clusters.pos    # cluster +beta neighbours on that dimension
result["dim-1"].stats           # cov_share, score_corr
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `k` | `int` | — | Number of dimensions. Required, `1 <= k <= min(q, n-1, D)`. At `k = q` every component is kept, so `cov_explained` is 1 by construction. |
| `rotate` | `None \| "words" \| "outcomes"` | `None` | `None` keeps the SVD dimensions. `"words"` runs varimax against the projection of the leading vocabulary rows onto the word-space directions. `"outcomes"` runs varimax against the outcome loadings (or the group rows below); it needs `k < q`. |
| `outcome_groups` | `dict[str, list[str]] \| None` | `None` | Only with `rotate="outcomes"`. Each group becomes one rotation target row. Outcomes in no group keep their own row. `k` cannot exceed the number of target rows. |
| `n_splits`, `n_perm` | `int` | `50`, `1000` | Splits and permutations for the LV1 test. |
| `random_state`, `verbose` | — | `2137`, `False` | Seed and progress output. |

A group's rotation target row and its `.outcomes` group loading are plain means of the group's outcome loadings. Outcomes scored in opposite directions cancel within a group.

Varimax modes reorder the dimensions by their share of the cross-covariance. In every mode, the sign of each dimension is set so its largest outcome loading is positive.

The container p-value comes from the split_exact PLS3 test at the first dimension (LV1) only. It does not depend on `rotate`. Leaves have no p-value and no `.test`. To rerun it with other settings, call `result.test(n_splits=100, n_perm=2000, seed=42)`.

Container attributes and views:

- `.stats` — `n`, `q`, `k`, `rotate`, `cov_explained` (share of squared cross-covariance captured by the `k` dimensions), `pvalue`
- `.pls_info` — `k`, `rotate`, `order`, `signs`, `sweeps`, `V_converged`, `kaiser_normalized`, `random_state`
- `.test` — LV1 test; call `.test(n_splits=..., n_perm=..., seed=...)` to rerun
- `.outcome_names`, `.outcome_groups`
- `.words`, `.clusters`, `.snippets` — fan out across dimensions
- `.report()`, `.attach(...)`

Leaf views (`result["dim-1"]`):

- `.outcomes` — columns `outcome`, `group`, `loading`; with `outcome_groups`, one extra row per group
- `.stats` — `cov_share` (share of the total cross-covariance of the `k` dimensions; sums to 1 over the dimensions), `score_corr` (in-sample correlation of the word-side and outcome-side scores)
- `.words`, `.clusters`, `.snippets`

`fit_multipls` needs `Embeddings` attached to the `SSD` (the per-dimension views and `rotate="words"` use the vocabulary). See [`examples/demo_multipls.py`](examples/demo_multipls.py) and [`docs/api_reference.md`](docs/api_reference.md).


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
# Number of components — the attribute name differs by backend:
result.n_components             # PLSResult only
result.pca_k                    # PCAOLSResult only
result.fit_info.n_components    # both backends (use this if the backend may vary)

# Comprehensive narrative report — every section is on by default; pass
# section=False to drop one.
print(result.report(clusters={"n": 100, "n_words": 10, "n_snippets": 2},
                    extreme_docs={"n": 30}, misdiagnosed={"n": 20}))
result.report().save("report.md")    # also .html / .docx / .tex

# Re-run the significance test in place
result.test(n_splits=200)            # PLSResult — overwrites stats.pvalue
```

For `MultiPLSResult` and `GroupResult`, see the "Several outcomes (PLS3)" and "Cross-Group Comparison" sections above.

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
# Not exported at top level:
from ssdiff.results.multi_pls_result import MultiPLSResult
```

### `Embeddings`

- `Embeddings.load(path="en", *, verbose=False, parallel=False)` — load `.wordpack`, `.kv`, `.bin`, `.txt`, `.vec` (and `.gz` variants), or a wordpack by short tag or full name
- `.normalize(l2=True, abtt=1)` — in-place L2 + ABTT (rows L2-normalized again after ABTT); tracks state, safe to call repeatedly
- `.save(filename=None, fmt="bin")` — save to binary, text, or gensim format
- `emb["word"]` / `emb.get_vector("word", norm=False)` — vector lookup
- `"word" in emb` — membership check
- `len(emb)` / `.vocab_size` — vocabulary size
- `.vector_size` (alias `.dim`) — embedding dimensionality
- `.similar_by_vector(vec, topn=10, restrict_vocab=None)` — nearest neighbor search

### `Corpus`

- `Corpus(texts, *, lang=None, model=None, nlp=None, stopwords=None, pretokenized=False, auto_download=None, embeddings=None)`
- `.docs` — lemmatized tokens per document
- `.pre_docs` — sentence-level structure for snippet extraction
- `.n_texts` — number of documents
- `.suggest_lexicon(y, *, top_k=30, ...)` -> `LexiconResult` — data-driven seed word selection
- `.evaluate_lexicon(y, lexicon, ...)` -> `LexiconResult` — score an existing lexicon (per-token + aggregate)

### `SSD`

- `SSD(embeddings, corpus, y, lexicon, *, window=3, sif_a=1e-3, use_full_doc=False)`
- `.fit_pls(*, k="auto", k_max=5, keep=None, n_splits=50, random_state=2137, verbose=False)` -> `PLSResult`
- `.fit_multipls(*, k, rotate=None, outcome_groups=None, n_splits=50, n_perm=1000, random_state=2137, verbose=False)` -> `MultiPLSResult`
- `.fit_ols(*, fixed_k=None, k_min=2, k_max=120, k_step=2, verbose=False)` -> `PCAOLSResult`
- `.fit_groups(*, median_split=False, n_perm=5000, correction="holm", random_state=2137, verbose=False)` -> `GroupResult`

### `PLSResult` / `PCAOLSResult`

**Direct array attributes**: `beta`, `gradient`, `beta_norm`, `alignment_scores`, `x`, `y`. The component count is named per backend: **PLS** exposes `n_components`, **PCA+OLS** exposes `pca_k` (both are also available as `result.fit_info.n_components`). PLS adds `component_scores`, `component_weights`, `find_k_result`, `cv_scores`. PCA+OLS adds `pca_components`, `pca_weights`, `sweep_result`.

**Scalar views** (all expose `.r2`, `.pvalue`, … as attributes; print to read, export with `.to_df()` / `.save(...)`):
- `.stats` — `backend`, `r2`, `r2_adj` (OLS only), `pvalue`, `n_raw`, `n_kept`, `n_dropped`, `y_mean`, `y_std`, `beta_norm`, `delta`, `iqr_effect`, `y_corr_pred`
- `.fit_info` — `n_components`, `p_at_k`, `n_splits`, `random_state`, `keep` (sparse PLS only), plus PCA-K sweep info for OLS

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

Plisiecki, H., Lenartowicz, P., Pokropek, A., Małyska, K., & Flakus, M. (2025). Measuring Individual Differences in Meaning: The Supervised Semantic Differential. PsyArXiv. https://doi.org/10.31234/osf.io/gvrsb_v1

---

## Questions / Contributions

- File issues and feature requests on the repo's Issues page.
- Pull requests welcome — especially for:
  - Robustness diagnostics and visualization helpers
  - Documentation improvements

Contact: hplisiecki@gmail.com

Project was funded by the National Science Centre, Poland (grant no. 2020/38/E/HS6/00302).
