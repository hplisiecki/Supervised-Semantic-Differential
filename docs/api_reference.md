# ssdiff — Usage Guide & API Reference

Complete workflow reference for the `ssdiff` Python package.

**Install**: `pip install ssdiff`
**Python**: 3.10+
**Core deps**: numpy, spacy, matplotlib

---

## Quick Start

```python
from ssdiff import Embeddings, Corpus, SSD

# 1. Load word embeddings
emb = Embeddings.load("glove_300d.txt")
emb.normalize(l2=True, abtt_m=1)

# 2. Tokenize texts
corpus = Corpus(texts, lang="en")

# 3. Build SSD and fit
ssd = SSD(emb, corpus, y=scores, lexicon=["happy", "sad", "joy", "anger"])
result = ssd.fit_pls()

# 4. Interpret
print(result.summary())
print(result.top_words(20))
```

---

## Step-by-Step Workflow

### Step 1: Load Embeddings

```python
emb = Embeddings.load(path, verbose=False, parallel=False)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `path` | `str` | — | Path to embedding file |
| `verbose` | `bool` | `False` | Print loading progress |
| `parallel` | `bool` | `False` | Multiprocess loading for `.txt`/`.vec` files |

**Supported formats**: `.ssdembed` (native pickle+npy, fastest), `.kv` (gensim; loads without gensim via internal shim), `.bin` (fastText binary), `.txt`/`.vec` (text), all with optional `.gz` compression.

#### Normalize embeddings

```python
emb.normalize(l2=True, abtt_m=1, re_normalize=True)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `l2` | `bool` | `True` | L2-normalize each word vector to unit length. Skipped if already applied. |
| `abtt_m` | `int` | `1` | Target number of top principal components to remove (ABTT). Absolute: if ABTT was already applied with a smaller m, only the remaining components are removed. 0 = skip. |
| `re_normalize` | `bool` | `True` | L2-normalize again after ABTT removal |

Returns `self` for chaining.

Processing state is tracked on the instance (`_l2_normalized`, `_abtt_m`) and persisted in `.ssdembed` files:
- **L2**: skipped if already applied.
- **ABTT**: absolute target — requesting m=3 on embeddings with m=1 already applied removes 2 more components. Requesting m < current raises `ValueError`. Requesting m == current is a no-op (warning).
- **`re_normalize`**: only runs if ABTT actually removed components.

> **Note:** Saving to non-`.ssdembed` formats (`.kv`, `.bin`, `.txt`) emits a warning that normalization and ABTT metadata will be lost.

#### Save embeddings

```python
emb.save(filename=None, fmt="ssdembed")
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `filename` | `str \| None` | `None` | Output path (without extension). Defaults to stem of source path |
| `fmt` | `str` | `"ssdembed"` | Format: `"ssdembed"`, `"kv"`, `"bin"`, `"txt"` |

#### Lookup & search

```python
vec = emb["word"]                         # Get vector by word
vec = emb.get_vector("word", norm=False)  # norm=True → L2-normalized
"word" in emb                             # Membership check
len(emb)                                  # Vocabulary size
emb.vector_size                           # Embedding dimensionality

emb.similar_by_vector(vec, topn=10, restrict_vocab=None)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `topn` | `int` | `10` | Number of nearest neighbors to return |
| `restrict_vocab` | `int \| None` | `None` | Only search the first N most frequent words |

Returns `list[tuple[str, float]]` — (word, cosine_similarity) pairs.

---

### Step 2: Prepare Corpus

```python
corpus = Corpus(
    texts,
    lang=None,
    model=None,
    nlp=None,
    stopwords=None,
    pretokenized=False,
)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `texts` | `Sequence[str \| list[str]]` | — | Raw text strings or pre-tokenized token lists |
| `lang` | `str \| None` | `None` | Language code (`"en"`, `"pl"`, `"de"`, etc.). Resolves to default spaCy model |
| `model` | `str \| None` | `None` | Explicit spaCy model name (e.g. `"en_core_web_sm"`). Overrides `lang` |
| `nlp` | `spacy.Language \| None` | `None` | Pre-loaded spaCy pipeline. Overrides both `lang` and `model` |
| `stopwords` | `Sequence[str] \| None` | `None` | Custom stopword list. If `None`, uses bundled/spaCy defaults |
| `pretokenized` | `bool` | `False` | If `True`, skip spaCy — `texts` are already token lists |

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `corpus.docs` | `list[list[str]]` | Lemmatized tokens per document (stopwords removed) |
| `corpus.pre_docs` | `list[PreprocessedDoc] \| None` | Sentence-level structure for snippet extraction |
| `corpus.lang` | `str` | Resolved language code |
| `corpus.n_texts` / `len(corpus)` | `int` | Number of documents |

**Supported languages**: ca, da, de, el, en, es, fr, hr, it, lt, mk, nb, nl, pl, pt, ro, ru, sl, sv, uk (20 total).

---

### Step 3: Create SSD Instance

```python
ssd = SSD(
    embeddings,
    corpus,
    y,
    lexicon,
    window=3,
    sif_a=1e-3,
    use_full_doc=False,
)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `embeddings` | `Embeddings` | — | Word embeddings instance |
| `corpus` | `Corpus` | — | Tokenized corpus (must match `y` length) |
| `y` | `array-like` | — | Outcome variable (float). NaN entries silently dropped with their docs |
| `lexicon` | `Sequence[str] \| set[str]` | — | Seed words for context-window extraction |
| `window` | `int` | `3` | Context window size in tokens around each seed word |
| `sif_a` | `float` | `1e-3` | SIF (Smooth Inverse Frequency) smoothing parameter. Must be > 0 |
| `use_full_doc` | `bool` | `False` | If `True`, use full-document vectors instead of seed-windowed contexts |

**Properties after construction**:

| Property | Type | Description |
|----------|------|-------------|
| `ssd.x` | `ndarray (n_kept, D)` | Document vectors |
| `ssd.y_kept` | `ndarray (n_kept,)` | Outcome for kept documents |
| `ssd.keep_mask` | `ndarray (n_raw,) bool` | Which original docs were kept |
| `ssd.n_raw` | `int` | Original document count |
| `ssd.n_kept` | `int` | Documents after preprocessing |
| `ssd.n_dropped` | `int` | Documents dropped (empty vectors or NaN y) |

---

### Step 4: Fit a Model

#### Option A: PLS (default, recommended)

```python
result = ssd.fit_pls(
    n_components=1,
    cv_folds=10,
    use_1se=True,
    pca_preprocess=None,
    p_method="auto",
    n_perm=1000,
    n_splits=50,
    split_ratio=0.5,
    random_state=2137,
    verbose=False,
)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `n_components` | `int \| "auto"` | `1` | Number of PLS components. `"auto"` selects via cross-validation |
| `cv_folds` | `int` | `10` | Number of CV folds for component selection |
| `use_1se` | `bool` | `True` | Use 1-SE rule for parsimonious component selection |
| `pca_preprocess` | `int \| str \| None` | `None` | Optional PCA dimensionality reduction before PLS. Int = component count, `"varNN"` = variance threshold (e.g. `"var95"` retains 95% variance) |
| `p_method` | `str \| None` | `"auto"` | Significance test method (see below) |
| `n_perm` | `int` | `1000` | Permutation iterations for `"perm"` and `"split_cal"` |
| `n_splits` | `int` | `50` | Number of random splits for `"split"` and `"split_cal"` |
| `split_ratio` | `float` | `0.5` | Training fraction for split-based tests |
| `random_state` | `int` | `2137` | Random seed for reproducibility |
| `verbose` | `bool` | `False` | Print progress |

**`p_method` options**:

| Value | Description |
|-------|-------------|
| `"auto"` | `"split"` when `n_components=1`, `"perm"` otherwise |
| `"perm"` | Permutation test on cross-validated R² |
| `"split"` | Split-half test with overlap-corrected t-test |
| `"split_cal"` | Permutation-calibrated split-half (exact FPR control, slower) |
| `None` | Skip significance testing (p-value = NaN) |

Returns → `PLSResult`.

#### Option B: PCA + OLS

```python
result = ssd.fit_ols(
    n_components=None,
    k_min=20,
    k_max=120,
    k_step=2,
    verbose=False,
)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `n_components` | `int \| None` | `None` | Number of PCA components. `None` = auto-select via interpretability+stability sweep |
| `k_min` | `int` | `20` | Minimum PCA-K for sweep |
| `k_max` | `int` | `120` | Maximum PCA-K for sweep |
| `k_step` | `int` | `2` | Step size for PCA-K sweep |
| `verbose` | `bool` | `False` | Print progress |

Returns → `PCAOLSResult`.

---

### Step 5: Inspect Results

Both `PLSResult` and `PCAOLSResult` share the same interpretation API.

#### Result attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `result.r2` | `float` | Coefficient of determination |
| `result.r2_adj` | `float` | Adjusted R² |
| `result.pvalue` | `float` | P-value (NaN if skipped) |
| `result.n_components` | `int` | Number of components used |
| `result.beta` | `ndarray` | Regression weight vector in embedding space |
| `result.beta_unit` | `ndarray` | Unit-length direction of beta |
| `result.beta_norm` | `float` | ‖beta‖ |
| `result.delta` | `float` | Predicted change in y per +0.10 cosine shift |
| `result.iqr_effect` | `float` | Predicted change in y across IQR of cosine alignment |
| `result.y_corr_pred` | `float` | \|Pearson r(y_true, y_pred)\| |
| `result.cos_align` | `ndarray` | Per-document cosine alignment to beta_unit |
| `result.y_mean` | `float` | Outcome mean (original scale) |
| `result.y_std` | `float` | Outcome SD (original scale) |

PLSResult-specific:

| Attribute | Type | Description |
|-----------|------|-------------|
| `result.p_method` | `str \| None` | Which significance test was used |
| `result.cv_scores` | `dict \| None` | Per-component CV R² (if `n_components="auto"`) |
| `result.perm_null` | `ndarray \| None` | Null distribution from permutation test |
| `result.split_mean_r` | `float \| None` | Mean Pearson r from split-half test |
| `result.pca_k` | `int \| None` | PCA components used for preprocessing |

PCAOLSResult-specific:

| Attribute | Type | Description |
|-----------|------|-------------|
| `result.sweep_result` | `object \| None` | Result from PCA-K sweep |

#### Sweep plot (PCAOLSResult only)

```python
result.plot_sweep(path=None, dpi=300)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `path` | `str \| None` | `None` | If given, save PNG to this path. If `None`, display interactively via `plt.show()` |
| `dpi` | `int` | `300` | Resolution |

Dual-axis plot: detrended interpretability (z-score, blue) and beta stability (smoothed 1 - cosine, orange), with a red vertical line at the selected best K.

Returns `bytes` (raw PNG) in all cases.

Raises `RuntimeError` if `fit_ols()` was called with an explicit `n_components` (no sweep data).

#### Summary

```python
print(result.summary())
```

Human-readable model summary with R², effect sizes, and p-value.

#### Comprehensive report

```python
print(result.report(top_words=10, clusters=100, extreme_docs=30, misdiagnosed=20))
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `top_words` | `int \| None` | `5` | Number of top words per side. `None` to skip |
| `clusters` | `int \| None` | `None` | Number of top neighbors to cluster (topn). `None` to skip |
| `extreme_docs` | `int \| None` | `None` | Number of extreme docs per side (top/bottom). `None` to skip |
| `misdiagnosed` | `int \| None` | `None` | Number of misdiagnosed docs per side. `None` to skip |

Prints and returns a comprehensive text report combining summary, top words, clusters, extreme docs, and misdiagnosed docs. Each section is only included if its argument is not `None`.

#### Semantic neighbors (top words)

```python
result.top_words(n=20)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `n` | `int` | `20` | Number of neighbors per pole |

Returns `list[dict]` with keys: `side` (`"pos"`/`"neg"`), `rank`, `word`, `cos`.

```python
result.neighbors(side="pos", n=20)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `side` | `str` | `"pos"` | `"pos"` for +beta neighbors, `"neg"` for -beta |
| `n` | `int` | `20` | Number of neighbors |

Returns `list[tuple[str, float]]` — (word, cosine) pairs.

#### Cluster neighbors

```python
result.cluster_neighbors(
    side="pos",
    topn=100,
    k=None,
    k_min=2,
    k_max=10,
    random_state=2137,
    min_cluster_size=2,
)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `side` | `str` | `"pos"` | Pole to cluster |
| `topn` | `int` | `100` | Size of candidate neighbor pool |
| `k` | `int \| None` | `None` | Fixed cluster count. `None` = auto-select via silhouette |
| `k_min` | `int` | `2` | Minimum k for auto-selection |
| `k_max` | `int` | `10` | Maximum k for auto-selection |
| `random_state` | `int` | `2137` | Random seed for K-Means |
| `min_cluster_size` | `int` | `2` | Discard clusters smaller than this |

Returns `list[dict]` with keys: `id`, `size`, `centroid_cos_beta`, `coherence`, `words`.

#### Document scores

```python
result.doc_scores()
```

Returns `dict` with keys:
- `keep_mask` — bool array (n_raw,), which docs were kept
- `cos_align` — float array (n_kept,), cosine alignment to beta_unit
- `score_std` — float array (n_kept,), standardized predicted scores
- `yhat_raw` — float array (n_kept,), predicted outcome in original scale

#### Extreme documents

```python
result.extreme_docs(k=50, by="predicted")
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `k` | `int` | `50` | Number of extremes per side |
| `by` | `str` | `"predicted"` | `"predicted"` or `"observed"` |

Returns `list[dict]` with keys: `idx`, `y_true`, `yhat`, `cos`, `side` (`"top"`/`"bottom"`).

#### Misdiagnosed documents

```python
result.misdiagnosed(k=20, side="both")
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `k` | `int` | `20` | Number of docs per side |
| `side` | `str` | `"both"` | `"both"`, `"over"` (model over-predicts), `"under"` |

Returns `list[dict]` with keys: `idx`, `y_true`, `yhat`, `cos`, `residual`, `side`.

#### Text snippets

```python
# Snippets along beta (all docs)
result.snippets(corpus.pre_docs, top_per_side=200)

# Snippets from extreme docs only
result.snippets_extreme(corpus.pre_docs, k=50, by="predicted", top_per_side=200)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `pre_docs` | `list[PreprocessedDoc]` | — | From `corpus.pre_docs` |
| `top_per_side` | `int` | `200` | Number of top snippets per side |
| `k` | `int` | `50` | (snippets_extreme) Number of extreme docs per side |
| `by` | `str` | `"predicted"` | (snippets_extreme) Ranking criterion |

Returns `dict` with keys `"pos"` and `"neg"`, each containing snippet metadata.

#### Re-run split-half test (PLSResult only)

```python
result.split_test(n_splits=50, split_ratio=0.5, seed=42, method="split", n_perm=200)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `n_splits` | `int` | `50` | Number of random splits |
| `split_ratio` | `float` | `0.5` | Training fraction |
| `seed` | `int` | `42` | Random seed |
| `method` | `str` | `"split"` | `"split"` or `"split_cal"` |
| `n_perm` | `int` | `200` | Permutations for `"split_cal"` |

**Mutates the result in place**, overwriting `result.pvalue`, `result.p_method`, `result.split_mean_r`, and the corresponding params (`n_splits`, `split_ratio`, `random_state`, and `n_perm` for `"split_cal"`). Clears `result.perm_null` (stale if the prior test was permutation). Returns `self` for convenience — read new values via `result.pvalue`, `result.split_mean_r`.

---

### Step 6: Group Analysis (Categorical)

Use `ssd.fit_groups()` to compare groups. Pass categorical labels as `y` when constructing SSD, or use `median_split=True` to split a continuous `y` at the median.

```python
# Categorical groups
ssd = SSD(emb, corpus, y=groups, lexicon=lexicon)
result = ssd.fit_groups(
    median_split=False,     # treat y as categorical labels (default)
    n_perm=5000,            # permutation iterations
    correction="holm",      # "holm", "bonferroni", "fdr_bh", or "none"
    random_state=2137,
)

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

Groups with fewer than 20 documents are automatically dropped (with a warning). If fewer than 2 groups remain, a `ValueError` is raised.

`fit_groups()` does **not** mutate `self.x` or `self.y_kept` — all filtering operates on local copies, so subsequent `fit_pls()`/`fit_ols()` calls are unaffected.

#### GroupResult attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `result.result_type` | `str` | Always `"group"` |
| `result.omnibus_T` | `float` | Mean pairwise cosine distance between centroids |
| `result.omnibus_p` | `float` | Permutation p-value for omnibus test |
| `result.group_labels` | `list` | Sorted unique group labels |
| `result.G` | `int` | Number of groups |
| `result.n_kept` | `int` | Docs used (after both preprocessing and small-group filtering) |
| `result.n_group_dropped` | `int` | Docs dropped by small-group filter |
| `result.pairwise` | `dict` | `(g1, g2)` → result dict (see below) |

Pairwise result dict keys: `T`, `p_raw`, `p_corrected`, `beta_unit`, `contrast_norm`, `cohens_d`, `n_g1`, `n_g2`.

#### GroupResult methods

```python
print(result.summary())         # Human-readable summary
print(result.report(top_words=10, clusters=100))  # Comprehensive report
result.results_table()          # list[dict] of pairwise results

# Interpretation (works across all contrasts, adds "contrast" key)
result.top_words(20)            # list[dict] with "contrast", "side", "rank", "word", "cos"
result.neighbors("pos", 10)
result.cluster_neighbors()
result.snippets(pre_docs)

# Filter to specific groups (returns new GroupResult, no recomputation)
r = result.filter_groups("A", "B")        # 1 contrast
r = result.filter_groups("A")             # all contrasts involving A
r = result.filter_groups("A", "B", "C")   # 3 contrasts (A-B, A-C, B-C)
```

`filter_groups()` returns a new `GroupResult` with subsetted pairwise results and doc vectors. The original omnibus stats are preserved for display.

---

## Utility: Lexicon Tools

### Suggest seed words

The recommended way is via `Corpus.suggest_lexicon()`, which uses the already-lemmatized tokens:

```python
corpus = Corpus(texts, lang="en")
candidates = corpus.suggest_lexicon(y, top_k=30)
ssd = SSD(emb, corpus, y, lexicon=candidates)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `y` | `array-like` | — | Outcome variable (numeric or categorical) |
| `top_k` | `int` | `30` | Maximum number of words to return |
| `min_docs` | `int` | `5` | Minimum document frequency |
| `n_bins` | `int` | `4` | Quantile bins for balanced coverage |
| `corr_cap` | `float` | `0.30` | Penalty threshold for outcome association |
| `var_type` | `str` | `"continuous"` | `"continuous"` or `"categorical"` |

Returns `list[str]` — tokens sorted by descending rank.

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

### Token presence stats

```python
from ssdiff.utils.lexicon import token_presence_stats

stats = token_presence_stats(
    texts, y, token,
    n_bins=4, corr_cap=0.30, verbose=False, var_type="continuous",
)
```

Returns `list[dict]` (single element) with keys: `token`, `frequency`, `association`, `pvalue`, `effect_direction`.

### Coverage analysis

```python
from ssdiff.utils.lexicon import coverage_by_lexicon

summary, per_token = coverage_by_lexicon(
    df_or_texts,
    text_col=None, score_col=None,
    lexicon=["word1", "word2"],
    n_bins=4, verbose=False, var_type="continuous",
)
```

Returns `(summary_dict, per_token_list)`:
- `summary`: `docs_any`, `cov_all`, `q1`, `q4`, `corr_any`, `hits_mean`, `hits_median`, `types_mean`, `types_median`
- `per_token`: list of dicts with `token`, `frequency`, `association`, `pvalue`, `effect_direction`

---

## Full Workflow Example

```python
import numpy as np
from ssdiff import Embeddings, Corpus, SSD
from ssdiff.utils.lexicon import suggest_lexicon, coverage_by_lexicon

# ── Data ──
texts = ["I feel very happy today", "This is so sad and depressing", ...]
scores = np.array([4.2, 1.8, ...])   # continuous outcome
groups = ["positive", "negative", ...]  # categorical labels

# ── Embeddings ──
emb = Embeddings.load("glove_300d.txt", verbose=True)
emb.normalize(l2=True, abtt_m=1)

# ── Corpus ──
corpus = Corpus(texts, lang="en")

# ── Lexicon selection ──
candidates = suggest_lexicon((corpus.docs, scores), top_k=100, min_docs=3)
summary, per_token = coverage_by_lexicon(
    (corpus.docs, scores), lexicon=candidates[:30], verbose=True,
)
lexicon = candidates[:20]  # pick top-20

# ── Continuous analysis (PLS) ──
ssd = SSD(emb, corpus, y=scores, lexicon=lexicon, window=3)
result = ssd.fit_pls(
    n_components="auto",
    p_method="split",
    n_splits=100,
    verbose=True,
)

print(result.summary())
print(f"R² = {result.r2:.4f}, p = {result.pvalue:.4g}")
print(f"Δy per +0.10 cos = {result.delta:.3f}")

# Top words on both poles
for w in result.top_words(15):
    print(f"  {w['side']:3s} #{w['rank']:2d}  {w['word']:20s}  cos={w['cos']:.3f}")

# Thematic clusters
for cl in result.cluster_neighbors("pos", topn=100, k_max=8):
    print(f"  Cluster {cl['id']}: {', '.join(cl['words'][:5])}  (coherence={cl['coherence']:.3f})")

# Document-level scores
scores_dict = result.doc_scores()
extremes = result.extreme_docs(k=30)

# Snippets
if corpus.pre_docs:
    snips = result.snippets(corpus.pre_docs, top_per_side=100)

# ── Continuous analysis (PCA+OLS) ──
result_ols = ssd.fit_ols(n_components=None, k_min=20, k_max=100, verbose=True)
print(result_ols.summary())
result_ols.plot_sweep()                  # display sweep plot
result_ols.plot_sweep("sweep.png")       # save to file

# ── Group analysis ──
ssd_g = SSD(emb, corpus, y=groups, lexicon=lexicon)
result_g = ssd_g.fit_groups(n_perm=5000, correction="holm")
print(result_g.summary())

for w in result_g.top_words(10):
    print(f"  {w['contrast']}  {w['side']:3s} #{w['rank']:2d}  {w['word']}")
```

---

## Pre-tokenized Input

If you've already tokenized your texts (e.g. with a custom pipeline):

```python
docs = [["happy", "day", "sunshine"], ["sad", "rain", "cold"], ...]
corpus = Corpus(docs, pretokenized=True, lang="en")
ssd = SSD(emb, corpus, y=scores, lexicon=lexicon)
```

## Custom spaCy Pipeline

```python
import spacy
nlp = spacy.load("en_core_web_lg", disable=["ner"])
corpus = Corpus(texts, nlp=nlp)
```

## Embedding Format Conversion

```python
emb = Embeddings.load("model.bin")       # Load fastText binary
emb.save("model_fast", fmt="ssdembed")   # Save as native format (fastest loading)
emb.save("model_fast", fmt="txt")        # Save as text format (portable)
```
