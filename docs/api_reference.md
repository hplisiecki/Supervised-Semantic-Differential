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
print(result.stats)
print(list(result.words)[:20])
result.report().to_text()
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
    fixed_k=None,
    k_min=2,
    k_max=120,
    k_step=2,
    verbose=False,
)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `fixed_k` | `int \| None` | `None` | Fixed number of PCA components. `None` = auto-select via interpretability+stability sweep |
| `k_min` | `int` | `2` | Minimum PCA-K for sweep |
| `k_max` | `int` | `120` | Maximum PCA-K for sweep |
| `k_step` | `int` | `2` | Step size for PCA-K sweep |
| `verbose` | `bool` | `False` | Print progress |

Returns → `PCAOLSResult`.

---

### Step 5: Inspect Results

Both `PLSResult` and `PCAOLSResult` share the same view-based interpretation API. See [`results.md`](results.md) for the full view / export / report reference.

#### Stats

```python
result.stats.r2          # float — coefficient of determination
result.stats.r2_adj      # float — adjusted R²
result.stats.pvalue      # float — p-value (NaN if skipped)
result.stats.n_kept      # int — documents used
result.stats.cos_align   # float — mean cosine alignment
```

PLSResult-specific attributes on `result.stats`: `p_method`, `split_mean_r`, `pca_k`.

PCAOLSResult-specific: `result.sweep` view (PCA-K sweep table).

#### Words

```python
list(result.words)[:20]                     # first 20 Word objects
result.words.to_dict()                      # list[dict], always stdlib-only
result.words.df()                           # pandas DataFrame (requires [results])
result.words.to_csv("words.csv")
```

#### Clusters

```python
result.clusters.pos                         # SidedClustersView (defaults: topn=100, k=auto)
result.clusters.neg
result.clusters.pos.recompute(topn=50, k=5) # recompute with different params
result.clusters.pos[0].words                # ClusterWordsView for cluster 0
result.clusters.pos.to_csv("clusters_pos.csv")
result.clusters.pos.to_markdown("clusters_pos.md")   # requires [results]
```

#### Snippets

After `result.attach(corpus=corpus)`:

```python
result.snippets                             # SnippetsView (default top_per_side=200)
result.snippets.recompute(top_per_side=500)
result.snippets.df()
```

#### Docs

```python
result.docs.top(5)       # DocsView of 5 highest-predicted docs
result.docs.bottom(5)    # DocsView of 5 lowest-predicted docs
result.docs.to_dict()
result.docs.df()
```

#### Report

```python
r = result.report(top_words=10, clusters=50)
r.to_text()          # plain terminal output
r.to_markdown()      # markdown source
r.to_html()          # styled HTML
r.save("report.md")  # extension dispatch: .txt .md .html .docx
```

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

Raises `RuntimeError` if `fit_ols()` was called with an explicit `fixed_k` (no sweep data).

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

#### GroupResult views and methods

```python
gr.stats.omnibus_T          # float — mean pairwise cosine distance between centroids
gr.stats.omnibus_p          # float — permutation p-value
gr.stats.G                  # int — number of groups
gr.stats.n_kept             # int — docs used

gr.pairs                    # PairsView — mapping-like, iterate or index by contrast
gr.pairs["A","B"]           # PairView — single-contrast result (same interface as ContinuousResult)
gr.pairs.df()               # pairwise table as DataFrame

gr.words                    # flat WordsView across all contrasts (with "contrast" column)
gr.clusters.pos             # flat clusters across all contrasts
gr.snippets                 # flat snippets across all contrasts (after attach(corpus=...))
```

Reports:

```python
gr.report(top_words=10, clusters=50).to_text()   # loops over all contrasts automatically
gr.report().save("group_report.md")
```

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
lex_result = corpus.suggest_lexicon(scores, top_k=100, min_docs=3)
lexicon = lex_result.tokens[:20]  # pick top-20

# ── Continuous analysis (PLS) ──
ssd = SSD(emb, corpus, y=scores, lexicon=lexicon, window=3)
result = ssd.fit_pls(
    n_components="auto",
    p_method="split",
    n_splits=100,
    verbose=True,
)

print(result.stats.r2, result.stats.pvalue)

# Top words on both poles
for w in list(result.words)[:15]:
    print(f"  {w.side:3s} #{w.rank:2d}  {w.word:20s}  cos={w.cos_beta:.3f}")

# Thematic clusters
for cl in result.clusters.pos.recompute(topn=100, k_max=8):
    words_preview = ", ".join(cw.word for cw in list(cl.words)[:5])
    print(f"  Cluster {cl.cluster_id}: {words_preview}  (coherence={cl.coherence:.3f})")

# Document-level scores
for doc in result.docs.top(30):
    print(doc.y_true, doc.y_hat, doc.cos_align)

# Snippets (after attaching corpus)
result.attach(corpus=corpus)
for snip in list(result.snippets)[:5]:
    print(snip.side, snip.text_surface)

# Report
result.report(top_words=10, clusters=50).save("report.md")

# ── Continuous analysis (PCA+OLS) ──
result_ols = ssd.fit_ols(fixed_k=None, k_min=2, k_max=100, verbose=True)
result_ols.plot_sweep()           # display sweep plot
result_ols.plot_sweep("sweep.png")  # save to file

# ── Group analysis ──
ssd_g = SSD(emb, corpus, y=groups, lexicon=lexicon)
result_g = ssd_g.fit_groups(n_perm=5000, correction="holm")
result_g.report(top_words=10, clusters=50).to_text()

for w in list(result_g.words)[:10]:
    print(f"  {w.contrast}  {w.side:3s} #{w.rank:2d}  {w.word}")
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
