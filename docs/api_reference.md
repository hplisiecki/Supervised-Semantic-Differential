# ssdiff — API Reference

Public API of the `ssdiff` package: main classes, their methods, and arguments.

**Install**: `pip install ssdiff` (core), `pip install ssdiff[results]` (pandas/Excel/docx/matplotlib export).
**Python**: 3.10+.
**Imports**: `from ssdiff import Embeddings, Corpus, SSD`.

For result objects (views, exports, reports) see [`results.md`](results.md).
For per-view column listings and defaults see [`results_tables.md`](results_tables.md).
For internals see [`architecture.md`](architecture.md).

---

## Quick start

```python
from ssdiff import Embeddings, Corpus, SSD

emb = Embeddings.load("glove_300d.ssdembed")
corpus = Corpus(texts, lang="en")
ssd = SSD(emb, corpus, y=scores, lexicon=["happy", "sad", "joy", "anger"])

result = ssd.fit_ols()         # PCA + OLS
# result = ssd.fit_pls()       # PLS
# result = ssd.fit_groups()    # categorical y

print(result.stats)
print(result.words)
result.report().save("report.md")
```

---

## `Embeddings`

Stores word vectors, handles normalization, persists to disk.

### Load

```python
emb = Embeddings.load(path, *, verbose=False, parallel=False)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | — | Path to the embedding file. |
| `verbose` | `bool` | `False` | Show a tqdm progress bar when loading `.txt`/`.vec`/`.bin`. |
| `parallel` | `bool` | `False` | Multiprocess loading for `.txt`/`.vec`. |

Supported formats: `.ssdembed` (native pickle+npy, fastest), `.kv` (gensim pickle; no gensim required on load), `.bin` (fastText binary), `.txt`/`.vec` (text), all with optional `.gz` compression.

### Normalize

```python
emb.normalize(*, l2=True, abtt=1, re_normalize=True)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `l2` | `bool` | `True` | L2-normalize each row. Skipped if already applied. |
| `abtt` | `int` | `1` | Absolute target number of top principal components to remove (ABTT). `0` skips. Requesting a smaller value than currently applied raises `ValueError` (ABTT is irreversible). |
| `re_normalize` | `bool` | `True` | L2-normalize again after ABTT removal. |

Mutates in place and returns `self` for chaining. State is tracked on the instance (`emb.l2_normalized`, `emb.abtt`) and persisted by the `.ssdembed` format.

### Save

```python
emb.save(filename=None, fmt="ssdembed")
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `filename` | `str \| None` | `None` | Output path **without** extension. Defaults to the stem of the source path. |
| `fmt` | `str` | `"ssdembed"` | One of `"ssdembed"`, `"kv"`, `"bin"`, `"txt"`. |

Non-`ssdembed` formats emit a warning — normalization and ABTT metadata are lost.

### Lookup

```python
vec = emb["word"]                           # raw vector, KeyError if missing
vec = emb.get_vector("word", norm=False)    # norm=True → L2-normalized
"word" in emb                               # membership
len(emb)                                    # vocab size
emb.vector_size                             # dimensionality
emb.vocab_size                              # alias for len(emb)

emb.similar_by_vector(vec, topn=10, restrict_vocab=None)
# → list[tuple[str, float]]  (word, cosine similarity)
```

---

## `Corpus`

Tokenizes and lemmatizes text via spaCy; exposes lexicon helpers.

### Constructor

```python
corpus = Corpus(texts, *, lang=None, model=None, nlp=None,
                stopwords=None, pretokenized=False)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `texts` | `Sequence[str \| list[str]]` | — | Raw strings, or token lists if `pretokenized=True`. |
| `lang` | `str \| None` | `None` | Language code (`"en"`, `"pl"`, `"de"`, …). Resolves to the default spaCy model. |
| `model` | `str \| None` | `None` | Explicit spaCy model name. Overrides `lang`. |
| `nlp` | `spacy.Language \| None` | `None` | Pre-loaded spaCy pipeline. Overrides `lang` / `model`. |
| `stopwords` | `Sequence[str] \| None` | `None` | Custom stopword list. Falls back to bundled defaults for the resolved language. |
| `pretokenized` | `bool` | `False` | Skip spaCy — `texts` are already token lists. |

Attributes:

| Property | Type | Description |
|---|---|---|
| `corpus.docs` | `list[list[str]]` | Lemmatized tokens per document (stopwords removed). |
| `corpus.pre_docs` | `list[PreprocessedDoc] \| None` | Sentence-level structure for snippet extraction. |
| `corpus.lang` | `str \| None` | Resolved language code. |
| `corpus.n_texts`, `len(corpus)` | `int` | Document count. |

Supported languages out of the box: ca, da, de, el, en, es, fr, hr, it, lt, mk, nb, nl, pl, pt, ro, ru, sl, sv, uk.

### Lexicon helpers

All three return a `LexiconResult` (see [`results.md`](results.md)).

```python
corpus.suggest_lexicon(y, *, top_k=30, min_docs=5, n_bins=4,
                       corr_cap=0.30, var_type="continuous")

corpus.evaluate_lexicon(y, lexicon, *, n_bins=4, corr_cap=0.30,
                        var_type="continuous")

corpus.token_stats(y, lexicon, *, n_bins=4, corr_cap=0.30,
                   var_type="continuous")   # → list[dict]

corpus.coverage_summary(y, lexicon, *, n_bins=4,
                        var_type="continuous")  # → dict
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `y` | array-like | — | Outcome aligned with `corpus.docs`. NaN / `None` filtered. |
| `top_k` | `int` | `30` | Maximum candidates to return (`suggest_lexicon`). |
| `min_docs` | `int` | `5` | Minimum document frequency. |
| `n_bins` | `int` | `4` | Quantile bins for balanced coverage (continuous). |
| `corr_cap` | `float` | `0.30` | Cap applied to `|corr|` in the rank penalty. |
| `var_type` | `str` | `"continuous"` | `"continuous"` or `"categorical"`. |

---

## `SSD`

Builds personal concept vectors (PCVs) from the corpus + lexicon, then fits a backend.

### Constructor

```python
ssd = SSD(embeddings, corpus, y, lexicon, *,
          window=3, sif_a=1e-3, use_full_doc=False)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `embeddings` | `Embeddings` | — | Word-embedding model. |
| `corpus` | `Corpus` | — | Tokenized corpus aligned with `y`. |
| `y` | array-like | — | Outcome. Numeric for `fit_pls` / `fit_ols`; categorical labels accepted for `fit_groups`. NaN / invalid entries drop their documents silently. |
| `lexicon` | `Sequence[str] \| set[str]` | — | Seed words for context-window extraction. |
| `window` | `int` | `3` | Context window size (tokens) around each seed. |
| `sif_a` | `float` | `1e-3` | SIF smoothing. Must be > 0. |
| `use_full_doc` | `bool` | `False` | Use full-document vectors instead of seed-windowed contexts. |

Attributes after construction:

| Property | Type | Description |
|---|---|---|
| `ssd.x` | `ndarray (n_kept, D)` | Document vectors. |
| `ssd.y` | `ndarray (n_kept,)` | Outcome aligned with retained documents. |
| `ssd.n_raw`, `ssd.n_kept`, `ssd.n_dropped` | `int` | Document counts. |
| `ssd.lexicon`, `ssd.window`, `ssd.sif_a`, `ssd.lang` | — | Echoes of constructor input. |

### `fit_ols()` — PCA + OLS

```python
result = ssd.fit_ols(*,
    fixed_k=None,
    k_min=2, k_max=120, k_step=2,
    verbose=False,
)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `fixed_k` | `int \| None` | `None` | Fixed number of PCA components. `None` selects K via the interpretability + stability sweep. |
| `k_min`, `k_max`, `k_step` | `int` | `2`, `120`, `2` | Sweep range (ignored when `fixed_k` is set). |
| `verbose` | `bool` | `False` | Print progress. |

Returns a `PCAOLSResult`. Significance is reported as the OLS F-test. Sweep diagnostics and `result.plot_sweep(path=None, dpi=300)` are available when `fixed_k is None`.

### `fit_pls()` — PLS1 NIPALS

```python
result = ssd.fit_pls(*,
    n_components=1,
    cv_folds=10,
    pca_preprocess=None,
    p_method="auto",
    n_perm=1000, n_splits=50, split_ratio=0.5,
    random_state=2137, verbose=False,
)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `n_components` | `int \| "auto"` | `1` | Number of PLS components. `"auto"` picks argmax mean CV R² over `cv_folds`-fold CV. |
| `cv_folds` | `int` | `10` | CV folds for `"auto"` component selection. |
| `pca_preprocess` | `int \| str \| None` | `None` | Pre-PLS PCA reduction. Int = component count; `"var95"` retains 95 % of variance. |
| `p_method` | `str \| None` | `"auto"` | `"perm"`, `"split"`, `"split_cal"`, `"auto"` (= `"split"` for 1 component, `"perm"` otherwise), or `None` to skip. |
| `n_perm` | `int` | `1000` | Permutations for `"perm"` / `"split_cal"`. |
| `n_splits` | `int` | `50` | Splits for `"split"` / `"split_cal"`. |
| `split_ratio` | `float` | `0.5` | Train fraction for split-based tests. |
| `random_state` | `int` | `2137` | Seed. |
| `verbose` | `bool` | `False` | Print progress. |

Returns a `PLSResult`. Re-run the significance test on a result via `result.test(...)` — see [`results.md`](results.md).

### `fit_groups()` — categorical outcome

```python
result = ssd.fit_groups(*,
    median_split=False,
    n_perm=5000,
    correction="holm",
    random_state=2137, verbose=False,
)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `median_split` | `bool` | `False` | Split continuous `y` at the median into `"low"`/`"high"`. |
| `n_perm` | `int` | `5000` | Permutations for omnibus + pairwise tests. |
| `correction` | `str` | `"holm"` | `"holm"`, `"bonferroni"`, `"fdr_bh"`, or `"none"`. |
| `random_state` | `int` | `2137` | Seed. |
| `verbose` | `bool` | `False` | Print progress. |

Groups with < 20 documents are dropped with a warning. Fewer than 2 surviving groups raises `ValueError`. `fit_groups()` does not mutate `ssd.x` / `ssd.y`, so continuous fits remain callable on the same instance.

Returns a `GroupResult`.

---

## Results — overview

Every fit method returns a result object with a consistent surface:

```python
result.stats                # headline metrics (StatsView / GroupStatsView / …)
result.report(...)          # narrative Report — .to_text() / .to_html() / .save(...)
result.save(...)            # — not defined; export per-view instead
result.words / .clusters / .snippets / .docs    # continuous; also .words/.clusters/.snippets on GroupResult
result.pairs                # groups only — PairsListView of Pair dataclass rows
gr[('g1', 'g2')].words      # idiomatic per-pair drill-down (also .clusters, .snippets, .stats, …)
gr.pairs[('g1', 'g2')]      # raw Pair dataclass (fields: T, p_raw, p_corrected, cohens_d, contrast, …)
result.suggestions / .tokens / .summary         # lexicon only
result.attach(corpus=..., embeddings=...)       # re-wire after un-pickling
result.clear_cache()
```

### Reports

`result.report(...)` builds a `Report`; extension dispatch picks the format.

```python
r = result.report(top_words=10, clusters=50, extreme_docs=5, misdiagnosed=5)
r.to_text()         # plain text
r.to_markdown()     # markdown source
r.to_html()         # styled HTML
r.save("report.md")   # .md .txt .html .tex .docx .json
```

### Lexicon suggestions (also a result)

```python
lex = corpus.suggest_lexicon(scores, top_k=30)
lex.tokens           # list[str] — ranked tokens, ready to pass back to SSD
lex.suggestions      # per-token table
lex.report().save("lexicon.md")

eval_ = corpus.evaluate_lexicon(scores, lex.tokens[:10])
eval_.summary        # aggregate coverage
```

For the full results surface (views, to_dict / to_df / to_records, per-view `save`, callable reruns, cache behaviour) see [`results.md`](results.md).

---

## Full example

```python
import numpy as np
from ssdiff import Embeddings, Corpus, SSD

texts  = [...]                     # list[str]
scores = np.array([...])           # continuous outcome
groups = [...]                     # categorical labels

emb = Embeddings.load("glove_300d.ssdembed").normalize(l2=True, abtt=1)
corpus = Corpus(texts, lang="en")

# Lexicon selection
lex = corpus.suggest_lexicon(scores, top_k=30)
lexicon = lex.tokens[:20]

# Continuous fit — PCA+OLS
ssd = SSD(emb, corpus, y=scores, lexicon=lexicon, window=3)
ols = ssd.fit_ols()
ols.plot_sweep("sweep.png")
ols.report(top_words=10, clusters=50).save("report_ols.md")

# Continuous fit — PLS on the same SSD
pls = ssd.fit_pls(n_components="auto", p_method="split")
pls.report(top_words=10, clusters=50).save("report_pls.md")

# Group comparison
ssd_g = SSD(emb, corpus, y=groups, lexicon=lexicon)
gr = ssd_g.fit_groups(n_perm=5000, correction="holm")
gr.report().save("report_groups.md")
```
