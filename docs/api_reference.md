# ssdiff — API Reference

Public API of the `ssdiff` package: main classes, their methods, and arguments.

**Install**: `pip install ssdiff` (core), `pip install ssdiff[results]` (pandas/Excel/docx/matplotlib export).
**Python**: 3.10 – 3.14.
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
# result = ssd.fit_multipls(k=2)  # several outcomes (y as a DataFrame), PLS3

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

Both return a `LexiconResult` (see [`results.md`](results.md)).

```python
corpus.suggest_lexicon(y, *, top_k=30, min_docs=5, n_bins=4,
                       corr_cap=0.30, var_type="continuous")

corpus.evaluate_lexicon(y, lexicon, *, n_bins=4, corr_cap=0.30,
                        var_type="continuous")
```

`suggest_lexicon` ranks corpus tokens for use as seeds; `evaluate_lexicon` scores an existing lexicon, returning per-token diagnostics (`.suggestions`) and aggregate coverage (`.summary`).

| Argument | Type | Default | Description |
|---|---|---|---|
| `y` | array-like | — | Outcome aligned with `corpus.docs`. NaN / `None` filtered. |
| `lexicon` | iterable[str] | — | Tokens to score (`evaluate_lexicon`). |
| `top_k` | `int` | `30` | Maximum candidates to return (`suggest_lexicon`). |
| `min_docs` | `int` | `5` | Minimum document frequency (`suggest_lexicon`). |
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
| `y` | array-like | — | Outcome. 1-D numeric for `fit_pls` / `fit_ols`; 1-D categorical labels accepted for `fit_groups`. A DataFrame or 2-D array `(n, q)` of numeric outcomes is for `fit_multipls` only (`SSD.outcome_names` holds the column names). NaN / invalid entries drop their documents silently; with several outcomes a row is dropped if any outcome is missing. |
| `lexicon` | `Sequence[str] \| set[str]` | — | Seed words for context-window extraction. |
| `window` | `int` | `3` | Context window size (tokens) around each seed. |
| `sif_a` | `float` | `1e-3` | SIF smoothing. Must be > 0. |
| `use_full_doc` | `bool` | `False` | Use full-document vectors instead of seed-windowed contexts. |

Attributes after construction:

| Property | Type | Description |
|---|---|---|
| `ssd.x` | `ndarray (n_kept, D)` | Document vectors. |
| `ssd.y` | `ndarray (n_kept,)` or `(n_kept, q)` | Outcome aligned with retained documents; 2-D for several outcomes. |
| `ssd.outcome_names` | `list[str] \| None` | Outcome column names for 2-D `y` (DataFrame columns, else `"y1"`, …); `None` for 1-D `y`. |
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

### `fit_pls()` — PLS1

```python
result = ssd.fit_pls(*,
    k="auto",
    k_max=5,
    keep=None,
    n_splits=50,
    random_state=2137, verbose=False,
)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `k` | `int \| "auto"` | `"auto"` | Number of PLS components. `int` fits at exactly that k. `"auto"` calls `plskit.pls1_find_k_optimal` (selector `r2_se`). The p-value is the `split_exact` test (permutation-calibrated split-half) at the fitted `k`; with `k="auto"` it is the k=1 `split_exact` test. `split_nb` is only the diagnostic chain used to pick `k`. |
| `k_max` | `int` | `5` | Cap for `k="auto"`, clamped to `min(k_max, n-1, D)`. Ignored when `k` is an int. |
| `keep` | `int \| "auto" \| None` | `None` | Sparse-PLS knob. `None` is dense PLS. An `int` in `[1, D]` gives each component exactly that many non-zero weights. `"auto"` cross-validates the keep-count at the fixed `k` (raises if none is found). Sparse mode needs an explicit integer `k`; `keep` with `k="auto"` raises `ValueError`. |
| `n_splits` | `int` | `50` | Random splits for the split-half tests. |
| `random_state` | `int` | `2137` | Seed. |
| `verbose` | `bool` | `False` | Print progress. |

With `keep` set, the p-value is the largest p-value in the sequential chain of `k` tests (`result.test.name` is `"split_nb_seq"`). It is a different test from the dense p-value, so do not compare the two. The chain is computed at fit time, so `result.test()` raises `RuntimeError` on a sparse result. To change `n_splits` or the seed, refit. `result.fit_info.keep` holds the keep-count used (`None` for dense fits).

Returns a `PLSResult`. Re-run the significance test on a dense result via `result.test(...)` — see [`results.md`](results.md).

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

### `fit_multipls()` — several outcomes (PLS3)

Fits the document vectors against several numeric outcomes at once. `SSD` must be built with a DataFrame or 2-D array `y` with at least two columns; for one outcome use `fit_pls`. Needs attached `Embeddings`.

PLS3 takes the SVD of the cross-covariance between document vectors and outcomes. Dimension `i` pairs one word-space direction with one vector of outcome loadings.

```python
result = ssd.fit_multipls(*,
    k,
    rotate=None,                     # None | "words" | "outcomes"
    outcome_groups=None,             # {"group": ["outcome_a", "outcome_b"], ...}
    n_splits=50, n_perm=1000,
    random_state=2137, verbose=False,
)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `k` | `int` | — | Number of dimensions. Required, `1 <= k <= min(q, n-1, D)`. At `k = q` every component is kept, so `cov_explained` is 1 by construction. Raises if plskit keeps fewer than `k` components. |
| `rotate` | `None \| "words" \| "outcomes"` | `None` | `None` keeps the SVD dimensions. `"words"`: varimax against the projection of the leading vocabulary rows (up to 50,000) onto the word-space directions. `"outcomes"`: varimax against the outcome loadings, or the group rows when `outcome_groups` is set; needs `k < q`. |
| `outcome_groups` | `dict[str, list[str]] \| None` | `None` | Only with `rotate="outcomes"`. Each group is one rotation target row, the mean of its outcomes' loadings. Outcomes in no group keep their own row. An unknown name, an outcome in two groups, or an empty group raises. `k` cannot exceed the number of target rows. |
| `n_splits` | `int` | `50` | Random splits for the LV1 test. |
| `n_perm` | `int` | `1000` | Permutations for the LV1 test. |
| `random_state`, `verbose` | — | `2137`, `False` | Seed and progress output. |

A group's rotation target row and its `.outcomes` group loading are plain means of the group's outcome loadings. Outcomes scored in opposite directions cancel within a group.

Varimax modes reorder the dimensions by their share of the cross-covariance. In every mode, each dimension's sign is set so its largest outcome loading is positive.

Returns a `MultiPLSResult`:

- `result["dim-1"]`, …, `result["dim-k"]` — one leaf per dimension
- `result["dim-1"].outcomes` — outcome loadings (`outcome`, `group`, `loading`); with `outcome_groups`, one extra row per group
- `result["dim-1"].words`, `.clusters`, `.snippets` — neighbours along the dimension
- `result["dim-1"].stats` — `cov_share`, `score_corr`
- `.stats` — `n`, `q`, `k`, `rotate`, `cov_explained`, `pvalue`
- `.pls_info` — `k`, `rotate`, `order`, `signs`, `sweeps`, `V_converged`, `kaiser_normalized`, `random_state`
- `.test` — split_exact PLS3 test at LV1, container level only; rerun with `.test(n_splits=..., n_perm=..., seed=...)`
- `.report(top_words={"n": 5})` — stats, rotation, per-dimension outcome and words tables

The p-value tests the first dimension (LV1) and does not depend on `rotate`. Leaves have no p-value.

Runnable example: [`examples/demo_multipls.py`](../examples/demo_multipls.py).

---

## Results — overview

Every fit method returns a result object with a consistent surface:

```python
result.stats                # headline metrics (StatsView / GroupStatsView / …)
result.report(...)          # narrative Report — .to_text() / .to_html() / .save(...)
result.save(...)            # — not defined; export per-view instead
result.words / .clusters / .snippets / .docs    # continuous; also .words/.clusters/.snippets on GroupResult
result.pairs                # groups only — PairsListView of Pair dataclass rows
gr[('g1', 'g2')].words      # idiomatic per-pair zoom (also .clusters, .snippets, .stats, …)
gr.pairs[('g1', 'g2')]      # raw Pair dataclass (fields: T, p_raw, p_corrected, cohens_d, contrast, …)
result.suggestions / .tokens / .summary         # lexicon only
result.attach(corpus=..., embeddings=...)       # re-wire after un-pickling
result.clear_cache()
```

### Reports

`result.report(...)` builds a `Report`; extension dispatch picks the format.
**Clusters are the main element**, with words and a representative excerpt
rendered inside each cluster row. Every section is on by default — pass
`section=False` to drop one. The Stats / Fit-info block (PLS / PCA+OLS) and
the Omnibus / Group-labels / Pairwise-contrasts block (`GroupResult`) are
always rendered. Section arguments accept `True` / `False` / `None` / `dict`
— never `int`.

```python
# Defaults — every section on
result.report()

# Customise the cluster table; drop the misdiagnosed section
r = result.report(
    clusters={"n": 50, "n_words": 10, "n_snippets": 2},
    misdiagnosed=False,
)
r.to_text()         # plain text
r.to_html()         # styled HTML
r.save("report.md") # .md .txt .html .tex .docx .json
```

`clusters` dict keys (with defaults): `n=10` (clusters per side), `n_words=5`
(words listed in each row), `n_snippets=1` (representative excerpts per row;
`0` drops the column, values > 1 join with `" / "`).

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
ols.report(clusters={"n": 50, "n_words": 10, "n_snippets": 2}).save("report_ols.md")

# Continuous fit — PLS on the same SSD
pls = ssd.fit_pls(k="auto")
pls.report(clusters={"n": 50, "n_words": 10, "n_snippets": 2}).save("report_pls.md")

# Group comparison
ssd_g = SSD(emb, corpus, y=groups, lexicon=lexicon)
gr = ssd_g.fit_groups(n_perm=5000, correction="holm")
gr.report().save("report_groups.md")
```
