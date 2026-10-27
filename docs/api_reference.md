# ssdiff — API Reference

Public API of the `ssdiff` package: main classes, their methods, and arguments.

**Install**: `pip install ssdiff` (core), `pip install ssdiff[results]` (pandas/Excel/docx/matplotlib export).
**Python**: 3.11 – 3.15.
**Imports**: `from ssdiff import Embeddings, SSD, fit_regression, contrast`.

For choosing and loading embeddings see [`embeddings.md`](embeddings.md).
For result objects (views, exports, reports) see [`results.md`](results.md).
For per-view column listings and defaults see [`results_tables.md`](results_tables.md).
For internals see [`architecture.md`](architecture.md).

---

## Quick start

```python
from ssdiff import Embeddings, SSD

emb = Embeddings.load()                    # the default English wordpack, downloaded once
ssd = SSD(df["text"], embeddings=emb, lexicon=["happy", "sad", "joy", "anger"])   # one text, no outcome

result = ssd.fit_regression("score", data=df)   # PCA regression
# result = ssd.fit_pls("score", data=df)        # PLS
# result = ssd.fit_groups("condition", data=df) # categorical outcome
# result = ssd.fit_multipls(["a", "b"], data=df) # several outcomes, PLS3

print(result.stats)
print(result.words)
result.report().save("report.md")
```

---

## `Embeddings`

Stores word vectors, handles normalization, persists to disk. [`embeddings.md`](embeddings.md) is the guide to the available packs and to loading your own file.

### Load

```python
emb = Embeddings.load(path="default", *, verbose=False, parallel=False)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | `"default"` | A wordpack's tag (`"en"`), a path to an embedding file, or a wordpack's full name (`"en-glove2024-dolma-1"`), tried in that order. The tags are listed in `ssdiff.lang_config.WORDPACK_TAGS`. A pack is downloaded once into the cache directory (`WORDPACK_CACHE`, or `~/.wordpacks`). A name that matches none of the three raises `FileNotFoundError` listing the tags. |
| `verbose` | `bool` | `False` | Show a tqdm progress bar when loading `.txt`/`.vec`/`.bin`. |
| `parallel` | `bool` | `False` | Multiprocess loading for `.txt`/`.vec`. |

Supported formats: `.wordpack` (tokenizer, stopwords and ABTT included; takes strings), `.kv` (gensim pickle; no gensim required on load), `.bin` (fastText binary), `.txt`/`.vec` (text), all with optional `.gz` compression.

### Normalize

```python
emb.normalize(*, l2=True, abtt=1)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `l2` | `bool` | `True` | L2-normalize each row. Skipped if already applied. |
| `abtt` | `int` | `1` | Absolute target number of top principal components to remove (ABTT). `0` skips. Requesting a smaller value than currently applied raises `ValueError` (ABTT is irreversible). On a wordpack, any value 0 to 10, up or down; `l2=False` raises. |

Rows are L2-normalized again after ABTT removal.

Mutates in place (a wordpack gets a new `vectors` array) and returns `self` for chaining. State is tracked on the instance (`emb.l2_normalized`, `emb.abtt`).

### Save

```python
emb.save(filename=None, fmt="bin")
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `filename` | `str \| None` | `None` | Output path **without** extension. Defaults to the stem of the source path. |
| `fmt` | `str` | `"bin"` | One of `"bin"`, `"kv"`, `"txt"`. |

Saving normalized embeddings emits a warning — normalization and ABTT metadata are lost.

### Lookup

```python
vec = emb["word"]                           # raw vector, KeyError if missing
vec = emb.get_vector("word", norm=False)    # norm=True → L2-normalized
"word" in emb                               # membership
len(emb)                                    # vocab size
emb.vector_size                             # dimensionality
emb.vocab_size                              # same as len(emb)

emb.similar_by_vector(vec, topn=10, restrict_vocab=None)
# → list[tuple[str, float]]  (word, cosine similarity)
```

---

## `SSD`

Holds one text, tokenized once, and builds personal concept vectors (PCVs) from it and a lexicon on first use. It knows nothing about an outcome: every fit names its own. Several fits, with different outcomes or backends, run on the same `SSD` without rebuilding the vectors.

### Constructor

```python
ssd = SSD(texts, *, embeddings=None, lexicon=None,
          window=3, sif_a=1e-3, use_full_doc=None)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `texts` | sequence of `str`, or sequence of token lists | — | One entry per document. Strings are tokenized with the wordpack's tokenizer and stopwords; an empty, `None` or NaN entry is an empty document. Token lists are used as given and are the way to use any embedding that is not a wordpack. A pandas Series gives the text its name. |
| `embeddings` | `Embeddings` or `None` | `None` | L2-normalized word embeddings. Strings need a wordpack. `None` loads `Embeddings.load()`, the `"default"` wordpack tag, which prints its tag message to stderr. |
| `lexicon` | `Sequence[str] \| set[str] \| None` | `None` | Seed words for context-window extraction. The lexicon helpers do not need it. |
| `window` | `int` | `3` | Context window size (tokens) around each seed. |
| `sif_a` | `float` or `None` | `1e-3` | SIF smoothing. Must be > 0; `None` turns SIF off (plain mean of word vectors). |
| `use_full_doc` | `bool` or `None` | `None` | `True`: full-document vectors. `False`: seed-windowed contexts, which need a lexicon. `None` means `True` without a lexicon and `False` with one. |

A single string as `texts` raises `TypeError`, and so do strings with embeddings that are not a wordpack. Embeddings that are not L2-normalized raise `RuntimeError`; call `emb.normalize()`.

```python
ssd = SSD(df["essay"])                               # default wordpack, full documents
ssd = SSD(df["essay"], embeddings=emb, lexicon=lexicon)   # strings, wordpack embeddings
ssd = SSD(token_lists, embeddings=emb, lexicon=lexicon)   # token lists, any embeddings
ssd2 = ssd.with_lexicon(other_lexicon)               # same tokens, another lexicon
```

`ssd.with_lexicon(lexicon, *, window=3, sif_a=1e-3, use_full_doc=False)` returns a new `SSD` on the same tokens with another lexicon and settings. The text is not tokenized again.

Attributes:

| Property | Type | Description |
|---|---|---|
| `ssd.docs` | `list[list[str]]` | Tokens of each document. |
| `ssd.pre_docs` | list or `None` | Sentence structure for snippets; `None` for token-list texts. |
| `ssd.name` | `str \| None` | The text's column name when `texts` is a pandas Series. |
| `ssd.tokenizer` | `str \| None` | `"wordpack:<name>-<version>"` for strings, `None` for token lists. |
| `ssd.embeddings` | `Embeddings` | The embeddings the vectors are built in. |
| `ssd.lexicon`, `ssd.window`, `ssd.sif_a`, `ssd.use_full_doc` | — | Echoes of the settings. |
| `ssd.n_kept`, `len(ssd)` | `int` | Documents that have a vector, and documents in the text. |

### The outcome argument

The first argument of every fit is the outcome, and each fit takes `data=` (a DataFrame or a `dict` of 1-D arrays with one row per document, in the order of the texts):

- **Values**: an array, list or Series with one value per document.
- **A column name**: `"score"`, with `data=df`.
- **A formula** (`fit_regression`, `fit_pls`): `"score ~ age + gender"`, with `data=df`. The left side is the outcome column.
- **Several outcomes** (`fit_multipls`): a list of column names, a DataFrame, or a 2-D array.

A document drops out of a fit when its outcome is missing or when it has no vector. Column names that start with `_ssd_` are reserved.

### `fit_regression()` — PCA regression

```python
result = ssd.fit_regression(outcome, *, data=None,
    family="gaussian", link=None,
    fixed_k=None,
    k_min=2, k_max=120, k_step=2,
    ssd_gradients=1,
    verbose=False,
    pca_sweep_args=None,
)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `outcome` | `array-like \| str` | — | Numeric outcome values, a column name of `data`, or a formula over columns of `data`, e.g. `"score ~ age + gender + (1 \| ID)"`. The text is the formula term `ssd()`; a formula without it gets `+ ssd()` added at the end. Without a formula the model is intercept plus components. |
| `data` | DataFrame or `dict` | `None` | One row per document: the outcome column, covariates and grouping factors. |
| `family` | `str` | `"gaussian"` | Any `glmm` family of the final model, e.g. `"binomial"` for a 0/1 outcome. |
| `link` | `str \| None` | `None` | Link function; `None` takes glmm's default for the family. |
| `fixed_k` | `int \| None` | `None` | Fixed number of PCA components. `None` selects K via the interpretability + stability sweep. |
| `k_min`, `k_max`, `k_step` | `int` | `2`, `120`, `2` | Sweep range (ignored when `fixed_k` is set). |
| `ssd_gradients` | `int` | `1` | Number of gradients returned, in `[1, K]`. At 1 the result is the standard fit alone. At N > 1 the K components are split into N consecutive blocks (`np.array_split`: K=10, N=3 gives 4/3/3), each block's part of the model's beta is one gradient, and the N gradients are rotated by words as in `fit_pls`; leaves `result["dim-1"]` … `result["dim-N"]` with `stats` (`score_corr`, `cov_share`), `words`, `clusters`, `snippets` and no p-value. `score_corr` and `cov_share` keep the covariates in, so with covariates, another family or random effects they are descriptive. The standard fit does not depend on it. N > 1 needs one plain `ssd()` term: `group*ssd()` or `age:ssd()` raises `ValueError`, and so do N > K and a zero or aliased block. |
| `verbose` | `bool` | `False` | Print progress. |
| `pca_sweep_args` | `dict \| None` | `None` | Extra keyword arguments for the PCA-K sweep (e.g. `cluster_topn`, `auck_radius`, `save_tables`, `out_dir`). Ignored when `fixed_k` is set. A key that `fit_regression` sets itself (`Xs`, `X_scale`, `x`, `ys`, `embeddings`, `pca_k_values`, `verbose`) raises `ValueError`. |

K is chosen by the sweep (OLS on standardized outcome, whatever the family, without interactions). The final model at that K is fit by `glmm`: the K component scores replace the `ssd()` term in the formula, and `result.model` holds that fit (response `_ssd_y`, components `_ssd_pca_1` … `_ssd_pca_K`). The p-value tests the text terms together: the formula without them against the full formula, both fit by ML on the same rows. An OLS model (Gaussian, no random effects) gives the F-test (`test.name == "f_test"`), and so do Gamma and inverse-Gaussian GLMs (Wald F); every other model gives the likelihood-ratio test (`"lrt"`). A column the formula uses must have no missing values in the retained documents. `ssd()` cannot go inside a random-effect term. Sweep diagnostics and `result.plot_sweep(path=None, dpi=300)` are available when `fixed_k is None`.

For a Gaussian identity-link fit, `beta` is on the standardized-outcome scale; for other families it is on the link scale (e.g. log-odds). With a formula, `r2` / `r2_adj` are the whole model's, covariates included, and NaN for GLMs and mixed models. For a Gaussian identity-link fit, `docs.y_hat` uses `beta` only, not the covariates or random effects; for other families it is the model's fitted mean on the response scale (e.g. a probability), covariates included. `predict()` raises `NotImplementedError` after a fit with a formula or another family.

```python
result = ssd.fit_regression("rating ~ age + gender + (1 | ID)", data=df)
result.stats.pvalue       # test of the text terms together
result.covariates         # covariate coefficients, with the text held fixed
result.params             # statsmodels names: params, bse, tvalues, conf_int(), cov_params()
result.pvalues            # covariates, then one Wald p-value per text term
result.llr_pvalue         # whole model vs. outcome ~ 1 (fvalue / f_pvalue for OLS)
result.summary()          # compact: glmm summary without the _ssd_* rows, a row per text term
result.model.summary()    # full glmm summary, component rows included
```

The result has statsmodels names, with statsmodels meanings, read from `result.model`: `params`, `bse`, `tvalues`, `conf_int(alpha=0.05)`, `cov_params()` (covariates only), `pvalues` (covariates, then one Wald row per text term), `fvalue` / `f_pvalue` (OLS only) or `llr` / `llr_pvalue` (every other model) for the whole model against `outcome ~ 1`, `df_model`, `df_resid`, `nobs`, `rsquared`, `rsquared_adj`, `aic`, `bic`, `llf`, `resid`, `fittedvalues` and `random_effects`. They need pandas. See [results.md](results.md#pcaolsresult).

Returns a `PCAOLSResult` when the formula has one text term.

#### Interactions

The text can interact with covariates: `"score ~ age + sex*ssd()"`. `glmm` expands the interaction, and the result is a `RegressionResult` with one `PCAOLSResult` leaf per term that contains the text.

```python
res = ssd.fit_regression("score ~ age + sex*ssd()", data=df)
res.keys()                       # ['ssd(essay)', 'sexM:ssd(essay)']
res["sexM:ssd(essay)"].words     # leaf: how the gradient for sexM differs from the reference level
res.test                         # all-text test, then one Wald test per term
res.summary()
```

- A key is the term with `ssd()` written as `ssd(<text name>)`; it is `ssd()` when the text has no name.
- A leaf's `beta` is the gradient of its term alone. The gradient of one level is the sum of the `beta` of the main text term and of that level's term.
- `res.test` is a table with columns `label`, `name`, `pvalue`, `statistic`, `df`, `df_resid`, `r2_delta`. The row with an empty label tests all text terms together by comparing fits (`f_test` or `lrt`). The other rows are Wald tests (`wald_f` or `wald_chisq`) of one term's coefficients together, and they include lower-order terms such as `ssd(essay)`.
- `res.model`, `res.covariates`, `res.fit_info`, `res.sweep`, `res.summary()`, `res.save_page(...)` and `res.attach(...)` belong to the whole fit. `res.report()` and `res.predict()` are on a leaf, not on the container.
- Two texts cannot interact in one term.

#### `ssdiff.fit_regression()` — texts by name

```python
res = ssdiff.fit_regression(formula, data, *, family="gaussian", link=None,
    fixed_k=None, k_min=2, k_max=120, k_step=2, verbose=False, pca_sweep_args=None,
    **texts)
```

```python
import ssdiff

res = ssdiff.fit_regression("score ~ age + ssd(essay)", df)   # the text from the column "essay"

both = ssdiff.fit_regression(
    "score ~ age + ssd(essay) + sex*ssd(diary)", df,
    essay=essay_ssd, diary=diary_ssd,
)
both.keys()               # ['essay', 'diary']
both["essay"].words       # a text with one term is a PCAOLSResult
both["diary"].keys()      # ['ssd(diary)', 'sexM:ssd(diary)']
both.test                 # all texts, then one row per text, then one Wald row per term
```

Each `ssd(name)` in the formula is a text: the keyword `name=` when one is passed, else the column `name` of `data`, built as `SSD(data[name])` with every default. All texts built from columns in one call share one `Embeddings.load()`. A text that needs other settings is built with `SSD` and passed by keyword; a keyword wins over a column of the same name. The formula needs an outcome on the left and `data`; a bare `ssd()` is not allowed, a name that is neither a keyword nor a column raises, and the formula must name each text that is passed. The texts and `data` must have the same number of rows, in the same order. A row stays when it has an outcome and a vector in every text. Each text gets its own standardization, PCA and K. `fixed_k`, `k_min`, `k_max`, `k_step` and `pca_sweep_args` apply to every text. The per-text test row compares the full formula with the formula without that text's terms. With one text the result has no text level, as from `SSD.fit_regression`.

`fit_ols(...)` is a deprecated alias of `fit_regression`: same arguments, warns with `FutureWarning`.

### `fit_pls()` — PLS1

```python
result = ssd.fit_pls(
    outcome, *,
    data=None,
    k="auto",
    k_max=5,
    ssd_gradients=1,
    rotate="words",                  # "words" | None; used when ssd_gradients > 1
    n_splits=50,
    random_state=2137, test=True, verbose=False,
)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `outcome` | `array-like \| str` | — | The numeric outcome, one value per document: values, a column name of `data`, or a formula such as `"score ~ age + gender"`. A formula's left side is a column of `data` and its right side holds fixed covariates. The outcome and every column of the document vectors are regressed on them by OLS, and PLS and its test run on the residuals; split_exact permutes residuals against residuals (Kennedy, 1995); the test can run slightly liberal when the covariates use many degrees of freedom relative to n. Random effects and an `ssd()` term raise `ValueError` (use `fit_regression`). With a formula, `r2` is the R² of the residualized outcome, `docs.residual` has the covariates taken out, and `predict()` is not available. The covariates' own coefficients are not reported; use `fit_regression` for them. Missing values drop the document. |
| `data` | DataFrame or `dict` of 1-D arrays | `None` | One row per document: the outcome column and the formula's covariates. |
| `k` | `int \| "auto"` | `"auto"` | Number of PLS components. `int` fits at exactly that k. `"auto"` calls `plskit.pls1_find_k_optimal` (selector `r2_se`). The p-value is plskit's default split-half test (`test_method="auto"`: `split_exact` or `split_nb`, chosen from X alone; `result.test.name` says which ran) at the fitted `k`; with `k="auto"` it is the k=1 test. The diagnostic chain used to pick `k` is `split_nb`. |
| `k_max` | `int` | `5` | Cap for `k="auto"`, clamped to `min(k_max, n-1, D)`. Ignored when `k` is an int. |
| `ssd_gradients` | `int` | `1` | Number of gradients returned, in `[1, min(n-1, D)]`. At 1 the result is the standard fit alone and `rotate` is ignored. At N > 1, an N-component PLS1 fit on the same standardized data gives N weight vectors, one leaf `result["dim-i"]` each. The standard fit (k selection, test, `beta`, `r2`, `docs`, `predict()`) does not depend on it. |
| `rotate` | `"words" \| None` | `"words"` | With `ssd_gradients > 1`: `"words"` varimax-rotates the weight vectors against the projection of the leading vocabulary rows, orders them by the correlation of their scores with the outcome (highest first) and flips each so that correlation is positive; it needs attached `Embeddings`. `None` keeps NIPALS order and signs; gradients 2…N then have zero covariance with the outcome. Any other value raises `ValueError`. |
| `n_splits` | `int` | `50` | Random splits for the split-half tests. |
| `random_state` | `int` | `2137` | Seed. |
| `test` | `bool` | `True` | Run the confirmatory test. `False` skips it: `pvalue` is NaN, and k selection for `k="auto"` still runs. `result.test(n_splits=..., seed=...)` runs it later; with no arguments it uses 50 splits and no seed. |
| `verbose` | `bool` | `False` | Print progress. |

Returns a `PLSResult`. Re-run the significance test via `result.test(...)` — see [`results.md`](results.md). With `ssd_gradients=N > 1` it also has:

- `result["dim-1"]`, …, `result["dim-N"]` (`result.keys()`) — one leaf per gradient, with `.stats` (`score_corr`, `cov_share`), `.words`, `.clusters`, `.snippets`, `.beta`, `.gradient`; no p-value
- `.pls_info` — `ssd_gradients`, `rotate`, `order`, `signs`, `sweeps`, `V_converged`, `kaiser_normalized`, `random_state`

`fit_info.n_components` stays the standard fit's k.

`result.summary()` prints and returns a compact summary: the text's test and its R² (partial R² with a formula).

### `fit_groups()` — categorical outcome

```python
result = ssd.fit_groups(outcome, *, data=None,
    median_split=False,
    n_perm=5000,
    correction="holm",
    random_state=2137, verbose=False,
)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `outcome` | `array-like \| str` | — | One label per document: values, or a column name of `data`. A document with an empty or missing label drops out. |
| `data` | DataFrame or `dict` | `None` | One row per document; holds the label column. |
| `median_split` | `bool` | `False` | Split a continuous outcome at the median into `"low"`/`"high"`. |
| `n_perm` | `int` | `5000` | Permutations for omnibus + pairwise tests. |
| `correction` | `str` | `"holm"` | `"holm"`, `"bonferroni"`, `"fdr_bh"`, or `"none"`. |
| `random_state` | `int` | `2137` | Seed. |
| `verbose` | `bool` | `False` | Print progress. |

Groups with < 20 documents are dropped with a warning. Fewer than 2 surviving groups raises `ValueError`. `fit_groups()` takes no formula and works on one text.

Returns a `GroupResult`.

### `fit_multipls()` — several outcomes (PLS3)

Fits the document vectors against several numeric outcomes at once. The outcomes are a list of column names of `data`, a DataFrame, or a 2-D array, with at least two columns; for one outcome use `fit_pls`. A row with a missing value in any outcome drops the document. Needs the embeddings attached to the `SSD`.

PLS3 takes the SVD of the cross-covariance between document vectors and outcomes. Dimension `i` pairs one word-space direction with one vector of outcome loadings.

```python
result = ssd.fit_multipls(outcome, *,
    data=None,
    ssd_gradients=None,              # None means q
    rotate=None,                     # None | "words" | "outcomes"
    outcome_groups=None,             # {"group": ["outcome_a", "outcome_b"], ...}
    n_splits=50, n_perm=1000,
    random_state=2137, verbose=False,
)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `outcome` | `list[str] \| DataFrame \| array` | — | The outcomes: column names of `data`, a DataFrame (column names become outcome names), or a 2-D array (names are `"y1"`, `"y2"`, …). |
| `data` | DataFrame or `dict` of 1-D arrays | `None` | One row per document; holds the outcome columns. |
| `ssd_gradients` | `int \| None` | `None` | Number of dimensions, in `[1, min(q, n-1, D)]`. `None` means `q`, or with `outcome_groups` the number of rotation target rows. At `q` every component is kept, so `cov_explained` is 1 by construction. Raises if plskit keeps fewer components. |
| `rotate` | `None \| "words" \| "outcomes"` | `None` | `None` keeps the SVD dimensions. `"words"`: varimax against the projection of the leading vocabulary rows (up to 50,000) onto the word-space directions. `"outcomes"`: varimax against the outcome loadings, or the group rows when `outcome_groups` is set. At `ssd_gradients = q` without groups it takes `R = V'` with no varimax, so each dimension loads 1 on one outcome. |
| `outcome_groups` | `dict[str, list[str]] \| None` | `None` | Only with `rotate="outcomes"`. Each group is one rotation target row, the mean of its outcomes' loadings. Outcomes in no group keep their own row. An unknown name, an outcome in two groups, or an empty group raises. `ssd_gradients` cannot exceed the number of target rows. |
| `n_splits` | `int` | `50` | Random splits for the LV1 test. |
| `n_perm` | `int` | `1000` | Permutations for the LV1 test. |
| `random_state`, `verbose` | — | `2137`, `False` | Seed and progress output. |

A group's rotation target row and its `.outcomes` group loading are plain means of the group's outcome loadings. Outcomes scored in opposite directions cancel within a group.

Rotated modes reorder the dimensions by their share of the cross-covariance. In every mode, each dimension's sign is set so its largest outcome loading is positive.

Returns a `MultiPLSResult`:

- `result["dim-1"]`, … — one leaf per dimension
- `result["dim-1"].outcomes` — outcome loadings (`outcome`, `group`, `loading`); with `outcome_groups`, one extra row per group
- `result["dim-1"].words`, `.clusters`, `.snippets` — neighbours along the dimension
- `result["dim-1"].stats` — `cov_share`, `score_corr`
- `.stats` — `n`, `q`, `ssd_gradients`, `rotate`, `cov_explained`, `pvalue`
- `.pls_info` — `ssd_gradients`, `rotate`, `order`, `signs`, `sweeps`, `V_converged`, `kaiser_normalized`, `random_state`
- `.test` — PLS3 test at LV1 (plskit `test_method="auto"`), container level only; rerun with `.test(n_splits=..., n_perm=..., seed=...)`
- `.report(top_words={"n": 5})` — stats, rotation, per-dimension outcome and words tables

The p-value tests the first dimension (LV1) and does not depend on `rotate`. Leaves have no p-value.

Runnable example: [`examples/several_outcomes.py`](../examples/several_outcomes.py).

---

## `contrast()` — the difference of two gradients

```python
c = ssdiff.contrast(g1, g2, embeddings, *, labels=("g1", "g2"))
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `g1`, `g2` | array-like | — | Two gradient vectors of the same length, e.g. `r.gradient` or the `beta` of a leaf. They live in the space of `embeddings`. |
| `embeddings` | `Embeddings` | — | The embeddings the words are read from. The two gradients may come from different embeddings. |
| `labels` | `tuple[str, str]` | `("g1", "g2")` | Names of the two gradients on the results page. |

The contrast is `unit(g1) − unit(g2)`, normalized, so the scale of each gradient does not matter. It raises `ValueError` when the two gradients point the same way.

```python
r1 = ssd.fit_pls("score", data=df)
r2 = ssd.fit_pls("age", data=df)
c = ssdiff.contrast(r1.gradient, r2.gradient, emb, labels=("score", "age"))
c.words.pos
c.clusters.pos
c.stats                 # cos_g1_g2, diff_norm
c.save_page("contrast.html")
```

Returns a `ContrastResult` with `.words`, `.clusters`, `.stats`, `.beta`, `.gradient` and `.labels`. It has no test, and no documents, so `.snippets`, `.docs` and `.alignment_scores` raise `RuntimeError`.

---

## Lexicon helpers

Methods on `SSD`. They score the tokens of the text, need no lexicon on the `SSD`, and return a `LexiconResult` (see [`results.md`](results.md)).

```python
ssd.suggest_lexicon(outcome=None, *, data=None, seeds=None, top_k=30, min_docs=5,
                    n_bins=4, corr_cap=0.30, var_type="continuous")

ssd.evaluate_lexicon(outcome, lexicon, *, data=None, n_bins=4, corr_cap=0.30,
                     var_type="continuous")

ssd.token_stats(outcome, lexicon, *, data=None, n_bins=4, corr_cap=0.30,
                var_type="continuous")        # list of dicts: the per-token part
ssd.coverage_summary(outcome, lexicon, *, data=None, n_bins=4,
                     var_type="continuous")   # dict: the aggregate part
```

`suggest_lexicon` ranks tokens for use as seeds; `evaluate_lexicon` scores an existing lexicon, returning per-token diagnostics (`.suggestions`) and aggregate coverage (`.summary`).

With `seeds`, `suggest_lexicon` lists each seed, its `top_k` closest tokens of the text, and the seed with an English ending (`-s`, `-es`, `-d`, `-ed`, `-ing`, `-ies`, `-ied`), in one table sorted by cosine. A token listed for several seeds has one row, with the seed it is closest to and the cosine to that seed. Only tokens in at least `min_docs` documents and in the embeddings are listed. A lexicon matches exact tokens, so plural forms must be listed.

```python
lex = ssd.suggest_lexicon("score", data=df, seeds=["immigrant", "muslim"])
ssd = ssd.with_lexicon(["immigrant", "immigrants", "muslim", "muslims"])
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `outcome` | array-like, column name, or `None` | — | Outcome values or a column name of `data`, one per document. NaN / `None` entries filtered. `None` keeps every document and returns only the statistics that need no outcome; the outcome columns (`cov_bal`, `corr`, `pvalue`, `direction`, `rank`, and `q1`, `q4`, `corr_any` in `.summary`) are `None`. `suggest_lexicon` takes `None` only with `seeds`. |
| `data` | DataFrame | `None` | Holds the column `outcome` names. |
| `lexicon` | iterable[str] | — | Tokens to score (`evaluate_lexicon`, `token_stats`, `coverage_summary`). |
| `seeds` | sequence[str] | `None` | Seed words, each a key of the embeddings (`suggest_lexicon`). |
| `top_k` | `int` | `30` | Maximum candidates to return (`suggest_lexicon`). With `seeds`, neighbours listed per seed. |
| `min_docs` | `int` | `5` | Minimum document frequency (`suggest_lexicon`). |
| `n_bins` | `int` | `4` | Quantile bins for balanced coverage (continuous). |
| `corr_cap` | `float` | `0.30` | Cap applied to `|corr|` in the rank penalty. |
| `var_type` | `str` | `"continuous"` | `"continuous"` or `"categorical"`. |

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
res["sexM:ssd(essay)"].words    # one term of an interaction fit
result.suggestions / .tokens / .summary         # LexiconResult
result.summary()            # PLS / PCA regression — compact summary
result.predict(new_texts)   # PLS / PCA regression without a formula
result.save_page(path) / result.page_data()     # one offline HTML page / its content as a dict; not LexiconResult
result.attach(texts=..., embeddings=...)        # re-wire after un-pickling
result.clear_cache()
```

A result keeps the text it was fit on as `result.text` (an `SSD`), which snippets and `docs.id()` read.

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
lex = ssd.suggest_lexicon("score", data=df, top_k=30)
lex.tokens           # list[str] — suggested tokens in table order, ready to pass to SSD
lex.suggestions      # per-token table
lex.report().save("lexicon.md")

eval_ = ssd.evaluate_lexicon("score", lex.tokens[:10], data=df)
eval_.summary        # aggregate coverage
```

For the full results surface (views, to_dict / to_df / to_records, per-view `save`, callable reruns, cache behaviour) see [`results.md`](results.md).

---

## Full example

```python
import numpy as np
import ssdiff
from ssdiff import Embeddings, SSD

# df: one row per document, with columns "text", "score", "age", "sex", "condition"
emb = Embeddings.load()

# Lexicon selection: the SSD needs no lexicon yet
ssd = SSD(df["text"], embeddings=emb)
lex = ssd.suggest_lexicon("score", data=df, top_k=30)
lexicon = lex.tokens[:20]

# Continuous fit — PCA regression
ssd = ssd.with_lexicon(lexicon, window=3)
ols = ssd.fit_regression("score", data=df)
ols.plot_sweep("sweep.png")
ols.report(clusters={"n": 50, "n_words": 10, "n_snippets": 2}).save("report_ols.md")

# The same, with a covariate and an interaction
inter = ssd.fit_regression("score ~ age + sex*ssd()", data=df)
inter.summary()
inter.save_page("interaction.html")

# Continuous fit — PLS on the same SSD
pls = ssd.fit_pls("score", data=df, k="auto")
pls.report(clusters={"n": 50, "n_words": 10, "n_snippets": 2}).save("report_pls.md")

# Group comparison
gr = ssd.fit_groups("condition", data=df, n_perm=5000, correction="holm")
gr.report().save("report_groups.md")

# What differs between two fitted gradients
age = ssd.fit_pls("age", data=df)
c = ssdiff.contrast(pls.gradient, age.gradient, emb, labels=("score", "age"))
c.words.pos
```
