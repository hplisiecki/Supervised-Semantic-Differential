# ssdiff — Results Guide

How to read, slice, rerun, and export the objects returned by `SSD.fit_*`, `ssdiff.fit_regression`, `contrast`, and the lexicon helpers of `SSD`.

For fit-method arguments, see [`api_reference.md`](api_reference.md).
For per-view column lists and how to change defaults, see [`results_tables.md`](results_tables.md).

---

## Contents

- [What you get back](#what-you-get-back)
- [The three view types](#the-three-view-types)
- [`PLSResult`](#plsresult)
- [`PCAOLSResult`](#pcaolsresult)
- [`RegressionResult`](#regressionresult)
- [`GroupResult`](#groupresult)
- [`MultiPLSResult`](#multiplsresult)
- [`ContrastResult`](#contrastresult)
- [`LexiconResult`](#lexiconresult)
- [Picking columns — `cols=`](#picking-columns--cols)
- [Resizing tables](#resizing-tables)
- [Exporting with `save()`](#exporting-with-save)
- [Rerunning statistical tests](#rerunning-statistical-tests)
- [Pickle & `attach()`](#pickle--attach)
- [Cache](#cache)
- [Reports](#reports)
- [Results page](#results-page)
- [Repr footers](#repr-footers)
- [Row dataclasses](#row-dataclasses)
- [Optional dependencies](#optional-dependencies)
- [Cheat sheet](#cheat-sheet)

---

## What you get back

| From | Object |
|---|---|
| `SSD.fit_pls()` | `PLSResult` |
| `SSD.fit_regression()` | `PCAOLSResult`; `RegressionResult` when the formula has an interaction with the text |
| `ssdiff.fit_regression()` | `RegressionResult` (keyed by text when the formula has several texts) |
| `SSD.fit_groups()` | `GroupResult` |
| `SSD.fit_multipls()` | `MultiPLSResult` |
| `contrast()` | `ContrastResult` |
| `SSD.suggest_lexicon()` / `.evaluate_lexicon()` | `LexiconResult` |

```python
result = ssd.fit_pls("score", data=df)
print(result)              # headline summary + discoverability block
print(result.stats)        # r², p, n_kept, effect sizes
print(result.words.pos)    # top positive-pole words
```

---

## The three view types

Every result exposes its data through **views** — small objects you can print, slice, or export.

| Type | What it is | Example |
|---|---|---|
| `View[T]` | Tabular. Iterates frozen dataclasses. | `result.words`, `result.docs` |
| `ScalarView` | Single row. Field access via `.r2` or `["r2"]`. | `result.stats`, `result.fit_info` |
| `TestView` | Callable ScalarView — calling it reruns the test. | `result.test(n_splits=100)` |

All views share the same output methods:

```python
v.to_dict()            # list[dict] or dict  — always works
v.to_records()         # list[tuple]
v.to_df()              # pandas.DataFrame    — needs ssdiff[results]
v.to_text()            # aligned plain text
v.to_html()            # HTML (used by Jupyter)
v.save(path, k=None)   # export — format from extension
```

Iteration yields the full dataclass:

```python
for w in result.words.pos:
    print(w.rank, w.word, w.cos_beta)
```

---

## `PLSResult`

Returned by `fit_pls()`.

**Scalar views** — `.stats`, `.fit_info`, `.test`; `.pls_info` with `ssd_gradients > 1` (see [Gradient leaves](#gradient-leaves--fit_plsssd_gradientsn))

**Arrays** (numpy):

| Attribute | Shape | Meaning |
|---|---|---|
| `.x` | `(n_kept, D)` | Per-doc concept vectors (PCVs) |
| `.y` | `(n_kept,)` | Outcome on the original scale |
| `.beta` | `(D,)` | Regression direction — carries magnitude |
| `.gradient` | `(D,)` | Unit-length β — the semantic gradient |
| `.alignment_scores` | `(n_kept,)` | `x @ gradient` |
| `.component_scores`, `.component_weights` | `(n_kept, A)`, `(D, A)` | PLS1 T, W |
| `.find_k_result`, `.cv_scores` | — | `k="auto"` selection diagnostics (`None` for an int `k`) |

**Tabular views:**

| View | Default | Notes |
|---|---|---|
| `.words` | top 100 / side | `.pos`, `.neg` — 20 rows each |
| `.clusters.pos` / `.neg` | `topn=100` | Callable; zooming with `(cluster_id)` renders a 5-row snippet sub-table. `top_words` (top-5 cluster words) is a default column; `top_snippet` is **opt-in** via `cols="all"` (filling it triggers full snippet extraction). |
| `.clusters.words` / `.pos.words` | — | Per-cluster word tables |
| `.clusters.pos.snippets` | 30 | Snippets filtered to pos-side clusters |
| `.snippets` | `top_per_side=30` | `.pos`, `.neg` for sides |
| `.docs` | preview | See below |

`.summary()` prints and returns a header and the Text block: the text's test and its R² (partial R² with a formula).

**`.docs` accessors:**

```python
result.docs.pos(10)                          # 10 most β-pos-aligned
result.docs.neg(10)                          # β-neg
result.docs.misdiagnosed(5)                  # largest |residual|
result.docs.misdiagnosed(5, direction="over")   # residual < 0
result.docs.id(42)                           # full detail view; needs the text attached, and shows its raw text for strings only
```

### Gradient leaves — `fit_pls(ssd_gradients=N)`

With `ssd_gradients=N > 1`, a `PLSResult` also holds N leaves keyed `"dim-1"`, …, `"dim-N"`, one per rotated gradient. Everything above describes the standard fit and stays the same as with `ssd_gradients=1`.

```python
res = ssd.fit_pls("score", data=df, ssd_gradients=3)

res.keys()                 # ['dim-1', 'dim-2', 'dim-3']
res["dim-1"].stats         # score_corr, cov_share
res["dim-1"].words         # top neighbors along gradient 1
res["dim-1"].beta          # the gradient's direction in doc-vector space
res.pls_info               # ssd_gradients, rotate, order, signs, rotation diagnostics
```

Leaves are the same `_PLSComponentResult` class as `MultiPLSResult` leaves, without `.outcomes`. They have `.stats`, `.words`, `.clusters`, `.snippets`, `.beta`, `.gradient`, `.alignment_scores`, and no p-value. `report()` adds the rotation, a row per gradient and each gradient's words after the standard sections; the results page lists the gradients in its "Dimension" menu after β.

---

## `PCAOLSResult`

Everything on `PLSResult`, plus:

- `.stats` adds `r2_adj`
- `.pca_k`, `.pca_components`, `.pca_weights` — PCA basis and the components' regression coefficients
- `.model` — the `glmm` fit of the outcome on the formula and the K component scores (`_ssd_pca_1` … `_ssd_pca_K`; response `_ssd_y`): covariate coefficients, random effects, `.summary()`. `.covariates` lists its covariate rows, with the components held fixed; empty without a formula.
- `.summary()` — glmm's summary of `.model` without the `_ssd_*` rows (fixed effects and their correlations), with an SSD effects row below the fixed effects that holds the text's ΔR² and p. The other blocks describe the full model, components included. `.model.summary()` prints the full glmm summary.
- `.test` — columns `name` (`"f_test"` or `"lrt"`), `pvalue`, `statistic` (F or χ²), `df` (the number of components), `df_resid` (the F-test's residual df; `None` for a likelihood-ratio test) and `r2_delta` (the R² the components add; NaN unless the model is OLS)
- `.sweep_result`, `.sweep` — full sweep diagnostics (`None` if `fixed_k` was set)
- `.plot_sweep(path=None, dpi=300)` — dual-axis chart of interpretability vs β-change. Returns the matplotlib `Figure`; writes to `path` when given, otherwise calls `plt.show()`. Needs matplotlib.
- `.predict(texts)` — predicted outcome for new documents, in the outcome's units. Strings for a fit on strings, token lists for a fit on token lists. Available for fits without a formula and, for `fit_regression`, with the Gaussian family; other fits raise `NotImplementedError`. A document with no vector gets `NaN`.

---

## `RegressionResult`

Returned by `fit_regression()` when the formula has an interaction with the text (`"score ~ age + sex*ssd()"`), and by `ssdiff.fit_regression()`. It is a container of `PCAOLSResult` leaves, one per term that contains the text.

```python
res = ssd.fit_regression("score ~ age + sex*ssd()", data=df)

res.keys()                     # ['ssd(essay)', 'sexM:ssd(essay)']
res["ssd(essay)"].words        # gradient at the reference level of sex
res["sexM:ssd(essay)"].words   # how the gradient for sexM differs from it
res.test                       # all-text test, then one Wald test per term
res.summary()                  # glmm summary without the text rows, plus one row per test
res.covariates                 # covariate coefficients
res.model                      # the glmm fit shared by all leaves
res.save_page("interaction.html")
```

- A key is the formula term with `ssd()` shown as `ssd(<text name>)`. The text name is the Series name given to `SSD`; without one the key reads `ssd()`.
- A leaf is a `PCAOLSResult` with `.beta`, `.gradient`, `.words`, `.clusters`, `.snippets`, `.docs`, `.report()`. Its `beta` is the gradient of that term alone. The gradient of one level is the sum of the `beta` of the main text term and of that level's term.
- `res.words`, `res.clusters`, `res.snippets`, `res.beta`, `res.gradient` fan out across the leaves, as on `GroupResult`.
- `.model`, `.fit_info`, `.sweep` and `.covariates` are the same for every leaf.
- The container has `.summary()`, `.save_page()` and `.attach()`, and no `.report()` or `.predict()`: call those on a leaf.

**`res.test`** has one row per test, with columns `label`, `name`, `pvalue`, `statistic`, `df`, `df_resid`, `r2_delta`:

| Row | `label` | `name` |
|---|---|---|
| All text terms together: the formula without them against the full formula | empty | `f_test` or `lrt` |
| One text, in a fit with several texts | the text's name | `f_test` or `lrt` |
| One term: its coefficients tested together | the term, e.g. `sex:ssd(essay)` | `wald_f` or `wald_chisq` |

A term row tests lower-order terms too (`ssd(essay)` has a row). `r2_delta` is set on comparison rows when the model is OLS, and empty on Wald rows.

### Several texts

`ssdiff.fit_regression("score ~ ssd(essay) + sex*ssd(diary)", df, essay=essay, diary=diary)` returns a `RegressionResult` keyed by text name:

```python
both.keys()              # ['essay', 'diary']
both["essay"]            # a PCAOLSResult (one term)
both["diary"].keys()     # ['ssd(diary)', 'sexM:ssd(diary)']
both.fit_info["diary"]   # fit_info, sweep and n_no_vector are dicts keyed by text name
both.test                # all texts, one row per text, then one Wald row per term
```

Each text has its own PCA and K. `both.text` and `both.embeddings` are `None`; each entry holds its own. After un-pickling, `both.attach(texts={"essay": ..., "diary": ...}, embeddings={"essay": emb, "diary": emb})` takes dicts keyed by text name.

---

## `GroupResult`

Returned by `fit_groups()`. Handles both 2-group (single-pair) and multi-group (multi-pair) fits.

**Scalars:** `.G`, `.n_kept`, `.group_labels` (canonical → original name).

**Arrays** — always `dict[pair, value]`, even for a 2-group fit (no scalar shortcut).

```python
gr.beta              # {('g1','g2'): (D,), ...}
gr.gradient          # same
gr.beta_norm         # same
gr.alignment_scores  # same
```

- Access a single pair's array via `gr.beta[('g1','g2')]` or (equivalently) `gr[('g1','g2')].beta`.

**Views:** `.stats`, `.test`, `.pairs`, `.words`, `.clusters`, `.snippets`.

### Zoom to one pair — `gr[pair]`

The canonical way to work with a specific pair. Indexing returns a `PairResult`:

```python
pair = gr[('g1', 'g2')]    # PairResult — full single-pair view
pair.words.pos(20)
pair.clusters.pos
pair.snippets
pair.beta                   # (D,) ndarray
```

The key must be a canonical tuple `('g1', 'g2')`. Reverse order is normalized. Unknown pair → `KeyError`.

*Power-user shortcut:* view-level indexing — `gr.words[('g1','g2')]` — skips constructing a `PairResult` when you only need one view.

### `.pairs`

```python
for p in gr.pairs:
    print(p.contrast, p.T, p.p_corrected, p.cohens_d)

gr.pairs[('g1', 'g2')]   # raw Pair dataclass
gr.pairs[0]              # first Pair
gr.pairs[:3]             # slice
```

### Cluster snippets across pairs

```python
gr.cluster_snippets(
    pair=None,            # required for multi-pair; canonical tuple
    side="pos",           # "pos" | "neg" | "both"
    top_per_cluster=100,
    min_cosine=None,
)
```

`pair=` is optional for 2-group fits.

---

## `MultiPLSResult`

Returned by `fit_multipls()` (PLS3, several outcomes). A container of per-dimension leaves keyed `"dim-1"`, `"dim-2"`, …. Mirrors the `GroupResult` / `PairResult` pattern. Runnable example: [`../examples/several_outcomes.py`](../examples/several_outcomes.py).

**Scalar views** (whole model):

- `.stats` — `n`, `q`, `ssd_gradients`, `rotate`, `cov_explained`, `pvalue`
- `.pls_info` — `ssd_gradients`, `rotate`, `order`, `signs`, `sweeps`, `V_converged`, `kaiser_normalized`, `random_state`
- `.test` — PLS3 test at LV1 (plskit `test_method="auto"`)

**Other attributes:** `.outcome_names`, `.outcome_groups`.

**Arrays:** `.x`, `.y`, `.U`, `.V`, `.singular_values` (unrotated PLS3 fit); `.U_rot`, `.V_rot`, `.R` (after rotation); `.x_scores`, `.y_scores`.

### Leaves

Each leaf is a `_PLSComponentResult` (a `_SingleResult`), so its interpretation views mirror `PLSResult`:

```python
res = ssd.fit_multipls(["valence", "arousal", "dominance"], data=df, ssd_gradients=2, rotate="outcomes")

res["dim-1"].outcomes       # outcome loadings on dimension 1
res["dim-1"].stats          # cov_share, score_corr
res["dim-1"].words          # top neighbors along dimension 1
res["dim-1"].beta           # the dimension's direction in doc-vector space
res["dim-2"].words.pos(20)
```

Per-leaf views:

- `.outcomes` — columns `outcome`, `group`, `loading`. With `outcome_groups`, one extra row per group holds the group's mean loading. Outcomes scored in opposite directions cancel in that mean.
- `.stats` — `cov_share`, `score_corr`
- `.words`, `.clusters`, `.snippets`

`res.words` / `res.clusters` / `res.snippets` fan out across leaves, same as `GroupResult`. Leaves have no p-value and no `.test`.

### Test

`res.test(...)` runs one whole-model test, PLS3 at LV1 with plskit's `test_method="auto"`. The p-value does not depend on `rotate`.

```python
res.test(n_splits=100, n_perm=2000, seed=42)
```

### Report

```python
res.report(top_words={"n": 5}).save("multipls.md")
```

Clusters and snippets sections are not part of the report; they are not accepted and raise `TypeError`.

---

## `ContrastResult`

Returned by `contrast(g1, g2, embeddings)`. It reads `unit(g1) − unit(g2)`, normalized, as a gradient of its own.

```python
c = contrast(r1.gradient, r2.gradient, emb, labels=("score", "age"))

c.stats           # cos_g1_g2 (cosine of the two gradients), diff_norm (length of the unit difference)
c.words.pos       # same word views as any gradient
c.clusters.pos
c.beta            # (D,) the difference, normalized
c.gradient        # same direction, unit length
c.labels          # ('score', 'age')
c.save_page("contrast.html")
```

A contrast has no documents. `.snippets`, `.docs` and `.alignment_scores` raise `RuntimeError`. It has no test and no `.report()`. The words and clusters need the embeddings given to `contrast`.

---

## `LexiconResult`

Returned by `SSD.suggest_lexicon(...)` or `.evaluate_lexicon(...)`.

| View | Columns |
|---|---|
| `.stats` | `var_type`, `n_docs`, `n_tokens` |
| `.suggestions` | `token`, `freq`, `cov_all`, `cov_bal`, `corr`, `pvalue`, `direction`, `rank`. From `suggest_lexicon(seeds=...)`: `seed` first and `cos` after `token` |
| `.tokens` | `list[str]` — each suggestion token once, in table order |
| `.summary` | Only from `evaluate_lexicon`: `docs_any`, `cov_all`, `q1`, `q4`, `corr_any`, `hits_mean/median`, `types_mean/median`, `group_cov` |

```python
lex.report(top={"n": 20}).save("lexicon.md")
```

With seeds, `.suggestions` is one table for all seeds, highest cosine first: `seed` is the seed the token is closest to and `cos` the cosine to it. Printing `lex.suggestions` shows the first 20 rows; `lex.suggestions[:]` prints all of them. A lexicon matches exact tokens, so plural forms must be listed.

---

## Picking columns — `cols=`

Every `to_*` and `save()` accepts `cols`:

```python
v.to_df()                              # view's default columns
v.to_df(cols="all")                    # every available column
v.to_df(cols=["word", "cos_beta"])     # explicit subset + order
```

Defaults are the same for `__repr__`, text, HTML, and exports — **what you see is what you save**.

To change a default globally:

```python
from ssdiff.results import display
display.DEFAULT_COLS["WordsView"] = ("side", "rank", "word", "cos_beta", "contrast")
```

See [`results_tables.md`](results_tables.md) for every view's columns.

---

## Resizing tables

Size-bearing views are **callable**. Calling re-slices (or recomputes) and the new size applies everywhere — display and export.

| View | Default | Call |
|---|---|---|
| `result.words.pos` / `.neg` | 20 rows | `words.pos(50)`, `words.pos(None)` |
| `result.docs.pos()` / `.neg()` / `.misdiagnosed()` | 20 | `docs.pos(50)` |
| `result.clusters.pos` / `.neg` | `topn=100` | `clusters.pos(topn=50, k=5, …)` — **re-clusters** |
| `result.snippets` | `top_per_side=30` | `snippets(top_per_side=200, min_cosine=0.4)` — **re-extracts** |

Or pass `k=` directly to `save()`:

```python
result.words.pos.save("top50.csv", k=50)
```

Terminal tables truncate to 20 rows by default (10 for snippet tables, 50 for `result.docs`) with a `… N more rows` footer. Slicing (`words.pos[:50]`) or calling with an explicit count prints everything.

---

## Exporting with `save()`

```python
v.save("words.csv")
v.save("words.xlsx", cols="all", k=100)
```

| Extension | Needs extra |
|---|---|
| `.csv`, `.json`, `.md`, `.tex`, `.txt`, `.html` | — |
| `.xlsx` | `ssdiff[results]` (pandas + openpyxl) |
| `.docx` | `ssdiff[results]` (python-docx) |

Unsupported extensions raise `ValueError` with the list of supported ones.

---

## Rerunning statistical tests

`result.test` is callable. Calling it replaces the test and updates `result.stats.pvalue` in place.

**PLS:**
```python
result.test(n_splits=100, seed=0)   # test_method="auto" at the fitted k

result.test.pvalue         # current p-value
result.test.name           # "split_exact" or "split_nb"
```

**Several outcomes:**
```python
res.test(n_splits=100, n_perm=2000, seed=42)   # LV1, test_method="auto"
```

**Groups:**
```python
gr.test(n_perm=10000, correction="fdr_bh", random_state=0)
gr.test.omnibus_T          # observed mean pairwise cosine distance
gr.test.omnibus_p          # omnibus p-value
```

---

## Pickle & `attach()`

A pickled result keeps its text (the tokens, and the raw strings of a string text) and its embeddings, except a wordpack: a wordpack is read from its file and cannot be pickled. The document vectors are not pickled; they are rebuilt when needed. Views that need embeddings (`words`, `clusters`, `snippets`, `predict()`) then raise `RuntimeError` until you attach the same pack again. A path is loaded at the fit's ABTT, and `attach()` checks the pack's name, version and ABTT against the fit:

```python
import pickle
result = pickle.load(open("result.pkl", "rb"))
result.attach(embeddings="en-glove2024-dolma-1.wordpack")
result.words   # works now
```

`attach(texts=..., embeddings=...)` also swaps in other texts or embeddings on any result; embeddings must have the fit's ABTT. `texts` are strings, or token lists when the fit was on token lists, with as many documents as the fit's text; they are tokenized with the result's embeddings. Strings given for a token-list fit, or token lists for a string fit, raise `TypeError`.

`attach()` returns `self`, so it chains: `pickle.load(f).attach(texts=t, embeddings=e)`. On a result with leaves (`GroupResult`, `MultiPLSResult`, `RegressionResult`, or `fit_pls(ssd_gradients=N)`), it re-wires every leaf too.

---

## Cache

Parameterised views (`clusters`, `snippets`) cache per parameter set:

```python
result.clusters.pos                 # cached (topn=100)
result.clusters.pos(topn=50)        # separate cache entry
result.clusters.pos                 # still returns the topn=100 one

result.clear_cache()                # drop all
result.clear_cache("clusters")      # drop only clusters.*
```

---

## Reports

Every result has a `.report(...)` method that builds a multi-section document.
**Clusters are the main element**; every section is on by default — pass
`section=False` to drop one.

**Section toggles** (`clusters`, `top_words`, `extreme_docs`, `misdiagnosed`,
`top`) all accept:

- `True` (the default for every section) — include with defaults
- `False` / `None` — skip the section
- `dict` — override defaults, e.g. `clusters={"n": 20}`

Passing an `int` raises `TypeError` — wrap in a dict instead.

The Stats / Fit-info block (PLS / PCA+OLS) and the Omnibus / Group-labels /
Pairwise-contrasts block (`GroupResult`) are **always rendered** — there is no
toggle for them.

**PLS / PCA+OLS:**
```python
# Default — Stats + Fit info + Clusters + Top words + Extreme docs + Misdiagnosed
result.report()

# Customize the cluster table
result.report(clusters={"n": 50, "n_words": 8, "n_snippets": 2})

# Drop the excerpt column entirely
result.report(clusters={"n_snippets": 0})

# Skip a section
result.report(misdiagnosed=False)

# Minimal — Stats + Fit info + Clusters only
result.report(top_words=False, extreme_docs=False, misdiagnosed=False)
```

`RegressionResult` and `ContrastResult` have no `.report()`: use `res.summary()` and `save_page()`, or `.report()` on one leaf of a `RegressionResult`.

A `fit_pls(ssd_gradients=N)` result with N > 1 adds, after these sections, a Rotation block, a Dimensions table (`cov_share`, `score_corr` per gradient) and each gradient's top words; `top_words` sets those too.

**Groups:**
```python
# Default — Omnibus + Group labels + Pairwise contrasts + Top words + Clusters
gr.report()

# Heavier cluster tables, no top-words section
gr.report(clusters={"n": 20, "n_words": 10, "n_snippets": 2}, top_words=False)
```

`clusters` dict keys: `n` (clusters per side, default 10), `n_words` (default
5) and `n_snippets` (default 1; `0` drops the "Representative Excerpt" column;
values > 1 join excerpts with `" / "`). Per pair × side on `GroupResult`.

**Render or save:**
```python
r = result.report(clusters={"n": 50, "n_words": 10})

print(r)                 # plain text
r.save("report.md")      # .md .txt .html .tex .json .docx
```

`.docx` needs `ssdiff[results]`. Every other format is zero-dep.

---

## Results page

`result.save_page(path)` writes one HTML file with the result's stats, words, clusters, snippets and documents. It opens in a browser from disk, with no network. It works on `PLSResult`, `PCAOLSResult`, `RegressionResult`, `GroupResult`, `MultiPLSResult` and `ContrastResult`.

```python
result.save_page("ssd_results.html", title="Valence (PLS)")
result.save_page("all_texts.html", doc_texts=None)   # keep every document's text
```

The page lists the texts it was fit on (embeddings, lexicon and window settings, K, sweep, documents without a vector). A regression page has one unit for each text term, and a fit with several texts has one section for each text. A contrast page shows the difference with the two labels as its poles and has no texts or documents.

`doc_texts` (default 50) keeps the text of that many highest- and lowest-scoring documents, and for PLS and PCA regression also of the most over- and under-predicted ones; `None` keeps every text, `0` none. `snippets_per_side` (200) and `snippets_per_cluster` (50) cap the snippet tables. `dataset={"File": "data.csv"}` adds a "Dataset" block of those key–value pairs. Word and cluster tables need the embeddings, and snippets and document texts also need the text's raw strings. A pickled result leaves them out until `result.attach(texts=..., embeddings=...)`, and a text of token lists has no snippets, and its documents appear as their tokens joined by spaces. `result.page_data()` returns the same content as a dict of plain JSON values; its `report_format` is 2.

---

## Repr footers

Every `__repr__` ends with a `Save: …` hint showing the idiomatic export call. To silence globally (e.g. for log consumers):

```python
from ssdiff import set_repr_hints
set_repr_hints(False)
```

Long text columns (`Snippet.text_window`) clip to 40 chars in the terminal only — exports keep the full value.

---

## Row dataclasses

Iterating a tabular view yields frozen dataclasses. Import from `ssdiff.results.schema` for `isinstance` checks.

```
Word         side, rank, word, cos_beta, contrast
Cluster      cluster_id, side, size, coherence, centroid_cos_beta, contrast,
             top_words, top_snippet
ClusterWord  cluster_id, side, word, cos_centroid, cos_beta, contrast
Snippet      snippet_id, side, doc_id, cosine, seed,
             start_token_idx, end_token_idx, start_sent_idx, end_sent_idx,
             text_window, text_surface, text_lemmas,
             cluster_id, contrast
Doc          doc_id, y_true, y_hat, residual, alignment_score
Pair         contrast, g1, g2, T, p_raw, p_corrected, cohens_d,
             n_g1, n_g2, contrast_norm
Suggestion   token, freq, cov_all, cov_bal, corr, pvalue, direction, rank,
             seed, cos
TermTest     label, name, pvalue, statistic, df, df_resid, r2_delta
Outcome      outcome, group, loading
```

- `side ∈ {"pos", "neg"}` — β-pole
- `direction` on `Suggestion`: `"positive" | "negative" | "none"` (correlation sign)
- `direction` on `docs.misdiagnosed()`: `"both" | "over" | "under"` (residual sign)
- `contrast` is `None` on continuous fits, `"g1_g2"` on group fits

---

## Optional dependencies

```bash
pip install ssdiff[results]
```

| For | Needs |
|---|---|
| `to_dict`, `to_records`, `save('.csv' / '.json' / '.md' / '.tex' / '.txt' / '.html')` | nothing |
| `to_df()`, `save('.xlsx')` | pandas + openpyxl |
| `save('.docx')` | python-docx |
| `result.plot_sweep(...)` | matplotlib |

Missing deps raise `ImportError` with an install hint.

---

## Cheat sheet

| Task | Code |
|---|---|
| Headline stats | `result.stats` |
| Full narrative | `print(result.report())` |
| Save report | `result.report().save("r.md")` |
| Top 50 pos words → DataFrame | `result.words.pos(50).to_df()` |
| All pos words → CSV | `result.words.pos(None).save("pos.csv")` |
| Worst-predicted docs | `result.docs.misdiagnosed(10, direction="over")` |
| Doc + raw text | `result.docs.id(42)` |
| Recompute snippets | `result.snippets(top_per_side=200, min_cosine=0.4)` |
| Snippets inside cluster 3 | `result.clusters.pos.snippets(cluster_id=3)` |
| Rerun PLS test | `result.test(n_splits=100)` |
| Rerun group test | `gr.test(n_perm=10000, correction="fdr_bh")` |
| Zoom to one pair | `gr[('g1','g2')]` |
| Per-pair top words | `gr[('g1','g2')].words.pos(20)` |
| Per-pair raw stats | `gr.pairs[('g1','g2')]` |
| Zoom to one term of an interaction | `res['sexM:ssd(essay)']` |
| Compare two gradients | `contrast(g1, g2, emb)` |
| Zoom to one dimension | `res['dim-1']` |
| Per-dim top words | `res['dim-1'].words.pos(20)` |
| Per-dim outcome loadings | `res['dim-1'].outcomes` |
| Drop cache | `result.clear_cache()` |
| Re-wire after unpickle | `result.attach(texts=t, embeddings=e)` |
| Silence repr footers | `ssdiff.set_repr_hints(False)` |

---

## See also

- [`api_reference.md`](api_reference.md) — `Embeddings`, `SSD`, fit methods, `contrast`
- [`results_tables.md`](results_tables.md) — every view's columns and defaults
- [`architecture.md`](architecture.md) — backends, cache internals, how views compose
- [`../examples/pls.py`](../examples/pls.py) — runnable end-to-end example
- [`../examples/several_outcomes.py`](../examples/several_outcomes.py) — `fit_multipls`
