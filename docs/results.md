# ssdiff — Results Guide

How to read, slice, rerun, and export the objects returned by `SSD.fit_*` and `Corpus.*_lexicon`.

For fit-method arguments, see [`api_reference.md`](api_reference.md).
For per-view column lists and how to change defaults, see [`results_tables.md`](results_tables.md).

---

## Contents

- [What you get back](#what-you-get-back)
- [The three view types](#the-three-view-types)
- [`PLSResult`](#plsresult)
- [`PCAOLSResult`](#pcaolsresult)
- [`GroupResult`](#groupresult)
- [`MultiPLSResult` *(in development)*](#multiplsresult-in-development)
- [`LexiconResult`](#lexiconresult)
- [Picking columns — `cols=`](#picking-columns--cols)
- [Resizing tables](#resizing-tables)
- [Exporting with `save()`](#exporting-with-save)
- [Rerunning statistical tests](#rerunning-statistical-tests)
- [Pickle & `attach()`](#pickle--attach)
- [Cache](#cache)
- [Reports](#reports)
- [Repr footers](#repr-footers)
- [Row dataclasses](#row-dataclasses)
- [Optional dependencies](#optional-dependencies)
- [Cheat sheet](#cheat-sheet)

---

## What you get back

| From | Object |
|---|---|
| `SSD.fit_pls()` | `PLSResult` |
| `SSD.fit_ols()` | `PCAOLSResult` |
| `SSD.fit_groups()` | `GroupResult` |
| `SSD.fit_multipls()` *(in development)* | `MultiPLSResult` |
| `Corpus.suggest_lexicon()` / `.evaluate_lexicon()` | `LexiconResult` |

```python
result = ssd.fit_pls()
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
| `TestView` | Callable ScalarView — calling it reruns the test. | `result.test("perm", n_perm=5000)` |

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

**Scalar views** — `.stats`, `.fit_info`, `.test`

**Arrays** (numpy):

| Attribute | Shape | Meaning |
|---|---|---|
| `.x` | `(n_kept, D)` | Per-doc concept vectors (PCVs) |
| `.y` | `(n_kept,)` | Outcome on the original scale |
| `.beta` | `(D,)` | Regression direction — carries magnitude |
| `.gradient` | `(D,)` | Unit-length β — the semantic gradient |
| `.alignment_scores` | `(n_kept,)` | `x @ gradient` |
| `.component_scores`, `.component_weights` | `(n_kept, A)`, `(D, A)` | PLS1 T, W |
| `.cv_result`, `.cv_scores`, `.perm_null` | — | CV + permutation diagnostics (may be `None`) |

**Tabular views:**

| View | Default | Notes |
|---|---|---|
| `.words` | top 100 / side | `.pos`, `.neg` — 20 rows each |
| `.clusters.pos` / `.neg` | `topn=100` | Callable — re-clusters |
| `.clusters.words` / `.pos.words` | — | Per-cluster word tables |
| `.clusters.pos.snippets` | 30 | Snippets filtered to pos-side clusters |
| `.snippets` | `top_per_side=30` | `.pos`, `.neg` for sides |
| `.docs` | preview | See below |

**`.docs` accessors:**

```python
result.docs.pos(10)                          # 10 most β-pos-aligned
result.docs.neg(10)                          # β-neg
result.docs.misdiagnosed(5)                  # largest |residual|
result.docs.misdiagnosed(5, direction="over")   # y_hat > y_true
result.docs.id(42)                           # full detail view; needs attached corpus
```

---

## `PCAOLSResult`

Everything on `PLSResult`, plus:

- `.stats` adds `r2_adj`
- `.pca_k`, `.pca_components`, `.pca_weights` — PCA basis and OLS coefficients
- `.sweep_result`, `.sweep` — full sweep diagnostics (`None` if `fixed_k` was set)
- `.plot_sweep(path=None, dpi=300)` — dual-axis chart of interpretability vs β-change; returns PNG bytes. Needs matplotlib.

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

## `MultiPLSResult` *(in development)*

Returned by `fit_multipls()`. A container of per-component leaves plus a `"combined"` leaf for the unrotated prediction direction. Mirrors the `GroupResult` / `PairResult` pattern.

> **Status**: API is stable for research use. Per-leaf feature parity with `PLSResult` (clusters, snippets, misdiagnosed docs, per-dim diagnostics) is still being rolled out. Runnable example: [`../examples/demo_multipls.py`](../examples/demo_multipls.py).

**Scalars:** `.n_components`, `.random_state`.

**Scalar views:** `.stats`, `.pls_info`, `.test` (shared, whole-model).

**Arrays:** `.W`, `.P`, `.Q` (unrotated PLS), `.W_rot`, `.T_rot` (rotated), `.beta_combined` (unrotated prediction β), `.x`, `.y`.

### Leaves

Each leaf is a `_PLSComponentResult` (a `_SingleResult`), so its interpretation views mirror `PLSResult`:

```python
res = ssd.fit_multipls(n_components=2, rotate="varimax")

res["dim-1"].words          # top neighbors along the first rotated axis
res["dim-1"].beta           # W_rot[:, 0]  (the axis direction)
res["dim-2"].words.pos(20)
res["combined"].words       # neighbors along the unrotated prediction β
res["combined"].beta        # β = W(P'W)⁻¹Q  — rotation-invariant
```

Leaf keys are `"dim-1"`, …, `"dim-k"`, `"combined"` (strings). `res.words` / `res.clusters` / `res.snippets` fan out across leaves via `_ShimView`, same as `GroupResult`.

### Shared test

`res.test(...)` runs one whole-model test — CV-R² is a model-level quantity, rotation is free for prediction. Same three backends as `PLSResult`:

```python
res.test("split",     n_splits=100)
res.test("perm",      n_perm=2000)
res.test("split_cal", n_splits=50, n_perm=2000)
```

### Rotation diagnostics

`res.pls_info` exposes: `n_components`, `rotate`, `pca_k`, `order`, `signs`, `kaiser_normalized`, `sweeps`, `V_converged`, `kappa`, `pvalue_source`, `random_state`.

### Minimal report (v1)

```python
res.report(top_words=5).save("multipls.md")
```

Clusters / snippets / misdiagnosed sections are reserved for a later milestone — passing them here is a no-op.

---

## `LexiconResult`

Returned by `Corpus.suggest_lexicon(...)` or `.evaluate_lexicon(...)`.

| View | Columns |
|---|---|
| `.stats` | `var_type`, `n_docs`, `n_tokens` |
| `.suggestions` | `token`, `freq`, `cov_all`, `cov_bal`, `corr`, `pvalue`, `direction`, `rank` |
| `.tokens` | `list[str]` — suggestion tokens in rank order |
| `.summary` | Only from `evaluate_lexicon`: `docs_any`, `cov_all`, `q1`, `q4`, `corr_any`, `hits_mean/median`, `types_mean/median`, `group_cov` |

```python
lex.report(top=20).save("lexicon.md")
```

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

Terminal tables truncate to 20 rows with a `… N more rows` footer. Slicing (`words.pos[:50]`) or calling with an explicit count prints everything.

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
result.test("perm",      n_perm=5000, seed=0)
result.test("split",     n_splits=100, split_ratio=0.5)
result.test("split_cal", n_splits=50, n_perm=2000)

result.test.pvalue         # current p-value
result.test.name           # "perm" | "split" | "split_cal"
```

**Groups:**
```python
gr.test(n_perm=10000, correction="fdr_bh", random_state=0)
gr.test.omnibus_T          # observed mean pairwise cosine distance
gr.test.omnibus_p          # omnibus p-value
```

---

## Pickle & `attach()`

Pickled results lose their corpus/embeddings link. Views that need live data (`words`, `clusters`, `snippets`, `docs.id(...)`) raise `RuntimeError` until re-wired:

```python
import pickle
result = pickle.load(open("result.pkl", "rb"))
result.attach(corpus=my_corpus, embeddings=my_embeddings)
result.words   # works now
```

`attach()` returns `self`, so it chains: `pickle.load(f).attach(corpus=c, embeddings=e)`.

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

**PLS / PCA+OLS:**
```python
result.report(
    top_words=5,              # words per pole (None skips)
    clusters=None,            # topn passed to cluster extractor
    extreme_docs=None,        # N most-pos + N most-neg
    misdiagnosed=None,        # N over + N under
)
```

**Groups:**
```python
gr.report(top_words=5, clusters=None, snippets_per_cluster=None)
```

**Render or save:**
```python
r = result.report(top_words=10, clusters=50, extreme_docs=5)

print(r)                 # plain text
r.save("report.md")      # .md .txt .html .tex .json .docx
```

`.docx` needs `ssdiff[results]`. Every other format is zero-dep.

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
Cluster      cluster_id, side, size, coherence, centroid_cos_beta, contrast
ClusterWord  cluster_id, side, word, cos_centroid, cos_beta, contrast
Snippet      snippet_id, side, doc_id, cosine, seed,
             start_token_idx, end_token_idx, start_sent_idx, end_sent_idx,
             text_window, text_surface, text_lemmas,
             cluster_id, contrast, post_id
Doc          doc_id, y_true, y_hat, residual, alignment_score
Pair         contrast, g1, g2, T, p_raw, p_corrected, cohens_d,
             n_g1, n_g2, contrast_norm
Suggestion   token, freq, cov_all, cov_bal, corr, pvalue, direction, rank
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
| Rerun PLS test | `result.test("perm", n_perm=5000)` |
| Rerun group test | `gr.test(n_perm=10000, correction="fdr_bh")` |
| Zoom to one pair | `gr[('g1','g2')]` |
| Per-pair top words | `gr[('g1','g2')].words.pos(20)` |
| Per-pair raw stats | `gr.pairs[('g1','g2')]` |
| Zoom to one rotated dim | `res['dim-1']` |
| Per-dim top words | `res['dim-1'].words.pos(20)` |
| Combined (unrotated) β words | `res['combined'].words` |
| Drop cache | `result.clear_cache()` |
| Re-wire after unpickle | `result.attach(corpus=c, embeddings=e)` |
| Silence repr footers | `ssdiff.set_repr_hints(False)` |

---

## See also

- [`api_reference.md`](api_reference.md) — `Embeddings`, `Corpus`, `SSD`, fit methods
- [`results_tables.md`](results_tables.md) — every view's columns and defaults
- [`architecture.md`](architecture.md) — backends, cache internals, how views compose
- [`../examples/demo_api.py`](../examples/demo_api.py) — runnable end-to-end demo
- [`../examples/demo_multipls.py`](../examples/demo_multipls.py) — rotated multi-component PLS (in development)
