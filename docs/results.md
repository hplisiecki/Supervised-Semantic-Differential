# ssdiff — Results API

Full surface of the result objects returned by `SSD.fit_*` and `Corpus.*_lexicon`.

For arguments of the fit methods themselves see [`api_reference.md`](api_reference.md).
For per-view column listings and how to change column defaults see [`results_tables.md`](results_tables.md).

---

## Contents

1. [Returned objects](#returned-objects)
2. [Common patterns](#common-patterns)
   - [View vs ScalarView vs TestView](#view-vs-scalarview-vs-testview)
   - [`to_*` conversion methods](#to_-conversion-methods)
   - [`save()`](#save)
   - [Columns: `cols=`, `"all"`, and defaults](#columns-cols-all-and-defaults)
   - [Row caps & callable resizing](#row-caps--callable-resizing)
   - [Repr hints & display toggles](#repr-hints--display-toggles)
3. [Result base class — `attach`, cache, reports](#result-base-class)
4. [`PLSResult`](#plsresult) / [`PCAOLSResult`](#pcaolsresult)
5. [`GroupResult`](#groupresult) and [`PairView`](#pairview)
6. [`LexiconResult`](#lexiconresult)
7. [`Report`](#report)
8. [Domain row dataclasses](#domain-row-dataclasses)
9. [Optional dependencies](#optional-dependencies)

---

## Returned objects

| Returned by | Class |
|---|---|
| `SSD.fit_pls()` | `PLSResult` |
| `SSD.fit_ols()` | `PCAOLSResult` |
| `SSD.fit_groups()` | `GroupResult` |
| `Corpus.suggest_lexicon()` / `.evaluate_lexicon()` | `LexiconResult` |
| `gr.pairs["A", "B"]` | `PairView` (ephemeral — built on access) |

All import from `ssdiff`:

```python
from ssdiff import (
    PLSResult, PCAOLSResult, GroupResult, LexiconResult,
    PairView, Result, set_repr_hints,
)
```

---

## Common patterns

### View vs ScalarView vs TestView

- **`View[T]`** — tabular views yielding frozen `T` dataclasses. Support `len(v)`, `v[i]`, `v[:k]`, `iter(v)`, `list(v)`.
- **`ScalarView`** — single-row views (`stats`, `fit_info`, `summary`). Expose row fields via attribute **and** `v["key"]` access: `stats.r2`, `stats["r2"]`.
- **`TestView`** — callable ScalarView that **reruns** the statistical test when invoked: `result.test(name="perm", n_perm=5000)` mutates the view in place and returns `self`.

Every view exposes:

| Attribute / method | Returns |
|---|---|
| `v.columns` | `tuple[str, ...]` of all column names |
| `v.params` | `dict` of parameters that produced it (empty for parameterless views) |
| `v.to_dict()` | `list[dict]` (tabular) or `dict` (scalar) |
| `v.to_records()` | `list[tuple]` |
| `v.to_df()` | `pandas.DataFrame` (needs `ssdiff[results]`) |
| `v.to_html()` | HTML string (used by Jupyter repr) |
| `v.to_text()` | aligned plain-text table |
| `v.save(path, *, cols=None, k=None)` | writes to disk, extension picks the format |

### `to_*` conversion methods

```python
v.to_dict(cols=None)      # list[dict] | dict     — always works, no optional deps
v.to_records(cols=None)   # list[tuple]
v.to_df(cols=None)        # pandas.DataFrame      — needs pandas
v.to_html(cols=None)      # HTML string           — used by _repr_html_, not a file writer
v.to_text(max_rows=None, cols=None)  # aligned text, cell truncation applied
```

- `cols=None` → view's **default** subset (see `results_tables.md`).
- `cols="all"` → every available column.
- `cols=[...]` → explicit selection + ordering. Unknown names emit a `UserWarning` and are dropped; if none are valid, falls back to the full column set.
- Iteration (`for row in v: …`) yields **full** dataclasses — column projection is an output concern, not a data concern.

### `save()`

```python
v.save(path, *, cols=None, k=None)
```

Extension dispatch. Supported:

| Extension | Needs extra | Notes |
|---|---|---|
| `.csv` | — | UTF-8, header row |
| `.json` | — | pretty-printed, numpy scalars coerced |
| `.md` | — | GitHub-flavoured table |
| `.tex` | — | booktabs-style `\begin{tabular}` |
| `.txt` | — | `to_text()` output (truncation applied) |
| `.html` | — | `to_html()` output |
| `.xlsx` | `ssdiff[results]` | pandas + openpyxl |
| `.docx` | `ssdiff[results]` | python-docx |

- `k=N` caps rows for size-bearing views before writing (words, docs, clusters, snippets, pairs, suggestions). Ignored on single-row scalar views.
- Passing an unsupported extension raises `ValueError` with the supported list.

### Columns: `cols=`, `"all"`, and defaults

Display-layer defaults live in `ssdiff.results.display.DEFAULT_COLS` keyed by view class name. Defaults apply **uniformly** to `__repr__`, `to_text`, `to_html`, `to_dict`, `to_records`, `to_df`, and `save()` — "what you see is what you save".

```python
from ssdiff.results import display

display.DEFAULT_COLS["WordsView"]
# ('side', 'rank', 'word', 'cos_beta')

# Override globally (mutates the registry):
display.DEFAULT_COLS["WordsView"] = ("side", "rank", "word", "cos_beta", "contrast")

# One-off override:
result.words.save("all_cols.csv", cols="all")
result.words.to_df(cols=["word", "cos_beta"])
```

See [`results_tables.md`](results_tables.md) for every view's default and the rationale.

### Row caps & callable resizing

Size-bearing views are **callable**. Calling re-slices or recomputes; the new view respects the cap everywhere (display **and** export).

| View | Default | Call syntax | Behaviour |
|---|---|---|---|
| `result.words.pos` / `.neg` | 20 rows | `result.words.pos(50)`, `result.words.pos(None)` | re-slice (no recompute) |
| `result.docs.pos()` / `.neg()` / `.misdiagnosed()` | 20 | `result.docs.pos(50)` | re-sort + slice |
| `result.clusters.pos` / `.neg` | `topn=100` | `result.clusters.pos(topn=50, k=5, …)` | **re-cluster** (cached per param set) |
| `result.snippets` | `top_per_side=30` | `result.snippets(top_per_side=200, min_cosine=0.4)` | **re-extract** (cached per param set) |
| `result.report(...)` | see below | `report(top_words=10, clusters=30, …)` | new `Report` object |

`save(..., k=N)` caps rows directly without recomputing:

```python
result.words.pos.save("top50.csv", k=50)       # equivalent to words.pos(50).save("top50.csv")
result.docs.misdiagnosed(10).to_df()
```

### Repr hints & display toggles

Every `__repr__` ends with a one-line `Save: …` hint showing the idiomatic export call. To silence it globally (e.g. for log-stream consumers like the SSD_APP GUI):

```python
from ssdiff import set_repr_hints
set_repr_hints(False)   # no footer on repr / _repr_html_
set_repr_hints(True)    # re-enable
```

Table display truncates to `DEFAULT_MAX_ROWS = 20` with a `... N more rows` footer; slicing (`view[:k]`) or calling the view with an explicit count produces a `_no_trunc=True` view that prints every row. Long text columns (e.g. `Snippet.text_window`) are clipped to 40 chars in the terminal only — data exports keep the full value.

---

## Result base class

All result classes inherit from `ssdiff.results.Result`:

```python
class Result:
    corpus: Corpus | None          # set at construction or via attach()
    embeddings: Embeddings | None  # ditto

    def attach(self, corpus=None, embeddings=None) -> Self
    def clear_cache(self, view: str | None = None) -> None
    def report(self, **kwargs) -> Report   # subclass-specific kwargs
    def to_text(self) -> str               # header + discoverability block
    def to_html(self) -> str               # ditto, HTML
```

### `attach(...)` — re-wire after un-pickling

After loading a pickled result, views that need live data (`words`, `clusters`, `snippets`, `docs.id(...)`) raise `RuntimeError` with an actionable message until you reattach:

```python
import pickle
with open("result.pkl", "rb") as f:
    result = pickle.load(f)

result.attach(corpus=my_corpus, embeddings=my_embeddings)
result.words   # now works
```

`attach()` returns `self` so it chains: `pickle.load(f).attach(corpus=c, embeddings=e)`.

### Cache behaviour

Views computed from parameters (`clusters`, `snippets`, some future extensions) are cached on first access and keyed by `(view_name, frozen_params)`. Requesting the same view with different parameters stores a **separate** cache entry — earlier results survive:

```python
result.clusters.pos                 # cached entry (topn=100, k=None, …)
result.clusters.pos(topn=50)        # separate entry (topn=50, k=None, …)
result.clusters.pos                 # still the topn=100 entry

result.clear_cache()                # drop everything
result.clear_cache("clusters")      # drop every clusters.* entry
result.clear_cache("snippets")      # drop every snippets.* entry
```

Parameterless views (`stats`, `fit_info`, `docs`, `words`) are computed once in `__init__` or on first property access and aren't stored in the cache at all — they live directly on the instance.

### Reports

`result.report(**kwargs) -> Report`. Each result subclass accepts its own kwargs (documented below). See [`Report`](#report) for how to render or save the returned object.

---

## `PLSResult`

Returned by `SSD.fit_pls(...)`. Inherits the full continuous-result surface.

### Scalar views

| Attribute | Class | Notes |
|---|---|---|
| `.stats` | `StatsView` | `backend`, `r2`, `pvalue`, `n_raw`, `n_kept`, `n_dropped`, `y_mean`, `y_std`, `beta_norm`, `delta`, `iqr_effect`, `y_corr_pred` |
| `.fit_info` | `FitInfoView` | `n_components`, `pca_k`, `p_method`, `n_perm`, `n_splits`, `split_ratio`, `split_mean_r`, `random_state`, `k_min/k_max/k_step/best_k` |
| `.test` | `PLSTestView` | rerun via `result.test(name="perm"|"split"|"split_cal", **params)` |

### Arrays (numpy, no pandas needed)

| Attribute | Shape | Meaning |
|---|---|---|
| `.x` | `(n_kept, D)` | Per-document **personal concept vectors** (PCVs) after filtering. |
| `.y` | `(n_kept,)` | Outcome on the original scale. |
| `.beta` | `(D,)` | Raw regression direction in embedding space (carries magnitude). |
| `.gradient` | `(D,)` | Unit-length β — the **semantic gradient**. |
| `.beta_norm` | scalar | `‖β‖`. |
| `.alignment_scores` | `(n_kept,)` | `d_i · gradient`, cached on first access. |
| `.component_scores` | `(n_kept, A)` | PLS1 X-scores `T`. |
| `.component_weights` | `(D, A)` | PLS1 X-weights `W` in embedding space. |
| `.cv_result`, `.cv_scores`, `.perm_null` | various | CV + permutation diagnostics (`None` if not computed). |
| `.n_components` | int | Echo of `fit_info.n_components`. |

### Tabular views

| Attribute | Class | Default | Callable? |
|---|---|---|---|
| `.words` | `WordsView` | top 100 / side | — |
| `.words.pos`, `.words.neg` | `SidedWordsView` | 20 rows | `words.pos(k)`, `words.pos(None)` |
| `.clusters` | `ClustersIndex` | — | — |
| `.clusters.pos`, `.clusters.neg` | `SidedClustersView` | `topn=100` | `clusters.pos(topn=50, k=5, k_min=2, k_max=10, min_cluster_size=2, random_state=2137)` |
| `.clusters.pos.words(cid)` | `ClusterWordsView` | — | — |
| `.clusters.pos.snippets` | `SidedSnippetsView` | — | `clusters.pos.snippets(cluster_id=0)` |
| `.snippets` | `SnippetsView` | `top_per_side=30` | `snippets(top_per_side=200, min_cosine=0.4, n_jobs=-1)` |
| `.docs` | `DocsView` | preview (β-pos 5 / β-neg 5) in terminal | `docs.pos(k)`, `docs.neg(k)`, `docs.misdiagnosed(k, direction="both"|"over"|"under")`, `docs.id(doc_id) → DocDetailView` |

`DocsView` extras:

```python
result.docs.pos(10)                                # 10 docs most aligned with β-pos
result.docs.neg(10)                                # β-neg
result.docs.misdiagnosed(5)                        # largest |residual|
result.docs.misdiagnosed(5, direction="over")      # y_hat > y_true
result.docs.misdiagnosed(5, direction="under")     # y_hat < y_true
result.docs.id(42)                                 # DocDetailView; needs corpus for raw text
```

### Rerunning the PLS test

```python
result.test                         # current test (from fit)
result.test.pvalue                  # p-value
result.test.name                    # "perm" | "split" | "split_cal"
result.test.params                  # dict of test-specific params

# Rerun — updates .test and propagates the new p-value into .stats
result.test("perm", n_perm=5000, seed=0)
result.test("split", n_splits=100, split_ratio=0.5)
result.test("split_cal", n_splits=50, n_perm=2000)
```

Calling with no `name` reuses the previous test name (falling back to `"split"` on a fresh fit).

### `PLSResult.report(...)`

```python
result.report(
    top_words=5,              # words per pole (None skips the section)
    clusters=None,             # topn passed to cluster extractor (None skips)
    snippets_per_cluster=None, # reserved
    extreme_docs=None,         # N most-pos + N most-neg docs (None skips)
    misdiagnosed=None,         # N over + N under (None skips)
) -> Report
```

---

## `PCAOLSResult`

Everything on `PLSResult` above, **plus**:

### Extra scalar / array attributes

| Attribute | Description |
|---|---|
| `.stats` | `OLSStatsView` — adds `r2_adj` after `r2`. |
| `.pca_k` | Number of PCA components used. |
| `.pca_components` | `(K, D)` PCA loadings `V_K`. |
| `.pca_weights` | `(K,)` OLS coefficients in PCA space. |
| `.sweep_result` | Full PCA-sweep diagnostics (or `None` when `fixed_k` was used). |

### Extra views

| Attribute | Class | Notes |
|---|---|---|
| `.sweep` | `SweepView` | Columns: `k`, `r2`, `r2_adj`, `pvalue`. One row per tested K. |
| `.test` | `PCAOLSTestView` | Currently F-test only; rerun is a no-op returning the stored p-value. |

### `plot_sweep`

```python
png_bytes = result.plot_sweep(path=None, *, dpi=300)
```

Dual-axis chart of detrended interpretability (left) vs smoothed β-change (right), with a vertical line at the selected K. Raises `RuntimeError` if `fixed_k` was set (no sweep to plot) and `ImportError` if matplotlib isn't installed. Returns the PNG bytes even when `path` is `None` (in which case the figure is also shown interactively).

---

## `GroupResult`

Returned by `SSD.fit_groups(...)`.

### Scalars

| Attribute | Type | Description |
|---|---|---|
| `.G` | int | Number of groups. |
| `.n_kept` | int | Documents retained after `filter_small_groups`. |
| `.n_perm` | int | Permutations used. |
| `.correction` | str | `"holm" \| "bonferroni" \| "fdr_bh" \| "none"`. |
| `.random_state` | int | Seed. |
| `.x` | `(n_kept, D)` ndarray | PCVs retained after filtering. |
| `.groups` | `(n_kept,)` ndarray | Group labels aligned with `.x`. |

### Views

| Attribute | Class | Notes |
|---|---|---|
| `.stats` | `GroupStatsView` | `G`, `n_kept`, `n_perm`, `correction`, `random_state`, `pvalue` (= omnibus p). |
| `.test` | `GroupTestView` | Permutation omnibus + pairwise; rerun via `gr.test(n_perm=..., correction=..., random_state=...)`. |
| `.pairs` | `PairsListView` | Iterable of `Pair` rows; tuple-indexed lookup returns a `PairView`. |

### `gr.test(...)` — rerun the permutation test

```python
gr.test                             # GroupTestView
gr.test.omnibus_T                   # observed mean pairwise cosine distance
gr.test.omnibus_p                   # omnibus p-value
gr.test.pvalue                      # alias (same value)

gr.test(n_perm=10000, correction="fdr_bh", random_state=0)
# replaces gr.pairs with new pair statistics
# refreshes gr.stats with the new pvalue/correction
```

Rerunning preserves the analysis-time `embeddings` / `corpus` attachments and the pair-level sub-view rows (`words`, `clusters`, `snippets`) — only the statistics change.

### `gr.pairs` — list view + tuple lookup

```python
for p in gr.pairs:
    print(p.contrast, p.T, p.p_corrected, p.cohens_d)

# Tuple lookup returns a PairView. Accessing in reverse order flips signs:
pv = gr.pairs["A", "B"]
pv_rev = gr.pairs["B", "A"]
pv.pair.T == -pv_rev.pair.T          # True
pv.pair.p_corrected == pv_rev.pair.p_corrected   # True (p-values unchanged)
```

`gr.pairs[i]` returns the raw `Pair` dataclass at index `i`. `gr.pairs[slice]` returns a sliced `PairsListView`.

---

## `PairView`

Ephemeral view built on `gr.pairs["A", "B"]`. Mirrors the continuous-result surface for the `A → B` contrast.

### Scalars / arrays

| Attribute | Description |
|---|---|
| `.pair` | `Pair` dataclass (sign-flipped if accessed reverse) |
| `.contrast` | Label `"A_vs_B"` (or `"B_vs_A"` when reversed) |
| `.beta` | `(D,)` — contrast vector `c_g1 − c_g2` (carries magnitude) |
| `.gradient` | `(D,)` — `beta / ‖beta‖` |
| `.beta_norm` | `‖beta‖` (invariant under reverse) |
| `.alignment_scores` | `(n_kept,)` — `x @ gradient` |

### Views

| Attribute | Class | Notes |
|---|---|---|
| `.stats` | `PairStatsView` | `T`, `p_raw`, `p_corrected`, `cohens_d`, `n_g1`, `n_g2`, `contrast_norm` |
| `.words` | `WordsView` | Filtered to this contrast; sides flipped on reverse access |
| `.clusters` | `_PairClustersIndex` | `.pos` / `.neg` → `SidedClustersView` (no callable recompute on this variant) |
| `.snippets` | `SnippetsView` | Filtered to this contrast; sides flipped on reverse access |

Sign handling: `PairView` lazily detects whether you looked it up in canonical or reverse order and flips `T`, `cohens_d`, `contrast_norm`, `beta`, `gradient`, `cos_beta`, and `side` labels accordingly. P-values and cluster `coherence` / `size` do not flip.

---

## `LexiconResult`

Returned by `Corpus.suggest_lexicon(...)` (without `.summary`) or `Corpus.evaluate_lexicon(...)` (with `.summary`).

### Views

| Attribute | Class | Notes |
|---|---|---|
| `.stats` | `LexiconStatsView` | `var_type`, `n_docs`, `n_tokens` |
| `.suggestions` | `SuggestionsView` | `token`, `freq`, `cov_all`, `cov_bal`, `corr`, `pvalue`, `direction`, `rank` |
| `.tokens` | `list[str]` | Convenience — suggestion tokens in rank order |
| `.summary` | `SummaryView \| None` | Only populated by `evaluate_lexicon`; `docs_any`, `cov_all`, `q1`, `q4`, `corr_any`, `hits_mean`, `hits_median`, `types_mean`, `types_median`, `group_cov` |

### Report

```python
lex.report(top=20).save("lexicon.md")
```

Builds a three-section report: stats (K/V), top-N suggestions (table), and coverage summary (K/V, only when `.summary` is populated). `cite=False` by default on lexicon reports.

---

## `Report`

Returned by every `result.report(...)` call. A lightweight multi-section builder with format-agnostic rendering.

### Construction fields

```python
@dataclass
class Report:
    title: str
    sections: list[Section]
    subtitle: str | None = None
    cite: bool = True            # appends the Plisiecki et al. (2025) citation
```

### Rendering & saving

```python
r = result.report(top_words=10, clusters=50, extreme_docs=5, misdiagnosed=5)

r.to_text()       # plain text (fixed-width, also used by __repr__)
r.to_html()       # HTML string (used by _repr_html_)

r.save("report.md")     # .md  .txt  .html  .tex  .docx  .json
```

Extension dispatch:

| Extension | Renderer | Needs extra |
|---|---|---|
| `.md` | GitHub-flavoured markdown | — |
| `.txt` | `to_text()` | — |
| `.html` | `to_html()` | — |
| `.tex` | booktabs LaTeX | — |
| `.json` | sections array (title / kind / headers / rows) | — |
| `.docx` | python-docx | `ssdiff[results]` |

Sections come in three `kind`s: `"kv"` (key/value pairs), `"table"` (multi-column), `"list"` (bullets). Custom reports can be assembled manually:

```python
from ssdiff.results.report import Report, Section

Report(
    title="My combined report",
    sections=[
        Section(title="Fit A", kind="kv",   rows=[("r²", 0.23), ("p", 0.004)]),
        Section(title="Words", kind="table", headers=["side", "word"],
                rows=[["pos", "hope"], ["neg", "fear"]],
                numeric=[False, False]),
    ],
    cite=False,
).save("combined.md")
```

---

## Domain row dataclasses

Iterating any view yields frozen dataclasses (not dicts) — IDEs autocomplete the fields.

```python
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

Stats        backend, r2, pvalue, n_raw, n_kept, n_dropped,
             y_mean, y_std, beta_norm, delta, iqr_effect, y_corr_pred, r2_adj
FitInfo      n_components, pca_k, p_method, n_perm, n_splits, split_ratio,
             split_mean_r, random_state, k_min, k_max, k_step, best_k
Summary      docs_any, cov_all, q1, q4, corr_any,
             hits_mean, hits_median, types_mean, types_median, group_cov
```

Conventions:

- `side ∈ {"pos", "neg"}` — β-direction pole.
- `direction ∈ {"positive", "negative", "none"}` on `Suggestion` — sign of the outcome correlation.
- `direction ∈ {"both", "over", "under"}` on `DocsView.misdiagnosed()` — residual sign (distinct axis from `side`).
- `contrast` is `None` on continuous results; `"g1_vs_g2"` on pairs / group results.

Import from `ssdiff.results.schema` if you need the types themselves (e.g. for `isinstance` checks or type hints).

---

## Optional dependencies

The core library has **no** hard dependency on pandas / openpyxl / python-docx. Install the extra when you need them:

```bash
pip install ssdiff[results]
```

| Need | Optional? |
|---|---|
| iterate views, `to_dict`, `to_records`, `save('.csv'/'.json'/'.md'/'.tex'/'.txt'/'.html')` | no |
| `to_df()` or `save('.xlsx')` | **pandas + openpyxl** |
| `save('.docx')` on views or reports | **python-docx** |
| `save('.md'/'.tex'/'.txt'/'.html'/'.json')` on reports | no |
| `result.plot_sweep(...)` | **matplotlib** (not bundled in the `results` extra) |

Missing optional deps raise `ImportError` with the install hint.

---

## Cheat sheet

| Task | Code |
|---|---|
| Headline stats | `result.stats`  (or `result.stats.r2`, `result.stats["pvalue"]`) |
| Full narrative | `print(result.report())` |
| Save report | `result.report().save("r.md" \| "r.docx" \| "r.tex" \| "r.html")` |
| Top 50 pos words → DataFrame | `result.words.pos(50).to_df()` |
| All pos words → CSV | `result.words.pos(None).save("pos.csv")` |
| Worst-predicted docs | `result.docs.misdiagnosed(10, direction="over")` |
| Single doc + raw text | `result.docs.id(42)` (corpus attached) |
| Recompute snippets | `result.snippets(top_per_side=200, min_cosine=0.4)` |
| Snippets inside cluster 3 (pos) | `result.clusters.pos.snippets(cluster_id=3)` |
| Rerun PLS test | `result.test("perm", n_perm=5000)` |
| Rerun group test | `gr.test(n_perm=10000, correction="fdr_bh")` |
| Per-pair view | `gr.pairs["A", "B"]` (reverse-order flips signs) |
| Drop all cached views | `result.clear_cache()` |
| Re-wire after unpickle | `result.attach(corpus=c, embeddings=e)` |
| Silence repr footers | `ssdiff.set_repr_hints(False)` |

---

## See also

- [`api_reference.md`](api_reference.md) — `Embeddings`, `Corpus`, `SSD`, fit methods.
- [`results_tables.md`](results_tables.md) — every view's columns, defaults, and rationale.
- [`architecture.md`](architecture.md) — backends, cache internals, how views compose.
- [`demo_new_api.py`](demo_new_api.py) — runnable end-to-end demo.
