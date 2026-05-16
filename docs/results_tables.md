# ssdiff — Results Tables

Reference list of every view that ships with `ssdiff`: columns, defaults, and how to change what's printed or exported.

For usage patterns (`to_df`, `save`, `cols=…`, callable resizing) see [`results.md`](results.md).

---

## How defaults work

Every tabular and scalar view declares two things:

- `_columns` — the **full** ordered tuple of every available column.
- A default subset, registered by class name in `ssdiff.results.display.DEFAULT_COLS`.

The default applies uniformly to `__repr__`, `to_text`, `to_html`, `to_dict`, `to_records`, `to_df`, and `save()` — "what you see is what you save". Iteration (`for row in view: …`) is unaffected: rows are always the full domain dataclass.

### Changing the column subset

Three layers, from most local to most global:

```python
# 1. One call only (never mutates defaults)
result.words.save("w.csv", cols=["word", "cos_beta"])   # explicit subset
result.words.save("w.csv", cols="all")                  # escape — every column
result.words.to_df(cols=["word", "cos_beta"])

# 2. Override the registry for a single view class (process-wide)
from ssdiff.results import display
display.DEFAULT_COLS["WordsView"] = ("side", "rank", "word", "cos_beta", "contrast")

# 3. Silence the save-hint footer on all reprs
from ssdiff import set_repr_hints
set_repr_hints(False)
```

`cols=` resolution rules:

| Value | Resolves to |
|---|---|
| `None` (omitted) | registry default for that class; falls through to full `_columns` if no entry |
| `"all"` | every column in `_columns` |
| explicit sequence | validated against `_columns` — unknown names warn and drop |

### Changing row caps

Row-cap knobs are **per view**, not global. Each size-bearing view is callable:

| View | Default cap | How to change |
|---|---|---|
| `result.words.pos` / `.neg` | 20 (one-side) | `words.pos(50)`, `words.pos(None)` for all |
| `result.docs.pos() / .neg()` | 20 | `docs.pos(50)`, `docs.neg(50)` |
| `result.docs.misdiagnosed()` | 20, `direction="both"` | `docs.misdiagnosed(10, direction="over"\|"under")` |
| `result.clusters.pos` / `.neg` | `topn=100` neighbors → auto-K clusters | `clusters.pos(topn=50, k=5, k_min=2, k_max=10, min_cluster_size=2, random_state=2137)` |
| `result.snippets` | `top_per_side=30` | `snippets(top_per_side=200, min_cosine=0.4, n_jobs=-1)` |
| Everything else (slicing) | `view[:k]` | returns a `_no_trunc=True` sub-view |
| Terminal display only | `DEFAULT_MAX_ROWS = 20` | `view.to_text(max_rows=50)` |
| On save only | — | `view.save("x.csv", k=50)` caps without recomputing |

Cluster and snippet reruns with different parameters are cached separately — the original call's view remains available until you call `result.clear_cache("clusters")` / `.clear_cache("snippets")`.

---

## Continuous results — `PLSResult` / `PCAOLSResult`

### `stats` — `StatsView` (ScalarView, PLS) / `OLSStatsView` (PCA+OLS)

| column | default | note |
|---|:---:|---|
| `backend` | ✓ | identifies which fit produced this |
| `r2` | ✓ | headline fit quality |
| `r2_adj` | ✓ (OLS) | **OLS only** — PLS omits this column |
| `pvalue` | ✓ | headline significance |
| `n_raw` |  | derivable from `n_kept + n_dropped` |
| `n_kept` | ✓ | sample actually modelled |
| `n_dropped` |  | diagnostic — surface via `cols="all"` |
| `y_mean` |  | describes input `y`, not the fit |
| `y_std` |  | same |
| `beta_norm` |  | internal scale; `iqr_effect` is the interpretable version |
| `delta` |  | `0.1 · beta_norm · y_std` — engineering quantity |
| `iqr_effect` | ✓ | interpretable effect size: Q4−Q1 change in `y` |
| `y_corr_pred` |  | ≈ `√r²` for a linear fit — duplicate signal |

### `fit_info` — `FitInfoView` (ScalarView)

| column | default | note |
|---|:---:|---|
| `n_components` | ✓ | defines the fit |
| `pca_k` | ✓ | meaningful when PCA preprocess / PCA+OLS; `None` otherwise |
| `p_method` | ✓ | which significance test was used |
| `n_perm` |  | test config — on `.test` |
| `n_splits` |  | same |
| `split_mean_r` |  | surfaced as `split_r2` on `.test` |
| `random_state` | ✓ | reproducibility anchor |
| `k_min` / `k_max` / `k_step` |  | PCA-sweep config — on `.sweep` |
| `best_k` |  | PCA-sweep output — on `.sweep` |

### `words` — `WordsView` (tabular)

| column | default | note |
|---|:---:|---|
| `side` | ✓ | `"pos"` / `"neg"` |
| `rank` | ✓ | 1-based, per side |
| `word` | ✓ | vocabulary token |
| `cos_beta` | ✓ | cosine similarity to the semantic gradient |
| `contrast` |  | group-only; `None` for continuous |

### `words.pos` / `.neg` — `SidedWordsView`

Same columns + same defaults as `WordsView`. Row cap default = 20 (per side). Use `words.pos(50)` or `words.pos(None)` to resize.

### `clusters.pos` / `.neg` — `SidedClustersView`

| column | default | note |
|---|:---:|---|
| `cluster_id` | ✓ | |
| `side` |  | implied by the `.pos` / `.neg` accessor |
| `size` | ✓ | |
| `coherence` | ✓ | mean pairwise cosine among members |
| `centroid_cos_beta` | ✓ | |
| `contrast` |  | group-only |
| `top_words` | ✓ | comma-joined top-5 cluster words by `cos_centroid` desc (eager, cheap) |
| `top_snippet` |  | text of the highest-cosine snippet in this cluster; **opt-in** via `cols="all"` or explicit cols list (filling triggers full snippet extraction on first request); clipped to 40 chars in repr, full text in exports |

**Zoom:** `clusters.pos(cluster_id)` filters to one cluster. The repr appends a "Top 5 cluster snippets" sub-table (columns `seed`, `cosine`, `doc_id`, `text_window`) — independent of the `top_snippet` column on the main row. Pass `clusters.pos(cluster_id, top_snippets=N)` to resize the sub-table. Without an attached corpus, the sub-table is replaced by `(attach corpus to populate)`. `save()` on a zoomed view writes only the 1-row summary (`cols="all"` to include `top_snippet`); the snippet table is reached via `clusters.pos(cluster_id).snippets.save(...)`.

### `clusters.pos(cid).words` — `ClusterWordsView`

| column | default | note |
|---|:---:|---|
| `cluster_id` | ✓ | kept so rows stand alone when saved |
| `side` |  | implied by parent `SidedClustersView` |
| `word` | ✓ | |
| `cos_centroid` | ✓ | |
| `cos_beta` | ✓ | |
| `contrast` |  | group-only |

### `snippets` — `SnippetsView` (+ `SidedSnippetsView`)

| column | default | note |
|---|:---:|---|
| `snippet_id` |  | internal numbering |
| `side` | ✓ | |
| `doc_id` | ✓ | jump point — `result.docs.id(doc_id)` |
| `cosine` | ✓ | alignment along β |
| `seed` | ✓ | which lexicon term anchored this snippet |
| `start_token_idx` / `end_token_idx` |  | reconstruction detail |
| `start_sent_idx` / `end_sent_idx` |  | reconstruction detail |
| `text_window` | ✓ | "what does this mean" column; truncated to 40 chars in terminal |
| `text_surface` |  | full enclosing sentence(s) |
| `text_lemmas` |  | lemmatized variant |
| `cluster_id` |  | set when clusters were computed |
| `post_id` |  | multi-post docs (forums) |
| `contrast` |  | group-only |

### `docs` — `DocsView`

| column | default | note |
|---|:---:|---|
| `doc_id` | ✓ | aligns with corpus row index |
| `y_true` | ✓ | |
| `y_hat` | ✓ | |
| `residual` | ✓ | `y_true − y_hat` |
| `alignment_score` | ✓ | cosine along β — complementary to `y_hat` |

Default = full. Nothing to prune.

### `test` — `PLSTestView` (ScalarView)

| column | default | note |
|---|:---:|---|
| `name` | ✓ | `"raw_perm" \| "split_nb" \| "split_perm"` |
| `pvalue` | ✓ | |
| `split_r2` | ✓ | only present after `"split_nb"` / `"split_perm"` |
| `n_splits` |  | config — reachable via `.params` |
| `n_perm` |  | config |
| `random_state` |  | config |

### `test` — `PCAOLSTestView` (ScalarView)

Columns: `name`, `pvalue`. Default = full (rerun is a no-op; F-test is analytic).

### `sweep` — `SweepView` (PCAOLSResult only)

Columns: `k`, `r2`, `r2_adj`, `pvalue`. One row per K tested. Default = full.

---

## `GroupResult`

### `stats` — `GroupStatsView` (ScalarView)

| column | default | note |
|---|:---:|---|
| `G` | ✓ | number of groups |
| `n_kept` | ✓ | |
| `n_perm` |  | config — on `.test.params` |
| `correction` |  | same |
| `random_state` |  | same |
| `pvalue` | ✓ | omnibus p |

### `test` — `GroupTestView` (ScalarView)

| column | default | note |
|---|:---:|---|
| `name` | ✓ | |
| `pvalue` | ✓ | canonical p — same value as `omnibus_p` |
| `omnibus_T` | ✓ | omnibus test statistic |
| `omnibus_p` |  | redundant with `pvalue` |
| `G` |  | on `.stats` |
| `n_kept` |  | on `.stats` |
| `n_perm` |  | config |
| `correction` |  | config |
| `random_state` |  | config |

### `pairs` — `PairsListView`

| column | default | note |
|---|:---:|---|
| `contrast` | ✓ | `"g1_g2"` |
| `g1` |  | encoded in `contrast` |
| `g2` |  | encoded in `contrast` |
| `T` | ✓ | observed cosine distance |
| `p_raw` |  | users should read corrected value |
| `p_corrected` | ✓ | headline p after correction |
| `cohens_d` | ✓ | standardized effect size |
| `n_g1` |  | stable within a GroupResult; on `.stats` |
| `n_g2` |  | same |
| `contrast_norm` |  | internal diagnostic |

Tuple-key lookup `gr.pairs[('g1', 'g2')]` returns the raw `Pair` dataclass directly. Canonical order only — reverse order raises `KeyError`, no sign-flip.

### Per-pair views — `gr[('g1', 'g2')].words` / `.clusters` / `.snippets`

`gr[('g1', 'g2')]` returns a `PairResult` for that pair. Its `.words`, `.clusters`, and `.snippets` attributes dispatch to the same view classes as continuous results, with the `contrast` column populated (`"g1_g2"`) rather than `None`. Reverse-order access `gr[('g2', 'g1')]` is normalized to canonical order, so it returns the identical view — no sign-flip.

**`gr[('g1', 'g2')].words` — `WordsView`**

Same columns and defaults as the continuous `WordsView` (see above). `contrast` is `"g1_g2"`.

**`gr[('g1', 'g2')].clusters.pos` / `.neg` — `ClustersViewSided`**

Same columns and defaults as the continuous `ClustersViewSided`. `contrast` is populated on every row.

**`gr[('g1', 'g2')].clusters.pos.words` — `ClusterWordsViewSided`**

Same columns and defaults as the continuous `ClusterWordsViewSided`. `contrast` is populated.

**`gr[('g1', 'g2')].snippets` — `SnippetsView`**

**`gr[('g1', 'g2')].snippets.pos` / `.neg` — `SnippetsViewSided`**

Same columns and defaults as the continuous `SnippetsView` / `SnippetsViewSided`. `contrast` is populated.

*Power-user shortcut:* top-level paired containers also accept tuple indexing directly — `gr.words[('g1', 'g2')]`, `gr.clusters[('g1', 'g2')]`, `gr.snippets[('g1', 'g2')]` — skipping the `PairResult` construction when you only need one view (canonical order only).

---

## `MultiPLSResult` *(in development)*

Shared model-level views. Per-leaf views (`res['dim-1'].words` etc.) reuse the continuous `WordsView` / `SidedWordsView` classes and their defaults documented above.

### `stats` — `MultiPLSStatsView` (ScalarView)

| column | default | note |
|---|:---:|---|
| `r2` | ✓ | model-level R² from the unrotated combined β |
| `pvalue` | ✓ | shared whole-model p (raw_perm / split_nb / split_perm) |
| `n` | ✓ | documents in fit |
| `n_components` | ✓ | number of rotated dims |
| `rotate` | ✓ | `"varimax"` / `"raw"` |

### `pls_info` — `PLSInfoView` (ScalarView)

| column | default | note |
|---|:---:|---|
| `n_components` | ✓ | |
| `rotate` | ✓ | |
| `order` |  | dim permutation applied post-rotation (identity once returned) |
| `signs` |  | per-dim sign flips applied for `corr(t_i, y) > 0` |
| `kaiser_normalized` |  | varimax Kaiser row-normalisation flag |
| `sweeps` |  | varimax pairwise-sweep count |
| `V_converged` |  | final varimax criterion value |
| `pvalue_source` | ✓ | which test produced `stats.pvalue` |
| `random_state` | ✓ | reproducibility anchor |

### `test` — `MultiPLSTestView` (ScalarView)

Same columns and semantics as `PLSTestView` (see above) — both go through `backends.pls.run_signal_test`, which dispatches to `plskit.pls1_signal_test`.

### Per-leaf — `res['dim-i'].words`, `res['combined'].words`

Same columns and defaults as the continuous `WordsView` (see [Continuous results](#continuous-results--plsresult--pcaolsresult)). Per-leaf `.clusters` / `.snippets` / `.docs` are reserved for a later milestone.

---

## `LexiconResult`

### `stats` — `LexiconStatsView` (ScalarView)

Columns: `var_type`, `n_docs`, `n_tokens`. Default = full.

### `suggestions` — `SuggestionsView`

| column | default | note |
|---|:---:|---|
| `rank` | ✓ | orders the table meaningfully |
| `token` | ✓ | |
| `freq` | ✓ | trust signal for the correlation |
| `cov_all` |  | coverage is secondary; `corr` drives ranking |
| `cov_bal` |  | same |
| `corr` | ✓ | primary ranking signal |
| `pvalue` | ✓ | |
| `direction` | ✓ | `"positive" \| "negative" \| "none"` |

### `summary` — `SummaryView` (ScalarView; present only after `evaluate_lexicon`)

| column | default | note |
|---|:---:|---|
| `docs_any` | ✓ | docs with ≥ 1 lexicon hit |
| `cov_all` | ✓ | headline coverage |
| `q1` |  | quartile breakdown — secondary |
| `q4` |  | same |
| `corr_any` | ✓ | headline lexicon↔y correlation |
| `hits_mean` |  | mean can be noisy with skewed counts |
| `hits_median` | ✓ | robust central tendency |
| `types_mean` |  | — |
| `types_median` |  | niche vs. `hits_median` |
| `group_cov` |  | only populated when groups are provided |

---

## Quick recipes

```python
# Change a default globally, process-wide
from ssdiff.results import display
display.DEFAULT_COLS["SnippetsView"] = (
    "side", "doc_id", "cosine", "seed", "text_window", "text_surface"
)

# One-off: save everything, including diagnostic columns
result.snippets.save("snippets_full.csv", cols="all")

# One-off: a bespoke column order
result.docs.to_df(cols=["doc_id", "y_hat", "y_true", "residual"])

# Resize a view before exporting
result.words.pos(50).save("top50_pos.xlsx")     # needs ssdiff[results]
result.words.pos.save("top50_pos.csv", k=50)    # equivalent

# Clear specific caches
result.clear_cache("clusters")
result.clear_cache("snippets")
result.clear_cache()     # everything

# Disable save-hint footers in all reprs
from ssdiff import set_repr_hints
set_repr_hints(False)
```

---

## See also

- [`results.md`](results.md) — usage patterns for views, reports, reruns.
- [`api_reference.md`](api_reference.md) — fit methods that produce these results.
- [`architecture.md`](architecture.md) — how views compose internally.
