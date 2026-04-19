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
| `split_ratio` |  | same |
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

### `clusters.pos.words(cid)` — `ClusterWordsView`

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
| `name` | ✓ | `"perm" \| "split" \| "split_cal"` |
| `pvalue` | ✓ | |
| `split_r2` | ✓ | only present after `"split"` / `"split_cal"` |
| `n_splits` |  | config — reachable via `.params` |
| `split_ratio` |  | config |
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
| `contrast` | ✓ | `"g1_vs_g2"` |
| `g1` |  | encoded in `contrast` |
| `g2` |  | encoded in `contrast` |
| `T` | ✓ | observed cosine distance |
| `p_raw` |  | users should read corrected value |
| `p_corrected` | ✓ | headline p after correction |
| `cohens_d` | ✓ | standardized effect size |
| `n_g1` |  | stable within a GroupResult; on `.stats` / `PairView.stats` |
| `n_g2` |  | same |
| `contrast_norm` |  | internal diagnostic |

### `pairs["A", "B"].stats` — `PairStatsView` (ScalarView)

| column | default | note |
|---|:---:|---|
| `T` | ✓ | |
| `p_raw` |  | corrected is headline |
| `p_corrected` | ✓ | |
| `cohens_d` | ✓ | |
| `n_g1` | ✓ | cell size matters at the pair level |
| `n_g2` | ✓ | same |
| `contrast_norm` |  | internal |

Asymmetry with `PairsListView`: the list is skimmed across many pairs, so cell sizes are a distraction; the per-pair scalar view is read in isolation, where imbalance is informative.

### `pairs["A", "B"].words` / `.clusters.pos` / `.snippets`

Same columns and defaults as the continuous `WordsView` / `SidedClustersView` / `SnippetsView`. The `contrast` column is populated (not `None`), and `side` / `cos_beta` / `T` are sign-flipped when the pair is accessed in reverse order.

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
