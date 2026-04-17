# ssdiff — Results Layer

Guide to working with results returned by `SSD.fit_pls()`, `SSD.fit_ols()`, `SSD.fit_groups()`, and `Corpus.suggest_lexicon()` / `Corpus.evaluate_lexicon()`.

> **Status:** This document describes the **v2 results layer** (planned for `ssdiff` ≥ 2.0). The v1 API (`result.cluster_neighbors("pos")`, `result.snippets(...)`, `gr.report()`) is still in place and will continue to work in 1.x. v2 is a clean rewrite of the result/display surface — see [Migration from v1](#migration-from-v1) at the bottom.

---

## At a glance

```python
result = ssd.fit_pls()

# Navigate
result.stats                            # scalars: r2, pvalue, n_kept, …
result.words.df()                       # DataFrame of top words per pole
result.clusters.pos[0]                  # Cluster object
result.clusters.pos[0].words            # words inside that cluster
result.clusters.pos[0].snippets         # snippets anchored in that cluster
result.snippets.df()                    # all snippets, flat

# Export
result.words.to_csv("words.csv")
result.clusters.pos.to_excel("clusters_pos.xlsx")
result.snippets.to_docx("snippets.docx")

# Report (multi-format)
result.report().text()                  # terminal
result.report().markdown()              # markdown
result.report().html()                  # styled HTML (also _repr_html_ in Jupyter)
result.report().save("report.docx")     # dispatched on extension
```

Group results expose the same interface, with `gr.pairs["A","B"]` giving you a per-contrast view shaped exactly like a continuous result.

---

## Result types

| Returned by | Class | Inherits |
|---|---|---|
| `SSD.fit_pls()` | `PLSResult` | `ContinuousResult` |
| `SSD.fit_ols()` | `PCAOLSResult` | `ContinuousResult` |
| `SSD.fit_groups()` | `GroupResult` | `Result` |
| `Corpus.suggest_lexicon(...)` | `LexiconResult` | `Result` |
| `Corpus.evaluate_lexicon(...)` | `LexiconResult` | `Result` |

All five share the same view / export / report / caching machinery. They differ only in **which views they expose** (e.g. `GroupResult` has `.pairs` and `.omnibus`; `LexiconResult` has `.suggestions` and `.summary`).

```
                 ┌──────────────┐
                 │   Result     │  abstract: .stats, .report(), .save(), .clear_cache()
                 └──────┬───────┘
        ┌───────────────┼─────────────────────┐
        ▼               ▼                     ▼
ContinuousResult    GroupResult           LexiconResult
.words              .pairs[g1,g2] →       .suggestions
.clusters.pos/neg     PairView            .summary
.snippets           .omnibus
.docs               (also flat .words,
                     .clusters, .snippets
                     across all pairs)
   ▼
PLSResult / PCAOLSResult
(small backend-specific extras)
```

---

## The view interface

Every "view" object on a Result implements the same contract:

| Method / attr | Returns | Notes |
|---|---|---|
| `len(view)` | `int` | Number of rows |
| `iter(view)` | yields domain objects | `Word`, `Cluster`, `Snippet`, … |
| `view[i]` | one domain object | by integer index |
| `view.where(**filters)` | sub-view | e.g. `clusters.where(side="pos", min_size=3)` |
| `view.df()` | `pandas.DataFrame` | requires `[results]` extra |
| `view.to_dict()` | `list[dict]` | always available, stdlib only |
| `view.to_records()` | `list[tuple]` | column-oriented; stdlib only |
| `view.to_csv(path)` | — | writes CSV (uses pandas if available, falls back to `csv`) |
| `view.to_excel(path, sheet=None)` | — | requires `[results]` |
| `view.to_docx(path)` | — | requires `[results]` |
| `view.recompute(**params)` | new view | re-runs the underlying computation with different params |
| `view.params` | `dict` | the params this view was computed with |
| `repr(view)` | short summary | shape + first few rows |

### Views on a continuous result

```python
result.stats             # dict-like: r2, r2_adj, pvalue, n_kept, y_mean, y_std, cos_align, …
result.words             # WordsView    (top words per pole, ranked)
result.clusters          # ClustersIndex   .pos / .neg → SidedClustersView
result.snippets          # SnippetsView (all snippets, including those tied to clusters)
result.docs              # DocsView     (predicted vs observed, residuals, extreme/misdiagnosed)
result.predictions       # PredictionsView (y_hat, residuals; thin wrapper over docs)
result.components        # PLSResult only — loadings, scores
result.sweep             # PCAOLSResult only — PCA-K sweep table + plot helper
```

### Views on a group result

```python
gr.stats                 # G (n groups), n_kept, n_perm, correction, random_state, …
gr.omnibus               # dict: omnibus_T, omnibus_p
gr.pairs                 # PairsView, mapping-like (also has .df() for the pairwise table)
gr.pairs["A","B"]        # PairView — a single-contrast result. Same interface as ContinuousResult,
                         #   minus .docs / .predictions (no per-doc prediction in group mode).
gr.words                 # flat WordsView across all pairs (with `contrast` column)
gr.clusters.pos          # flat clusters across all pairs
gr.snippets              # flat snippets across all pairs
```

### Views on a lexicon result

```python
lex.stats                # var_type, n_docs, n_tokens
lex.suggestions          # SuggestionsView — per-token rows (token, freq, cov_bal, corr, p, direction, rank)
lex.summary              # dict | None  (present after evaluate_lexicon, absent after suggest_lexicon)
lex.tokens               # convenience: list[str] in rank order
```

---

## Domain objects

Iterating a view yields **frozen dataclasses**, not dicts. Fields are typed and IDE-autocomplete-friendly.

```python
@dataclass(frozen=True, slots=True)
class Word:
    side: str          # "pos" | "neg"
    rank: int          # 1, 2, 3, …
    word: str
    cos_beta: float
    contrast: str | None = None   # set on group / pair results

@dataclass(frozen=True, slots=True)
class Cluster:
    cluster_id: int
    side: str
    size: int
    coherence: float            # mean cos(word, centroid) inside cluster
    centroid_cos_beta: float    # cos(centroid, beta)
    contrast: str | None = None
    # joins (lazy):
    @property
    def words(self) -> "ClusterWordsView": ...
    @property
    def snippets(self) -> "SnippetsView": ...

@dataclass(frozen=True, slots=True)
class ClusterWord:
    cluster_id: int
    word: str
    cos_centroid: float
    cos_beta: float

@dataclass(frozen=True, slots=True)
class Snippet:
    snippet_id: int
    side: str
    profile_id: int
    post_id: int
    cosine: float
    seed: str
    start_token_idx: int
    end_token_idx: int
    start_sent_idx: int
    end_sent_idx: int
    text_surface: str
    text_lemmas: str
    cluster_id: int | None = None     # set when the snippet is anchored in a cluster
    contrast: str | None = None

@dataclass(frozen=True, slots=True)
class Doc:
    doc_id: int
    y_true: float
    y_hat: float
    residual: float
    cos_align: float

@dataclass(frozen=True, slots=True)
class Pair:
    contrast: str          # "A_vs_B"
    g1: str
    g2: str
    T: float
    p_raw: float
    p_corrected: float
    cohens_d: float
    n_g1: int
    n_g2: int
    contrast_norm: float

@dataclass(frozen=True, slots=True)
class Suggestion:
    rank: int
    token: str
    freq: int
    cov_all: float
    cov_bal: float
    corr: float
    pvalue: float
    direction: str         # "+" | "-" | "n/a" for categorical
```

---

## Data schema (canonical tables)

For power users and `.df()` consumers — these are the underlying tidy tables. **Joins are by id**, never by nested structure.

### Continuous result (`PLSResult`, `PCAOLSResult`)

| Table | Key | Columns |
|---|---|---|
| `stats` | — | backend, r2, r2_adj, pvalue, n_raw, n_kept, n_dropped, y_mean, y_std, cos_align, … |
| `words` | (side, rank) | side, rank, word, cos_beta |
| `clusters` | (cluster_id) | cluster_id, side, size, coherence, centroid_cos_beta |
| `cluster_words` | (cluster_id, word) | cluster_id, word, cos_centroid, cos_beta |
| `snippets` | (snippet_id) | snippet_id, side, profile_id, post_id, cosine, seed, start_token_idx, end_token_idx, start_sent_idx, end_sent_idx, text_surface, text_lemmas, **cluster_id (nullable)** |
| `docs` | (doc_id) | doc_id, y_true, y_hat, residual, cos_align |

### Group result (`GroupResult`)

Adds a `contrast` column to `words`, `clusters`, `cluster_words`, `snippets`. Replaces `stats` / `docs` with:

| Table | Key | Columns |
|---|---|---|
| `omnibus` | — | omnibus_T, omnibus_p, n_perm, correction, G |
| `pairs` | (contrast) | contrast, g1, g2, T, p_raw, p_corrected, cohens_d, n_g1, n_g2, contrast_norm |

### Lexicon result (`LexiconResult`)

| Table | Key | Columns |
|---|---|---|
| `stats` | — | var_type, n_docs, n_tokens |
| `suggestions` | (rank) | rank, token, freq, cov_all, cov_bal, corr, pvalue, direction |
| `summary` | — | docs_any, cov_all, q1, q4, corr_any, hits_mean, hits_median, types_mean, types_median, group_cov? *(only after `evaluate_lexicon`)* |

---

## Computing & re-computing

Views with parameters (clusters, snippets) are computed on first access with sensible defaults and **cached by parameter set**.

```python
# First access — computed with defaults
result.clusters.pos                                 # topn=100, k=auto, k_min=2, k_max=10

# Same params — cache hit
result.clusters.pos                                 # ← returns cached

# Different params — separate cache entry, recomputed
result.clusters.pos.recompute(topn=50, k=5)        # new view, also cached
result.clusters.pos                                 # ← still the original cached default

# Force recompute with same params
result.clusters.pos.recompute()
```

Snippet views similarly:

```python
result.snippets                                     # default top_per_side=200
result.snippets.recompute(top_per_side=500)        # separate cache entry
```

Cache management:

```python
result.clear_cache()                                # everything
result.clear_cache("clusters")                      # one view family
result.clear_cache("clusters", side="pos")          # one view, one set of params
```

> **What changed from v1:** in v1, calling `cluster_neighbors(topn=50)` after `cluster_neighbors(topn=100)` silently returned the topn=100 result. v2 keys the cache on params, so this can't happen.

---

## Reports

`result.report(...)` returns a `Report` object — a builder for multi-format rendering of the same content.

```python
r = result.report(
    top_words=10,
    clusters=50,
    snippets_per_cluster=3,
    extreme_docs=5,         # ignored on GroupResult / LexiconResult
    misdiagnosed=5,         # ignored on GroupResult / LexiconResult
)

print(r)                   # → r.text()
r.text()                   # plain terminal output
r.markdown()               # markdown source
r.html()                   # styled HTML (also _repr_html_ for Jupyter)
r.docx()                   # python-docx Document object
r.save("report.md")        # extension dispatch: .txt .md .html .docx
```

`GroupResult.report()` automatically expands sections per pairwise contrast — you don't write the loop.

`LexiconResult.report()` renders the suggestions table + summary block (matches v1 output but pluggable).

---

## Export — quick reference

| Want | Code |
|---|---|
| One table as DataFrame | `result.words.df()` |
| One table as CSV | `result.words.to_csv("w.csv")` |
| One table as Excel | `result.words.to_excel("w.xlsx")` |
| All tables as one Excel workbook (one sheet each) | `result.to_excel("everything.xlsx")` |
| All tables as a folder of CSVs | `result.to_csv_folder("out_dir/")` |
| Full report as DOCX | `result.report().save("report.docx")` |
| Full report as Markdown | `result.report().save("report.md")` |
| Pickle (round-trip with cache, no embeddings) | `result.save("r.pkl")`  /  `Result.load("r.pkl")` |

---

## Optional dependencies

Tabular and document export require the `[results]` extra:

```bash
pip install ssdiff[results]
```

This adds:

- `pandas` — for `.df()`, `.to_csv()` (when available), `.to_excel()`
- `openpyxl` — Excel write backend (used through pandas)
- `python-docx` — DOCX rendering

Without `[results]` you can still:

- Iterate views (yields domain objects)
- Call `.to_dict()` / `.to_records()`
- Call `.to_csv()` (falls back to stdlib `csv`)
- Call `report().text()` / `report().markdown()` / `report().html()`
- Save and load via `result.save()` / `Result.load()` (pickle)

Methods that need a missing dep raise `ImportError` with a clear hint:

```
ImportError: Cluster.df() requires pandas. Install with: pip install ssdiff[results]
```

---

## Save / load

```python
result.save("model_results.pkl")
loaded = PLSResult.load("model_results.pkl")        # or Result.load(...) for auto-dispatch
```

What's persisted:

- All canonical tables (`stats`, `words`, `clusters`, `cluster_words`, `snippets`, `docs`, …)
- Cached computed views (so `.clusters.pos` is instant after load)
- Backend metadata (n_components, cv_result, sweep_result, perm_null, …)

What's **not** persisted:

- The `Embeddings` object (recompute neighbors with `result.attach(embeddings=...)` if you need to re-cluster)
- The `Corpus` (re-attach to compute new snippets)

---

## Code organization

Internally the results layer lives in `ssdiff/results/`:

```
ssdiff/results/
  __init__.py         # public exports
  _base.py            # Result, View, domain dataclasses, schema
  _cache.py           # parameter-keyed cache
  _export.py          # to_csv / to_excel / to_docx / to_dict shims
  _report.py          # Report builder + text/md/html/docx renderers
  ssd.py              # ContinuousResult, PLSResult, PCAOLSResult
  group.py            # GroupResult, PairView
  lexicon.py          # LexiconResult
```

Notably, `LexiconResult` shares the same view / export / report machinery as the SSD results — if you understand one, you understand all.

---

## Migration from v1

| v1 | v2 |
|---|---|
| `result.summary()` | `print(result.report())` or `result.stats` |
| `result.top_words(20)` | `result.words[:20]` or `result.words.df().head(20)` |
| `result.cluster_neighbors("pos", topn=100)` | `result.clusters.pos.recompute(topn=100)` (or `result.clusters.pos` for defaults) |
| `result.snippets(corpus.pre_docs)` | `result.snippets` (auto-attached after `attach(corpus=…)`) |
| `result.cluster_snippets(...)` | `result.clusters.pos[i].snippets` (per-cluster) or `result.snippets.where(cluster_id__notnull=True)` |
| `result.report(top_words=10, clusters=50)` | `result.report(top_words=10, clusters=50).text()` |
| `gr.report()` (loops over pairs) | `gr.report()` (still loops; same output, multi-format) |
| `gr.cluster_neighbors("pos")` (flat) | `gr.clusters.pos` (flat, with `contrast` column) |
| `gr.filter_groups("A","B")` | `gr.pairs["A","B"]` (cheaper, just a view) |
| `LexiconResult.tokens` | `lex.tokens` (unchanged) or `lex.suggestions.df()` |
| `LexiconResult.report()` | `lex.report().text()` |

A v1.x → v2 migration script will be provided that rewrites the most common call patterns automatically.

---

## See also

- [`api_reference.md`](api_reference.md) — full ssdiff API (fitting, embeddings, corpus)
- [`architecture.md`](architecture.md) — fitting backends and statistical methods
- [`demo_new_api.py`](demo_new_api.py) — runnable continuous-results example
- [`demo_lexicon_api.py`](demo_lexicon_api.py) — runnable lexicon example
