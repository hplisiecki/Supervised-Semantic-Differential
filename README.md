# Supervised Semantic Differential (SSD)

**SSD** is a small, researcher-friendly Python package that lets you recover **interpretable semantic directions** (à la Osgood’s Semantic Differential) directly from open-ended text and relate them to **numeric outcomes** (e.g., scales, judgments). It builds per-essay concept vectors from **local contexts around seed words**, learns a **semantic gradient** that best predicts the outcome, and then provides multiple interpretability layers:

- **Nearest neighbors** of each pole (+β̂ / −β̂)
- **Clustering** of neighbors into themes
- **Text snippets**: top sentences whose local context aligns with each cluster centroid or the β̂ axis
- **Per-essay scores** (cosine alignments) to analyze or merge back into your data frame

The package focuses on **clarity, reproducibility, and explainability**, not production serving.

---

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Core Concepts](#core-concepts)
- [Preprocessing (spaCy)](#preprocessing-spacy)
- [Lexicon Utilities](#lexicon-utilities)
- [Fitting SSD](#fitting-ssd)
- [Neighbors & Clustering](#neighbors--clustering)
- [Interpreting with Snippets](#interpreting-with-snippets)
- [Per-Essay SSD Scores](#per-essay-ssd-scores)
- [API Summary](#api-summary)
- [Citing & License](#citing--license)

---

## Installation

```bash
pip install ssd-semdiff
```

Dependencies (installed automatically): `numpy`, `pandas`, `scikit-learn`, `gensim`, `spacy`, `tqdm`.

To use the Polish pipeline:
```bash
pip install ssd-semdiff
```
(For other languages, swap the model accordingly.)

---

## Quickstart

Below is an end-to-end minimal example using the Polish model and a “climate” dataset. 
Adjust paths and column names to your data.

```python
from ssd import (
    SSD, load_embeddings, normalize_kv,
    load_spacy, load_stopwords, preprocess_texts, build_docs_from_preprocessed,
    suggest_lexicon, token_presence_stats, coverage_by_lexicon,
)

import pandas as pd

MODEL_PATH = r"D:\resources\NLP\PL\nkjp+wiki-lemmas-all-300-cbow-hs.txt\nkjp+wiki-lemmas-all-300-cbow-hs.kv"
DATA_PATH  = r"D:\Github\BetweenWords\data\kalibra_klimat.csv"

# 1) Load and normalize embeddings (L2 + ABTT on word space)
kv = normalize_kv(load_embeddings(MODEL_PATH), l2=True, abtt_m=1)

# 2) Load your data
df = pd.read_csv(DATA_PATH)
text_raw_col = "klimat_open"
y_col        = "klimat_closed"

# 3) Preprocess (spaCy) — keep original sentences and lemmas linked
nlp = load_spacy("pl_core_news_lg")
stopwords = load_stopwords("pl")
texts_raw = df[text_raw_col].fillna("").astype(str).tolist()
pre_docs = preprocess_texts(texts_raw, nlp, stopwords)

# 4) Build lemma docs for modeling and filter to non-NaN y
docs = build_docs_from_preprocessed(pre_docs)       # list[list[str]]
y = pd.to_numeric(df[y_col], errors="coerce")
mask = ~y.isna()
docs = [docs[i] for i in range(len(docs)) if mask.iat[i]]
pre_docs = [pre_docs[i] for i in range(len(pre_docs)) if mask.iat[i]]
y = y[mask].to_numpy()

# 5) Define a lexicon (tokens must match your preprocessing)
lexicon = {"zmiana", "klimatyczny", "klimat"}

# 6) Choose PCA dimensionality based on sample size
n_kept = len(docs)
PCA_K = min(10, max(3, n_kept // 10))

# 7) Fit SSD
ssd = SSD(
    kv=kv,
    docs=docs,
    y=y,
    lexicon=lexicon,
    l2_normalize_docs=True,
    abtt_m_docs=0,
    N_PCA=PCA_K,
    use_unit_beta=True,
)

# 8) Inspect regression readout
print({
    "R2": ssd.r2,
    "F": ssd.f_stat,
    "p": ssd.f_pvalue,
    "beta_norm": ssd.beta_norm_stdCN,        # ||β|| in SD(y) per +1.0 cosine
    "delta_per_0.10_raw": ssd.delta_per_0p10_raw,
    "IQR_effect_raw": ssd.iqr_effect_raw,
    "corr_y_pred": ssd.y_corr_pred,
    "n_kept": ssd.n_kept,
})

# 9) Neighbors
pos_words = [w for (w, _sim) in ssd.nbrs(sign=+1, n=20)]
neg_words = [w for (w, _sim) in ssd.nbrs(sign=-1, n=20)]
print("Neighbors +β̂:", pos_words[:10])
print("Neighbors −β̂:", neg_words[:10])

# 10) Cluster themes (e.g., 4 clusters per pole)
df_pos_clusters, df_pos_members = ssd.cluster_neighbors_sign(side="pos", topn=100, k=4, verbose=True)
df_neg_clusters, df_neg_members = ssd.cluster_neighbors_sign(side="neg", topn=100, k=4, verbose=True)

# 11) Snippets for interpretation
snips = ssd.snippets_from_clusters(
    pre_docs=pre_docs, window_sentences=1, top_per_cluster=100, sif_a=1e-3
)
df_pos_snip = snips["pos"]
df_neg_snip = snips["neg"]

# 12) Optional: per-essay SSD scores (cosine of each essay’s context vector to β̂)
scores = ssd.ssd_scores(docs)
df.loc[mask, "ssd_cosine"] = scores
```
---

## Core Concepts

- **Seed lexicon**: a small set of tokens (lemmas) indicating the concept of interest (e.g., {klimat, klimatyczny, zmiana}).
- **Per-essay vector**: SIF-weighted average of context vectors around each seed occurrence (±3 tokens), then averaged across occurrences.
- **SSD fitting**: PCA on standardized doc vectors, OLS from components to standardized outcome 𝑦, then back-project to doc space to get β (the semantic gradient).
- **Interpretation**: nearest neighbors to +β̂/−β̂, clustering neighbors into themes, and showing original sentences whose local context aligns with centroids or β̂.

---

## Preprocessing (spaCy)

SSD uses spaCy to keep original sentences and lemmas aligned for later snippet extraction.

```python
from ssd import load_spacy, load_stopwords, preprocess_texts, build_docs_from_preprocessed

nlp = load_spacy("pl_core_news_lg")   # or another language model
stopwords = load_stopwords("pl")      # same stopword source across app & package

pre_docs = preprocess_texts(texts_raw, nlp, stopwords)
docs = build_docs_from_preprocessed(pre_docs)  # → list[list[str]] (lemmas without stopwords/punct)
```

Each PreprocessedDoc stores:

- **sents_surface**: list[str], original sentences
- **sents_lemmas**: list[list[str]]
- **doc_lemmas**: flattened lemmas (list[str])
- **token_to_sent**: index mapping lemma positions → sentence index

---
## Lexicon Utilities

These helpers make lexicon selection transparent and data-driven (you can also hand-pick tokens).

### `suggest_lexicon(...)`

Rank tokens by balanced coverage with a mild penalty for strong correlation with
- `cov_all`: fraction of essays containing the token (0/1 presence)
- `cov_bal`: average presence across 𝑛 quantile bins of 𝑦 (default: 4 bins)
- `corr`: Pearson correlation between 0/1 presence and standardized 𝑦
- `rank = cov_bal * (1 - min(1, |corr|/corr_cap))` (default `corr_cap=0.30`)

Accepts a DataFrame (`text_col`, `score_col`) or a `(texts, y)` tuple where texts can be raw strings or token lists.

```python
from ssd import suggest_lexicon

# Using a DataFrame
cands_df = suggest_lexicon(df, text_col="stemmed", score_col="klimat_closed", top_k=150)

# Or using a tuple (texts, y)
texts = [" ".join(doc) for doc in docs]
cands_df2 = suggest_lexicon((texts, y), top_k=150)
```
### `token_presence_stats(...)`

Per-token coverage & correlation diagnostics:
```python
from ssd import token_presence_stats
stats = token_presence_stats((texts, y), token="klimat", n_bins=4, verbose=True)
print(stats)  # dict: token, docs, cov_all, cov_bal, corr, rank
```

### `coverage_by_lexicon(...)`

Summary for your chosen lexicon:
- `summary` : `docs_any`, `cov_all`, `q1`. `q4`, `corr_any`
  - `q1` / `q4`: coverage within the lowest/highest 𝑦 bins (by quantile)
- `per_token_df`: stats for each token

```python
from ssd import coverage_by_lexicon

summary, per_tok = coverage_by_lexicon(
    (texts, y),
    lexicon={"klimat", "klimatyczny", "zmiana"},
    n_bins=4,
    verbose=True
)
```
---
## Fitting SSD

Instantiate `SSD` with your normalized embeddings, tokenized documents, numeric outcome, and lexicon:
```python
from ssd import SSD, load_embeddings, normalize_kv

kv = normalize_kv(load_embeddings(MODEL_PATH), l2=True, abtt_m=1)

PCA_K = min(10, max(3, len(docs)//10))
ssd = SSD(
    kv=kv,
    docs=docs,
    y=y,
    lexicon={"klimat", "klimatyczny", "zmiana"},
    l2_normalize_docs=True,
    abtt_m_docs=0,     # no ABTT on doc matrix in the main pipeline
    N_PCA=PCA_K,
    use_unit_beta=True # unit β̂ for neighbors/interpretation
)

print(ssd.r2, ssd.f_stat, ssd.f_pvalue)
```
Key outputs attached to the instance:
- `beta` / `beta_unit` — semantic gradient (doc space)
- `r2`, `f_stat`, `f_pvalue`
- `beta_norm_stdCN` — ||β|| in SD(y) per +1.0 cosine
- `delta_per_0p10_raw` — change in raw 𝑦 per +0.10 cosine
- `iqr_effect_raw` — IQR(of cosine)*slope in raw 𝑦
- `y_corr_pred` — correlation of standardized 𝑦 with predicted values

---
## Neighbors & Clustering

### Nearest neighbors
```python
# Top neighbors of +β̂
ssd.nbrs(sign=+1, n=20)  # → list of (word, cosine), filtered for readability

# Top neighbors of −β̂
ssd.nbrs(sign=-1, n=20)
```

### Clustering neighbors into themes
Use `cluster_neighbors_sign` to group the top N neighbors of +β̂/−β̂ into k clusters (k-means; Euclidean on unit vectors ≈ cosine):

```python
# POSITIVE pole (+β̂)
df_pos_clusters, df_pos_members = ssd.cluster_neighbors_sign(
    side="pos",        # or "neg"
    topn=100,
    k=4,               # or let the function choose via silhouette: k=None, k_min=2, k_max=10
    restrict_vocab=50000,
    random_state=13,
    min_cluster_size=2,
    top_words=10,      # top preview words in the cluster summary
    verbose=True,      # pretty console preview
)

# NEGATIVE pole (−β̂)
df_neg_clusters, df_neg_members = ssd.cluster_neighbors_sign(side="neg", topn=100, k=4, verbose=True)
```

Returns
- df_*_clusters (one row per cluster):
- side, cluster_rank, size, centroid_cos_beta, coherence, top_words
- df_*_members (one row per word):
  side, cluster_rank, word, cos_to_centroid, cos_to_beta

The raw clusters (with all per-word cosines and internal ids) are kept internally as:
- ssd.pos_clusters_raw
- ssd.neg_clusters_raw

---
## Interpreting with Snippets
After clustering, SSD lets you **link the abstract directions in embedding space back to actual language** by inspecting **text snippets**.  
 The script:
1. Locates each **occurrence of a seed word** (from your lexicon) in the corpus.  
2. Extracts a **small window of surrounding context** (±3 tokens).  
3. Represents that window as a **SIF-weighted context vector** in the same embedding space as β̂ and the cluster centroids.  
4. Computes the **cosine similarity** between each such local context vector and  
   - a **cluster centroid** (to find passages representative of that theme), or  
   - the overall **semantic gradient β̂** (to find passages aligned with the global direction).

### Snippets by cluster centroids

```python
snips = ssd.snippets_from_clusters(
    pre_docs=pre_docs,    # from preprocess_texts(...)
    window_sentences=1,   # [sent-1, sent, sent+1]
    seeds=ssd.lexicon,    # seed lemmas to anchor context windows
    sif_a=1e-3,
    top_per_cluster=100,  # keep best K per cluster
)

df_pos_snip = snips["pos"]  # columns: centroid_label, doc_id, cosine, seed, sentence_before, sentence_anchor, sentence_after, window_text_surface, ...
df_neg_snip = snips["neg"]
```
Each returned row represents a seed occurrence window, not a whole essay.  
The `cosine` column is the similarity between the context vector (built around that seed occurrence) and the cluster centroid.  
Surface text (`sentence_before`, `sentence_anchor`, `sentence_after`) lets you read the passage in context.

### Snippets along β̂
You can also extract windows that best illustrate the main semantic direction (rather than specific clusters):
```python
beta_snips = ssd.snippets_along_beta(
    pre_docs=pre_docs,
    window_sentences=1,
    seeds=ssd.lexicon,
    sif_a=1e-3,
    top_per_side=200,
)
df_beta_pos = beta_snips["beta_pos"]
df_beta_neg = beta_snips["beta_neg"]
```
Here, the cosine is taken between each seed-centered context vector and β̂ (the main semantic gradient).
Sorting by this cosine reveals which local language usages most strongly express the positive or negative pole of your concept.

---
## Per-Essay SSD Scores

The **SSD score** for each essay quantifies **how closely the text’s meaning aligns with the main semantic direction (β̂)** discovered by the model.  
These scores can be used for individual-difference analyses, correlations with psychological scales, or visualization of semantic alignment across groups.

Internally, each essay is represented by a **SIF-weighted average of local context vectors** (around the lexicon seeds).  
The SSD score is then computed as the **cosine similarity between that essay’s vector and β̂**.  
In addition, the model’s regression weights allow you to compute the **predicted outcome** for each essay — both in standardized units and in the original scale of your dependent variable.


### How scores are computed

For each document \(i\):
- (x_i) — document vector (normalized if `l2_normalize_docs=True`)
- (β̂) — unit semantic gradient in embedding space  
- `cos[i] = cos(x_i, β̂)` → **semantic alignment score**  
- `yhat_std[i] = x_i · β` → predicted standardized outcome  
- `yhat_raw[i] = mean(y) + std(y) * yhat_std[i]` → prediction in original units  

These are available for **all documents**, with NaNs for those that did not contain any lexicon occurrences (i.e., were dropped before fitting).

```python
scores = ssd.ssd_scores(include_all=True, return_df=True)

print(scores.head())
```

Returned columns:
- `doc_index`	Original document index (0-based)
- `kept`	Whether the essay had valid seed contexts (True/False)
- `cos`	Cosine alignment of essay vector to β̂
- `yhat_std`	Predicted outcome (standardized units)
- `yhat_raw`	Predicted outcome (original scale of your dependent variable)
- `y_true_std`	True standardized outcome (NaN for dropped docs)
- `y_true_raw`	True raw outcome (NaN for dropped docs)

---
## API Summary
The `ssd` top-level package re-exports the main objects so you can write:

```python
from ssd import (
  SSD,                       # the analysis class (fit, neighbors, clustering, snippets, scores)
  load_embeddings, normalize_kv,
  load_spacy, load_stopwords, preprocess_texts, build_docs_from_preprocessed,
  suggest_lexicon, token_presence_stats, coverage_by_lexicon,
)
```

### `SSD` (class)

- `__init__(kv, docs, y, lexicon, *, l2_normalize_docs=True,  N_PCA=20, use_unit_beta=True)`
- Attributes after fit: `beta`, `beta_unit`, `r2`, `f_stat`, `f_pvalue`, `beta_norm_stdCN`,  
`delta_per_0p10_raw`, `iqr_effect_raw`, `y_corr_pred`, `n_kept`, etc.
- Methods:
  - nbrs(sign=+1, n=20) → list[(word, cosine)]
  - cluster_neighbors_sign(side="pos", topn=100, k=None, k_min=2, k_max=10, restrict_vocab=50000, random_state=13, min_cluster_size=2, top_words=10, verbose=False) → (df_clusters, df_members) and stores raw clusters in pos_clusters_raw/neg_clusters_raw 
  - snippets_from_clusters(pre_docs, window_sentences=1, seeds=None, sif_a=1e-3, top_per_cluster=100) → dict with "pos"/"neg" DataFrames 
  - snippets_along_beta(pre_docs, window_sentences=1, seeds=None, sif_a=1e-3, top_per_side=200) → dict with "beta_pos"/"beta_neg" DataFrames 
  - ssd_scores(docs) → numpy array of per-essay cosines

### Embeddings
- load_embeddings(path) → gensim.models.KeyedVectors
- normalize_kv(kv, l2=True, abtt_m=0) → new KeyedVectors with L2 + optional ABTT (“all-but-the-top”, top-m PCs removed)
- 
### Preprocessing
- load_spacy(model_name="pl_core_news_lg") → spaCy nlp
- load_stopwords(lang="pl") → list of stopwords (remote Polish list with sensible fallback)
- preprocess_texts(texts, nlp, stopwords) → list of PreprocessedDoc
- build_docs_from_preprocessed(pre_docs) → list[list[str]] (lemmas for modeling)

### Lexicon
- suggest_lexicon(df_or_tuple, text_col=None, score_col=None, top_k=150, min_docs=5, n_bins=4, corr_cap=0.30) → DataFrame
- token_presence_stats(df_or_tuple, token, n_bins=4, corr_cap=0.30, verbose=False) → dict
- coverage_by_lexicon(df_or_tuple, lexicon, n_bins=4, verbose=False) → (summary, per_token_df)

--- 
## Citing & License

- License: MIT (see LICENSE).
- If you use SSD in published work, please cite the package (and the classic Semantic Differential literature that motivated the method). 
- A suggested citation:

---
## Questions / Contributions
- File issues and feature requests on the repo’s Issues page.
- Pull requests welcome — especially for:
  - Robustness diagnostics and visualization helpers
  - Documentation improvements

Contact: hplisiecki@gmail.com