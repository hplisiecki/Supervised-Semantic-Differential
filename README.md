# Supervised Semantic Differential (SSD)

[![Tests](https://github.com/hplisiecki/Supervised-Semantic-Differential/workflows/Tests/badge.svg)](https://github.com/hplisiecki/Supervised-Semantic-Differential/actions)
[![PyPI](https://img.shields.io/pypi/v/ssdiff)](https://pypi.org/project/ssdiff/)
[![Python](https://img.shields.io/pypi/pyversions/ssdiff)](https://pypi.org/project/ssdiff/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://img.shields.io/badge/DOI-10.31234%2Fosf.io%2Fgvrsb__v1-blue)](https://doi.org/10.31234/osf.io/gvrsb_v1)

**SSD** lets you recover **interpretable semantic directions** related to specific concepts directly from open-ended text and relate them to **numeric outcomes**
(e.g., psychometric scales, judgments) or **categorical groups** (e.g., clinical diagnosis, experimental condition). It builds per-document concept vectors from **local contexts around seed words**,
learns a **semantic gradient (beta)** that best predicts the outcome, and then provides multiple interpretability layers:

- **Nearest neighbors** of each pole (+beta / -beta)
- **Clustering** of neighbors into themes
- **Text snippets**: top sentences whose local contexts align with each cluster centroid or the beta axis
- **Per-document scores** (cosine alignments) for further analysis
- **Cross-group comparisons** with permutation inference

The method has been presented in the following preprint:
https://doi.org/10.31234/osf.io/gvrsb_v3

> **No-code option:** a GUI desktop application for SSD is available at [hplisiecki/SSD_APP](https://github.com/hplisiecki/SSD_APP). It wraps this package into a point-and-click interface with a guided three-stage workflow, interactive lexicon builder, and APA-formatted export — pre-built binaries for Windows, Linux, and macOS are available with no Python installation required.

---


## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Core Concepts](#core-concepts)
- [Word Embeddings](#word-embeddings)
- [Texts and Tokenization](#texts-and-tokenization)
- [Lexicon Utilities](#lexicon-utilities)
- [Fitting SSD](#fitting-ssd)
  - [PCA regression](#pca-regression)
  - [PLS](#pls)
  - [Several outcomes (PLS3)](#several-outcomes-pls3)
  - [Cross-Group Comparison](#cross-group-comparison)
  - [Contrast between two gradients](#contrast-between-two-gradients)
  - [Inspecting results](#inspecting-results)
- [Neighbors & Clustering](#neighbors--clustering)
- [Interpreting with Snippets](#interpreting-with-snippets)
- [Per-Document SSD Scores](#per-document-ssd-scores)
- [API Summary](#api-summary)
- [Citing & License](#citing--license)

---

## Installation

```bash
pip install ssdiff
```

**Python**: 3.11 – 3.15.

Core dependencies (installed automatically): `numpy`, `plskit`, `glmm`.

Optional extras:
- `ssdiff[results]` — pandas / openpyxl / python-docx / matplotlib for `to_df()`, `.xlsx`/`.docx` export, and `plot_sweep()`.
- `ssdiff[gensim]` — only needed to *save* embeddings in `.kv` format.

> Loading `.kv` files works without gensim (handled by an internal unpickler shim).

---

## Quickstart

Below is an end-to-end minimal example for English text. For another language, or your own embedding file, see [Word Embeddings](#word-embeddings) and [Texts and Tokenization](#texts-and-tokenization).

```python
from ssdiff import Embeddings, SSD

# 1) Load embeddings: the default English wordpack, downloaded once (380 MB)
emb = Embeddings.load()

# 2) Load your data: one row per document
df = ...                               # DataFrame with a text column and a numeric outcome column

# 3) Build SSD on one text, with a lexicon (the forms as they occur in the texts)
ssd = SSD(df["text"], embeddings=emb, lexicon=["happy", "sad", "joy", "anger"])

# 4) Fit. The outcome is the first argument of each fit.
result = ssd.fit_pls("score", data=df)      # or ssd.fit_regression("score", data=df)

# 5) Inspect
print(result.stats)            # r², p-value, n_kept, β-norm, IQR effect
print(result.words.pos)        # top β-positive neighbours
print(result.words.neg)        # top β-negative neighbours
print(result.clusters.pos(topn=100))   # cluster the 100 nearest +β neighbours
result.report().save("report.md")
```

The outcome can also be given as values: `ssd.fit_pls(df["score"])` or `ssd.fit_pls(scores_array)`.

With every part at its default, a regression is one line: `ssdiff.fit_regression("score ~ ssd(text)", df)` builds the text from the column `text` with the default wordpack and full-document vectors (see [Several texts](#several-texts)).

Every result attribute is a view: print it, slice with `(n)`, dispatch
to one side with `.pos` / `.neg`, or export with `.to_df()` / `.save(...)`.

---

## Core Concepts

- **Seed lexicon**: a small set of tokens (word forms) indicating the concept of interest (e.g., {climate, warming, change}).
- **One text per `SSD`**: an `SSD` holds the tokens of one text column and builds its document vectors once. It has no outcome. Every fit names its own outcome, so one `SSD` serves many outcomes.
- **Per-document vector**: SIF-weighted average of context vectors around each seed occurrence (+-3 tokens), then averaged across occurrences.
- **SSD fitting**: Learn a semantic gradient (beta) that best predicts the outcome. Two backends are available:
  - **PLS**: Partial Least Squares regression directly in embedding space.
  - **PCA regression**: PCA dimensionality reduction followed by a regression on the first K components (matches original SSD paper). The text is a term in a formula, `ssd()`, so it can sit next to covariates, random effects and interactions.
- **Interpretation**: nearest neighbors to +beta/-beta, clustering neighbors into themes, and showing original sentences whose local context aligns with centroids or beta.

---

## Word Embeddings

For English, `Embeddings.load()` downloads a ready wordpack. For another language, load your own embedding file. The full guide is [`docs/embeddings.md`](docs/embeddings.md).

### Wordpacks

A wordpack carries its own tokenizer and stopword list, so `SSD` can take plain strings with it. Load one by tag or by full name; it is downloaded once into a cache directory and loaded from there. With no argument, `Embeddings.load()` loads the tag `default`, which is the same pack as `en`:

```python
from ssdiff import Embeddings, SSD

emb = Embeddings.load()                                # the default: the same pack as Embeddings.load("en")
# emb = Embeddings.load("en-wiki")                     # tag
# emb = Embeddings.load("en-glove2024-dolma-1")        # full name: always this file
# emb = Embeddings.load("path/to/file.wordpack")       # a file you built yourself
ssd = SSD(df["text"], embeddings=emb)                  # strings, tokenized by the pack
res = ssd.fit_pls("score", data=df)
pred = res.predict(new_texts)
```

| tag | full name | use it for |
|---|---|---|
| `en`, `en-dolma`, `default` | `en-glove2024-dolma-1` | informal or recent text: comments, social media, open survey answers, forums |
| `en-wiki` | `en-glove2024-wikigiga-1` | formal text: news, essays, reports, older press |

A full name always means the same file, and it is what `report()` and `ssd.tokenizer` record, so use it in scripts whose results must be reproducible. A tag means the same pack for a given ssdiff version, and a later version can move it to a newer pack. Loading by tag prints one line that names the pack.

Packs are kept in `~/.wordpacks` (`C:\Users\<name>\.wordpacks` on Windows); set `WORDPACK_CACHE` to use another directory. Nothing removes them: delete files there to free space.

A pack loads with ABTT (m = 1) already applied and keeps its 50,000 most frequent words in RAM; other words are read from the file when needed, so keep the file in place while you use `emb`.

`emb.normalize(abtt=m)` sets any m from 0 to 10, up or down.

English packs are keyed by surface form, not by lemma. A seed lexicon lists the forms that occur in the texts: `dog` and `dogs`, not the lemma alone. ssdiff does no lemma expansion.

### Your own embedding file

For another language, or your own model, load an embedding file in one of these formats:

| Format | Extension | Notes |
|--------|-----------|-------|
| Wordpack | `.wordpack` | Tokenizer, stopwords and ABTT included; takes strings |
| gensim KeyedVectors | `.kv` | Loads without gensim via internal shim |
| word2vec binary | `.bin` | Standard binary format |
| Text | `.txt`, `.vec` | One word per line + floats |
| Compressed | `.txt.gz`, `.vec.gz`, `.bin.gz` | Gzip-compressed versions of the above |

Embeddings that are not a wordpack take token lists, not strings (see [Texts and Tokenization](#texts-and-tokenization)). `SSD` needs L2-normalized rows. To capture semantic information without frequency-based artifacts, also apply All-But-The-Top (ABTT):

```python
from ssdiff import Embeddings

emb = Embeddings.load("path/to/model.bin", verbose=True)
emb.normalize(l2=True, abtt=1)   # L2 + ABTT (remove top-1 PC)
```

Calling `normalize()` with no arguments applies both L2 and ABTT (m=1) by default.
Processing state is tracked — calling it again safely skips already-applied steps.

Look for pre-trained static word embeddings in your language with good vocabulary coverage for your domain. GloVe and word2vec trained on large general corpora are a reliable starting point.

For Polish, the `nkjp+wiki-lemmas-all-300-cbow-hs.txt.gz` (no. 25) from the [Polish Word2Vec model list](https://dsmodels.nlp.ipipan.waw.pl) was found to work well. Its keys are lemmas, so lemmatize the texts first and pass the lemmas as token lists.

---

## Texts and Tokenization

`SSD(texts, embeddings=emb, lexicon=lexicon)` takes one text: a sequence with one entry per document. A pandas Series works; its name becomes the text's name (`ssd.name`). Without `embeddings`, it loads `Embeddings.load()`, the default wordpack.

```python
ssd = SSD(df["text"], embeddings=emb, lexicon=lexicon)
```

**Strings.** With a wordpack, each entry is a string. The pack's tokenizer and stopword list turn it into tokens. A wordpack exists for English, so strings work for English text. An empty, `None` or NaN entry is an empty document.

**Token lists.** With any other embedding, each entry is a list of tokens. They are used as given, and each token is matched to the embedding's keys exactly. Words that are not keys are skipped. Use the form the embedding is keyed by: lowercase tokens for a lowercase embedding, lemmas for a lemmatized embedding.

```python
tokens = [["happy", "day", "sunshine"], ["sad", "rain", "cold"], ...]
ssd = SSD(tokens, embeddings=emb, lexicon=lexicon)
```

A text of token lists has no raw text, so it has no snippets, and `result.docs.id(i)` shows no original text.

Lemmatize or tokenize with any tool you like and pass the tokens. `examples/custom_embedding.py` does this with spaCy, which is optional and is not an ssdiff dependency.

`SSD` options:

| Argument | Default | Description |
|---|---|---|
| `embeddings` | `None` | The word embeddings. `None` loads `Embeddings.load()`, the default wordpack. |
| `lexicon` | `None` | Seed words for seed-context vectors. |
| `window` | `3` | Context window (tokens) around each seed. |
| `sif_a` | `1e-3` | SIF smoothing. `None` turns SIF off (plain mean). |
| `use_full_doc` | `None` | `True`: the whole document; `False`: seed contexts. `None` means `True` without a lexicon and `False` with one. |

`ssd.with_lexicon(other_lexicon)` returns a new `SSD` on the same tokens with another lexicon (and optionally other `window`, `sif_a`, `use_full_doc`). The texts are not tokenized again.

`ssd.docs` holds the tokens of each document, `ssd.tokenizer` is `"wordpack:<name>-<version>"` for strings and `None` for token lists, `len(ssd)` is the document count and `ssd.n_kept` counts the documents that have a vector.

---

## Lexicon Utilities

These helpers make lexicon selection transparent and data-driven (you can also hand-pick tokens). They are **methods on `SSD`** — they score the tokens of the text, so what they score is exactly what the fits consume. They need no lexicon on the `SSD`.

### `ssd.suggest_lexicon(outcome, data=None, ...)`

Rank tokens by balanced coverage with a mild penalty for strong association with the outcome. Returns a `LexiconResult` view (printable, exportable, sliceable). `outcome` is values or a column name of `data`.

```python
ssd = SSD(df["text"], embeddings=emb)
result = ssd.suggest_lexicon("score", data=df, top_k=30)
print(result)                      # tabular view
ssd = ssd.with_lexicon(result.tokens)
```

With `seeds`, it lists the text's tokens closest to the seeds in the embedding instead, so you can see which forms and related words occur in your texts and choose the ones that go into the lexicon. `outcome` may then be `None`:

```python
emb = Embeddings.load("en")
ssd = SSD(df["text"], embeddings=emb)

lex = ssd.suggest_lexicon("score", data=df, seeds=["immigrant", "muslim"])
lex.suggestions        # one table for all seeds, highest cosine first
lex.suggestions[:]     # every row (printing shows the first 20)
lex.tokens             # the listed tokens, in table order

ssd = ssd.with_lexicon(["immigrant", "immigrants", "muslim", "muslims"])
```

Each seed lists itself, its `top_k` closest tokens by cosine, and the seed with an English ending (`-s`, `-es`, `-d`, `-ed`, `-ing`, `-ies`, `-ied`). A token listed for several seeds has one row, with the seed it is closest to and the cosine to that seed. Only tokens in at least `min_docs` documents and in the embedding are listed. A lexicon matches exact tokens: `immigrant` does not count `immigrants`, so plural forms must be listed.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `outcome` | `array-like \| str \| None` | `None` | Outcome values (numeric or categorical), or a column name of `data`. `None` only with `seeds`. |
| `data` | `DataFrame` | `None` | Holds the column `outcome` names |
| `seeds` | `sequence[str]` | `None` | Seed words; each must be in the embeddings |
| `top_k` | `int` | `30` | Maximum number of words to return; with `seeds`, neighbours listed per seed |
| `min_docs` | `int` | `5` | Minimum document frequency |
| `n_bins` | `int` | `4` | Quantile bins for balanced coverage |
| `corr_cap` | `float` | `0.30` | Penalty threshold for outcome association |
| `var_type` | `str` | `"continuous"` | `"continuous"` or `"categorical"` |

### `ssd.evaluate_lexicon(outcome, lexicon, data=None, ...)`

Score an existing lexicon against an outcome. Returns a `LexiconResult` bundling per-token diagnostics (`.suggestions`) and an aggregate coverage summary (`.summary`) — both saveable, with `.report()` producing a narrative markdown overview. `ssd.token_stats(...)` and `ssd.coverage_summary(...)` return the two parts alone, as a list of dicts and a dict.

```python
lex = ssd.evaluate_lexicon("score", lexicon=["happy", "sad", "anger"], data=df)

print(lex)                              # tabular view
lex.suggestions.save("tokens.csv")      # per-token rows
lex.summary.save("coverage.csv")        # aggregate stats
lex.report().save("lexicon.md")         # narrative overview
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `outcome` | `array-like \| str \| None` | — | Outcome values, or a column name of `data`. `None` returns only the statistics that need no outcome. |
| `lexicon` | `iterable[str]` | — | Tokens to evaluate (matched against the text's tokens) |
| `data` | `DataFrame` | `None` | Holds the column `outcome` names |
| `n_bins` | `int` | `4` | Quantile bins for balanced coverage |
| `corr_cap` | `float` | `0.30` | Penalty threshold for outcome association |
| `var_type` | `str` | `"continuous"` | `"continuous"` or `"categorical"` |

`.suggestions` columns: `token, freq, cov_all, cov_bal, corr, pvalue, direction, rank`.
`.summary` fields: `docs_any, cov_all, q1, q4, corr_any, hits_mean, hits_median, types_mean, types_median` (plus `group_cov` for categorical `outcome`).

---

## Fitting SSD

Create an `SSD` with one text, and optionally embeddings and a lexicon.
The constructor tokenizes the text and does **not** fit a model — call `fit_pls()`, `fit_multipls()`, `fit_regression()`, or `fit_groups()` explicitly. The outcome is the first argument of each fit: values (array, list, Series), or the name of a column of `data`. `fit_regression` and `fit_pls` also take a formula (`"score ~ age + gender"`) with `data=`. `fit_multipls` takes several numeric outcomes (a DataFrame, a 2-D array, or a list of column names). `data` has one row per document, in the order of the texts.

```python
from ssdiff import Embeddings, SSD

emb = Embeddings.load()
ssd = SSD(
    df["text"],
    embeddings=emb,     # None (the default) loads Embeddings.load()
    lexicon=["word1", "word2", "word3"],
    window=3,           # context window +/-3 tokens around lexicon hits
    sif_a=1e-3,         # SIF weighting parameter; None turns SIF off
    use_full_doc=None,  # None = seed contexts with a lexicon, full documents without
)
```

A document drops out of a fit when its outcome is missing or when it has no vector (no lexicon word, or none of its words is in the embeddings). `result.stats` reports `n_kept` and `n_dropped`.

### PCA regression

Original SSD algorithm from the paper: PCA of the document vectors, then a regression of the outcome on the first K components. The final model is fit by [`glmm`](https://github.com/pawlenartowicz/glmm).

```python
result = ssd.fit_regression(
    "score",              # values, a column name, or a formula such as "score ~ age + ssd()"
    data=df,
    family="gaussian",    # any glmm family, e.g. "binomial" for a 0/1 outcome
    fixed_k=None,         # None = auto-select via interpretability+stability sweep
    k_min=2,
    k_max=120,
    k_step=2,
    ssd_gradients=1,      # N > 1 also returns N rotated gradients from this model
    verbose=False,
    pca_sweep_args=None,  # extra sweep settings; ignored when fixed_k is set
)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `outcome` | `array-like \| str` | — | Outcome values, a column name of `data`, or a formula whose left side is a column of `data` |
| `data` | `DataFrame \| dict` | `None` | One row per document: the outcome column, covariates and grouping factors |
| `family` | `str` | `"gaussian"` | glmm family of the final model |
| `link` | `str \| None` | `None` | Link function; `None` = the family's default |
| `fixed_k` | `int \| None` | `None` | Fixed PCA components. `None` = auto-select via sweep |
| `k_min` | `int` | `2` | Minimum PCA-K for sweep |
| `k_max` | `int` | `120` | Maximum PCA-K for sweep |
| `k_step` | `int` | `2` | Step size |
| `ssd_gradients` | `int` | `1` | Number of gradients returned, in `[1, K]`. At 1 the result is the standard fit alone. See [Several gradients](#several-gradients). |
| `verbose` | `bool` | `False` | Print progress |
| `pca_sweep_args` | `dict \| None` | `None` | Extra keyword arguments for the PCA-K sweep (e.g. `cluster_topn`, `auck_radius`, `save_tables`, `out_dir`). Ignored when `fixed_k` is set. A key that `fit_regression` sets itself (`Xs`, `X_scale`, `x`, `ys`, `embeddings`, `pca_k_values`, `verbose`) raises `ValueError`. |

`fit_ols` is a deprecated alias of `fit_regression` and gives a `FutureWarning`.

`ssd_gradients=N` with N > 1 also returns N gradients, `result["dim-1"]` … `result["dim-N"]`, built from this regression's own model: the K components are split into N consecutive blocks, each block's part of the model's beta is one gradient, and the N gradients are rotated by words as in [Several gradients](#several-gradients). The standard fit does not change. It needs the text as one plain `ssd()` term; with covariates, another family or random effects, the gradients' `score_corr` and `cov_share` are descriptive.

#### Formulas: covariates, random effects and the text term

In a formula the text is the term `ssd()`. ssdiff replaces it with the K component scores before `glmm` fits the model. A formula without `ssd()` gets `+ ssd()` added at the end, so these two calls are the same:

```python
ssd.fit_regression("rating ~ age + gender + (1 | ID)", data=df)
ssd.fit_regression("rating ~ age + gender + (1 | ID) + ssd()", data=df)
```

K is still chosen by the sweep below. Random effects such as `(1 | ID)` handle several texts per person. `ssd()` cannot go inside a random-effect term.

```python
result.stats.pvalue      # the text terms together: formula without them vs full formula
result.covariates        # covariate coefficients, with the text held fixed
result.summary()         # compact glmm summary, without the component rows, plus the text's test
result.model.summary()   # full glmm summary, component rows included
```

The p-value comes from an F-test for an OLS model (Gaussian, no random effects) and for Gamma and inverse-Gaussian GLMs, and from a likelihood-ratio test otherwise (`result.test.name` is `"f_test"` or `"lrt"`). Rows of `data` follow the documents the fit keeps. A column the formula uses must have no missing values in those documents. With a formula, `r2` is the whole model's R², covariates included; it is NaN for mixed models and GLMs. `predict()` is not available after a fit with a formula or a non-Gaussian family.

#### Interactions with the text

The text can interact with covariates. `glmm` expands the interaction, and ssdiff returns one gradient for each term that contains the text:

```python
res = ssd.fit_regression("score ~ age + sex*ssd()", data=df)

res.keys()                  # ['ssd(essay)', 'sexM:ssd(essay)']
res["ssd(essay)"].words     # the gradient at the reference level of sex
res["sexM:ssd(essay)"].words   # how the gradient for sexM differs from it
res.test                    # all-text test, then one Wald test per term
res.summary()
```

The key of a term is the formula term with `ssd()` shown as `ssd(<text name>)` (`ssd()` when the text has no name). Each leaf is a regression result with `.beta`, `.gradient`, `.words`, `.clusters`, `.snippets`. A leaf's `beta` is the gradient of its term alone: the gradient of one level is the sum of the `beta` of `res["ssd(essay)"]` and that level's term.

`res.test` has one row per test. The row with an empty label tests all the text terms together (the formula without them against the full formula). Each other row is one Wald test of one term, which tests that term's coefficients together, so lower-order terms such as `ssd(essay)` have a row too. A formula with a single text term gives a plain regression result, not a container.

`res.report()` is not defined on the container: use `res.summary()`, `res.save_page(...)`, or `res["sexM:ssd(essay)"].report()` for one term.

#### Several texts

Several texts can meet in one model. Name each text in the formula as `ssd(name)` and call the module-level function `ssdiff.fit_regression`. A name that is a column of `data` is built from that column as `SSD(data[name])`, with every default; all texts built this way in one call share one `Embeddings.load()`. A text that needs other settings is built with `SSD` and passed by keyword, which wins over a column of the same name:

```python
import ssdiff

ssdiff.fit_regression("score ~ age + ssd(essay) + ssd(diary)", df)   # both texts from columns

essay = SSD(df["essay"], embeddings=emb, lexicon=lex)
diary = SSD(df["diary"], embeddings=emb, lexicon=lex)

both = ssdiff.fit_regression(
    "score ~ age + ssd(essay) + sex*ssd(diary)", df,
    essay=essay, diary=diary,
)

both.keys()               # ['essay', 'diary']
both["essay"].words       # the essay text's gradient
both["diary"].keys()      # ['ssd(diary)', 'sexM:ssd(diary)']
both.test                 # all texts, then one row per text, then one Wald row per term
```

A name in the formula that is neither a keyword nor a column raises `ValueError`, and so does a keyword the formula does not name. The texts and `data` have one row per document in the same order. A row stays when it has an outcome and a vector in every text. Each text gets its own PCA and its own K, and `fixed_k`, `k_min`, `k_max`, `k_step` and `pca_sweep_args` apply to every text. Two texts cannot meet in one interaction term. Use `ssd(name)` in the formula; a bare `ssd()` is for the method `SSD.fit_regression`.

#### Automatic K selection (PCA sweep)

Selecting the number of PCA components (`fixed_k = K`) can be a researcher degree of freedom. Pass `fixed_k=None` (the default) to run an automatic **PCA sweep** that evaluates a range of K values and selects the most robust solution. The sweep fits the outcome on the components without interactions.

For each candidate PCA dimensionality K, the sweep fits SSD and tracks:

1. **Interpretability quality** — based on clustering the nearest neighbors at each pole of the semantic gradient and computing aggregate cluster coherence and alignment with beta.

2. **Stability of the semantic gradient** — measured as the cosine change between consecutive gradients: `beta_delta = 1 - cos(gradient(K-1), gradient(K))`. Smaller values mean more stable gradients.

These signals are smoothed using an AUCK window.

```python
result = ssd.fit_regression("score", data=df, fixed_k=None, k_min=2, k_max=120, verbose=True)
print(f"Selected K = {result.pca_k}")   # PCAOLSResult exposes .pca_k (PLSResult uses .n_components)
print(result.stats)

result.plot_sweep("sweep.png")   # save sweep plot
result.plot_sweep()              # display interactively
```

The **blue curve** shows **detrended interpretability** as a function of K. The **orange curve** shows **solution stability**. The **red vertical line** marks the selected K.


### PLS

PLS regression operates directly in the full embedding space, finding latent directions that maximize covariance between document vectors and the outcome without a separate dimensionality-reduction step. With a single component it recovers one semantic gradient in a single pass. With `k="auto"` (default) the number of components is selected via selector `r2_se`); the p-value comes from plskit's default split-half test (`test_method="auto"`: `split_exact` or `split_nb`, chosen from X alone), at the fitted `k` (for `k="auto"`, at k=1, independent of selection).

```python
result = ssd.fit_pls(
    "score",              # values, a column name, or a formula such as "score ~ age + gender"
    data=df,
    k="auto",             # int, or "auto" for find_k_optimal
    k_max=5,              # cap for "auto"
    ssd_gradients=1,      # >1 also returns that many rotated gradients
    rotate="words",       # rotation of those gradients: "words" or None
    n_splits=50,          # split-half iterations
    random_state=2137,
    test=True,            # False skips the confirmatory test; pvalue is NaN
    verbose=False,
)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `outcome` | `array-like \| str` | — | Outcome values, a column name of `data`, or a formula with fixed covariates (see below) |
| `data` | `DataFrame \| dict` | `None` | One row per document |
| `k` | `int \| "auto"` | `"auto"` | Number of PLS components. `int` fits at exactly that k. `"auto"` calls `plskit.pls1_find_k_optimal` (selector `r2_se`, diagnostic `split_nb`). The p-value is plskit's default split-half test (`test_method="auto"`: `split_exact` or `split_nb`, chosen from X alone; `result.test.name` says which ran) at the fitted `k`; with `k="auto"` it is the k=1 test. |
| `k_max` | `int` | `5` | Cap for `k="auto"`, clamped to `min(k_max, n-1, D)`. Ignored when `k` is an int. |
| `ssd_gradients` | `int` | `1` | Number of gradients returned, in `[1, min(n-1, D)]`. At 1 the result is the standard fit alone. See [Several gradients](#several-gradients). |
| `rotate` | `"words" \| None` | `"words"` | Rotation of the gradients when `ssd_gradients > 1`; ignored at 1. |
| `n_splits` | `int` | `50` | Random splits for the split-half tests. |
| `random_state` | `int` | `2137` | Random seed. |
| `test` | `bool` | `True` | Run the confirmatory test. `False` skips it: `pvalue` is NaN, and k selection for `k="auto"` still runs. `result.test(n_splits=..., seed=...)` runs it later; with no arguments it uses 50 splits and no seed. |
| `verbose` | `bool` | `False` | Print K-selection chain and confirmatory test progress. |

To re-run the test with different settings, call `result.test(n_splits=200)` — it overwrites `result.stats.pvalue` and `result.test.pvalue` in place.

#### Several gradients

`ssd_gradients=N` with N > 1 also returns N gradients from one outcome. The standard fit does not change: the p-value, `beta`, `r2`, `docs` and `predict()` are the same as with `ssd_gradients=1`. The N gradients come from a separate N-component PLS1 fit on the same data, one leaf each.

```python
result = ssd.fit_pls("score", data=df, ssd_gradients=3)   # works with a formula too
result.keys()                               # ['dim-1', 'dim-2', 'dim-3']
result["dim-1"].words                       # words along gradient 1
result["dim-1"].stats                       # score_corr, cov_share
result.pls_info                             # rotation diagnostics
```

With `rotate="words"` (default) the N weight vectors are varimax-rotated against the projection of the leading vocabulary rows. The gradients are then ordered by the correlation of their scores with the outcome, highest first, and each is flipped so that correlation is positive. With `rotate=None` they stay in NIPALS order: gradient 1 is the first PLS weight vector, and gradients 2…N have zero covariance with the outcome, so their `score_corr` and `cov_share` are 0.

Each leaf has `.stats` (`score_corr`: correlation of its scores with the outcome; `cov_share`: its share of the squared covariance with it), `.words`, `.clusters`, `.snippets`, `.beta` and `.gradient`. Leaves have no p-value. `result.fit_info.n_components` stays the standard fit's k, which `k="auto"` may set to a value other than N. `report()` and `save_page()` show the gradients after the standard fit.

#### Covariates

With a formula, `fit_pls` controls for fixed covariates: the outcome and every column of the document vectors are regressed on the formula by OLS, and PLS runs on the residuals.

```python
result = ssd.fit_pls("rating ~ age + gender", data=df)
result.summary()         # the text's test and partial R²
```

When the test is split_exact, it permutes the residualized outcome against the residualized document vectors (Kennedy, 1995). The test can run slightly liberal when the covariates use many degrees of freedom relative to the number of documents, e.g. a factor with many levels. Random effects such as `(1 | ID)` and an `ssd()` term raise `ValueError`; use `fit_regression` for them. With a formula, `r2` is the R² of the residualized outcome, `docs.residual` has the covariates taken out, and `predict()` is not available.

`fit_pls` does not report the covariates' own coefficients. Their standard errors and p-values would have to hold the text fixed through its score on `beta`, and beta was fitted to the same outcome, so they come out too small when the number of embedding dimensions is close to the number of documents or larger. Use `fit_regression` for covariate effects.

### Several outcomes (PLS3)

When you have several numeric outcomes, `fit_multipls()` fits them together. Pass the outcomes as a list of column names of `data`, a DataFrame (column names become outcome names) or a 2-D array (names are `"y1"`, `"y2"`, …). Rows with a missing value in any outcome are dropped.

PLS3 takes the SVD of the cross-covariance between document vectors and outcomes. Dimension `i` pairs one direction in word space with one vector of outcome loadings. The returned `MultiPLSResult` is a container of per-dimension leaves keyed by `"dim-1"`, `"dim-2"`, …

```python
result = ssd.fit_multipls(
    ["valence", "arousal", "dominance"],   # column names of data, or a DataFrame / 2-D array
    data=df,
    ssd_gradients=None,   # number of dimensions; None means q
    rotate=None,          # None, "words" or "outcomes"
    outcome_groups=None,  # dict of group name -> outcome names; needs rotate="outcomes"
    n_splits=50,
    n_perm=1000,
    random_state=2137,
    verbose=False,
)

print(result.stats)             # n, q, ssd_gradients, rotate, cov_explained, pvalue
print(result.test)              # LV1 test (split_exact or split_nb)
result["dim-1"].outcomes        # outcome loadings on dimension 1
result["dim-1"].words           # words along dimension 1
result["dim-1"].clusters.pos    # cluster +beta neighbours on that dimension
result["dim-1"].stats           # cov_share, score_corr
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `outcome` | `list[str] \| DataFrame \| array` | — | The outcomes: column names of `data`, a DataFrame, or a 2-D array. At least two. |
| `data` | `DataFrame \| dict` | `None` | One row per document; holds the outcome columns |
| `ssd_gradients` | `int \| None` | `None` | Number of dimensions, in `[1, min(q, n-1, D)]`. `None` means `q`, or with `outcome_groups` the number of rotation target rows. At `q` every component is kept, so `cov_explained` is 1 by construction. |
| `rotate` | `None \| "words" \| "outcomes"` | `None` | `None` keeps the SVD dimensions. `"words"` runs varimax against the projection of the leading vocabulary rows onto the word-space directions. `"outcomes"` runs varimax against the outcome loadings (or the group rows below). At `ssd_gradients = q` without groups, it rotates each dimension onto one outcome: each loads 1 on its own outcome and 0 on the others. |
| `outcome_groups` | `dict[str, list[str]] \| None` | `None` | Only with `rotate="outcomes"`. Each group becomes one rotation target row. Outcomes in no group keep their own row. `ssd_gradients` cannot exceed the number of target rows. |
| `n_splits`, `n_perm` | `int` | `50`, `1000` | Splits and permutations for the LV1 test. |
| `random_state`, `verbose` | — | `2137`, `False` | Seed and progress output. |

A group's rotation target row and its `.outcomes` group loading are plain means of the group's outcome loadings. Outcomes scored in opposite directions cancel within a group.

Rotated modes reorder the dimensions by their share of the cross-covariance. In every mode, the sign of each dimension is set so its largest outcome loading is positive.

The container p-value comes from plskit's default PLS3 test (`test_method="auto"`: `split_exact` or `split_nb`, chosen from X alone) at the first dimension (LV1) only. It does not depend on `rotate`. Leaves have no p-value and no `.test`. To rerun it with other settings, call `result.test(n_splits=100, n_perm=2000, seed=42)`.

Container attributes and views:

- `.stats` — `n`, `q`, `ssd_gradients`, `rotate`, `cov_explained` (share of squared cross-covariance captured by the dimensions), `pvalue`
- `.pls_info` — `ssd_gradients`, `rotate`, `order`, `signs`, `sweeps`, `V_converged`, `kaiser_normalized`, `random_state`
- `.test` — LV1 test; call `.test(n_splits=..., n_perm=..., seed=...)` to rerun
- `.outcome_names`, `.outcome_groups`
- `.words`, `.clusters`, `.snippets` — fan out across dimensions
- `.report()`, `.save_page(...)`, `.attach(...)`

Leaf views (`result["dim-1"]`):

- `.outcomes` — columns `outcome`, `group`, `loading`; with `outcome_groups`, one extra row per group
- `.stats` — `cov_share` (share of the total cross-covariance of the dimensions; sums to 1 over the dimensions), `score_corr` (in-sample correlation of the word-side and outcome-side scores)
- `.words`, `.clusters`, `.snippets`

`fit_multipls` needs the embeddings attached to the `SSD` (the per-dimension views and `rotate="words"` use the vocabulary). See [`examples/several_outcomes.py`](examples/several_outcomes.py) and [`docs/api_reference.md`](docs/api_reference.md).


### Cross-Group Comparison

When your research question involves **categorical groups** rather than a continuous outcome, use `ssd.fit_groups(labels)`.

| Scenario | Use |
|---|---|
| Continuous outcome (scale score, rating) | `fit_pls()` or `fit_regression()` |
| Categorical groups (diagnosis, condition) | `fit_groups()` |
| Continuous outcome AND group labels | Both — `fit_pls()` for the continuous analysis, `fit_groups()` for the group comparison |

```python
# Categorical groups: labels as values, or a column name of data
result = ssd.fit_groups("condition", data=df, n_perm=5000, correction="holm")

# Or: median split on a continuous outcome
result = ssd.fit_groups("score", data=df, median_split=True, n_perm=5000)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `outcome` | `array-like \| str` | — | One label per document, or a column name of `data` |
| `data` | `DataFrame \| dict` | `None` | One row per document |
| `median_split` | `bool` | `False` | Split a continuous outcome into "low"/"high" at median |
| `n_perm` | `int` | `5000` | Permutation iterations |
| `correction` | `str` | `"holm"` | P-value correction: `"holm"`, `"bonferroni"`, `"fdr_bh"`, `"none"` |
| `random_state` | `int` | `2137` | Random seed |

Groups with fewer than 20 documents are automatically dropped.

Groups are canonicalised internally — original labels are remapped to `"g1"`, `"g2"`, … (in sorted order). The original-label mapping is exposed on `result.group_labels`.

`fit_groups` takes one text. It has no formula.

#### Interpreting group results

```python
print(result)                  # header + view directory
print(result.stats)            # G, n_kept, n_perm, correction, omnibus pvalue
print(result.test)             # omnibus pvalue + pairwise contrasts block

# Pairwise rows (T, p_raw, p_corrected, cohens_d, n_g1, n_g2 per contrast)
result.pairs                   # PairsListView — exports via .to_df() / .save()

# Zoom into one pair → PairResult (canonical keys: "g1", "g2", ...)
pair = result[("g1", "g2")]
pair.words.pos
pair.clusters.pos
pair.snippets

# Pivoted interpretation across all contrasts (adds a "contrast" column)
result.words.pos
result.clusters.pos(topn=100)
result.snippets.pos

# Re-run the permutation test with different settings
result.test(n_perm=10_000, correction="fdr_bh")
```

Key attributes:
- `result.G` — number of retained groups (after the 20-doc minimum filter)
- `result.n_kept` — total documents across retained groups
- `result.group_labels` — `dict` mapping canonical keys (`"g1"`, …) to original labels
- `result.test.omnibus_T`, `result.test.omnibus_p` — omnibus statistic and permutation p
- `result.pairs` — list-like view of `Pair` rows with per-contrast `T`, `p_raw`, `p_corrected`, `cohens_d`, `n_g1`, `n_g2`, `contrast_norm`

### Contrast between two gradients

`contrast(g1, g2, embeddings)` reads the difference between two gradients as a gradient of its own. It takes two gradient vectors and the embeddings to read words from, so you pick the two gradients to compare: two fits of the same text, two terms of an interaction, two dimensions, or gradients from different embeddings.

```python
from ssdiff import contrast

r1 = ssd.fit_pls("score", data=df)
r2 = ssd.fit_pls("age", data=df)
c = contrast(r1.gradient, r2.gradient, emb, labels=("score", "age"))

c.words.pos                # words toward g1 and away from g2
c.clusters.pos
c.stats                    # cos_g1_g2, diff_norm
c.save_page("contrast.html")
```

For an interaction, the gradient of a level is the sum of the `beta` of the main text term and of that level's term:

```python
res = ssd.fit_regression("score ~ sex*ssd()", data=df)
g_f = res["ssd(essay)"].beta
g_m = g_f + res["sexM:ssd(essay)"].beta
c = contrast(g_f, g_m, emb, labels=("F", "M"))
```

The contrast is `unit(g1) − unit(g2)`, normalized, so the scale of each gradient does not matter. `c.stats.cos_g1_g2` is the cosine between the two gradients. The result has `.words` and `.clusters` and no test. It has no documents, so it has no `.snippets`, `.docs` or `.alignment_scores`.

### Inspecting results

`PLSResult` and `PCAOLSResult` share the same interpretation API — everything is a printable, exportable view:

```python
print(result)                   # header + view/array directory
print(result.stats)             # backend, r², r²_adj (OLS only), p, n_kept,
                                # β-norm, Δ (per +0.10 cosine), IQR effect,
                                # |corr(y, ŷ)|, y_mean, y_std
print(result.fit_info)          # n_components, p_at_k, random_state,
                                # plus PCA-K sweep info for OLS

# Direct array attributes (numpy ndarrays)
result.beta                     # raw direction in embedding space
result.gradient                 # unit-length version of beta
result.beta_norm                # ||beta|| (effect-size summary)
result.alignment_scores         # per-doc cosine to gradient
# Number of components — the attribute name differs by backend:
result.n_components             # PLSResult only
result.pca_k                    # PCAOLSResult only
result.fit_info.n_components    # both backends (use this if the backend may vary)

# Comprehensive narrative report — every section is on by default; pass
# section=False to drop one.
print(result.report(clusters={"n": 100, "n_words": 10, "n_snippets": 2},
                    extreme_docs={"n": 30}, misdiagnosed={"n": 20}))
result.report().save("report.md")    # also .html / .docx / .tex

# One self-contained HTML results page
result.save_page("results.html")

# Re-run the significance test in place
result.test(n_splits=200)            # PLSResult — overwrites stats.pvalue

# Predict the outcome for new documents (fits without a formula)
result.predict(new_texts)            # strings, or token lists for a token-list fit
```

For `MultiPLSResult`, `GroupResult` and the container from an interaction formula, see the sections above.

After un-pickling a result, re-attach the embeddings (and texts) with `result.attach(embeddings=emb, texts=new_texts)`. Details are in [`docs/results.md`](docs/results.md).

---

## Neighbors & Clustering

### Nearest neighbors

`result.words` is a tabular view with columns `side`, `rank`, `word`, `cos_beta`:

```python
result.words            # default: top 20 per pole
result.words.pos        # one-sided, default 20 rows
result.words.pos(50)    # resize to 50 rows
result.words.neg(None)  # all available rows on this side

# Standard view exports
result.words.to_df()              # pandas DataFrame
result.words.save("words.csv")    # csv / json / md / xlsx / docx / tex
```

### Clustering neighbors into themes

`result.clusters` k-means clusters the top neighbours per pole (k auto-selected via silhouette unless pinned):

```python
result.clusters.pos              # default topn=100
result.clusters.pos(topn=200, k=4)         # recompute with different params
result.clusters.pos(cluster_id=0).words    # zoom into one cluster
result.clusters.pos(cluster_id=0).snippets # snippets aligned with that centroid
result.clusters.words            # flat per-side cluster-words table

# Columns: cluster_id, side, size, coherence, centroid_cos_beta
result.clusters.pos.to_df()
result.clusters.save("clusters.csv")
```

---

## Interpreting with Snippets

After fitting, SSD lets you link the abstract directions in embedding space back to actual language by inspecting text snippets near seed-word occurrences. Snippets are pulled from the text the result was fit on (`result.text`) — no need to pass anything. A text of token lists has no raw text, so it has no snippets.

```python
result.snippets                       # default: top 30 per pole
result.snippets.pos                   # SnippetsViewSided, top 30
result.snippets.pos(50)               # resize
result.snippets(top_per_side=200, min_cosine=0.1)   # recompute extraction

# Snippets aligned with a specific cluster centroid
result.clusters.pos(cluster_id=0).snippets

# Columns: snippet_id, side, doc_id, cosine, seed, start/end indices,
# text_window, text_surface, text_lemmas, cluster_id, contrast
result.snippets.to_df()
result.snippets.save("snippets.xlsx")
```

The snippet extraction:
1. Locates each occurrence of a seed word in the text.
2. Extracts a small window of surrounding context.
3. Represents that window as a SIF-weighted context vector.
4. Computes cosine similarity between the context vector and β, ranking snippets by alignment.

---

## Per-Document SSD Scores

`result.docs` exposes per-document predictions and the cosine alignment score (the SSD score, ⟨d_i, gradient⟩):

```python
result.docs                          # all rows; columns: doc_id, y_true,
                                     # y_hat, residual, alignment_score
result.docs.pos(20)                  # 20 most β-positive (highest y_hat)
result.docs.neg(20)                  # 20 most β-negative
result.docs.id(42)                   # single-doc detail (incl. raw text)

# Misdiagnosed — largest |residual|
result.docs.misdiagnosed(20)                     # both over and under
result.docs.misdiagnosed(20, direction="over")   # residual < 0
result.docs.misdiagnosed(20, direction="under")  # y_hat < y_true

result.docs.to_df()
result.docs.save("docs.csv")
```

`doc_id` is the position of the document in the text you gave `SSD`.

The full per-document alignment vector is also available directly:

```python
result.alignment_scores              # ndarray of shape (n_kept,)
```

---

## API Summary

The `ssdiff` top-level package exports two primary classes and one function, plus result and view classes:

```python
from ssdiff import Embeddings, SSD, fit_regression, contrast
# Result / view classes (re-exported for type hints, isinstance checks, pickling):
from ssdiff import (
    PLSResult, PCAOLSResult, RegressionResult, GroupResult, LexiconResult, ContrastResult,
    WordsView, WordsViewSided, ClustersView, ClustersViewSided,
    ClusterWordsView, ClusterWordsViewSided, SnippetsView, SnippetsViewSided,
)
# Not exported at top level:
from ssdiff.results.multi_pls_result import MultiPLSResult
```

### `Embeddings`

- `Embeddings.load(path="default", *, verbose=False, parallel=False)` — load `.wordpack`, `.kv`, `.bin`, `.txt`, `.vec` (and `.gz` variants), or a wordpack by tag or full name
- `.normalize(l2=True, abtt=1)` — in-place L2 + ABTT (rows L2-normalized again after ABTT); tracks state, safe to call repeatedly
- `.save(filename=None, fmt="bin")` — save to binary, text, or gensim format
- `emb["word"]` / `emb.get_vector("word", norm=False)` — vector lookup
- `"word" in emb` — membership check
- `len(emb)` / `.vocab_size` — vocabulary size
- `.vector_size` (alias `.dim`) — embedding dimensionality
- `.similar_by_vector(vec, topn=10, restrict_vocab=None)` — nearest neighbor search

### `SSD`

- `SSD(texts, *, embeddings=None, lexicon=None, window=3, sif_a=1e-3, use_full_doc=None)` — `texts`: strings (wordpack) or token lists; `embeddings=None` loads the default wordpack
- `.with_lexicon(lexicon, *, window=3, sif_a=1e-3, use_full_doc=False)` -> `SSD` — same tokens, another lexicon
- `.docs` — tokens per document; `.name`, `.tokenizer`, `.lexicon`, `.window`, `.sif_a`, `.use_full_doc`, `.n_kept`, `len(ssd)`
- `.fit_pls(outcome, *, data=None, k="auto", k_max=5, ssd_gradients=1, rotate="words", n_splits=50, random_state=2137, test=True, verbose=False)` -> `PLSResult`
- `.fit_multipls(outcome, *, data=None, ssd_gradients=None, rotate=None, outcome_groups=None, n_splits=50, n_perm=1000, random_state=2137, verbose=False)` -> `MultiPLSResult`
- `.fit_regression(outcome, *, data=None, family="gaussian", link=None, fixed_k=None, k_min=2, k_max=120, k_step=2, ssd_gradients=1, verbose=False, pca_sweep_args=None)` -> `PCAOLSResult` (`RegressionResult` when the text has several terms)
- `.fit_groups(outcome, *, data=None, median_split=False, n_perm=5000, correction="holm", random_state=2137, verbose=False)` -> `GroupResult`
- `.suggest_lexicon(outcome=None, *, data=None, seeds=None, top_k=30, min_docs=5, n_bins=4, corr_cap=0.30, var_type="continuous")` -> `LexiconResult` — data-driven seed word selection; with `seeds`, the tokens closest to the seeds
- `.evaluate_lexicon(outcome, lexicon, *, data=None, n_bins=4, corr_cap=0.30, var_type="continuous")` -> `LexiconResult` — score an existing lexicon (per-token + aggregate)
- `.token_stats(...)`, `.coverage_summary(...)` — the two parts of `evaluate_lexicon` alone

### Module functions

- `fit_regression(formula, data, *, family="gaussian", link=None, fixed_k=None, k_min=2, k_max=120, k_step=2, verbose=False, pca_sweep_args=None, **texts)` -> `RegressionResult` (`PCAOLSResult` for one text with one term) — texts in one formula: `fit_regression("y ~ ssd(a) + ssd(b)", df, a=ssd_a)`; `ssd(b)` with no keyword is built from the column `b`
- `contrast(g1, g2, embeddings, *, labels=("g1", "g2"))` -> `ContrastResult`

### `PLSResult` / `PCAOLSResult`

**Direct array attributes**: `beta`, `gradient`, `beta_norm`, `alignment_scores`, `x`, `y`. The component count is named per backend: **PLS** exposes `n_components`, **PCA regression** exposes `pca_k` (both are also available as `result.fit_info.n_components`). PLS adds `component_scores`, `component_weights`, `find_k_result`, `cv_scores`. PCA regression adds `pca_components`, `pca_weights`, `sweep_result`, and `model` (the glmm fit).

**Scalar views** (all expose `.r2`, `.pvalue`, … as attributes; print to read, export with `.to_df()` / `.save(...)`):
- `.stats` — `backend`, `r2`, `r2_adj` (OLS only), `pvalue`, `n_raw`, `n_kept`, `n_dropped`, `y_mean`, `y_std`, `beta_norm`, `delta`, `iqr_effect`, `y_corr_pred`
- `.fit_info` — `n_components`, `p_at_k`, `n_splits`, `random_state`, `ssd_gradients` and `rotate` (PLS), plus PCA-K sweep info for OLS
- `.pls_info` — PLS with `ssd_gradients > 1` only: `ssd_gradients`, `rotate`, `order`, `signs`, `sweeps`, `V_converged`, `kaiser_normalized`, `random_state`

**Gradient leaves** (PLS with `ssd_gradients > 1`): `result.keys()` lists `"dim-1"` … `"dim-N"`; `result["dim-1"]` has `.stats`, `.words`, `.clusters`, `.snippets`, `.beta`, `.gradient`.

**Tabular views**:
- `.words` → `WordsView` (with `.pos` / `.neg` → `WordsViewSided`, callable `(n)` to resize)
- `.clusters` → `ClustersView` (with `.pos` / `.neg` → `ClustersViewSided`, callable `(topn=…, k=…)` to recompute or `(cluster_id)` to zoom)
- `.snippets` → `SnippetsView` (with `.pos` / `.neg`, callable `(top_per_side=…)` to recompute)
- `.docs` → `DocsView` with `.pos(k)`, `.neg(k)`, `.misdiagnosed(k, direction=…)`, `.id(doc_id)`
- `.sweep` → `SweepView` (PCA+OLS only) — per-K interpretability/stability rows
- `.test` → `TestView` — callable to **re-run** the test in place (`result.test(n_splits=200)` overwrites `stats.pvalue` and `test.pvalue`)

**Methods**:
- `.report(clusters=True, top_words=True, extreme_docs=True, misdiagnosed=True)` -> `Report` — every section is on by default; pass `section=False` to drop one. Each section toggle is `True` / `False` / `None` / `dict` (e.g. `clusters={"n": 20, "n_words": 5, "n_snippets": 1}`). Stats + Fit info are always included. Use `.to_text()`, `.to_html()`, `.save("report.md")`.
- `.summary()` — compact summary of the fit and the text's test
- `.predict(texts)` — predicted outcome for new documents (fits without a formula; Gaussian for `fit_regression`)
- `.save_page(path="ssd_results.html")` — self-contained HTML results page (see `docs/results.md`)
- `.attach(texts=None, embeddings=None)` — re-attach after un-pickling
- `.plot_sweep(path=None)` — PCA-K sweep chart (`PCAOLSResult` only)

### `RegressionResult`

From a formula with several text terms (`group*ssd()`), or from `ssdiff.fit_regression`. `res.keys()` lists the terms (`'ssd(essay)'`, `'sexM:ssd(essay)'`); with several texts it lists the text names, and `res["name"]` is that text's result. `res[term]` is a `PCAOLSResult` with `.beta`, `.gradient`, `.words`, `.clusters`, `.snippets`. `.test` lists the all-text test, the per-text tests (several texts) and the per-term Wald tests. `.model`, `.covariates`, `.summary()`, `.save_page(...)` and `.attach(...)` work on the container.

### `GroupResult`

**Direct attributes**: `G`, `n_kept`, `n_perm`, `correction`, `random_state`, `group_labels` (canonical → original label dict), `x`, `groups`, `beta`, `gradient`, `beta_norm`, `alignment_scores`.

**Views**: `.stats`, `.test` (omnibus `pvalue`, `omnibus_T`, `omnibus_p`), `.pairs` (per-contrast `T`, `p_raw`, `p_corrected`, `cohens_d`, `n_g1`, `n_g2`), `.words`, `.clusters`, `.snippets` (all pivoted across contrasts, add a `contrast` column).

**Pair access**: `result[("g1", "g2")]` → `PairResult` (canonical keys only) with its own `.words`, `.clusters`, `.snippets`, `.gradient`, `.beta`. Use `result.keys()` to list available pair keys; `result.group_labels` to map canonical → original.

**Methods**: `.report(clusters=True, top_words=True)` — both sections on by default; pass `section=False` to drop one. Each toggle is `True` / `False` / `None` / `dict` (e.g. `clusters={"n": 20, "n_words": 5, "n_snippets": 1}`). Omnibus + Group labels + Pairwise contrasts are always included. `.test(n_perm=…, correction=…)` (re-runs in place); `.save_page(...)`; `.attach(...)`.

### `ContrastResult`

`.words`, `.clusters`, `.stats` (`cos_g1_g2`, `diff_norm`), `.beta`, `.gradient`, `.labels`, `.save_page(...)`. No `.snippets`, `.docs`, `.alignment_scores` or test.

### `LexiconResult`

From `ssd.suggest_lexicon(...)` and `ssd.evaluate_lexicon(...)`. Views (`.suggestions`, `.summary`, `.tokens`) and `.report()` support `.to_df()` (requires `ssdiff[results]`), `.to_dict()`, `.to_records()`, and `.save("file.{csv,json,md,xlsx,docx,tex,html}")`.

---

## Citing & License

- License: GPL v3 (see LICENSE).
- If you use SSD in published work, please cite the associated paper.
- A suggested citation:

Plisiecki, H., Lenartowicz, P., Pokropek, A., Małyska, K., & Flakus, M. (2025). Measuring Individual Differences in Meaning: The Supervised Semantic Differential. PsyArXiv. https://doi.org/10.31234/osf.io/gvrsb_v1

---

## Questions / Contributions

- File issues and feature requests on the repo's Issues page.
- Pull requests welcome — especially for:
  - Robustness diagnostics and visualization helpers
  - Documentation improvements

Contact: hplisiecki@gmail.com

Project was funded by the National Science Centre, Poland (grant no. 2020/38/E/HS6/00302).
