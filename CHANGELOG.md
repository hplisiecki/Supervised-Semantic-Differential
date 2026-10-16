# Changelog

## 3.3.0

- `fit_ols()` is renamed `fit_regression()`. The old name still works with the same arguments and warns with `FutureWarning`. Without a formula the results are the same as before.
- `fit_regression(formula, family=..., link=...)` controls for covariates and nesting: `fit_regression("~ age + gender + (1 | ID)")`. K is still chosen by the PCA sweep; the final model at that K is fit by `glmm`, with the K component scores added to the formula. The p-value tests the components together (F-test for OLS and for Gamma and inverse-Gaussian GLMs, likelihood-ratio test otherwise), and `result.model` holds the glmm fit. `predict()` is not available after a fit with a formula or a non-Gaussian family.
- `SSD(..., data=df)` takes a table with one row per document; `y` can then be a column name, or a list of column names for `fit_multipls()`.
- `glmm` (>= 0.6.0) is now a dependency.
- `token_stats`, `coverage_summary`, `evaluate_lexicon` and `suggest_lexicon(seeds=...)` take `y=None`: they then return only the statistics that need no outcome, with the outcome fields `None`.
- New `result.save_page(path)` on `fit_pls()`, `fit_regression()`, `fit_groups()` and `fit_multipls()` results writes one HTML file with the result's tables that opens in a browser from disk, with no network. `result.page_data()` returns the same content as a dict.

## 3.2.0

- spaCy is no longer installed by default: `pip install ssdiff[spacy]` to keep using `Corpus(texts, lang=...)`. New `.wordpack` embeddings tokenize without spaCy: `Corpus(texts, embeddings=Embeddings.load("en"))`.
- `Embeddings.load` takes a wordpack's tag (`"en"`) or full name (`"en-glove2024-dolma-1"`): the pack is downloaded once into a cache directory and loaded from there. With no argument it loads the tag `"default"`, the same pack as `"en"`. `report()` and `corpus.tokenizer` record the full name.
- The `.ssdembed` format is deprecated: loading one raises an error, and `Embeddings.save` no longer writes it (`fmt` now defaults to `"bin"`).
- `Embeddings.load` no longer takes `ram_efficient`. A `.wordpack` always keeps only its 50,000 most frequent rows in RAM.
- `Embeddings.normalize` no longer takes `re_normalize`: rows are always L2-normalized again after ABTT, and `l2_normalized` then reports it.
- New `result.predict(texts)` on `fit_pls()` and `fit_ols()` results: the predicted outcome for new documents, given as a `Corpus` or, for a fit on a wordpack, as raw texts.
- `fit_pls(test=False)` skips the significance test; `pvalue` is then `NaN`.
- `SSD(sif_a=None)` gives every word weight 1 (plain mean, no SIF weighting).
- `docs.y_hat` now subtracts the training mean of the document vectors before applying `beta`, so `y_hat` and `residual` differ from 3.1.0 by a constant for the same fit.
- `ssd.n_raw` now counts every corpus document and `ssd.n_dropped` includes documents with an invalid outcome; `doc_id` is the document's row in the corpus.
- `Corpus` raises `TypeError` when `texts` is a single string.
