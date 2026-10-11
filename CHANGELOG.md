# Changelog

## 3.2.0

- spaCy is no longer installed by default: `pip install ssdiff[spacy]` to keep using `Corpus(texts, lang=...)`. New `.wordpack` embeddings tokenize without spaCy: `Corpus(texts, embeddings=Embeddings.load("en"))`.
- `Embeddings.load` takes a wordpack's short tag (`"en"`) or full name (`"en-glove2024-dolma-1"`): the pack is downloaded once into a cache directory and loaded from there. With no argument it loads `"en"`. `report()` and `corpus.tokenizer` record the full name.
- The `.ssdembed` format is deprecated: loading one raises an error, and `Embeddings.save` no longer writes it (`fmt` now defaults to `"bin"`).
- `Embeddings.load` no longer takes `ram_efficient`. A `.wordpack` always keeps only its 50,000 most frequent rows in RAM.
- `Embeddings.normalize` no longer takes `re_normalize`: rows are always L2-normalized again after ABTT, and `l2_normalized` then reports it.
- New `result.predict(texts)` on `fit_pls()` and `fit_ols()` results: the predicted outcome for new documents, given as a `Corpus` or, for a fit on a wordpack, as raw texts.
- `fit_pls(test=False)` skips the significance test; `pvalue` is then `NaN`.
- `SSD(sif_a=None)` gives every word weight 1 (plain mean, no SIF weighting).
- `docs.y_hat` now subtracts the training mean of the document vectors before applying `beta`, so `y_hat` and `residual` differ from 3.1.0 by a constant for the same fit.
- `ssd.n_raw` now counts every corpus document and `ssd.n_dropped` includes documents with an invalid outcome; `doc_id` is the document's row in the corpus.
- `Corpus` raises `TypeError` when `texts` is a single string.
