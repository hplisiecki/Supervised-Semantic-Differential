# ssdiff — Embeddings

How to get word embeddings into `ssdiff`: a ready English pack, or your own file.

For the argument tables of `Embeddings` see [`api_reference.md`](api_reference.md#embeddings).

---

## Load the English default

```python
from ssdiff import Embeddings, Corpus

emb = Embeddings.load()                  # same as Embeddings.load("en")
corpus = Corpus(texts, embeddings=emb)   # the pack's own tokenizer and stopwords; no spaCy
```

The first call downloads the pack (380 MB) into a cache directory. Later calls load it from there.

What you get is a **wordpack**: one file with the word vectors, the tokenizer and the stopword list. It is ready to use: rows are L2-normalized and ABTT (m = 1) is already applied, so you do not call `normalize()`.

---

## Choose a pack

| short tag | full name | words | use it for |
|---|---|---|---|
| `en` | `en-glove2024-dolma-1` | 1,200,000 | informal or recent text: comments, social media, open survey answers, forums |
| `en-wiki` | `en-glove2024-wikigiga-1` | 1,291,146 | formal text: news, essays, reports, older press |

```python
emb = Embeddings.load("en-wiki")
```

Both are GloVe 2024 (Carlson, Bauer & Manning, 2025, arXiv:2507.18103): 300 dimensions, lowercase, licence PDDL-1.0.

---

## Short tag or full name

```python
emb = Embeddings.load("en")                     # short tag: the current default for English
emb = Embeddings.load("en-glove2024-dolma-1")   # full name: always this file
```

- A **full name** always means the same file. Once the file is in the cache it loads with no internet connection.
- A **short tag** follows the published default and can move to a newer pack. It is looked up online on every load. With no connection, the last answer is used.

Use the full name in scripts whose results must be reproducible. A fit records it for you:

```python
emb.name, emb.version    # ("en-glove2024-dolma", "1")
corpus.tokenizer         # "wordpack:en-glove2024-dolma-1"
result.report()          # the "embeddings" row reads en-glove2024-dolma-1
```

A name is used only when no file exists at that path. A file called `en` in the working directory is loaded as a file.

---

## Where packs are stored

| System | Directory |
|---|---|
| Linux | `~/.cache/wordpack` (`$XDG_CACHE_HOME/wordpack` when that variable is set) |
| macOS | `~/Library/Caches/wordpack` |
| Windows | `%LOCALAPPDATA%\wordpack` |

Set the environment variable `WORDPACK_CACHE` to use another directory. Nothing removes the files: delete them there to free space.

---

## Working with a wordpack

- **Lexicon.** English packs are keyed by surface form, not by lemma. List the forms that occur in the texts: `dog` and `dogs`, not the lemma alone. `ssdiff` does no lemma expansion.
- **ABTT.** `emb.normalize(abtt=m)` sets any m from 0 to 10, up or down.
- **Memory.** A pack keeps its 50,000 most frequent words in RAM. Other words are read from the file when needed, so keep the file in place while you use `emb`.
- **Saving.** A wordpack cannot be saved with `emb.save()`. Load it again by name.

---

## Use your own embedding file

For another language, or your own model, load a file:

```python
emb = Embeddings.load("path/to/model.bin", verbose=True)
emb.normalize(l2=True, abtt=1)        # L2 + ABTT (remove the top principal component)
corpus = Corpus(texts, lang="pl")     # spaCy tokenization; needs pip install ssdiff[spacy]
```

| Format | Extension | Notes |
|---|---|---|
| Wordpack | `.wordpack` | A pack file you built yourself; loads like a named pack |
| gensim KeyedVectors | `.kv` | Loads without gensim |
| word2vec binary | `.bin` | Standard binary format |
| Text | `.txt`, `.vec` | One word per line + floats |
| Compressed | `.txt.gz`, `.vec.gz`, `.bin.gz` | Gzip-compressed versions of the above |

`normalize()` with no arguments applies both L2 and ABTT (m = 1). It tracks what was applied, so calling it again skips the steps already done.

Look for pre-trained static word embeddings in your language with good vocabulary coverage for your domain. GloVe and word2vec trained on large general corpora are a reliable starting point. For Polish, `nkjp+wiki-lemmas-all-300-cbow-hs.txt.gz` (no. 25) from the [Polish Word2Vec model list](https://dsmodels.nlp.ipipan.waw.pl) was found to work well.

With spaCy the corpus holds lemmas, so the lexicon lists lemmas.

---

## When loading fails

| Message | What it means |
|---|---|
| `No file or wordpack named 'glove'. Short tags: en, en-wiki.` | There is no file at that path and no pack with that tag. Check the path. |
| `No wordpack named 'en-glove2024-dolma-2'. Versions of en-glove2024-dolma: 1.` | The pack exists, but not in that version. |
| `No file named 'en', and it cannot be looked up as a wordpack: no host answered …` | There is no file at that path, and the pack list cannot be read: no internet connection, or the host is down. A full name that is already in the cache still loads. |
| `… Upgrade ssdiff.` | The pack is newer than this `ssdiff` can read. |
