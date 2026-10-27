# ssdiff — Embeddings

How to get word embeddings and texts into `ssdiff`: a ready English pack with strings, or your own file with token lists.

For the argument tables of `Embeddings` see [`api_reference.md`](api_reference.md#embeddings).

---

## Load the English default

```python
from ssdiff import Embeddings, SSD

emb = Embeddings.load()                    # the same pack as Embeddings.load("en")
ssd = SSD(df["text"], embeddings=emb, lexicon=lexicon)   # strings: the pack's own tokenizer and stopwords
```

The first call downloads the pack (380 MB) into a cache directory. Later calls load it from there.

What you get is a **wordpack**: one file with the word vectors, the tokenizer and the stopword list. It is ready to use: rows are L2-normalized and ABTT (m = 1) is already applied, so you do not call `normalize()`. `SSD` tokenizes strings with the pack, so you pass plain text.

---

## Choose a pack

| tag | full name | words | use it for |
|---|---|---|---|
| `en`, `en-dolma`, `default` | `en-glove2024-dolma-1` | 1,200,000 | informal or recent text: comments, social media, open survey answers, forums |
| `en-wiki` | `en-glove2024-wikigiga-1` | 1,291,146 | formal text: news, essays, reports, older press |

```python
emb = Embeddings.load("en-wiki")
```

Both are GloVe 2024 (Carlson, Bauer & Manning, 2025, arXiv:2507.18103): 300 dimensions, lowercase, licence PDDL-1.0.

---

## Tag or full name

```python
emb = Embeddings.load("en")                     # tag: the English default of this ssdiff version
emb = Embeddings.load("en-glove2024-dolma-1")   # full name: always this file
```

- A **full name** always means the same file. Once the file is in the cache it loads with no internet connection.
- A **tag** is a short name that ssdiff ships with. It means the same pack for a given ssdiff version, and a later version can move it to a newer pack. Loading by tag prints one line that names the pack. Once the pack is in the cache it loads with no internet connection.

Use the full name in scripts whose results must be reproducible. A fit records it for you:

```python
emb.name, emb.version    # ("en-glove2024-dolma", "1")
ssd.tokenizer            # "wordpack:en-glove2024-dolma-1"
result.report()          # the "embeddings" row reads en-glove2024-dolma-1
```

A tag is checked first: `Embeddings.load("en")` loads the pack even when a file called `en` is in the working directory. Write `"./en"` to load that file. A full name is used only when no file exists at that path. A name that is no tag, no file and no full name raises `FileNotFoundError`, and the message lists the tags.

---

## Where packs are stored

Packs are kept in `.wordpacks` in your home directory: `~/.wordpacks` on Linux and macOS, `C:\Users\<name>\.wordpacks` on Windows.

Set the environment variable `WORDPACK_CACHE` to use another directory. Nothing removes the files: delete them there to free space.

---

## Working with a wordpack

- **Lexicon.** English packs are keyed by surface form, not by lemma. List the forms that occur in the texts: `dog` and `dogs`, not the lemma alone. `ssdiff` does no lemma expansion.
- **ABTT.** `emb.normalize(abtt=m)` sets any m from 0 to 10, up or down.
- **Memory.** A pack keeps its 50,000 most frequent words in RAM. Other words are read from the file when needed, so keep the file in place while you use `emb`.
- **Saving.** A wordpack cannot be saved with `emb.save()`. Load it again by name.

---

## Use your own embedding file

For another language, or your own model, load a file and pass token lists:

```python
emb = Embeddings.load("path/to/model.bin", verbose=True)
emb.normalize(l2=True, abtt=1)        # L2 + ABTT (remove the top principal component)

tokens = [["happy", "day", "sunshine"], ["sad", "rain", "cold"], ...]   # one token list per document
ssd = SSD(tokens, embeddings=emb, lexicon=lexicon)
```

| Format | Extension | Notes |
|---|---|---|
| Wordpack | `.wordpack` | A pack file you built yourself; loads like a named pack |
| gensim KeyedVectors | `.kv` | Loads without gensim |
| word2vec binary | `.bin` | Standard binary format |
| Text | `.txt`, `.vec` | One word per line + floats |
| Compressed | `.txt.gz`, `.vec.gz`, `.bin.gz` | Gzip-compressed versions of the above |

`SSD` needs L2-normalized rows. `normalize()` with no arguments applies both L2 and ABTT (m = 1). It tracks what was applied, so calling it again skips the steps already done. Loading a file that is not a wordpack gives a warning that reminds you to pass token lists.

Look for pre-trained static word embeddings in your language with good vocabulary coverage for your domain. GloVe and word2vec trained on large general corpora are a reliable starting point. For Polish, `nkjp+wiki-lemmas-all-300-cbow-hs.txt.gz` (no. 25) from the [Polish Word2Vec model list](https://dsmodels.nlp.ipipan.waw.pl) was found to work well.

---

## Texts for your own embedding

Only a wordpack tokenizes strings, and wordpacks exist for English. For any other embedding, each document is a list of tokens.

- The tokens are used as given. Each is matched to the embedding's keys exactly, and a token that is not a key is skipped.
- Make the tokens in the form the embedding is keyed by: lowercase tokens for a lowercase embedding, lemmas for a lemmatized one. The Polish model above is lemmatized, so lemmatize the Polish texts first.
- The lexicon lists tokens in that same form.
- A text of token lists has no raw text, so a result from it has no snippets and `result.docs.id(i)` shows no original text. Predicting for new documents (`result.predict(...)`) and `result.attach(texts=...)` take token lists too.

Any tool can make the tokens. `examples/custom_embedding.py` lemmatizes with spaCy; spaCy is optional and ssdiff does not depend on it.

---

## When loading fails

| Message | What it means |
|---|---|
| `No wordpack tag or file named 'glove'. Tags: default, en, en-dolma, en-wiki.` | There is no file at that path and no pack with that tag. Check the path. |
| `No wordpack named 'en-glove2024-dolma-2'. Versions of en-glove2024-dolma: 1.` | The pack exists, but not in that version. |
| `Wordpack 'en-glove2024-dolma-1' is not in the cache (<dir>) and cannot be downloaded: no host answered for index.json …` | The pack is not in the local cache, and the pack list cannot be read: no internet connection, or the host is down. A full name that is already in the cache still loads. |
| `… Upgrade ssdiff.` | The pack is newer than this `ssdiff` can read. |
