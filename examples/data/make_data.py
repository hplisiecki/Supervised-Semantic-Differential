"""Build the two derived example data files from local sources.

Outputs, written next to this script:
  - measuring_hate_speech.csv.gz : one row per comment (text, mean score, six 0/1 target flags)
  - glove_10k.bin                : the 10,000 most frequent comment lemmas found in GloVe 42B 300d

Inputs (outside this repository):
  - Corpuses/measuring_hate_speech.parquet
  - Models/glove.42B.300d.txt

Run once; the outputs are committed. No randomness: reruns give the same files.

    python examples/data/make_data.py
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
import spacy

from ssdiff import Embeddings

HERE = os.path.dirname(os.path.abspath(__file__))
SSD_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PARQUET = os.path.join(SSD_ROOT, "Corpuses", "measuring_hate_speech.parquet")
GLOVE = os.path.join(SSD_ROOT, "Models", "glove.42B.300d.txt")

TARGETS = {
    "women": "target_gender_women",
    "black": "target_race_black",
    "muslim": "target_religion_muslim",
    "gay": "target_sexuality_gay",
    "immigrant": "target_origin_immigrant",
    "jewish": "target_religion_jewish",
}
N_WORDS = 10_000
DIM = 300

# ---- hate speech: annotation rows -> one row per comment ----
ann = pd.read_parquet(PARQUET)
for name, col in TARGETS.items():
    ann[name] = ann[col].astype(float)
g = ann.groupby("comment_id", sort=True)
comments = pd.DataFrame({
    "text": g["text"].first(),
    "score": g["hate_speech_score"].mean().round(3),
})
# A comment carries a target when at least half of its annotators flagged it.
for name in TARGETS:
    comments[name] = (g[name].mean() >= 0.5).astype(int)
comments = comments.reset_index()
out_csv = os.path.join(HERE, "measuring_hate_speech.csv.gz")
comments.to_csv(out_csv, index=False, compression={"method": "gzip", "mtime": 0})
print(f"{out_csv}: {len(comments)} rows")
print(comments[list(TARGETS)].sum().to_string())

# ---- glove_10k.bin: the comments' most frequent lemmas that GloVe has ----
nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner"])
counts = Counter()
for doc in nlp.pipe(comments["text"].tolist(), batch_size=256):
    counts.update(t.lemma_.lower() for t in doc if t.is_alpha)

found = {}
with open(GLOVE, encoding="utf8") as f:
    for line in f:
        parts = line.rstrip("\n").split(" ")
        word = " ".join(parts[:-DIM])
        if word in counts and word not in found:
            found[word] = np.asarray(parts[-DIM:], dtype=np.float32)

keys = sorted(found, key=lambda w: (-counts[w], w))[:N_WORDS]
emb = Embeddings(keys, np.stack([found[w] for w in keys]))
emb.save(os.path.join(HERE, "glove_10k"), fmt="bin")
print(f"glove_10k.bin: {len(keys)} words, least frequent kept: {keys[-1]!r} ({counts[keys[-1]]})")
