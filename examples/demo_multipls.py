"""Demo of SSD.fit_multipls — varimax rotation on Imbir rating dimensions."""

import pandas as pd

from ssdiff import Corpus, Embeddings, SSD

embeddings_path = "Models/glove_800_normalized.ssdembed"
imbir_path = "Corpuses/Imbir/imbir.csv"

word_col = "polish word"
dimensions = {
    "arousal":      "arousal_M",
    "valence":      "Valence_M",
    "dominance":    "dominance_M",
    "significance": "significance_M",
}

emb = Embeddings.load(embeddings_path)

df = pd.read_csv(imbir_path, encoding="utf-8")
words = df[word_col].astype(str).str.strip().tolist()
ratings = {name: pd.to_numeric(df[col], errors="coerce").to_numpy()
           for name, col in dimensions.items()}
corpus_full = Corpus(words, lang="pl")

for dim, col in dimensions.items():
    print(f"\n══ {dim} ({col}) ══")

    ssd = SSD(emb, corpus_full, ratings[dim], use_full_doc=True)

    print("── varimax, k=auto ──")
    res_varimax = ssd.fit_multipls(k="auto", rotate="varimax")
    print(res_varimax.stats)
    print(res_varimax.words)
    # res_varimax.report(top_words=10).save(f"multipls_varimax_{dim}.md")
