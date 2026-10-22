# Example data

| file | content | licence |
|---|---|---|
| `BRM-emot-submit.csv` | Warriner et al. (2013) valence, arousal and dominance norms, the original file | CC BY-NC-ND 3.0 |
| `measuring_hate_speech.csv.gz` | one row per comment of the Measuring Hate Speech corpus | CC BY 4.0 |
| `glove_10k.bin` | 10,000 rows of GloVe 42B 300d, word2vec binary | PDDL 1.0 |

`make_data.py` builds the last two from local copies of the sources. It needs
`spacy` with `en_core_web_sm`.

## BRM-emot-submit.csv

Valence, arousal and dominance ratings for 13,915 English lemmas, overall and
split by gender, age and education (65 columns). The file is unmodified.

Source: Electronic Supplementary Material 1 of the article,
<https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0314-x/MediaObjects/13428_2012_314_MOESM1_ESM.zip>

Licence: Creative Commons Attribution-NonCommercial-NoDerivs 3.0 (CC BY-NC-ND 3.0),
<https://creativecommons.org/licenses/by-nc-nd/3.0/>. It is not covered by the
licence of the repository that contains it.

- Attribution: cite the article below when you use or share the file.
- Non-commercial: do not use or share it for commercial purposes.
- No derivatives: share it only as it is. Do not share edited, trimmed or converted copies.

Citation: Warriner, A. B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence,
arousal, and dominance for 13,915 English lemmas. *Behavior Research Methods*,
45(4), 1191-1207. <https://doi.org/10.3758/s13428-012-0314-x>

## measuring_hate_speech.csv.gz

39,565 comments from the Measuring Hate Speech corpus
(<https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech>), which
holds one row per annotation. Columns:

- `comment_id`
- `text`: the first text of the comment
- `score`: the mean `hate_speech_score` over the comment's annotations, rounded to 3 decimals
- `women`, `black`, `muslim`, `gay`, `immigrant`, `jewish`: 1 when at least half of the
  comment's annotators flagged `target_gender_women`, `target_race_black`,
  `target_religion_muslim`, `target_sexuality_gay`, `target_origin_immigrant`,
  `target_religion_jewish`, else 0

Licence: Creative Commons Attribution 4.0 (CC BY 4.0),
<https://creativecommons.org/licenses/by/4.0/>.

Citation:

- Kennedy, C. J., Bacon, G., Sahn, A., & von Vacano, C. (2020). Constructing
  interval variables via faceted Rasch measurement and multitask deep learning:
  a hate speech application. arXiv:2009.10277.
- Sachdeva, P., Barreto, R., Bacon, G., Sahn, A., von Vacano, C., & Kennedy, C. (2022).
  The Measuring Hate Speech Corpus: Leveraging Rasch Measurement Theory for Data
  Perspectivism. In *Proceedings of the 1st Workshop on Perspectivist Approaches to
  NLP @ LREC 2022*, 83-94.

The comments contain hateful and offensive language.

## glove_10k.bin

The 10,000 most frequent spaCy lemmas (`en_core_web_sm`, lower-cased, alphabetic
tokens) of the comments above that GloVe has a vector for, with their GloVe 42B
300d vectors unchanged. Rows are ordered by lemma frequency.

Source: GloVe, Common Crawl (42B tokens, 1.9M vocab, uncased, 300d),
<https://nlp.stanford.edu/projects/glove/>

Licence: Open Data Commons Public Domain Dedication and License v1.0 (PDDL 1.0),
<https://opendatacommons.org/licenses/pddl/1-0/>.

Citation: Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global
Vectors for Word Representation. In *Proceedings of EMNLP 2014*, 1532-1543.
