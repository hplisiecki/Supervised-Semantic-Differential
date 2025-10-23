# words_apart_old/__init__.py
from .core import SSD
from .clusters import cluster_top_neighbors
from .utils import (
    load_embeddings, normalize_kv, compute_global_sif,
    build_doc_vectors, filtered_neighbors
)
from .lexicon import suggest_lexicon, coverage_by_lexicon, token_presence_stats
from .preprocess import load_spacy, load_stopwords, preprocess_texts, build_docs_from_preprocessed


__all__ = [
    "SSD",
    "cluster_top_neighbors",
    "load_embeddings", "normalize_kv", "compute_global_sif",
    "build_doc_vectors", "filtered_neighbors", "build_docs_from_preprocessed",
    "suggest_lexicon", "coverage_by_lexicon", "token_presence_stats",
    "load_spacy", 'load_stopwords', 'preprocess_texts',
    "suggest_lexicon", "coverage_by_lexicon", "token_presence_stats",
]