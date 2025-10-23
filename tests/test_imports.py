def test_public_api_imports():
    from ssdiff import (
        SSD, load_embeddings, normalize_kv,
        load_spacy, load_stopwords, preprocess_texts, build_docs_from_preprocessed,
        suggest_lexicon, token_presence_stats, coverage_by_lexicon,
        cluster_top_neighbors, 
    )
    # Sanity references to avoid “imported but unused”
    assert SSD is not None
