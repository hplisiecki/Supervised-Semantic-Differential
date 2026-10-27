"""Wordpack tags: what ``Embeddings.load("en")`` loads."""

from __future__ import annotations

# tag -> (pack file name on the host, message printed to stderr when the tag is loaded)
WORDPACK_TAGS: dict[str, tuple[str, str]] = {
    "default": (
        "en-glove2024-dolma-1.wordpack",
        "No embedding named: loading the default English wordpack, en-glove2024-dolma-1 "
        "(web pages, Reddit, books). For formal text such as news, essays and reports, "
        'use Embeddings.load("en-wiki").',
    ),
    "en": (
        "en-glove2024-dolma-1.wordpack",
        "wordpack en: en-glove2024-dolma-1 (web pages, Reddit, books). "
        'For formal text such as news, essays and reports, use Embeddings.load("en-wiki").',
    ),
    "en-dolma": (
        "en-glove2024-dolma-1.wordpack",
        "wordpack en-dolma: en-glove2024-dolma-1 (web pages, Reddit, books).",
    ),
    "en-wiki": (
        "en-glove2024-wikigiga-1.wordpack",
        "wordpack en-wiki: en-glove2024-wikigiga-1 (Wikipedia 2024 and newswire 1994-2010). "
        'For informal or recent text, use Embeddings.load("en").',
    ),
}
