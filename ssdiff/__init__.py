"""ssdiff — Supervised Semantic Differential.

Top-level package: re-exports the primary public API (``SSD``, ``Corpus``,
``Embeddings``, result classes, view classes) so users can do ``from ssdiff import SSD``
without knowing the internal module structure.
"""

from ssdiff.corpus import Corpus
from ssdiff.embeddings import Embeddings
from ssdiff.results import (
    ClusterWordsView,
    ClusterWordsViewSided,
    ClustersView,
    ClustersViewSided,
    GroupResult,
    LexiconResult,
    PCAOLSResult,
    PLSResult,
    Result,
    SnippetsView,
    SnippetsViewSided,
    WordsView,
    WordsViewSided,
    set_repr_hints,
)
from ssdiff.ssd import SSD
from ssdiff.utils.diagnostics import progress_hook

__all__ = [
    "SSD",
    "Corpus",
    "Embeddings",
    "ClusterWordsView",
    "ClusterWordsViewSided",
    "ClustersView",
    "ClustersViewSided",
    "GroupResult",
    "LexiconResult",
    "PCAOLSResult",
    "PLSResult",
    "Result",
    "SnippetsView",
    "SnippetsViewSided",
    "WordsView",
    "WordsViewSided",
    "progress_hook",
    "set_repr_hints",
]

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("ssdiff")
except PackageNotFoundError:
    __version__ = "1.0.0"
