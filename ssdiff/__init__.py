"""ssdiff — Supervised Semantic Differential.

Top-level package: re-exports the primary public API (``SSD``,
``Embeddings``, result classes, view classes) so users can do ``from ssdiff import SSD``
without knowing the internal module structure.
"""

from ssdiff.embeddings import Embeddings
from ssdiff.results import (
    ClustersView,
    ClustersViewSided,
    ClusterWordsView,
    ClusterWordsViewSided,
    ContrastResult,
    GroupResult,
    LexiconResult,
    PCAOLSResult,
    PLSResult,
    RegressionResult,
    Result,
    SnippetsView,
    SnippetsViewSided,
    WordsView,
    WordsViewSided,
    contrast,
    set_repr_hints,
)
from ssdiff.ssd import SSD, fit_regression
from ssdiff.utils.diagnostics import progress_hook

__all__ = [
    "SSD",
    "Embeddings",
    "ClusterWordsView",
    "ClusterWordsViewSided",
    "ClustersView",
    "ClustersViewSided",
    "ContrastResult",
    "GroupResult",
    "LexiconResult",
    "PCAOLSResult",
    "PLSResult",
    "RegressionResult",
    "Result",
    "SnippetsView",
    "SnippetsViewSided",
    "WordsView",
    "WordsViewSided",
    "contrast",
    "fit_regression",
    "progress_hook",
    "set_repr_hints",
]

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("ssdiff")
except PackageNotFoundError:
    __version__ = "4.0.1"
