"""ssdiff — Supervised Semantic Differential.

Top-level package: re-exports the primary public API (``SSD``, ``Corpus``,
``Embeddings``, result classes) so users can do ``from ssdiff import SSD``
without knowing the internal module structure.
"""

from ssdiff.corpus import Corpus
from ssdiff.embeddings import Embeddings
from ssdiff.results import (
    GroupResult,
    LexiconResult,
    PairView,
    PCAOLSResult,
    PLSResult,
    Result,
    set_repr_hints,
)
from ssdiff.ssd import SSD
from ssdiff.utils.diagnostics import progress_hook

__all__ = [
    "SSD",
    "Corpus",
    "Embeddings",
    "GroupResult",
    "LexiconResult",
    "PCAOLSResult",
    "PLSResult",
    "PairView",
    "Result",
    "progress_hook",
    "set_repr_hints",
]

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("ssdiff")
except PackageNotFoundError:
    __version__ = "1.0.0"
