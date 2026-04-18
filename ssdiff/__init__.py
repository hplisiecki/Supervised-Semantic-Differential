"""ssdiff — Supervised Semantic Differential."""

from ssdiff.utils.diagnostics import progress_hook
from ssdiff.corpus import Corpus
from ssdiff.embeddings import Embeddings
from ssdiff.ssd import SSD
from ssdiff.results import (
    Result,
    PLSResult,
    PCAOLSResult,
    GroupResult,
    PairView,
    LexiconResult,
    set_repr_hints,
)

__all__ = [
    "SSD", "Corpus", "Embeddings", "progress_hook",
    "Result", "PLSResult", "PCAOLSResult",
    "GroupResult", "PairView", "LexiconResult",
    "set_repr_hints",
]

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("ssdiff")
except PackageNotFoundError:
    __version__ = "1.0.0"
