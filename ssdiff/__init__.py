"""ssdiff — Supervised Semantic Differential."""

from ssdiff.corpus import Corpus
from ssdiff.embeddings import Embeddings
from ssdiff.ssd import SSD

__all__ = ["SSD", "Corpus", "Embeddings"]

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("ssdiff")
except PackageNotFoundError:
    __version__ = "1.0.0"  # fallback for uninstalled dev
