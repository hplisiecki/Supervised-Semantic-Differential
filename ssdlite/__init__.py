"""SSDLite — Supervised Semantic Differential."""

from ssdlite.embeddings import Embeddings
from ssdlite.corpus import Corpus
from ssdlite.ssd import SSD
from ssdlite.utils.group import SSDGroup, SSDContrast

__all__ = ["Embeddings", "Corpus", "SSD", "SSDGroup", "SSDContrast"]

from importlib.metadata import version as _version, PackageNotFoundError
try:
    __version__ = _version("ssdlite")
except PackageNotFoundError:
    __version__ = "1.0.0"  # fallback for uninstalled dev
