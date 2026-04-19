"""Public API of the ``ssdiff.results`` package.

Re-exports the result classes (``PLSResult``, ``PCAOLSResult``,
``GroupResult``, ``LexiconResult``), their shared base ``Result``,
the ``PairView`` contrast view, and the ``set_repr_hints`` display toggle.
"""

from ssdiff.results.continuous_result import ContinuousResult, PCAOLSResult, PLSResult
from ssdiff.results.core import Result
from ssdiff.results.display import set_repr_hints
from ssdiff.results.group_result import GroupResult, PairView
from ssdiff.results.lexicon_result import LexiconResult

__all__ = [
    "ContinuousResult",
    "GroupResult",
    "LexiconResult",
    "PCAOLSResult",
    "PLSResult",
    "PairView",
    "Result",
    "set_repr_hints",
]
