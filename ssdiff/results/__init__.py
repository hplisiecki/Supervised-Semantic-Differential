"""Public exports of the results package."""

from ssdiff.results.core import Result
from ssdiff.results.display import set_repr_hints
from ssdiff.results.group_result import GroupResult, PairView
from ssdiff.results.lexicon_result import LexiconResult
from ssdiff.results.continuous_result import ContinuousResult, PCAOLSResult, PLSResult

__all__ = [
    "Result", "ContinuousResult", "PLSResult", "PCAOLSResult",
    "GroupResult", "PairView", "LexiconResult",
    "set_repr_hints",
]
