"""Public API of the ``ssdiff.results`` package.

Re-exports the result classes (``PLSResult``, ``PCAOLSResult``,
``GroupResult``, ``LexiconResult``), their shared base ``Result``,
the sided view classes (``WordsViewSided``, ``ClustersViewSided``,
``SnippetsViewSided``), the paired view classes
(``WordsViewPaired``, ``ClustersIndexPaired``, ``ClustersViewSidedPaired``,
``SnippetsViewPaired``), and the ``set_repr_hints`` display toggle.
"""

from ssdiff.results.continuous_result import (
    ClustersViewSided,
    ContinuousResult,
    PCAOLSResult,
    PLSResult,
    SnippetsViewSided,
    WordsViewSided,
)
from ssdiff.results.core import Result
from ssdiff.results.display import set_repr_hints
from ssdiff.results.group_result import GroupResult
from ssdiff.results.lexicon_result import LexiconResult
from ssdiff.results.paired_view import (
    ClustersIndexPaired,
    ClustersViewSidedPaired,
    SnippetsViewPaired,
    WordsViewPaired,
)

__all__ = [
    "ClustersIndexPaired",
    "ClustersViewSided",
    "ClustersViewSidedPaired",
    "ContinuousResult",
    "GroupResult",
    "LexiconResult",
    "PCAOLSResult",
    "PLSResult",
    "Result",
    "SnippetsViewPaired",
    "SnippetsViewSided",
    "WordsViewPaired",
    "WordsViewSided",
    "set_repr_hints",
]
