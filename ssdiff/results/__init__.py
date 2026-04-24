"""Public API of the ``ssdiff.results`` package.

Re-exports the result classes (``PLSResult``, ``PCAOLSResult``,
``GroupResult``, ``LexiconResult``, ``UMAPVisResult``), their shared base
``Result``, the sided view classes (``WordsViewSided``, ``ClustersViewSided``,
``SnippetsViewSided``), and the ``set_repr_hints`` display toggle.
"""

from ssdiff.results.continuous_result import (
    ClusterWordsView,
    ClusterWordsViewSided,
    ClustersView,
    ClustersViewSided,
    ContinuousResult,
    PCAOLSResult,
    PLSResult,
    SnippetsView,
    SnippetsViewSided,
    WordsView,
    WordsViewSided,
)
from ssdiff.results.core import Result
from ssdiff.results.display import set_repr_hints
from ssdiff.results.group_result import GroupResult, PairResult
from ssdiff.results.lexicon_result import LexiconResult

__all__ = [
    "ClusterWordsView",
    "ClusterWordsViewSided",
    "ClustersView",
    "ClustersViewSided",
    "ContinuousResult",
    "GroupResult",
    "LexiconResult",
    "PairResult",
    "PCAOLSResult",
    "PLSResult",
    "Result",
    "SnippetsView",
    "SnippetsViewSided",
    "WordsView",
    "WordsViewSided",
    "set_repr_hints",
]
