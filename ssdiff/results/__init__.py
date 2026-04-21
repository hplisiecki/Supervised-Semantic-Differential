"""Public API of the ``ssdiff.results`` package.

Re-exports the result classes (``PLSResult``, ``PCAOLSResult``,
``GroupResult``, ``LexiconResult``), their shared base ``Result``,
the sided view classes (``WordsViewSided``, ``ClustersViewSided``,
``SnippetsViewSided``), the paired view classes
(``WordsViewPaired``, ``ClustersViewPaired``, ``ClustersViewSidedPaired``,
``SnippetsViewPaired``, ``SnippetsViewSidedPaired``), and the
``set_repr_hints`` display toggle.
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
from ssdiff.results.group_result import GroupResult
from ssdiff.results.lexicon_result import LexiconResult
from ssdiff.results.paired_view import (
    ClusterWordsViewPaired,
    ClustersViewPaired,
    ClustersViewSidedPaired,
    SnippetsViewPaired,
    SnippetsViewSidedPaired,
    WordsViewPaired,
)

__all__ = [
    "ClusterWordsView",
    "ClusterWordsViewPaired",
    "ClusterWordsViewSided",
    "ClustersView",
    "ClustersViewPaired",
    "ClustersViewSided",
    "ClustersViewSidedPaired",
    "ContinuousResult",
    "GroupResult",
    "LexiconResult",
    "PCAOLSResult",
    "PLSResult",
    "Result",
    "SnippetsView",
    "SnippetsViewPaired",
    "SnippetsViewSided",
    "SnippetsViewSidedPaired",
    "WordsView",
    "WordsViewPaired",
    "WordsViewSided",
    "set_repr_hints",
]
