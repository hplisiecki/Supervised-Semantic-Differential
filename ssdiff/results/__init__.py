"""Public API of the ``ssdiff.results`` package.

Re-exports the result classes (``PLSResult``, ``PCAOLSResult``,
``GroupResult``, ``LexiconResult``, ``UMAPVisResult``), their shared base
``Result``, the sided view classes (``WordsViewSided``, ``ClustersViewSided``,
``SnippetsViewSided``), and the ``set_repr_hints`` display toggle.
"""

from ssdiff.results.continuous_result import (
    ClustersView,
    ClustersViewSided,
    ClusterWordsView,
    ClusterWordsViewSided,
    ContinuousResult,
    PCAOLSResult,
    PLSResult,
    RegressionResult,
    SnippetsView,
    SnippetsViewSided,
    WordsView,
    WordsViewSided,
)
from ssdiff.results.contrast_result import ContrastResult, contrast
from ssdiff.results.core import Result
from ssdiff.results.display import set_repr_hints
from ssdiff.results.group_result import GroupResult, PairResult
from ssdiff.results.lexicon_result import LexiconResult
from ssdiff.results.multi_pls_result import (
    MultiPLSResult,
)
from ssdiff.results.multi_pls_result import (
    _PLSComponentResult as PLSComponentResult,
)

__all__ = [
    "ClusterWordsView",
    "ClusterWordsViewSided",
    "ClustersView",
    "ClustersViewSided",
    "ContinuousResult",
    "ContrastResult",
    "GroupResult",
    "LexiconResult",
    "MultiPLSResult",
    "PairResult",
    "PLSComponentResult",
    "PCAOLSResult",
    "PLSResult",
    "RegressionResult",
    "Result",
    "SnippetsView",
    "SnippetsViewSided",
    "WordsView",
    "WordsViewSided",
    "contrast",
    "set_repr_hints",
]
