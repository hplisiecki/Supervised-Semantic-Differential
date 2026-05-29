"""Utility helpers shared across ssdiff.

Re-exports the two most commonly needed internal utilities:
``_diagnostic`` (conditional print) and ``_progress`` (tqdm + hook wrapper).
"""

from ssdiff.utils.diagnostics import _diagnostic, _progress

__all__ = ["_diagnostic", "_progress"]
