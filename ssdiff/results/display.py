"""Display-layer scaffolding for ssdiff results.

Holds the module-level ``set_repr_hints`` flag and the
``_save_hint_enabled`` helper used by ``__repr__`` / ``_repr_html_``
across ``Result``, ``View``, ``ScalarView``, and ``Report``.
"""

from __future__ import annotations


_REPR_HINTS_ENABLED: bool = True


def set_repr_hints(enabled: bool) -> None:
    """Toggle save-hint footers in __repr__ / _repr_html_ output.

    Default: True. Disable for log-stream consumers (SSD_APP).
    """
    if not isinstance(enabled, bool):
        raise TypeError(f"enabled must be bool, got {type(enabled).__name__}")
    global _REPR_HINTS_ENABLED
    _REPR_HINTS_ENABLED = enabled


def _save_hint_enabled() -> bool:
    return _REPR_HINTS_ENABLED
