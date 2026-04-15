"""Verbose output: tqdm progress bars, diagnostic prints, and progress hook."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from typing import TypeVar

_T = TypeVar("_T")

# ---------------------------------------------------------------------------
# Progress hook (thread-local, for GUI integration)
# ---------------------------------------------------------------------------

_local = threading.local()


@contextmanager
def progress_hook(callback: Callable[[int, int, str], None]):
    """Set a progress callback for the duration of the block.

    Parameters
    ----------
    callback : (current, total, desc) -> None
        Called on each iteration of a progress-tracked loop.
        *current* and *total* are iteration counts; *desc* is a
        human-readable phase label (e.g. ``"Permutation test"``).

    Examples
    --------
    >>> from ssdiff import progress_hook
    >>> def on_progress(current, total, desc):
    ...     print(f"{desc}: {current}/{total}")
    >>> with progress_hook(on_progress):
    ...     result = ssd.fit_pls(n_perm=500)
    """
    prev = getattr(_local, "callback", None)
    _local.callback = callback
    try:
        yield
    finally:
        _local.callback = prev


def _get_hook() -> Callable[[int, int, str], None] | None:
    """Return the active progress callback, or None."""
    return getattr(_local, "callback", None)


# ---------------------------------------------------------------------------
# Progress wrapper
# ---------------------------------------------------------------------------

def _hooked_iter(
    iterable: Iterable[_T],
    total: int | None,
    desc: str,
    hook: Callable[[int, int, str], None],
) -> Iterable[_T]:
    """Wrap *iterable* to fire *hook(current, total, desc)* on each step."""
    t = total or 0
    for i, item in enumerate(iterable, 1):
        yield item
        hook(i, t, desc)


def _progress(
    iterable: Iterable[_T],
    *,
    verbose: bool = False,
    total: int | None = None,
    desc: str = "",
) -> Iterable[_T]:
    """Wrap *iterable* in a tqdm progress bar and/or fire the progress hook.

    - If a ``progress_hook`` callback is active, fires it on each iteration
      (regardless of *verbose*).
    - If *verbose* is True, also wraps in ``tqdm.auto`` (falls back to bare
      iterable if tqdm is not installed).
    """
    hook = _get_hook()
    if hook is not None:
        iterable = _hooked_iter(iterable, total, desc, hook)
    if not verbose:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


# ---------------------------------------------------------------------------
# Diagnostic print
# ---------------------------------------------------------------------------

def _diagnostic(verbose: bool, message: str) -> None:
    """Print a diagnostic message when *verbose* is True."""
    if verbose:
        print(message)
