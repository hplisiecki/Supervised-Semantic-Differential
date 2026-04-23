"""`_MultiContainer` — key-opaque container of leaf ``_SingleResult``s.

Holds ``dict[Hashable, _SingleResult]`` + aggregate shims. Knows nothing about
pair semantics (canonical ordering, numeric-g sort) — subclasses plug those
in via ``_key_to_str`` / ``_key_repr`` hooks and their own ``__getitem__``
override if needed.

Shim views (``.words`` / ``.clusters`` / ``.snippets``) are pure dict
wrappers: single-leaf access by key, no flat row iteration, no summed ``__len__``.
"""

from __future__ import annotations

from typing import Any, Hashable

from ssdiff.results.core import Result


class _ShimView:
    """Dict wrapper around per-leaf views.

    Parameters
    ----------
    leaves : dict[Hashable, View]
        Per-leaf view instance keyed by container key.
    view_name : str
        Logical name (``"words"`` / ``"clusters"`` / ``"snippets"``) — used for
        save fan-out filenames and ``__repr__``.
    container : _MultiContainer
        Source container — consulted for ``_key_to_str`` / ``_key_repr`` hooks.
    """

    def __init__(
        self,
        *,
        leaves: dict[Hashable, Any],
        view_name: str,
        container: "_MultiContainer",
    ):
        self._leaves = dict(leaves)
        self._view_name = view_name
        self._container = container

    _LEAF_TYPE_NAMES = {
        "words": "WordsView",
        "clusters": "ClustersView",
        "snippets": "SnippetsView",
    }

    def __getitem__(self, key):
        if key not in self._leaves:
            raise KeyError(key)
        return self._leaves[key]

    def keys(self) -> list:
        return list(self._leaves.keys())

    def __len__(self) -> int:
        return len(self._leaves)

    def _leaf_type_name(self) -> str:
        return self._LEAF_TYPE_NAMES.get(self._view_name, "per-pair view")

    def to_text(self) -> str:
        cont_name = type(self._container).__name__
        n = len(self._leaves)
        header = f"{cont_name}.{self._view_name} shim — {n} pair view(s)"
        if not self._leaves:
            return header + "\n\n(no pairs)"
        return header + "\n\nPairs:\n" + self._format_keys()

    def _format_keys(self) -> str:
        reprs = [repr(k) for k in self._leaves]
        per_row = 4
        rows = [
            "  " + "   ".join(reprs[i:i + per_row])
            for i in range(0, len(reprs), per_row)
        ]
        return "\n".join(rows)

    def _save_hint(self) -> str:
        first = next(iter(self._leaves), None)
        leaf_type = self._leaf_type_name()
        save_line = (
            f"Save:   .save('{self._view_name}.csv')   "
            f"→ per-pair fan-out (one file per pair)"
        )
        if first is None:
            return save_line
        fk = repr(first)
        return (
            f"Zoom:   result[{fk}].{self._view_name}   → {leaf_type} "
            f"(canonical)\n"
            f"        this_shim[{fk}]             → same "
            f"(power-user shortcut)\n"
            + save_line
        )

    def _save_hint_html(self) -> str:
        import html as _h
        return f"<pre class='ssd-save-hint'>{_h.escape(self._save_hint())}</pre>"

    def __repr__(self) -> str:
        from ssdiff.results.display import _save_hint_enabled
        body = self.to_text()
        if _save_hint_enabled():
            return body + "\n\n" + self._save_hint()
        return body

    def _repr_html_(self) -> str:
        from ssdiff.results.display import _save_hint_enabled
        body = f"<pre>{self.to_text()}</pre>"
        if _save_hint_enabled():
            return body + "\n" + self._save_hint_html()
        return body

    def save(self, path=None, *, cols=None, k: int | None = None) -> None:
        """Write the shim to ``path``; delegates to ``_paired_save`` helper."""
        from ssdiff.results.paired_view import _paired_save
        _paired_save(
            view_name=self._view_name,
            views=self._leaves,
            path=path,
            cols=cols,
            k=k,
            key_to_str=self._container._key_to_str,
            key_heading=self._container._key_repr,
        )


class _MultiContainer(Result):
    """Keyed collection of ``_SingleResult`` leaves + aggregate shim views.

    Subclasses provide:
    - ``self._leaves : dict[Hashable, _SingleResult]`` (set in ``__init__``)
    - ``.stats``, ``.test`` — omnibus stats + the one rerunnable test
    - ``_key_to_str(key) -> str`` — filename / sheet / JSON-key form
    - ``_key_repr(key) -> str`` — human-readable form for headings / repr
    - per-subclass ``report()`` assembly
    """

    _leaves: dict[Hashable, Any]

    def _key_to_str(self, key: Hashable) -> str:
        return str(key)

    def _key_repr(self, key: Hashable) -> str:
        return repr(key)

    def __getitem__(self, key):
        return self._leaves[key]

    def keys(self) -> list:
        return list(self._leaves.keys())

    def __len__(self) -> int:
        return len(self._leaves)

    @property
    def beta(self) -> dict:
        return {k: leaf.beta for k, leaf in self._leaves.items()}

    @property
    def gradient(self) -> dict:
        return {k: leaf.gradient for k, leaf in self._leaves.items()}

    @property
    def beta_norm(self) -> dict:
        return {k: leaf.beta_norm for k, leaf in self._leaves.items()}

    @property
    def alignment_scores(self) -> dict:
        return {k: leaf.alignment_scores for k, leaf in self._leaves.items()}

    @property
    def words(self) -> _ShimView:
        return _ShimView(
            leaves={k: leaf.words for k, leaf in self._leaves.items()},
            view_name="words", container=self,
        )

    @property
    def clusters(self) -> _ShimView:
        return _ShimView(
            leaves={k: leaf.clusters for k, leaf in self._leaves.items()},
            view_name="clusters", container=self,
        )

    @property
    def snippets(self) -> _ShimView:
        return _ShimView(
            leaves={k: leaf.snippets for k, leaf in self._leaves.items()},
            view_name="snippets", container=self,
        )
