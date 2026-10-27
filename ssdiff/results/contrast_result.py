"""`ContrastResult` — the direction in which two gradients differ, with words and clusters only."""

from __future__ import annotations

import numpy as np

from ssdiff.results.core import ScalarView
from ssdiff.results.single_result import _SingleResult
from ssdiff.utils.math import unit_vector


class ContrastStatsView(ScalarView):
    """Scalar view: cosine of the two gradients and the length of their unit difference."""

    _name = "stats"
    _columns = ("cos_g1_g2", "diff_norm")

    def __init__(self, *, cos_g1_g2: float, diff_norm: float):
        super().__init__()
        self._row = {"cos_g1_g2": float(cos_g1_g2), "diff_norm": float(diff_norm)}

    def __iter__(self):
        yield dict(self._row)


class ContrastResult(_SingleResult):
    """``unit(g1) − unit(g2)``, normalized, read in the embeddings.

    Has no documents, so it offers ``words`` and ``clusters`` but no ``snippets``,
    ``docs`` or ``alignment_scores``. No test: ``cos_g1_g2`` is the only
    statement about how the two gradients relate.
    """

    _access = ("stats", "words", "clusters")
    _arrays = ("beta", "gradient")

    def __init__(self, g1, g2, embeddings, *, labels=("g1", "g2")):
        g1 = np.asarray(g1, dtype=float).ravel()
        g2 = np.asarray(g2, dtype=float).ravel()
        if g1.shape != g2.shape:
            raise ValueError(f"g1 and g2 must have the same length, got {g1.size} and {g2.size}.")
        if len(labels) != 2:
            raise ValueError("labels must hold two names, one per gradient.")
        u1, u2 = unit_vector(g1), unit_vector(g2)
        diff = u1 - u2
        diff_norm = float(np.linalg.norm(diff))
        if diff_norm < 1e-12:
            raise ValueError("g1 and g2 are the same direction (or zero): the contrast is empty.")
        super().__init__(x=None, beta=diff / diff_norm, embeddings=embeddings)
        self.labels = (str(labels[0]), str(labels[1]))
        self.stats = ContrastStatsView(cos_g1_g2=float(u1 @ u2), diff_norm=diff_norm)

    @property
    def snippets(self):
        raise RuntimeError("A contrast has no documents, so it has no snippets.")

    def cluster_snippets(self, *args, **kwargs):
        raise RuntimeError("A contrast has no documents, so it has no snippets.")

    @property
    def docs(self):
        raise RuntimeError("A contrast has no documents.")

    @property
    def alignment_scores(self):
        raise RuntimeError("A contrast has no documents, so it has no alignment scores.")

    def _pole_names(self) -> tuple[str, str]:
        return f"toward {self.labels[0]}", f"toward {self.labels[1]}"

    def _summary(self) -> str:
        from ssdiff.results.format import fmt_r
        s = self.stats._row
        return (
            f"ContrastResult  {self.labels[0]} − {self.labels[1]}  "
            f"cos(g1, g2)={fmt_r(s['cos_g1_g2'])}  diff_norm={s['diff_norm']:.3f}"
        )

    def _summary_html(self) -> str:
        return f"<p>{self._summary()}</p>"

    def _save_hint(self) -> str:
        return (
            "Save:  c.words.save('words.csv')\n"
            "       c.save_page('contrast.html')"
        )

    def _save_hint_html(self) -> str:
        return f"<pre class='ssd-save-hint'>{self._save_hint()}</pre>"


def contrast(g1, g2, embeddings, *, labels=("g1", "g2")) -> ContrastResult:
    """The difference of two gradients, as a gradient: ``unit(g1) − unit(g2)``, normalized.

    ``g1`` and ``g2`` are gradient vectors (e.g. ``r.gradient``) in the space of
    ``embeddings``; unit vectors keep beta scale out of the difference.
    ``labels`` names the two gradients on the results page.
    """
    return ContrastResult(g1, g2, embeddings, labels=labels)
