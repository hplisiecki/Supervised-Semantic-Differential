"""`_SingleResult` — leaf base for one gradient's views and derived arrays.

Holds everything derivable from ``(beta, x, embeddings, corpus, lexicon,
window, sif_a, lang)``. Key-agnostic: knows nothing about whether it lives
standalone (as ``ContinuousResult``) or inside a container (as ``PairResult``).

Lazy views (`.words`, `.clusters`, `.snippets`, `cluster_snippets`) and the
cached `.alignment_scores` live here. Subclass-specific state — y/outcome-scale
stats, rerunnable `.test`, `.docs`, `fit_info`, `.report()` — stays on
subclasses (`ContinuousResult`, `PairResult`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ssdiff.results.core import Result
from ssdiff.utils.math import unit_vector


_SNIPPET_EXTRACTION_KWARGS = {"top_per_side", "min_cosine", "n_jobs"}


class _SingleResult(Result):
    """Leaf base: one gradient direction + its derived views/arrays.

    Subclasses are responsible for setting ``self.stats`` and (optionally)
    ``self.test``; this base handles only what's derivable from
    ``(x, beta, embeddings, corpus, …)``.
    """

    def __init__(
        self,
        *,
        x: np.ndarray,
        beta: np.ndarray,
        embeddings: Any = None,
        corpus: Any = None,
        lexicon: set | None = None,
        window: int = 3,
        sif_a: float = 1e-3,
        lang: str | None = None,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.corpus = corpus
        self.lexicon = set(lexicon) if lexicon else set()
        self.window = int(window)
        self.sif_a = float(sif_a)
        self.lang = lang

        # Keep raw x; derive gradient/beta_norm up front (cheap).
        self._x = x
        self.beta = np.asarray(beta, dtype=float)
        self.gradient = unit_vector(self.beta)
        self.beta_norm = float(np.linalg.norm(self.beta))

        # Eager: ClustersView is a thin wrapper that defers all real work
        # to _clusters_for (which caches via self._cache). Assigning here
        # keeps caching discipline consistent with every other view.
        from ssdiff.results.continuous_result import ClustersView
        self.clusters = ClustersView(self)

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def alignment_scores(self) -> np.ndarray:
        """Per-document cosine alignment ``d_i · gradient`` (shape ``(n,)``).

        Cached by reference — read-only once computed.
        """
        cached = getattr(self, "_alignment_scores_cache", None)
        if cached is not None:
            return cached
        x = self.x
        x_norms = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        out = ((x / x_norms) @ self.gradient).ravel()
        out.setflags(write=False)
        self._alignment_scores_cache = out
        return out

    @property
    def words(self):
        from ssdiff.results.continuous_result import WordsView

        key = ("words", ())
        if key in self._cache:
            return self._cache[key]
        self._require_resource("embeddings", "words")
        rows = self._compute_words_rows()
        view = WordsView(rows)
        self._cache[key] = view
        return view

    def _compute_words_rows(self, *, contrast: str | None = None):
        from ssdiff.results.schema import Word
        from ssdiff.utils.neighbors import filtered_neighbors

        out: list[Word] = []
        lang = self.lang or "pl"
        for side, vec, sign in [("pos", self.gradient, 1.0),
                                 ("neg", -self.gradient, -1.0)]:
            rank = 0
            for word, cos in filtered_neighbors(
                self.embeddings, vec, topn=100, lang=lang,
            ):
                signed_cos = float(cos) * sign
                if (side == "pos" and signed_cos < 0) or \
                   (side == "neg" and signed_cos > 0):
                    continue
                rank += 1
                out.append(Word(
                    side=side, rank=rank, word=word,
                    cos_beta=signed_cos, contrast=contrast,
                ))
        return out

    def _clusters_for(self, side: str, **params):
        from ssdiff.results.continuous_result import ClustersViewSided

        defaults = {"topn": 100, "k": None, "k_min": 2, "k_max": 10,
                    "random_state": 2137, "min_cluster_size": 2}
        params = {**defaults, **params, "side": side}

        def _compute():
            self._require_resource("embeddings", "clusters")
            rows, words_rows = self._compute_clusters_for_side(**params)
            return ClustersViewSided(
                parent=self, side=side, rows=rows, words_rows=words_rows,
                params=params,
            )
        return self._cache_get("clusters", params, _compute)

    def _compute_clusters_for_side(
        self, *, side, topn, k, k_min, k_max, random_state, min_cluster_size,
        contrast: str | None = None,
    ):
        from ssdiff.results.schema import Cluster, ClusterWord
        from ssdiff.utils.neighbors import cluster_top_neighbors

        raw = cluster_top_neighbors(
            self.embeddings, self.gradient, topn=topn, k=k,
            k_min=k_min, k_max=k_max, random_state=random_state,
            min_cluster_size=min_cluster_size, side=side, lang=self.lang or "pl",
        )
        rows: list[Cluster] = []
        words_rows: list[ClusterWord] = []
        for c in raw:
            rows.append(Cluster(
                cluster_id=int(c["id"]), side=side, size=int(c["size"]),
                coherence=float(c["coherence"]),
                centroid_cos_beta=float(c["centroid_cos_beta"]),
                contrast=contrast,
            ))
            for w in c["words"]:
                words_rows.append(ClusterWord(
                    cluster_id=int(c["id"]), side=side, word=w["word"],
                    cos_centroid=float(w.get("cos_centroid", 0.0)),
                    cos_beta=float(w["cos_beta"]),
                    contrast=contrast,
                ))
        return rows, words_rows

    @property
    def snippets(self):
        return self._snippets_for(top_per_side=30)

    def _snippets_for(self, **params):
        from ssdiff.results.continuous_result import SnippetsView

        defaults = {"top_per_side": 30}
        extraction_params = {
            k: v for k, v in params.items() if k in _SNIPPET_EXTRACTION_KWARGS
        }
        params = {**defaults, **extraction_params}

        def _compute():
            self._require_resource("corpus", "snippets")
            self._require_resource("embeddings", "snippets")
            return SnippetsView(
                self._compute_snippets_rows(**params),
                params=params, parent=self,
            )
        return self._cache_get("snippets", params, _compute)

    def _compute_snippets_rows(self, *, contrast: str | None = None, **params):
        from ssdiff.results.schema import Snippet
        from ssdiff.utils.snippets import snippets_along_beta

        out = snippets_along_beta(
            pre_docs=self.corpus.pre_docs,
            ssd=self,
            token_window=self.window,
            seeds=self.lexicon or None,
            sif_a=self.sif_a,
            top_per_side=params.get("top_per_side", 30),
            min_cosine=params.get("min_cosine"),
            n_jobs=params.get("n_jobs", -1),
            verbose=False,
        )
        rows: list[Snippet] = []
        sid = 0
        for side in ("pos", "neg"):
            for d in out[side]:
                rows.append(Snippet(
                    snippet_id=sid, side=side,
                    doc_id=int(d["profile_id"]),
                    cosine=float(d["cosine"]),
                    seed=d["seed"],
                    start_token_idx=int(d["start_token_idx"]),
                    end_token_idx=int(d["end_token_idx"]),
                    start_sent_idx=int(d["start_sent_idx"]),
                    end_sent_idx=int(d["end_sent_idx"]),
                    text_window=d["snippet_anchor"],
                    text_surface=d["essay_text_surface"],
                    text_lemmas=d["essay_text_lemmas"],
                    cluster_id=None,
                    contrast=contrast,
                    post_id=d.get("post_id"),
                ))
                sid += 1
        return rows

    def _cluster_snippets_for(
        self, side: str, *,
        top_per_cluster: int = 100,
        min_cosine: float | None = None,
        n_jobs: int = -1,
        contrast: str | None = None,
        **cluster_params,
    ):
        from types import SimpleNamespace

        from ssdiff.results.continuous_result import SnippetsViewSided
        from ssdiff.results.schema import Snippet
        from ssdiff.utils.snippets import cluster_snippets_by_centroids

        cluster_view = self._clusters_for(side, **cluster_params)
        effective_cluster_params = {
            k: v for k, v in cluster_view._params.items() if k != "side"
        }
        cache_params = {
            "side": side,
            "top_per_cluster": top_per_cluster,
            "min_cosine": min_cosine,
            "n_jobs": n_jobs,
            **effective_cluster_params,
        }

        def _compute():
            self._require_resource("corpus", "cluster_snippets")
            self._require_resource("embeddings", "cluster_snippets")
            words_by_cid: dict[int, list[dict]] = {}
            for cw in cluster_view._words_rows:
                words_by_cid.setdefault(cw.cluster_id, []).append({"word": cw.word})
            clusters_arg = [
                {"words": words_by_cid.get(c.cluster_id, [])}
                for c in cluster_view._rows
            ]
            rank_to_cid = {
                i + 1: c.cluster_id for i, c in enumerate(cluster_view._rows)
            }

            shim = SimpleNamespace(
                embeddings=self.embeddings,
                gradient=self.gradient,
                beta=self.gradient,
                lexicon=self.lexicon,
                window=self.window,
                sif_a=self.sif_a,
            )
            pos_arg = clusters_arg if side == "pos" else None
            neg_arg = clusters_arg if side == "neg" else None

            out = cluster_snippets_by_centroids(
                pre_docs=self.corpus.pre_docs, ssd=shim,
                pos_clusters=pos_arg, neg_clusters=neg_arg,
                token_window=self.window, seeds=self.lexicon or None,
                sif_a=self.sif_a, top_per_cluster=top_per_cluster,
                n_jobs=n_jobs, verbose=False,
            )
            side_rows = out.get(side, [])
            if min_cosine is not None:
                side_rows = [d for d in side_rows if d["cosine"] >= min_cosine]

            rows: list[Snippet] = []
            for sid, d in enumerate(side_rows):
                rank = int(d["centroid_label"].rsplit("_", 1)[-1])
                rows.append(Snippet(
                    snippet_id=sid, side=side,
                    doc_id=int(d["profile_id"]),
                    cosine=float(d["cosine"]),
                    seed=d["seed"],
                    start_token_idx=int(d["start_token_idx"]),
                    end_token_idx=int(d["end_token_idx"]),
                    start_sent_idx=int(d["start_sent_idx"]),
                    end_sent_idx=int(d["end_sent_idx"]),
                    text_window=d["snippet_anchor"],
                    text_surface=d["essay_text_surface"],
                    text_lemmas=d["essay_text_lemmas"],
                    cluster_id=rank_to_cid.get(rank),
                    contrast=contrast,
                    post_id=d.get("post_id"),
                ))
            return SnippetsViewSided(side=side, all_rows=rows)

        return self._cache_get("cluster_snippets", cache_params, _compute)

    def cluster_snippets(
        self, *, side: str,
        top_per_cluster: int = 100,
        min_cosine: float | None = None,
        n_jobs: int = -1,
        **cluster_params,
    ):
        return self._cluster_snippets_for(
            side=side, top_per_cluster=top_per_cluster,
            min_cosine=min_cosine, n_jobs=n_jobs, **cluster_params,
        )
