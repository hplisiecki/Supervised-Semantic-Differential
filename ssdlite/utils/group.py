"""SSDGroup + SSDContrast: categorical group analysis via centroid contrasts."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ssdlite.embeddings import Embeddings
from ssdlite.corpus import Corpus
from ssdlite.utils.math import unit_vector
from ssdlite.utils.vectors import build_and_normalize_doc_vectors
from ssdlite.utils.neighbors import filtered_neighbors, cluster_top_neighbors


class SSDGroup:
    """Categorical group SSD — permutation-based pairwise contrasts.

    >>> emb = Embeddings.load("model.ssdembed")
    >>> corpus = Corpus(texts, lang="pl")
    >>> sg = SSDGroup(emb, corpus, groups, lexicon)
    >>> sg.omnibus_p
    >>> contrast = sg.get_contrast("A", "B")
    """

    def __init__(
        self,
        embeddings: Embeddings,
        corpus: Corpus,
        groups,
        lexicon,
        *,
        n_perm: int = 5000,
        random_state: int = 42,
        window: int = 3,
        sif_a: float = 1e-3,
        use_full_doc: bool = False,
    ) -> None:
        """Fit categorical group SSD with permutation-based significance tests.

        Parameters
        ----------
        embeddings : Embeddings
            Word embeddings used for vector construction.
        corpus : Corpus
            Tokenized corpus of documents.
        groups : array-like
            Group labels per document (same length as *corpus*).
        lexicon : sequence or set of str
            Seed words defining the semantic dimension.
        n_perm : int, optional
            Number of permutations for significance tests, by default 5000.
        random_state : int, optional
            Random seed for reproducibility, by default 42.
        window : int, optional
            Context window (in tokens) around seed words, by default 3.
        sif_a : float, optional
            SIF smoothing parameter, by default 1e-3.
        use_full_doc : bool, optional
            If ``True``, use full-document vectors instead of
            seed-windowed contexts, by default ``False``.

        Raises
        ------
        ValueError
            If ``len(groups) != len(corpus)`` or fewer than 2 groups
            remain after filtering invalid labels.
        """
        self.kv = embeddings
        self.lexicon = set(lexicon)
        self.window = window
        self.sif_a = sif_a
        self.lang = getattr(corpus, "lang", None) or "pl"

        docs = corpus.docs
        groups_raw = np.asarray(groups, dtype=object)
        if len(groups_raw) != len(docs):
            raise ValueError(f"len(groups)={len(groups_raw)} != len(docs)={len(docs)}")

        # Drop invalid group labels
        group_valid = np.array([
            g is not None and g != "" and (not isinstance(g, float) or np.isfinite(g))
            for g in groups_raw
        ], dtype=bool)

        if not group_valid.all():
            docs = [d for d, v in zip(docs, group_valid) if v]
            groups_raw = groups_raw[group_valid]

        # Build doc vectors
        X, keep = build_and_normalize_doc_vectors(
            docs, embeddings, self.lexicon,
            window=window, sif_a=sif_a, use_full_doc=use_full_doc,
        )

        self.keep_mask = keep
        self.n_raw = len(keep)
        self.n_kept = int(keep.sum())
        self.n_dropped = self.n_raw - self.n_kept
        self.x = np.asarray(X, dtype=np.float64)

        # Apply keep_mask to groups
        self.groups_kept = groups_raw[keep]
        self.group_labels = sorted(set(self.groups_kept))
        self.G = len(self.group_labels)

        if self.G < 2:
            raise ValueError(f"Need at least 2 groups after filtering, got {self.G}")

        self.n_perm = int(n_perm)
        self._rng = np.random.default_rng(random_state)

        # Compute group centroids
        self.centroids = self._compute_centroids(self.x, self.groups_kept)

        # Permutation tests
        if self.G == 2:
            self.pairwise = self._pairwise_tests()
            only = list(self.pairwise.values())[0]
            self.omnibus_T = only["T"]
            self.omnibus_p = only["p_raw"]
            self.omnibus_null = only["null_dist"]
        else:
            self.omnibus_T, self.omnibus_p, self.omnibus_null = (
                self._omnibus_permutation_test()
            )
            self.pairwise = self._pairwise_tests()

    def _compute_centroids(self, X, groups) -> dict:
        centroids = {}
        for g in self.group_labels:
            c = X[groups == g].mean(axis=0)
            centroids[g] = unit_vector(c)
        return centroids

    def _compute_centroids_matrix(self, X, group_idx, G):
        D = X.shape[1]
        counts = np.bincount(group_idx, minlength=G)
        centroids = np.zeros((G, D), dtype=np.float64)
        for dim in range(D):
            centroids[:, dim] = np.bincount(group_idx, weights=X[:, dim], minlength=G)
        mask = counts > 0
        centroids[mask] /= counts[mask, np.newaxis]
        norms = np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
        centroids /= norms
        return centroids

    @staticmethod
    def _cosine_distance(a, b) -> float:
        return 1.0 - float(np.clip(np.dot(a, b), -1.0, 1.0))

    def _compute_omnibus_T(self, X, groups) -> float:
        centroids = self._compute_centroids(X, groups)
        pairs = list(combinations(self.group_labels, 2))
        if not pairs:
            return 0.0
        return float(np.mean([self._cosine_distance(centroids[a], centroids[b]) for a, b in pairs]))

    def _omnibus_T_from_matrix(self, centroid_matrix, pair_indices) -> float:
        if len(pair_indices) == 0:
            return 0.0
        dots = np.sum(
            centroid_matrix[pair_indices[:, 0]] * centroid_matrix[pair_indices[:, 1]], axis=1
        )
        return float(np.mean(1.0 - np.clip(dots, -1.0, 1.0)))

    def _omnibus_permutation_test(self):
        """Omnibus permutation test for multi-group differences.

        Test statistic *T* is the mean pairwise cosine distance between
        group centroids.  The null distribution is built by shuffling
        group labels and recomputing *T* for each permutation.

        Returns
        -------
        T_obs : float
            Observed test statistic.
        p_value : float
            Proportion of null values >= *T_obs*.
        null_dist : np.ndarray
            Array of *n_perm* null test-statistic values.
        """
        T_obs = self._compute_omnibus_T(self.x, self.groups_kept)
        label_to_idx = {g: i for i, g in enumerate(self.group_labels)}
        group_idx = np.array([label_to_idx[g] for g in self.groups_kept], dtype=np.intp)
        G = len(self.group_labels)
        pair_indices = np.array(list(combinations(range(G), 2)), dtype=np.intp)

        null_dist = np.empty(self.n_perm, dtype=np.float64)
        for i in range(self.n_perm):
            self._rng.shuffle(group_idx)
            centroids = self._compute_centroids_matrix(self.x, group_idx, G)
            null_dist[i] = self._omnibus_T_from_matrix(centroids, pair_indices)

        p_value = float((np.sum(null_dist >= T_obs) + 1) / (self.n_perm + 1))
        return T_obs, p_value, null_dist

    def _pairwise_tests(self) -> dict:
        """Pairwise permutation tests with Bonferroni correction.

        For each group pair, computes cosine distance between centroids
        as the test statistic, builds a null distribution via label
        permutation, and derives Cohen's *d* from projections onto the
        contrast vector.  P-values are Bonferroni-corrected by
        multiplying by the number of pairs.

        Returns
        -------
        dict
            Mapping ``(group_a, group_b)`` to result dicts with keys:
            ``T``, ``p_raw``, ``p_corrected``, ``null_dist``,
            ``contrast_raw``, ``contrast_unit``, ``contrast_norm``,
            ``cohens_d``, ``n_g1``, ``n_g2``.
        """
        results = {}
        pairs = list(combinations(self.group_labels, 2))
        n_pairs = len(pairs)

        for g1, g2 in pairs:
            mask = (self.groups_kept == g1) | (self.groups_kept == g2)
            X_pair = self.x[mask]
            g_pair = self.groups_kept[mask]

            # Observed test statistic
            c1 = unit_vector(X_pair[g_pair == g1].mean(axis=0))
            c2 = unit_vector(X_pair[g_pair == g2].mean(axis=0))
            T_obs = 1.0 - float(np.clip(np.dot(c1, c2), -1.0, 1.0))

            # Permutation
            n_g1 = int((g_pair == g1).sum())
            n_pair = len(g_pair)
            null_dist = np.empty(self.n_perm, dtype=np.float64)
            for i in range(self.n_perm):
                perm_idx = self._rng.permutation(n_pair)
                pc1 = unit_vector(X_pair[perm_idx[:n_g1]].mean(axis=0))
                pc2 = unit_vector(X_pair[perm_idx[n_g1:]].mean(axis=0))
                null_dist[i] = 1.0 - float(np.clip(np.dot(pc1, pc2), -1.0, 1.0))

            p_raw = float((np.sum(null_dist >= T_obs) + 1) / (self.n_perm + 1))

            # Contrast vector
            contrast_raw = self.centroids[g1] - self.centroids[g2]
            contrast_norm = float(np.linalg.norm(contrast_raw))
            contrast_unit = unit_vector(contrast_raw)

            # Cohen's d
            proj = (self.x @ contrast_unit).ravel()
            proj_g1 = proj[self.groups_kept == g1]
            proj_g2 = proj[self.groups_kept == g2]
            dof = len(proj_g1) + len(proj_g2) - 2
            if dof > 0:
                pooled_std = np.sqrt(
                    ((len(proj_g1) - 1) * np.var(proj_g1, ddof=1) +
                     (len(proj_g2) - 1) * np.var(proj_g2, ddof=1))
                    / dof
                )
            else:
                pooled_std = 0.0
            cohens_d = (np.mean(proj_g1) - np.mean(proj_g2)) / max(pooled_std, 1e-12)

            results[(g1, g2)] = {
                "T": T_obs,
                "p_raw": p_raw,
                "p_corrected": min(p_raw * n_pairs, 1.0),
                "null_dist": null_dist,
                "contrast_raw": contrast_raw,
                "contrast_unit": contrast_unit,
                "contrast_norm": contrast_norm,
                "cohens_d": float(cohens_d),
                "n_g1": int(np.sum(self.groups_kept == g1)),
                "n_g2": int(np.sum(self.groups_kept == g2)),
            }

        return results

    def get_contrast(self, group_a, group_b) -> SSDContrast:
        """Return an SSDContrast for a pair of groups.

        Parameters
        ----------
        group_a, group_b
            Group labels.  Order determines the contrast direction:
            the contrast vector points from *group_b* toward *group_a*
            (i.e. A minus B).

        Returns
        -------
        SSDContrast
            Pairwise contrast object with interpretation methods
            (neighbors, clustering, summary).

        Raises
        ------
        KeyError
            If the group pair was not found among computed pairwise
            contrasts.
        """
        key = (group_a, group_b)
        flipped = False
        if key not in self.pairwise:
            key = (group_b, group_a)
            flipped = True
            if key not in self.pairwise:
                raise KeyError(f"Pair ({group_a}, {group_b}) not found.")

        r = self.pairwise[key]
        contrast_unit = -r["contrast_unit"] if flipped else r["contrast_unit"]
        contrast_raw = -r["contrast_raw"] if flipped else r["contrast_raw"]

        perm_result = dict(r)
        if flipped:
            perm_result["cohens_d"] = -perm_result["cohens_d"]

        mask_pair = (self.groups_kept == group_a) | (self.groups_kept == group_b)

        return SSDContrast(
            kv=self.kv,
            beta=contrast_raw,
            beta_unit=contrast_unit,
            x=self.x,
            x_pair=self.x[mask_pair],
            groups_kept=self.groups_kept,
            groups_pair=self.groups_kept[mask_pair],
            group_a=group_a,
            group_b=group_b,
            lexicon=self.lexicon,
            window=self.window,
            sif_a=self.sif_a,
            lang=self.lang,
            perm_result=perm_result,
        )

    def contrast_scores(self, group_a, group_b) -> dict:
        """Project all kept participants onto the (A-B) contrast vector.

        Parameters
        ----------
        group_a, group_b
            Group labels defining the contrast direction (A minus B).

        Returns
        -------
        dict
            ``"group"`` : list of group labels for each kept participant.
            ``"cos_to_contrast"`` : list of float cosine similarity scores
            between each participant's vector and the contrast vector.
        """
        contrast = self.get_contrast(group_a, group_b)
        bu = contrast.beta_unit
        x_norms = np.maximum(np.sqrt(np.einsum("ij,ij->i", self.x, self.x))[:, None], 1e-12)
        x_unit = self.x / x_norms
        cos_vals = (x_unit @ bu).ravel()
        return {
            "group": self.groups_kept.tolist(),
            "cos_to_contrast": cos_vals.tolist(),
        }

    def results_table(self) -> list[dict]:
        """Pairwise results as a list of dicts.

        Returns
        -------
        list[dict]
            Each dict contains keys: ``group_A``, ``group_B``, ``n_A``,
            ``n_B``, ``cosine_distance``, ``p_raw``, ``p_corrected``
            (Bonferroni), ``cohens_d``, ``contrast_norm``.
        """
        rows = []
        for (g1, g2), r in self.pairwise.items():
            rows.append({
                "group_A": g1, "group_B": g2,
                "n_A": r["n_g1"], "n_B": r["n_g2"],
                "cosine_distance": r["T"],
                "p_raw": r["p_raw"], "p_corrected": r["p_corrected"],
                "cohens_d": r["cohens_d"], "contrast_norm": r["contrast_norm"],
            })
        return rows

    def summary(self) -> str:
        """Human-readable group analysis summary."""
        labels_str = ", ".join(str(g) for g in self.group_labels)
        title = "SSDGroup Summary"
        sep = "─" * len(title)
        lines = [title, sep]
        lines.append(f"Groups: {self.G} ({labels_str})   Docs: {self.n_kept} kept / {self.n_raw} total")
        lines.append(f"Omnibus: T = {self.omnibus_T:.4f}   p = {self.omnibus_p:.4f}")
        lines.append("")
        lines.append("Pairwise:")
        for (g1, g2), r in self.pairwise.items():
            lines.append(
                f"  {g1} vs {g2}: "
                f"cos_dist={r['T']:.4f}  "
                f"p={r['p_raw']:.4f} (corrected={r['p_corrected']:.4f})  "
                f"Cohen's d={r['cohens_d']:.2f}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"SSDGroup({self.G} groups, n_kept={self.n_kept}, omnibus_p={self.omnibus_p:.4f})"


class SSDContrast:
    """Single pairwise group contrast — same interpretation API as SSD.

    Duck-types with SSD: has .kv, .beta, .beta_unit, .use_unit_beta,
    .lexicon, .window, .sif_a for use with neighbors, clustering, snippets.
    """

    def __init__(
        self, *, kv, beta, beta_unit, x, x_pair,
        groups_kept, groups_pair, group_a, group_b,
        lexicon, window, sif_a, lang="pl", perm_result,
    ):
        self.kv = kv
        self.beta = beta
        self.beta_unit = beta_unit
        self.use_unit_beta = True
        self.x = x
        self.x_pair = x_pair
        self.groups_kept = groups_kept
        self.groups_pair = groups_pair
        self.group_a = group_a
        self.group_b = group_b
        self.lexicon = lexicon
        self.window = window
        self.sif_a = sif_a
        self.lang = lang
        self.perm_result = perm_result
        self.pos_clusters_raw = None
        self.neg_clusters_raw = None

    def top_words(self, n: int = 20) -> list[dict]:
        """Top neighbor words for both poles of the contrast."""
        b = self.beta_unit
        out = []
        for side, group, vec in [("pos", self.group_a, b), ("neg", self.group_b, -b)]:
            pairs = filtered_neighbors(self.kv, vec, topn=n, lang=self.lang)
            for rank, (word, cos) in enumerate(pairs, 1):
                out.append({"side": side, "group": group, "rank": rank, "word": word, "cos": float(cos)})
        return out

    def neighbors(self, side: str = "pos", n: int = 20) -> list[tuple[str, float]]:
        """Nearest embedding neighbors along one pole of the contrast.

        Parameters
        ----------
        side : {"pos", "neg"}, optional
            ``"pos"`` returns neighbors toward *group_a*; ``"neg"`` toward
            *group_b*.  Default ``"pos"``.
        n : int, optional
            Number of neighbors to return, by default 20.

        Returns
        -------
        list[tuple[str, float]]
            ``(word, cosine_similarity)`` pairs sorted descending.
        """
        b = self.beta_unit
        vec = b if side == "pos" else -b
        return filtered_neighbors(self.kv, vec, topn=n, lang=self.lang)

    def cluster_neighbors(self, side: str = "pos", **kwargs) -> list[dict]:
        """Cluster nearest neighbors along one pole of the contrast.

        Parameters
        ----------
        side : {"pos", "neg"}, optional
            Pole to cluster, by default ``"pos"``.
        **kwargs
            Forwarded to :func:`cluster_top_neighbors`.

        Returns
        -------
        list[dict]
            Cluster dicts (same format as
            ``_SSDResultBase.cluster_neighbors``).
        """
        clusters = cluster_top_neighbors(
            self.kv, self.beta, use_unit_beta=True, side=side,
            lang=self.lang, **kwargs,
        )
        if side == "pos":
            self.pos_clusters_raw = clusters
        else:
            self.neg_clusters_raw = clusters
        return clusters

    def summary(self) -> str:
        """Human-readable contrast summary."""
        r = self.perm_result
        title = f"SSDContrast: {self.group_a} vs {self.group_b}"
        sep = "─" * len(title)
        lines = [title, sep]
        lines.append(f"cos_dist = {r['T']:.4f}   p = {r['p_raw']:.4f} (corrected = {r['p_corrected']:.4f})")
        lines.append(f"Cohen's d = {r['cohens_d']:.2f}   ‖contrast‖ = {r['contrast_norm']:.4f}")
        lines.append(f"n_{self.group_a} = {r['n_g1']}   n_{self.group_b} = {r['n_g2']}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"SSDContrast({self.group_a} vs {self.group_b})"
