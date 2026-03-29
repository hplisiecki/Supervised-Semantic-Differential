"""Tests for ssdlite.utils.group — SSDGroup and SSDContrast."""

import numpy as np
import pytest

from ssdlite.utils.group import SSDGroup, SSDContrast
from ssdlite.corpus import Corpus


class TestSSDGroup:
    @pytest.fixture(scope="class")
    def fitted(self, tiny_kv, sample_docs, sample_groups, lexicon):
        corpus = Corpus(sample_docs, pretokenized=True)
        return SSDGroup(
            tiny_kv, corpus, sample_groups, lexicon,
            n_perm=50, random_state=42,
        )

    def test_attributes(self, fitted):
        assert fitted.G == 2
        assert set(fitted.group_labels) == {"A", "B"}
        assert fitted.n_kept > 0
        assert np.isfinite(fitted.omnibus_T)
        assert 0 <= fitted.omnibus_p <= 1

    def test_pairwise(self, fitted):
        assert len(fitted.pairwise) == 1  # 2 groups → 1 pair
        key = list(fitted.pairwise.keys())[0]
        r = fitted.pairwise[key]
        assert "T" in r
        assert "p_raw" in r
        assert "p_corrected" in r
        assert "cohens_d" in r
        assert "contrast_unit" in r

    def test_get_contrast(self, fitted):
        contrast = fitted.get_contrast("A", "B")
        assert isinstance(contrast, SSDContrast)
        assert contrast.group_a == "A"
        assert contrast.group_b == "B"

    def test_get_contrast_flipped(self, fitted):
        contrast = fitted.get_contrast("B", "A")
        assert contrast.group_a == "B"
        assert contrast.group_b == "A"

    def test_cohens_d_flips_on_reverse(self, fitted):
        """Cohen's d should negate when get_contrast order is reversed."""
        c_ab = fitted.get_contrast(fitted.group_labels[0], fitted.group_labels[-1])
        c_ba = fitted.get_contrast(fitted.group_labels[-1], fitted.group_labels[0])
        assert c_ab.perm_result["cohens_d"] == pytest.approx(
            -c_ba.perm_result["cohens_d"], abs=1e-12
        )

    def test_contrast_scores(self, fitted):
        scores = fitted.contrast_scores("A", "B")
        assert "group" in scores
        assert "cos_to_contrast" in scores
        assert len(scores["group"]) == fitted.n_kept

    def test_results_table(self, fitted):
        table = fitted.results_table()
        assert isinstance(table, list)
        assert len(table) == 1
        assert "group_A" in table[0]

    def test_repr(self, fitted):
        r = repr(fitted)
        assert "2 groups" in r


class TestSSDGroup3Groups:
    def test_three_groups(self, tiny_kv, sample_docs, sample_groups_3, lexicon):
        corpus = Corpus(sample_docs, pretokenized=True)
        sg = SSDGroup(
            tiny_kv, corpus, sample_groups_3, lexicon,
            n_perm=50, random_state=42,
        )
        assert sg.G == 3
        assert len(sg.pairwise) == 3  # C(3,2) = 3 pairs
        assert 0 <= sg.omnibus_p <= 1


class TestSSDContrast:
    @pytest.fixture(scope="class")
    def contrast(self, tiny_kv, sample_docs, sample_groups, lexicon):
        corpus = Corpus(sample_docs, pretokenized=True)
        sg = SSDGroup(
            tiny_kv, corpus, sample_groups, lexicon,
            n_perm=50, random_state=42,
        )
        return sg.get_contrast("A", "B")

    def test_top_words(self, contrast):
        words = contrast.top_words(n=3)
        assert isinstance(words, list)
        assert len(words) > 0
        assert "group" in words[0]

    def test_neighbors(self, contrast):
        nbrs = contrast.neighbors("pos", n=3)
        assert isinstance(nbrs, list)

    def test_repr(self, contrast):
        assert "A vs B" in repr(contrast)


class TestSummary:
    def test_group_summary(self, tiny_kv, sample_docs, sample_groups, lexicon):
        corpus = Corpus(sample_docs, pretokenized=True)
        sg = SSDGroup(tiny_kv, corpus, sample_groups, lexicon, n_perm=50)
        s = sg.summary()
        assert isinstance(s, str)
        assert "Omnibus" in s or "omnibus" in s
        assert "kept" in s

    def test_group_3_summary(self, tiny_kv, sample_docs, sample_groups_3, lexicon):
        corpus = Corpus(sample_docs, pretokenized=True)
        sg = SSDGroup(tiny_kv, corpus, sample_groups_3, lexicon, n_perm=50)
        s = sg.summary()
        assert "Pairwise" in s or "pairwise" in s
        # All 3 pairs should appear
        assert s.count("vs") == 3

    def test_contrast_summary(self, tiny_kv, sample_docs, sample_groups, lexicon):
        corpus = Corpus(sample_docs, pretokenized=True)
        sg = SSDGroup(tiny_kv, corpus, sample_groups, lexicon, n_perm=50)
        c = sg.get_contrast("A", "B")
        s = c.summary()
        assert isinstance(s, str)
        assert "A" in s and "B" in s
        assert "cos_dist" in s
        assert "Cohen" in s
