"""Tests for SSD.fit_groups() — group comparison via permutation tests."""

import numpy as np
import pytest

from ssdiff.corpus import Corpus
from ssdiff.results import GroupResult
from ssdiff.ssd import SSD


class TestFitGroups2Groups:
    """SSD.fit_groups() with 2 categorical groups (large data)."""

    @pytest.fixture(scope="class")
    def result(self, tiny_kv, large_docs, large_groups_2, lexicon):
        corpus = Corpus(large_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, large_groups_2, lexicon)
        return ssd.fit_groups(n_perm=50, random_state=42)

    def test_returns_group_result(self, result):
        assert isinstance(result, GroupResult)
        assert result.result_type == "group"

    def test_attributes(self, result):
        assert result.G == 2
        assert set(result.group_labels) == {"A", "B"}
        assert result.n_kept > 0
        assert np.isfinite(result.omnibus_T)
        assert 0 <= result.omnibus_p <= 1
        assert result.omnibus_T >= 0  # T-statistic from permutation should be non-negative

    def test_pairwise(self, result):
        assert len(result.pairwise) == 1  # 2 groups → 1 pair
        key = list(result.pairwise.keys())[0]
        r = result.pairwise[key]
        assert "T" in r
        assert "p_raw" in r
        assert "p_corrected" in r
        assert "cohens_d" in r
        assert "beta_unit" in r
        # Values should be reasonable
        assert np.isfinite(r["T"])
        assert 0 <= r["p_raw"] <= 1
        assert 0 <= r["p_corrected"] <= 1
        assert r["p_corrected"] >= r["p_raw"]  # Correction only inflates
        assert np.isfinite(r["cohens_d"])
        assert r["beta_unit"].ndim == 1
        assert np.linalg.norm(r["beta_unit"]) == pytest.approx(1.0, abs=1e-6)

    def test_results_table(self, result):
        table = result.results_table()
        assert isinstance(table, list)
        assert len(table) == 1
        assert "group_A" in table[0]

    def test_correction_default_holm(self, result):
        assert result.correction == "holm"

    def test_no_mutation(self, tiny_kv, large_docs, large_groups_2, lexicon):
        """fit_groups must not mutate self.x or self.y_kept."""
        corpus = Corpus(large_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, large_groups_2, lexicon)
        x_before = ssd.x.copy()
        y_before = ssd.y_kept.copy()
        ssd.fit_groups(n_perm=50)
        np.testing.assert_array_equal(ssd.x, x_before)
        np.testing.assert_array_equal(ssd.y_kept, y_before)

    def test_repr(self, result):
        r = repr(result)
        assert "2 groups" in r
        assert "omnibus_p" in r


class TestFitGroups3Groups:
    """SSD.fit_groups() with 3 categorical groups."""

    def test_three_groups_no_filter(self, tiny_kv, large_docs_3x20, large_groups_3x20, lexicon):
        """3 groups each with >=20 docs."""
        corpus = Corpus(large_docs_3x20, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, large_groups_3x20, lexicon)
        result = ssd.fit_groups(n_perm=50, random_state=42)
        assert result.G == 3
        assert len(result.pairwise) == 3  # C(3,2) = 3 pairs
        assert 0 <= result.omnibus_p <= 1


class TestFitGroupsSmallGroupFilter:
    """Small-group filtering behavior."""

    def test_small_groups_dropped_with_warning(self, tiny_kv, lexicon):
        """Groups below threshold are dropped with a warning."""
        rng = np.random.default_rng(99)
        seeds = ["kraj", "narod", "panstwo"]
        context = ["piekny", "silny", "zly", "dobry", "wielki", "maly",
                   "stary", "nowy", "dom", "szkola", "praca", "miasto"]
        docs = []
        for i in range(45):
            seed = seeds[i % len(seeds)]
            ctx = list(rng.choice(context, size=3, replace=False))
            docs.append([seed] + ctx)
        # A=20, B=20, C=5 — C should be dropped
        groups = np.array(
            ["A"] * 20 + ["B"] * 20 + ["C"] * 5, dtype=object,
        )
        corpus = Corpus(docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, groups, lexicon)
        with pytest.warns(UserWarning, match="Group 'C' dropped"):
            result = ssd.fit_groups(n_perm=50)
        assert result.G == 2
        assert result.n_group_dropped == 5

    def test_too_few_groups_after_filter_raises(self, tiny_kv, sample_docs, sample_groups, lexicon):
        """If all groups are <20, raise ValueError."""
        corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, sample_groups, lexicon)
        with pytest.raises(ValueError, match="Need at least 2 groups"):
            ssd.fit_groups(n_perm=50)


class TestFitGroupsMedianSplit:
    """SSD.fit_groups(median_split=True)."""

    def test_median_split(self, tiny_kv, sample_docs, sample_y, lexicon):
        corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, sample_y, lexicon)
        result = ssd.fit_groups(median_split=True, n_perm=50)
        assert result.G == 2
        assert set(result.group_labels) == {"high", "low"}

    def test_median_split_identical_y_raises(self, tiny_kv, sample_docs, lexicon):
        y_const = np.ones(len(sample_docs))
        corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, y_const, lexicon)
        with pytest.raises(ValueError, match="all y values are identical"):
            ssd.fit_groups(median_split=True, n_perm=50)


class TestFitGroupsCorrection:
    """P-value correction methods."""

    @pytest.fixture(scope="class")
    def ssd(self, tiny_kv, large_docs_3x20, large_groups_3x20, lexicon):
        corpus = Corpus(large_docs_3x20, pretokenized=True, lang="pl")
        return SSD(tiny_kv, corpus, large_groups_3x20, lexicon)

    def test_holm(self, ssd):
        r = ssd.fit_groups(n_perm=50, correction="holm")
        for pw in r.pairwise.values():
            assert pw["p_corrected"] >= pw["p_raw"]

    def test_bonferroni(self, ssd):
        r = ssd.fit_groups(n_perm=50, correction="bonferroni")
        for pw in r.pairwise.values():
            assert pw["p_corrected"] >= pw["p_raw"]

    def test_fdr_bh(self, ssd):
        r = ssd.fit_groups(n_perm=50, correction="fdr_bh")
        assert r.correction == "fdr_bh"
        for pw in r.pairwise.values():
            assert 0 <= pw["p_corrected"] <= 1
            assert pw["p_corrected"] >= pw["p_raw"]

    def test_none(self, ssd):
        r = ssd.fit_groups(n_perm=50, correction="none")
        for pw in r.pairwise.values():
            assert pw["p_corrected"] == pw["p_raw"]

    def test_invalid_raises(self, ssd):
        with pytest.raises(ValueError, match="Unknown correction"):
            ssd.fit_groups(n_perm=50, correction="invalid")


class TestGroupResultInterpretation:
    """GroupResult interpretation methods."""

    @pytest.fixture(scope="class")
    def result(self, tiny_kv, large_docs, large_groups_2, lexicon):
        corpus = Corpus(large_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, large_groups_2, lexicon)
        return ssd.fit_groups(n_perm=50, random_state=42)

    def test_top_words_has_contrast(self, result):
        words = result.top_words(n=3)
        assert isinstance(words, list)
        assert len(words) > 0
        assert "contrast" in words[0]
        assert "side" in words[0]

    def test_neighbors(self, result):
        nbrs = result.neighbors("pos", n=3)
        assert isinstance(nbrs, list)

    def test_cluster_neighbors(self, result):
        clusters = result.cluster_neighbors("pos", topn=10, k=2)
        assert isinstance(clusters, list)


class TestGroupResultFilterGroups:
    """GroupResult.filter_groups()."""

    @pytest.fixture(scope="class")
    def result_3g(self, tiny_kv, large_docs_3x20, large_groups_3x20, lexicon):
        corpus = Corpus(large_docs_3x20, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, large_groups_3x20, lexicon)
        return ssd.fit_groups(n_perm=50, random_state=42)

    def test_filter_two_groups(self, result_3g):
        filtered = result_3g.filter_groups("X", "Y")
        assert filtered.G == 2
        assert len(filtered.pairwise) == 1

    def test_filter_one_group(self, result_3g):
        filtered = result_3g.filter_groups("X")
        for g1, g2 in filtered.pairwise:
            assert "X" in (g1, g2)

    def test_filter_invalid_label_raises(self, result_3g):
        with pytest.raises(ValueError, match="not found"):
            result_3g.filter_groups("NONEXISTENT")

    def test_filtered_summary(self, result_3g):
        filtered = result_3g.filter_groups("X", "Y")
        s = filtered.summary()
        assert "filtered" in s.lower()


class TestGroupSummary:
    """GroupResult.summary() display."""

    def test_summary_2groups(self, tiny_kv, large_docs, large_groups_2, lexicon):
        corpus = Corpus(large_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, large_groups_2, lexicon)
        result = ssd.fit_groups(n_perm=50)
        s = result.summary()
        assert isinstance(s, str)
        assert "Omnibus" in s
        assert "Pairwise" in s

    def test_summary_3groups(self, tiny_kv, large_docs_3x20, large_groups_3x20, lexicon):
        corpus = Corpus(large_docs_3x20, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, large_groups_3x20, lexicon)
        result = ssd.fit_groups(n_perm=50)
        s = result.summary()
        assert s.count("vs") == 3


class TestGroupReport:
    """GroupResult.report() method."""

    @pytest.fixture(scope="class")
    def result(self, tiny_kv, large_docs, large_groups_2, lexicon):
        corpus = Corpus(large_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, large_groups_2, lexicon)
        return ssd.fit_groups(n_perm=50, random_state=42)

    def test_default(self, result, capsys):
        text = result.report()
        captured = capsys.readouterr()
        assert captured.out.strip() == text.strip()
        assert "Group Analysis" in text
        assert "Top Words" in text

    def test_top_words_none_skips(self, result):
        text = result.report(top_words=None)
        assert "Top Words" not in text

    def test_top_words_has_contrast_label(self, result):
        text = result.report(top_words=3)
        assert "vs" in text

    def test_clusters(self, result):
        text = result.report(top_words=None, clusters=10)
        assert "Clusters" in text

    def test_extreme_docs_ignored(self, result):
        text = result.report(top_words=None, extreme_docs=5)
        assert "Extreme" not in text

    def test_misdiagnosed_ignored(self, result):
        text = result.report(top_words=None, misdiagnosed=5)
        assert "Misdiagnosed" not in text

    def test_summary_always_present(self, result):
        text = result.report(top_words=None)
        assert "Omnibus" in text
        assert "Pairwise" in text


class TestSSDReuseFitGroups:
    """SSD instance can be reused for fit_groups + other backends."""

    def test_categorical_y_rejects_pls(self, tiny_kv, large_docs, large_groups_2, lexicon):
        """SSD with categorical y can fit_groups but not fit_pls/fit_ols."""
        corpus = Corpus(large_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, large_groups_2, lexicon)
        gr = ssd.fit_groups(n_perm=50)
        assert isinstance(gr, GroupResult)
        with pytest.raises(ValueError, match="requires numeric y"):
            ssd.fit_pls()
        with pytest.raises(ValueError, match="requires numeric y"):
            ssd.fit_ols()

    def test_median_split_then_pls(self, tiny_kv, large_docs, large_y, lexicon):
        """Same SSD can do median_split groups and then PLS."""
        corpus = Corpus(large_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, large_y, lexicon)
        gr = ssd.fit_groups(median_split=True, n_perm=50)
        pls = ssd.fit_pls(n_components=2, p_method=None)
        assert gr.result_type == "group"
        assert pls.result_type == "pls"
        # Results should be independent
        assert gr.G == 2
        assert pls.n_components == 2

    def test_pls_then_groups(self, tiny_kv, large_docs, large_y, lexicon):
        """Same SSD can fit PLS first and then groups."""
        corpus = Corpus(large_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, large_y, lexicon)
        pls = ssd.fit_pls(n_components=2, p_method=None)
        gr = ssd.fit_groups(median_split=True, n_perm=50)
        assert pls.result_type == "pls"
        assert gr.result_type == "group"

    def test_pls_then_ols(self, tiny_kv, large_docs, large_y, lexicon):
        """Same SSD can fit PLS and then OLS."""
        corpus = Corpus(large_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, large_y, lexicon)
        pls = ssd.fit_pls(n_components=2, p_method=None)
        ols = ssd.fit_ols(n_components=3)
        assert pls.result_type == "pls"
        assert ols.result_type == "pca_ols"
