"""Tests for SSD.fit_groups() — group comparison via permutation tests."""

import numpy as np
import pytest

from ssdiff.corpus import Corpus
from ssdiff.results import GroupResult
from ssdiff.results.group_result import PairView
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

    def test_attributes(self, result):
        assert result.G == 2
        # group_labels removed; check via pairs
        g_labels = {p.g1 for p in result.pairs} | {p.g2 for p in result.pairs}
        assert g_labels == {"A", "B"}
        assert result.n_kept > 0
        assert np.isfinite(result.test.omnibus_T)
        assert 0 <= result.test.omnibus_p <= 1
        assert result.test.omnibus_T >= 0

    def test_pairwise(self, result):
        assert len(result.pairs) == 1  # 2 groups → 1 pair
        p = list(result.pairs)[0]
        assert np.isfinite(p.T)
        assert 0 <= p.p_raw <= 1
        assert 0 <= p.p_corrected <= 1
        assert p.p_corrected >= p.p_raw  # Correction only inflates
        assert np.isfinite(p.cohens_d)
        assert p.n_g1 > 0 and p.n_g2 > 0

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
        assert "GroupResult" in r
        assert "omnibus" in r or "omnibus_T" in r or "p=" in r


class TestFitGroups3Groups:
    """SSD.fit_groups() with 3 categorical groups."""

    def test_three_groups_no_filter(self, tiny_kv, large_docs_3x20, large_groups_3x20, lexicon):
        """3 groups each with >=20 docs."""
        corpus = Corpus(large_docs_3x20, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, large_groups_3x20, lexicon)
        result = ssd.fit_groups(n_perm=50, random_state=42)
        assert result.G == 3
        assert len(result.pairs) == 3  # C(3,2) = 3 pairs
        assert 0 <= result.test.omnibus_p <= 1


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
        # n_group_dropped not in new API; core behavior verified via G==2

    def test_too_few_groups_after_filter_raises(self, tiny_kv, sample_docs, sample_groups, lexicon):
        """If all groups are <20, raise ValueError."""
        corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, sample_groups, lexicon)
        with pytest.warns(UserWarning, match="dropped"), \
             pytest.raises(ValueError, match="Need at least 2 groups"):
            ssd.fit_groups(n_perm=50)


class TestFitGroupsMedianSplit:
    """SSD.fit_groups(median_split=True)."""

    def test_median_split(self, tiny_kv, sample_docs, sample_y, lexicon):
        corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, sample_y, lexicon)
        result = ssd.fit_groups(median_split=True, n_perm=50)
        assert result.G == 2
        g_labels = {p.g1 for p in result.pairs} | {p.g2 for p in result.pairs}
        assert g_labels == {"high", "low"}

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
        for p in r.pairs:
            assert p.p_corrected >= p.p_raw

    def test_bonferroni(self, ssd):
        r = ssd.fit_groups(n_perm=50, correction="bonferroni")
        for p in r.pairs:
            assert p.p_corrected >= p.p_raw

    def test_fdr_bh(self, ssd):
        r = ssd.fit_groups(n_perm=50, correction="fdr_bh")
        assert r.correction == "fdr_bh"
        for p in r.pairs:
            assert 0 <= p.p_corrected <= 1
            assert p.p_corrected >= p.p_raw

    def test_none(self, ssd):
        r = ssd.fit_groups(n_perm=50, correction="none")
        for p in r.pairs:
            assert p.p_corrected == p.p_raw

    def test_invalid_raises(self, ssd):
        with pytest.raises(ValueError, match="Unknown correction"):
            ssd.fit_groups(n_perm=50, correction="invalid")


class TestGroupResultPairsAccess:
    """GroupResult.pairs access: tuple key, iteration, PairView semantics."""

    @pytest.fixture(scope="class")
    def result(self, tiny_kv, large_docs, large_groups_2, lexicon):
        corpus = Corpus(large_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, large_groups_2, lexicon)
        return ssd.fit_groups(n_perm=50, random_state=42)

    def test_pairs_iterable(self, result):
        pairs = list(result.pairs)
        assert len(pairs) == 1

    def test_pairs_tuple_key_access(self, result):
        """gr.pairs["A", "B"] returns a PairView."""
        pv = result.pairs["A", "B"]
        assert isinstance(pv, PairView)
        assert "A" in pv.contrast and "B" in pv.contrast

    def test_pairs_missing_key_raises(self, result):
        with pytest.raises(KeyError):
            _ = result.pairs["X", "Y"]


class TestGroupResultPairViewSemantics:
    """PairView sign-flip and field-swap on reverse access."""

    @pytest.fixture(scope="class")
    def result_3g(self, tiny_kv, large_docs_3x20, large_groups_3x20, lexicon):
        corpus = Corpus(large_docs_3x20, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, large_groups_3x20, lexicon)
        return ssd.fit_groups(n_perm=50, random_state=42)

    def test_pair_access_all_contrasts(self, result_3g):
        """3 groups → 3 pairs accessible by tuple key."""
        pairs = list(result_3g.pairs)
        g_set = {(p.g1, p.g2) for p in pairs}
        # C(3,2) = 3 unique ordered pairs
        assert len(g_set) == 3

    def test_reverse_access_flips_t(self, result_3g):
        """Accessing pair in reverse order flips the T statistic sign."""
        # Find one pair
        p_canonical = list(result_3g.pairs)[0]
        g1, g2 = p_canonical.g1, p_canonical.g2
        pv_forward = result_3g.pairs[g1, g2]
        pv_reverse = result_3g.pairs[g2, g1]
        assert abs(pv_forward.pair.T + pv_reverse.pair.T) < 1e-9

    def test_reverse_access_preserves_pvalues(self, result_3g):
        """P-values are symmetric — must not flip on reverse access."""
        p_canonical = list(result_3g.pairs)[0]
        g1, g2 = p_canonical.g1, p_canonical.g2
        pv_forward = result_3g.pairs[g1, g2]
        pv_reverse = result_3g.pairs[g2, g1]
        assert pv_forward.pair.p_raw == pv_reverse.pair.p_raw
        assert pv_forward.pair.p_corrected == pv_reverse.pair.p_corrected


class TestGroupReport:
    """GroupResult.report() method."""

    @pytest.fixture(scope="class")
    def result(self, tiny_kv, large_docs, large_groups_2, lexicon):
        corpus = Corpus(large_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, large_groups_2, lexicon)
        return ssd.fit_groups(n_perm=50, random_state=42)

    def test_default_report_is_report_object(self, result):
        from ssdiff.results.report import Report
        assert isinstance(result.report(), Report)

    def test_report_text_has_omnibus(self, result):
        text = result.report().to_text()
        assert "Omnibus" in text

    def test_report_text_has_pairwise(self, result):
        text = result.report().to_text()
        assert "Pairwise" in text

    def test_report_text_has_vs(self, result):
        text = result.report().to_text()
        assert "vs" in text

    def test_report_not_auto_printed(self, result, capsys):
        """report() does NOT auto-print to stdout."""
        _ = result.report()
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_report_citation_present(self, result):
        """Every report contains the citation."""
        text = result.report().to_text()
        assert "Plisiecki" in text


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
        assert isinstance(gr, GroupResult)
        assert gr.G == 2
        assert pls.fit_info.n_components == 2

    def test_pls_then_groups(self, tiny_kv, large_docs, large_y, lexicon):
        """Same SSD can fit PLS first and then groups."""
        corpus = Corpus(large_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, large_y, lexicon)
        pls = ssd.fit_pls(n_components=2, p_method=None)
        gr = ssd.fit_groups(median_split=True, n_perm=50)
        assert isinstance(gr, GroupResult)

    def test_pls_then_ols(self, tiny_kv, large_docs, large_y, lexicon):
        """Same SSD can fit PLS and then OLS."""
        corpus = Corpus(large_docs, pretokenized=True, lang="pl")
        ssd = SSD(tiny_kv, corpus, large_y, lexicon)
        pls = ssd.fit_pls(n_components=2, p_method=None)
        from ssdiff.results import PCAOLSResult
        ols = ssd.fit_ols(fixed_k=3)
        assert isinstance(ols, PCAOLSResult)
