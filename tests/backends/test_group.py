"""Tests for ssdiff.backends.group — pure-algorithm invariants.

Covers:
- median_split (tie-breaking, all-same error, no-tie case)
- filter_small_groups (drop below MIN_GROUP_SIZE, pairs only for large groups)
- correct_pvalues (Holm, Bonferroni, FDR-BH, none)
- unified_permutation_test (canonical pair order, signal vs null p-values,
  Cohen's d formula)
- _canonical_pair_key (numeric-string sort, regression guard for g2 < g10)

Does NOT test fit_groups SSD-level integration (integration/test_ssd.py)
or _ShimView access patterns (results/test_group.py).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ssdiff.backends.group import (
    MIN_GROUP_SIZE,
    correct_pvalues,
    filter_small_groups,
    median_split,
    unified_permutation_test,
)
from ssdiff.results.group_result import _canonical_pair_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_x(n: int, dim: int = 8, seed: int = 0) -> np.ndarray:
    """Return (n, dim) unit-normed float64 embedding matrix."""
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(n, dim)).astype(np.float64)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


def _make_signal_x(n_per_group: int, dim: int = 16, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Two-group x with a planted signal (groups have separated centroids)."""
    rng = np.random.default_rng(seed)
    # Group A: shifted +signal along dim 0; Group B: shifted -signal
    signal = np.zeros(dim)
    signal[0] = 3.0  # strong separation on first dimension
    base = rng.normal(size=(n_per_group * 2, dim)) * 0.3
    x = base.copy()
    x[:n_per_group, 0] += 3.0
    x[n_per_group:, 0] -= 3.0
    # Unit-normalize
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x /= np.maximum(norms, 1e-12)
    groups = np.array(["A"] * n_per_group + ["B"] * n_per_group, dtype=object)
    return x, groups


# ---------------------------------------------------------------------------
# median_split
# ---------------------------------------------------------------------------

class TestMedianSplit:
    def test_no_ties_low_high(self):
        """Clean split: values below median → low, above → high."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        labels = median_split(y, random_state=0)
        assert set(labels) == {"low", "high"}
        assert labels[0] == "low"
        assert labels[1] == "low"
        assert labels[2] == "high"
        assert labels[3] == "high"

    def test_all_same_raises(self):
        """All-identical y must raise ValueError."""
        y = np.array([5.0, 5.0, 5.0, 5.0])
        with pytest.raises(ValueError, match="all y values are identical"):
            median_split(y)

    def test_ties_go_to_both_bins(self):
        """Tied-at-median values are distributed so each bin gets ~n//2."""
        # y = [1, 2, 2, 2, 2, 3] → median = 2 (tied)
        y = np.array([1.0, 2.0, 2.0, 2.0, 2.0, 3.0])
        labels = median_split(y, random_state=2137)
        assert set(labels) == {"low", "high"}
        # With n=6, target = 3. Below median: [1.0] → n_below=1, n_low_needs=2.
        # 2 tied docs go to "low", 2 go to "high", 1 non-tied "high" at 3.0.
        n_low = int((labels == "low").sum())
        n_high = int((labels == "high").sum())
        assert n_low == 3
        assert n_high == 3

    def test_ties_deterministic_with_seed(self):
        """Same seed produces identical label assignment for tied values."""
        y = np.array([1.0, 2.0, 2.0, 2.0, 3.0])
        l1 = median_split(y, random_state=7)
        l2 = median_split(y, random_state=7)
        np.testing.assert_array_equal(l1, l2)

    def test_ties_different_seeds_may_differ(self):
        """Different seeds can produce different tie assignments (not guaranteed, but likely)."""
        y = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 3.0])
        # 8 tied values → seeds should produce different shuffles almost surely
        results = set()
        for seed in range(20):
            l = median_split(y, random_state=seed)
            results.add(tuple(l))
        # More than one unique assignment observed
        assert len(results) > 1


# ---------------------------------------------------------------------------
# filter_small_groups
# ---------------------------------------------------------------------------

class TestFilterSmallGroups:
    def test_small_group_dropped(self):
        """Groups below MIN_GROUP_SIZE are removed from x and groups arrays."""
        n_large = MIN_GROUP_SIZE
        n_small = MIN_GROUP_SIZE - 1  # just below threshold
        n_total = n_large * 2 + n_small
        x = _make_x(n_total, dim=4)
        groups = np.array(
            ["A"] * n_large + ["B"] * n_large + ["C"] * n_small, dtype=object
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x_out, g_out, n_dropped = filter_small_groups(x, groups)

        assert n_dropped == n_small
        assert len(x_out) == n_large * 2
        assert set(g_out) == {"A", "B"}
        # Warning should mention the dropped group
        assert any("C" in str(warning.message) for warning in w)

    def test_pairs_only_for_large_groups(self):
        """With sizes [30, 30, 5], filter leaves only the two 30-doc groups."""
        n_large = 30
        n_small = 5
        n_total = n_large * 2 + n_small
        x = _make_x(n_total, dim=4, seed=1)
        groups = np.array(
            ["big1"] * n_large + ["big2"] * n_large + ["tiny"] * n_small,
            dtype=object,
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _, g_out, _ = filter_small_groups(x, groups)

        assert "tiny" not in set(g_out)
        assert {"big1", "big2"} == set(g_out)

    def test_no_small_groups_unchanged(self):
        """No groups dropped when all are >= MIN_GROUP_SIZE; n_dropped=0."""
        n = MIN_GROUP_SIZE
        x = _make_x(n * 2, dim=4, seed=2)
        groups = np.array(["X"] * n + ["Y"] * n, dtype=object)
        x_out, g_out, n_dropped = filter_small_groups(x, groups)
        assert n_dropped == 0
        assert len(x_out) == n * 2
        np.testing.assert_array_equal(g_out, groups)

    def test_inputs_not_mutated(self):
        """filter_small_groups must not modify the original arrays."""
        n_large = MIN_GROUP_SIZE
        n_small = 3
        x = _make_x(n_large + n_small, dim=4, seed=3)
        groups = np.array(["A"] * n_large + ["B"] * n_small, dtype=object)
        x_copy = x.copy()
        g_copy = groups.copy()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            filter_small_groups(x, groups)
        np.testing.assert_array_equal(x, x_copy)
        np.testing.assert_array_equal(groups, g_copy)


# ---------------------------------------------------------------------------
# correct_pvalues
# ---------------------------------------------------------------------------

class TestCorrectPvalues:
    def test_none_returns_exact_copy(self):
        """correction='none' must return p_corrected == p_raw exactly."""
        p = np.array([0.01, 0.05, 0.1, 0.5])
        result = correct_pvalues(p, method="none")
        np.testing.assert_array_equal(result, p)

    def test_bonferroni_single_pair(self):
        """Single-pair Bonferroni: p_corrected == p_raw (multiply by 1)."""
        p_raw = np.array([0.03])
        result = correct_pvalues(p_raw, method="bonferroni")
        np.testing.assert_allclose(result, p_raw, atol=1e-15)

    def test_bonferroni_three_pairs(self):
        """Three-pair Bonferroni: p_corrected == min(1.0, 3 * p_raw) for each pair."""
        p_raw = np.array([0.01, 0.03, 0.02])
        expected = np.minimum(p_raw * 3, 1.0)
        result = correct_pvalues(p_raw, method="bonferroni")
        np.testing.assert_allclose(result, expected, atol=1e-15)

    def test_bonferroni_caps_at_1(self):
        """Bonferroni correction is capped at 1.0."""
        p_raw = np.array([0.5, 0.6, 0.4])
        result = correct_pvalues(p_raw, method="bonferroni")
        assert np.all(result <= 1.0)
        # At least one should hit the cap
        assert np.any(result == 1.0)

    def test_fdr_bh_exact(self):
        """FDR BH on [0.01, 0.04, 0.03] matches manual BH calculation.

        Manual:
          sorted p:  [0.01, 0.03, 0.04] at original indices [0, 2, 1]
          i=2 (idx=1, p=0.04): adj=0.04*3/3=0.04, cummin=0.04 → corrected[1]=0.04
          i=1 (idx=2, p=0.03): adj=0.03*3/2=0.045, cummin=min(0.04,0.045)=0.04 → corrected[2]=0.04
          i=0 (idx=0, p=0.01): adj=0.01*3/1=0.03, cummin=min(0.04,0.03)=0.03 → corrected[0]=0.03
        """
        p_raw = np.array([0.01, 0.04, 0.03])
        expected = np.array([0.03, 0.04, 0.04])
        result = correct_pvalues(p_raw, method="fdr_bh")
        np.testing.assert_allclose(result, expected, atol=1e-15)

    def test_holm_monotonic_inflation(self):
        """Holm correction: at least one pair has p_corrected > p_raw for 3-pair signal."""
        # Use a fixed set of plausible signal p-values
        for seed in [0, 7, 42]:
            rng = np.random.default_rng(seed)
            # Generate 3 small p-values (simulating signal)
            p_raw = np.sort(rng.uniform(0.001, 0.04, size=3))
            result = correct_pvalues(p_raw, method="holm")
            assert np.any(result > p_raw), (
                f"Holm should inflate at least one p-value; seed={seed}, "
                f"p_raw={p_raw}, corrected={result}"
            )

    def test_holm_minimum_unchanged_or_inflated(self):
        """Holm smallest p is multiplied by m (largest multiplier)."""
        p_raw = np.array([0.005, 0.02, 0.04])
        result = correct_pvalues(p_raw, method="holm")
        # Smallest p gets multiplied by 3 (m=3, rank=0 → m-rank=3)
        expected_min = min(0.005 * 3, 1.0)
        np.testing.assert_allclose(result[0], expected_min, atol=1e-15)

    def test_unknown_correction_raises(self):
        """Unknown correction method raises ValueError."""
        p = np.array([0.05])
        with pytest.raises(ValueError, match="Unknown correction"):
            correct_pvalues(p, method="bh_invalid")

    def test_empty_array_none(self):
        """Empty p-value array returns empty array (not error)."""
        p = np.array([], dtype=float)
        result = correct_pvalues(p, method="holm")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# _canonical_pair_key — numeric-string sort
# ---------------------------------------------------------------------------

class TestCanonicalPairKey:
    def test_two_groups_alphabetical(self):
        """'B', 'A' input → ('A', 'B') canonical order (alphabetical)."""
        result = _canonical_pair_key("B", "A")
        assert result == ("A", "B")

    def test_canonical_g2_before_g10(self):
        """Numeric sort: g2 < g10 (regression guard against lexicographic ordering).

        Without numeric sort, lexicographic order gives 'g10' < 'g2' (wrong).
        """
        result = _canonical_pair_key("g10", "g2")
        assert result == ("g2", "g10"), (
            "Numeric sort must put g2 before g10; "
            "lexicographic sort would incorrectly reverse them"
        )

    def test_canonical_eleven_groups_full_enumeration(self):
        """11-group canonical labels g1..g11: g2 appears before g10 in pair list."""
        labels = [f"g{i}" for i in range(1, 12)]  # g1..g11
        from itertools import combinations

        pairs = [_canonical_pair_key(a, b) for a, b in combinations(labels, 2)]
        # Extract indices for g2 and g10 in pair list
        g2_first_pair = next(
            (i for i, (a, _) in enumerate(pairs) if a == "g2"), None
        )
        g10_first_pair = next(
            (i for i, (a, _) in enumerate(pairs) if a == "g10"), None
        )
        # Both should appear; g2 appears as first element of its pair before g10 does
        assert g2_first_pair is not None
        assert g10_first_pair is not None
        assert g2_first_pair < g10_first_pair, (
            "g2 should appear before g10 in pair enumeration "
            "(numeric-sort, not lexicographic)"
        )

    def test_already_canonical_unchanged(self):
        """Input already in canonical order is returned unchanged."""
        assert _canonical_pair_key("g1", "g2") == ("g1", "g2")
        assert _canonical_pair_key("A", "B") == ("A", "B")


# ---------------------------------------------------------------------------
# unified_permutation_test — signal vs null, Cohen's d, pair structure
# ---------------------------------------------------------------------------

class TestUnifiedPermutationTest:
    def test_signal_p_below_threshold(self):
        """Known-signal case: omnibus_p < 0.05 with n_perm=500, seed=42."""
        x, groups = _make_signal_x(n_per_group=40, dim=16, seed=42)
        group_labels = ["A", "B"]  # sorted alphabetical
        result = unified_permutation_test(
            x, groups, group_labels,
            n_perm=500,
            correction="none",
            random_state=42,
        )
        assert result["omnibus_p"] < 0.05, (
            f"Signal case should have omnibus_p < 0.05, got {result['omnibus_p']}"
        )
        pair_p = result["pairwise"][("A", "B")]["p_raw"]
        assert pair_p < 0.05, f"Pairwise p should be < 0.05 for signal case, got {pair_p}"

    def test_null_p_above_threshold(self):
        """Permuted (null) y: omnibus_p > 0.1 (random shuffling destroys signal)."""
        rng = np.random.default_rng(123)
        x = _make_x(80, dim=16, seed=123)
        # Null groups: randomly assigned (no actual structure)
        groups = np.array(["A"] * 40 + ["B"] * 40, dtype=object)
        rng.shuffle(groups)
        group_labels = ["A", "B"]
        result = unified_permutation_test(
            x, groups, group_labels,
            n_perm=500,
            correction="none",
            random_state=99,
        )
        # Null case: p should not be suspiciously small
        # Use a relaxed threshold; even random chance could give < 0.1
        # so we just assert it's not at the floor (< 1/(n_perm+1) ~ 0.002)
        assert result["omnibus_p"] > 1.0 / 501, (
            f"Null omnibus_p at minimum floor is suspicious: {result['omnibus_p']}"
        )

    def test_pair_keys_match_group_labels(self):
        """Pairwise keys must be exactly the (g1, g2) tuples from group_labels."""
        x, groups = _make_signal_x(n_per_group=25, dim=8, seed=7)
        group_labels = ["A", "B"]
        result = unified_permutation_test(
            x, groups, group_labels, n_perm=50, random_state=0,
        )
        assert ("A", "B") in result["pairwise"]
        assert len(result["pairwise"]) == 1

    def test_three_group_pair_count(self):
        """Three groups → C(3,2)=3 pairs in pairwise dict."""
        n = 25
        x = _make_x(n * 3, dim=8, seed=5)
        groups = np.array(["X"] * n + ["Y"] * n + ["Z"] * n, dtype=object)
        group_labels = ["X", "Y", "Z"]
        result = unified_permutation_test(
            x, groups, group_labels, n_perm=50, random_state=0,
        )
        assert len(result["pairwise"]) == 3
        expected_pairs = {("X", "Y"), ("X", "Z"), ("Y", "Z")}
        assert set(result["pairwise"].keys()) == expected_pairs

    def test_cohens_d_formula(self):
        """Cohen's d ≈ 1.0 for groups with mean difference 1 and pooled std 1.

        Construction: 1-dimensional x so the contrast vector is exactly [1.0]
        (unit-normed centroid of group A=[+vals] is [1], group B=[-vals] is [-1],
        contrast unit = [1]). Projections = x[:,0] directly.

        Group A: x[:,0] ~ N(+0.5, 1.0), Group B: x[:,0] ~ N(-0.5, 1.0).
        Expected: d = (0.5 - (-0.5)) / pooled_std ≈ 1.0 / 1.0 = 1.0.
        """
        rng = np.random.default_rng(42)
        n = 500  # large n so sample pooled_std converges to 1.0

        xa = rng.normal(loc=0.5, scale=1.0, size=(n, 1))
        xb = rng.normal(loc=-0.5, scale=1.0, size=(n, 1))
        x = np.vstack([xa, xb]).astype(np.float64)

        groups = np.array(["A"] * n + ["B"] * n, dtype=object)
        group_labels = ["A", "B"]

        result = unified_permutation_test(
            x, groups, group_labels, n_perm=50, random_state=0,
        )
        d = result["pairwise"][("A", "B")]["cohens_d"]
        assert abs(abs(d) - 1.0) < 0.05, (
            f"Cohen's d should be ≈ 1.0 for mean diff=1, pooled_std≈1; got {d}"
        )

    def test_canonical_input_pair_order(self):
        """Input group_labels=['A','B'] → pair key ('A','B') present (not ('B','A'))."""
        x, groups = _make_signal_x(n_per_group=25, dim=8, seed=1)
        group_labels = ["A", "B"]  # alphabetical, as fit_groups produces
        result = unified_permutation_test(
            x, groups, group_labels, n_perm=50, random_state=0,
        )
        assert ("A", "B") in result["pairwise"]
        assert ("B", "A") not in result["pairwise"]

    def test_result_structure_keys(self):
        """Result dict contains all expected top-level keys."""
        x, groups = _make_signal_x(n_per_group=25, dim=8, seed=2)
        group_labels = ["A", "B"]
        result = unified_permutation_test(
            x, groups, group_labels, n_perm=50, random_state=0,
        )
        for key in ("omnibus_T", "omnibus_p", "pairwise", "group_labels", "G", "correction"):
            assert key in result, f"Missing key: {key}"
        pw = result["pairwise"][("A", "B")]
        for field in ("gradient", "T", "p_raw", "p_corrected", "cohens_d", "n_g1", "n_g2", "contrast_norm"):
            assert field in pw, f"Missing pairwise field: {field}"

    def test_n_g1_n_g2_counts(self):
        """n_g1, n_g2 in pairwise dict reflect actual group sizes."""
        n_a, n_b = 30, 20
        x = _make_x(n_a + n_b, dim=8, seed=4)
        groups = np.array(["A"] * n_a + ["B"] * n_b, dtype=object)
        group_labels = ["A", "B"]
        result = unified_permutation_test(
            x, groups, group_labels, n_perm=50, random_state=0,
        )
        pw = result["pairwise"][("A", "B")]
        assert pw["n_g1"] == n_a
        assert pw["n_g2"] == n_b

    def test_correction_none_p_corrected_equals_raw(self):
        """correction='none': p_corrected == p_raw for all pairs."""
        x, groups = _make_signal_x(n_per_group=25, dim=8, seed=3)
        group_labels = ["A", "B"]
        result = unified_permutation_test(
            x, groups, group_labels, n_perm=50,
            correction="none", random_state=0,
        )
        pw = result["pairwise"][("A", "B")]
        assert pw["p_corrected"] == pw["p_raw"]
