"""Tests for ssdiff/utils/lexicon.py and Corpus lexicon methods."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import pointbiserialr

from ssdiff.corpus import Corpus
from ssdiff.embeddings import Embeddings
from ssdiff.utils.lexicon import (
    _as_float_array,
    _chi2_pvalue,
    _cramers_v,
    _crosstab,
    _effect_direction,
    _pointbiserial_pvalue,
    _quantile_bins,
    _validate_var_type,
)

# ---------------------------------------------------------------------------
# _as_float_array
# ---------------------------------------------------------------------------

def test_as_float_array_basic():
    result = _as_float_array([1, 2, 3])
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_as_float_array_none_becomes_nan():
    result = _as_float_array([1.0, None, 3.0])
    assert np.isnan(result[1])
    assert result[0] == 1.0
    assert result[2] == 3.0


# ---------------------------------------------------------------------------
# _quantile_bins
# ---------------------------------------------------------------------------

def test_quantile_bins_5_elements_2_bins():
    # edges = np.percentile([1,2,3,4,5], [0, 50, 100]) = [1, 3, 5]
    # unique edges = [1, 3, 5]; interior = [3]
    # searchsorted([3], [1,2,3,4,5], side='right') = [0, 0, 1, 1, 1]
    result = _quantile_bins(np.array([1, 2, 3, 4, 5]), n_bins=2)
    expected = np.array([0, 0, 1, 1, 1])
    np.testing.assert_array_equal(result, expected)


def test_quantile_bins_returns_integer_labels():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    result = _quantile_bins(y, n_bins=2)
    assert result.dtype in (np.int64, np.int32, np.intp)
    assert set(result).issubset({0, 1})


def test_quantile_bins_constant_falls_back_to_median_split():
    # All same value — can't form edges, should fallback to median split
    y = np.array([5.0, 5.0, 5.0, 5.0])
    result = _quantile_bins(y, n_bins=4)
    # Fallback: (arr > median).astype(int), median=5, so all False → all 0
    assert all(r == 0 for r in result)


# ---------------------------------------------------------------------------
# _crosstab
# ---------------------------------------------------------------------------

def test_crosstab_2x2_exact():
    # a=[0,0,1,1], b=[0,1,0,1] → row_labels=[0,1], col_labels=[0,1]
    # table[0,0]=1, table[0,1]=1, table[1,0]=1, table[1,1]=1
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 1, 0, 1])
    table, row_labels, col_labels = _crosstab(a, b)
    assert row_labels == [0, 1]
    assert col_labels == [0, 1]
    np.testing.assert_array_equal(table, [[1, 1], [1, 1]])


def test_crosstab_perfect_association():
    # a=[0,0,1,1], b=[0,0,1,1] → only diagonal cells filled
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 0, 1, 1])
    table, row_labels, col_labels = _crosstab(a, b)
    np.testing.assert_array_equal(table, [[2, 0], [0, 2]])


# ---------------------------------------------------------------------------
# _cramers_v
# ---------------------------------------------------------------------------

def test_cramers_v_uniform_table_is_zero():
    # 2x2 table all 25 → no association → V=0
    presence = np.array([0, 0, 1, 1] * 25)
    groups = np.array([0, 1, 0, 1] * 25)
    v = _cramers_v(presence, groups)
    assert abs(v) < 1e-12


def test_cramers_v_perfect_association_is_one():
    # Perfect 2x2: [0,0] and [1,1] exclusively → V=1
    presence = np.array([0, 0, 1, 1])
    groups = np.array([0, 0, 1, 1])
    v = _cramers_v(presence, groups)
    assert abs(v - 1.0) < 1e-12


def test_cramers_v_single_group_returns_zero():
    # Only one group — crosstab is 2x1 → V=0.0 (fallback)
    presence = np.array([0, 1, 0, 1])
    groups = np.array([0, 0, 0, 0])
    v = _cramers_v(presence, groups)
    assert v == 0.0


# ---------------------------------------------------------------------------
# _effect_direction
# ---------------------------------------------------------------------------

def test_effect_direction_categorical_positive():
    # groups: A and B; B (last sorted) has higher presence → positive
    presence = np.array([0.0, 0.0, 1.0, 1.0])
    groups = np.array(["A", "A", "B", "B"], dtype=object)
    direction = _effect_direction(presence, groups, categorical=True)
    assert direction == "positive"


def test_effect_direction_categorical_negative():
    # A has higher presence than B → negative (last < first)
    presence = np.array([1.0, 1.0, 0.0, 0.0])
    groups = np.array(["A", "A", "B", "B"], dtype=object)
    direction = _effect_direction(presence, groups, categorical=True)
    assert direction == "negative"


def test_effect_direction_continuous_positive():
    # presence correlates positively with y
    presence = np.array([0.0, 0.0, 1.0, 1.0])
    y = [1.0, 2.0, 3.0, 4.0]
    direction = _effect_direction(presence, y, categorical=False)
    assert direction == "positive"


def test_effect_direction_continuous_none_constant_presence():
    # Constant presence → std < 1e-12 → "none"
    presence = np.array([1.0, 1.0, 1.0, 1.0])
    y = [1.0, 2.0, 3.0, 4.0]
    direction = _effect_direction(presence, y, categorical=False)
    assert direction == "none"


# ---------------------------------------------------------------------------
# _chi2_pvalue (real signature: presence, groups)
# ---------------------------------------------------------------------------

def test_chi2_pvalue_no_association_returns_large_p():
    # Uniform table → chi2~0 → p close to 1
    presence = np.array([0, 0, 1, 1] * 25)
    groups = np.array([0, 1, 0, 1] * 25)
    p = _chi2_pvalue(presence, groups)
    assert np.isfinite(p)
    assert p > 0.5


def test_chi2_pvalue_perfect_association_returns_small_p():
    # Perfect association → large chi2 → p very small
    presence = np.array([0] * 50 + [1] * 50)
    groups = np.array([0] * 50 + [1] * 50)
    p = _chi2_pvalue(presence, groups)
    assert np.isfinite(p)
    assert p < 0.001


def test_chi2_pvalue_single_group_returns_nan():
    # Only one group → crosstab is 2x1 → shape[1] < 2 → nan
    presence = np.array([0, 1, 0, 1])
    groups = np.array([0, 0, 0, 0])
    p = _chi2_pvalue(presence, groups)
    assert math.isnan(p)


# ---------------------------------------------------------------------------
# _pointbiserial_pvalue (vs. scipy)
# ---------------------------------------------------------------------------

def test_pointbiserial_pvalue_matches_scipy():
    presence = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    y = np.array([1.0, 1.2, 0.9, 0.8, 1.5, 1.1, 0.7, 1.3])
    our_p = _pointbiserial_pvalue(presence, y)
    _, scipy_p = pointbiserialr(presence, y)
    assert abs(our_p - scipy_p) < 1e-10


def test_pointbiserial_pvalue_constant_returns_nan():
    presence = np.array([1.0, 1.0, 1.0, 1.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = _pointbiserial_pvalue(presence, y)
    assert math.isnan(p)


# ---------------------------------------------------------------------------
# _validate_var_type
# ---------------------------------------------------------------------------

def test_validate_var_type_valid():
    _validate_var_type("continuous")
    _validate_var_type("categorical")


def test_validate_var_type_invalid():
    with pytest.raises(ValueError, match="var_type"):
        _validate_var_type("ordinal")


# ---------------------------------------------------------------------------
# Corpus.suggest_lexicon
# ---------------------------------------------------------------------------

def test_suggest_lexicon_returns_lexicon_result(sample_docs, sample_y):
    from ssdiff.results.lexicon_result import LexiconResult
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    result = corpus.suggest_lexicon(sample_y, top_k=5, min_docs=1)
    assert isinstance(result, LexiconResult)


def test_suggest_lexicon_top_k_respected(sample_docs, sample_y):
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    result = corpus.suggest_lexicon(sample_y, top_k=3, min_docs=1)
    assert len(result.tokens) <= 3


def test_suggest_lexicon_tokens_list_not_empty(sample_docs, sample_y):
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    result = corpus.suggest_lexicon(sample_y, min_docs=1)
    assert len(result.tokens) > 0
    assert all(isinstance(t, str) for t in result.tokens)


def test_suggest_lexicon_without_seeds_returns_the_rank_table():
    # y = 1..4 falls in two bins. "a" is in every document: corr 0, balanced
    # coverage 1. "c" is in one document of each bin: corr 0, balanced
    # coverage 0.5. "b" is in documents 0 and 2: corr -0.447, past corr_cap.
    corpus = Corpus([["a", "b", "c"], ["a"], ["a", "b"], ["a", "c"]], pretokenized=True)
    lex = corpus.suggest_lexicon([1.0, 2.0, 3.0, 4.0], min_docs=1, n_bins=2)
    assert lex.suggestions.columns == (
        "token", "freq", "cov_all", "cov_bal", "corr", "pvalue", "direction", "rank",
    )
    rows = lex.suggestions.to_records(cols=["token", "freq", "rank"])
    assert [r[:2] for r in rows] == [("a", 4), ("c", 2), ("b", 2)]
    assert [r[2] for r in rows] == pytest.approx([1.0, 0.5, 0.0])


# ---------------------------------------------------------------------------
# Corpus.suggest_lexicon with seeds
# ---------------------------------------------------------------------------

# "zebra" is not in the embedding, "cating" is in one document only, and
# "cated" is in the embedding but in no document.
SEED_DOCS = [
    ["cat", "cats", "kitten", "feline", "pet", "dog", "zebra"],
    ["cat", "cats", "kitten", "feline", "pet", "dog", "zebra"],
    ["cating", "kitten"],
    ["feline"],
]
SEED_Y = [1.0, 2.0, 3.0, 4.0]


def _angle_embeddings() -> Embeddings:
    """2-D vectors at known angles (degrees), all of length 3."""
    angles = {
        "cat": 0, "cated": 4, "cating": 5, "kitten": 10, "feline": 20,
        "pet": 45, "cats": 60, "dog": 90,
    }
    rad = np.deg2rad(list(angles.values()))
    mat = (3.0 * np.stack([np.cos(rad), np.sin(rad)], axis=1)).astype(np.float32)
    return Embeddings(list(angles), mat)


def _suggest_with_seeds(seeds, *, top_k, min_docs=2):
    corpus = Corpus(SEED_DOCS, pretokenized=True)
    return corpus.suggest_lexicon(
        SEED_Y, seeds=seeds, embeddings=_angle_embeddings(), top_k=top_k, min_docs=min_docs,
    )


def test_seed_rows_are_in_cosine_order_with_the_seed_first():
    rows = list(_suggest_with_seeds(["dog"], top_k=10).suggestions)
    assert [r.token for r in rows] == ["dog", "cats", "pet", "feline", "kitten", "cat"]
    assert [r.seed for r in rows] == ["dog"] * 6
    np.testing.assert_allclose(
        [r.cos for r in rows], np.cos(np.deg2rad([0, 30, 45, 70, 80, 90])), atol=1e-6,
    )


def test_seed_ending_outside_top_k_is_listed():
    # "pet" is closer to "cat" than "cats" is, and is the third neighbour.
    lex = _suggest_with_seeds(["cat"], top_k=2)
    assert [r.token for r in lex.suggestions] == ["cat", "kitten", "feline", "cats"]


def test_seed_ending_absent_from_corpus_or_below_min_docs_is_not_listed():
    lex = _suggest_with_seeds(["cat"], top_k=10)
    assert "cats" in lex.tokens
    assert "cated" not in lex.tokens
    assert "cating" not in lex.tokens


def test_seeded_result_with_no_rows_keeps_the_seed_columns():
    # No token is in 50 documents, so nothing is listed.
    lex = _suggest_with_seeds(["cat"], top_k=10, min_docs=50)
    assert len(lex.suggestions) == 0
    assert lex.suggestions.columns[:3] == ("seed", "token", "cos")


@pytest.mark.parametrize("seed, expected", [
    ("bake", ["bake", "baked", "baking", "bakes"]),
    ("try", ["try", "tried", "tries", "trying"]),
    ("box", ["box", "boxed", "boxes"]),
])
def test_seed_endings_are_listed_in_cosine_order(seed, expected):
    # Each list is in angle order from its seed; top_k=0 leaves only the
    # seed and its endings.
    angles = {
        "bake": 0, "baked": 10, "baking": 20, "bakes": 30,
        "try": 90, "tried": 100, "tries": 110, "trying": 120,
        "box": 200, "boxed": 210, "boxes": 220,
    }
    rad = np.deg2rad(list(angles.values()))
    emb = Embeddings(list(angles), np.stack([np.cos(rad), np.sin(rad)], axis=1).astype(np.float32))
    corpus = Corpus([list(angles)] * 2, pretokenized=True)
    lex = corpus.suggest_lexicon([1.0, 2.0], seeds=[seed], embeddings=emb, top_k=0, min_docs=2)
    assert [r.token for r in lex.suggestions] == expected


def test_token_listed_for_two_seeds_has_one_row_with_the_higher_cosine():
    # "cats" is 60 degrees from "cat" and 30 degrees from "dog".
    lex = _suggest_with_seeds(["cat", "dog"], top_k=10)
    assert lex.tokens == ["cat", "dog", "kitten", "feline", "cats", "pet"]
    assert len(lex.suggestions) == lex.stats.n_tokens == 6
    by_token = {r.token: r for r in lex.suggestions}
    assert by_token["cats"].seed == "dog"
    assert by_token["cats"].cos == pytest.approx(np.cos(np.deg2rad(30)), abs=1e-6)
    assert by_token["kitten"].seed == "cat"


def test_seeds_with_the_same_vector_each_have_their_own_row():
    emb = Embeddings(["colour", "color"], np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))
    corpus = Corpus([["colour", "color"]] * 2, pretokenized=True)
    lex = corpus.suggest_lexicon(
        [1.0, 2.0], seeds=["colour", "color"], embeddings=emb, top_k=5, min_docs=2,
    )
    assert lex.suggestions.to_records(cols=["seed", "token"]) == [
        ("colour", "colour"), ("color", "color"),
    ]


def test_a_repeated_seed_gives_the_table_of_the_seed_given_once():
    once = _suggest_with_seeds(["cat"], top_k=10)
    twice = _suggest_with_seeds(["cat", "cat"], top_k=10)
    assert twice.suggestions.to_records(cols=["seed", "token", "cos"]) == (
        once.suggestions.to_records(cols=["seed", "token", "cos"])
    )


@pytest.mark.parametrize("kwargs, error, match", [
    ({"seeds": "cat", "embeddings": _angle_embeddings()}, TypeError, "single string"),
    ({"seeds": [], "embeddings": _angle_embeddings()}, ValueError, "seeds is empty"),
    ({"seeds": ["cat"]}, ValueError, "needs embeddings"),
    ({"embeddings": _angle_embeddings()}, ValueError, "only used with seeds"),
    ({"seeds": ["cat", "nope", "nada"], "embeddings": _angle_embeddings()},
     ValueError, "nope.*nada"),
    ({"seeds": ["cat"], "embeddings": _angle_embeddings(), "top_k": -1}, ValueError, "top_k"),
])
def test_suggest_lexicon_seed_argument_errors(kwargs, error, match):
    corpus = Corpus(SEED_DOCS, pretokenized=True)
    with pytest.raises(error, match=match):
        corpus.suggest_lexicon(SEED_Y, **kwargs)


def test_suggest_lexicon_seeds_reject_a_corpus_tokenized_for_other_embeddings(wordpack_path):
    corpus = Corpus(["I like the dog."] * 4, embeddings=Embeddings.load(wordpack_path))
    with pytest.raises(ValueError, match="not a wordpack"):
        corpus.suggest_lexicon(SEED_Y, seeds=["dog"], embeddings=_angle_embeddings())


def test_suggest_lexicon_seeds_without_y_list_the_same_neighbours():
    with_y = _suggest_with_seeds(["cat", "dog"], top_k=10)
    lex = Corpus(SEED_DOCS, pretokenized=True).suggest_lexicon(
        None, seeds=["cat", "dog"], embeddings=_angle_embeddings(), top_k=10, min_docs=2,
    )
    free = ["seed", "token", "cos", "freq", "cov_all"]
    assert lex.suggestions.to_records(cols=free) == with_y.suggestions.to_records(cols=free)
    outcome = ["cov_bal", "corr", "pvalue", "direction", "rank"]
    assert set(lex.suggestions.to_records(cols=outcome)) == {(None,) * 5}


def test_suggest_lexicon_without_y_or_seeds_raises():
    with pytest.raises(ValueError, match="needs an outcome"):
        Corpus(SEED_DOCS, pretokenized=True).suggest_lexicon(None)


# ---------------------------------------------------------------------------
# Corpus.token_stats
# ---------------------------------------------------------------------------

def test_token_stats_returns_list_of_dicts(sample_docs, sample_y):
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    lex = ["kraj", "narod", "panstwo"]
    rows = corpus.token_stats(sample_y, lex)
    assert isinstance(rows, list)
    assert len(rows) == 3
    for row in rows:
        assert isinstance(row, dict)
        assert "token" in row
        assert "freq" in row
        assert "cov_all" in row
        assert "rank" in row
        assert "direction" in row


def test_token_stats_freq_correct(sample_docs, sample_y):
    # From conftest: "kraj" appears in docs 0, 3, 6 → freq=3
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    rows = corpus.token_stats(sample_y, ["kraj"])
    kraj_row = next(r for r in rows if r["token"] == "kraj")
    assert kraj_row["freq"] == 3


def test_token_stats_cov_all_correct(sample_docs, sample_y):
    # 8 docs, "kraj" in 3 → cov_all = 3/8 = 0.375
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    rows = corpus.token_stats(sample_y, ["kraj"])
    kraj_row = next(r for r in rows if r["token"] == "kraj")
    assert abs(kraj_row["cov_all"] - 3 / 8) < 1e-10


def test_token_stats_without_y_gives_the_counts_sorted_by_freq(sample_docs, sample_y):
    # "kraj" and "narod" are in 3 documents, "panstwo" in 2.
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    lex = ["panstwo", "kraj", "narod"]
    rows = corpus.token_stats(None, lex)
    assert [r["token"] for r in rows] == ["kraj", "narod", "panstwo"]
    with_y = {r["token"]: r for r in corpus.token_stats(sample_y, lex)}
    for r in rows:
        assert (r["freq"], r["cov_all"]) == (with_y[r["token"]]["freq"], with_y[r["token"]]["cov_all"])
        assert [r[k] for k in ("cov_bal", "corr", "rank", "pvalue", "direction")] == [None] * 5


# ---------------------------------------------------------------------------
# Corpus.coverage_summary
# ---------------------------------------------------------------------------

def test_coverage_summary_returns_dict(sample_docs, sample_y):
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    summary = corpus.coverage_summary(sample_y, {"kraj"})
    assert isinstance(summary, dict)
    assert "docs_any" in summary
    assert "cov_all" in summary
    assert "q1" in summary
    assert "q4" in summary


def test_coverage_summary_kraj_fraction(sample_docs, sample_y):
    # "kraj" appears in docs 0, 3, 6 → 3/8 docs = 0.375
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    summary = corpus.coverage_summary(sample_y, {"kraj"})
    assert summary["docs_any"] == 3
    assert abs(summary["cov_all"] - 3 / 8) < 1e-10


def test_coverage_summary_full_lexicon_all_docs(sample_docs, sample_y):
    # All docs contain at least one of kraj/narod/panstwo → docs_any=8
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    summary = corpus.coverage_summary(sample_y, {"kraj", "narod", "panstwo"})
    assert summary["docs_any"] == 8
    assert abs(summary["cov_all"] - 1.0) < 1e-10


def test_coverage_summary_without_y_gives_the_counts_only(sample_docs, sample_y):
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    lex = {"kraj", "narod"}
    # var_type is ignored without y, so there is no group_cov.
    summary = corpus.coverage_summary(None, lex, var_type="categorical")
    with_y = corpus.coverage_summary(sample_y, lex)
    free = ("docs_any", "cov_all", "hits_mean", "hits_median", "types_mean", "types_median")
    assert {k: summary[k] for k in free} == {k: with_y[k] for k in free}
    assert (summary["q1"], summary["q4"], summary["corr_any"]) == (None, None, None)
    assert "group_cov" not in summary


# ---------------------------------------------------------------------------
# Corpus.evaluate_lexicon
# ---------------------------------------------------------------------------

def test_evaluate_lexicon_returns_lexicon_result(sample_docs, sample_y):
    from ssdiff.results.lexicon_result import LexiconResult
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    result = corpus.evaluate_lexicon(sample_y, ["kraj", "narod"])
    assert isinstance(result, LexiconResult)
    assert result.summary is not None


def test_evaluate_lexicon_empty_lexicon_graceful(sample_docs, sample_y):
    # Empty lexicon should not raise — zero-token case
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    result = corpus.evaluate_lexicon(sample_y, [])
    assert result is not None
    assert len(result.tokens) == 0
    assert result.summary is not None
    # cov_all=0 since no tokens; docs_any=0
    assert result.summary.docs_any == 0
    assert result.summary.cov_all == 0.0


def test_evaluate_lexicon_counts_a_repeated_token_once(sample_docs, sample_y):
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    result = corpus.evaluate_lexicon(sample_y, ["kraj", "kraj", "narod"])
    assert result.stats.n_tokens == len(result.tokens) == len(result.suggestions) == 2


def test_evaluate_lexicon_summary_matches_coverage_summary(sample_docs, sample_y):
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    lex = ["kraj", "narod", "panstwo"]
    result = corpus.evaluate_lexicon(sample_y, lex)
    direct = corpus.coverage_summary(sample_y, lex)
    assert result.summary.docs_any == direct["docs_any"]
    assert abs(result.summary.cov_all - direct["cov_all"]) < 1e-12


def test_evaluate_lexicon_without_y_keeps_every_document(sample_docs, sample_y_with_nan):
    # The NaN in sample_y_with_nan drops document 3, which holds "kraj".
    corpus = Corpus(sample_docs, pretokenized=True, lang="pl")
    with_nan = corpus.evaluate_lexicon(sample_y_with_nan, ["kraj", "narod"])
    lex = corpus.evaluate_lexicon(None, ["kraj", "narod"])
    assert (with_nan.stats.n_docs, with_nan.summary.docs_any) == (7, 5)
    assert (lex.stats.n_docs, lex.summary.docs_any) == (8, 6)
    assert (lex.summary.q1, lex.summary.q4, lex.summary.corr_any) == (None, None, None)
    assert [s.rank for s in lex.suggestions] == [None, None]


def test_lexicon_results_without_y_render(sample_docs, tmp_path):
    evaluated = Corpus(sample_docs, pretokenized=True, lang="pl").evaluate_lexicon(
        None, ["kraj", "narod"],
    )
    seeded = Corpus(SEED_DOCS, pretokenized=True).suggest_lexicon(
        None, seeds=["cat"], embeddings=_angle_embeddings(), min_docs=2,
    )
    for lex in (evaluated, seeded):
        repr(lex)
        lex._repr_html_()
        report = lex.report()
        assert all(t in report.to_text() for t in lex.tokens)
        report.to_html()
        lex.suggestions.save(tmp_path / "suggestions.csv")
