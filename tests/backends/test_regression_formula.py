import pytest

from ssdiff.backends.regression import (
    check_text_terms,
    expand,
    label_of,
    null_formula,
    term_groups,
    text_labels,
    text_names,
    to_placeholders,
    with_text_term,
)


def test_names_and_append():
    assert text_names("y ~ a + sex*ssd() + ssd(diary):b") == [None, "diary"]
    assert with_text_term("y ~ age") == "y ~ age + ssd()"
    assert with_text_term("y ~ sex*ssd()") == "y ~ sex*ssd()"
    assert with_text_term("y ~") == "y ~ ssd()"


def test_text_in_random_effect_raises():
    with pytest.raises(ValueError, match="random"):
        check_text_terms("y ~ x + (ssd() | id)")
    check_text_terms("y ~ x + ssd() + (1 | id)")


def test_null_formula_keeps_random_and_drops_labels():
    f = to_placeholders("y ~ age + sex*ssd() + (1 | id)")
    assert null_formula(f, {"_ssd_text", "sex:_ssd_text"}) == "y ~ age + sex + (1 | id)"
    assert null_formula(f, {"sex:_ssd_text"}) == "y ~ age + sex + _ssd_text + (1 | id)"


def test_null_formula_copies_nested_random_and_offset():
    f = to_placeholders("y ~ ssd() + offset(log(n)) + (1 + (x) | id)")
    assert null_formula(f, {"_ssd_text"}) == "y ~ 1 + (1 + (x) | id) + offset(log(n))"


def test_null_formula_intercept_removed():
    assert null_formula("y ~ 0 + x + _ssd_text", {"_ssd_text"}) == "y ~ 0 + x"
    with pytest.raises(ValueError, match="no fixed term"):
        null_formula(to_placeholders("y ~ 0 + sex:ssd()"), {"sex:_ssd_text"})


def test_expand():
    assert expand("y ~ sex*_ssd_text", {"_ssd_text": ["_ssd_pca_1", "_ssd_pca_2"]}) == \
        "y ~ sex*(_ssd_pca_1 + _ssd_pca_2)"


def test_term_groups_both_orders():
    comps = {"_ssd_pca_1": ("ssd(e)", 0), "_ssd_pca_2": ("ssd(e)", 1)}
    names = ["(Intercept)", "age", "sexM", "_ssd_pca_1", "_ssd_pca_2", "sexM:_ssd_pca_1", "sexM:_ssd_pca_2"]
    assert term_groups(names, comps) == {"ssd(e)": [(0, 3), (1, 4)], "sexM:ssd(e)": [(0, 5), (1, 6)]}
    assert list(term_groups(["_ssd_pca_1:sexM", "_ssd_pca_2:sexM"], comps)) == ["ssd(e):sexM"]


def test_label_of():
    labels = ["age", "sex", "ssd(e)", "sex:ssd(e)"]
    assert label_of("sexM:ssd(e)", labels) == "sex:ssd(e)"
    assert label_of("ssd(e)", labels) == "ssd(e)"
    assert label_of("ssd(e):sexM", ["ssd(e)", "ssd(e):sex"]) == "ssd(e):sex"


def test_text_labels():
    assert text_labels(["age", "_ssd_text", "sex:_ssd_text", "_ssd_text2"], "_ssd_text") == \
        ["_ssd_text", "sex:_ssd_text"]
