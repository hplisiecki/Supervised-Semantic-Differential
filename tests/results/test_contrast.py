"""ssdiff.contrast: difference of two unit gradients, words and clusters only, and its results page."""

import json

import numpy as np
import pytest

import ssdiff
from ssdiff.results.page import page_data, save_page

@pytest.fixture(scope="module")
def contrast_result(ssd_instance, pls_result, pcaols_result):
    return ssdiff.contrast(pls_result.gradient, pcaols_result.gradient, ssd_instance.embeddings,
                           labels=("pls", "ols"))


def test_beta_is_normalized_difference_of_unit_gradients():
    rng = np.random.default_rng(0)
    g1, g2 = rng.normal(size=5), 10 * rng.normal(size=5)
    c = ssdiff.contrast(g1, g2, None)
    u1, u2 = g1 / np.linalg.norm(g1), g2 / np.linalg.norm(g2)
    d = u1 - u2
    np.testing.assert_allclose(c.beta, d / np.linalg.norm(d))
    assert c.beta_norm == pytest.approx(1.0)
    assert c.stats.cos_g1_g2 == pytest.approx(u1 @ u2)
    assert c.stats.diff_norm == pytest.approx(np.linalg.norm(d))


def test_same_direction_and_length_mismatch_raise():
    g = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="same direction"):
        ssdiff.contrast(g, 4 * g, None)
    with pytest.raises(ValueError, match="same length"):
        ssdiff.contrast(g, g[:2], None)


def test_words_and_clusters(contrast_result):
    assert len(contrast_result.words.pos) > 0 and len(contrast_result.words.neg) > 0
    assert len(contrast_result.clusters) > 0
    assert len(contrast_result.clusters.words) > 0


@pytest.mark.parametrize("name", ["snippets", "docs", "alignment_scores"])
def test_document_views_raise(contrast_result, name):
    with pytest.raises(RuntimeError, match="no documents"):
        getattr(contrast_result, name)
    with pytest.raises(RuntimeError, match="no documents"):
        contrast_result.cluster_snippets(side="pos")


def test_repr_names_both_sides(contrast_result):
    assert "pls − ols" in repr(contrast_result)


def test_contrast_page(contrast_result, tmp_path):
    d = page_data(contrast_result)
    json.dumps(d, allow_nan=False)
    assert d["kind"] == "contrast" and d["report_format"] == 2
    assert d["texts"] == [{"name": None, "embeddings": d["texts"][0]["embeddings"], "settings": None,
                           "k": None, "sweep": None, "n_no_vector": None, "snippets": None}]
    assert d["texts"][0]["embeddings"]["name"] is not None
    assert set(d["tables"]) == {"stats"}
    assert d["tables"]["stats"]["columns"] == ["cos_g1_g2", "diff_norm"]
    [u] = d["units"]
    assert (u["key"], u["label"], u["pos_label"], u["neg_label"], u["text"]) == (
        "contrast", "pls − ols", "pls", "ols", 0)
    assert set(u["tables"]) == {"words", "clusters", "cluster_words"}
    save_page(contrast_result, tmp_path / "c.html")
