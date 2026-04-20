"""ContinuousResult.report() includes a Fit info section."""

from __future__ import annotations


def test_report_pls_has_fit_info(pls_result):
    text = pls_result.report().to_text()
    assert "Fit info" in text
    assert "p_method" in text
    assert "n_components" in text
    assert "random_state" in text


def test_report_pca_ols_has_fit_info_with_pca_k_source(pcaols_result):
    text = pcaols_result.report().to_text()
    assert "Fit info" in text
    assert "pca_k" in text
    assert "pca_k_source" in text


def test_pca_k_source_is_sweep_when_fixed_k_none(pcaols_result_sweep):
    assert pcaols_result_sweep.fit_info.pca_k_source == "sweep"


def test_pca_k_source_is_fixed_when_fixed_k_passed(pcaols_result):
    assert pcaols_result.fit_info.pca_k_source == "fixed"
