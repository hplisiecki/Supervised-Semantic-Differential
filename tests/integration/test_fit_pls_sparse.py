import numpy as np
import pytest


def test_keep_int_fits_and_records_keep(ssd_instance):
    res = ssd_instance.fit_pls(k=1, keep=4, n_splits=10, random_state=0)
    assert res.fit_info.keep == 4
    W = res.component_weights
    assert np.count_nonzero(W[:, 0]) == 4

def test_keep_auto_fits(ssd_instance):
    res = ssd_instance.fit_pls(k=2, keep="auto", n_splits=10, random_state=0)
    assert 1 <= res.fit_info.keep <= 8

def test_keep_with_k_auto_raises(ssd_instance):
    with pytest.raises(ValueError, match="explicit integer k"):
        ssd_instance.fit_pls(k="auto", keep=4)

@pytest.mark.parametrize("keep", [0, 9])
def test_keep_out_of_range_raises(ssd_instance, keep):
    with pytest.raises(ValueError, match=r"keep must be in \[1, 8\]"):
        ssd_instance.fit_pls(k=1, keep=keep)

def test_sparse_test_name_and_no_rerun(ssd_instance):
    res = ssd_instance.fit_pls(k=2, keep=4, n_splits=10, random_state=0)
    assert res.test.name == "split_nb_seq"
    with pytest.raises(RuntimeError, match="spls1_find_k_sequence"):
        res.test(n_splits=10)

def test_dense_keep_is_none(ssd_instance):
    assert ssd_instance.fit_pls(k=1, n_splits=10).fit_info.keep is None

def test_k_auto_k_star_zero_raises(ssd_instance, monkeypatch):
    import dataclasses
    import plskit
    real = plskit.pls1_find_k_optimal
    def fake(*a, **kw):
        return dataclasses.replace(real(*a, **kw), k_star=0)
    monkeypatch.setattr(plskit, "pls1_find_k_optimal", fake)
    with pytest.raises(ValueError, match="k_star"):
        ssd_instance.fit_pls(k="auto", n_splits=10)
