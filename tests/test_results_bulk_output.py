"""Per-view save() and report save dispatch (acceptance #6, #13)."""

import pytest

from ssdiff.results.continuous_result import PLSResult


def _mk():
    import numpy as np
    rng = np.random.default_rng(0)
    d, n = 4, 30
    beta = rng.standard_normal(d)
    x = rng.standard_normal((n, d))
    y = x @ beta + 0.1 * rng.standard_normal(n)
    return PLSResult(
        x=x, beta=beta, keep_mask=np.ones(n, dtype=bool),
        n_raw=n, n_kept=n, n_dropped=0,
        y_kept=y, _y_mean=np.array([y.mean()]), _y_scale=np.array([y.std() + 1e-12]),
        r2=0.47, pvalue=1e-4,
    )


def test_per_view_save_writes_files(tmp_path):
    """Per-view save() is how users export tables (Result has no bundle-save)."""
    r = _mk()
    r.stats.save(str(tmp_path / "stats.csv"))
    r.docs.save(str(tmp_path / "docs.csv"))
    assert (tmp_path / "stats.csv").is_file()
    assert (tmp_path / "docs.csv").is_file()


def test_view_save_to_json(tmp_path):
    r = _mk()
    r.docs.save(str(tmp_path / "docs.json"))
    assert (tmp_path / "docs.json").is_file()


def test_view_save_to_xlsx(tmp_path):
    pytest.importorskip("openpyxl")
    pytest.importorskip("pandas")
    r = _mk()
    r.docs.save(str(tmp_path / "docs.xlsx"))
    assert (tmp_path / "docs.xlsx").is_file()


def test_report_save_dispatches_by_extension(tmp_path):
    r = _mk()
    r.report().save(str(tmp_path / "r.md"))
    r.report().save(str(tmp_path / "r.txt"))
    r.report().save(str(tmp_path / "r.html"))
    r.report().save(str(tmp_path / "r.tex"))
    for name in ("r.md", "r.txt", "r.html", "r.tex"):
        assert (tmp_path / name).is_file()


def test_view_save_k_caps_rows_on_real_docs_view(tmp_path):
    """End-to-end: k= trims a live DocsView to the first k rows."""
    import csv
    r = _mk()
    p = tmp_path / "docs_k.csv"
    r.docs.save(str(p), k=5)
    rows = list(csv.reader(p.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 1 + 5  # header + 5 body rows


def test_scalar_view_save_k_is_silent_noop(tmp_path):
    """k= on a single-row ScalarView must not raise or drop the row."""
    import csv
    r = _mk()
    p = tmp_path / "stats.csv"
    r.stats.save(str(p), k=0)
    rows = list(csv.reader(p.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 2  # header + the stats row


def test_report_save_rejects_k_kwarg(tmp_path):
    """Report doesn't accept k= — size is fixed at report(top_words=..., clusters=...)."""
    r = _mk()
    rep = r.report()
    with pytest.raises(TypeError):
        rep.save(str(tmp_path / "r.md"), k=5)
