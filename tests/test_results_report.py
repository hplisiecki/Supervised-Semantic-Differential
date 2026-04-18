"""Report builder — D7 (builder), D8 (path rule)."""

import pytest

from ssdiff.results.report import CITATION, Report, Section


def test_citation_text_is_apa_formatted():
    assert "Plisiecki" in CITATION
    assert "2025" in CITATION
    assert "doi.org/10.31234/osf.io/gvrsb_v1" in CITATION


def test_report_to_text_contains_all_sections():
    sections = [
        Section(title="Stats", kind="kv", rows=[("r2", ".47"), ("n", "1,240")]),
        Section(title="Top words", kind="table",
                headers=["rank", "word"], rows=[[1, "excellent"]],
                numeric=[True, False]),
    ]
    r = Report(title="PLSResult", sections=sections)
    out = r.to_text()
    assert "Stats" in out
    assert "Top words" in out
    assert "r2" in out
    assert "excellent" in out


def test_report_includes_citation_footer():
    r = Report(title="X", sections=[])
    assert "Plisiecki" in r.to_text()
    assert "doi.org/10.31234/osf.io/gvrsb_v1" in r.to_text()


def test_report_repr_equals_to_text():
    import ssdiff
    r = Report(title="X", sections=[])
    ssdiff.set_repr_hints(False)
    try:
        assert repr(r) == r.to_text()
    finally:
        ssdiff.set_repr_hints(True)


def test_report_save_markdown_uses_atx_headers(tmp_path):
    r = Report(title="MyTitle",
               sections=[Section(title="S1", kind="kv", rows=[("k", "v")])])
    p = tmp_path / "r.md"
    r.save(str(p))
    md = p.read_text(encoding="utf-8")
    assert "# MyTitle" in md
    assert "## S1" in md


def test_report_save_dispatches_by_extension(tmp_path):
    r = Report(title="X", sections=[])
    for ext, needle in [(".txt", "Plisiecki"), (".md", "# X"), (".html", "<h1>X</h1>")]:
        p = tmp_path / f"r{ext}"
        r.save(str(p))
        content = p.read_text(encoding="utf-8")
        assert needle in content


def test_report_save_unknown_extension_raises(tmp_path):
    r = Report(title="X", sections=[])
    with pytest.raises(ValueError, match=r"[Uu]nsupported extension"):
        r.save(str(tmp_path / "r.xyz"))
