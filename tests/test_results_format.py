"""D11 APA-inspired formatting primitives — exact-string assertions."""

import pytest

from ssdiff.results.format import (
    fmt_count,
    fmt_d,
    fmt_p,
    fmt_pct,
    fmt_r,
    fmt_sig,
    fmt_table,
    truncate,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0004, "<.001"),
        (0.0009999, "<.001"),
        (0.001, ".001"),
        (0.037, ".037"),
        (0.5, ".500"),
        (1.0, "1.000"),
    ],
)
def test_fmt_p(value, expected):
    assert fmt_p(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.42, ".42"),
        (0.423, ".42"),
        (-0.18, "−.18"),       # en-dash minus
        (0.0, ".00"),
        (1.0, "1.00"),          # exact ±1 keeps leading digit
        (-1.0, "−1.00"),
    ],
)
def test_fmt_r(value, expected):
    assert fmt_r(value) == expected


def test_fmt_r_signed():
    assert fmt_r(0.42, signed=True) == "+.42"
    assert fmt_r(-0.18, signed=True) == "−.18"
    assert fmt_r(0.0, signed=True) == "+.00"


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.73, "0.73"),
        (1.24, "1.24"),
        (-0.05, "-0.05"),
        (0.0, "0.00"),
    ],
)
def test_fmt_d(value, expected):
    assert fmt_d(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.3014621812015017, "0.301"),
        (5.605681417160174e-09, "5.61e-09"),
        (1234567.0, "1.23e+06"),
        (0.0, "0"),
        (1.0, "1"),
        (-0.000123, "-0.000123"),
        (None, ""),
        (True, "True"),
        (False, "False"),
        (597, "597"),
        (float("nan"), "nan"),
        (float("inf"), "inf"),
        (float("-inf"), "-inf"),
    ],
)
def test_fmt_sig(value, expected):
    assert fmt_sig(value) == expected


def test_fmt_sig_custom_digits():
    assert fmt_sig(0.3014621812015017, digits=4) == "0.3015"
    assert fmt_sig(0.3014621812015017, digits=2) == "0.3"


@pytest.mark.parametrize(
    "value,expected",
    [
        (1240, "1,240"),
        (0, "0"),
        (1_000_000, "1,000,000"),
    ],
)
def test_fmt_count(value, expected):
    assert fmt_count(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.374, "37.4%"),
        (0.0, "0.0%"),
        (1.0, "100.0%"),
    ],
)
def test_fmt_pct(value, expected):
    assert fmt_pct(value) == expected


def test_truncate_respects_width():
    s = "x" * 80
    assert truncate(s, 70) == "x" * 69 + "…"
    assert truncate("short", 70) == "short"


def test_fmt_table_alignment_and_gutter():
    rows = [
        ["pos", 1, ".42"],
        ["neg", 1, "-.41"],
    ]
    headers = ["side", "rank", "cos_b"]
    numeric = [False, True, True]
    out = fmt_table(rows, headers=headers, numeric=numeric)
    # Gutter is 2 spaces between columns
    assert "  " in out.splitlines()[0]
    # Header matches first data row width
    lines = out.splitlines()
    assert len(lines[0]) == len(lines[1])


def test_fmt_table_fits_80_cols_with_truncation():
    long_text = "x" * 200
    rows = [[long_text, 1]]
    out = fmt_table(rows, headers=["text", "rank"], numeric=[False, True],
                    max_width=80, text_truncate=70)
    for line in out.splitlines():
        assert len(line) <= 80
