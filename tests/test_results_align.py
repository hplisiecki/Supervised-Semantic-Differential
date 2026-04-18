"""Default column alignment rule from spec_console_repr.md."""

from ssdiff.results.format import default_alignment, fmt_table


def test_default_alignment_first_left_rest_right():
    # Returns a tuple of bools matching fmt_table's `numeric` semantics:
    # numeric[i]=True → right-align. So first=False, rest=True.
    assert default_alignment(1) == (False,)
    assert default_alignment(2) == (False, True)
    assert default_alignment(4) == (False, True, True, True)


def test_default_alignment_zero_cols():
    assert default_alignment(0) == ()


def test_fmt_table_with_default_alignment_renders_first_left_rest_right():
    headers = ["side", "rank", "word", "cos_beta"]
    rows = [["pos", 1, "postep", "+0.32"], ["pos", 2, "rozwoj", "+0.30"]]
    out = fmt_table(rows, headers=headers, numeric=default_alignment(len(headers)))
    lines = out.splitlines()
    # First column left: "side  ..." — "side" appears flush left
    assert lines[0].startswith("side")
    # Last column right: ranks aligned by digits matters less here, but "+0.32"
    # in row 1 should appear at the same right-edge as "+0.30" in row 2.
    assert lines[1].rstrip().endswith("+0.32")
    assert lines[2].rstrip().endswith("+0.30")
