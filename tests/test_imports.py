"""Tests for clean imports — no pandas at import time."""

import subprocess
import sys


def _check_no_module_imported(
    module_substring: str,
    imports: list[str],
    exclude: list[str] | None = None,
) -> None:
    """Run imports in a subprocess and verify module_substring is not in sys.modules."""
    exclude = exclude or []
    exclude_cond = " and ".join(
        f"'{ex}' not in m" for ex in exclude
    ) if exclude else "True"
    code = (
        "import sys; "
        + "; ".join(f"import {m}" for m in imports)
        + f"; mods = [m for m in sys.modules if '{module_substring}' in m and {exclude_cond}]; "
        f"print(','.join(mods) if mods else 'CLEAN')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=30,
    )
    output = result.stdout.strip()
    assert output == "CLEAN", (
        f"'{module_substring}' found in sys.modules after importing "
        f"{imports}: {output}"
    )


CORE_IMPORTS = [
    "ssdiff.embeddings",
    "ssdiff.ssd",
    "ssdiff.corpus",
    "ssdiff.utils.math",
    "ssdiff.utils.vectors",
    "ssdiff.utils.neighbors",
    "ssdiff.backends.pls",
]


def test_no_pandas_in_ssdiff_code():
    """ssdiff core imports should not pull in pandas."""
    # tqdm registers a _tqdm_pandas shim at import time — not a real pandas import
    _check_no_module_imported("pandas", CORE_IMPORTS, exclude=["tqdm"])


def test_no_sklearn_at_import():
    """ssdiff core imports should not pull in sklearn."""
    _check_no_module_imported("sklearn", CORE_IMPORTS)


def test_public_api():
    """All documented public names should be importable."""
    from ssdiff import SSD, Corpus, Embeddings
    assert all([Embeddings, Corpus, SSD])


def test_version():
    import ssdiff
    assert isinstance(ssdiff.__version__, str)
    assert len(ssdiff.__version__) > 0
