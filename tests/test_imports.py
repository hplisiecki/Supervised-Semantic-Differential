"""Tests for clean imports — no pandas/sklearn/matplotlib at import time."""

import subprocess
import sys

import pytest

CORE_IMPORTS = [
    "ssdiff.embeddings",
    "ssdiff.ssd",
    "ssdiff.corpus",
    "ssdiff.utils.math",
    "ssdiff.utils.vectors",
    "ssdiff.utils.neighbors",
    "ssdiff.backends.pls",
]


@pytest.fixture(scope="session")
def ssdiff_loaded_modules() -> list[str]:
    """Import CORE_IMPORTS in a fresh subprocess and return sys.modules.

    Running one subprocess and reusing its module list across the three
    import-hygiene tests avoids paying Python startup + ssdiff-import cost
    three times (~6-10s saved on the fast suite).
    """
    code = (
        "import sys; "
        + "; ".join(f"import {m}" for m in CORE_IMPORTS)
        + "; print('\\n'.join(sorted(sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=30,
    )
    result.check_returncode()
    return result.stdout.splitlines()


def test_no_pandas_in_ssdiff_code(ssdiff_loaded_modules):
    """ssdiff core imports should not pull in pandas."""
    # tqdm registers a _tqdm_pandas shim at import time — not a real pandas import
    bad = [m for m in ssdiff_loaded_modules if "pandas" in m and "tqdm" not in m]
    assert not bad, f"pandas found in sys.modules: {bad}"


def test_no_sklearn_at_import(ssdiff_loaded_modules):
    """ssdiff core imports should not pull in sklearn."""
    bad = [m for m in ssdiff_loaded_modules if "sklearn" in m]
    assert not bad, f"sklearn found in sys.modules: {bad}"


def test_no_matplotlib_at_import(ssdiff_loaded_modules):
    """ssdiff core imports should not pull in matplotlib."""
    bad = [m for m in ssdiff_loaded_modules if "matplotlib" in m]
    assert not bad, f"matplotlib found in sys.modules: {bad}"


def test_public_api():
    """All documented public names should be importable."""
    from ssdiff import SSD, Corpus, Embeddings
    assert all([Embeddings, Corpus, SSD])


def test_version():
    import ssdiff
    assert isinstance(ssdiff.__version__, str)
    assert len(ssdiff.__version__) > 0
