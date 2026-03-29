"""Tests for clean imports — no pandas/sklearn at import time."""

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
        f"import sys; "
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
    "ssdlite.embeddings",
    "ssdlite.ssd",
    "ssdlite.corpus",
    "ssdlite.utils.math",
    "ssdlite.utils.vectors",
    "ssdlite.utils.neighbors",
    "ssdlite.backends.pls",
]


def test_no_pandas_in_ssdlite_code():
    """ssdlite core imports should not pull in pandas."""
    # tqdm registers a _tqdm_pandas shim at import time — not a real pandas import
    _check_no_module_imported("pandas", CORE_IMPORTS, exclude=["tqdm"])


def test_no_sklearn_at_import():
    """ssdlite core imports should not pull in sklearn."""
    _check_no_module_imported("sklearn", CORE_IMPORTS)


def test_public_api():
    """All documented public names should be importable."""
    from ssdlite import Embeddings, Corpus, SSD, SSDGroup, SSDContrast
    assert all([Embeddings, Corpus, SSD, SSDGroup, SSDContrast])


def test_version():
    import ssdlite
    assert isinstance(ssdlite.__version__, str)
    assert len(ssdlite.__version__) > 0
