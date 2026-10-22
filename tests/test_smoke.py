"""Smoke tests for ssdiff — import hygiene, public API, version, __all__."""

import subprocess
import sys

import ssdiff


def test_no_heavy_imports_on_load():
    """import ssdiff must not pull in pandas, sklearn, matplotlib, or scipy."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ssdiff; import sys; "
            "print(sorted(m for m in ('pandas','sklearn','matplotlib','scipy') if m in sys.modules))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "[]", (
        f"Heavy modules leaked into sys.modules on 'import ssdiff': {proc.stdout.strip()}"
    )


def test_public_api_importable():
    """SSD and Embeddings must both be importable from ssdiff."""
    from ssdiff import SSD, Embeddings

    assert callable(SSD), "SSD should be a callable class"
    assert callable(Embeddings), "Embeddings should be a callable class"


def test_version_matches_pyproject():
    import re
    from pathlib import Path
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")

    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    assert match is not None

    assert ssdiff.__version__ == match.group(1)

def test_all_contains_core_names():
    """ssdiff.__all__ must exist and include SSD and Embeddings."""
    assert hasattr(ssdiff, "__all__"), "ssdiff does not define __all__"
    all_names = ssdiff.__all__
    for name in ("SSD", "Embeddings"):
        assert name in all_names, f"{name!r} missing from ssdiff.__all__"
