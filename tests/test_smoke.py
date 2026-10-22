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


def test_version_matches_pyproject():
    import re
    from pathlib import Path
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")

    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    assert match is not None

    assert ssdiff.__version__ == match.group(1)

def test_all_names_resolve():
    """Every name in ssdiff.__all__ exists, and the core classes are exported."""
    ns = {}
    exec("from ssdiff import *", ns)
    assert {"SSD", "Embeddings"} <= set(ssdiff.__all__) <= ns.keys()
