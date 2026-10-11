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
    """SSD, Corpus, and Embeddings must all be importable from ssdiff."""
    from ssdiff import Corpus, Embeddings, SSD  # noqa: F401

    assert callable(SSD), "SSD should be a callable class"
    assert callable(Corpus), "Corpus should be a callable class"
    assert callable(Embeddings), "Embeddings should be a callable class"


def test_version_matches_pyproject():
    from pathlib import Path
    import re
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")

    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    assert match is not None

    assert ssdiff.__version__ == match.group(1)

def test_all_contains_core_names():
    """ssdiff.__all__ must exist and include SSD, Corpus, and Embeddings."""
    assert hasattr(ssdiff, "__all__"), "ssdiff does not define __all__"
    all_names = ssdiff.__all__
    for name in ("SSD", "Corpus", "Embeddings"):
        assert name in all_names, f"{name!r} missing from ssdiff.__all__"


def test_wordpack_and_pretokenized_routes_never_import_spacy(wordpack_path):
    code = (
        "import sys\n"
        "from ssdiff import Corpus, Embeddings\n"
        f"emb = Embeddings.load({wordpack_path!r})\n"
        "Corpus(['I like the dog.'], embeddings=emb)\n"
        "Corpus([['dog']], pretokenized=True)\n"
        "print('spacy' in sys.modules)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == "False"


def test_without_spacy_the_wordpack_route_works_and_lang_gives_the_install_hint(wordpack_path):
    # A None entry in sys.modules makes `import spacy` raise ImportError,
    # as on a machine where spaCy is not installed.
    code = (
        "import sys\n"
        "sys.modules['spacy'] = None\n"
        "from ssdiff import Corpus, Embeddings\n"
        f"emb = Embeddings.load({wordpack_path!r})\n"
        "print(Corpus(['I like the dog.'], embeddings=emb).docs)\n"
        "try:\n"
        "    Corpus(['x'], lang='en')\n"
        "except ImportError as e:\n"
        "    print(e)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert proc.stdout.splitlines() == [
        "[['like', 'dog']]",
        "spacy is required for this method; install with: pip install ssdiff[spacy]",
    ]
