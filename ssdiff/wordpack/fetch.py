"""Find a wordpack by full name: download it once into a cache and return its path."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import sys
import tempfile
import urllib.request
from pathlib import Path

from ssdiff.lang_config import WORDPACK_TAGS
from ssdiff.wordpack.reader import FORMAT_VERSION

# Base URLs, tried in order. Every host holds the same files under the same names.
HOSTS = ["https://resources.s-on.pl/wordpacks"]

# A request that fails is sent to the same host this many more times before the next host is tried.
RETRIES = 2

# A full name is `<name>-<version>` with a whole-number version. Pack names
# never end in a hyphen followed only by digits.
_FULL_NAME = re.compile(r"([a-z0-9-]+)-([0-9]+)")


def is_full_name(text: str) -> bool:
    """Whether `text` has the shape of a pack's full name."""
    return _FULL_NAME.fullmatch(text) is not None


def cache_dir() -> Path:
    """The directory pack files are kept in.

    Named after the format, not after ssdiff, so any wordpack reader can
    share one copy of a pack. It is in the home directory on every system,
    not in the platform's cache directory, because Chromium browsers never
    let a web page open `~/.cache`, `~/Library` or `%LOCALAPPDATA%`, and a
    reader running in a browser shares this directory too.
    """
    if os.environ.get("WORDPACK_CACHE"):
        return Path(os.environ["WORDPACK_CACHE"])
    return Path.home() / ".wordpacks"


def _open(url: str):
    from ssdiff import __version__

    # Cloudflare answers 403 to urllib's default User-Agent.
    request = urllib.request.Request(url, headers={"User-Agent": f"ssdiff/{__version__}"})
    # urllib applies the timeout to each blocking operation, not to the whole download.
    return urllib.request.urlopen(request, timeout=60)


def _failed(full: str, host: str, error: Exception, attempt: int, failures: list[str]) -> None:
    """Say on stderr that `host` is asked again; after its last try, add the failure to `failures` instead."""
    if attempt <= RETRIES:
        print(f"wordpack {full}: {host} failed ({error}); trying again ({attempt} of {RETRIES})", file=sys.stderr)
    else:
        failures.append(f"  {host}: {error}")


def _index(full: str) -> dict:
    """`index.json` from the first host that gives it in the right shape. OSError naming every host when none does."""
    failures = []
    for host in HOSTS:
        for attempt in range(1, RETRIES + 2):
            try:
                with _open(f"{host}/index.json") as response:
                    index = json.loads(response.read())
                if not isinstance(index, dict) or not isinstance(index.get("packs"), dict):
                    raise ValueError("index.json does not have the expected contents")
                return index
            # A connection cut in the middle of an answer raises http.client.IncompleteRead, which is not an OSError.
            except (OSError, http.client.HTTPException, ValueError) as e:
                _failed(full, host, e, attempt, failures)
    raise OSError(
        f"Wordpack {full!r} is not in the cache ({cache_dir()}) and cannot be downloaded: "
        f"no host answered for index.json in {RETRIES + 1} tries each:\n" + "\n".join(failures)
    )


def _download(full: str, entry: dict, dest: Path) -> None:
    """Download the file `entry` describes to `dest`. It appears there only after its size and SHA-256 are checked."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading wordpack {full} ({entry['size'] / 1e6:,.0f} MB) to {dest}", file=sys.stderr)
    failures = []
    for host in HOSTS:
        for attempt in range(1, RETRIES + 2):
            fd, part = tempfile.mkstemp(dir=dest.parent, suffix=".part")
            try:
                digest, size = hashlib.sha256(), 0
                with os.fdopen(fd, "wb") as f, _open(f"{host}/{entry['file']}") as response:
                    while chunk := response.read(1 << 20):
                        f.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                if size != entry["size"] or digest.hexdigest() != entry["sha256"]:
                    raise OSError(
                        f"got {size} bytes with SHA-256 {digest.hexdigest()}; "
                        f"the index lists {entry['size']} bytes with SHA-256 {entry['sha256']}"
                    )
                # mkstemp makes the file readable by its owner only; a cache directory shared by a group needs more.
                os.chmod(part, 0o644)
                try:
                    os.replace(part, dest)
                except PermissionError:
                    # Windows refuses to replace a file another process has
                    # open: that process finished the same download first.
                    if not dest.exists():
                        raise
            except (OSError, http.client.HTTPException) as e:
                _failed(full, host, e, attempt, failures)
            else:
                print(f"downloaded wordpack {full}", file=sys.stderr)
                return
            finally:
                if os.path.exists(part):
                    os.remove(part)
    raise OSError(
        f"Cannot fetch wordpack {full!r}: no host gave its file in {RETRIES + 1} tries each:\n" + "\n".join(failures)
    )


def resolve(full: str) -> str:
    """The path of the cached pack file for a full name, downloading the file when it is missing.

    A full name that is in the cache makes no network request.
    """
    path = cache_dir() / f"{full}.wordpack"
    if path.exists():
        return str(path)
    index = _index(full)
    name, version = _FULL_NAME.fullmatch(full).groups()
    pack = index["packs"].get(name)
    if pack is None:
        raise ValueError(f"No wordpack named {full!r}. Tags: {', '.join(WORDPACK_TAGS)}.")
    entry = pack["versions"].get(version)
    if entry is None:
        versions = ", ".join(sorted(pack["versions"], key=int))
        raise ValueError(f"No wordpack named {full!r}. Versions of {name}: {versions}.")
    if entry["format_version"] != FORMAT_VERSION:
        raise ValueError(
            f"Wordpack {full} has format version {entry['format_version']}; "
            f"this ssdiff reads format version {FORMAT_VERSION}. Upgrade ssdiff."
        )
    _download(full, entry, path)
    return str(path)
