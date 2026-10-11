"""Find a wordpack by short tag or full name: download it once into a cache and return its path."""

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

from ssdiff._wordpack_reader import FORMAT_VERSION

# Base URLs, tried in order. Every host holds the same files under the same names.
HOSTS = ["https://opensoftwarelab.s-on.pl/wordpacks"]

# A request that fails is sent to the same host this many more times before the next host is tried.
RETRIES = 2

# A full name is `<name>-<version>` with a whole-number version. Pack names and
# short tags never end in a hyphen followed only by digits, so that ending
# tells a full name from a short tag.
_FULL_NAME = re.compile(r"([a-z0-9-]+)-([0-9]+)")


def is_pack_request(text: str) -> bool:
    """Whether `text` reads as a pack's full name or short tag rather than as a file path."""
    return re.fullmatch(r"[a-z0-9-]+", text) is not None


def cache_dir() -> Path:
    """The directory pack files are kept in.

    Named after the format, not after ssdiff, so any wordpack reader can
    share one copy of a pack.
    """
    if os.environ.get("WORDPACK_CACHE"):
        return Path(os.environ["WORDPACK_CACHE"])
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "wordpack"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "wordpack"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "wordpack"


def _open(url: str):
    from ssdiff import __version__

    # Cloudflare answers 403 to urllib's default User-Agent.
    request = urllib.request.Request(url, headers={"User-Agent": f"ssdiff/{__version__}"})
    # urllib applies the timeout to each blocking operation, not to the whole download.
    return urllib.request.urlopen(request, timeout=60)


def _failed(request: str, host: str, error: Exception, attempt: int, failures: list[str]) -> None:
    """Say on stderr that `host` is asked again; after its last try, add the failure to `failures` instead."""
    if attempt <= RETRIES:
        print(f"wordpack {request}: {host} failed ({error}); trying again ({attempt} of {RETRIES})", file=sys.stderr)
    else:
        failures.append(f"  {host}: {error}")


def _check(name: str, data) -> None:
    """ValueError when `data` does not have the shape of `tags.json` or of `index.json`."""
    if name == "tags.json":
        ok = isinstance(data, dict) and all(isinstance(v, str) and _FULL_NAME.fullmatch(v) for v in data.values())
    else:
        ok = isinstance(data, dict) and isinstance(data.get("packs"), dict)
    if not ok:
        raise ValueError(f"{name} does not have the expected contents")


def _fetch_json(name: str, request: str) -> dict:
    """`name` from the first host whose answer is JSON of the right shape. OSError naming every host when none does."""
    failures = []
    for host in HOSTS:
        for attempt in range(1, RETRIES + 2):
            try:
                with _open(f"{host}/{name}") as response:
                    data = json.loads(response.read())
                _check(name, data)
                return data
            # A connection cut in the middle of an answer raises http.client.IncompleteRead, which is not an OSError.
            except (OSError, http.client.HTTPException, ValueError) as e:
                _failed(request, host, e, attempt, failures)
    raise OSError(
        f"No file named {request!r}, and it cannot be looked up as a wordpack: "
        f"no host answered for {name} in {RETRIES + 1} tries each:\n" + "\n".join(failures)
    )


def _tags(request: str) -> dict:
    """`tags.json` from a host, with a copy saved in the cache; the cached copy when no host answers."""
    cached = cache_dir() / "tags.json"
    try:
        tags = _fetch_json("tags.json", request)
    except OSError as e:
        # A cached copy that cannot be used is the same as none: the error names the hosts.
        try:
            tags = json.loads(cached.read_text("utf-8"))
            _check("tags.json", tags)
        except (OSError, ValueError):
            raise e from None
        return tags
    # The cached copy only serves a later load when no host answers, so a
    # cache that cannot be written does not fail this one.
    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        # A temporary name of its own: several processes may save their copy at once.
        fd, part = tempfile.mkstemp(dir=cached.parent, suffix=".part")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(tags, f)
            os.chmod(part, 0o644)  # as in _download
            os.replace(part, cached)
        finally:
            if os.path.exists(part):
                os.remove(part)
    except OSError:
        pass
    return tags


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


def resolve(request: str) -> str:
    """The path of the cached pack file for a full name or a short tag, downloading the file when it is missing.

    A full name that is in the cache makes no network request. A short tag is
    looked up on a host every time, so a moved tag is followed at the next load.
    """
    full = request
    if _FULL_NAME.fullmatch(request) is None:
        tags = _tags(request)
        if request not in tags:
            raise ValueError(f"No file or wordpack named {request!r}. Short tags: {', '.join(sorted(tags))}.")
        full = tags[request]
        print(f"wordpack {request}: {full}", file=sys.stderr)
    path = cache_dir() / f"{full}.wordpack"
    if path.exists():
        return str(path)
    index = _fetch_json("index.json", request)
    name, version = _FULL_NAME.fullmatch(full).groups()
    pack = index["packs"].get(name)
    if pack is None:
        try:
            known = f" Short tags: {', '.join(sorted(_tags(request)))}."
        except OSError:
            known = ""
        raise ValueError(f"No wordpack named {full!r}.{known}")
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
