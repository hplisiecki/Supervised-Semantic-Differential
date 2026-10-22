"""Loading a wordpack by tag or full name: hosts, cache, checks and errors.

A host is a temporary directory holding two copies of the fixture pack, listed
as en-fixture-1 and en-fixture-2. It is read through a file:// URL, or served
on 127.0.0.1 where a test needs the request headers.
"""

from __future__ import annotations

import functools
import hashlib
import http.server
import json
import sys
import threading
from pathlib import Path

import pytest

import ssdiff
from ssdiff import SSD, Embeddings
from ssdiff.lang_config import WORDPACK_TAGS
from ssdiff.wordpack import fetch


def make_host(directory: Path, pack: str, **changes) -> str:
    """Write a host into `directory` and return its base URL. `changes` overrides fields of every index entry."""
    directory.mkdir()
    data = Path(pack).read_bytes()
    versions = {}
    for version in ("1", "2"):
        file = f"en-fixture-{version}.wordpack"
        (directory / file).write_bytes(data)
        versions[version] = {
            "file": file,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "format_version": 1,
            **changes,
        }
    pack_entry = {"language": "en", "description": "Synthetic test pack.", "license": "CC0-1.0", "versions": versions}
    index = {"index_version": 1, "packs": {"en-fixture": pack_entry}}
    (directory / "index.json").write_text(json.dumps(index), "utf-8")
    return directory.as_uri()


@pytest.fixture
def cache(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("WORDPACK_CACHE", str(tmp_path / "cache"))
    return tmp_path / "cache"


@pytest.fixture
def host(tmp_path, monkeypatch, wordpack_path, cache) -> Path:
    """A reachable host, with the tags `default` and `en` standing for en-fixture-1."""
    monkeypatch.setattr(fetch, "HOSTS", [make_host(tmp_path / "host", wordpack_path)])
    monkeypatch.setitem(WORDPACK_TAGS, "default", ("en-fixture-1.wordpack", "the default is en-fixture-1"))
    monkeypatch.setitem(WORDPACK_TAGS, "en", ("en-fixture-1.wordpack", "en is en-fixture-1"))
    return tmp_path / "host"


def unreachable(tmp_path) -> str:
    return (tmp_path / "no-such-host").as_uri()


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture
def serve(monkeypatch):
    """Serve a directory on 127.0.0.1 through a handler class and make that server the only host."""
    servers = []

    def start(directory: Path, handler_class) -> None:
        handler = functools.partial(handler_class, directory=str(directory))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        monkeypatch.setenv("no_proxy", "127.0.0.1")
        monkeypatch.setattr(fetch, "HOSTS", [f"http://127.0.0.1:{server.server_address[1]}"])

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def dropping(directory: Path, file: str, drops: int, cut: str) -> tuple[type, list]:
    """A handler class that cuts the connection on the first `drops` requests for `file`, and those requests."""
    seen = []

    class Handler(QuietHandler):
        def do_GET(self):
            if self.path.rsplit("/", 1)[-1] != file:
                return super().do_GET()
            seen.append(self.path)
            if len(seen) > drops:
                return super().do_GET()
            if cut == "in the middle of the answer":
                data = (directory / file).read_bytes()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data[: len(data) // 2])

    return Handler, seen


def test_a_cached_full_name_resolves_without_a_host(tmp_path, monkeypatch, wordpack_path, cache):
    monkeypatch.setattr(fetch, "HOSTS", [unreachable(tmp_path)])
    cache.mkdir()
    cached = cache / "en-fixture-1.wordpack"
    cached.write_bytes(Path(wordpack_path).read_bytes())
    assert fetch.resolve("en-fixture-1") == str(cached)


def test_a_full_name_that_is_not_cached_is_downloaded(host, cache, wordpack_path):
    path = fetch.resolve("en-fixture-2")
    assert path == str(cache / "en-fixture-2.wordpack")
    assert Path(path).read_bytes() == Path(wordpack_path).read_bytes()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no permission bits for group and others")
def test_the_cached_files_are_readable_by_other_users(host, cache):
    fetch.resolve("en-fixture-1")
    assert all(p.stat().st_mode & 0o044 == 0o044 for p in cache.iterdir())


def test_no_host_and_an_empty_cache_names_every_host(tmp_path, monkeypatch, cache):
    hosts = [(tmp_path / "gone-a").as_uri(), (tmp_path / "gone-b").as_uri()]
    monkeypatch.setattr(fetch, "HOSTS", hosts)
    with pytest.raises(OSError, match=r"'en-fixture-1' is not in the cache .* cannot be downloaded") as e:
        fetch.resolve("en-fixture-1")
    assert all(h in str(e.value) for h in hosts)


def test_a_format_version_this_reader_does_not_read_asks_for_an_upgrade(tmp_path, monkeypatch, wordpack_path, cache):
    monkeypatch.setattr(fetch, "HOSTS", [make_host(tmp_path / "host", wordpack_path, format_version=2)])
    with pytest.raises(ValueError, match="Upgrade ssdiff"):
        fetch.resolve("en-fixture-1")
    assert not cache.exists() or not list(cache.iterdir())


@pytest.mark.parametrize("changes", [{"sha256": "0" * 64}, {"size": 1}])
def test_a_download_that_fails_its_check_leaves_nothing_in_the_cache(
    tmp_path, monkeypatch, wordpack_path, cache, changes
):
    monkeypatch.setattr(fetch, "HOSTS", [make_host(tmp_path / "host", wordpack_path, **changes)])
    with pytest.raises(OSError, match="SHA-256"):
        fetch.resolve("en-fixture-1")
    assert list(cache.iterdir()) == []


@pytest.mark.parametrize("answer", ["<html>Sign in to this network</html>", "5", '{"status": "blocked"}'])
def test_a_host_that_is_missing_or_answers_with_something_else_is_skipped(
    tmp_path, monkeypatch, wordpack_path, cache, answer
):
    # The first host answers index.json with something else, as a captive portal does, and has no other file.
    portal = tmp_path / "portal"
    portal.mkdir()
    (portal / "index.json").write_text(answer, "utf-8")
    good = make_host(tmp_path / "host", wordpack_path)
    monkeypatch.setattr(fetch, "HOSTS", [unreachable(tmp_path), portal.as_uri(), good])
    assert fetch.resolve("en-fixture-1") == str(cache / "en-fixture-1.wordpack")


@pytest.mark.parametrize(
    "full, lists",
    [
        ("en-fixture-7", r"'en-fixture-7'.*Versions of en-fixture: 1, 2\."),
        ("pl-nothing-1", r"'pl-nothing-1'.*Tags: .*\ben\b"),
    ],
)
def test_an_unknown_full_name_lists_what_is_available(host, full, lists):
    with pytest.raises(ValueError, match=lists):
        fetch.resolve(full)


def test_every_request_names_ssdiff_as_the_user_agent(tmp_path, wordpack_path, cache, serve):
    seen = []

    class Handler(QuietHandler):
        def do_GET(self):
            seen.append(self.headers["User-Agent"])
            super().do_GET()

    make_host(tmp_path / "host", wordpack_path)
    serve(tmp_path / "host", Handler)
    fetch.resolve("en-fixture-1")
    # index.json and the pack file
    assert seen == [f"ssdiff/{ssdiff.__version__}"] * 2


@pytest.mark.parametrize("cut", ["before the answer", "in the middle of the answer"])
@pytest.mark.parametrize("file", ["index.json", "en-fixture-1.wordpack"])
def test_a_request_that_fails_twice_is_sent_a_third_time(tmp_path, wordpack_path, cache, serve, capsys, file, cut):
    make_host(tmp_path / "host", wordpack_path)
    handler, seen = dropping(tmp_path / "host", file, 2, cut)
    serve(tmp_path / "host", handler)
    assert fetch.resolve("en-fixture-1") == str(cache / "en-fixture-1.wordpack")
    assert len(seen) == 3
    assert capsys.readouterr().err.count("trying again") == 2


def test_a_host_that_keeps_failing_is_asked_three_times(tmp_path, wordpack_path, cache, serve):
    make_host(tmp_path / "host", wordpack_path)
    handler, seen = dropping(tmp_path / "host", "index.json", 10, "before the answer")
    serve(tmp_path / "host", handler)
    with pytest.raises(OSError, match="3 tries"):
        fetch.resolve("en-fixture-1")
    assert len(seen) == 3


@pytest.mark.parametrize(
    "platform, env, expected",
    [
        ("linux", {"WORDPACK_CACHE": "/somewhere/else"}, "/somewhere/else"),
        ("linux", {"XDG_CACHE_HOME": "{home}/xdg"}, "{home}/.wordpacks"),
        ("linux", {}, "{home}/.wordpacks"),
        ("darwin", {}, "{home}/.wordpacks"),
        ("win32", {"LOCALAPPDATA": "{home}/AppData/Local"}, "{home}/.wordpacks"),
    ],
)
def test_the_cache_directory_is_in_home_on_every_platform(tmp_path, monkeypatch, platform, env, expected):
    home = str(tmp_path)
    for name in ("WORDPACK_CACHE", "XDG_CACHE_HOME", "LOCALAPPDATA"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)
    for name, value in env.items():
        monkeypatch.setenv(name, value.format(home=home))
    monkeypatch.setattr(sys, "platform", platform)
    assert fetch.cache_dir() == Path(expected.format(home=home))


def test_load_takes_a_tag_and_a_full_name(host, cache):
    by_tag = Embeddings.load("en")
    by_name = Embeddings.load("en-fixture-1")
    assert (by_tag.name, by_tag.version) == (by_name.name, by_name.version) == ("en-fixture", "1")
    assert by_name._source_path == str(cache / "en-fixture-1.wordpack")


def test_load_with_no_argument_loads_the_tag_default(host, cache, capsys):
    emb = Embeddings.load()
    assert emb._source_path == str(cache / "en-fixture-1.wordpack")
    assert "the default is en-fixture-1" in capsys.readouterr().err


def test_a_tag_prints_its_message_on_every_load(host, capsys):
    Embeddings.load("en")
    Embeddings.load("en")
    assert capsys.readouterr().err.count("en is en-fixture-1") == 2


def test_a_tag_whose_pack_is_cached_loads_without_a_host(host, tmp_path, monkeypatch):
    first = Embeddings.load("en")
    monkeypatch.setattr(fetch, "HOSTS", [unreachable(tmp_path)])
    assert Embeddings.load("en")._source_path == first._source_path


def test_the_name_a_text_records_loads_the_same_pack(host):
    emb = Embeddings.load("en")
    recorded = SSD(emb, ["the dog swims"], use_full_doc=True).tokenizer.removeprefix("wordpack:")
    again = Embeddings.load(recorded)
    assert (again.name, again.version) == (emb.name, emb.version)


@pytest.mark.parametrize("name", ["xx", "EN", "english", "missing.kv", Path("en")])
def test_a_name_that_is_no_tag_and_no_file_lists_the_tags_without_a_request(host, tmp_path, monkeypatch, name):
    def no_request(url):
        raise AssertionError(f"requested {url}")

    monkeypatch.setattr(fetch, "_open", no_request)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match=rf"No wordpack tag or file named '{name}'\. Tags: .*\ben\b"):
        Embeddings.load(name)


def test_a_tag_wins_over_a_file_of_the_same_name(host, cache, tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    (work / "en").write_bytes(b"not an embedding")
    monkeypatch.chdir(work)
    assert Embeddings.load("en").name == "en-fixture"
    # A string with a dot is read as a path, so the file is opened and fails to parse.
    with pytest.raises(Exception) as e:
        Embeddings.load("./en")
    assert not isinstance(e.value, FileNotFoundError)


def test_a_directory_named_like_a_tag_does_not_block_the_lookup(host, tmp_path, monkeypatch):
    work = tmp_path / "work"
    (work / "en").mkdir(parents=True)
    monkeypatch.chdir(work)
    assert Embeddings.load("en").name == "en-fixture"


def test_every_shipped_tag_names_a_pack_file():
    for tag, (file, message) in WORDPACK_TAGS.items():
        assert "." not in tag
        assert file.endswith(".wordpack") and fetch.is_full_name(file.removesuffix(".wordpack"))
        assert message
