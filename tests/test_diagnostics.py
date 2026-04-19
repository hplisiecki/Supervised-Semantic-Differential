"""Tests for ssdiff.utils.diagnostics — progress hook + _progress/_diagnostic."""

from __future__ import annotations

import pytest

from ssdiff.utils import diagnostics as _diag
from ssdiff.utils.diagnostics import (
    _diagnostic,
    _hooked_iter,
    _progress,
    progress_hook,
)


class TestProgressHookContextManager:
    def test_sets_and_restores_callback(self):
        assert _diag._get_hook() is None
        calls = []
        with progress_hook(lambda c, t, d: calls.append((c, t, d))):
            assert _diag._get_hook() is not None
        assert _diag._get_hook() is None

    def test_restores_previous_callback_on_nested_use(self):
        def outer(c, t, d):
            return None
        with progress_hook(outer):
            assert _diag._get_hook() is outer
            with progress_hook(lambda c, t, d: None):
                assert _diag._get_hook() is not outer
            assert _diag._get_hook() is outer

    def test_restores_on_exception(self):
        with pytest.raises(RuntimeError, match="boom"), progress_hook(lambda c, t, d: None):
            raise RuntimeError("boom")
        assert _diag._get_hook() is None

    def test_thread_local_isolation(self):
        import threading
        observed = {}

        def worker():
            observed["child"] = _diag._get_hook()

        def cb(c, t, d):
            return None
        with progress_hook(cb):
            t = threading.Thread(target=worker)
            t.start()
            t.join()
        # Child thread's local is independent; should not see parent's hook.
        assert observed["child"] is None


class TestHookedIter:
    def test_fires_hook_each_iteration(self):
        calls = []
        out = list(_hooked_iter(
            iter(["a", "b", "c"]),
            total=3, desc="phase",
            hook=lambda c, t, d: calls.append((c, t, d)),
        ))
        assert out == ["a", "b", "c"]
        assert calls == [(1, 3, "phase"), (2, 3, "phase"), (3, 3, "phase")]

    def test_total_none_becomes_zero(self):
        calls = []
        list(_hooked_iter(
            iter([10, 20]),
            total=None, desc="",
            hook=lambda c, t, d: calls.append(t),
        ))
        assert calls == [0, 0]


class TestProgress:
    def test_no_hook_no_verbose_is_passthrough(self):
        out = list(_progress(iter([1, 2, 3]), verbose=False))
        assert out == [1, 2, 3]

    def test_hook_fires_when_active(self):
        calls = []
        with progress_hook(lambda c, t, d: calls.append((c, t, d))):
            out = list(_progress(iter([1, 2]), total=2, desc="X"))
        assert out == [1, 2]
        assert calls == [(1, 2, "X"), (2, 2, "X")]

    def test_verbose_without_tqdm_falls_back(self, monkeypatch):
        # Simulate tqdm missing.
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name.startswith("tqdm"):
                raise ImportError("mocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        out = list(_progress(iter([1, 2, 3]), verbose=True))
        assert out == [1, 2, 3]

    def test_hook_and_verbose_compose(self, monkeypatch):
        # With a hook active, the hook fires regardless of verbose.
        calls = []
        # Force tqdm import failure so behavior is deterministic on any env.
        import builtins
        real_import = builtins.__import__
        monkeypatch.setattr(
            builtins, "__import__",
            lambda name, *a, **kw: (_ for _ in ()).throw(ImportError())
                                   if name.startswith("tqdm")
                                   else real_import(name, *a, **kw),
        )
        with progress_hook(lambda c, t, d: calls.append(c)):
            list(_progress(iter([1, 2, 3]), verbose=True, total=3, desc="Y"))
        assert calls == [1, 2, 3]


class TestDiagnostic:
    def test_prints_when_verbose(self, capsys):
        _diagnostic(True, "hello")
        captured = capsys.readouterr()
        assert captured.out == "hello\n"

    def test_silent_when_not_verbose(self, capsys):
        _diagnostic(False, "hello")
        captured = capsys.readouterr()
        assert captured.out == ""
