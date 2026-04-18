"""Verifies the new results package skeleton exists and is importable."""

import importlib

MODULES = [
    "ssdiff.results",
    "ssdiff.results.core",
    "ssdiff.results.schema",
    "ssdiff.results.format",
    "ssdiff.results.report",
    "ssdiff.results.continuous_result",
    "ssdiff.results.group_result",
    "ssdiff.results.lexicon_result",
]


def test_all_modules_importable():
    for name in MODULES:
        importlib.import_module(name)
