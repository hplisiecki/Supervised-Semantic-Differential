"""Tests for ssdiff/lang_config.py — language config registry."""

from __future__ import annotations

import pytest

from ssdiff.lang_config import _ALIASES, LANG_TO_MODEL, LANGUAGES, get_config, lang_to_model


class TestLangToModel:
    def test_iso_code_pl(self):
        assert lang_to_model("pl") == "pl_core_news_lg"

    def test_iso_code_en(self):
        assert lang_to_model("en") == "en_core_web_lg"

    def test_iso_code_ru(self):
        assert lang_to_model("ru") == "ru_core_news_lg"

    def test_full_name_polish(self):
        assert lang_to_model("polish") == "pl_core_news_lg"

    def test_full_name_english(self):
        assert lang_to_model("english") == "en_core_web_lg"

    def test_case_insensitive_upper(self):
        assert lang_to_model("PL") == "pl_core_news_lg"

    def test_case_insensitive_mixed(self):
        assert lang_to_model("English") == "en_core_web_lg"

    def test_whitespace_stripped(self):
        assert lang_to_model("  pl  ") == "pl_core_news_lg"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown language"):
            lang_to_model("klingon")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            lang_to_model("")


class TestGetConfig:
    def test_returns_config(self):
        cfg = get_config("pl")
        assert cfg.spacy_model == "pl_core_news_lg"
        assert cfg.stopwords_file == "polish_stopwords.txt"
        assert cfg.bad_token_re is not None

    def test_bad_token_re_filters_digits(self):
        cfg = get_config("en")
        assert cfg.bad_token_re.match("abc123")

    def test_bad_token_re_filters_uppercase(self):
        cfg = get_config("en")
        assert cfg.bad_token_re.match("Hello")

    def test_bad_token_re_passes_lowercase(self):
        cfg = get_config("en")
        assert not cfg.bad_token_re.match("hello")

    def test_cyrillic_uppercase(self):
        cfg = get_config("ru")
        assert cfg.bad_token_re.match("Привет")  # starts with uppercase Cyrillic


class TestMappingCompleteness:
    def test_all_aliases_resolve(self):
        for name, iso in _ALIASES.items():
            assert iso in LANGUAGES, f"Alias '{name}' → '{iso}' not in LANGUAGES"

    def test_all_codes_have_model(self):
        for _code, cfg in LANGUAGES.items():
            assert isinstance(cfg.spacy_model, str)
            assert len(cfg.spacy_model) > 0

    def test_expected_languages_present(self):
        expected = {"pl", "en", "de", "fr", "es", "it", "ru"}
        assert expected.issubset(set(LANGUAGES.keys()))

    def test_lang_to_model_compat_dict(self):
        """LANG_TO_MODEL flat dict matches LANGUAGES."""
        for code, model in LANG_TO_MODEL.items():
            assert LANGUAGES[code].spacy_model == model
