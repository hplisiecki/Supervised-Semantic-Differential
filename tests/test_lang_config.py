"""Tests for ssdiff.lang_config — LangConfig regex behavior and resolver."""

from __future__ import annotations

import pytest

from ssdiff.lang_config import (
    LANG_TO_MODEL,
    LANGUAGES,
    LangConfig,
    _resolve_lang,
    get_config,
    lang_to_model,
)


class TestBadTokenRegexLatin:
    """Latin-script config rejects digits and uppercase starts, accepts lowercase."""

    def _re(self):
        return LANGUAGES["pl"].bad_token_re

    @pytest.mark.parametrize("token", [
        "kraj", "narod", "łąka", "ążśź", "café",
    ])
    def test_accepts_lowercase_tokens(self, token):
        assert self._re().match(token) is None, token

    @pytest.mark.parametrize("token", [
        "Warszawa", "ABC", "Łódź", "Zürich", "Ödön",
    ])
    def test_rejects_uppercase_first_letter(self, token):
        assert self._re().match(token) is not None, token

    @pytest.mark.parametrize("token", [
        "abc1", "v2", "covid19", "data2024",
    ])
    def test_rejects_digit_containing_tokens(self, token):
        assert self._re().match(token) is not None, token


class TestBadTokenRegexCyrillic:
    def _re(self):
        return LANGUAGES["ru"].bad_token_re

    @pytest.mark.parametrize("token", ["привет", "народ", "ёжик"])
    def test_accepts_cyrillic_lowercase(self, token):
        assert self._re().match(token) is None, token

    @pytest.mark.parametrize("token", ["Москва", "Ёж", "Привет"])
    def test_rejects_cyrillic_uppercase(self, token):
        assert self._re().match(token) is not None, token

    def test_does_not_reject_latin_uppercase_on_cyrillic_re(self):
        # A Latin uppercase "A" should NOT match the Cyrillic bad-token pattern.
        # (Separate scripts — language-specific regexes are intentional.)
        assert LANGUAGES["ru"].bad_token_re.match("Apple") is None


class TestResolveLang:
    def test_iso_code_roundtrip(self):
        assert _resolve_lang("pl") == "pl"
        assert _resolve_lang("PL") == "pl"
        assert _resolve_lang(" pl ") == "pl"

    def test_full_name_maps_to_iso(self):
        assert _resolve_lang("polish") == "pl"
        assert _resolve_lang("English") == "en"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown language"):
            _resolve_lang("klingon")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="Language is required"):
            _resolve_lang(None)


class TestPublicHelpers:
    def test_get_config_returns_langconfig(self):
        cfg = get_config("pl")
        assert isinstance(cfg, LangConfig)
        assert cfg.spacy_model == "pl_core_news_lg"
        assert cfg.stopwords_file == "polish_stopwords.txt"

    def test_lang_to_model_matches_registry(self):
        for code, cfg in LANGUAGES.items():
            assert lang_to_model(code) == cfg.spacy_model

    def test_lang_to_model_flat_dict_consistent(self):
        for code, cfg in LANGUAGES.items():
            assert LANG_TO_MODEL[code] == cfg.spacy_model
