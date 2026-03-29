"""Per-language configuration — single source of truth for the package.

Stores spaCy model defaults, bundled stopword file names, and token-filter
patterns for each supported language.  Every language-dependent decision in
ssdlite should read from this module.

# Future: allow users to load custom config (e.g. from a YAML/JSON file)
# to override defaults without touching library code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Language config dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LangConfig:
    """Configuration for a single language.

    Parameters
    ----------
    spacy_model : str
        Default spaCy model name (e.g. ``"pl_core_news_lg"``).
    stopwords_file : str or None
        Bundled stopword file name under ``ssdlite/utils/``, or ``None``
        to fall back to spaCy built-in stopwords.
    bad_token_re : re.Pattern
        Compiled regex — tokens matching this are filtered out in neighbor
        search.  The default pattern rejects tokens containing digits or
        starting with an uppercase letter (ASCII + common diacritics for
        the language).
    """
    spacy_model: str
    stopwords_file: str | None = None
    bad_token_re: re.Pattern = field(
        default_factory=lambda: re.compile(r".*\d|^[A-Z]")
    )


# ---------------------------------------------------------------------------
# Per-language registry
# ---------------------------------------------------------------------------

# Uppercase letter classes used in bad_token patterns:
#   _LATIN_UPPER  — covers Western/Central European languages
#   _CYRILLIC_UPPER — covers Slavic Cyrillic scripts
#   Each language picks the pattern that fits its script.

_LATIN_UPPER = r"A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ"
_LATIN_UPPER += r"ĄĆĘŁŃÓŚŹŻĈĜĤĴŜŬŠŽĐ"  # Polish, Croatian, etc.
_CYRILLIC_UPPER = r"А-ЯЁЂЄІЇҐЉЊЋЏЎЍ"

_RE_LATIN = re.compile(rf".*\d|^[{_LATIN_UPPER}]")
_RE_CYRILLIC = re.compile(rf".*\d|^[{_CYRILLIC_UPPER}]")
_RE_CJK_DIGIT = re.compile(r".*\d")  # CJK: no uppercase concept, filter digits only


LANGUAGES: dict[str, LangConfig] = {
    "ca": LangConfig("ca_core_news_lg", bad_token_re=_RE_LATIN),
    "da": LangConfig("da_core_news_lg", bad_token_re=_RE_LATIN),
    "de": LangConfig("de_core_news_lg", bad_token_re=_RE_LATIN),
    "el": LangConfig("el_core_news_lg"),  # Greek script — default pattern OK
    "en": LangConfig("en_core_web_lg", bad_token_re=_RE_LATIN),
    "es": LangConfig("es_core_news_lg", bad_token_re=_RE_LATIN),
    "fr": LangConfig("fr_core_news_lg", bad_token_re=_RE_LATIN),
    "hr": LangConfig("hr_core_news_lg", bad_token_re=_RE_LATIN),
    "it": LangConfig("it_core_news_lg", bad_token_re=_RE_LATIN),
    "ja": LangConfig("ja_core_news_lg", bad_token_re=_RE_CJK_DIGIT),
    "ko": LangConfig("ko_core_news_lg", bad_token_re=_RE_CJK_DIGIT),
    "lt": LangConfig("lt_core_news_lg", bad_token_re=_RE_LATIN),
    "mk": LangConfig("mk_core_news_lg", bad_token_re=_RE_CYRILLIC),
    "nb": LangConfig("nb_core_news_lg", bad_token_re=_RE_LATIN),
    "nl": LangConfig("nl_core_news_lg", bad_token_re=_RE_LATIN),
    "pl": LangConfig("pl_core_news_lg", stopwords_file="polish_stopwords.txt",
                      bad_token_re=_RE_LATIN),
    "pt": LangConfig("pt_core_news_lg", bad_token_re=_RE_LATIN),
    "ro": LangConfig("ro_core_news_lg", bad_token_re=_RE_LATIN),
    "ru": LangConfig("ru_core_news_lg", bad_token_re=_RE_CYRILLIC),
    "sl": LangConfig("sl_core_news_lg", bad_token_re=_RE_LATIN),
    "sv": LangConfig("sv_core_news_lg", bad_token_re=_RE_LATIN),
    "uk": LangConfig("uk_core_news_lg", bad_token_re=_RE_CYRILLIC),
    "zh": LangConfig("zh_core_web_lg", bad_token_re=_RE_CJK_DIGIT),
}

# Full-name aliases → ISO code
_ALIASES: dict[str, str] = {
    "catalan": "ca",
    "danish": "da",
    "german": "de",
    "greek": "el",
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "croatian": "hr",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "lithuanian": "lt",
    "macedonian": "mk",
    "norwegian": "nb",
    "dutch": "nl",
    "polish": "pl",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "slovenian": "sl",
    "swedish": "sv",
    "ukrainian": "uk",
    "chinese": "zh",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _resolve_lang(lang: str) -> str:
    """Resolve a language string (ISO code or full name) to an ISO code."""
    key = lang.strip().lower()
    if key in LANGUAGES:
        return key
    iso = _ALIASES.get(key)
    if iso is not None:
        return iso
    raise ValueError(
        f"Unknown language '{lang}'. "
        f"Supported codes: {sorted(LANGUAGES.keys())}"
    )


def get_config(lang: str) -> LangConfig:
    """Return the ``LangConfig`` for a language code or name."""
    return LANGUAGES[_resolve_lang(lang)]


def lang_to_model(lang: str) -> str:
    """Map a language code or name to the default spaCy model name.

    Convenience wrapper — equivalent to ``get_config(lang).spacy_model``.
    """
    return get_config(lang).spacy_model


# Backward-compat: flat dict used by tests and external code
LANG_TO_MODEL: dict[str, str] = {
    code: cfg.spacy_model for code, cfg in LANGUAGES.items()
}
