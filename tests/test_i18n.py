"""Tests for the bilingual translation module."""

from __future__ import annotations

from race_planners.i18n import (
    AVAILABLE_LOCALES,
    DEFAULT_LOCALE,
    TRANSLATIONS,
    t,
)


def test_every_key_has_both_en_and_fr() -> None:
    en_keys = set(TRANSLATIONS["en"].keys())
    fr_keys = set(TRANSLATIONS["fr"].keys())

    missing_in_fr = en_keys - fr_keys
    missing_in_en = fr_keys - en_keys

    assert not missing_in_fr, f"Keys missing French translation: {sorted(missing_in_fr)}"
    assert not missing_in_en, f"Keys missing English translation: {sorted(missing_in_en)}"


def test_t_returns_english_for_default_locale() -> None:
    assert t("app.title") == "Unified Event Planner"


def test_t_returns_french_for_fr_locale() -> None:
    assert t("app.title", "fr") == "Planificateur d'Événements Unifié"


def test_t_falls_back_to_english_for_unknown_locale() -> None:
    assert t("app.title", "de") == "Unified Event Planner"


def test_t_returns_key_itself_for_unknown_key() -> None:
    assert t("nonexistent.key.somewhere") == "nonexistent.key.somewhere"


def test_t_handles_named_placeholders() -> None:
    result = t("caption.derived_hr_cap", "en", hr=160)
    assert "160 bpm" in result

    result_fr = t("caption.derived_hr_cap", "fr", hr=160)
    assert "160 bpm" in result_fr


def test_t_handles_format_specifiers() -> None:
    result = t("assumption.weather_heat", "en", temp=30)
    assert "30°C" in result


def test_t_does_not_raise_on_missing_placeholder() -> None:
    result = t("caption.derived_hr_cap", "en")
    assert "{hr}" in result


def test_available_locales_contains_en_and_fr() -> None:
    assert "en" in AVAILABLE_LOCALES
    assert "fr" in AVAILABLE_LOCALES


def test_default_locale_is_en() -> None:
    assert DEFAULT_LOCALE == "en"
