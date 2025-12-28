"""Tests for i18n plurals module."""

import pytest
from truthound.datadocs.i18n.plurals import (
    PluralCategory,
    PluralRules,
    get_plural_category,
    pluralize,
    pluralize_with_forms,
)


class TestPluralCategory:
    """Tests for PluralCategory enum."""

    def test_categories(self):
        """Test all categories exist."""
        assert PluralCategory.ZERO.value == "zero"
        assert PluralCategory.ONE.value == "one"
        assert PluralCategory.TWO.value == "two"
        assert PluralCategory.FEW.value == "few"
        assert PluralCategory.MANY.value == "many"
        assert PluralCategory.OTHER.value == "other"


class TestPluralRules:
    """Tests for PluralRules class."""

    def test_english_plurals(self):
        """Test English plural rules."""
        rules = PluralRules()
        assert rules.get_category(1, "en") == PluralCategory.ONE
        assert rules.get_category(2, "en") == PluralCategory.OTHER
        assert rules.get_category(0, "en") == PluralCategory.OTHER
        assert rules.get_category(100, "en") == PluralCategory.OTHER

    def test_french_plurals(self):
        """Test French plural rules."""
        rules = PluralRules()
        assert rules.get_category(0, "fr") == PluralCategory.ONE
        assert rules.get_category(1, "fr") == PluralCategory.ONE
        assert rules.get_category(2, "fr") == PluralCategory.OTHER

    def test_russian_plurals(self):
        """Test Russian plural rules."""
        rules = PluralRules()
        assert rules.get_category(1, "ru") == PluralCategory.ONE
        assert rules.get_category(2, "ru") == PluralCategory.FEW
        assert rules.get_category(3, "ru") == PluralCategory.FEW
        assert rules.get_category(4, "ru") == PluralCategory.FEW
        assert rules.get_category(5, "ru") == PluralCategory.MANY
        assert rules.get_category(11, "ru") == PluralCategory.MANY
        assert rules.get_category(21, "ru") == PluralCategory.ONE
        assert rules.get_category(22, "ru") == PluralCategory.FEW
        assert rules.get_category(25, "ru") == PluralCategory.MANY

    def test_arabic_plurals(self):
        """Test Arabic plural rules."""
        rules = PluralRules()
        assert rules.get_category(0, "ar") == PluralCategory.ZERO
        assert rules.get_category(1, "ar") == PluralCategory.ONE
        assert rules.get_category(2, "ar") == PluralCategory.TWO
        assert rules.get_category(3, "ar") == PluralCategory.FEW
        assert rules.get_category(11, "ar") == PluralCategory.MANY

    def test_japanese_no_plurals(self):
        """Test Japanese (no plural distinction)."""
        rules = PluralRules()
        assert rules.get_category(1, "ja") == PluralCategory.OTHER
        assert rules.get_category(100, "ja") == PluralCategory.OTHER

    def test_korean_no_plurals(self):
        """Test Korean (no plural distinction)."""
        rules = PluralRules()
        assert rules.get_category(1, "ko") == PluralCategory.OTHER
        assert rules.get_category(100, "ko") == PluralCategory.OTHER

    def test_unsupported_language_fallback(self):
        """Test fallback for unsupported language."""
        rules = PluralRules()
        assert rules.get_category(1, "xyz") == PluralCategory.OTHER

    def test_supported_languages(self):
        """Test getting supported languages."""
        rules = PluralRules()
        languages = rules.get_supported_languages()
        assert "en" in languages
        assert "ru" in languages
        assert "ja" in languages


class TestPluralize:
    """Tests for pluralize function."""

    def test_english_singular(self):
        """Test English singular."""
        result = pluralize(1, "file", "files", "en")
        assert result == "1 file"

    def test_english_plural(self):
        """Test English plural."""
        result = pluralize(5, "file", "files", "en")
        assert result == "5 files"

    def test_zero_is_plural(self):
        """Test that zero uses plural form."""
        result = pluralize(0, "file", "files", "en")
        assert result == "0 files"


class TestPluralizeWithForms:
    """Tests for pluralize_with_forms function."""

    def test_english_forms(self):
        """Test English with two forms."""
        forms = {
            PluralCategory.ONE: "{count} file",
            PluralCategory.OTHER: "{count} files",
        }
        assert pluralize_with_forms(1, forms, "en") == "1 file"
        assert pluralize_with_forms(5, forms, "en") == "5 files"

    def test_russian_forms(self):
        """Test Russian with multiple forms."""
        forms = {
            PluralCategory.ONE: "{count} файл",
            PluralCategory.FEW: "{count} файла",
            PluralCategory.MANY: "{count} файлов",
            PluralCategory.OTHER: "{count} файла",
        }
        assert pluralize_with_forms(1, forms, "ru") == "1 файл"
        assert pluralize_with_forms(2, forms, "ru") == "2 файла"
        assert pluralize_with_forms(5, forms, "ru") == "5 файлов"
        assert pluralize_with_forms(21, forms, "ru") == "21 файл"

    def test_string_keys(self):
        """Test using string keys."""
        forms = {
            "one": "{count} item",
            "other": "{count} items",
        }
        assert pluralize_with_forms(1, forms, "en") == "1 item"
        assert pluralize_with_forms(5, forms, "en") == "5 items"

    def test_with_additional_params(self):
        """Test with additional format parameters."""
        forms = {
            PluralCategory.ONE: "{count} {name}",
            PluralCategory.OTHER: "{count} {name}s",
        }
        result = pluralize_with_forms(1, forms, "en", name="user")
        assert result == "1 user"

    def test_fallback_to_other(self):
        """Test fallback to OTHER category."""
        forms = {
            PluralCategory.OTHER: "{count} things",
        }
        # Even for 1, should use OTHER if ONE not provided
        result = pluralize_with_forms(1, forms, "en")
        assert result == "1 things"


class TestGetPluralCategory:
    """Tests for get_plural_category function."""

    def test_get_category(self):
        """Test getting plural category."""
        assert get_plural_category(1, "en") == PluralCategory.ONE
        assert get_plural_category(2, "en") == PluralCategory.OTHER

    def test_with_locale_region(self):
        """Test with locale including region."""
        assert get_plural_category(1, "en_US") == PluralCategory.ONE
        assert get_plural_category(1, "en-GB") == PluralCategory.ONE
