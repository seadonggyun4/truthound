"""Translation Management System integration tests.

Tests TMS functionality including:
- Project management
- String management
- Translation operations
- Export/import operations
- Webhook handling

These tests can run against:
- Mock TMS (default)
- Crowdin API (with credentials)
- Lokalise API (with credentials)
- Other TMS providers
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.integration.external.backends.tms_backend import TMSBackend
    from tests.integration.external.providers.mock_provider import MockTMSService


# =============================================================================
# Mock TMS Tests (Fast, No External Dependencies)
# =============================================================================


class TestMockTMS:
    """Tests using mock TMS for fast execution."""

    def test_project_creation(self, mock_tms: "MockTMSService") -> None:
        """Test project creation."""
        project = mock_tms.create_project("test-project", "Test Project")

        assert project["id"] == "test-project"
        assert project["name"] == "Test Project"
        assert "created" in project

    def test_get_project(self, mock_tms: "MockTMSService") -> None:
        """Test getting project details."""
        mock_tms.create_project("my-project", "My Project")

        project = mock_tms.get_project("my-project")
        assert project is not None
        assert project["id"] == "my-project"

        # Non-existent project
        assert mock_tms.get_project("nonexistent") is None

    def test_list_projects(self, mock_tms: "MockTMSService") -> None:
        """Test listing projects."""
        mock_tms.create_project("project-1")
        mock_tms.create_project("project-2")

        projects = mock_tms.list_projects()
        assert len(projects) == 2

        project_ids = [p["id"] for p in projects]
        assert "project-1" in project_ids
        assert "project-2" in project_ids

    def test_delete_project(self, mock_tms: "MockTMSService") -> None:
        """Test project deletion."""
        mock_tms.create_project("to-delete")
        assert mock_tms.get_project("to-delete") is not None

        result = mock_tms.delete_project("to-delete")
        assert result is True
        assert mock_tms.get_project("to-delete") is None

        # Deleting non-existent project
        assert mock_tms.delete_project("nonexistent") is False

    def test_add_string(self, mock_tms: "MockTMSService") -> None:
        """Test adding source strings."""
        mock_tms.create_project("test-project")

        string_data = mock_tms.add_string(
            "test-project",
            "hello",
            "Hello, World!",
            context="Greeting message",
        )

        assert string_data["key"] == "hello"
        assert string_data["text"] == "Hello, World!"
        assert string_data["context"] == "Greeting message"

    def test_get_string(self, mock_tms: "MockTMSService") -> None:
        """Test getting a string."""
        mock_tms.create_project("test-project")
        mock_tms.add_string("test-project", "hello", "Hello")

        string_data = mock_tms.get_string("test-project", "hello")
        assert string_data is not None
        assert string_data["key"] == "hello"

        # Non-existent string
        assert mock_tms.get_string("test-project", "nonexistent") is None

    def test_list_strings(self, mock_tms: "MockTMSService") -> None:
        """Test listing strings in a project."""
        mock_tms.create_project("test-project")
        mock_tms.add_string("test-project", "key1", "Text 1")
        mock_tms.add_string("test-project", "key2", "Text 2")
        mock_tms.add_string("test-project", "key3", "Text 3")

        strings = mock_tms.list_strings("test-project")
        assert len(strings) == 3

        keys = [s["key"] for s in strings]
        assert "key1" in keys
        assert "key2" in keys
        assert "key3" in keys

    def test_delete_string(self, mock_tms: "MockTMSService") -> None:
        """Test deleting a string."""
        mock_tms.create_project("test-project")
        mock_tms.add_string("test-project", "to-delete", "Delete me")

        assert mock_tms.delete_string("test-project", "to-delete") is True
        assert mock_tms.get_string("test-project", "to-delete") is None

    def test_add_translation(self, mock_tms: "MockTMSService") -> None:
        """Test adding translations."""
        mock_tms.create_project("test-project")
        mock_tms.add_string("test-project", "hello", "Hello")

        translation = mock_tms.add_translation(
            "test-project",
            "hello",
            "es",
            "Hola",
        )

        assert translation["key"] == "hello"
        assert translation["locale"] == "es"
        assert translation["translation"] == "Hola"

    def test_get_translation(self, mock_tms: "MockTMSService") -> None:
        """Test getting a translation."""
        mock_tms.create_project("test-project")
        mock_tms.add_string("test-project", "hello", "Hello")
        mock_tms.add_translation("test-project", "hello", "es", "Hola")
        mock_tms.add_translation("test-project", "hello", "fr", "Bonjour")

        assert mock_tms.get_translation("test-project", "hello", "es") == "Hola"
        assert mock_tms.get_translation("test-project", "hello", "fr") == "Bonjour"
        assert mock_tms.get_translation("test-project", "hello", "de") is None

    def test_export_translations(self, mock_tms: "MockTMSService") -> None:
        """Test exporting translations."""
        mock_tms.create_project("test-project")
        mock_tms.add_string("test-project", "hello", "Hello")
        mock_tms.add_string("test-project", "goodbye", "Goodbye")
        mock_tms.add_translation("test-project", "hello", "es", "Hola")
        mock_tms.add_translation("test-project", "goodbye", "es", "Adiós")
        mock_tms.add_translation("test-project", "hello", "fr", "Bonjour")

        # Export Spanish
        es_translations = mock_tms.export_translations("test-project", "es")
        assert es_translations == {"hello": "Hola", "goodbye": "Adiós"}

        # Export French
        fr_translations = mock_tms.export_translations("test-project", "fr")
        assert fr_translations == {"hello": "Bonjour"}

    def test_list_translations(self, mock_tms: "MockTMSService") -> None:
        """Test listing all translations."""
        mock_tms.create_project("test-project")
        mock_tms.add_string("test-project", "hello", "Hello")
        mock_tms.add_translation("test-project", "hello", "es", "Hola")
        mock_tms.add_translation("test-project", "hello", "fr", "Bonjour")

        # List all
        all_translations = mock_tms.list_translations("test-project")
        assert "hello" in all_translations
        assert all_translations["hello"] == {"es": "Hola", "fr": "Bonjour"}

        # Filter by locale
        es_only = mock_tms.list_translations("test-project", locale="es")
        assert es_only == {"hello": {"es": "Hola"}}


# =============================================================================
# TMS Backend Tests
# =============================================================================


@pytest.mark.tms
@pytest.mark.integration
class TestTMSBackend:
    """Tests using TMS backend."""

    def test_backend_connection(self, tms_backend: "TMSBackend") -> None:
        """Test TMS backend connection."""
        assert tms_backend.is_running
        assert tms_backend.is_healthy

    def test_health_check(self, tms_backend: "TMSBackend") -> None:
        """Test health check."""
        result = tms_backend.health_check()
        assert result.healthy

    def test_project_lifecycle(
        self,
        tms_backend: "TMSBackend",
        test_project_id: str,
    ) -> None:
        """Test complete project lifecycle."""
        # Create
        project = tms_backend.create_project(test_project_id, "Test Project")
        assert project is not None

        # Get
        retrieved = tms_backend.get_project(test_project_id)
        assert retrieved is not None

        # List
        projects = tms_backend.list_projects()
        assert any(p.get("id") == test_project_id for p in projects)

        # Delete
        assert tms_backend.delete_project(test_project_id) is True

    def test_string_operations(
        self,
        tms_backend: "TMSBackend",
        test_project_id: str,
    ) -> None:
        """Test string operations."""
        tms_backend.create_project(test_project_id)

        try:
            # Add strings
            tms_backend.add_string(test_project_id, "key1", "Text 1")
            tms_backend.add_string(test_project_id, "key2", "Text 2")

            # List strings
            strings = tms_backend.list_strings(test_project_id)
            assert len(strings) >= 2

            # Get string
            string = tms_backend.get_string(test_project_id, "key1")
            assert string is not None
            assert string["text"] == "Text 1"

        finally:
            tms_backend.delete_project(test_project_id)

    def test_translation_operations(
        self,
        tms_backend: "TMSBackend",
        test_project_id: str,
    ) -> None:
        """Test translation operations."""
        tms_backend.create_project(test_project_id)

        try:
            # Add source string
            tms_backend.add_string(test_project_id, "greeting", "Hello")

            # Add translations
            tms_backend.add_translation(test_project_id, "greeting", "es", "Hola")
            tms_backend.add_translation(test_project_id, "greeting", "fr", "Bonjour")
            tms_backend.add_translation(test_project_id, "greeting", "de", "Hallo")

            # Get translations
            assert tms_backend.get_translation(test_project_id, "greeting", "es") == "Hola"
            assert tms_backend.get_translation(test_project_id, "greeting", "fr") == "Bonjour"
            assert tms_backend.get_translation(test_project_id, "greeting", "de") == "Hallo"

            # Export
            es_export = tms_backend.export_translations(test_project_id, "es")
            assert es_export.get("greeting") == "Hola"

        finally:
            tms_backend.delete_project(test_project_id)

    def test_bulk_operations(
        self,
        tms_backend: "TMSBackend",
        test_project_id: str,
        sample_translation_data: list,
    ) -> None:
        """Test bulk string/translation operations."""
        tms_backend.create_project(test_project_id)

        try:
            # Bulk add strings
            results = tms_backend.bulk_add_strings(test_project_id, sample_translation_data)
            assert len(results) == len(sample_translation_data)

            # Verify
            strings = tms_backend.list_strings(test_project_id)
            assert len(strings) >= len(sample_translation_data)

            # Bulk add translations
            translations = [
                {"key": "hello", "locale": "es", "translation": "Hola"},
                {"key": "goodbye", "locale": "es", "translation": "Adiós"},
                {"key": "welcome", "locale": "es", "translation": "Bienvenido"},
            ]
            results = tms_backend.bulk_add_translations(test_project_id, translations)
            assert len(results) == 3

        finally:
            tms_backend.delete_project(test_project_id)


# =============================================================================
# i18n Integration Tests
# =============================================================================


@pytest.mark.tms
@pytest.mark.integration
class TestI18nIntegration:
    """Tests for internationalization workflow integration."""

    def test_complete_localization_workflow(
        self,
        tms_backend: "TMSBackend",
        test_project_id: str,
    ) -> None:
        """Test complete localization workflow."""
        tms_backend.create_project(test_project_id, "Localization Test")

        try:
            # 1. Add source strings (English)
            strings = [
                {"key": "app.title", "text": "My Application", "context": "App title"},
                {"key": "app.welcome", "text": "Welcome, {name}!", "context": "Welcome message with placeholder"},
                {"key": "app.items_count", "text": "{count} item(s)", "context": "Item count with plural"},
                {"key": "button.submit", "text": "Submit", "context": "Submit button"},
                {"key": "button.cancel", "text": "Cancel", "context": "Cancel button"},
            ]
            tms_backend.bulk_add_strings(test_project_id, strings)

            # 2. Add translations for multiple locales
            locales = {
                "es": {
                    "app.title": "Mi Aplicación",
                    "app.welcome": "¡Bienvenido, {name}!",
                    "app.items_count": "{count} elemento(s)",
                    "button.submit": "Enviar",
                    "button.cancel": "Cancelar",
                },
                "fr": {
                    "app.title": "Mon Application",
                    "app.welcome": "Bienvenue, {name}!",
                    "app.items_count": "{count} élément(s)",
                    "button.submit": "Soumettre",
                    "button.cancel": "Annuler",
                },
                "ja": {
                    "app.title": "マイアプリケーション",
                    "app.welcome": "ようこそ、{name}さん！",
                    "app.items_count": "{count}件",
                    "button.submit": "送信",
                    "button.cancel": "キャンセル",
                },
            }

            for locale, translations in locales.items():
                for key, translation in translations.items():
                    tms_backend.add_translation(test_project_id, key, locale, translation)

            # 3. Export and verify
            for locale, expected in locales.items():
                exported = tms_backend.export_translations(test_project_id, locale)
                for key, value in expected.items():
                    assert exported.get(key) == value, f"Mismatch for {locale}/{key}"

        finally:
            tms_backend.delete_project(test_project_id)

    def test_placeholder_preservation(
        self,
        tms_backend: "TMSBackend",
        test_project_id: str,
    ) -> None:
        """Test that placeholders are preserved in translations."""
        tms_backend.create_project(test_project_id)

        try:
            # Add string with placeholders
            tms_backend.add_string(
                test_project_id,
                "greeting",
                "Hello {firstName} {lastName}!",
            )

            # Add translation with placeholders
            tms_backend.add_translation(
                test_project_id,
                "greeting",
                "es",
                "¡Hola {firstName} {lastName}!",
            )

            # Verify placeholders are preserved
            translation = tms_backend.get_translation(test_project_id, "greeting", "es")
            assert "{firstName}" in translation
            assert "{lastName}" in translation

        finally:
            tms_backend.delete_project(test_project_id)

    def test_rtl_language_support(
        self,
        tms_backend: "TMSBackend",
        test_project_id: str,
    ) -> None:
        """Test right-to-left language support."""
        tms_backend.create_project(test_project_id)

        try:
            tms_backend.add_string(test_project_id, "hello", "Hello")

            # Arabic (RTL)
            tms_backend.add_translation(
                test_project_id, "hello", "ar", "مرحبا"
            )

            # Hebrew (RTL)
            tms_backend.add_translation(
                test_project_id, "hello", "he", "שלום"
            )

            # Verify
            assert tms_backend.get_translation(test_project_id, "hello", "ar") == "مرحبا"
            assert tms_backend.get_translation(test_project_id, "hello", "he") == "שלום"

        finally:
            tms_backend.delete_project(test_project_id)

    def test_unicode_handling(
        self,
        tms_backend: "TMSBackend",
        test_project_id: str,
    ) -> None:
        """Test Unicode character handling."""
        tms_backend.create_project(test_project_id)

        try:
            # Source with emoji
            tms_backend.add_string(
                test_project_id,
                "status.success",
                "Success! ✓",
            )

            # Translations with various Unicode
            tms_backend.add_translation(
                test_project_id, "status.success", "zh", "成功！✓"
            )
            tms_backend.add_translation(
                test_project_id, "status.success", "ko", "성공! ✓"
            )
            tms_backend.add_translation(
                test_project_id, "status.success", "th", "สำเร็จ! ✓"
            )

            # Verify
            assert "成功" in tms_backend.get_translation(test_project_id, "status.success", "zh")
            assert "성공" in tms_backend.get_translation(test_project_id, "status.success", "ko")
            assert "สำเร็จ" in tms_backend.get_translation(test_project_id, "status.success", "th")

        finally:
            tms_backend.delete_project(test_project_id)
