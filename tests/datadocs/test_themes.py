"""Tests for datadocs themes module."""

import pytest

from truthound.datadocs.themes import (
    LIGHT_THEME,
    DARK_THEME,
    PROFESSIONAL_THEME,
    MINIMAL_THEME,
    MODERN_THEME,
    THEMES,
    get_theme,
    get_available_themes,
    BaseTheme,
)
from truthound.datadocs.base import ReportTheme, ThemeConfig


class TestPredefinedThemes:
    """Test predefined theme configurations."""

    def test_light_theme_exists(self):
        """Test light theme is properly configured."""
        assert LIGHT_THEME is not None
        assert isinstance(LIGHT_THEME, BaseTheme)
        assert LIGHT_THEME.name == "light"
        assert LIGHT_THEME.config.colors.background == "#ffffff"

    def test_dark_theme_exists(self):
        """Test dark theme is properly configured."""
        assert DARK_THEME is not None
        assert isinstance(DARK_THEME, BaseTheme)
        assert DARK_THEME.name == "dark"
        # Dark theme should have dark background
        bg = DARK_THEME.config.colors.background
        assert bg.startswith("#0") or bg.startswith("#1")

    def test_professional_theme_exists(self):
        """Test professional theme is properly configured."""
        assert PROFESSIONAL_THEME is not None
        assert isinstance(PROFESSIONAL_THEME, BaseTheme)
        assert PROFESSIONAL_THEME.name == "professional"

    def test_minimal_theme_exists(self):
        """Test minimal theme is properly configured."""
        assert MINIMAL_THEME is not None
        assert isinstance(MINIMAL_THEME, BaseTheme)
        assert MINIMAL_THEME.name == "minimal"

    def test_modern_theme_exists(self):
        """Test modern theme is properly configured."""
        assert MODERN_THEME is not None
        assert isinstance(MODERN_THEME, BaseTheme)
        assert MODERN_THEME.name == "modern"

    def test_all_themes_have_chart_palette(self):
        """Test all themes have chart color palettes."""
        for theme in [LIGHT_THEME, DARK_THEME, PROFESSIONAL_THEME, MINIMAL_THEME, MODERN_THEME]:
            assert len(theme.config.colors.chart_palette) >= 8
            assert all(c.startswith("#") for c in theme.config.colors.chart_palette)


class TestThemesDict:
    """Test THEMES dictionary."""

    def test_themes_dict_populated(self):
        """Test THEMES dict has all themes (6: default, light, dark, minimal, modern, professional)."""
        assert len(THEMES) == 6
        assert "light" in THEMES
        assert "dark" in THEMES
        assert "professional" in THEMES
        assert "minimal" in THEMES
        assert "modern" in THEMES
        assert "default" in THEMES

    def test_themes_values_are_classes(self):
        """Test THEMES dict values are theme classes."""
        for name, theme_cls in THEMES.items():
            assert isinstance(theme_cls, type) or callable(theme_cls)


class TestGetTheme:
    """Test get_theme function."""

    def test_get_theme_by_enum(self):
        """Test getting theme by enum."""
        theme = get_theme(ReportTheme.LIGHT)
        assert theme is not None
        assert theme.name == "light"

    def test_get_theme_by_string(self):
        """Test getting theme by string."""
        theme = get_theme("dark")
        assert theme is not None
        assert theme.name == "dark"

    def test_get_theme_invalid_string(self):
        """Test invalid theme string raises error."""
        with pytest.raises(KeyError) as exc_info:
            get_theme("nonexistent")
        assert "not found" in str(exc_info.value)

    def test_get_all_themes_by_string(self):
        """Test all themes can be retrieved by string."""
        for theme_name in ["light", "dark", "professional", "minimal", "modern"]:
            theme = get_theme(theme_name)
            assert theme is not None
            assert theme.name == theme_name


class TestGetAvailableThemes:
    """Test get_available_themes function."""

    def test_returns_list(self):
        """Test function returns a list."""
        themes = get_available_themes()
        assert isinstance(themes, list)

    def test_returns_all_themes(self):
        """Test all themes are returned (6 themes including default)."""
        themes = get_available_themes()
        assert len(themes) == 6
        assert "light" in themes
        assert "dark" in themes
        assert "professional" in themes
        assert "minimal" in themes
        assert "modern" in themes
        assert "default" in themes

    def test_returns_strings(self):
        """Test returned values are strings."""
        themes = get_available_themes()
        assert all(isinstance(t, str) for t in themes)


class TestThemeCSSGeneration:
    """Test CSS generation from themes."""

    def test_light_theme_css_vars(self):
        """Test light theme generates valid CSS."""
        css = LIGHT_THEME.get_css()
        assert ":root {" in css
        assert "--color-background:" in css
        assert "--color-primary:" in css

    def test_dark_theme_css_vars(self):
        """Test dark theme generates valid CSS."""
        css = DARK_THEME.get_css()
        assert ":root {" in css
        # Dark theme should have different background
        assert "--color-background:" in css

    def test_css_vars_contain_all_colors(self):
        """Test CSS vars contain all required colors."""
        css = PROFESSIONAL_THEME.get_css()

        required_vars = [
            "--color-background",
            "--color-surface",
            "--color-text-primary",
            "--color-primary",
            "--color-success",
            "--color-warning",
            "--color-error",
            "--color-border",
        ]

        for var in required_vars:
            assert var in css, f"Missing CSS variable: {var}"

    def test_css_vars_contain_chart_colors(self):
        """Test CSS vars contain chart colors."""
        css = LIGHT_THEME.get_css()

        # Should have chart color variables
        assert "--chart-color-0:" in css
        assert "--chart-color-1:" in css
