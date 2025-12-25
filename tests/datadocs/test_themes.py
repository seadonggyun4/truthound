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
)
from truthound.datadocs.base import ReportTheme, ThemeConfig


class TestPredefinedThemes:
    """Test predefined theme configurations."""

    def test_light_theme_exists(self):
        """Test light theme is properly configured."""
        assert LIGHT_THEME is not None
        assert isinstance(LIGHT_THEME, ThemeConfig)
        assert LIGHT_THEME.name == "light"
        assert LIGHT_THEME.colors.background == "#ffffff"

    def test_dark_theme_exists(self):
        """Test dark theme is properly configured."""
        assert DARK_THEME is not None
        assert isinstance(DARK_THEME, ThemeConfig)
        assert DARK_THEME.name == "dark"
        # Dark theme should have dark background
        assert DARK_THEME.colors.background.startswith("#0") or DARK_THEME.colors.background.startswith("#1")

    def test_professional_theme_exists(self):
        """Test professional theme is properly configured."""
        assert PROFESSIONAL_THEME is not None
        assert isinstance(PROFESSIONAL_THEME, ThemeConfig)
        assert PROFESSIONAL_THEME.name == "professional"

    def test_minimal_theme_exists(self):
        """Test minimal theme is properly configured."""
        assert MINIMAL_THEME is not None
        assert isinstance(MINIMAL_THEME, ThemeConfig)
        assert MINIMAL_THEME.name == "minimal"

    def test_modern_theme_exists(self):
        """Test modern theme is properly configured."""
        assert MODERN_THEME is not None
        assert isinstance(MODERN_THEME, ThemeConfig)
        assert MODERN_THEME.name == "modern"

    def test_all_themes_have_chart_palette(self):
        """Test all themes have chart color palettes."""
        for theme in [LIGHT_THEME, DARK_THEME, PROFESSIONAL_THEME, MINIMAL_THEME, MODERN_THEME]:
            assert len(theme.colors.chart_palette) >= 8
            assert all(c.startswith("#") for c in theme.colors.chart_palette)


class TestThemesDict:
    """Test THEMES dictionary."""

    def test_themes_dict_populated(self):
        """Test THEMES dict has all themes."""
        assert len(THEMES) == 5
        assert ReportTheme.LIGHT in THEMES
        assert ReportTheme.DARK in THEMES
        assert ReportTheme.PROFESSIONAL in THEMES
        assert ReportTheme.MINIMAL in THEMES
        assert ReportTheme.MODERN in THEMES

    def test_themes_match_constants(self):
        """Test THEMES dict values match theme constants."""
        assert THEMES[ReportTheme.LIGHT] is LIGHT_THEME
        assert THEMES[ReportTheme.DARK] is DARK_THEME
        assert THEMES[ReportTheme.PROFESSIONAL] is PROFESSIONAL_THEME


class TestGetTheme:
    """Test get_theme function."""

    def test_get_theme_by_enum(self):
        """Test getting theme by enum."""
        theme = get_theme(ReportTheme.LIGHT)
        assert theme is LIGHT_THEME

    def test_get_theme_by_string(self):
        """Test getting theme by string."""
        theme = get_theme("dark")
        assert theme is DARK_THEME

    def test_get_theme_invalid_string(self):
        """Test invalid theme string raises error."""
        with pytest.raises(ValueError) as exc_info:
            get_theme("nonexistent")
        assert "Unknown theme" in str(exc_info.value)

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
        """Test all themes are returned."""
        themes = get_available_themes()
        assert len(themes) == 5
        assert "light" in themes
        assert "dark" in themes
        assert "professional" in themes
        assert "minimal" in themes
        assert "modern" in themes

    def test_returns_strings(self):
        """Test returned values are strings."""
        themes = get_available_themes()
        assert all(isinstance(t, str) for t in themes)


class TestThemeCSSGeneration:
    """Test CSS generation from themes."""

    def test_light_theme_css_vars(self):
        """Test light theme generates valid CSS."""
        css = LIGHT_THEME.to_css_vars()
        assert ":root {" in css
        assert "--color-background: #ffffff" in css
        assert "--color-primary:" in css

    def test_dark_theme_css_vars(self):
        """Test dark theme generates valid CSS."""
        css = DARK_THEME.to_css_vars()
        assert ":root {" in css
        # Dark theme should have different background
        assert "--color-background:" in css
        assert "#ffffff" not in css.split("--color-background:")[1].split(";")[0]

    def test_css_vars_contain_all_colors(self):
        """Test CSS vars contain all required colors."""
        css = PROFESSIONAL_THEME.to_css_vars()

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
        css = LIGHT_THEME.to_css_vars()

        # Should have chart color variables
        assert "--chart-color-0:" in css
        assert "--chart-color-1:" in css
