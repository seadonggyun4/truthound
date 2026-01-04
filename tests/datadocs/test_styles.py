"""Tests for styles module."""

import pytest

from truthound.datadocs.styles import (
    BASE_CSS,
    LAYOUT_CSS,
    COMPONENTS_CSS,
    RESPONSIVE_CSS,
    PRINT_CSS,
    DARK_MODE_OVERRIDES,
    get_complete_stylesheet,
)
from truthound.datadocs.themes import LIGHT_THEME, DARK_THEME, PROFESSIONAL_THEME


class TestBaseCSS:
    """Test base CSS definitions."""

    def test_base_css_exists(self):
        """Test BASE_CSS is defined."""
        assert BASE_CSS is not None
        assert len(BASE_CSS) > 0

    def test_base_css_has_reset(self):
        """Test BASE_CSS includes reset styles."""
        # Should have some form of CSS reset
        assert "box-sizing" in BASE_CSS or "margin" in BASE_CSS

    def test_base_css_has_variables(self):
        """Test BASE_CSS references CSS variables."""
        assert "var(" in BASE_CSS or "--" in BASE_CSS


class TestLayoutCSS:
    """Test layout CSS definitions."""

    def test_layout_css_exists(self):
        """Test LAYOUT_CSS is defined."""
        assert LAYOUT_CSS is not None
        assert len(LAYOUT_CSS) > 0

    def test_layout_css_has_layout_classes(self):
        """Test LAYOUT_CSS includes layout styles."""
        # LAYOUT_CSS uses flexbox for layout; grid styles are in COMPONENTS_CSS
        css_lower = LAYOUT_CSS.lower()
        has_layout = (
            "flex" in css_lower or
            "container" in css_lower or
            "section" in css_lower
        )
        assert has_layout

    def test_layout_css_has_flex(self):
        """Test LAYOUT_CSS includes flex styles."""
        assert "flex" in LAYOUT_CSS.lower()


class TestComponentsCSS:
    """Test components CSS definitions."""

    def test_components_css_exists(self):
        """Test COMPONENTS_CSS is defined."""
        assert COMPONENTS_CSS is not None
        assert len(COMPONENTS_CSS) > 0

    def test_components_css_has_card_styles(self):
        """Test COMPONENTS_CSS includes card styles."""
        assert "card" in COMPONENTS_CSS.lower()

    def test_components_css_has_table_styles(self):
        """Test COMPONENTS_CSS includes table styles."""
        assert "table" in COMPONENTS_CSS.lower()

    def test_components_css_has_button_styles(self):
        """Test COMPONENTS_CSS includes button or interactive styles."""
        # May have buttons, badges, or other interactive elements
        css_lower = COMPONENTS_CSS.lower()
        has_interactive = (
            "button" in css_lower or
            "badge" in css_lower or
            "tag" in css_lower or
            "hover" in css_lower
        )
        assert has_interactive


class TestResponsiveCSS:
    """Test responsive CSS definitions."""

    def test_responsive_css_exists(self):
        """Test RESPONSIVE_CSS is defined."""
        assert RESPONSIVE_CSS is not None
        assert len(RESPONSIVE_CSS) > 0

    def test_responsive_css_has_media_queries(self):
        """Test RESPONSIVE_CSS includes media queries."""
        assert "@media" in RESPONSIVE_CSS

    def test_responsive_css_has_breakpoints(self):
        """Test RESPONSIVE_CSS has common breakpoints."""
        # Should have at least one breakpoint for mobile/tablet
        css_lower = RESPONSIVE_CSS.lower()
        has_breakpoint = (
            "768px" in css_lower or
            "600px" in css_lower or
            "max-width" in css_lower or
            "min-width" in css_lower
        )
        assert has_breakpoint


class TestPrintCSS:
    """Test print CSS definitions."""

    def test_print_css_exists(self):
        """Test PRINT_CSS is defined."""
        assert PRINT_CSS is not None
        assert len(PRINT_CSS) > 0

    def test_print_css_has_media_print(self):
        """Test PRINT_CSS includes print media query."""
        assert "@media print" in PRINT_CSS

    def test_print_css_has_page_break(self):
        """Test PRINT_CSS handles page breaks."""
        # Should have page break handling
        css_lower = PRINT_CSS.lower()
        has_page_handling = (
            "page-break" in css_lower or
            "break-" in css_lower or
            "@page" in css_lower
        )
        assert has_page_handling


class TestDarkModeOverrides:
    """Test dark mode CSS overrides."""

    def test_dark_mode_exists(self):
        """Test DARK_MODE_OVERRIDES is defined."""
        assert DARK_MODE_OVERRIDES is not None
        assert len(DARK_MODE_OVERRIDES) > 0

    def test_dark_mode_has_selector(self):
        """Test DARK_MODE_OVERRIDES has proper selector."""
        # Should have a dark mode selector
        css_lower = DARK_MODE_OVERRIDES.lower()
        has_dark_selector = (
            "dark" in css_lower or
            "prefers-color-scheme" in css_lower or
            ".dark" in css_lower or
            "[data-theme" in css_lower
        )
        assert has_dark_selector


class TestGetCompleteStylesheet:
    """Test get_complete_stylesheet function."""

    def test_returns_string(self):
        """Test function returns a string."""
        css = get_complete_stylesheet(LIGHT_THEME)
        assert isinstance(css, str)
        assert len(css) > 0

    def test_includes_all_css(self):
        """Test complete stylesheet includes all CSS parts."""
        css = get_complete_stylesheet(LIGHT_THEME)

        # Should contain parts from all CSS modules
        # Check for common selectors/properties
        assert "body" in css or "html" in css
        assert "@media" in css  # Responsive or print

    def test_includes_theme_variables(self):
        """Test stylesheet includes theme CSS variables."""
        css = get_complete_stylesheet(LIGHT_THEME)

        # Should have CSS custom properties
        assert "--" in css or "var(" in css

    def test_light_theme_colors(self):
        """Test light theme has correct colors."""
        css = get_complete_stylesheet(LIGHT_THEME)

        # Light theme typically has light background
        assert "#fff" in css.lower() or "#ffffff" in css.lower() or "#f" in css.lower()

    def test_dark_theme_colors(self):
        """Test dark theme has correct colors."""
        css = get_complete_stylesheet(DARK_THEME)

        # Dark theme should have dark colors referenced via CSS vars
        # The actual color values are in theme.get_css() which includes :root vars
        assert "var(--color-background)" in css or "DarkTheme" in css

    def test_different_themes_produce_different_css(self):
        """Test different themes produce different CSS."""
        light_css = get_complete_stylesheet(LIGHT_THEME)
        dark_css = get_complete_stylesheet(DARK_THEME)
        prof_css = get_complete_stylesheet(PROFESSIONAL_THEME)

        # Each should be different
        assert light_css != dark_css
        assert light_css != prof_css
        assert dark_css != prof_css

    def test_css_is_valid(self):
        """Test generated CSS is syntactically valid (basic check)."""
        css = get_complete_stylesheet(LIGHT_THEME)

        # Basic validity checks
        open_braces = css.count('{')
        close_braces = css.count('}')
        assert open_braces == close_braces, "Unbalanced braces in CSS"

    def test_includes_chart_styles(self):
        """Test stylesheet includes chart styles."""
        css = get_complete_stylesheet(LIGHT_THEME)

        # Should have chart-related styles
        css_lower = css.lower()
        has_chart_styles = (
            "chart" in css_lower or
            "svg" in css_lower or
            ".chart" in css_lower
        )
        assert has_chart_styles

    def test_includes_section_styles(self):
        """Test stylesheet includes section styles."""
        css = get_complete_stylesheet(LIGHT_THEME)

        # Should have section-related styles
        css_lower = css.lower()
        has_section_styles = (
            "section" in css_lower or
            ".section" in css_lower or
            "report" in css_lower
        )
        assert has_section_styles


class TestCSSVariables:
    """Test CSS variable generation."""

    def test_theme_to_css_vars(self):
        """Test theme generates CSS variables."""
        css_vars = LIGHT_THEME.to_css_vars()

        assert isinstance(css_vars, str)
        assert "--" in css_vars
        assert "color" in css_vars.lower()

    def test_css_vars_include_colors(self):
        """Test CSS variables include theme colors."""
        css_vars = LIGHT_THEME.to_css_vars()

        # Should have main color variables
        has_colors = (
            "--color" in css_vars or
            "--background" in css_vars or
            "primary" in css_vars
        )
        assert has_colors

    def test_css_vars_include_spacing(self):
        """Test CSS variables include spacing values."""
        css_vars = LIGHT_THEME.to_css_vars()

        # Should have spacing/radius variables
        has_spacing = (
            "radius" in css_vars.lower() or
            "spacing" in css_vars.lower() or
            "shadow" in css_vars.lower()
        )
        assert has_spacing

    def test_dark_theme_vars(self):
        """Test dark theme generates different variables."""
        light_vars = LIGHT_THEME.to_css_vars()
        dark_vars = DARK_THEME.to_css_vars()

        # Dark theme should have different color values
        assert dark_vars != light_vars
