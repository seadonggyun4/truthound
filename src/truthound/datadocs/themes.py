"""Pre-defined themes for Data Docs reports.

This module provides carefully designed themes that are clean, modern,
and suitable for enterprise use.
"""

from truthound.datadocs.base import (
    ReportTheme,
    ThemeColors,
    ThemeTypography,
    ThemeSpacing,
    ThemeConfig,
)


# =============================================================================
# Light Theme - Clean and Professional
# =============================================================================

LIGHT_THEME = ThemeConfig(
    name="light",
    colors=ThemeColors(
        # Main colors
        background="#ffffff",
        surface="#f8fafc",
        text_primary="#0f172a",
        text_secondary="#64748b",

        # Brand colors
        primary="#3b82f6",
        secondary="#8b5cf6",
        accent="#ec4899",

        # Semantic colors
        success="#22c55e",
        warning="#f59e0b",
        error="#ef4444",
        info="#0ea5e9",

        # Border and shadow
        border="#e2e8f0",
        shadow="rgba(15, 23, 42, 0.08)",

        # Chart colors (Blue-based palette)
        chart_palette=(
            "#3b82f6", "#8b5cf6", "#ec4899", "#06b6d4",
            "#22c55e", "#f59e0b", "#ef4444", "#6366f1",
            "#14b8a6", "#f97316"
        ),
    ),
    typography=ThemeTypography(
        font_family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        font_family_mono="'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
    ),
    spacing=ThemeSpacing(
        border_radius_sm="6px",
        border_radius_md="8px",
        border_radius_lg="12px",
        border_radius_xl="16px",
        shadow_md="0 4px 6px -1px rgba(15, 23, 42, 0.1), 0 2px 4px -1px rgba(15, 23, 42, 0.06)",
    ),
)


# =============================================================================
# Dark Theme - Modern and Elegant
# =============================================================================

DARK_THEME = ThemeConfig(
    name="dark",
    colors=ThemeColors(
        # Main colors
        background="#0f172a",
        surface="#1e293b",
        text_primary="#f1f5f9",
        text_secondary="#94a3b8",

        # Brand colors
        primary="#60a5fa",
        secondary="#a78bfa",
        accent="#f472b6",

        # Semantic colors
        success="#4ade80",
        warning="#fbbf24",
        error="#f87171",
        info="#38bdf8",

        # Border and shadow
        border="#334155",
        shadow="rgba(0, 0, 0, 0.25)",

        # Chart colors (Vibrant for dark background)
        chart_palette=(
            "#60a5fa", "#a78bfa", "#f472b6", "#22d3ee",
            "#4ade80", "#fbbf24", "#f87171", "#818cf8",
            "#2dd4bf", "#fb923c"
        ),
    ),
    typography=ThemeTypography(
        font_family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        font_family_mono="'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
    ),
    spacing=ThemeSpacing(
        shadow_md="0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)",
        shadow_lg="0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.2)",
    ),
)


# =============================================================================
# Professional Theme - Corporate and Subdued
# =============================================================================

PROFESSIONAL_THEME = ThemeConfig(
    name="professional",
    colors=ThemeColors(
        # Main colors
        background="#fafbfc",
        surface="#ffffff",
        text_primary="#1f2937",
        text_secondary="#6b7280",

        # Brand colors (Corporate blue)
        primary="#2563eb",
        secondary="#4f46e5",
        accent="#7c3aed",

        # Semantic colors
        success="#059669",
        warning="#d97706",
        error="#dc2626",
        info="#0284c7",

        # Border and shadow
        border="#d1d5db",
        shadow="rgba(0, 0, 0, 0.05)",

        # Chart colors (Professional palette)
        chart_palette=(
            "#2563eb", "#4f46e5", "#7c3aed", "#0891b2",
            "#059669", "#d97706", "#dc2626", "#4338ca",
            "#0d9488", "#ea580c"
        ),
    ),
    typography=ThemeTypography(
        font_family="'Segoe UI', 'SF Pro Display', -apple-system, BlinkMacSystemFont, Roboto, sans-serif",
        font_family_mono="'Cascadia Code', 'SF Mono', 'Consolas', monospace",
    ),
    spacing=ThemeSpacing(
        border_radius_sm="4px",
        border_radius_md="6px",
        border_radius_lg="8px",
        border_radius_xl="12px",
    ),
)


# =============================================================================
# Minimal Theme - Clean and Unobtrusive
# =============================================================================

MINIMAL_THEME = ThemeConfig(
    name="minimal",
    colors=ThemeColors(
        # Main colors
        background="#ffffff",
        surface="#fafafa",
        text_primary="#171717",
        text_secondary="#737373",

        # Brand colors (Monochromatic)
        primary="#404040",
        secondary="#525252",
        accent="#262626",

        # Semantic colors (Muted)
        success="#16a34a",
        warning="#ca8a04",
        error="#dc2626",
        info="#2563eb",

        # Border and shadow
        border="#e5e5e5",
        shadow="rgba(0, 0, 0, 0.03)",

        # Chart colors (Grayscale with accent)
        chart_palette=(
            "#404040", "#525252", "#737373", "#a3a3a3",
            "#16a34a", "#ca8a04", "#dc2626", "#2563eb",
            "#0d9488", "#ea580c"
        ),
    ),
    typography=ThemeTypography(
        font_family="'Helvetica Neue', Arial, sans-serif",
        font_family_mono="'Monaco', 'Menlo', monospace",
    ),
    spacing=ThemeSpacing(
        border_radius_sm="2px",
        border_radius_md="4px",
        border_radius_lg="6px",
        border_radius_xl="8px",
        shadow_sm="0 1px 2px rgba(0, 0, 0, 0.03)",
        shadow_md="0 2px 4px rgba(0, 0, 0, 0.05)",
    ),
)


# =============================================================================
# Modern Theme - Vibrant and Contemporary
# =============================================================================

MODERN_THEME = ThemeConfig(
    name="modern",
    colors=ThemeColors(
        # Main colors
        background="#fefefe",
        surface="#f5f5f7",
        text_primary="#1d1d1f",
        text_secondary="#86868b",

        # Brand colors (Gradient-inspired)
        primary="#5e5ce6",
        secondary="#bf5af2",
        accent="#ff375f",

        # Semantic colors
        success="#32d74b",
        warning="#ff9f0a",
        error="#ff453a",
        info="#0a84ff",

        # Border and shadow
        border="#d2d2d7",
        shadow="rgba(0, 0, 0, 0.08)",

        # Chart colors (Apple-inspired)
        chart_palette=(
            "#5e5ce6", "#bf5af2", "#ff375f", "#30d158",
            "#ff9f0a", "#64d2ff", "#ff453a", "#ffd60a",
            "#ac8e68", "#0a84ff"
        ),
    ),
    typography=ThemeTypography(
        font_family="'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif",
        font_family_mono="'SF Mono', 'Monaco', monospace",
    ),
    spacing=ThemeSpacing(
        border_radius_sm="8px",
        border_radius_md="12px",
        border_radius_lg="16px",
        border_radius_xl="22px",
        shadow_md="0 4px 12px rgba(0, 0, 0, 0.08)",
        shadow_lg="0 8px 24px rgba(0, 0, 0, 0.12)",
    ),
)


# =============================================================================
# Theme Registry
# =============================================================================

THEMES: dict[ReportTheme, ThemeConfig] = {
    ReportTheme.LIGHT: LIGHT_THEME,
    ReportTheme.DARK: DARK_THEME,
    ReportTheme.PROFESSIONAL: PROFESSIONAL_THEME,
    ReportTheme.MINIMAL: MINIMAL_THEME,
    ReportTheme.MODERN: MODERN_THEME,
}


def get_theme(theme: ReportTheme | str) -> ThemeConfig:
    """Get a theme configuration by name or enum.

    Args:
        theme: Theme name or enum value

    Returns:
        ThemeConfig for the requested theme

    Raises:
        ValueError: If theme is not found
    """
    if isinstance(theme, str):
        try:
            theme = ReportTheme(theme)
        except ValueError:
            available = [t.value for t in ReportTheme]
            raise ValueError(
                f"Unknown theme '{theme}'. Available: {available}"
            )

    if theme not in THEMES:
        raise ValueError(f"Theme '{theme.value}' not configured")

    return THEMES[theme]


def get_available_themes() -> list[str]:
    """Get list of available theme names."""
    return [t.value for t in THEMES.keys()]
