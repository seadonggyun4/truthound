"""Default theme implementations for Data Docs.

This module provides the built-in themes that are available out of the box.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from truthound.datadocs.themes.base import (
    BaseTheme,
    ThemeConfig,
    ThemeColors,
    ThemeTypography,
    ThemeSpacing,
    ThemeAssets,
)


class DefaultTheme(BaseTheme):
    """The default Truthound theme.

    A clean, professional theme suitable for most use cases.
    """

    def __init__(
        self,
        config: ThemeConfig | None = None,
        **overrides: Any,
    ) -> None:
        """Initialize the default theme.

        Args:
            config: Optional custom configuration.
            **overrides: Direct attribute overrides.
        """
        if config is None:
            config = ThemeConfig(
                name="default",
                display_name="Default",
                description="Clean, professional default theme",
                colors=ThemeColors(
                    background="#fafbfc",
                    surface="#ffffff",
                    text_primary="#1f2937",
                    text_secondary="#6b7280",
                    primary="#2563eb",
                    secondary="#4f46e5",
                    accent="#7c3aed",
                    success="#059669",
                    warning="#d97706",
                    error="#dc2626",
                    info="#0284c7",
                    border="#d1d5db",
                    shadow="rgba(0, 0, 0, 0.05)",
                    chart_palette=(
                        "#2563eb", "#4f46e5", "#7c3aed", "#0891b2",
                        "#059669", "#d97706", "#dc2626", "#4338ca",
                        "#0d9488", "#ea580c"
                    ),
                ),
            )

        # Apply overrides
        if overrides:
            config = self._apply_overrides(config, overrides)

        super().__init__(config)

    def _apply_overrides(
        self,
        config: ThemeConfig,
        overrides: dict[str, Any],
    ) -> ThemeConfig:
        """Apply overrides to the configuration.

        Args:
            config: Base configuration.
            overrides: Override values.

        Returns:
            Updated configuration.
        """
        # Handle nested overrides
        colors_overrides = {}
        typography_overrides = {}
        spacing_overrides = {}
        assets_overrides = {}
        config_overrides = {}

        for key, value in overrides.items():
            if key.startswith("color_") or key in ("primary", "secondary", "accent", "background", "surface"):
                colors_overrides[key.replace("color_", "")] = value
            elif key.startswith("font_") or key.startswith("line_height"):
                typography_overrides[key] = value
            elif key.startswith("border_radius") or key.startswith("spacing") or key.startswith("shadow"):
                spacing_overrides[key] = value
            elif key in ("logo_url", "logo_base64", "favicon_url"):
                assets_overrides[key] = value
            else:
                config_overrides[key] = value

        # Apply nested overrides
        if colors_overrides:
            new_colors = replace(config.colors, **colors_overrides)
            config = replace(config, colors=new_colors)
        if typography_overrides:
            new_typography = replace(config.typography, **typography_overrides)
            config = replace(config, typography=new_typography)
        if spacing_overrides:
            new_spacing = replace(config.spacing, **spacing_overrides)
            config = replace(config, spacing=new_spacing)
        if assets_overrides:
            new_assets = replace(config.assets, **assets_overrides)
            config = replace(config, assets=new_assets)
        if config_overrides:
            config = replace(config, **config_overrides)

        return config

    def customize(self, **overrides: Any) -> "DefaultTheme":
        """Create a customized version of this theme.

        Args:
            **overrides: Values to override.

        Returns:
            New theme instance.
        """
        new_config = self._apply_overrides(self._config, overrides)
        return DefaultTheme(config=new_config)


class LightTheme(BaseTheme):
    """Light theme with bright colors.

    A clean, light theme with subtle shadows.
    """

    def __init__(self, **overrides: Any) -> None:
        config = ThemeConfig(
            name="light",
            display_name="Light",
            description="Clean light theme",
            colors=ThemeColors(
                background="#ffffff",
                surface="#f8fafc",
                text_primary="#0f172a",
                text_secondary="#64748b",
                primary="#3b82f6",
                secondary="#8b5cf6",
                accent="#ec4899",
                success="#22c55e",
                warning="#f59e0b",
                error="#ef4444",
                info="#0ea5e9",
                border="#e2e8f0",
                shadow="rgba(15, 23, 42, 0.08)",
                chart_palette=(
                    "#3b82f6", "#8b5cf6", "#ec4899", "#06b6d4",
                    "#22c55e", "#f59e0b", "#ef4444", "#6366f1",
                    "#14b8a6", "#f97316"
                ),
            ),
            spacing=ThemeSpacing(
                shadow_md="0 4px 6px -1px rgba(15, 23, 42, 0.1)",
            ),
        )
        super().__init__(config)

    def customize(self, **overrides: Any) -> "LightTheme":
        # Simplified customization
        return self


class DarkTheme(BaseTheme):
    """Dark theme for low-light environments.

    A modern dark theme with vibrant accent colors.
    """

    def __init__(self, **overrides: Any) -> None:
        config = ThemeConfig(
            name="dark",
            display_name="Dark",
            description="Modern dark theme",
            colors=ThemeColors(
                background="#0f172a",
                surface="#1e293b",
                text_primary="#f1f5f9",
                text_secondary="#94a3b8",
                primary="#60a5fa",
                secondary="#a78bfa",
                accent="#f472b6",
                success="#4ade80",
                warning="#fbbf24",
                error="#f87171",
                info="#38bdf8",
                border="#334155",
                shadow="rgba(0, 0, 0, 0.25)",
                chart_palette=(
                    "#60a5fa", "#a78bfa", "#f472b6", "#22d3ee",
                    "#4ade80", "#fbbf24", "#f87171", "#818cf8",
                    "#2dd4bf", "#fb923c"
                ),
            ),
            spacing=ThemeSpacing(
                shadow_md="0 4px 6px -1px rgba(0, 0, 0, 0.3)",
                shadow_lg="0 10px 15px -3px rgba(0, 0, 0, 0.3)",
            ),
        )
        super().__init__(config)

    def _get_base_css(self) -> str:
        """Add dark mode specific CSS."""
        return """
/* Dark mode adjustments */
body {
    color-scheme: dark;
}
::selection {
    background: var(--color-primary);
    color: var(--color-text-primary);
}
"""

    def customize(self, **overrides: Any) -> "DarkTheme":
        return self


class MinimalTheme(BaseTheme):
    """Minimal theme with reduced visual elements.

    A clean, distraction-free theme focused on content.
    """

    def __init__(self, **overrides: Any) -> None:
        config = ThemeConfig(
            name="minimal",
            display_name="Minimal",
            description="Clean minimal theme",
            colors=ThemeColors(
                background="#ffffff",
                surface="#fafafa",
                text_primary="#171717",
                text_secondary="#737373",
                primary="#404040",
                secondary="#525252",
                accent="#262626",
                success="#16a34a",
                warning="#ca8a04",
                error="#dc2626",
                info="#2563eb",
                border="#e5e5e5",
                shadow="rgba(0, 0, 0, 0.03)",
                chart_palette=(
                    "#404040", "#525252", "#737373", "#a3a3a3",
                    "#16a34a", "#ca8a04", "#dc2626", "#2563eb",
                    "#0d9488", "#ea580c"
                ),
            ),
            typography=ThemeTypography(
                font_family="'Helvetica Neue', Arial, sans-serif",
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
        super().__init__(config)

    def customize(self, **overrides: Any) -> "MinimalTheme":
        return self


class ModernTheme(BaseTheme):
    """Modern theme with vibrant colors and smooth curves.

    An Apple-inspired design with gradient-like color palette.
    """

    def __init__(self, **overrides: Any) -> None:
        config = ThemeConfig(
            name="modern",
            display_name="Modern",
            description="Vibrant modern theme",
            colors=ThemeColors(
                background="#fefefe",
                surface="#f5f5f7",
                text_primary="#1d1d1f",
                text_secondary="#86868b",
                primary="#5e5ce6",
                secondary="#bf5af2",
                accent="#ff375f",
                success="#32d74b",
                warning="#ff9f0a",
                error="#ff453a",
                info="#0a84ff",
                border="#d2d2d7",
                shadow="rgba(0, 0, 0, 0.08)",
                chart_palette=(
                    "#5e5ce6", "#bf5af2", "#ff375f", "#30d158",
                    "#ff9f0a", "#64d2ff", "#ff453a", "#ffd60a",
                    "#ac8e68", "#0a84ff"
                ),
            ),
            typography=ThemeTypography(
                font_family="'SF Pro Display', -apple-system, sans-serif",
                font_family_mono="'SF Mono', Monaco, monospace",
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
        super().__init__(config)

    def customize(self, **overrides: Any) -> "ModernTheme":
        return self


class ProfessionalTheme(BaseTheme):
    """Professional theme for corporate reports.

    A subdued, business-appropriate theme.
    """

    def __init__(self, **overrides: Any) -> None:
        config = ThemeConfig(
            name="professional",
            display_name="Professional",
            description="Corporate professional theme",
            colors=ThemeColors(
                background="#fafbfc",
                surface="#ffffff",
                text_primary="#1f2937",
                text_secondary="#6b7280",
                primary="#2563eb",
                secondary="#4f46e5",
                accent="#7c3aed",
                success="#059669",
                warning="#d97706",
                error="#dc2626",
                info="#0284c7",
                border="#d1d5db",
                shadow="rgba(0, 0, 0, 0.05)",
                chart_palette=(
                    "#2563eb", "#4f46e5", "#7c3aed", "#0891b2",
                    "#059669", "#d97706", "#dc2626", "#4338ca",
                    "#0d9488", "#ea580c"
                ),
            ),
            typography=ThemeTypography(
                font_family="'Segoe UI', 'SF Pro Display', system-ui, sans-serif",
                font_family_mono="'Cascadia Code', 'SF Mono', monospace",
            ),
            spacing=ThemeSpacing(
                border_radius_sm="4px",
                border_radius_md="6px",
                border_radius_lg="8px",
                border_radius_xl="12px",
            ),
        )
        super().__init__(config)

    def customize(self, **overrides: Any) -> "ProfessionalTheme":
        return self


# Theme registry for easy access
BUILT_IN_THEMES = {
    "default": DefaultTheme,
    "light": LightTheme,
    "dark": DarkTheme,
    "minimal": MinimalTheme,
    "modern": ModernTheme,
    "professional": ProfessionalTheme,
}


def get_theme(name: str, **overrides: Any) -> BaseTheme:
    """Get a theme by name.

    Args:
        name: Theme name.
        **overrides: Optional customizations.

    Returns:
        Theme instance.

    Raises:
        KeyError: If theme not found.
    """
    if name not in BUILT_IN_THEMES:
        raise KeyError(
            f"Theme '{name}' not found. "
            f"Available: {list(BUILT_IN_THEMES.keys())}"
        )
    return BUILT_IN_THEMES[name](**overrides)


def list_themes() -> list[str]:
    """List available theme names.

    Returns:
        List of theme names.
    """
    return list(BUILT_IN_THEMES.keys())
