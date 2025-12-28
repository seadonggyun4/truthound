"""Theme loader for loading themes from external files.

This module provides utilities for loading themes from YAML, JSON,
or dictionary configurations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from truthound.datadocs.themes.base import (
    ThemeConfig,
    ThemeColors,
    ThemeTypography,
    ThemeSpacing,
    ThemeAssets,
)
from truthound.datadocs.themes.enterprise import (
    EnterpriseTheme,
    EnterpriseThemeConfig,
    BrandingConfig,
)


class ThemeLoader:
    """Loader for theme configurations from various sources.

    Supports loading from:
    - YAML files
    - JSON files
    - Python dictionaries

    Example:
        loader = ThemeLoader()

        # Load from YAML
        theme = loader.load("./themes/my-theme.yaml")

        # Load from JSON
        theme = loader.load("./themes/my-theme.json")

        # Load from dict
        theme = loader.from_dict({
            "name": "my-theme",
            "colors": {"primary": "#FF5722"},
        })
    """

    def load(self, path: Path | str) -> EnterpriseTheme:
        """Load a theme from a file.

        Args:
            path: Path to theme file (YAML or JSON).

        Returns:
            Loaded theme instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is unsupported.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Theme file not found: {path}")

        suffix = path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            return self.load_yaml(path)
        elif suffix == ".json":
            return self.load_json(path)
        else:
            raise ValueError(f"Unsupported theme file format: {suffix}")

    def load_yaml(self, path: Path | str) -> EnterpriseTheme:
        """Load a theme from a YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            Loaded theme instance.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML theme loading. "
                "Install with: pip install pyyaml"
            )

        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return self.from_dict(data)

    def load_json(self, path: Path | str) -> EnterpriseTheme:
        """Load a theme from a JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            Loaded theme instance.
        """
        import json

        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self.from_dict(data)

    def from_dict(self, data: dict[str, Any]) -> EnterpriseTheme:
        """Create a theme from a dictionary.

        Args:
            data: Theme configuration dictionary.

        Returns:
            EnterpriseTheme instance.
        """
        # Parse branding
        branding_data = data.get("branding", {})
        branding = BrandingConfig(
            company_name=branding_data.get("company_name", data.get("company_name", "")),
            tagline=branding_data.get("tagline", ""),
            logo_url=branding_data.get("logo_url", data.get("logo_url")),
            logo_base64=branding_data.get("logo_base64", data.get("logo_base64")),
            logo_height=branding_data.get("logo_height", "40px"),
            favicon_url=branding_data.get("favicon_url"),
            copyright_text=branding_data.get("copyright_text", ""),
            website_url=branding_data.get("website_url", ""),
            support_email=branding_data.get("support_email", ""),
        )

        # Parse colors
        colors_data = data.get("colors", {})
        colors = ThemeColors(
            background=colors_data.get("background", "#fafbfc"),
            surface=colors_data.get("surface", "#ffffff"),
            text_primary=colors_data.get("text_primary", "#1f2937"),
            text_secondary=colors_data.get("text_secondary", "#6b7280"),
            primary=colors_data.get("primary", data.get("primary_color", "#2563eb")),
            secondary=colors_data.get("secondary", data.get("secondary_color", "#4f46e5")),
            accent=colors_data.get("accent", "#7c3aed"),
            success=colors_data.get("success", "#059669"),
            warning=colors_data.get("warning", "#d97706"),
            error=colors_data.get("error", "#dc2626"),
            info=colors_data.get("info", "#0284c7"),
            border=colors_data.get("border", "#d1d5db"),
            shadow=colors_data.get("shadow", "rgba(0, 0, 0, 0.05)"),
            chart_palette=tuple(colors_data.get("chart_palette", [])) or ThemeColors().chart_palette,
        )

        # Parse typography
        typography_data = data.get("typography", {})
        typography = ThemeTypography(
            font_family=typography_data.get("font_family", data.get("font_family", "'Inter', system-ui, sans-serif")),
            font_family_mono=typography_data.get("font_family_mono", "'JetBrains Mono', monospace"),
        )

        # Parse spacing
        spacing_data = data.get("spacing", {})
        spacing_kwargs = {}
        for key in ("border_radius_sm", "border_radius_md", "border_radius_lg", "border_radius_xl",
                    "spacing_xs", "spacing_sm", "spacing_md", "spacing_lg", "spacing_xl",
                    "shadow_sm", "shadow_md", "shadow_lg", "shadow_xl"):
            if key in spacing_data:
                spacing_kwargs[key] = spacing_data[key]
        spacing = ThemeSpacing(**spacing_kwargs) if spacing_kwargs else ThemeSpacing()

        # Build config
        config = EnterpriseThemeConfig(
            name=data.get("name", "custom"),
            display_name=data.get("display_name", data.get("name", "Custom Theme")),
            description=data.get("description", ""),
            branding=branding,
            colors=colors,
            typography=typography,
            spacing=spacing,
            custom_css=data.get("custom_css", ""),
            print_css=data.get("print_css", ""),
            show_header=data.get("show_header", True),
            show_footer=data.get("show_footer", True),
            show_toc=data.get("show_toc", True),
            show_branding=data.get("show_branding", True),
            pdf_options=data.get("pdf_options", {}),
        )

        return EnterpriseTheme.from_config(config)

    def to_dict(self, theme: EnterpriseTheme) -> dict[str, Any]:
        """Convert a theme to a dictionary.

        Args:
            theme: Theme to convert.

        Returns:
            Dictionary representation.
        """
        config = theme.enterprise_config
        branding = config.branding
        colors = config.colors
        typography = config.typography
        spacing = config.spacing

        return {
            "name": config.name,
            "display_name": config.display_name,
            "description": config.description,
            "branding": {
                "company_name": branding.company_name,
                "tagline": branding.tagline,
                "logo_url": branding.logo_url,
                "logo_base64": branding.logo_base64,
                "logo_height": branding.logo_height,
                "favicon_url": branding.favicon_url,
                "copyright_text": branding.copyright_text,
                "website_url": branding.website_url,
                "support_email": branding.support_email,
            },
            "colors": {
                "background": colors.background,
                "surface": colors.surface,
                "text_primary": colors.text_primary,
                "text_secondary": colors.text_secondary,
                "primary": colors.primary,
                "secondary": colors.secondary,
                "accent": colors.accent,
                "success": colors.success,
                "warning": colors.warning,
                "error": colors.error,
                "info": colors.info,
                "border": colors.border,
                "shadow": colors.shadow,
                "chart_palette": list(colors.chart_palette),
            },
            "typography": {
                "font_family": typography.font_family,
                "font_family_mono": typography.font_family_mono,
            },
            "spacing": {
                "border_radius_sm": spacing.border_radius_sm,
                "border_radius_md": spacing.border_radius_md,
                "border_radius_lg": spacing.border_radius_lg,
                "border_radius_xl": spacing.border_radius_xl,
            },
            "custom_css": config.custom_css,
            "show_header": config.show_header,
            "show_footer": config.show_footer,
            "show_toc": config.show_toc,
            "show_branding": config.show_branding,
        }

    def save_yaml(self, theme: EnterpriseTheme, path: Path | str) -> None:
        """Save a theme to a YAML file.

        Args:
            theme: Theme to save.
            path: Output file path.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML export. "
                "Install with: pip install pyyaml"
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict(theme)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def save_json(self, theme: EnterpriseTheme, path: Path | str) -> None:
        """Save a theme to a JSON file.

        Args:
            theme: Theme to save.
            path: Output file path.
        """
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict(theme)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# Convenience functions

_loader = ThemeLoader()


def load_theme_from_yaml(path: Path | str) -> EnterpriseTheme:
    """Load a theme from a YAML file.

    Args:
        path: Path to YAML file.

    Returns:
        Loaded theme.
    """
    return _loader.load_yaml(path)


def load_theme_from_json(path: Path | str) -> EnterpriseTheme:
    """Load a theme from a JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Loaded theme.
    """
    return _loader.load_json(path)


def load_theme_from_dict(data: dict[str, Any]) -> EnterpriseTheme:
    """Create a theme from a dictionary.

    Args:
        data: Theme configuration dictionary.

    Returns:
        EnterpriseTheme instance.
    """
    return _loader.from_dict(data)
