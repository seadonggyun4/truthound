"""Component registry for the report generation pipeline.

This module provides a centralized registry for all pipeline components:
- Transformers
- Renderers
- Themes
- Exporters

The registry supports both programmatic registration and decorator-based registration.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from truthound.datadocs.transformers.base import Transformer
    from truthound.datadocs.renderers.base import Renderer
    from truthound.datadocs.themes.base import Theme
    from truthound.datadocs.exporters.base import Exporter


T = TypeVar("T")


class ComponentRegistry:
    """Centralized registry for pipeline components.

    This registry maintains collections of:
    - Transformers: Data transformation stages
    - Renderers: Template rendering engines
    - Themes: Visual styling configurations
    - Exporters: Output format handlers

    Components can be registered by name and retrieved for use in the pipeline.

    Example:
        registry = ComponentRegistry()

        # Register components
        registry.register_transformer("i18n", I18nTransformer)
        registry.register_theme("enterprise", EnterpriseTheme)

        # Retrieve components
        transformer = registry.get_transformer("i18n")
        theme = registry.get_theme("enterprise")
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._transformers: dict[str, type] = {}
        self._renderers: dict[str, type] = {}
        self._themes: dict[str, type | Any] = {}
        self._exporters: dict[str, type] = {}
        self._theme_instances: dict[str, Any] = {}

    # Transformer registration

    def register_transformer(
        self,
        name: str,
        transformer_class: type,
        replace: bool = False,
    ) -> None:
        """Register a transformer class.

        Args:
            name: Unique name for the transformer.
            transformer_class: Transformer class to register.
            replace: If True, replace existing registration.

        Raises:
            ValueError: If name already exists and replace is False.
        """
        if name in self._transformers and not replace:
            raise ValueError(f"Transformer '{name}' already registered")
        self._transformers[name] = transformer_class

    def get_transformer(
        self,
        name: str,
        **kwargs: Any,
    ) -> "Transformer":
        """Get a transformer instance by name.

        Args:
            name: Transformer name.
            **kwargs: Arguments to pass to the constructor.

        Returns:
            Transformer instance.

        Raises:
            KeyError: If transformer not found.
        """
        if name not in self._transformers:
            available = list(self._transformers.keys())
            raise KeyError(
                f"Transformer '{name}' not found. Available: {available}"
            )
        return self._transformers[name](**kwargs)

    def list_transformers(self) -> list[str]:
        """List all registered transformer names.

        Returns:
            List of transformer names.
        """
        return list(self._transformers.keys())

    def has_transformer(self, name: str) -> bool:
        """Check if a transformer is registered.

        Args:
            name: Transformer name.

        Returns:
            True if registered.
        """
        return name in self._transformers

    # Renderer registration

    def register_renderer(
        self,
        name: str,
        renderer_class: type,
        replace: bool = False,
    ) -> None:
        """Register a renderer class.

        Args:
            name: Unique name for the renderer.
            renderer_class: Renderer class to register.
            replace: If True, replace existing registration.

        Raises:
            ValueError: If name already exists and replace is False.
        """
        if name in self._renderers and not replace:
            raise ValueError(f"Renderer '{name}' already registered")
        self._renderers[name] = renderer_class

    def get_renderer(
        self,
        name: str,
        **kwargs: Any,
    ) -> "Renderer":
        """Get a renderer instance by name.

        Args:
            name: Renderer name.
            **kwargs: Arguments to pass to the constructor.

        Returns:
            Renderer instance.

        Raises:
            KeyError: If renderer not found.
        """
        if name not in self._renderers:
            available = list(self._renderers.keys())
            raise KeyError(
                f"Renderer '{name}' not found. Available: {available}"
            )
        return self._renderers[name](**kwargs)

    def list_renderers(self) -> list[str]:
        """List all registered renderer names.

        Returns:
            List of renderer names.
        """
        return list(self._renderers.keys())

    def has_renderer(self, name: str) -> bool:
        """Check if a renderer is registered.

        Args:
            name: Renderer name.

        Returns:
            True if registered.
        """
        return name in self._renderers

    # Theme registration

    def register_theme(
        self,
        name: str,
        theme: type | Any,
        replace: bool = False,
    ) -> None:
        """Register a theme class or instance.

        Args:
            name: Unique name for the theme.
            theme: Theme class or instance to register.
            replace: If True, replace existing registration.

        Raises:
            ValueError: If name already exists and replace is False.
        """
        if name in self._themes and not replace:
            raise ValueError(f"Theme '{name}' already registered")
        self._themes[name] = theme

        # If it's an instance, cache it
        if not isinstance(theme, type):
            self._theme_instances[name] = theme

    def get_theme(
        self,
        name: str,
        **kwargs: Any,
    ) -> "Theme":
        """Get a theme instance by name.

        Args:
            name: Theme name.
            **kwargs: Arguments to pass to the constructor (if class).

        Returns:
            Theme instance.

        Raises:
            KeyError: If theme not found.
        """
        if name not in self._themes:
            available = list(self._themes.keys())
            raise KeyError(
                f"Theme '{name}' not found. Available: {available}"
            )

        # Return cached instance if available
        if name in self._theme_instances and not kwargs:
            return self._theme_instances[name]

        theme = self._themes[name]
        if isinstance(theme, type):
            instance = theme(**kwargs)
            # Cache if no kwargs
            if not kwargs:
                self._theme_instances[name] = instance
            return instance
        return theme

    def list_themes(self) -> list[str]:
        """List all registered theme names.

        Returns:
            List of theme names.
        """
        return list(self._themes.keys())

    def has_theme(self, name: str) -> bool:
        """Check if a theme is registered.

        Args:
            name: Theme name.

        Returns:
            True if registered.
        """
        return name in self._themes

    # Exporter registration

    def register_exporter(
        self,
        name: str,
        exporter_class: type,
        replace: bool = False,
    ) -> None:
        """Register an exporter class.

        Args:
            name: Unique name for the exporter (usually format like 'html', 'pdf').
            exporter_class: Exporter class to register.
            replace: If True, replace existing registration.

        Raises:
            ValueError: If name already exists and replace is False.
        """
        if name in self._exporters and not replace:
            raise ValueError(f"Exporter '{name}' already registered")
        self._exporters[name] = exporter_class

    def get_exporter(
        self,
        name: str,
        **kwargs: Any,
    ) -> "Exporter":
        """Get an exporter instance by name.

        Args:
            name: Exporter name (format).
            **kwargs: Arguments to pass to the constructor.

        Returns:
            Exporter instance.

        Raises:
            KeyError: If exporter not found.
        """
        if name not in self._exporters:
            available = list(self._exporters.keys())
            raise KeyError(
                f"Exporter '{name}' not found. Available: {available}"
            )
        return self._exporters[name](**kwargs)

    def list_exporters(self) -> list[str]:
        """List all registered exporter names.

        Returns:
            List of exporter names (formats).
        """
        return list(self._exporters.keys())

    def has_exporter(self, name: str) -> bool:
        """Check if an exporter is registered.

        Args:
            name: Exporter name.

        Returns:
            True if registered.
        """
        return name in self._exporters

    # Utility methods

    def clear(self) -> None:
        """Clear all registrations."""
        self._transformers.clear()
        self._renderers.clear()
        self._themes.clear()
        self._exporters.clear()
        self._theme_instances.clear()

    def stats(self) -> dict[str, int]:
        """Get registration statistics.

        Returns:
            Dictionary with counts of each component type.
        """
        return {
            "transformers": len(self._transformers),
            "renderers": len(self._renderers),
            "themes": len(self._themes),
            "exporters": len(self._exporters),
        }


# Global registry instance
component_registry = ComponentRegistry()


# Decorator factories for registration


def register_transformer(
    name: str,
    registry: ComponentRegistry | None = None,
) -> Callable[[type[T]], type[T]]:
    """Decorator to register a transformer class.

    Args:
        name: Unique name for the transformer.
        registry: Registry to use (default: global registry).

    Returns:
        Decorator function.

    Example:
        @register_transformer("i18n")
        class I18nTransformer:
            ...
    """
    reg = registry or component_registry

    def decorator(cls: type[T]) -> type[T]:
        reg.register_transformer(name, cls)
        return cls

    return decorator


def register_renderer(
    name: str,
    registry: ComponentRegistry | None = None,
) -> Callable[[type[T]], type[T]]:
    """Decorator to register a renderer class.

    Args:
        name: Unique name for the renderer.
        registry: Registry to use (default: global registry).

    Returns:
        Decorator function.

    Example:
        @register_renderer("jinja")
        class JinjaRenderer:
            ...
    """
    reg = registry or component_registry

    def decorator(cls: type[T]) -> type[T]:
        reg.register_renderer(name, cls)
        return cls

    return decorator


def register_theme(
    name: str,
    registry: ComponentRegistry | None = None,
) -> Callable[[type[T]], type[T]]:
    """Decorator to register a theme class.

    Args:
        name: Unique name for the theme.
        registry: Registry to use (default: global registry).

    Returns:
        Decorator function.

    Example:
        @register_theme("enterprise")
        class EnterpriseTheme:
            ...
    """
    reg = registry or component_registry

    def decorator(cls: type[T]) -> type[T]:
        reg.register_theme(name, cls)
        return cls

    return decorator


def register_exporter(
    name: str,
    registry: ComponentRegistry | None = None,
) -> Callable[[type[T]], type[T]]:
    """Decorator to register an exporter class.

    Args:
        name: Unique name for the exporter (format).
        registry: Registry to use (default: global registry).

    Returns:
        Decorator function.

    Example:
        @register_exporter("pdf")
        class PdfExporter:
            ...
    """
    reg = registry or component_registry

    def decorator(cls: type[T]) -> type[T]:
        reg.register_exporter(name, cls)
        return cls

    return decorator
