"""Factory functions for creating quality reporters.

This module provides a registry-based factory pattern for creating
quality reporter instances. New reporter types can be registered
at runtime.
"""

from __future__ import annotations

from typing import Any, Callable, Type

from truthound.reporters.quality.base import (
    BaseQualityReporter,
    QualityReporterError,
)
from truthound.reporters.quality.config import QualityReporterConfig
from truthound.reporters.quality.reporters import (
    ConsoleQualityReporter,
    JsonQualityReporter,
    MarkdownQualityReporter,
    HtmlQualityReporter,
    JUnitQualityReporter,
)


# =============================================================================
# Registry
# =============================================================================

# Type for reporter constructor
QualityReporterConstructor = Callable[..., BaseQualityReporter[Any]]

# Registry of reporter constructors
_quality_reporter_registry: dict[str, QualityReporterConstructor] = {}

# Format aliases
_FORMAT_ALIASES: dict[str, str] = {
    "md": "markdown",
    "terminal": "console",
    "rich": "console",
    "xml": "junit",
}


# =============================================================================
# Registration
# =============================================================================


def register_quality_reporter(name: str) -> Callable[[QualityReporterConstructor], QualityReporterConstructor]:
    """Decorator to register a quality reporter type.

    Args:
        name: Name to register the reporter under.

    Returns:
        Decorator function.

    Example:
        >>> @register_quality_reporter("my_format")
        ... class MyQualityReporter(BaseQualityReporter):
        ...     pass
    """
    def decorator(cls: QualityReporterConstructor) -> QualityReporterConstructor:
        _quality_reporter_registry[name.lower()] = cls
        return cls

    return decorator


def unregister_quality_reporter(name: str) -> bool:
    """Unregister a quality reporter type.

    Args:
        name: Name of the reporter to unregister.

    Returns:
        True if the reporter was unregistered, False if not found.
    """
    name = name.lower()
    if name in _quality_reporter_registry:
        del _quality_reporter_registry[name]
        return True
    return False


# =============================================================================
# Factory Function
# =============================================================================


def get_quality_reporter(
    format: str,
    config: QualityReporterConfig | None = None,
    **kwargs: Any,
) -> BaseQualityReporter[Any]:
    """Create a quality reporter instance for the specified format.

    This is the primary entry point for creating quality reporters.
    It handles lazy loading and provides a uniform interface.

    Args:
        format: Name of the report format. Options:
            - "console": Console/terminal output with Rich markup
            - "json": JSON format output
            - "html": HTML format with optional charts
            - "markdown" or "md": Markdown format output
            - "junit" or "xml": JUnit XML for CI/CD
        config: Optional reporter configuration.
        **kwargs: Format-specific configuration options.

    Returns:
        Configured quality reporter instance.

    Raises:
        QualityReporterError: If the format is not available.

    Example:
        >>> # Console reporter
        >>> reporter = get_quality_reporter("console")
        >>>
        >>> # JSON reporter with indent
        >>> reporter = get_quality_reporter("json", indent=4)
        >>>
        >>> # HTML reporter with charts
        >>> reporter = get_quality_reporter("html", include_charts=True)
    """
    # Normalize format name
    format = format.lower().strip()

    # Apply aliases
    format = _FORMAT_ALIASES.get(format, format)

    # Check registry first (custom reporters)
    if format in _quality_reporter_registry:
        return _quality_reporter_registry[format](config=config, **kwargs)

    # Built-in reporters
    if format == "console":
        return ConsoleQualityReporter(config=config, **kwargs)

    elif format == "json":
        return JsonQualityReporter(config=config, **kwargs)

    elif format == "markdown":
        return MarkdownQualityReporter(config=config, **kwargs)

    elif format == "html":
        return HtmlQualityReporter(config=config, **kwargs)

    elif format == "junit":
        return JUnitQualityReporter(config=config, **kwargs)

    else:
        available = list_quality_formats()
        raise QualityReporterError(
            f"Unknown quality reporter format: {format}. "
            f"Available formats: {', '.join(sorted(available))}"
        )


def list_quality_formats() -> list[str]:
    """List all available quality report formats.

    Returns:
        List of format names that can be used with get_quality_reporter().
    """
    # Built-in formats
    formats = ["console", "json", "html", "markdown", "junit"]

    # Add registered custom formats
    formats.extend(_quality_reporter_registry.keys())

    # Add aliases
    formats.extend(_FORMAT_ALIASES.keys())

    return sorted(set(formats))


def is_quality_format_available(format: str) -> bool:
    """Check if a quality report format is available.

    Args:
        format: Name of the format to check.

    Returns:
        True if the format is available.
    """
    format = format.lower().strip()
    format = _FORMAT_ALIASES.get(format, format)

    if format in _quality_reporter_registry:
        return True

    if format in ("console", "json", "html", "markdown", "junit"):
        return True

    return False


# =============================================================================
# Convenience Functions
# =============================================================================


def create_console_reporter(**kwargs: Any) -> ConsoleQualityReporter:
    """Create a console quality reporter.

    Args:
        **kwargs: Reporter configuration options.

    Returns:
        Console quality reporter.
    """
    return ConsoleQualityReporter(**kwargs)


def create_json_reporter(**kwargs: Any) -> JsonQualityReporter:
    """Create a JSON quality reporter.

    Args:
        **kwargs: Reporter configuration options.

    Returns:
        JSON quality reporter.
    """
    return JsonQualityReporter(**kwargs)


def create_html_reporter(**kwargs: Any) -> HtmlQualityReporter:
    """Create an HTML quality reporter.

    Args:
        **kwargs: Reporter configuration options.

    Returns:
        HTML quality reporter.
    """
    return HtmlQualityReporter(**kwargs)


def create_markdown_reporter(**kwargs: Any) -> MarkdownQualityReporter:
    """Create a Markdown quality reporter.

    Args:
        **kwargs: Reporter configuration options.

    Returns:
        Markdown quality reporter.
    """
    return MarkdownQualityReporter(**kwargs)
