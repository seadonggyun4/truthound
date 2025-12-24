"""Factory functions for creating reporters.

This module provides a registry-based factory pattern for creating reporter
instances. New reporter types can be registered at runtime.
"""

from __future__ import annotations

from typing import Any, Callable

from truthound.reporters.base import BaseReporter, ReporterError

# Type for reporter constructor functions
ReporterConstructor = Callable[..., BaseReporter[Any, Any]]

# Registry of reporter constructors
_reporter_registry: dict[str, ReporterConstructor] = {}


def register_reporter(name: str) -> Callable[[ReporterConstructor], ReporterConstructor]:
    """Decorator to register a reporter type.

    Args:
        name: Name to register the reporter under.

    Returns:
        Decorator function.

    Example:
        >>> @register_reporter("my_format")
        ... class MyReporter(BaseReporter):
        ...     pass
    """

    def decorator(cls: ReporterConstructor) -> ReporterConstructor:
        _reporter_registry[name] = cls
        return cls

    return decorator


def get_reporter(format: str, **kwargs: Any) -> BaseReporter[Any, Any]:
    """Create a reporter instance for the specified format.

    This is the primary entry point for creating reporters. It handles
    lazy loading of reporter modules and provides a uniform interface.

    Args:
        format: Name of the report format to use. Options:
            - "json": JSON format output
            - "html": HTML format output (requires jinja2)
            - "console": Console/terminal output (using Rich)
            - "markdown": Markdown format output
        **kwargs: Format-specific configuration options.

    Returns:
        Configured reporter instance.

    Raises:
        ReporterError: If the format is not available or configuration fails.

    Example:
        >>> # JSON reporter
        >>> reporter = get_reporter("json", output_path="report.json")
        >>>
        >>> # HTML reporter
        >>> reporter = get_reporter("html", title="My Validation Report")
        >>>
        >>> # Console reporter
        >>> reporter = get_reporter("console", color=True)
    """
    # Normalize format name
    format = format.lower().strip()

    # Check if already registered
    if format in _reporter_registry:
        return _reporter_registry[format](**kwargs)

    # Lazy load built-in reporters
    if format == "json":
        from truthound.reporters.json_reporter import JSONReporter

        return JSONReporter(**kwargs)

    elif format == "html":
        try:
            from truthound.reporters.html_reporter import HTMLReporter

            return HTMLReporter(**kwargs)
        except ImportError as e:
            raise ReporterError(
                f"HTML reporter requires jinja2. Install with: pip install truthound[all]\n"
                f"Original error: {e}"
            )

    elif format in ("console", "terminal", "rich"):
        from truthound.reporters.console_reporter import ConsoleReporter

        return ConsoleReporter(**kwargs)

    elif format in ("markdown", "md"):
        from truthound.reporters.markdown_reporter import MarkdownReporter

        return MarkdownReporter(**kwargs)

    else:
        available = list(_reporter_registry.keys()) + [
            "json",
            "html",
            "console",
            "markdown",
        ]
        raise ReporterError(
            f"Unknown reporter format: {format}. "
            f"Available formats: {', '.join(sorted(set(available)))}"
        )


def list_available_formats() -> list[str]:
    """List all available report formats.

    Returns:
        List of format names that can be used with get_reporter().
    """
    # Built-in formats always available
    formats = ["json", "console", "markdown"]

    # Check optional formats
    try:
        import jinja2

        formats.append("html")
    except ImportError:
        pass

    # Add registered formats
    formats.extend(_reporter_registry.keys())

    return sorted(set(formats))


def is_format_available(format: str) -> bool:
    """Check if a report format is available.

    Args:
        format: Name of the format to check.

    Returns:
        True if the format is available, False otherwise.
    """
    format = format.lower().strip()

    if format in ("json", "console", "terminal", "rich", "markdown", "md"):
        return True

    if format in _reporter_registry:
        return True

    if format == "html":
        try:
            import jinja2

            return True
        except ImportError:
            return False

    return False
