"""Factory functions for creating reporters.

This module provides a registry-based factory pattern for creating reporter
instances. New reporter types can be registered at runtime.

Includes support for CI/CD platform-specific reporters.
"""

from __future__ import annotations

from typing import Any, Callable

from truthound.reporters.base import BaseReporter, ReporterError

# Type for reporter constructor functions
ReporterConstructor = Callable[..., BaseReporter[Any, Any]]

# Registry of reporter constructors
_reporter_registry: dict[str, ReporterConstructor] = {}

# CI platform identifiers for routing
_CI_PLATFORMS = frozenset({
    "github", "github_actions", "github-actions",
    "gitlab", "gitlab_ci", "gitlab-ci",
    "jenkins",
    "azure", "azure_devops", "azure-devops", "azdo", "vsts",
    "circleci", "circle", "circle_ci", "circle-ci",
    "bitbucket", "bitbucket_pipelines", "bitbucket-pipelines",
    "ci", "ci-auto", "auto-ci",
})


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
            - CI platforms: "github", "gitlab", "jenkins", "azure",
              "circleci", "bitbucket", or "ci" for auto-detection
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
        >>>
        >>> # CI reporter (auto-detect platform)
        >>> reporter = get_reporter("ci")
        >>>
        >>> # Specific CI platform
        >>> reporter = get_reporter("github")
    """
    # Normalize format name
    format = format.lower().strip()

    # Check if already registered
    if format in _reporter_registry:
        return _reporter_registry[format](**kwargs)

    # Route CI platforms to CI reporter factory
    if format in _CI_PLATFORMS:
        from truthound.reporters.ci import get_ci_reporter

        # Map generic "ci" to auto-detection
        if format in ("ci", "ci-auto", "auto-ci"):
            return get_ci_reporter(None, **kwargs)
        return get_ci_reporter(format, **kwargs)

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
            "ci",
            "github",
            "gitlab",
            "jenkins",
            "azure",
            "circleci",
            "bitbucket",
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

    # CI platforms
    formats.extend(["ci", "github", "gitlab", "jenkins", "azure", "circleci", "bitbucket"])

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

    # CI platforms are always available
    if format in _CI_PLATFORMS:
        return True

    return False
