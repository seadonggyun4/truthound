"""Protocol definitions for reporter dependencies.

This module defines structural typing protocols for external library clients,
allowing type-safe code without requiring type stubs packages.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Jinja2TemplateProtocol(Protocol):
    """Protocol for Jinja2 Template."""

    def render(self, **context: Any) -> str:
        """Render the template with given context."""
        ...


@runtime_checkable
class Jinja2EnvironmentProtocol(Protocol):
    """Protocol for Jinja2 Environment.

    Defines the minimal interface used by HTMLReporter.
    """

    def get_template(self, name: str) -> Jinja2TemplateProtocol:
        """Get a template by name."""
        ...
