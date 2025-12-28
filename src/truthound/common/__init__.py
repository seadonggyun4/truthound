"""Common utilities shared across Truthound modules.

This package contains shared infrastructure components:
- resilience: Circuit breaker, retry, bulkhead patterns
"""

from __future__ import annotations

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    """Lazy import for common modules."""
    if name == "resilience":
        from truthound.common import resilience
        return resilience
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__: list[str] = ["resilience"]
