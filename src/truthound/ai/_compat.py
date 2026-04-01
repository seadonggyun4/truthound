"""Compatibility helpers for the optional ``truthound.ai`` namespace."""

from __future__ import annotations


def ensure_ai_dependencies() -> None:
    """Ensure the optional AI dependency set is available."""

    try:
        import pydantic  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via import contract tests
        raise ImportError(
            "truthound.ai requires the optional AI dependency set. "
            "Install it with: pip install truthound[ai]"
        ) from exc
