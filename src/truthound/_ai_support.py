"""Root-level support probes for the optional ``truthound.ai`` namespace."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field

DEFAULT_AI_PROVIDER = "openai"
DEFAULT_AI_INSTALL_HINT = "pip install truthound[ai]"


@dataclass(frozen=True)
class AISupportStatus:
    """Resolved availability of the optional Truthound AI surface."""

    ready: bool
    provider_name: str = DEFAULT_AI_PROVIDER
    pydantic_available: bool = False
    openai_available: bool = False
    missing_dependencies: tuple[str, ...] = field(default_factory=tuple)
    install_hint: str = DEFAULT_AI_INSTALL_HINT


def get_ai_support_status() -> AISupportStatus:
    """Return whether the optional AI surface is ready for use."""

    pydantic_available = _dependency_available("pydantic")
    openai_available = _dependency_available("openai")
    missing: list[str] = []
    if not pydantic_available:
        missing.append("pydantic")
    if not openai_available:
        missing.append("openai")
    return AISupportStatus(
        ready=pydantic_available and openai_available,
        pydantic_available=pydantic_available,
        openai_available=openai_available,
        missing_dependencies=tuple(missing),
    )


def has_ai_support() -> bool:
    """Return ``True`` when the default OpenAI-backed AI surface is ready."""

    return get_ai_support_status().ready


def _dependency_available(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


__all__ = [
    "AISupportStatus",
    "DEFAULT_AI_INSTALL_HINT",
    "DEFAULT_AI_PROVIDER",
    "get_ai_support_status",
    "has_ai_support",
]
