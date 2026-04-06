"""Public source-key helpers for ``truthound.ai`` consumers."""

from __future__ import annotations

from typing import Any

from truthound.context import TruthoundContext, get_context


def resolve_source_key(
    data: Any = None,
    source: Any = None,
    *,
    context: TruthoundContext | None = None,
) -> str:
    """Resolve the canonical source key for a dashboard or AI route consumer."""

    active_context = context or get_context()
    return active_context.resolve_source_key(data=data, source=source)


__all__ = ["resolve_source_key"]
