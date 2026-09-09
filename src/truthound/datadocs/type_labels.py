"""Shared presentation labels for canonical and legacy Profile column types."""

from collections.abc import Mapping
from typing import Any


def profile_type_label(column: Mapping[str, Any]) -> str:
    """Return the first supplied nonempty type label without inferring a type.

    Canonical semantic and physical labels take precedence over legacy ``dtype``.
    Keep the selected text unchanged; each HTML or chart renderer owns escaping.
    An explicit ``unknown`` label is meaningful and is not replaced by inference.
    """
    for key in ("inferred_type", "physical_type", "dtype"):
        value = column.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return "unknown"
