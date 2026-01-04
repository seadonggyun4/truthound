"""Validator name resolution utilities.

This module provides simple functions for resolving validator names
from various formats to canonical snake_case form.

Example:
    from truthound.profiler.integration.naming import resolve_validator_name

    resolve_validator_name("ColumnTypeValidator")  # Returns "column_type"
    resolve_validator_name("column_type")  # Returns "column_type"
    resolve_validator_name("column-type")  # Returns "column_type"
"""

from __future__ import annotations

import re
from functools import lru_cache


@lru_cache(maxsize=256)
def resolve_validator_name(name: str) -> str:
    """Convert any validator name format to canonical snake_case.

    Supports:
    - PascalCase: "ColumnTypeValidator" → "column_type"
    - snake_case: "column_type" → "column_type"
    - kebab-case: "column-type" → "column_type"

    Args:
        name: The validator name in any format.

    Returns:
        Canonical snake_case name.

    Raises:
        ValueError: If name is empty.

    Example:
        >>> resolve_validator_name("ColumnTypeValidator")
        'column_type'
        >>> resolve_validator_name("not_null")
        'not_null'
        >>> resolve_validator_name("column-type")
        'column_type'
    """
    if not name:
        raise ValueError("Validator name cannot be empty")

    result = name

    # Remove "Validator" suffix (PascalCase)
    if result.endswith("Validator"):
        result = result[:-9]

    # Normalize separators first (before case conversion)
    result = result.replace("-", "_")
    result = result.replace(".", "_")

    # Convert PascalCase to snake_case (only if mixed case)
    # Skip if already all uppercase or all lowercase
    if not result.isupper() and not result.islower():
        result = re.sub(r'(?<!^)(?=[A-Z])', '_', result)

    # Lowercase
    result = result.lower()

    # Remove _validator suffix if still present
    if result.endswith("_validator"):
        result = result[:-10]

    # Clean up multiple underscores
    while "__" in result:
        result = result.replace("__", "_")

    return result
