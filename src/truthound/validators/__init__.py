"""Built-in validators for data quality checks."""

from __future__ import annotations

from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.duplicate import DuplicateValidator
from truthound.validators.format import FormatValidator
from truthound.validators.null import NullValidator
from truthound.validators.outlier import OutlierValidator
from truthound.validators.range import RangeValidator
from truthound.validators.type import TypeValidator
from truthound.validators.unique import UniqueValidator

__all__ = [
    "Validator",
    "ValidationIssue",
    "NullValidator",
    "DuplicateValidator",
    "TypeValidator",
    "RangeValidator",
    "OutlierValidator",
    "FormatValidator",
    "UniqueValidator",
]

# Registry of built-in validators
BUILTIN_VALIDATORS: dict[str, type[Validator]] = {
    "null": NullValidator,
    "duplicate": DuplicateValidator,
    "type": TypeValidator,
    "range": RangeValidator,
    "outlier": OutlierValidator,
    "format": FormatValidator,
    "unique": UniqueValidator,
}


def get_validator(name: str) -> type[Validator]:
    """Get a validator class by name.

    Args:
        name: Name of the validator.

    Returns:
        Validator class.

    Raises:
        ValueError: If the validator name is not found.
    """
    if name not in BUILTIN_VALIDATORS:
        raise ValueError(
            f"Unknown validator: {name}. "
            f"Available validators: {', '.join(BUILTIN_VALIDATORS.keys())}"
        )
    return BUILTIN_VALIDATORS[name]
