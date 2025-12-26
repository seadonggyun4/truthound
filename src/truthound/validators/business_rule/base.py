"""Base classes for business rule validators.

This module provides extensible base classes for implementing
domain-specific business rule validation algorithms.
"""

from abc import abstractmethod
from typing import Any

import polars as pl

from truthound.validators.base import (
    Validator,
    ValidationIssue,
    StringValidatorMixin,
)


class BusinessRuleValidator(Validator, StringValidatorMixin):
    """Base class for business rule validators.

    Business rule validators check domain-specific constraints
    like checksum algorithms, format standards, and regulatory
    compliance requirements.

    Subclasses should implement:
        - validate_value(): Single value validation
        - validate(): Full validation logic
    """

    category = "business_rule"

    def __init__(
        self,
        column: str,
        strip_whitespace: bool = True,
        ignore_case: bool = False,
        allow_null: bool = True,
        **kwargs: Any,
    ):
        """Initialize business rule validator.

        Args:
            column: Column to validate
            strip_whitespace: Whether to strip whitespace before validation
            ignore_case: Whether to ignore case
            allow_null: Whether to allow null values
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.column = column
        self.strip_whitespace = strip_whitespace
        self.ignore_case = ignore_case
        self.allow_null = allow_null

    def _preprocess_value(self, value: str | None) -> str | None:
        """Preprocess a value before validation.

        Args:
            value: Input value

        Returns:
            Preprocessed value
        """
        if value is None:
            return None

        if self.strip_whitespace:
            value = value.strip()

        if self.ignore_case:
            value = value.upper()

        return value

    def _remove_separators(self, value: str, separators: str = " -") -> str:
        """Remove common separators from a value.

        Args:
            value: Input string
            separators: Characters to remove

        Returns:
            Cleaned string
        """
        result = value
        for sep in separators:
            result = result.replace(sep, "")
        return result

    @abstractmethod
    def validate_value(self, value: str) -> bool:
        """Validate a single value.

        Args:
            value: Value to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    def _get_invalid_mask(self, df: pl.DataFrame) -> pl.Series:
        """Get mask of invalid values.

        Default implementation uses validate_value for each row.
        Override for better performance with vectorized operations.

        Args:
            df: Input DataFrame

        Returns:
            Boolean Series where True = invalid
        """
        values = df[self.column].to_list()
        invalid = []

        for val in values:
            if val is None:
                invalid.append(not self.allow_null)
            else:
                processed = self._preprocess_value(str(val))
                if processed is None or processed == "":
                    invalid.append(not self.allow_null)
                else:
                    invalid.append(not self.validate_value(processed))

        return pl.Series(invalid)


class ChecksumValidator(BusinessRuleValidator):
    """Base class for checksum-based validators.

    Provides common utilities for validators that use
    checksum algorithms (Luhn, ISBN, etc.).
    """

    def _extract_digits(self, value: str) -> list[int]:
        """Extract numeric digits from a string.

        Args:
            value: Input string

        Returns:
            List of integer digits
        """
        return [int(c) for c in value if c.isdigit()]
