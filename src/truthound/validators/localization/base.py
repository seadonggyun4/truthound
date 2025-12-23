"""Base classes for localization validators.

This module provides extensible base classes for implementing
region-specific validation algorithms for Asian locales.
"""

from abc import abstractmethod
from typing import Any
import re

import polars as pl

from truthound.validators.base import (
    Validator,
    ValidationIssue,
    StringValidatorMixin,
)


class LocalizationValidator(Validator, StringValidatorMixin):
    """Base class for localization validators.

    Localization validators check region-specific formats
    like national IDs, phone numbers, postal codes, and
    bank account numbers.

    Subclasses should implement:
        - validate_value(): Single value validation
        - validate(): Full validation logic
    """

    category = "localization"

    def __init__(
        self,
        column: str,
        strip_whitespace: bool = True,
        remove_separators: bool = True,
        allow_null: bool = True,
        **kwargs: Any,
    ):
        """Initialize localization validator.

        Args:
            column: Column to validate
            strip_whitespace: Whether to strip whitespace before validation
            remove_separators: Whether to remove common separators
            allow_null: Whether to allow null values
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.column = column
        self.strip_whitespace = strip_whitespace
        self.remove_separators = remove_separators
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

        if self.remove_separators:
            value = self._remove_common_separators(value)

        return value

    def _remove_common_separators(self, value: str) -> str:
        """Remove common separators from a value.

        Args:
            value: Input string

        Returns:
            Cleaned string
        """
        return re.sub(r"[\s\-\.]", "", value)

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
