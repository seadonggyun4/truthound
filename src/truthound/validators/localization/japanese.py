"""Japanese localization validators.

This module provides validators for Japanese-specific formats:
- Postal codes (郵便番号)
- My Number (マイナンバー)
"""

import re
from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.registry import register_validator
from truthound.validators.localization.base import LocalizationValidator


@register_validator
class JapanesePostalCodeValidator(LocalizationValidator):
    """Validates Japanese postal codes (郵便番号).

    Format: XXX-XXXX (7 digits)
    The first 3 digits indicate prefecture/region.

    Valid ranges:
    - 001-0xx to 099-xxxx: Hokkaido
    - 100-xxxx to 209-xxxx: Tokyo
    - 210-xxxx to 259-xxxx: Kanagawa
    - etc.

    Example:
        validator = JapanesePostalCodeValidator(column="postal_code")
    """

    name = "japanese_postal_code"

    # Valid first 3 digit ranges (simplified - covers most cases)
    # Format: (min, max) for first 3 digits
    VALID_RANGES = [
        (1, 999),  # All 001-999 are potentially valid
    ]

    # Known invalid ranges (placeholder for future extension)
    INVALID_RANGES: list[tuple[int, int]] = []

    def __init__(
        self,
        column: str,
        strict_format: bool = False,
        **kwargs: Any,
    ):
        """Initialize Japanese postal code validator.

        Args:
            column: Column to validate
            strict_format: If True, require XXX-XXXX format with hyphen
            **kwargs: Additional config
        """
        super().__init__(column=column, **kwargs)
        self.strict_format = strict_format

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

        # For strict format, don't remove separators
        if not self.strict_format and self.remove_separators:
            value = re.sub(r"[\s\-ー－]", "", value)

        # Convert full-width digits to half-width
        value = self._to_half_width(value)

        return value

    def _to_half_width(self, value: str) -> str:
        """Convert full-width digits to half-width.

        Args:
            value: Input string

        Returns:
            String with half-width digits
        """
        full_width = "０１２３４５６７８９"
        half_width = "0123456789"

        result = value
        for fw, hw in zip(full_width, half_width):
            result = result.replace(fw, hw)
        return result

    def validate_value(self, value: str) -> bool:
        """Validate a Japanese postal code.

        Args:
            value: Postal code

        Returns:
            True if valid, False otherwise
        """
        if self.strict_format:
            # Must be in XXX-XXXX format
            if not re.match(r"^\d{3}-\d{4}$", value):
                return False
            value = value.replace("-", "")
        else:
            # Just digits
            if not value.isdigit():
                return False

        if len(value) != 7:
            return False

        # Validate first 3 digits are in valid range
        first_three = int(value[:3])

        # Check not in invalid ranges
        for min_val, max_val in self.INVALID_RANGES:
            if min_val <= first_three <= max_val:
                return False

        # Check in valid ranges
        for min_val, max_val in self.VALID_RANGES:
            if min_val <= first_three <= max_val:
                return True

        return False

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate the LazyFrame.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        df = lf.collect()
        if len(df) == 0:
            return []

        invalid_mask = self._get_invalid_mask(df)
        invalid_count = invalid_mask.sum() or 0

        issues: list[ValidationIssue] = []

        if invalid_count > 0:
            sample_invalid = df.filter(invalid_mask)[self.column].head(5).to_list()
            expected_format = "XXX-XXXX" if self.strict_format else "7 digits"
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="invalid_japanese_postal_code",
                    count=invalid_count,
                    severity=Severity.LOW,
                    details=(
                        f"Found {invalid_count} invalid Japanese postal codes. "
                        f"Sample: {sample_invalid}"
                    ),
                    expected=f"Valid Japanese postal code ({expected_format})",
                )
            )

        return issues


@register_validator
class JapaneseMyNumberValidator(LocalizationValidator):
    """Validates Japanese My Number (マイナンバー).

    Format: 12 digits
    Uses modulo 11 check digit algorithm.

    Example:
        validator = JapaneseMyNumberValidator(column="my_number")
    """

    name = "japanese_my_number"

    def __init__(
        self,
        column: str,
        mask_output: bool = True,
        **kwargs: Any,
    ):
        """Initialize Japanese My Number validator.

        Args:
            column: Column to validate
            mask_output: Whether to mask numbers in error output
            **kwargs: Additional config
        """
        super().__init__(column=column, **kwargs)
        self.mask_output = mask_output

    def validate_value(self, value: str) -> bool:
        """Validate a Japanese My Number.

        Args:
            value: My Number (digits only)

        Returns:
            True if valid, False otherwise
        """
        # Must be exactly 12 digits
        if not value.isdigit() or len(value) != 12:
            return False

        digits = [int(d) for d in value]

        # Calculate check digit using modulo 11 algorithm
        # Weights: q_n = ((n + 1) mod 6) + 1 for position n (1-indexed from right)
        total = 0
        for i in range(11):
            pos = 11 - i  # Position from right (1-11)
            weight = (pos % 6) + 1 if pos <= 6 else ((pos - 6) % 6) + 1
            total += digits[i] * weight

        remainder = total % 11
        if remainder <= 1:
            check_digit = 0
        else:
            check_digit = 11 - remainder

        return check_digit == digits[11]

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate the LazyFrame.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        df = lf.collect()
        if len(df) == 0:
            return []

        invalid_mask = self._get_invalid_mask(df)
        invalid_count = invalid_mask.sum() or 0

        issues: list[ValidationIssue] = []

        if invalid_count > 0:
            if self.mask_output:
                sample_details = f"Found {invalid_count} invalid My Numbers (masked for privacy)"
            else:
                sample_invalid = df.filter(invalid_mask)[self.column].head(5).to_list()
                sample_details = f"Sample: {sample_invalid}"

            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="invalid_japanese_my_number",
                    count=invalid_count,
                    severity=Severity.HIGH,
                    details=f"Found {invalid_count} invalid Japanese My Numbers. {sample_details}",
                    expected="Valid Japanese My Number (12 digits with valid check digit)",
                )
            )

        return issues
