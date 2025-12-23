"""Checksum-based business rule validators.

This module provides validators for numbers that use checksum
algorithms for validation (credit cards, ISBN, etc.).
"""

import re
from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.registry import register_validator
from truthound.validators.business_rule.base import ChecksumValidator


@register_validator
class LuhnValidator(ChecksumValidator):
    """Validates numbers using the Luhn algorithm (mod 10).

    The Luhn algorithm is used for:
    - Credit card numbers (Visa, MasterCard, Amex, etc.)
    - IMEI numbers
    - National identification numbers
    - Various other identification numbers

    Example:
        validator = LuhnValidator(
            column="credit_card",
            min_length=13,
            max_length=19,
        )
    """

    name = "luhn"

    def __init__(
        self,
        column: str,
        min_length: int = 1,
        max_length: int = 19,
        allowed_prefixes: list[str] | None = None,
        **kwargs: Any,
    ):
        """Initialize Luhn validator.

        Args:
            column: Column containing numbers to validate
            min_length: Minimum number of digits
            max_length: Maximum number of digits
            allowed_prefixes: Optional list of allowed prefixes (e.g., ["4", "5"])
            **kwargs: Additional config
        """
        super().__init__(column=column, **kwargs)
        self.min_length = min_length
        self.max_length = max_length
        self.allowed_prefixes = allowed_prefixes

    def _luhn_checksum(self, digits: list[int]) -> int:
        """Calculate Luhn checksum.

        Args:
            digits: List of digits

        Returns:
            Checksum value (0 if valid)
        """
        total = 0
        for i, digit in enumerate(reversed(digits)):
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit
        return total % 10

    def validate_value(self, value: str) -> bool:
        """Validate a value using Luhn algorithm.

        Args:
            value: Value to validate

        Returns:
            True if valid
        """
        # Remove separators
        cleaned = self._remove_separators(value)

        # Check if all digits
        if not cleaned.isdigit():
            return False

        # Check length
        if len(cleaned) < self.min_length or len(cleaned) > self.max_length:
            return False

        # Check prefix if specified
        if self.allowed_prefixes:
            if not any(cleaned.startswith(p) for p in self.allowed_prefixes):
                return False

        # Luhn checksum
        digits = [int(d) for d in cleaned]
        return self._luhn_checksum(digits) == 0

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate column for Luhn checksum.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        df = lf.select(pl.col(self.column)).collect()

        if len(df) == 0:
            return []

        invalid_mask = self._get_invalid_mask(df)
        invalid_count = int(invalid_mask.sum())

        if invalid_count == 0:
            return []

        total_count = len(df)
        invalid_ratio = invalid_count / total_count

        # Get sample invalid values
        invalid_df = df.filter(invalid_mask)
        samples = invalid_df[self.column].head(5).to_list()

        return [
            ValidationIssue(
                column=self.column,
                issue_type="invalid_luhn_checksum",
                count=invalid_count,
                severity=self._calculate_severity(invalid_ratio),
                details=(
                    f"Found {invalid_count} values ({invalid_ratio:.2%}) "
                    f"failing Luhn checksum validation. "
                    f"Samples: {samples}"
                ),
                expected="Valid Luhn checksum (mod 10)",
            )
        ]

    def _calculate_severity(self, ratio: float) -> Severity:
        """Calculate severity based on invalid ratio."""
        if ratio < 0.01:
            return Severity.LOW
        elif ratio < 0.05:
            return Severity.MEDIUM
        elif ratio < 0.1:
            return Severity.HIGH
        else:
            return Severity.CRITICAL


@register_validator
class ISBNValidator(ChecksumValidator):
    """Validates ISBN (International Standard Book Number).

    Supports both ISBN-10 and ISBN-13 formats with proper
    checksum validation.

    ISBN-10: Uses modulo 11 checksum
    ISBN-13: Uses modulo 10 checksum (EAN-13)

    Example:
        validator = ISBNValidator(
            column="isbn",
            allow_isbn10=True,
            allow_isbn13=True,
        )
    """

    name = "isbn"

    def __init__(
        self,
        column: str,
        allow_isbn10: bool = True,
        allow_isbn13: bool = True,
        require_format: str | None = None,
        **kwargs: Any,
    ):
        """Initialize ISBN validator.

        Args:
            column: Column containing ISBNs
            allow_isbn10: Whether to accept ISBN-10
            allow_isbn13: Whether to accept ISBN-13
            require_format: Require specific format ("isbn10" or "isbn13")
            **kwargs: Additional config
        """
        super().__init__(column=column, **kwargs)
        self.allow_isbn10 = allow_isbn10 if require_format is None else require_format == "isbn10"
        self.allow_isbn13 = allow_isbn13 if require_format is None else require_format == "isbn13"

    def _validate_isbn10(self, value: str) -> bool:
        """Validate ISBN-10.

        Args:
            value: ISBN-10 string (digits only, may end with X)

        Returns:
            True if valid
        """
        if len(value) != 10:
            return False

        # Last digit can be X (representing 10)
        try:
            total = 0
            for i, char in enumerate(value):
                if char == "X" or char == "x":
                    if i != 9:
                        return False
                    digit = 10
                else:
                    digit = int(char)
                total += digit * (10 - i)

            return total % 11 == 0
        except ValueError:
            return False

    def _validate_isbn13(self, value: str) -> bool:
        """Validate ISBN-13.

        Args:
            value: ISBN-13 string (digits only)

        Returns:
            True if valid
        """
        if len(value) != 13 or not value.isdigit():
            return False

        # Must start with 978 or 979
        if not (value.startswith("978") or value.startswith("979")):
            return False

        # EAN-13 checksum
        total = 0
        for i, char in enumerate(value):
            digit = int(char)
            if i % 2 == 0:
                total += digit
            else:
                total += digit * 3

        return total % 10 == 0

    def validate_value(self, value: str) -> bool:
        """Validate an ISBN value.

        Args:
            value: ISBN to validate

        Returns:
            True if valid
        """
        # Remove separators
        cleaned = self._remove_separators(value)

        # Try ISBN-13 first (more common now)
        if self.allow_isbn13 and len(cleaned) == 13:
            return self._validate_isbn13(cleaned)

        # Try ISBN-10
        if self.allow_isbn10 and len(cleaned) == 10:
            return self._validate_isbn10(cleaned)

        return False

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate column for valid ISBNs.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        df = lf.select(pl.col(self.column)).collect()

        if len(df) == 0:
            return []

        invalid_mask = self._get_invalid_mask(df)
        invalid_count = int(invalid_mask.sum())

        if invalid_count == 0:
            return []

        total_count = len(df)
        invalid_ratio = invalid_count / total_count

        # Get sample invalid values
        invalid_df = df.filter(invalid_mask)
        samples = invalid_df[self.column].head(5).to_list()

        formats = []
        if self.allow_isbn10:
            formats.append("ISBN-10")
        if self.allow_isbn13:
            formats.append("ISBN-13")

        return [
            ValidationIssue(
                column=self.column,
                issue_type="invalid_isbn",
                count=invalid_count,
                severity=self._calculate_severity(invalid_ratio),
                details=(
                    f"Found {invalid_count} invalid ISBNs ({invalid_ratio:.2%}). "
                    f"Accepted formats: {', '.join(formats)}. "
                    f"Samples: {samples}"
                ),
                expected=f"Valid {' or '.join(formats)}",
            )
        ]

    def _calculate_severity(self, ratio: float) -> Severity:
        """Calculate severity based on invalid ratio."""
        if ratio < 0.01:
            return Severity.LOW
        elif ratio < 0.05:
            return Severity.MEDIUM
        elif ratio < 0.1:
            return Severity.HIGH
        else:
            return Severity.CRITICAL


@register_validator
class CreditCardValidator(ChecksumValidator):
    """Validates credit card numbers with brand detection.

    Validates credit cards using Luhn algorithm and also
    validates card brand based on prefix and length.

    Supported brands:
    - Visa: 4xxx (13 or 16 digits)
    - MasterCard: 51-55, 2221-2720 (16 digits)
    - American Express: 34, 37 (15 digits)
    - Discover: 6011, 65, 644-649 (16 digits)
    - JCB: 3528-3589 (16 digits)

    Example:
        validator = CreditCardValidator(
            column="card_number",
            allowed_brands=["visa", "mastercard"],
        )
    """

    name = "credit_card"

    # Card brand patterns: (prefixes, lengths)
    CARD_PATTERNS: dict[str, tuple[list[str], list[int]]] = {
        "visa": (["4"], [13, 16, 19]),
        "mastercard": (
            ["51", "52", "53", "54", "55"] +
            [str(i) for i in range(2221, 2721)],
            [16]
        ),
        "amex": (["34", "37"], [15]),
        "discover": (
            ["6011", "65"] + [f"64{i}" for i in range(4, 10)],
            [16]
        ),
        "jcb": ([str(i) for i in range(3528, 3590)], [16]),
        "diners": (["36", "38", "39"] + [f"30{i}" for i in range(6)], [14]),
    }

    def __init__(
        self,
        column: str,
        allowed_brands: list[str] | None = None,
        validate_brand: bool = True,
        **kwargs: Any,
    ):
        """Initialize credit card validator.

        Args:
            column: Column containing card numbers
            allowed_brands: List of allowed brands (None = all)
            validate_brand: Whether to validate brand prefix
            **kwargs: Additional config
        """
        super().__init__(column=column, min_length=13, max_length=19, **kwargs)
        self.allowed_brands = (
            [b.lower() for b in allowed_brands]
            if allowed_brands
            else None
        )
        self.validate_brand = validate_brand

    def _detect_brand(self, number: str) -> str | None:
        """Detect card brand from number.

        Args:
            number: Card number (digits only)

        Returns:
            Brand name or None if not detected
        """
        for brand, (prefixes, lengths) in self.CARD_PATTERNS.items():
            if len(number) in lengths:
                for prefix in prefixes:
                    if number.startswith(prefix):
                        return brand
        return None

    def validate_value(self, value: str) -> bool:
        """Validate a credit card number.

        Args:
            value: Card number to validate

        Returns:
            True if valid
        """
        # Remove separators
        cleaned = self._remove_separators(value)

        # Check if all digits
        if not cleaned.isdigit():
            return False

        # Check length
        if len(cleaned) < 13 or len(cleaned) > 19:
            return False

        # Validate brand if required
        if self.validate_brand:
            brand = self._detect_brand(cleaned)
            if brand is None:
                return False
            if self.allowed_brands and brand not in self.allowed_brands:
                return False

        # Luhn checksum
        digits = [int(d) for d in cleaned]
        return self._luhn_checksum(digits) == 0

    def _luhn_checksum(self, digits: list[int]) -> int:
        """Calculate Luhn checksum."""
        total = 0
        for i, digit in enumerate(reversed(digits)):
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit
        return total % 10

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate column for valid credit card numbers.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        df = lf.select(pl.col(self.column)).collect()

        if len(df) == 0:
            return []

        invalid_mask = self._get_invalid_mask(df)
        invalid_count = int(invalid_mask.sum())

        if invalid_count == 0:
            return []

        total_count = len(df)
        invalid_ratio = invalid_count / total_count

        # Get sample invalid values (masked for security)
        invalid_df = df.filter(invalid_mask)
        raw_samples = invalid_df[self.column].head(5).to_list()
        samples = [
            f"{str(s)[:4]}****{str(s)[-4:]}" if s and len(str(s)) > 8 else "****"
            for s in raw_samples
        ]

        brand_info = ""
        if self.allowed_brands:
            brand_info = f" Allowed brands: {self.allowed_brands}."

        return [
            ValidationIssue(
                column=self.column,
                issue_type="invalid_credit_card",
                count=invalid_count,
                severity=Severity.HIGH,  # Credit card issues are serious
                details=(
                    f"Found {invalid_count} invalid credit card numbers "
                    f"({invalid_ratio:.2%}).{brand_info} "
                    f"Sample patterns: {samples}"
                ),
                expected="Valid credit card number with Luhn checksum",
            )
        ]
