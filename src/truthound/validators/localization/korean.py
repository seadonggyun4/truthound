"""Korean localization validators.

This module provides validators for Korean-specific formats:
- Business registration numbers (사업자등록번호)
- Resident registration numbers (주민등록번호)
- Phone numbers (전화번호)
- Bank account numbers (계좌번호)
"""

import re
from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.registry import register_validator
from truthound.validators.localization.base import LocalizationValidator


@register_validator
class KoreanBusinessNumberValidator(LocalizationValidator):
    """Validates Korean business registration numbers (사업자등록번호).

    Format: XXX-XX-XXXXX (10 digits)
    Uses weighted checksum algorithm.

    Example:
        validator = KoreanBusinessNumberValidator(column="business_number")
    """

    name = "korean_business_number"

    # Checksum weights for validation
    WEIGHTS = [1, 3, 7, 1, 3, 7, 1, 3, 5]

    def validate_value(self, value: str) -> bool:
        """Validate a Korean business registration number.

        Args:
            value: Business number (digits only)

        Returns:
            True if valid, False otherwise
        """
        # Must be exactly 10 digits
        if not value.isdigit() or len(value) != 10:
            return False

        digits = [int(d) for d in value]

        # Calculate checksum
        total = sum(d * w for d, w in zip(digits[:9], self.WEIGHTS))
        total += (digits[8] * 5) // 10

        check_digit = (10 - (total % 10)) % 10

        return check_digit == digits[9]

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
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="invalid_korean_business_number",
                    count=invalid_count,
                    severity=Severity.MEDIUM,
                    details=(
                        f"Found {invalid_count} invalid Korean business numbers. "
                        f"Sample: {sample_invalid}"
                    ),
                    expected="Valid Korean business registration number (XXX-XX-XXXXX)",
                )
            )

        return issues


@register_validator
class KoreanRRNValidator(LocalizationValidator):
    """Validates Korean resident registration numbers (주민등록번호).

    Format: YYMMDD-GXXXXXX (13 digits)
    - YYMMDD: Birth date
    - G: Gender/century digit (1-4 for Koreans, 5-8 for foreigners)
    - XXXXXX: Serial + check digit

    Example:
        validator = KoreanRRNValidator(column="rrn")
    """

    name = "korean_rrn"

    # Checksum weights
    WEIGHTS = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]

    def __init__(
        self,
        column: str,
        validate_date: bool = True,
        mask_output: bool = True,
        **kwargs: Any,
    ):
        """Initialize Korean RRN validator.

        Args:
            column: Column to validate
            validate_date: Whether to validate birthdate portion
            mask_output: Whether to mask RRN in error output
            **kwargs: Additional config
        """
        super().__init__(column=column, **kwargs)
        self.validate_date = validate_date
        self.mask_output = mask_output

    def _validate_date(self, rrn: str) -> bool:
        """Validate the date portion of RRN.

        Args:
            rrn: 13-digit RRN

        Returns:
            True if date is valid
        """
        year = int(rrn[0:2])
        month = int(rrn[2:4])
        day = int(rrn[4:6])
        gender = int(rrn[6])

        # Determine century based on gender digit
        if gender in [1, 2, 5, 6]:
            year += 1900
        elif gender in [3, 4, 7, 8]:
            year += 2000
        elif gender in [9, 0]:
            year += 1800
        else:
            return False

        # Validate month
        if month < 1 or month > 12:
            return False

        # Validate day (simplified - doesn't check exact days per month)
        if day < 1 or day > 31:
            return False

        return True

    def validate_value(self, value: str) -> bool:
        """Validate a Korean RRN.

        Args:
            value: RRN (digits only)

        Returns:
            True if valid, False otherwise
        """
        # Must be exactly 13 digits
        if not value.isdigit() or len(value) != 13:
            return False

        # Validate date portion
        if self.validate_date and not self._validate_date(value):
            return False

        # Validate gender digit (1-8 are valid)
        gender = int(value[6])
        if gender < 1 or gender > 8:
            return False

        # Calculate checksum
        digits = [int(d) for d in value]
        total = sum(d * w for d, w in zip(digits[:12], self.WEIGHTS))
        check_digit = (11 - (total % 11)) % 10

        return check_digit == digits[12]

    def _mask_rrn(self, rrn: str) -> str:
        """Mask an RRN for privacy.

        Args:
            rrn: RRN to mask

        Returns:
            Masked RRN
        """
        if len(rrn) >= 7:
            return rrn[:6] + "-" + "*" * (len(rrn) - 6)
        return "*" * len(rrn)

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
                sample_details = f"Found {invalid_count} invalid RRNs (masked for privacy)"
            else:
                sample_invalid = df.filter(invalid_mask)[self.column].head(5).to_list()
                sample_details = f"Sample: {sample_invalid}"

            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="invalid_korean_rrn",
                    count=invalid_count,
                    severity=Severity.HIGH,
                    details=f"Found {invalid_count} invalid Korean RRNs. {sample_details}",
                    expected="Valid Korean resident registration number (YYMMDD-GXXXXXX)",
                )
            )

        return issues


@register_validator
class KoreanPhoneValidator(LocalizationValidator):
    """Validates Korean phone numbers.

    Supported formats:
    - Mobile: 010-XXXX-XXXX, 011/016/017/018/019-XXX-XXXX
    - Landline: 02-XXX(X)-XXXX, 0XX-XXX(X)-XXXX
    - Toll-free: 080-XXX-XXXX, 1588-XXXX, 1577-XXXX

    Example:
        validator = KoreanPhoneValidator(column="phone")
    """

    name = "korean_phone"

    # Valid patterns
    MOBILE_PATTERN = re.compile(r"^01[016789]\d{7,8}$")
    LANDLINE_PATTERN = re.compile(r"^0[2-6]\d{7,9}$")
    SPECIAL_PATTERN = re.compile(r"^(080\d{7}|1[0-9]{3}\d{4})$")

    def __init__(
        self,
        column: str,
        allow_mobile: bool = True,
        allow_landline: bool = True,
        allow_special: bool = True,
        **kwargs: Any,
    ):
        """Initialize Korean phone validator.

        Args:
            column: Column to validate
            allow_mobile: Allow mobile numbers
            allow_landline: Allow landline numbers
            allow_special: Allow special numbers (toll-free, etc.)
            **kwargs: Additional config
        """
        super().__init__(column=column, **kwargs)
        self.allow_mobile = allow_mobile
        self.allow_landline = allow_landline
        self.allow_special = allow_special

    def validate_value(self, value: str) -> bool:
        """Validate a Korean phone number.

        Args:
            value: Phone number (digits only)

        Returns:
            True if valid, False otherwise
        """
        if not value.isdigit():
            return False

        if self.allow_mobile and self.MOBILE_PATTERN.match(value):
            return True

        if self.allow_landline and self.LANDLINE_PATTERN.match(value):
            return True

        if self.allow_special and self.SPECIAL_PATTERN.match(value):
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
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="invalid_korean_phone",
                    count=invalid_count,
                    severity=Severity.LOW,
                    details=(
                        f"Found {invalid_count} invalid Korean phone numbers. "
                        f"Sample: {sample_invalid}"
                    ),
                    expected="Valid Korean phone number (mobile, landline, or special)",
                )
            )

        return issues


@register_validator
class KoreanBankAccountValidator(LocalizationValidator):
    """Validates Korean bank account numbers.

    Validates account numbers by bank with correct length ranges.

    Banks supported:
    - KB국민은행 (KB): 12-14 digits
    - 신한은행 (SHINHAN): 11-14 digits
    - 우리은행 (WOORI): 13 digits
    - 하나은행 (HANA): 14 digits
    - NH농협 (NH): 11-16 digits
    - IBK기업은행 (IBK): 11-14 digits
    - SC제일은행 (SC): 11 digits
    - 카카오뱅크 (KAKAO): 13 digits
    - 케이뱅크 (KBANK): 13 digits
    - 토스뱅크 (TOSS): 12-13 digits

    Example:
        validator = KoreanBankAccountValidator(column="account_number")
        validator = KoreanBankAccountValidator(
            column="account_number",
            bank_column="bank_code"
        )
    """

    name = "korean_bank_account"

    # Bank account length ranges
    BANK_LENGTHS: dict[str, tuple[int, int]] = {
        "KB": (12, 14),
        "SHINHAN": (11, 14),
        "WOORI": (13, 13),
        "HANA": (14, 14),
        "NH": (11, 16),
        "IBK": (11, 14),
        "SC": (11, 11),
        "KAKAO": (13, 13),
        "KBANK": (13, 13),
        "TOSS": (12, 13),
    }

    # Valid length range across all banks
    MIN_LENGTH = 10
    MAX_LENGTH = 16

    def __init__(
        self,
        column: str,
        bank_column: str | None = None,
        bank_code: str | None = None,
        **kwargs: Any,
    ):
        """Initialize Korean bank account validator.

        Args:
            column: Column containing account numbers
            bank_column: Optional column containing bank codes
            bank_code: Optional fixed bank code (if all accounts are from same bank)
            **kwargs: Additional config
        """
        super().__init__(column=column, **kwargs)
        self.bank_column = bank_column
        self.bank_code = bank_code.upper() if bank_code else None

    def validate_value(self, value: str) -> bool:
        """Validate a Korean bank account number (length only).

        Args:
            value: Account number (digits only)

        Returns:
            True if valid, False otherwise
        """
        if not value.isdigit():
            return False

        length = len(value)

        # If specific bank is set, use that bank's length
        if self.bank_code and self.bank_code in self.BANK_LENGTHS:
            min_len, max_len = self.BANK_LENGTHS[self.bank_code]
            return min_len <= length <= max_len

        # Otherwise, check if length is valid for any bank
        return self.MIN_LENGTH <= length <= self.MAX_LENGTH

    def _validate_with_bank(self, account: str, bank: str) -> bool:
        """Validate account with specific bank code.

        Args:
            account: Account number
            bank: Bank code

        Returns:
            True if valid
        """
        if not account.isdigit():
            return False

        bank_upper = bank.upper() if bank else None
        if bank_upper and bank_upper in self.BANK_LENGTHS:
            min_len, max_len = self.BANK_LENGTHS[bank_upper]
            return min_len <= len(account) <= max_len

        return self.MIN_LENGTH <= len(account) <= self.MAX_LENGTH

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

        issues: list[ValidationIssue] = []

        if self.bank_column and self.bank_column in df.columns:
            # Validate with per-row bank codes
            invalid_count = 0
            for row in df.iter_rows(named=True):
                account = row[self.column]
                bank = row[self.bank_column]

                if account is None:
                    if not self.allow_null:
                        invalid_count += 1
                    continue

                processed = self._preprocess_value(str(account))
                if not processed or not self._validate_with_bank(processed, bank):
                    invalid_count += 1

            if invalid_count > 0:
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="invalid_korean_bank_account",
                        count=invalid_count,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Found {invalid_count} invalid Korean bank accounts "
                            f"(validated against bank-specific formats)"
                        ),
                        expected="Valid Korean bank account number for specified bank",
                    )
                )
        else:
            # Use standard validation
            invalid_mask = self._get_invalid_mask(df)
            invalid_count = invalid_mask.sum() or 0

            if invalid_count > 0:
                sample_invalid = df.filter(invalid_mask)[self.column].head(5).to_list()
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="invalid_korean_bank_account",
                        count=invalid_count,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Found {invalid_count} invalid Korean bank accounts. "
                            f"Sample: {sample_invalid}"
                        ),
                        expected=f"Valid Korean bank account number ({self.MIN_LENGTH}-{self.MAX_LENGTH} digits)",
                    )
                )

        return issues
