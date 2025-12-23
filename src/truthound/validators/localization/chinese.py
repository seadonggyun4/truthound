"""Chinese localization validators.

This module provides validators for Chinese-specific formats:
- National ID (身份证号码)
- Unified Social Credit Code (统一社会信用代码)
"""

import re
from datetime import datetime
from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.registry import register_validator
from truthound.validators.localization.base import LocalizationValidator


@register_validator
class ChineseIDValidator(LocalizationValidator):
    """Validates Chinese National ID numbers (身份证号码).

    Supports both:
    - First generation (15 digits): AAAAAA YYMMDD XXS
    - Second generation (18 digits): AAAAAA YYYYMMDD XXX C

    Components:
    - AAAAAA: Administrative division code (6 digits)
    - YYMMDD/YYYYMMDD: Birth date
    - XXX/XX: Sequential code (odd for male, even for female)
    - C: Check digit (only for 18-digit)

    Example:
        validator = ChineseIDValidator(column="id_number")
    """

    name = "chinese_id"

    # Check digit weights for 18-digit ID
    WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]

    # Check digit mapping
    CHECK_MAP = "10X98765432"

    # Valid province codes (first 2 digits)
    PROVINCE_CODES = {
        "11": "北京",
        "12": "天津",
        "13": "河北",
        "14": "山西",
        "15": "内蒙古",
        "21": "辽宁",
        "22": "吉林",
        "23": "黑龙江",
        "31": "上海",
        "32": "江苏",
        "33": "浙江",
        "34": "安徽",
        "35": "福建",
        "36": "江西",
        "37": "山东",
        "41": "河南",
        "42": "湖北",
        "43": "湖南",
        "44": "广东",
        "45": "广西",
        "46": "海南",
        "50": "重庆",
        "51": "四川",
        "52": "贵州",
        "53": "云南",
        "54": "西藏",
        "61": "陕西",
        "62": "甘肃",
        "63": "青海",
        "64": "宁夏",
        "65": "新疆",
        "71": "台湾",
        "81": "香港",
        "82": "澳门",
    }

    def __init__(
        self,
        column: str,
        allow_15_digit: bool = True,
        validate_date: bool = True,
        validate_province: bool = True,
        mask_output: bool = True,
        **kwargs: Any,
    ):
        """Initialize Chinese ID validator.

        Args:
            column: Column to validate
            allow_15_digit: Whether to allow first-generation 15-digit IDs
            validate_date: Whether to validate birthdate portion
            validate_province: Whether to validate province code
            mask_output: Whether to mask ID in error output
            **kwargs: Additional config
        """
        super().__init__(column=column, **kwargs)
        self.allow_15_digit = allow_15_digit
        self.validate_date = validate_date
        self.validate_province = validate_province
        self.mask_output = mask_output

    def _validate_date(self, year: int, month: int, day: int) -> bool:
        """Validate a birthdate.

        Args:
            year: Birth year
            month: Birth month
            day: Birth day

        Returns:
            True if date is valid
        """
        try:
            date = datetime(year, month, day)
            # Must be in the past and not before 1900
            return 1900 <= year <= datetime.now().year and date <= datetime.now()
        except ValueError:
            return False

    def _validate_15_digit(self, value: str) -> bool:
        """Validate a 15-digit ID (first generation).

        Args:
            value: 15-digit ID

        Returns:
            True if valid
        """
        if not value.isdigit():
            return False

        # Validate province code
        if self.validate_province:
            province = value[:2]
            if province not in self.PROVINCE_CODES:
                return False

        # Validate birthdate (format: YYMMDD, assume 1900s)
        if self.validate_date:
            try:
                year = 1900 + int(value[6:8])
                month = int(value[8:10])
                day = int(value[10:12])
                if not self._validate_date(year, month, day):
                    return False
            except (ValueError, IndexError):
                return False

        return True

    def _validate_18_digit(self, value: str) -> bool:
        """Validate an 18-digit ID (second generation).

        Args:
            value: 18-digit ID

        Returns:
            True if valid
        """
        # First 17 must be digits, last can be digit or X
        if not value[:17].isdigit():
            return False

        last_char = value[17].upper()
        if not (last_char.isdigit() or last_char == "X"):
            return False

        # Validate province code
        if self.validate_province:
            province = value[:2]
            if province not in self.PROVINCE_CODES:
                return False

        # Validate birthdate (format: YYYYMMDD)
        if self.validate_date:
            try:
                year = int(value[6:10])
                month = int(value[10:12])
                day = int(value[12:14])
                if not self._validate_date(year, month, day):
                    return False
            except (ValueError, IndexError):
                return False

        # Validate check digit
        digits = [int(d) for d in value[:17]]
        total = sum(d * w for d, w in zip(digits, self.WEIGHTS))
        expected_check = self.CHECK_MAP[total % 11]

        return last_char == expected_check

    def validate_value(self, value: str) -> bool:
        """Validate a Chinese National ID.

        Args:
            value: ID number

        Returns:
            True if valid, False otherwise
        """
        value = value.upper()

        if len(value) == 18:
            return self._validate_18_digit(value)
        elif len(value) == 15 and self.allow_15_digit:
            return self._validate_15_digit(value)
        else:
            return False

    def _mask_id(self, id_num: str) -> str:
        """Mask a Chinese ID for privacy.

        Args:
            id_num: ID to mask

        Returns:
            Masked ID
        """
        if len(id_num) >= 8:
            return id_num[:4] + "*" * (len(id_num) - 8) + id_num[-4:]
        return "*" * len(id_num)

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
                sample_details = f"Found {invalid_count} invalid IDs (masked for privacy)"
            else:
                sample_invalid = df.filter(invalid_mask)[self.column].head(5).to_list()
                sample_details = f"Sample: {sample_invalid}"

            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="invalid_chinese_id",
                    count=invalid_count,
                    severity=Severity.HIGH,
                    details=f"Found {invalid_count} invalid Chinese National IDs. {sample_details}",
                    expected="Valid Chinese National ID (15 or 18 digits)",
                )
            )

        return issues


@register_validator
class ChineseUSCCValidator(LocalizationValidator):
    """Validates Chinese Unified Social Credit Code (统一社会信用代码).

    Format: 18 characters
    - Position 1: Registration management department code
    - Position 2: Organization category code
    - Position 3-8: Administrative division code
    - Position 9-17: Organization code
    - Position 18: Check digit

    Uses modulo 31 check digit algorithm with base-31 encoding.

    Example:
        validator = ChineseUSCCValidator(column="credit_code")
    """

    name = "chinese_uscc"

    # Valid characters for the code (0-9, A-Z excluding I,O,S,V,Z)
    VALID_CHARS = "0123456789ABCDEFGHJKLMNPQRTUWXY"
    CHAR_VALUES = {c: i for i, c in enumerate(VALID_CHARS)}

    # Weights for check digit calculation
    WEIGHTS = [1, 3, 9, 27, 19, 26, 16, 17, 20, 29, 25, 13, 8, 24, 10, 30, 28]

    def validate_value(self, value: str) -> bool:
        """Validate a Chinese USCC.

        Args:
            value: USCC (18 characters)

        Returns:
            True if valid, False otherwise
        """
        value = value.upper()

        if len(value) != 18:
            return False

        # Check all characters are valid
        for char in value:
            if char not in self.VALID_CHARS:
                return False

        # Calculate check digit
        total = 0
        for i in range(17):
            char_value = self.CHAR_VALUES.get(value[i], -1)
            if char_value < 0:
                return False
            total += char_value * self.WEIGHTS[i]

        remainder = total % 31
        expected_check_value = (31 - remainder) % 31
        expected_check = self.VALID_CHARS[expected_check_value]

        return value[17] == expected_check

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
                    issue_type="invalid_chinese_uscc",
                    count=invalid_count,
                    severity=Severity.MEDIUM,
                    details=(
                        f"Found {invalid_count} invalid Chinese USCC codes. "
                        f"Sample: {sample_invalid}"
                    ),
                    expected="Valid Chinese Unified Social Credit Code (18 characters)",
                )
            )

        return issues
