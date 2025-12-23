"""Localization validators for Asian regions.

This module provides validators for region-specific formats in Asia:

- **Korean Validators**: Business numbers, RRN, phone, bank accounts
- **Japanese Validators**: Postal codes, My Number
- **Chinese Validators**: National ID, USCC

Validators:
    KoreanBusinessNumberValidator: Korean business registration number (사업자등록번호)
    KoreanRRNValidator: Korean resident registration number (주민등록번호)
    KoreanPhoneValidator: Korean phone numbers (전화번호)
    KoreanBankAccountValidator: Korean bank account numbers (계좌번호)
    JapanesePostalCodeValidator: Japanese postal codes (郵便番号)
    JapaneseMyNumberValidator: Japanese My Number (マイナンバー)
    ChineseIDValidator: Chinese National ID (身份证号码)
    ChineseUSCCValidator: Chinese Unified Social Credit Code (统一社会信用代码)
"""

from truthound.validators.localization.base import LocalizationValidator

from truthound.validators.localization.korean import (
    KoreanBusinessNumberValidator,
    KoreanRRNValidator,
    KoreanPhoneValidator,
    KoreanBankAccountValidator,
)

from truthound.validators.localization.japanese import (
    JapanesePostalCodeValidator,
    JapaneseMyNumberValidator,
)

from truthound.validators.localization.chinese import (
    ChineseIDValidator,
    ChineseUSCCValidator,
)

__all__ = [
    # Base class
    "LocalizationValidator",
    # Korean validators
    "KoreanBusinessNumberValidator",
    "KoreanRRNValidator",
    "KoreanPhoneValidator",
    "KoreanBankAccountValidator",
    # Japanese validators
    "JapanesePostalCodeValidator",
    "JapaneseMyNumberValidator",
    # Chinese validators
    "ChineseIDValidator",
    "ChineseUSCCValidator",
]
