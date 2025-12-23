"""Financial identifier validators.

This module provides validators for financial identifiers
like IBAN, VAT numbers, and other regulatory codes.
"""

import re
from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.registry import register_validator
from truthound.validators.business_rule.base import BusinessRuleValidator


@register_validator
class IBANValidator(BusinessRuleValidator):
    """Validates IBAN (International Bank Account Number).

    IBAN validation includes:
    - Country code check (2 letters)
    - Check digits validation (mod 97)
    - Length validation per country
    - Format validation (alphanumeric)

    Example:
        validator = IBANValidator(
            column="bank_account",
            allowed_countries=["DE", "FR", "GB"],
        )
    """

    name = "iban"

    # IBAN lengths by country (ISO 3166-1 alpha-2)
    IBAN_LENGTHS: dict[str, int] = {
        "AL": 28, "AD": 24, "AT": 20, "AZ": 28, "BH": 22, "BY": 28,
        "BE": 16, "BA": 20, "BR": 29, "BG": 22, "CR": 22, "HR": 21,
        "CY": 28, "CZ": 24, "DK": 18, "DO": 28, "TL": 23, "EE": 20,
        "FO": 18, "FI": 18, "FR": 27, "GE": 22, "DE": 22, "GI": 23,
        "GR": 27, "GL": 18, "GT": 28, "HU": 28, "IS": 26, "IQ": 23,
        "IE": 22, "IL": 23, "IT": 27, "JO": 30, "KZ": 20, "XK": 20,
        "KW": 30, "LV": 21, "LB": 28, "LI": 21, "LT": 20, "LU": 20,
        "MK": 19, "MT": 31, "MR": 27, "MU": 30, "MC": 27, "MD": 24,
        "ME": 22, "NL": 18, "NO": 15, "PK": 24, "PS": 29, "PL": 28,
        "PT": 25, "QA": 29, "RO": 24, "SM": 27, "SA": 24, "RS": 22,
        "SC": 31, "SK": 24, "SI": 19, "ES": 24, "SE": 24, "CH": 21,
        "TN": 24, "TR": 26, "UA": 29, "AE": 23, "GB": 22, "VA": 22,
        "VG": 24,
    }

    def __init__(
        self,
        column: str,
        allowed_countries: list[str] | None = None,
        **kwargs: Any,
    ):
        """Initialize IBAN validator.

        Args:
            column: Column containing IBANs
            allowed_countries: List of allowed country codes (None = all)
            **kwargs: Additional config
        """
        super().__init__(column=column, **kwargs)
        self.allowed_countries = (
            [c.upper() for c in allowed_countries]
            if allowed_countries
            else None
        )

    def _mod97(self, iban_numeric: str) -> int:
        """Calculate IBAN mod 97.

        Args:
            iban_numeric: IBAN with letters converted to numbers

        Returns:
            Remainder when divided by 97
        """
        # Process in chunks to avoid integer overflow
        remainder = 0
        for i in range(0, len(iban_numeric), 7):
            chunk = str(remainder) + iban_numeric[i:i+7]
            remainder = int(chunk) % 97
        return remainder

    def validate_value(self, value: str) -> bool:
        """Validate an IBAN.

        Args:
            value: IBAN to validate

        Returns:
            True if valid
        """
        # Remove spaces and convert to uppercase
        iban = self._remove_separators(value.upper(), " -")

        # Check minimum length and alphanumeric
        if len(iban) < 15 or not iban.isalnum():
            return False

        # Extract country code
        country = iban[:2]
        if not country.isalpha():
            return False

        # Check country is known
        if country not in self.IBAN_LENGTHS:
            return False

        # Check length for country
        expected_length = self.IBAN_LENGTHS[country]
        if len(iban) != expected_length:
            return False

        # Check allowed countries
        if self.allowed_countries and country not in self.allowed_countries:
            return False

        # Rearrange: move first 4 chars to end
        rearranged = iban[4:] + iban[:4]

        # Convert letters to numbers (A=10, B=11, ..., Z=35)
        numeric = ""
        for char in rearranged:
            if char.isdigit():
                numeric += char
            else:
                numeric += str(ord(char) - ord("A") + 10)

        # Check mod 97
        return self._mod97(numeric) == 1

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate column for valid IBANs.

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

        # Get sample invalid values (masked)
        invalid_df = df.filter(invalid_mask)
        raw_samples = invalid_df[self.column].head(5).to_list()
        samples = [
            f"{str(s)[:4]}****{str(s)[-4:]}" if s and len(str(s)) > 8 else "****"
            for s in raw_samples
        ]

        country_info = ""
        if self.allowed_countries:
            country_info = f" Allowed countries: {self.allowed_countries}."

        return [
            ValidationIssue(
                column=self.column,
                issue_type="invalid_iban",
                count=invalid_count,
                severity=Severity.HIGH,
                details=(
                    f"Found {invalid_count} invalid IBANs ({invalid_ratio:.2%}).{country_info} "
                    f"Sample patterns: {samples}"
                ),
                expected="Valid IBAN with correct check digits",
            )
        ]


@register_validator
class VATValidator(BusinessRuleValidator):
    """Validates EU VAT (Value Added Tax) numbers.

    VAT numbers have country-specific formats and validation rules.
    This validator supports major EU countries.

    Supported countries:
    - DE (Germany): DE + 9 digits
    - FR (France): FR + 2 chars + 9 digits
    - GB (UK): GB + 9 or 12 digits
    - IT (Italy): IT + 11 digits
    - ES (Spain): ES + letter + 7 digits + letter/digit
    - NL (Netherlands): NL + 9 digits + B + 2 digits
    - BE (Belgium): BE + 10 digits
    - AT (Austria): ATU + 8 digits
    - PL (Poland): PL + 10 digits

    Example:
        validator = VATValidator(
            column="vat_number",
            allowed_countries=["DE", "FR", "IT"],
        )
    """

    name = "vat"

    # VAT patterns by country: (regex pattern, optional checksum function)
    VAT_PATTERNS: dict[str, tuple[str, Any]] = {
        "AT": (r"^ATU\d{8}$", None),
        "BE": (r"^BE[01]\d{9}$", None),
        "BG": (r"^BG\d{9,10}$", None),
        "CY": (r"^CY\d{8}[A-Z]$", None),
        "CZ": (r"^CZ\d{8,10}$", None),
        "DE": (r"^DE\d{9}$", None),
        "DK": (r"^DK\d{8}$", None),
        "EE": (r"^EE\d{9}$", None),
        "EL": (r"^EL\d{9}$", None),  # Greece
        "ES": (r"^ES[A-Z0-9]\d{7}[A-Z0-9]$", None),
        "FI": (r"^FI\d{8}$", None),
        "FR": (r"^FR[A-Z0-9]{2}\d{9}$", None),
        "GB": (r"^GB(\d{9}|\d{12}|(GD|HA)\d{3})$", None),
        "HR": (r"^HR\d{11}$", None),
        "HU": (r"^HU\d{8}$", None),
        "IE": (r"^IE(\d{7}[A-Z]{1,2}|\d[A-Z+*]\d{5}[A-Z])$", None),
        "IT": (r"^IT\d{11}$", None),
        "LT": (r"^LT(\d{9}|\d{12})$", None),
        "LU": (r"^LU\d{8}$", None),
        "LV": (r"^LV\d{11}$", None),
        "MT": (r"^MT\d{8}$", None),
        "NL": (r"^NL\d{9}B\d{2}$", None),
        "PL": (r"^PL\d{10}$", None),
        "PT": (r"^PT\d{9}$", None),
        "RO": (r"^RO\d{2,10}$", None),
        "SE": (r"^SE\d{12}$", None),
        "SI": (r"^SI\d{8}$", None),
        "SK": (r"^SK\d{10}$", None),
    }

    def __init__(
        self,
        column: str,
        allowed_countries: list[str] | None = None,
        strict_format: bool = True,
        **kwargs: Any,
    ):
        """Initialize VAT validator.

        Args:
            column: Column containing VAT numbers
            allowed_countries: List of allowed country codes (None = all)
            strict_format: Whether to require exact format matching
            **kwargs: Additional config
        """
        super().__init__(column=column, **kwargs)
        self.allowed_countries = (
            [c.upper() for c in allowed_countries]
            if allowed_countries
            else None
        )
        self.strict_format = strict_format

        # Compile regex patterns
        self._compiled_patterns = {
            country: re.compile(pattern)
            for country, (pattern, _) in self.VAT_PATTERNS.items()
        }

    def _extract_country(self, vat: str) -> str | None:
        """Extract country code from VAT number.

        Args:
            vat: VAT number

        Returns:
            Country code or None
        """
        if len(vat) < 2:
            return None

        # First two chars should be country code
        country = vat[:2].upper()

        # Greece uses EL instead of GR
        if country == "GR":
            country = "EL"

        return country if country in self.VAT_PATTERNS else None

    def validate_value(self, value: str) -> bool:
        """Validate a VAT number.

        Args:
            value: VAT number to validate

        Returns:
            True if valid
        """
        # Remove spaces and convert to uppercase
        vat = self._remove_separators(value.upper(), " -.")

        # Extract country
        country = self._extract_country(vat)
        if country is None:
            return False

        # Check allowed countries
        if self.allowed_countries and country not in self.allowed_countries:
            return False

        # Check pattern
        pattern = self._compiled_patterns.get(country)
        if pattern is None:
            return False

        return bool(pattern.match(vat))

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate column for valid VAT numbers.

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

        country_info = ""
        if self.allowed_countries:
            country_info = f" Allowed countries: {self.allowed_countries}."

        return [
            ValidationIssue(
                column=self.column,
                issue_type="invalid_vat_number",
                count=invalid_count,
                severity=Severity.MEDIUM,
                details=(
                    f"Found {invalid_count} invalid VAT numbers ({invalid_ratio:.2%}).{country_info} "
                    f"Samples: {samples}"
                ),
                expected="Valid EU VAT number format",
            )
        ]


@register_validator
class SWIFTValidator(BusinessRuleValidator):
    """Validates SWIFT/BIC codes.

    SWIFT (Society for Worldwide Interbank Financial Telecommunication)
    codes, also known as BIC (Bank Identifier Code), are used to
    identify banks globally.

    Format: AAAA BB CC DDD
    - AAAA: Bank code (4 letters)
    - BB: Country code (2 letters)
    - CC: Location code (2 alphanumeric)
    - DDD: Branch code (3 alphanumeric, optional)

    Example:
        validator = SWIFTValidator(
            column="swift_code",
            require_branch=False,
        )
    """

    name = "swift"

    # Valid country codes (ISO 3166-1 alpha-2) - subset
    VALID_COUNTRIES = {
        "AD", "AE", "AF", "AG", "AI", "AL", "AM", "AO", "AQ", "AR", "AS", "AT",
        "AU", "AW", "AX", "AZ", "BA", "BB", "BD", "BE", "BF", "BG", "BH", "BI",
        "BJ", "BL", "BM", "BN", "BO", "BQ", "BR", "BS", "BT", "BV", "BW", "BY",
        "BZ", "CA", "CC", "CD", "CF", "CG", "CH", "CI", "CK", "CL", "CM", "CN",
        "CO", "CR", "CU", "CV", "CW", "CX", "CY", "CZ", "DE", "DJ", "DK", "DM",
        "DO", "DZ", "EC", "EE", "EG", "EH", "ER", "ES", "ET", "FI", "FJ", "FK",
        "FM", "FO", "FR", "GA", "GB", "GD", "GE", "GF", "GG", "GH", "GI", "GL",
        "GM", "GN", "GP", "GQ", "GR", "GS", "GT", "GU", "GW", "GY", "HK", "HM",
        "HN", "HR", "HT", "HU", "ID", "IE", "IL", "IM", "IN", "IO", "IQ", "IR",
        "IS", "IT", "JE", "JM", "JO", "JP", "KE", "KG", "KH", "KI", "KM", "KN",
        "KP", "KR", "KW", "KY", "KZ", "LA", "LB", "LC", "LI", "LK", "LR", "LS",
        "LT", "LU", "LV", "LY", "MA", "MC", "MD", "ME", "MF", "MG", "MH", "MK",
        "ML", "MM", "MN", "MO", "MP", "MQ", "MR", "MS", "MT", "MU", "MV", "MW",
        "MX", "MY", "MZ", "NA", "NC", "NE", "NF", "NG", "NI", "NL", "NO", "NP",
        "NR", "NU", "NZ", "OM", "PA", "PE", "PF", "PG", "PH", "PK", "PL", "PM",
        "PN", "PR", "PS", "PT", "PW", "PY", "QA", "RE", "RO", "RS", "RU", "RW",
        "SA", "SB", "SC", "SD", "SE", "SG", "SH", "SI", "SJ", "SK", "SL", "SM",
        "SN", "SO", "SR", "SS", "ST", "SV", "SX", "SY", "SZ", "TC", "TD", "TF",
        "TG", "TH", "TJ", "TK", "TL", "TM", "TN", "TO", "TR", "TT", "TV", "TW",
        "TZ", "UA", "UG", "UM", "US", "UY", "UZ", "VA", "VC", "VE", "VG", "VI",
        "VN", "VU", "WF", "WS", "XK", "YE", "YT", "ZA", "ZM", "ZW",
    }

    def __init__(
        self,
        column: str,
        require_branch: bool = False,
        allowed_countries: list[str] | None = None,
        **kwargs: Any,
    ):
        """Initialize SWIFT validator.

        Args:
            column: Column containing SWIFT codes
            require_branch: Whether to require branch code
            allowed_countries: List of allowed country codes
            **kwargs: Additional config
        """
        super().__init__(column=column, **kwargs)
        self.require_branch = require_branch
        self.allowed_countries = (
            {c.upper() for c in allowed_countries}
            if allowed_countries
            else None
        )

    def validate_value(self, value: str) -> bool:
        """Validate a SWIFT code.

        Args:
            value: SWIFT code to validate

        Returns:
            True if valid
        """
        # Remove spaces and convert to uppercase
        swift = self._remove_separators(value.upper(), " -")

        # Check length (8 or 11 characters)
        if len(swift) not in (8, 11):
            return False

        # Check if branch is required
        if self.require_branch and len(swift) != 11:
            return False

        # Bank code (4 letters)
        bank_code = swift[:4]
        if not bank_code.isalpha():
            return False

        # Country code (2 letters)
        country = swift[4:6]
        if not country.isalpha():
            return False
        if country not in self.VALID_COUNTRIES:
            return False
        if self.allowed_countries and country not in self.allowed_countries:
            return False

        # Location code (2 alphanumeric)
        location = swift[6:8]
        if not location.isalnum():
            return False

        # Branch code if present (3 alphanumeric)
        if len(swift) == 11:
            branch = swift[8:11]
            if not branch.isalnum():
                return False

        return True

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate column for valid SWIFT codes.

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

        samples = df.filter(invalid_mask)[self.column].head(5).to_list()

        return [
            ValidationIssue(
                column=self.column,
                issue_type="invalid_swift_code",
                count=invalid_count,
                severity=Severity.MEDIUM,
                details=(
                    f"Found {invalid_count} invalid SWIFT/BIC codes ({invalid_ratio:.2%}). "
                    f"Samples: {samples}"
                ),
                expected="Valid SWIFT/BIC code (8 or 11 characters)",
            )
        ]
