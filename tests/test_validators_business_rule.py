"""Tests for business rule validators."""

import polars as pl
import pytest

from truthound.validators.business_rule import (
    LuhnValidator,
    ISBNValidator,
    CreditCardValidator,
    IBANValidator,
    VATValidator,
    SWIFTValidator,
)


class TestLuhnValidator:
    """Tests for LuhnValidator."""

    @pytest.fixture
    def valid_numbers(self):
        """Create data with valid Luhn numbers."""
        return pl.LazyFrame({
            "card": [
                "79927398713",  # Standard test number
                "4532015112830366",  # Visa-like
                "6011514433546201",  # Discover-like
            ]
        })

    @pytest.fixture
    def invalid_numbers(self):
        """Create data with invalid Luhn numbers."""
        return pl.LazyFrame({
            "card": [
                "79927398710",  # Invalid checksum
                "1234567890123",  # Random
                "abc123",  # Non-numeric
            ]
        })

    def test_valid_luhn_numbers(self, valid_numbers):
        """Test validation of valid Luhn numbers."""
        validator = LuhnValidator(column="card")
        issues = validator.validate(valid_numbers)
        assert len(issues) == 0

    def test_invalid_luhn_numbers(self, invalid_numbers):
        """Test detection of invalid Luhn numbers."""
        validator = LuhnValidator(column="card")
        issues = validator.validate(invalid_numbers)
        assert len(issues) == 1
        assert issues[0].issue_type == "invalid_luhn_checksum"
        assert issues[0].count == 3

    def test_length_constraints(self):
        """Test minimum/maximum length constraints."""
        data = pl.LazyFrame({"card": ["123"]})  # Too short
        validator = LuhnValidator(column="card", min_length=10)
        issues = validator.validate(data)
        assert len(issues) == 1

    def test_prefix_filter(self):
        """Test allowed prefix filtering."""
        data = pl.LazyFrame({
            "card": [
                "4532015112830366",  # Starts with 4
                "5532015112830366",  # Starts with 5
            ]
        })
        validator = LuhnValidator(column="card", allowed_prefixes=["4"])
        issues = validator.validate(data)
        assert len(issues) == 1  # One doesn't start with 4


class TestISBNValidator:
    """Tests for ISBNValidator."""

    @pytest.fixture
    def valid_isbns(self):
        """Create data with valid ISBNs."""
        return pl.LazyFrame({
            "isbn": [
                "978-0-306-40615-7",  # ISBN-13
                "0-306-40615-2",  # ISBN-10
                "9780306406157",  # ISBN-13 without hyphens
            ]
        })

    @pytest.fixture
    def invalid_isbns(self):
        """Create data with invalid ISBNs."""
        return pl.LazyFrame({
            "isbn": [
                "978-0-306-40615-8",  # Wrong check digit
                "1234567890",  # Invalid ISBN-10
                "abc",  # Not an ISBN
            ]
        })

    def test_valid_isbns(self, valid_isbns):
        """Test validation of valid ISBNs."""
        validator = ISBNValidator(column="isbn")
        issues = validator.validate(valid_isbns)
        assert len(issues) == 0

    def test_invalid_isbns(self, invalid_isbns):
        """Test detection of invalid ISBNs."""
        validator = ISBNValidator(column="isbn")
        issues = validator.validate(invalid_isbns)
        assert len(issues) == 1
        assert issues[0].issue_type == "invalid_isbn"

    def test_isbn10_only(self):
        """Test ISBN-10 only mode."""
        data = pl.LazyFrame({"isbn": ["9780306406157"]})  # ISBN-13
        validator = ISBNValidator(column="isbn", allow_isbn13=False)
        issues = validator.validate(data)
        assert len(issues) == 1

    def test_isbn13_only(self):
        """Test ISBN-13 only mode."""
        data = pl.LazyFrame({"isbn": ["0306406152"]})  # ISBN-10
        validator = ISBNValidator(column="isbn", allow_isbn10=False)
        issues = validator.validate(data)
        assert len(issues) == 1


class TestCreditCardValidator:
    """Tests for CreditCardValidator."""

    @pytest.fixture
    def valid_cards(self):
        """Create data with valid credit card numbers."""
        return pl.LazyFrame({
            "card": [
                "4532015112830366",  # Visa
                "5425233430109903",  # MasterCard
                "374245455400126",  # Amex
            ]
        })

    def test_valid_cards(self, valid_cards):
        """Test validation of valid credit cards."""
        validator = CreditCardValidator(column="card")
        issues = validator.validate(valid_cards)
        assert len(issues) == 0

    def test_brand_filtering(self, valid_cards):
        """Test brand-specific filtering."""
        validator = CreditCardValidator(
            column="card",
            allowed_brands=["visa"],
        )
        issues = validator.validate(valid_cards)
        assert len(issues) == 1  # 2 non-Visa cards

    def test_invalid_cards(self):
        """Test detection of invalid cards."""
        data = pl.LazyFrame({"card": ["1234567890123456"]})
        validator = CreditCardValidator(column="card")
        issues = validator.validate(data)
        assert len(issues) == 1


class TestIBANValidator:
    """Tests for IBANValidator."""

    @pytest.fixture
    def valid_ibans(self):
        """Create data with valid IBANs."""
        return pl.LazyFrame({
            "iban": [
                "DE89370400440532013000",  # Germany
                "GB82WEST12345698765432",  # UK
                "FR7630006000011234567890189",  # France
            ]
        })

    @pytest.fixture
    def invalid_ibans(self):
        """Create data with invalid IBANs."""
        return pl.LazyFrame({
            "iban": [
                "DE89370400440532013001",  # Invalid check digits
                "XX00123456789",  # Invalid country
                "not-an-iban",  # Invalid format
            ]
        })

    def test_valid_ibans(self, valid_ibans):
        """Test validation of valid IBANs."""
        validator = IBANValidator(column="iban")
        issues = validator.validate(valid_ibans)
        assert len(issues) == 0

    def test_invalid_ibans(self, invalid_ibans):
        """Test detection of invalid IBANs."""
        validator = IBANValidator(column="iban")
        issues = validator.validate(invalid_ibans)
        assert len(issues) == 1
        assert issues[0].issue_type == "invalid_iban"

    def test_country_filtering(self, valid_ibans):
        """Test country-specific filtering."""
        validator = IBANValidator(
            column="iban",
            allowed_countries=["DE"],
        )
        issues = validator.validate(valid_ibans)
        assert len(issues) == 1  # 2 non-DE IBANs


class TestVATValidator:
    """Tests for VATValidator."""

    @pytest.fixture
    def valid_vats(self):
        """Create data with valid VAT numbers."""
        return pl.LazyFrame({
            "vat": [
                "DE123456789",  # Germany
                "FR12345678901",  # France
                "GB123456789",  # UK
                "IT12345678901",  # Italy
            ]
        })

    @pytest.fixture
    def invalid_vats(self):
        """Create data with invalid VAT numbers."""
        return pl.LazyFrame({
            "vat": [
                "DE12345",  # Too short
                "XX123456789",  # Invalid country
                "not-a-vat",  # Invalid format
            ]
        })

    def test_valid_vats(self, valid_vats):
        """Test validation of valid VAT numbers."""
        validator = VATValidator(column="vat")
        issues = validator.validate(valid_vats)
        assert len(issues) == 0

    def test_invalid_vats(self, invalid_vats):
        """Test detection of invalid VAT numbers."""
        validator = VATValidator(column="vat")
        issues = validator.validate(invalid_vats)
        assert len(issues) == 1
        assert issues[0].issue_type == "invalid_vat_number"

    def test_country_filtering(self, valid_vats):
        """Test country-specific filtering."""
        validator = VATValidator(
            column="vat",
            allowed_countries=["DE", "FR"],
        )
        issues = validator.validate(valid_vats)
        assert len(issues) == 1  # GB and IT not allowed


class TestSWIFTValidator:
    """Tests for SWIFTValidator."""

    @pytest.fixture
    def valid_swifts(self):
        """Create data with valid SWIFT codes."""
        return pl.LazyFrame({
            "swift": [
                "DEUTDEFF",  # 8 chars
                "DEUTDEFF500",  # 11 chars with branch
                "BOFAUS3N",  # Bank of America
            ]
        })

    @pytest.fixture
    def invalid_swifts(self):
        """Create data with invalid SWIFT codes."""
        return pl.LazyFrame({
            "swift": [
                "DEUT",  # Too short
                "DEUTXXFF",  # Invalid country
                "12345678",  # All digits
            ]
        })

    def test_valid_swifts(self, valid_swifts):
        """Test validation of valid SWIFT codes."""
        validator = SWIFTValidator(column="swift")
        issues = validator.validate(valid_swifts)
        assert len(issues) == 0

    def test_invalid_swifts(self, invalid_swifts):
        """Test detection of invalid SWIFT codes."""
        validator = SWIFTValidator(column="swift")
        issues = validator.validate(invalid_swifts)
        assert len(issues) == 1
        assert issues[0].issue_type == "invalid_swift_code"

    def test_require_branch(self, valid_swifts):
        """Test branch requirement."""
        validator = SWIFTValidator(column="swift", require_branch=True)
        issues = validator.validate(valid_swifts)
        assert len(issues) == 1  # 2 without branch code


class TestNullHandling:
    """Tests for null value handling."""

    def test_null_values_allowed(self):
        """Test that null values are allowed by default."""
        data = pl.LazyFrame({"card": [None, "79927398713", None]})
        validator = LuhnValidator(column="card", allow_null=True)
        issues = validator.validate(data)
        assert len(issues) == 0

    def test_null_values_not_allowed(self):
        """Test that null values can be disallowed."""
        data = pl.LazyFrame({"card": [None, "79927398713"]})
        validator = LuhnValidator(column="card", allow_null=False)
        issues = validator.validate(data)
        assert len(issues) == 1
        assert issues[0].count == 1
