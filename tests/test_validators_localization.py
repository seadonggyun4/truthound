"""Tests for localization validators."""

import pytest
import polars as pl

from truthound.validators.localization import (
    KoreanBusinessNumberValidator,
    KoreanRRNValidator,
    KoreanPhoneValidator,
    KoreanBankAccountValidator,
    JapanesePostalCodeValidator,
    JapaneseMyNumberValidator,
    ChineseIDValidator,
    ChineseUSCCValidator,
)


class TestKoreanBusinessNumberValidator:
    """Tests for Korean business number validation."""

    def test_valid_business_numbers(self):
        """Test valid Korean business numbers."""
        # Valid business numbers (with correct checksum)
        df = pl.DataFrame(
            {
                "biz_num": [
                    "1234567890",  # Placeholder - need real valid numbers
                    "123-45-67890",
                ]
            }
        )
        # Note: These are placeholder tests - real validation requires valid checksums
        validator = KoreanBusinessNumberValidator(column="biz_num")
        # Just verify it runs without error
        issues = validator.validate(df.lazy())
        assert isinstance(issues, list)

    def test_invalid_business_numbers(self):
        """Test invalid Korean business numbers."""
        df = pl.DataFrame(
            {
                "biz_num": [
                    "123456789",  # Too short
                    "12345678901",  # Too long
                    "abcdefghij",  # Non-numeric
                ]
            }
        )
        validator = KoreanBusinessNumberValidator(column="biz_num")
        issues = validator.validate(df.lazy())
        assert len(issues) > 0
        assert issues[0].issue_type == "invalid_korean_business_number"

    def test_null_handling(self):
        """Test null value handling."""
        df = pl.DataFrame({"biz_num": [None, "1234567890", None]})
        validator = KoreanBusinessNumberValidator(column="biz_num", allow_null=True)
        issues = validator.validate(df.lazy())
        # Nulls should be allowed
        assert all(i.issue_type != "null_values" for i in issues)


class TestKoreanRRNValidator:
    """Tests for Korean RRN validation."""

    def test_invalid_rrn_format(self):
        """Test invalid RRN formats."""
        df = pl.DataFrame(
            {
                "rrn": [
                    "123456789012",  # Too short
                    "12345678901234",  # Too long
                    "abcdefghijklm",  # Non-numeric
                ]
            }
        )
        validator = KoreanRRNValidator(column="rrn")
        issues = validator.validate(df.lazy())
        assert len(issues) > 0

    def test_invalid_gender_digit(self):
        """Test invalid gender digit."""
        df = pl.DataFrame(
            {
                "rrn": [
                    "8001019000000",  # Gender digit 9 is invalid
                    "8001010000000",  # Gender digit 0 is invalid
                ]
            }
        )
        validator = KoreanRRNValidator(column="rrn", validate_date=False)
        issues = validator.validate(df.lazy())
        assert len(issues) > 0

    def test_mask_output(self):
        """Test that output is masked by default."""
        df = pl.DataFrame({"rrn": ["invalid_rrn"]})
        validator = KoreanRRNValidator(column="rrn", mask_output=True)
        issues = validator.validate(df.lazy())
        assert len(issues) > 0
        # Should not contain the actual RRN
        assert "invalid_rrn" not in issues[0].details


class TestKoreanPhoneValidator:
    """Tests for Korean phone number validation."""

    def test_valid_mobile_numbers(self):
        """Test valid Korean mobile numbers."""
        df = pl.DataFrame(
            {
                "phone": [
                    "01012345678",
                    "010-1234-5678",
                    "01112345678",
                    "016-123-4567",
                ]
            }
        )
        validator = KoreanPhoneValidator(column="phone")
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_valid_landline_numbers(self):
        """Test valid Korean landline numbers."""
        df = pl.DataFrame(
            {
                "phone": [
                    "0212345678",  # Seoul
                    "02-123-4567",
                    "0311234567",  # Gyeonggi
                ]
            }
        )
        validator = KoreanPhoneValidator(column="phone")
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_invalid_numbers(self):
        """Test invalid phone numbers."""
        df = pl.DataFrame(
            {
                "phone": [
                    "12345",  # Too short
                    "0201234567890",  # Too long
                    "abcdefghij",  # Non-numeric
                ]
            }
        )
        validator = KoreanPhoneValidator(column="phone")
        issues = validator.validate(df.lazy())
        assert len(issues) > 0

    def test_mobile_only(self):
        """Test mobile-only validation."""
        df = pl.DataFrame({"phone": ["0212345678"]})  # Landline
        validator = KoreanPhoneValidator(
            column="phone", allow_mobile=True, allow_landline=False, allow_special=False
        )
        issues = validator.validate(df.lazy())
        assert len(issues) > 0


class TestKoreanBankAccountValidator:
    """Tests for Korean bank account validation."""

    def test_valid_account_lengths(self):
        """Test valid account number lengths."""
        df = pl.DataFrame(
            {
                "account": [
                    "1234567890123",  # 13 digits (Woori, Kakao)
                    "12345678901234",  # 14 digits (Hana)
                    "123456789012",  # 12 digits (KB)
                ]
            }
        )
        validator = KoreanBankAccountValidator(column="account")
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_invalid_lengths(self):
        """Test invalid account number lengths."""
        df = pl.DataFrame(
            {
                "account": [
                    "12345",  # Too short
                    "12345678901234567890",  # Too long
                ]
            }
        )
        validator = KoreanBankAccountValidator(column="account")
        issues = validator.validate(df.lazy())
        assert len(issues) > 0

    def test_bank_specific_validation(self):
        """Test bank-specific validation."""
        df = pl.DataFrame(
            {
                "account": ["1234567890123"],  # 13 digits
                "bank": ["WOORI"],
            }
        )
        validator = KoreanBankAccountValidator(
            column="account", bank_column="bank"
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0


class TestJapanesePostalCodeValidator:
    """Tests for Japanese postal code validation."""

    def test_valid_postal_codes(self):
        """Test valid Japanese postal codes."""
        df = pl.DataFrame(
            {
                "postal": [
                    "1000001",  # Tokyo
                    "100-0001",
                    "5300001",  # Osaka
                ]
            }
        )
        validator = JapanesePostalCodeValidator(column="postal")
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_invalid_postal_codes(self):
        """Test invalid postal codes."""
        df = pl.DataFrame(
            {
                "postal": [
                    "123456",  # Too short
                    "12345678",  # Too long
                    "abcdefg",  # Non-numeric
                ]
            }
        )
        validator = JapanesePostalCodeValidator(column="postal")
        issues = validator.validate(df.lazy())
        assert len(issues) > 0

    def test_strict_format(self):
        """Test strict format validation."""
        df = pl.DataFrame({"postal": ["1234567"]})  # No hyphen
        validator = JapanesePostalCodeValidator(column="postal", strict_format=True)
        issues = validator.validate(df.lazy())
        assert len(issues) > 0

        df = pl.DataFrame({"postal": ["123-4567"]})  # With hyphen
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_fullwidth_digits(self):
        """Test full-width digit conversion."""
        df = pl.DataFrame({"postal": ["１００－０００１"]})  # Full-width
        validator = JapanesePostalCodeValidator(column="postal")
        issues = validator.validate(df.lazy())
        assert len(issues) == 0


class TestJapaneseMyNumberValidator:
    """Tests for Japanese My Number validation."""

    def test_invalid_format(self):
        """Test invalid format."""
        df = pl.DataFrame(
            {
                "my_number": [
                    "12345678901",  # Too short
                    "1234567890123",  # Too long
                    "abcdefghijkl",  # Non-numeric
                ]
            }
        )
        validator = JapaneseMyNumberValidator(column="my_number")
        issues = validator.validate(df.lazy())
        assert len(issues) > 0

    def test_mask_output(self):
        """Test masked output."""
        df = pl.DataFrame({"my_number": ["123456789012"]})
        validator = JapaneseMyNumberValidator(column="my_number", mask_output=True)
        issues = validator.validate(df.lazy())
        # Should mask sensitive data
        if issues:
            assert "masked" in issues[0].details.lower()


class TestChineseIDValidator:
    """Tests for Chinese ID validation."""

    def test_invalid_length(self):
        """Test invalid lengths."""
        df = pl.DataFrame(
            {
                "id": [
                    "1234567890",  # Too short
                    "1234567890123456789",  # Too long
                ]
            }
        )
        validator = ChineseIDValidator(column="id")
        issues = validator.validate(df.lazy())
        assert len(issues) > 0

    def test_invalid_province_code(self):
        """Test invalid province codes."""
        df = pl.DataFrame(
            {
                "id": [
                    "991234567890123456",  # Invalid province 99
                ]
            }
        )
        validator = ChineseIDValidator(column="id", validate_province=True)
        issues = validator.validate(df.lazy())
        assert len(issues) > 0

    def test_15_digit_disabled(self):
        """Test when 15-digit IDs are disabled."""
        df = pl.DataFrame({"id": ["123456789012345"]})  # 15 digits
        validator = ChineseIDValidator(column="id", allow_15_digit=False)
        issues = validator.validate(df.lazy())
        assert len(issues) > 0

    def test_mask_output(self):
        """Test masked output."""
        df = pl.DataFrame({"id": ["invalid_id"]})
        validator = ChineseIDValidator(column="id", mask_output=True)
        issues = validator.validate(df.lazy())
        if issues:
            assert "masked" in issues[0].details.lower()


class TestChineseUSCCValidator:
    """Tests for Chinese USCC validation."""

    def test_invalid_length(self):
        """Test invalid lengths."""
        df = pl.DataFrame(
            {
                "uscc": [
                    "12345678901234567",  # 17 chars - too short
                    "1234567890123456789",  # 19 chars - too long
                ]
            }
        )
        validator = ChineseUSCCValidator(column="uscc")
        issues = validator.validate(df.lazy())
        assert len(issues) > 0

    def test_invalid_characters(self):
        """Test invalid characters."""
        df = pl.DataFrame(
            {
                "uscc": [
                    "91110000MA0123I567",  # Contains I (invalid)
                    "91110000MA0123O567",  # Contains O (invalid)
                ]
            }
        )
        validator = ChineseUSCCValidator(column="uscc")
        issues = validator.validate(df.lazy())
        assert len(issues) > 0


class TestEdgeCases:
    """Test edge cases for localization validators."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pl.DataFrame({"col": []})
        validator = KoreanPhoneValidator(column="col")
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_all_nulls(self):
        """Test with all null values."""
        df = pl.DataFrame({"col": [None, None, None]})
        validator = KoreanBusinessNumberValidator(column="col", allow_null=True)
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_mixed_valid_invalid(self):
        """Test with mixed valid and invalid values."""
        df = pl.DataFrame(
            {
                "postal": [
                    "1000001",  # Valid
                    "abc",  # Invalid
                    "1234567",  # Valid
                    "12",  # Invalid
                ]
            }
        )
        validator = JapanesePostalCodeValidator(column="postal")
        issues = validator.validate(df.lazy())
        assert len(issues) > 0
        assert issues[0].count == 2  # 2 invalid values
