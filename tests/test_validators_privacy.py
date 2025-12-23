"""Tests for privacy compliance validators.

Tests cover:
- GDPR compliance validators
- CCPA compliance validators
- Global privacy validators
- Data retention validators
- Consent validators
- Special category detection
"""

import pytest
import polars as pl
from datetime import datetime, timedelta

from truthound.validators.privacy import (
    # Enums
    PrivacyRegulation,
    PIICategory,
    ConsentStatus,
    LegalBasis,
    # Base validators
    DataRetentionValidator,
    ConsentValidator,
    # GDPR validators
    GDPRComplianceValidator,
    GDPRSpecialCategoryValidator,
    GDPRDataMinimizationValidator,
    GDPRRightToErasureValidator,
    # CCPA validators
    CCPAComplianceValidator,
    CCPASensitiveInfoValidator,
    CCPADoNotSellValidator,
    CCPAConsumerRightsValidator,
    # Global validators
    GlobalPrivacyValidator,
    LGPDComplianceValidator,
    PIPEDAComplianceValidator,
    APPIComplianceValidator,
)
from truthound.types import Severity


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_pii_data():
    """Sample data with various PII types."""
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "email": [
            "john.doe@example.com",
            "jane.smith@company.org",
            "test@test.co.uk",
            None,
            "invalid-email",
        ],
        "phone": [
            "+1-555-123-4567",
            "+44 20 7123 4567",
            "555-123-4567",
            None,
            "invalid",
        ],
        "ip_address": [
            "192.168.1.1",
            "10.0.0.1",
            "172.16.0.1",
            "8.8.8.8",
            None,
        ],
        "name": [
            "John Doe",
            "Jane Smith",
            "Bob Wilson",
            None,
            "Alice Brown",
        ],
    })


@pytest.fixture
def eu_national_id_data():
    """Sample data with EU national IDs."""
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "uk_nino": [
            "AB123456C",
            "CD987654D",
            None,
            "EF111222A",
            "invalid",
        ],
        "spanish_dni": [
            "12345678Z",
            "X1234567L",  # NIE format
            None,
            "87654321X",
            "invalid",
        ],
        "italian_cf": [
            "RSSMRA85M01H501Z",
            None,
            "VRDLGI90A01F839X",
            "invalid",
            "BNCFNC95D15H501S",
        ],
        "german_id": [
            "T220001293",
            None,
            "L01X00T478",
            "invalid",
            "C01X0006H7",
        ],
    })


@pytest.fixture
def special_category_data():
    """Sample data with GDPR Article 9 special categories."""
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "health_status": ["healthy", "diabetic", None, "asthma", "none"],
        "ethnicity": ["Caucasian", "Asian", None, "Hispanic", "Black"],
        "religion": ["Christian", "Muslim", None, "Hindu", "Atheist"],
        "political_party": ["Democrat", "Republican", None, "Independent", "Green"],
        "union_membership": ["Yes", "No", None, "Yes", "No"],
    })


@pytest.fixture
def ccpa_data():
    """Sample data for CCPA testing."""
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "ssn": [
            "123-45-6789",
            "987-65-4321",
            None,
            "111-22-3333",
            "invalid",
        ],
        "drivers_license": [
            "A1234567",
            "B9876543",
            None,
            "C1111222",
            "invalid",
        ],
        "credit_card": [
            "4111-1111-1111-1111",
            "5500-0000-0000-0004",
            None,
            "3400-0000-0000-009",
            "invalid",
        ],
        "geolocation": [
            "37.7749,-122.4194",
            "34.0522,-118.2437",
            None,
            "40.7128,-74.0060",
            "invalid",
        ],
    })


@pytest.fixture
def brazil_data():
    """Sample data with Brazilian identifiers."""
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "cpf": [
            "123.456.789-09",
            "987.654.321-00",
            None,
            "111.222.333-44",
            "invalid",
        ],
        "cnpj": [
            "12.345.678/0001-99",
            "98.765.432/0001-00",
            None,
            "11.222.333/0001-44",
            "invalid",
        ],
        "telefone": [
            "+55 11 98765-4321",
            "+55 21 99876-5432",
            None,
            "+55 31 97654-3210",
            "invalid",
        ],
    })


@pytest.fixture
def canada_data():
    """Sample data with Canadian identifiers."""
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "sin": [
            "123-456-789",
            "987-654-321",
            None,
            "111-222-333",
            "invalid",
        ],
        "health_card": [
            "1234567890ON",
            "9876543210AB",
            None,
            "1111222233BC",
            "invalid",
        ],
        "postal_code": [
            "M5V 3L9",
            "V6B 1A1",
            None,
            "K1A 0B1",
            "invalid",
        ],
    })


@pytest.fixture
def japan_data():
    """Sample data with Japanese identifiers."""
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "my_number": [
            "123456789012",
            "987654321098",
            None,
            "111222333444",
            "invalid",
        ],
        "postal_code": [
            "100-0001",
            "160-0023",
            None,
            "530-0001",
            "invalid",
        ],
        "phone": [
            "03-1234-5678",
            "06-9876-5432",
            None,
            "090-1111-2222",
            "invalid",
        ],
    })


@pytest.fixture
def retention_data():
    """Sample data for retention testing."""
    today = datetime.now()
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "created_at": [
            today - timedelta(days=10),
            today - timedelta(days=100),
            today - timedelta(days=400),  # Over 1 year
            today - timedelta(days=500),  # Over 1 year
            today - timedelta(days=5),
        ],
        "email": [
            "user1@example.com",
            "user2@example.com",
            "old_user@example.com",
            "ancient@example.com",
            "new@example.com",
        ],
    })


@pytest.fixture
def consent_data():
    """Sample data for consent testing."""
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "email": [
            "user1@example.com",
            "user2@example.com",
            "user3@example.com",
            "user4@example.com",
            "user5@example.com",
        ],
        "consent_given": [
            "yes",
            "true",
            "no",
            None,
            "yes",
        ],
    })


@pytest.fixture
def deletion_data():
    """Sample data for right to erasure testing."""
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "email": [
            "active@example.com",
            None,  # Properly erased
            "should_be_deleted@example.com",  # Not erased
            None,  # Properly erased
            "active2@example.com",
        ],
        "is_deleted": [
            "false",
            "true",
            "true",  # Still has email
            "deleted",
            "false",
        ],
    })


@pytest.fixture
def optout_data():
    """Sample data for do-not-sell testing."""
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "email": [
            "user1@example.com",
            "user2@example.com",
            "user3@example.com",
            "user4@example.com",
            "user5@example.com",
        ],
        "do_not_sell": [
            "false",
            "true",
            "opted_out",
            "false",
            "true",
        ],
        "shared_with": [
            "partner_a",
            "partner_b",  # Violation: opted out but shared
            None,
            "partner_c",
            None,
        ],
    })


# ============================================================================
# GDPR Compliance Validator Tests
# ============================================================================

class TestGDPRComplianceValidator:
    """Tests for GDPRComplianceValidator."""

    def test_detects_email_pii(self, sample_pii_data):
        """Test detection of email addresses."""
        validator = GDPRComplianceValidator()
        issues = validator.validate(sample_pii_data.lazy())

        email_issues = [i for i in issues if i.column == "email"]
        assert len(email_issues) >= 1
        assert any("Email" in i.details for i in email_issues)

    def test_detects_ip_address(self, sample_pii_data):
        """Test detection of IP addresses."""
        validator = GDPRComplianceValidator()
        issues = validator.validate(sample_pii_data.lazy())

        ip_issues = [i for i in issues if i.column == "ip_address"]
        assert len(ip_issues) >= 1
        assert any("IP" in i.details for i in ip_issues)

    def test_detects_eu_national_ids(self, eu_national_id_data):
        """Test detection of EU national IDs."""
        validator = GDPRComplianceValidator()
        issues = validator.validate(eu_national_id_data.lazy())

        # Should detect at least some national ID patterns
        assert len(issues) >= 1

    def test_empty_dataframe_no_issues(self):
        """Test that empty DataFrame returns no issues."""
        df = pl.DataFrame({"email": []}).cast({"email": pl.Utf8})
        validator = GDPRComplianceValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 0

    def test_min_confidence_filter(self, sample_pii_data):
        """Test minimum confidence filtering."""
        # High confidence - should find fewer
        validator_high = GDPRComplianceValidator(min_confidence=95)
        issues_high = validator_high.validate(sample_pii_data.lazy())

        # Low confidence - should find more
        validator_low = GDPRComplianceValidator(min_confidence=50)
        issues_low = validator_low.validate(sample_pii_data.lazy())

        # Low confidence should find at least as many as high
        assert len(issues_low) >= len(issues_high)

    def test_specific_columns(self, sample_pii_data):
        """Test validation of specific columns only."""
        validator = GDPRComplianceValidator(columns=["email"])
        issues = validator.validate(sample_pii_data.lazy())

        # Should only have issues for email column
        for issue in issues:
            assert issue.column == "email"


class TestGDPRSpecialCategoryValidator:
    """Tests for GDPRSpecialCategoryValidator."""

    def test_detects_health_data(self, special_category_data):
        """Test detection of health data."""
        validator = GDPRSpecialCategoryValidator()
        issues = validator.validate(special_category_data.lazy())

        health_issues = [i for i in issues if "health" in i.column.lower()]
        assert len(health_issues) >= 1
        assert all(i.severity == Severity.CRITICAL for i in health_issues)

    def test_detects_ethnicity_data(self, special_category_data):
        """Test detection of racial/ethnic data."""
        validator = GDPRSpecialCategoryValidator()
        issues = validator.validate(special_category_data.lazy())

        ethnicity_issues = [i for i in issues if "ethnicity" in i.column.lower()]
        assert len(ethnicity_issues) >= 1

    def test_detects_religious_data(self, special_category_data):
        """Test detection of religious data."""
        validator = GDPRSpecialCategoryValidator()
        issues = validator.validate(special_category_data.lazy())

        religion_issues = [i for i in issues if "religion" in i.column.lower()]
        assert len(religion_issues) >= 1

    def test_detects_political_data(self, special_category_data):
        """Test detection of political data."""
        validator = GDPRSpecialCategoryValidator()
        issues = validator.validate(special_category_data.lazy())

        political_issues = [i for i in issues if "political" in i.column.lower()]
        assert len(political_issues) >= 1

    def test_detects_union_membership(self, special_category_data):
        """Test detection of trade union data."""
        validator = GDPRSpecialCategoryValidator()
        issues = validator.validate(special_category_data.lazy())

        union_issues = [i for i in issues if "union" in i.column.lower()]
        assert len(union_issues) >= 1

    def test_all_special_categories_critical(self, special_category_data):
        """Test that all special category findings are CRITICAL."""
        validator = GDPRSpecialCategoryValidator()
        issues = validator.validate(special_category_data.lazy())

        for issue in issues:
            assert issue.severity == Severity.CRITICAL


class TestGDPRDataMinimizationValidator:
    """Tests for GDPRDataMinimizationValidator."""

    def test_detects_mostly_null_pii_column(self):
        """Test detection of mostly null PII columns."""
        df = pl.DataFrame({
            "id": list(range(100)),
            "email": ["test@example.com"] + [None] * 99,  # 99% null
        })

        validator = GDPRDataMinimizationValidator(max_null_ratio=0.95)
        issues = validator.validate(df.lazy())

        null_issues = [i for i in issues if "unnecessary" in i.issue_type.lower()]
        assert len(null_issues) >= 1

    def test_detects_excessive_pii_columns(self):
        """Test detection of too many PII columns."""
        # Create data with many PII-like columns
        data = {"id": [1, 2, 3]}
        for i in range(15):
            data[f"email_{i}"] = [f"user{j}@example.com" for j in range(3)]

        df = pl.DataFrame(data)

        validator = GDPRDataMinimizationValidator(max_pii_columns=10)
        issues = validator.validate(df.lazy())

        excessive_issues = [i for i in issues if "excessive" in i.issue_type.lower()]
        assert len(excessive_issues) >= 1


class TestGDPRRightToErasureValidator:
    """Tests for GDPRRightToErasureValidator."""

    def test_detects_unerased_pii(self, deletion_data):
        """Test detection of PII not erased in deleted records."""
        validator = GDPRRightToErasureValidator(
            deletion_flag_column="is_deleted",
            deleted_values=["true", "deleted"],
        )
        issues = validator.validate(deletion_data.lazy())

        unerased_issues = [i for i in issues if "not_erased" in i.issue_type.lower()]
        assert len(unerased_issues) >= 1
        assert any(i.severity == Severity.CRITICAL for i in unerased_issues)

    def test_missing_deletion_column(self, sample_pii_data):
        """Test handling of missing deletion flag column."""
        validator = GDPRRightToErasureValidator(
            deletion_flag_column="nonexistent_column",
        )
        issues = validator.validate(sample_pii_data.lazy())

        missing_issues = [i for i in issues if "missing" in i.issue_type.lower()]
        assert len(missing_issues) >= 1

    def test_no_deletion_column_specified(self, sample_pii_data):
        """Test suggestion when no deletion column specified."""
        validator = GDPRRightToErasureValidator()
        issues = validator.validate(sample_pii_data.lazy())

        # Should suggest adding deletion tracking
        tracking_issues = [i for i in issues if "tracking" in i.issue_type.lower()]
        assert len(tracking_issues) >= 1


# ============================================================================
# CCPA Compliance Validator Tests
# ============================================================================

class TestCCPAComplianceValidator:
    """Tests for CCPAComplianceValidator."""

    def test_detects_ssn(self, ccpa_data):
        """Test detection of Social Security Numbers."""
        validator = CCPAComplianceValidator()
        issues = validator.validate(ccpa_data.lazy())

        ssn_issues = [i for i in issues if i.column == "ssn"]
        assert len(ssn_issues) >= 1

    def test_detects_drivers_license(self, ccpa_data):
        """Test detection of driver's license numbers."""
        validator = CCPAComplianceValidator()
        issues = validator.validate(ccpa_data.lazy())

        dl_issues = [i for i in issues if i.column == "drivers_license"]
        assert len(dl_issues) >= 1

    def test_detects_credit_card(self, ccpa_data):
        """Test detection of credit card numbers."""
        validator = CCPAComplianceValidator()
        issues = validator.validate(ccpa_data.lazy())

        cc_issues = [i for i in issues if i.column == "credit_card"]
        assert len(cc_issues) >= 1

    def test_detects_geolocation(self, ccpa_data):
        """Test detection of precise geolocation."""
        validator = CCPAComplianceValidator()
        issues = validator.validate(ccpa_data.lazy())

        geo_issues = [i for i in issues if i.column == "geolocation"]
        assert len(geo_issues) >= 1


class TestCCPASensitiveInfoValidator:
    """Tests for CCPASensitiveInfoValidator."""

    def test_detects_sensitive_pi(self, special_category_data):
        """Test detection of CPRA sensitive PI."""
        validator = CCPASensitiveInfoValidator()
        issues = validator.validate(special_category_data.lazy())

        # Should detect health, ethnicity, religion as sensitive
        assert len(issues) >= 3
        assert all(i.severity == Severity.CRITICAL for i in issues)


class TestCCPADoNotSellValidator:
    """Tests for CCPADoNotSellValidator."""

    def test_detects_sold_after_optout(self, optout_data):
        """Test detection of data shared after opt-out."""
        validator = CCPADoNotSellValidator(
            optout_column="do_not_sell",
            opted_out_values=["true", "opted_out"],
            sharing_indicator_columns=["shared_with"],
        )
        issues = validator.validate(optout_data.lazy())

        violation_issues = [i for i in issues if "sold_after_optout" in i.issue_type.lower()]
        assert len(violation_issues) >= 1
        assert any(i.severity == Severity.CRITICAL for i in violation_issues)

    def test_missing_optout_column(self, sample_pii_data):
        """Test handling of missing opt-out column."""
        validator = CCPADoNotSellValidator(
            optout_column="nonexistent",
        )
        issues = validator.validate(sample_pii_data.lazy())

        missing_issues = [i for i in issues if "missing" in i.issue_type.lower()]
        assert len(missing_issues) >= 1


class TestCCPAConsumerRightsValidator:
    """Tests for CCPAConsumerRightsValidator."""

    def test_suggests_rights_infrastructure(self, sample_pii_data):
        """Test suggestion of consumer rights infrastructure."""
        validator = CCPAConsumerRightsValidator()
        issues = validator.validate(sample_pii_data.lazy())

        # Should suggest adding rights tracking columns
        infra_issues = [i for i in issues if "infrastructure" in i.issue_type.lower()]
        assert len(infra_issues) >= 1


# ============================================================================
# Data Retention Validator Tests
# ============================================================================

class TestDataRetentionValidator:
    """Tests for DataRetentionValidator."""

    def test_detects_expired_records(self, retention_data):
        """Test detection of records beyond retention period."""
        validator = DataRetentionValidator(
            date_column="created_at",
            retention_days=365,  # 1 year
        )
        issues = validator.validate(retention_data.lazy())

        expired_issues = [i for i in issues if "exceeded" in i.issue_type.lower()]
        assert len(expired_issues) >= 1
        # Should find 2 records older than 365 days
        assert any(i.count == 2 for i in expired_issues)

    def test_missing_date_column(self, sample_pii_data):
        """Test handling of missing date column."""
        validator = DataRetentionValidator(
            date_column="nonexistent",
            retention_days=365,
        )
        issues = validator.validate(sample_pii_data.lazy())

        missing_issues = [i for i in issues if "not found" in i.details.lower()]
        assert len(missing_issues) >= 1


# ============================================================================
# Consent Validator Tests
# ============================================================================

class TestConsentValidator:
    """Tests for ConsentValidator."""

    def test_detects_missing_consent(self, consent_data):
        """Test detection of records without valid consent."""
        validator = ConsentValidator(
            consent_column="consent_given",
            pii_columns=["email"],
            valid_consent_values=["yes", "true"],
        )
        issues = validator.validate(consent_data.lazy())

        consent_issues = [i for i in issues if "consent_missing" in i.issue_type.lower()]
        assert len(consent_issues) >= 1
        # Should find 2 records without valid consent (no and None)
        assert any(i.count == 2 for i in consent_issues)

    def test_missing_consent_column(self, sample_pii_data):
        """Test handling of missing consent column."""
        validator = ConsentValidator(
            consent_column="nonexistent",
            pii_columns=["email"],
        )
        issues = validator.validate(sample_pii_data.lazy())

        missing_issues = [i for i in issues if "column_missing" in i.issue_type.lower()]
        assert len(missing_issues) >= 1
        assert any(i.severity == Severity.CRITICAL for i in missing_issues)


# ============================================================================
# Global Privacy Validator Tests
# ============================================================================

class TestGlobalPrivacyValidator:
    """Tests for GlobalPrivacyValidator."""

    def test_multi_jurisdiction_detection(self, sample_pii_data):
        """Test detection across multiple jurisdictions."""
        validator = GlobalPrivacyValidator()
        issues = validator.validate(sample_pii_data.lazy())

        # Should detect email and IP as global PII
        assert len(issues) >= 2

    def test_specific_regulations(self, sample_pii_data):
        """Test filtering by specific regulations."""
        validator = GlobalPrivacyValidator(
            regulations=[PrivacyRegulation.GDPR, PrivacyRegulation.CCPA]
        )
        issues = validator.validate(sample_pii_data.lazy())

        # Should mention GDPR and CCPA in details
        for issue in issues:
            assert "GDPR" in issue.details or "CCPA" in issue.details


class TestLGPDComplianceValidator:
    """Tests for LGPDComplianceValidator."""

    def test_detects_cpf(self, brazil_data):
        """Test detection of Brazilian CPF."""
        validator = LGPDComplianceValidator()
        issues = validator.validate(brazil_data.lazy())

        cpf_issues = [i for i in issues if i.column == "cpf"]
        assert len(cpf_issues) >= 1

    def test_detects_cnpj(self, brazil_data):
        """Test detection of Brazilian CNPJ."""
        validator = LGPDComplianceValidator()
        issues = validator.validate(brazil_data.lazy())

        cnpj_issues = [i for i in issues if i.column == "cnpj"]
        assert len(cnpj_issues) >= 1


class TestPIPEDAComplianceValidator:
    """Tests for PIPEDAComplianceValidator."""

    def test_detects_sin(self, canada_data):
        """Test detection of Canadian SIN."""
        validator = PIPEDAComplianceValidator()
        issues = validator.validate(canada_data.lazy())

        sin_issues = [i for i in issues if i.column == "sin"]
        assert len(sin_issues) >= 1

    def test_detects_health_card(self, canada_data):
        """Test detection of Canadian health card."""
        validator = PIPEDAComplianceValidator()
        issues = validator.validate(canada_data.lazy())

        health_issues = [i for i in issues if i.column == "health_card"]
        assert len(health_issues) >= 1


class TestAPPIComplianceValidator:
    """Tests for APPIComplianceValidator."""

    def test_detects_my_number(self, japan_data):
        """Test detection of Japanese My Number."""
        validator = APPIComplianceValidator()
        issues = validator.validate(japan_data.lazy())

        my_number_issues = [i for i in issues if i.column == "my_number"]
        assert len(my_number_issues) >= 1

    def test_detects_postal_code(self, japan_data):
        """Test detection of Japanese postal code."""
        validator = APPIComplianceValidator()
        issues = validator.validate(japan_data.lazy())

        postal_issues = [i for i in issues if i.column == "postal_code"]
        assert len(postal_issues) >= 1


# ============================================================================
# Enum Tests
# ============================================================================

class TestPrivacyEnums:
    """Tests for privacy-related enums."""

    def test_privacy_regulation_values(self):
        """Test PrivacyRegulation enum values."""
        assert PrivacyRegulation.GDPR.value == "gdpr"
        assert PrivacyRegulation.CCPA.value == "ccpa"
        assert PrivacyRegulation.LGPD.value == "lgpd"
        assert PrivacyRegulation.PIPEDA.value == "pipeda"
        assert PrivacyRegulation.APPI.value == "appi"

    def test_pii_category_values(self):
        """Test PIICategory enum values."""
        assert PIICategory.DIRECT_IDENTIFIER.value == "direct_identifier"
        assert PIICategory.INDIRECT_IDENTIFIER.value == "indirect_identifier"
        assert PIICategory.HEALTH.value == "health"
        assert PIICategory.BIOMETRIC.value == "biometric"

    def test_consent_status_values(self):
        """Test ConsentStatus enum values."""
        assert ConsentStatus.EXPLICIT.value == "explicit"
        assert ConsentStatus.WITHDRAWN.value == "withdrawn"
        assert ConsentStatus.UNKNOWN.value == "unknown"

    def test_legal_basis_values(self):
        """Test LegalBasis enum values."""
        assert LegalBasis.CONSENT.value == "consent"
        assert LegalBasis.CONTRACT.value == "contract"
        assert LegalBasis.LEGITIMATE_INTEREST.value == "legitimate_interest"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestPrivacyEdgeCases:
    """Tests for edge cases in privacy validators."""

    def test_empty_dataframe(self):
        """Test all validators handle empty DataFrames."""
        df = pl.DataFrame({"email": []}).cast({"email": pl.Utf8})

        validators = [
            GDPRComplianceValidator(),
            GDPRSpecialCategoryValidator(),
            CCPAComplianceValidator(),
            CCPASensitiveInfoValidator(),
            GlobalPrivacyValidator(),
        ]

        for validator in validators:
            issues = validator.validate(df.lazy())
            assert len(issues) == 0

    def test_all_null_column(self):
        """Test handling of all-null columns."""
        df = pl.DataFrame({"email": [None, None, None]})

        validator = GDPRComplianceValidator()
        issues = validator.validate(df.lazy())

        # Should not crash, may or may not report issues
        assert isinstance(issues, list)

    def test_non_string_columns_ignored(self):
        """Test that non-string columns are ignored."""
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "value": [1.0, 2.0, 3.0],
            "count": [10, 20, 30],
        })

        validator = GDPRComplianceValidator()
        issues = validator.validate(df.lazy())

        # No string columns, so no PII detection
        assert len(issues) == 0

    def test_unicode_column_names(self):
        """Test handling of Unicode column names."""
        df = pl.DataFrame({
            "이메일": ["test@example.com"],  # Korean
            "電話番号": ["03-1234-5678"],     # Japanese
            "电子邮件": ["user@test.com"],    # Chinese
        })

        validator = GlobalPrivacyValidator()
        issues = validator.validate(df.lazy())

        # Should handle Unicode column names without crashing
        assert isinstance(issues, list)

    def test_mixed_valid_invalid_data(self, sample_pii_data):
        """Test handling of mixed valid/invalid data."""
        validator = GDPRComplianceValidator()
        issues = validator.validate(sample_pii_data.lazy())

        # Should still detect valid PII even with some invalid values
        assert len(issues) >= 1
