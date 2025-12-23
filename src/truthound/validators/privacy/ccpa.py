"""CCPA (California Consumer Privacy Act) compliance validators.

This module provides validators for California CCPA/CPRA compliance including:
- Personal information detection (Section 1798.140)
- Sensitive personal information (CPRA addition)
- Sale/sharing of personal information tracking
- Consumer rights compliance (access, delete, opt-out)

Reference: https://oag.ca.gov/privacy/ccpa
"""

from dataclasses import field
from typing import Any
import re

import polars as pl

from truthound.validators.base import ValidationIssue
from truthound.validators.privacy.base import (
    PrivacyValidator,
    PrivacyRegulation,
    PIICategory,
    PIIFieldDefinition,
    PrivacyFinding,
)
from truthound.types import Severity


# CCPA-specific PII patterns (California focus)
CCPA_PII_DEFINITIONS: list[PIIFieldDefinition] = [
    # Direct Identifiers
    PIIFieldDefinition(
        name="Real Name",
        pattern=re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$"),
        column_hints=["name", "full_name", "first_name", "last_name", "customer_name"],
        category=PIICategory.DIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,  # CCPA uses opt-out model
        confidence_base=75,
        description="Real name including alias per CCPA 1798.140(v)",
    ),
    PIIFieldDefinition(
        name="Postal Address",
        pattern=re.compile(r"^\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr|way|blvd)\b", re.IGNORECASE),
        column_hints=["address", "street", "mailing_address", "home_address", "postal"],
        category=PIICategory.DIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,
        confidence_base=80,
        description="Postal address per CCPA 1798.140(v)",
    ),
    PIIFieldDefinition(
        name="Email Address",
        pattern=re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
        column_hints=["email", "e-mail", "mail", "contact_email"],
        category=PIICategory.DIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,
        confidence_base=95,
        description="Email address per CCPA 1798.140(v)",
    ),
    PIIFieldDefinition(
        name="Phone Number",
        pattern=re.compile(r"^(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}$"),
        column_hints=["phone", "telephone", "mobile", "cell", "contact_phone"],
        category=PIICategory.DIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,
        confidence_base=90,
        description="Telephone number per CCPA 1798.140(v)",
    ),

    # US Government Identifiers
    PIIFieldDefinition(
        name="Social Security Number",
        pattern=re.compile(r"^\d{3}-\d{2}-\d{4}$"),
        column_hints=["ssn", "social_security", "ss_number", "tax_id"],
        category=PIICategory.GOVERNMENT_ID,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,
        is_special_category=True,  # Sensitive PI under CPRA
        confidence_base=98,
        description="SSN per CCPA 1798.140(v) - Sensitive PI under CPRA",
    ),
    PIIFieldDefinition(
        name="Driver's License Number",
        pattern=re.compile(r"^[A-Z]\d{7}$|^[A-Z]\d{4}-\d{5}-\d{5}$"),  # CA formats
        column_hints=["drivers_license", "dl_number", "license_number", "dmv"],
        category=PIICategory.GOVERNMENT_ID,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,
        is_special_category=True,
        confidence_base=90,
        description="Driver's license per CCPA 1798.140(v)",
    ),
    PIIFieldDefinition(
        name="State ID Number",
        pattern=re.compile(r"^[A-Z0-9]{7,12}$"),
        column_hints=["state_id", "id_number", "identification"],
        category=PIICategory.GOVERNMENT_ID,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,
        is_special_category=True,
        confidence_base=75,
        description="State ID per CCPA 1798.140(v)",
    ),
    PIIFieldDefinition(
        name="Passport Number",
        pattern=re.compile(r"^[A-Z0-9]{6,9}$"),
        column_hints=["passport", "passport_number", "travel_doc"],
        category=PIICategory.GOVERNMENT_ID,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,
        is_special_category=True,
        confidence_base=85,
        description="Passport number per CCPA 1798.140(v)",
    ),

    # Financial Information
    PIIFieldDefinition(
        name="Bank Account Number",
        pattern=re.compile(r"^\d{8,17}$"),
        column_hints=["bank_account", "account_number", "routing", "aba"],
        category=PIICategory.FINANCIAL,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,
        is_special_category=True,  # Sensitive PI under CPRA
        confidence_base=80,
        description="Bank account per CCPA 1798.140(v) - Sensitive under CPRA",
    ),
    PIIFieldDefinition(
        name="Credit/Debit Card Number",
        pattern=re.compile(r"^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$"),
        column_hints=["credit_card", "card_number", "cc_number", "debit_card"],
        category=PIICategory.FINANCIAL,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,
        is_special_category=True,
        confidence_base=95,
        description="Credit/debit card per CCPA 1798.140(v) - Sensitive under CPRA",
    ),
    PIIFieldDefinition(
        name="Financial Account Credentials",
        pattern=None,
        column_hints=["pin", "security_code", "cvv", "password", "login_credentials"],
        category=PIICategory.FINANCIAL,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,
        is_special_category=True,
        confidence_base=95,
        description="Financial credentials per CCPA 1798.140(v)",
    ),

    # Biometric & Genetic (Sensitive PI under CPRA)
    PIIFieldDefinition(
        name="Biometric Information",
        pattern=None,
        column_hints=[
            "biometric", "fingerprint", "face_scan", "voice_print",
            "retina", "palm_print", "facial_geometry",
        ],
        category=PIICategory.BIOMETRIC,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,
        is_special_category=True,
        confidence_base=95,
        description="Biometric data - Sensitive PI under CPRA 1798.140(ae)",
    ),
    PIIFieldDefinition(
        name="Genetic Information",
        pattern=None,
        column_hints=["genetic", "dna", "genome", "hereditary", "gene_test"],
        category=PIICategory.GENETIC,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,
        is_special_category=True,
        confidence_base=98,
        description="Genetic data - Sensitive PI under CPRA 1798.140(ae)",
    ),

    # Health Information (Sensitive PI under CPRA)
    PIIFieldDefinition(
        name="Health Information",
        pattern=None,
        column_hints=[
            "health", "medical", "diagnosis", "treatment", "prescription",
            "condition", "insurance_claim", "healthcare",
        ],
        category=PIICategory.HEALTH,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,
        is_special_category=True,
        confidence_base=90,
        description="Health data - Sensitive PI under CPRA 1798.140(ae)",
    ),

    # Sensitive PI Categories (CPRA additions)
    PIIFieldDefinition(
        name="Racial/Ethnic Origin",
        pattern=None,
        column_hints=["race", "ethnicity", "ethnic_origin", "racial"],
        category=PIICategory.RACIAL_ETHNIC,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,
        is_special_category=True,
        confidence_base=95,
        description="Racial/ethnic origin - Sensitive PI under CPRA",
    ),
    PIIFieldDefinition(
        name="Religious/Philosophical Beliefs",
        pattern=None,
        column_hints=["religion", "religious", "faith", "belief", "philosophical"],
        category=PIICategory.RELIGIOUS,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,
        is_special_category=True,
        confidence_base=95,
        description="Religious beliefs - Sensitive PI under CPRA",
    ),
    PIIFieldDefinition(
        name="Union Membership",
        pattern=None,
        column_hints=["union", "union_member", "labor_union"],
        category=PIICategory.TRADE_UNION,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,
        is_special_category=True,
        confidence_base=90,
        description="Union membership - Sensitive PI under CPRA",
    ),
    PIIFieldDefinition(
        name="Sexual Orientation",
        pattern=None,
        column_hints=["sexual_orientation", "sexuality", "lgbtq", "gender_identity"],
        category=PIICategory.SEX_LIFE,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,
        is_special_category=True,
        confidence_base=98,
        description="Sexual orientation - Sensitive PI under CPRA",
    ),
    PIIFieldDefinition(
        name="Citizenship/Immigration Status",
        pattern=None,
        column_hints=["citizenship", "immigration", "visa_status", "residency_status"],
        category=PIICategory.GOVERNMENT_ID,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,
        is_special_category=True,
        confidence_base=90,
        description="Immigration status - Sensitive PI under CPRA",
    ),

    # Internet/Online Identifiers
    PIIFieldDefinition(
        name="IP Address",
        pattern=re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$"),
        column_hints=["ip", "ip_address", "client_ip", "user_ip"],
        category=PIICategory.INDIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,
        confidence_base=95,
        description="IP address per CCPA 1798.140(v)",
    ),
    PIIFieldDefinition(
        name="Device Identifier",
        pattern=re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"),
        column_hints=["device_id", "udid", "advertising_id", "idfa", "gaid"],
        category=PIICategory.INDIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,
        confidence_base=90,
        description="Device identifier per CCPA 1798.140(v)",
    ),
    PIIFieldDefinition(
        name="Cookie/Tracking ID",
        pattern=re.compile(r"^[a-zA-Z0-9_-]{16,}$"),
        column_hints=["cookie", "tracking_id", "session_id", "visitor_id"],
        category=PIICategory.INDIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,
        confidence_base=75,
        description="Cookies/tracking IDs per CCPA 1798.140(v)",
    ),

    # Geolocation (Sensitive PI under CPRA if precise)
    PIIFieldDefinition(
        name="Precise Geolocation",
        pattern=re.compile(r"^-?\d{1,3}\.\d{4,},\s*-?\d{1,3}\.\d{4,}$"),
        column_hints=["geolocation", "gps", "lat_long", "coordinates", "location"],
        category=PIICategory.INDIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.CCPA],
        requires_consent=False,
        is_special_category=True,  # Sensitive under CPRA
        confidence_base=90,
        description="Precise geolocation - Sensitive PI under CPRA 1798.140(ae)",
    ),
]


class CCPAComplianceValidator(PrivacyValidator):
    """CCPA personal information detection validator.

    Detects personal information as defined in CCPA Section 1798.140(v):
    Information that identifies, relates to, describes, is reasonably
    capable of being associated with, or could reasonably be linked,
    directly or indirectly, with a particular consumer or household.

    Also covers CPRA (California Privacy Rights Act) sensitive PI categories.
    """

    name = "ccpa_compliance"
    regulation = PrivacyRegulation.CCPA

    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        """Get CCPA-specific PII definitions."""
        return CCPA_PII_DEFINITIONS

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate CCPA compliance.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []
        pii_defs = self.get_pii_definitions()

        schema = lf.collect_schema()
        df = lf.collect()

        if df.height == 0:
            return issues

        # Get columns to check
        if self.columns:
            columns = [c for c in self.columns if c in schema.names()]
        else:
            columns = [
                c for c in schema.names()
                if schema[c] in (pl.String, pl.Utf8)
            ]

        # Detect PII in each column
        all_findings: list[PrivacyFinding] = []

        for column in columns:
            findings = self._detect_pii_in_column(df, column, pii_defs)
            all_findings.extend(findings)

        # Convert findings to issues
        issues = self._convert_findings_to_issues(all_findings)

        return issues


class CCPASensitiveInfoValidator(PrivacyValidator):
    """CPRA sensitive personal information validator.

    Specifically detects sensitive personal information as defined
    in CPRA Section 1798.140(ae):
    - SSN, driver's license, state ID, passport
    - Account credentials
    - Precise geolocation
    - Racial/ethnic origin, religious beliefs, union membership
    - Personal communications (mail, email, text)
    - Genetic/biometric data
    - Health information
    - Sex life/sexual orientation
    """

    name = "ccpa_sensitive_info"
    regulation = PrivacyRegulation.CCPA

    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        """Get only sensitive PI definitions."""
        return [d for d in CCPA_PII_DEFINITIONS if d.is_special_category]

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate for sensitive personal information.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []
        pii_defs = self.get_pii_definitions()

        schema = lf.collect_schema()
        df = lf.collect()

        if df.height == 0:
            return issues

        for pii_def in pii_defs:
            for column in schema.names():
                if pii_def.matches_column_name(column):
                    issues.append(ValidationIssue(
                        column=column,
                        issue_type="ccpa_sensitive_pi_detected",
                        count=df.height,
                        severity=Severity.CRITICAL,
                        details=(
                            f"CRITICAL: Column '{column}' appears to contain "
                            f"{pii_def.name} (CPRA Sensitive Personal Information). "
                            f"{pii_def.description}. "
                            "Consumers have right to limit use/disclosure."
                        ),
                        expected="No sensitive PI without proper disclosure",
                        actual=f"Potential {pii_def.name} data detected",
                    ))

        return issues


class CCPADoNotSellValidator(PrivacyValidator):
    """CCPA Do Not Sell/Share validator.

    Validates compliance with CCPA right to opt-out of sale/sharing
    of personal information (Section 1798.120).

    Checks:
    - Presence of opt-out tracking column
    - Records with PI but no opt-out status
    - Potential sharing indicators
    """

    name = "ccpa_do_not_sell"
    regulation = PrivacyRegulation.CCPA

    def __init__(
        self,
        optout_column: str | None = None,
        opted_out_values: list[str] | None = None,
        sharing_indicator_columns: list[str] | None = None,
        **kwargs: Any,
    ):
        """Initialize do-not-sell validator.

        Args:
            optout_column: Column tracking opt-out status
            opted_out_values: Values indicating user opted out
            sharing_indicator_columns: Columns indicating data sharing
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.optout_column = optout_column
        self.opted_out_values = opted_out_values or [
            "true", "1", "yes", "opted_out", "do_not_sell"
        ]
        self.sharing_indicator_columns = sharing_indicator_columns or [
            "shared_with", "third_party", "partner", "sold_to", "disclosed_to"
        ]

    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        """Get CCPA PII definitions."""
        return CCPA_PII_DEFINITIONS

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate do-not-sell compliance.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        schema = lf.collect_schema()
        df = lf.collect()

        if df.height == 0:
            return issues

        # Check for opt-out column
        if self.optout_column:
            if self.optout_column not in schema.names():
                issues.append(ValidationIssue(
                    column=self.optout_column,
                    issue_type="ccpa_optout_column_missing",
                    count=0,
                    severity=Severity.HIGH,
                    details=(
                        f"Opt-out tracking column '{self.optout_column}' not found. "
                        "CCPA 1798.120 requires honoring 'Do Not Sell' requests."
                    ),
                ))
            else:
                # Check for records opted out but potentially shared
                opted_out_values = [v.lower() for v in self.opted_out_values]

                opted_out_records = df.filter(
                    pl.col(self.optout_column)
                    .cast(pl.Utf8)
                    .str.to_lowercase()
                    .is_in(opted_out_values)
                )

                if opted_out_records.height > 0:
                    # Check if sharing indicators exist for opted-out users
                    for share_col in self.sharing_indicator_columns:
                        if share_col in schema.names():
                            shared_after_optout = opted_out_records.filter(
                                pl.col(share_col).is_not_null()
                            ).height

                            if shared_after_optout > 0:
                                issues.append(ValidationIssue(
                                    column=share_col,
                                    issue_type="ccpa_sold_after_optout",
                                    count=shared_after_optout,
                                    severity=Severity.CRITICAL,
                                    details=(
                                        f"Found {shared_after_optout} records with "
                                        f"opt-out status but sharing indicator in "
                                        f"'{share_col}'. This may violate CCPA 1798.120."
                                    ),
                                    expected="No sharing for opted-out consumers",
                                    actual=f"{shared_after_optout} potential violations",
                                ))
        else:
            issues.append(ValidationIssue(
                column="_table",
                issue_type="ccpa_optout_tracking_missing",
                count=0,
                severity=Severity.MEDIUM,
                details=(
                    "No opt-out tracking column specified. CCPA requires businesses "
                    "to track and honor 'Do Not Sell My Personal Information' requests."
                ),
            ))

        return issues


class CCPAConsumerRightsValidator(PrivacyValidator):
    """CCPA consumer rights compliance validator.

    Validates support for CCPA consumer rights:
    - Right to know (1798.100)
    - Right to delete (1798.105)
    - Right to opt-out (1798.120)
    - Right to non-discrimination (1798.125)
    - Right to correct (CPRA 1798.106)
    - Right to limit sensitive PI use (CPRA 1798.121)

    Checks for tracking columns that support these rights.
    """

    name = "ccpa_consumer_rights"
    regulation = PrivacyRegulation.CCPA

    def __init__(
        self,
        consumer_id_column: str | None = None,
        deletion_request_column: str | None = None,
        access_request_column: str | None = None,
        correction_request_column: str | None = None,
        **kwargs: Any,
    ):
        """Initialize consumer rights validator.

        Args:
            consumer_id_column: Column identifying consumer
            deletion_request_column: Column tracking deletion requests
            access_request_column: Column tracking access requests
            correction_request_column: Column tracking correction requests
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.consumer_id_column = consumer_id_column
        self.deletion_request_column = deletion_request_column
        self.access_request_column = access_request_column
        self.correction_request_column = correction_request_column

    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        """Get CCPA PII definitions."""
        return CCPA_PII_DEFINITIONS

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate consumer rights infrastructure.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        schema = lf.collect_schema()
        df = lf.collect()

        # Check for consumer identification
        if self.consumer_id_column:
            if self.consumer_id_column not in schema.names():
                issues.append(ValidationIssue(
                    column=self.consumer_id_column,
                    issue_type="ccpa_consumer_id_missing",
                    count=0,
                    severity=Severity.HIGH,
                    details=(
                        f"Consumer ID column '{self.consumer_id_column}' not found. "
                        "Consumer identification needed to fulfill rights requests."
                    ),
                ))

        # Check deletion request tracking
        if self.deletion_request_column:
            if self.deletion_request_column not in schema.names():
                issues.append(ValidationIssue(
                    column=self.deletion_request_column,
                    issue_type="ccpa_deletion_tracking_missing",
                    count=0,
                    severity=Severity.MEDIUM,
                    details=(
                        "Deletion request tracking not found. "
                        "CCPA 1798.105 requires responding to deletion requests."
                    ),
                ))
            else:
                # Check for pending deletion requests
                pending_deletions = df.filter(
                    pl.col(self.deletion_request_column).is_not_null()
                ).height

                if pending_deletions > 0:
                    issues.append(ValidationIssue(
                        column=self.deletion_request_column,
                        issue_type="ccpa_pending_deletion_requests",
                        count=pending_deletions,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Found {pending_deletions} records with deletion request "
                            "flags. Ensure these are processed within 45 days per CCPA."
                        ),
                    ))

        # Suggest best practices
        suggested_columns = {
            "deletion_request": "Track deletion requests (CCPA 1798.105)",
            "access_request": "Track access/know requests (CCPA 1798.100)",
            "optout_sale": "Track sale opt-outs (CCPA 1798.120)",
            "optout_share": "Track sharing opt-outs (CPRA)",
            "limit_sensitive": "Track sensitive PI limits (CPRA 1798.121)",
        }

        missing_suggestions = []
        for col, purpose in suggested_columns.items():
            if not any(col in c.lower() for c in schema.names()):
                missing_suggestions.append(f"  - {col}: {purpose}")

        if missing_suggestions:
            issues.append(ValidationIssue(
                column="_table",
                issue_type="ccpa_rights_infrastructure_incomplete",
                count=0,
                severity=Severity.LOW,
                details=(
                    "Consider adding columns to support consumer rights:\n" +
                    "\n".join(missing_suggestions)
                ),
            ))

        return issues
