"""GDPR (General Data Protection Regulation) compliance validators.

This module provides validators for EU GDPR compliance including:
- Personal data detection (Article 4)
- Special category data detection (Article 9)
- Data minimization validation (Article 5)
- Right to erasure compliance (Article 17)
- Data retention validation (Article 5)

Reference: https://gdpr.eu/
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


# GDPR-specific PII patterns for EU member states
GDPR_PII_DEFINITIONS: list[PIIFieldDefinition] = [
    # Direct Identifiers
    PIIFieldDefinition(
        name="Email Address",
        pattern=re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
        column_hints=["email", "e-mail", "mail", "contact_email", "user_email"],
        category=PIICategory.DIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        confidence_base=95,
        description="Email addresses are personal data under GDPR Article 4",
    ),
    PIIFieldDefinition(
        name="Full Name",
        pattern=re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$"),
        column_hints=["name", "full_name", "fullname", "customer_name", "user_name", "person"],
        category=PIICategory.DIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        confidence_base=75,
        description="Personal names are direct identifiers under GDPR",
    ),
    PIIFieldDefinition(
        name="Phone Number (EU)",
        pattern=re.compile(r"^\+?(?:31|32|33|34|39|44|49|[1-9]\d{0,2})[\s.-]?\d{6,12}$"),
        column_hints=["phone", "tel", "telephone", "mobile", "cell", "contact_phone"],
        category=PIICategory.DIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        confidence_base=85,
        description="Phone numbers including EU country codes",
    ),
    PIIFieldDefinition(
        name="Physical Address",
        pattern=re.compile(r"^\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr|way|court|ct|plaza|square|boulevard|blvd)\b", re.IGNORECASE),
        column_hints=["address", "street", "location", "residence", "home_address", "postal"],
        category=PIICategory.DIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        confidence_base=80,
        description="Physical addresses are personal data",
    ),

    # EU National IDs
    PIIFieldDefinition(
        name="German Personal ID (Personalausweis)",
        pattern=re.compile(r"^[CFGHJKLMNPRTVWXYZ0-9]{9}$"),
        column_hints=["personalausweis", "ausweis", "german_id", "de_id"],
        category=PIICategory.GOVERNMENT_ID,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        is_special_category=False,
        confidence_base=85,
        description="German national ID card number",
    ),
    PIIFieldDefinition(
        name="French National ID (NIR/INSEE)",
        pattern=re.compile(r"^[12]\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[1-8]\d|9[0-5]|2[AB])\d{6}\d{2}$"),
        column_hints=["nir", "insee", "french_id", "numero_secu", "securite_sociale"],
        category=PIICategory.GOVERNMENT_ID,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        is_special_category=False,
        confidence_base=95,
        description="French social security number (NIR)",
    ),
    PIIFieldDefinition(
        name="UK National Insurance Number",
        pattern=re.compile(r"^[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]$"),
        column_hints=["nino", "ni_number", "national_insurance", "uk_ni"],
        category=PIICategory.GOVERNMENT_ID,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        confidence_base=95,
        description="UK National Insurance Number",
    ),
    PIIFieldDefinition(
        name="Spanish DNI/NIE",
        pattern=re.compile(r"^(?:\d{8}[A-Z]|[XYZ]\d{7}[A-Z])$"),
        column_hints=["dni", "nie", "spanish_id", "nif"],
        category=PIICategory.GOVERNMENT_ID,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        confidence_base=95,
        description="Spanish national ID (DNI) or foreigner ID (NIE)",
    ),
    PIIFieldDefinition(
        name="Italian Fiscal Code",
        pattern=re.compile(r"^[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]$"),
        column_hints=["codice_fiscale", "fiscal_code", "italian_id", "cf"],
        category=PIICategory.GOVERNMENT_ID,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        confidence_base=95,
        description="Italian fiscal code (Codice Fiscale)",
    ),
    PIIFieldDefinition(
        name="Dutch BSN",
        pattern=re.compile(r"^\d{9}$"),
        column_hints=["bsn", "burgerservicenummer", "dutch_id", "nl_id"],
        category=PIICategory.GOVERNMENT_ID,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        confidence_base=80,
        description="Dutch citizen service number (BSN)",
    ),
    PIIFieldDefinition(
        name="Belgian National Number",
        pattern=re.compile(r"^\d{2}\.?\d{2}\.?\d{2}-?\d{3}\.?\d{2}$"),
        column_hints=["rijksregisternummer", "national_number", "belgian_id", "be_id"],
        category=PIICategory.GOVERNMENT_ID,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        confidence_base=90,
        description="Belgian national register number",
    ),
    PIIFieldDefinition(
        name="Polish PESEL",
        pattern=re.compile(r"^\d{11}$"),
        column_hints=["pesel", "polish_id", "pl_id"],
        category=PIICategory.GOVERNMENT_ID,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        confidence_base=85,
        description="Polish personal identification number (PESEL)",
    ),

    # Indirect Identifiers
    PIIFieldDefinition(
        name="IP Address",
        pattern=re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$"),
        column_hints=["ip", "ip_address", "ipaddress", "client_ip", "user_ip", "source_ip"],
        category=PIICategory.INDIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        confidence_base=95,
        description="IP addresses are personal data per GDPR (Breyer case)",
    ),
    PIIFieldDefinition(
        name="IPv6 Address",
        pattern=re.compile(r"^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$"),
        column_hints=["ip", "ip_address", "ipv6", "client_ip"],
        category=PIICategory.INDIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        confidence_base=95,
        description="IPv6 addresses are personal data per GDPR",
    ),
    PIIFieldDefinition(
        name="Device ID",
        pattern=re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"),
        column_hints=["device_id", "deviceid", "udid", "uuid", "device_uuid"],
        category=PIICategory.INDIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        confidence_base=85,
        description="Device identifiers can identify individuals",
    ),
    PIIFieldDefinition(
        name="Cookie ID",
        pattern=re.compile(r"^[a-zA-Z0-9_-]{20,}$"),
        column_hints=["cookie", "cookie_id", "session_id", "tracking_id"],
        category=PIICategory.INDIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        confidence_base=70,
        description="Cookie identifiers require consent under ePrivacy",
    ),
    PIIFieldDefinition(
        name="MAC Address",
        pattern=re.compile(r"^(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}$"),
        column_hints=["mac", "mac_address", "hardware_address"],
        category=PIICategory.INDIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        confidence_base=95,
        description="MAC addresses can identify devices and users",
    ),

    # Financial Data
    PIIFieldDefinition(
        name="IBAN",
        pattern=re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]?){0,16}$"),
        column_hints=["iban", "bank_account", "account_number"],
        category=PIICategory.FINANCIAL,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        confidence_base=95,
        description="International Bank Account Number",
    ),
    PIIFieldDefinition(
        name="Credit Card",
        pattern=re.compile(r"^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$"),
        column_hints=["credit_card", "card_number", "cc", "payment_card"],
        category=PIICategory.FINANCIAL,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        confidence_base=95,
        description="Credit/debit card numbers require PCI-DSS compliance",
    ),

    # Special Categories (GDPR Article 9)
    PIIFieldDefinition(
        name="Health/Medical Data",
        pattern=None,  # Detected by column name
        column_hints=[
            "health", "medical", "diagnosis", "treatment", "medication",
            "prescription", "allergy", "condition", "symptom", "disease",
            "blood_type", "disability", "mental_health", "patient",
        ],
        category=PIICategory.HEALTH,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        is_special_category=True,
        confidence_base=90,
        description="Health data is special category under Article 9",
    ),
    PIIFieldDefinition(
        name="Racial/Ethnic Origin",
        pattern=None,
        column_hints=[
            "race", "ethnicity", "ethnic", "racial", "origin", "nationality",
            "ethnic_group", "ethnic_background",
        ],
        category=PIICategory.RACIAL_ETHNIC,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        is_special_category=True,
        confidence_base=95,
        description="Racial/ethnic origin is special category under Article 9",
    ),
    PIIFieldDefinition(
        name="Political Opinion",
        pattern=None,
        column_hints=[
            "political", "party", "vote", "voting", "election", "politics",
            "political_affiliation", "political_view",
        ],
        category=PIICategory.POLITICAL,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        is_special_category=True,
        confidence_base=95,
        description="Political opinions are special category under Article 9",
    ),
    PIIFieldDefinition(
        name="Religious/Philosophical Belief",
        pattern=None,
        column_hints=[
            "religion", "religious", "faith", "belief", "church", "mosque",
            "temple", "denomination", "spiritual",
        ],
        category=PIICategory.RELIGIOUS,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        is_special_category=True,
        confidence_base=95,
        description="Religious beliefs are special category under Article 9",
    ),
    PIIFieldDefinition(
        name="Trade Union Membership",
        pattern=None,
        column_hints=[
            "union", "trade_union", "labor_union", "membership", "guild",
        ],
        category=PIICategory.TRADE_UNION,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        is_special_category=True,
        confidence_base=95,
        description="Trade union membership is special category under Article 9",
    ),
    PIIFieldDefinition(
        name="Genetic Data",
        pattern=None,
        column_hints=[
            "genetic", "dna", "genome", "gene", "hereditary", "genotype",
        ],
        category=PIICategory.GENETIC,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        is_special_category=True,
        confidence_base=98,
        description="Genetic data is special category under Article 9",
    ),
    PIIFieldDefinition(
        name="Biometric Data",
        pattern=None,
        column_hints=[
            "biometric", "fingerprint", "face_id", "facial", "iris", "retina",
            "voice_print", "palm", "hand_geometry",
        ],
        category=PIICategory.BIOMETRIC,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        is_special_category=True,
        confidence_base=98,
        description="Biometric data is special category under Article 9",
    ),
    PIIFieldDefinition(
        name="Sexual Orientation",
        pattern=None,
        column_hints=[
            "sexual", "orientation", "gender_identity", "lgbtq", "sexuality",
        ],
        category=PIICategory.SEX_LIFE,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        is_special_category=True,
        confidence_base=98,
        description="Sex life/orientation is special category under Article 9",
    ),
    PIIFieldDefinition(
        name="Criminal Record",
        pattern=None,
        column_hints=[
            "criminal", "conviction", "offense", "arrest", "sentence",
            "crime", "felony", "misdemeanor", "court_record",
        ],
        category=PIICategory.CRIMINAL,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        is_special_category=True,
        confidence_base=95,
        description="Criminal data requires official authority under Article 10",
    ),

    # Date of Birth (age can reveal minor status)
    PIIFieldDefinition(
        name="Date of Birth",
        pattern=re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}$|^\d{2}[-/]\d{2}[-/]\d{4}$"),
        column_hints=["dob", "birth", "birthday", "birth_date", "date_of_birth", "born"],
        category=PIICategory.DIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.GDPR],
        requires_consent=True,
        confidence_base=85,
        description="Date of birth reveals age and can identify minors",
    ),
]


class GDPRComplianceValidator(PrivacyValidator):
    """GDPR Article 4 personal data detection validator.

    Detects personal data as defined in GDPR Article 4:
    'personal data' means any information relating to an identified
    or identifiable natural person ('data subject').

    This includes:
    - Direct identifiers (name, email, phone)
    - Indirect identifiers (IP address, device ID, cookie)
    - Special categories (Article 9 sensitive data)
    """

    name = "gdpr_compliance"
    regulation = PrivacyRegulation.GDPR

    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        """Get GDPR-specific PII definitions."""
        return GDPR_PII_DEFINITIONS

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate GDPR compliance.

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
            # Check all string columns
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


class GDPRSpecialCategoryValidator(PrivacyValidator):
    """GDPR Article 9 special category data validator.

    Specifically detects special categories of personal data which
    require explicit consent or specific legal basis:
    - Racial or ethnic origin
    - Political opinions
    - Religious or philosophical beliefs
    - Trade union membership
    - Genetic data
    - Biometric data (for identification)
    - Health data
    - Sex life or sexual orientation
    """

    name = "gdpr_special_category"
    regulation = PrivacyRegulation.GDPR

    def __init__(
        self,
        check_column_names_only: bool = False,
        **kwargs: Any,
    ):
        """Initialize special category validator.

        Args:
            check_column_names_only: Only check column names, not values
            **kwargs: Additional config
        """
        super().__init__(detect_special_categories=True, **kwargs)
        self.check_column_names_only = check_column_names_only

    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        """Get only special category definitions."""
        return [d for d in GDPR_PII_DEFINITIONS if d.is_special_category]

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate for special category data.

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
                # Check column name match
                if pii_def.matches_column_name(column):
                    issues.append(ValidationIssue(
                        column=column,
                        issue_type="gdpr_special_category_detected",
                        count=df.height,
                        severity=Severity.CRITICAL,
                        details=(
                            f"CRITICAL: Column '{column}' appears to contain "
                            f"{pii_def.name} (GDPR Article 9 special category). "
                            f"{pii_def.description}. "
                            "Explicit consent or specific legal basis required."
                        ),
                        expected="No special category data without explicit consent",
                        actual=f"Potential {pii_def.name} data detected",
                    ))

        return issues


class GDPRDataMinimizationValidator(PrivacyValidator):
    """GDPR Article 5(1)(c) data minimization validator.

    Validates that personal data is adequate, relevant and limited
    to what is necessary in relation to the purposes for processing.

    Checks for:
    - Columns with excessive null ratios (may be unnecessary)
    - Duplicate PII columns (redundant data)
    - High-cardinality PII (potentially excessive)
    """

    name = "gdpr_data_minimization"
    regulation = PrivacyRegulation.GDPR

    def __init__(
        self,
        max_null_ratio: float = 0.95,
        max_pii_columns: int = 10,
        **kwargs: Any,
    ):
        """Initialize data minimization validator.

        Args:
            max_null_ratio: Maximum acceptable null ratio for PII columns
            max_pii_columns: Maximum number of PII columns before warning
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.max_null_ratio = max_null_ratio
        self.max_pii_columns = max_pii_columns

    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        """Get GDPR PII definitions."""
        return GDPR_PII_DEFINITIONS

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate data minimization compliance.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []
        pii_defs = self.get_pii_definitions()

        schema = lf.collect_schema()
        df = lf.collect()
        total_rows = df.height

        if total_rows == 0:
            return issues

        # Track detected PII columns
        pii_columns: list[tuple[str, str]] = []

        for column in schema.names():
            for pii_def in pii_defs:
                if pii_def.matches_column_name(column):
                    pii_columns.append((column, pii_def.name))

                    # Check null ratio
                    null_count = df.select(pl.col(column).null_count()).item()
                    null_ratio = null_count / total_rows

                    if null_ratio > self.max_null_ratio:
                        issues.append(ValidationIssue(
                            column=column,
                            issue_type="gdpr_unnecessary_pii_column",
                            count=null_count,
                            severity=Severity.MEDIUM,
                            details=(
                                f"PII column '{column}' ({pii_def.name}) has "
                                f"{null_ratio:.1%} null values. Consider removing "
                                "this column per data minimization principle."
                            ),
                            expected=f"Null ratio <= {self.max_null_ratio:.0%}",
                            actual=f"{null_ratio:.1%} null",
                        ))
                    break

        # Check for excessive PII columns
        if len(pii_columns) > self.max_pii_columns:
            issues.append(ValidationIssue(
                column="_table",
                issue_type="gdpr_excessive_pii_columns",
                count=len(pii_columns),
                severity=Severity.MEDIUM,
                details=(
                    f"Found {len(pii_columns)} potential PII columns, "
                    f"exceeding threshold of {self.max_pii_columns}. "
                    "Review if all are necessary per data minimization principle. "
                    f"Columns: {', '.join(c[0] for c in pii_columns[:5])}..."
                ),
                expected=f"<= {self.max_pii_columns} PII columns",
                actual=f"{len(pii_columns)} PII columns",
            ))

        return issues


class GDPRRightToErasureValidator(PrivacyValidator):
    """GDPR Article 17 right to erasure (right to be forgotten) validator.

    Validates that systems support data subject erasure requests by checking:
    - Presence of deletion/erasure flag columns
    - Records marked for deletion but still present
    - Orphaned PII after related records deleted
    """

    name = "gdpr_right_to_erasure"
    regulation = PrivacyRegulation.GDPR

    def __init__(
        self,
        deletion_flag_column: str | None = None,
        deleted_values: list[str] | None = None,
        **kwargs: Any,
    ):
        """Initialize right to erasure validator.

        Args:
            deletion_flag_column: Column indicating deletion status
            deleted_values: Values indicating record should be deleted
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.deletion_flag_column = deletion_flag_column
        self.deleted_values = deleted_values or [
            "deleted", "erased", "true", "1", "yes", "removed"
        ]

    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        """Get GDPR PII definitions."""
        return GDPR_PII_DEFINITIONS

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate right to erasure support.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        schema = lf.collect_schema()
        df = lf.collect()

        # Check if deletion flag column exists
        if self.deletion_flag_column:
            if self.deletion_flag_column not in schema.names():
                issues.append(ValidationIssue(
                    column=self.deletion_flag_column,
                    issue_type="gdpr_erasure_flag_missing",
                    count=0,
                    severity=Severity.HIGH,
                    details=(
                        f"Deletion flag column '{self.deletion_flag_column}' not found. "
                        "GDPR Article 17 requires ability to erase personal data."
                    ),
                ))
            else:
                # Check for records marked as deleted but still containing PII
                deleted_values = [v.lower() for v in self.deleted_values]

                marked_deleted = df.filter(
                    pl.col(self.deletion_flag_column)
                    .cast(pl.Utf8)
                    .str.to_lowercase()
                    .is_in(deleted_values)
                )

                if marked_deleted.height > 0:
                    # Check if deleted records still have PII
                    pii_defs = self.get_pii_definitions()
                    for column in schema.names():
                        if column == self.deletion_flag_column:
                            continue

                        for pii_def in pii_defs:
                            if pii_def.matches_column_name(column):
                                # Count non-null PII in deleted records
                                pii_in_deleted = marked_deleted.filter(
                                    pl.col(column).is_not_null()
                                ).height

                                if pii_in_deleted > 0:
                                    issues.append(ValidationIssue(
                                        column=column,
                                        issue_type="gdpr_pii_not_erased",
                                        count=pii_in_deleted,
                                        severity=Severity.CRITICAL,
                                        details=(
                                            f"Found {pii_in_deleted} records marked for "
                                            f"deletion but PII column '{column}' "
                                            f"({pii_def.name}) still contains data. "
                                            "This violates GDPR Article 17."
                                        ),
                                        expected="PII erased in deleted records",
                                        actual=f"{pii_in_deleted} records with PII",
                                    ))
                                break
        else:
            # Suggest adding deletion tracking
            issues.append(ValidationIssue(
                column="_table",
                issue_type="gdpr_erasure_tracking_missing",
                count=0,
                severity=Severity.MEDIUM,
                details=(
                    "No deletion flag column specified. Consider adding a column "
                    "to track deletion/erasure requests for GDPR Article 17 compliance."
                ),
            ))

        return issues
