"""Base classes for privacy compliance validators.

This module provides extensible base classes for implementing
GDPR, CCPA, and other privacy regulation compliance validators.

Privacy Regulations Supported:
    - GDPR (General Data Protection Regulation) - EU
    - CCPA (California Consumer Privacy Act) - US/California
    - LGPD (Lei Geral de Proteção de Dados) - Brazil
    - PIPEDA (Personal Information Protection) - Canada
    - APPI (Act on Protection of Personal Information) - Japan
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import re

import polars as pl

from truthound.validators.base import (
    Validator,
    ValidationIssue,
    ValidatorConfig,
    StringValidatorMixin,
)
from truthound.types import Severity


class PrivacyRegulation(str, Enum):
    """Supported privacy regulations."""

    GDPR = "gdpr"           # EU General Data Protection Regulation
    CCPA = "ccpa"           # California Consumer Privacy Act
    LGPD = "lgpd"           # Brazil Lei Geral de Proteção de Dados
    PIPEDA = "pipeda"       # Canada Personal Information Protection
    APPI = "appi"           # Japan Act on Protection of Personal Information
    HIPAA = "hipaa"         # US Health Insurance Portability and Accountability


class PIICategory(str, Enum):
    """Categories of personally identifiable information per GDPR Article 9."""

    # Standard PII
    DIRECT_IDENTIFIER = "direct_identifier"       # Name, email, phone
    INDIRECT_IDENTIFIER = "indirect_identifier"   # IP, device ID, cookie
    FINANCIAL = "financial"                       # Credit card, bank account
    GOVERNMENT_ID = "government_id"               # SSN, passport, national ID

    # Special Categories (GDPR Article 9 - requires explicit consent)
    RACIAL_ETHNIC = "racial_ethnic"               # Racial or ethnic origin
    POLITICAL = "political"                       # Political opinions
    RELIGIOUS = "religious"                       # Religious or philosophical beliefs
    TRADE_UNION = "trade_union"                   # Trade union membership
    GENETIC = "genetic"                           # Genetic data
    BIOMETRIC = "biometric"                       # Biometric data
    HEALTH = "health"                             # Health data
    SEX_LIFE = "sex_life"                         # Sex life or sexual orientation
    CRIMINAL = "criminal"                         # Criminal convictions


class ConsentStatus(str, Enum):
    """Data consent status for GDPR compliance."""

    EXPLICIT = "explicit"           # Explicit opt-in consent
    IMPLICIT = "implicit"           # Implied consent (may not be GDPR compliant)
    WITHDRAWN = "withdrawn"         # Consent withdrawn
    NOT_REQUIRED = "not_required"   # Consent not required (legitimate interest)
    UNKNOWN = "unknown"             # Consent status unknown


class LegalBasis(str, Enum):
    """GDPR Article 6 legal basis for processing."""

    CONSENT = "consent"                     # Data subject consent
    CONTRACT = "contract"                   # Contract performance
    LEGAL_OBLIGATION = "legal_obligation"   # Legal obligation
    VITAL_INTERESTS = "vital_interests"     # Protect vital interests
    PUBLIC_TASK = "public_task"             # Public interest task
    LEGITIMATE_INTEREST = "legitimate_interest"  # Legitimate interests


@dataclass
class PIIFieldDefinition:
    """Definition of a PII field pattern."""

    name: str
    pattern: re.Pattern | None = None
    column_hints: list[str] = field(default_factory=list)
    category: PIICategory = PIICategory.DIRECT_IDENTIFIER
    regulations: list[PrivacyRegulation] = field(default_factory=list)
    requires_consent: bool = True
    is_special_category: bool = False  # GDPR Article 9
    retention_sensitive: bool = True
    confidence_base: int = 85
    description: str = ""

    def matches_column_name(self, column_name: str) -> bool:
        """Check if column name matches hints."""
        col_lower = column_name.lower().replace("_", " ").replace("-", " ")
        return any(hint.lower() in col_lower for hint in self.column_hints)

    def matches_value(self, value: str) -> bool:
        """Check if value matches pattern."""
        if self.pattern is None:
            return False
        return bool(self.pattern.match(value))


@dataclass
class PrivacyFinding:
    """Represents a privacy compliance finding."""

    column: str
    pii_type: str
    category: PIICategory
    regulation: PrivacyRegulation
    violation_type: str
    count: int
    confidence: int
    severity: Severity
    recommendation: str
    legal_basis_required: bool = True
    requires_consent: bool = True
    sample_values: list[Any] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "column": self.column,
            "pii_type": self.pii_type,
            "category": self.category.value,
            "regulation": self.regulation.value,
            "violation_type": self.violation_type,
            "count": self.count,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "recommendation": self.recommendation,
            "legal_basis_required": self.legal_basis_required,
            "requires_consent": self.requires_consent,
            "sample_values": self.sample_values,
        }


class PrivacyValidator(Validator, StringValidatorMixin):
    """Base class for privacy compliance validators.

    Provides common functionality for detecting PII and
    validating compliance with privacy regulations.

    Subclasses should implement:
        - get_pii_definitions(): Return list of PII patterns to check
        - validate(): Full validation logic
    """

    category = "privacy"
    regulation: PrivacyRegulation = PrivacyRegulation.GDPR

    def __init__(
        self,
        columns: list[str] | None = None,
        sample_size: int = 1000,
        min_confidence: int = 70,
        detect_special_categories: bool = True,
        check_retention: bool = False,
        retention_days: int | None = None,
        date_column: str | None = None,
        **kwargs: Any,
    ):
        """Initialize privacy validator.

        Args:
            columns: Specific columns to check (None = all string columns)
            sample_size: Number of rows to sample for pattern detection
            min_confidence: Minimum confidence threshold for reporting
            detect_special_categories: Whether to detect GDPR Article 9 data
            check_retention: Whether to check data retention compliance
            retention_days: Maximum retention period in days
            date_column: Column containing record date for retention check
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.columns = columns
        self.sample_size = sample_size
        self.min_confidence = min_confidence
        self.detect_special_categories = detect_special_categories
        self.check_retention = check_retention
        self.retention_days = retention_days
        self.date_column = date_column

    @abstractmethod
    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        """Get PII field definitions for this regulation.

        Returns:
            List of PIIFieldDefinition objects
        """
        pass

    def _detect_pii_in_column(
        self,
        df: pl.DataFrame,
        column: str,
        pii_defs: list[PIIFieldDefinition],
    ) -> list[PrivacyFinding]:
        """Detect PII in a single column.

        Args:
            df: Input DataFrame
            column: Column to check
            pii_defs: PII definitions to check against

        Returns:
            List of privacy findings
        """
        findings: list[PrivacyFinding] = []

        col_data = df.get_column(column).drop_nulls()
        if len(col_data) == 0:
            return findings

        # Sample for performance
        sample = col_data.head(min(len(col_data), self.sample_size))
        sample_values = [str(v) for v in sample.to_list() if v is not None]

        for pii_def in pii_defs:
            # Skip special categories if not requested
            if pii_def.is_special_category and not self.detect_special_categories:
                continue

            # Calculate confidence
            confidence = self._calculate_confidence(
                column, sample_values, pii_def
            )

            if confidence < self.min_confidence:
                continue

            # Count matches
            match_count = sum(
                1 for v in sample_values
                if pii_def.matches_value(v)
            )

            if match_count == 0 and not pii_def.matches_column_name(column):
                continue

            # Extrapolate to full dataset
            match_ratio = match_count / len(sample_values) if sample_values else 0
            estimated_count = int(len(col_data) * match_ratio) if match_ratio > 0 else 0

            if estimated_count == 0 and not pii_def.matches_column_name(column):
                continue

            # Determine severity based on category
            severity = self._get_severity_for_category(pii_def.category)

            # Create finding
            finding = PrivacyFinding(
                column=column,
                pii_type=pii_def.name,
                category=pii_def.category,
                regulation=self.regulation,
                violation_type=f"potential_{pii_def.category.value}_detected",
                count=estimated_count or len(col_data),
                confidence=confidence,
                severity=severity,
                recommendation=self._get_recommendation(pii_def),
                legal_basis_required=True,
                requires_consent=pii_def.requires_consent,
                sample_values=sample_values[:3] if sample_values else None,
            )
            findings.append(finding)

        return findings

    def _calculate_confidence(
        self,
        column: str,
        values: list[str],
        pii_def: PIIFieldDefinition,
    ) -> int:
        """Calculate confidence score for PII detection.

        Args:
            column: Column name
            values: Sample values
            pii_def: PII definition

        Returns:
            Confidence score (0-100)
        """
        confidence = pii_def.confidence_base

        # Boost for column name match
        if pii_def.matches_column_name(column):
            confidence = min(99, confidence + 15)

        # Adjust based on pattern match ratio
        if values and pii_def.pattern:
            match_count = sum(1 for v in values if pii_def.matches_value(v))
            match_ratio = match_count / len(values)

            if match_ratio > 0.8:
                confidence = min(99, confidence + 10)
            elif match_ratio > 0.5:
                confidence = min(99, confidence + 5)
            elif match_ratio < 0.1:
                confidence = max(50, confidence - 20)

        return confidence

    def _get_severity_for_category(self, category: PIICategory) -> Severity:
        """Get severity level for PII category.

        Args:
            category: PII category

        Returns:
            Appropriate severity level
        """
        # Special categories (GDPR Article 9) are always critical
        special_categories = {
            PIICategory.RACIAL_ETHNIC,
            PIICategory.POLITICAL,
            PIICategory.RELIGIOUS,
            PIICategory.TRADE_UNION,
            PIICategory.GENETIC,
            PIICategory.BIOMETRIC,
            PIICategory.HEALTH,
            PIICategory.SEX_LIFE,
            PIICategory.CRIMINAL,
        }

        if category in special_categories:
            return Severity.CRITICAL

        # Government IDs and financial data are high severity
        if category in {PIICategory.GOVERNMENT_ID, PIICategory.FINANCIAL}:
            return Severity.HIGH

        # Direct identifiers are medium-high
        if category == PIICategory.DIRECT_IDENTIFIER:
            return Severity.MEDIUM

        return Severity.LOW

    def _get_recommendation(self, pii_def: PIIFieldDefinition) -> str:
        """Get compliance recommendation for PII finding.

        Args:
            pii_def: PII definition

        Returns:
            Recommendation string
        """
        if pii_def.is_special_category:
            return (
                f"CRITICAL: {pii_def.name} is a special category under GDPR Article 9. "
                "Explicit consent or specific legal basis required. "
                "Consider data minimization or pseudonymization."
            )

        if pii_def.category == PIICategory.GOVERNMENT_ID:
            return (
                f"HIGH: {pii_def.name} detected. "
                "Ensure proper encryption, access controls, and documented legal basis. "
                "Consider masking or tokenization."
            )

        if pii_def.category == PIICategory.FINANCIAL:
            return (
                f"HIGH: {pii_def.name} detected. "
                "Apply PCI-DSS compliant handling. "
                "Use encryption and limit access."
            )

        return (
            f"MEDIUM: {pii_def.name} detected. "
            "Document legal basis for processing and ensure appropriate consent."
        )

    def _convert_findings_to_issues(
        self,
        findings: list[PrivacyFinding],
    ) -> list[ValidationIssue]:
        """Convert privacy findings to validation issues.

        Args:
            findings: List of privacy findings

        Returns:
            List of ValidationIssue objects
        """
        issues: list[ValidationIssue] = []

        for finding in findings:
            issue = ValidationIssue(
                column=finding.column,
                issue_type=f"{self.regulation.value}_{finding.violation_type}",
                count=finding.count,
                severity=finding.severity,
                details=(
                    f"{finding.pii_type} ({finding.category.value}) detected with "
                    f"{finding.confidence}% confidence. {finding.recommendation}"
                ),
                expected=f"No unprotected {finding.pii_type}",
                actual=f"Found {finding.count} potential instances",
                sample_values=finding.sample_values,
            )
            issues.append(issue)

        return issues


class DataRetentionValidator(PrivacyValidator):
    """Validates data retention compliance.

    Checks that personal data is not retained beyond the specified
    retention period as required by GDPR Article 5(1)(e).
    """

    name = "data_retention"

    def __init__(
        self,
        date_column: str,
        retention_days: int,
        pii_columns: list[str] | None = None,
        **kwargs: Any,
    ):
        """Initialize retention validator.

        Args:
            date_column: Column containing record creation/update date
            retention_days: Maximum retention period in days
            pii_columns: Columns containing PII to check
            **kwargs: Additional config
        """
        super().__init__(
            check_retention=True,
            retention_days=retention_days,
            date_column=date_column,
            **kwargs,
        )
        self.pii_columns = pii_columns

    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        """Return empty list - retention validator doesn't detect PII types."""
        return []

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate data retention compliance.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        schema = lf.collect_schema()

        # Check if date column exists
        if self.date_column not in schema.names():
            return [ValidationIssue(
                column=self.date_column,
                issue_type=f"{self.regulation.value}_retention_check_failed",
                count=0,
                severity=Severity.HIGH,
                details=f"Date column '{self.date_column}' not found for retention check",
            )]

        # Calculate retention threshold
        from datetime import datetime, timedelta
        threshold = datetime.now() - timedelta(days=self.retention_days)

        # Count records beyond retention
        df = lf.collect()

        try:
            # Handle different date formats
            date_col = df[self.date_column]

            if date_col.dtype in (pl.Date, pl.Datetime):
                expired_count = df.filter(
                    pl.col(self.date_column) < threshold
                ).height
            else:
                # Try to parse as string
                expired_count = df.filter(
                    pl.col(self.date_column).str.to_datetime() < threshold
                ).height
        except Exception:
            return [ValidationIssue(
                column=self.date_column,
                issue_type=f"{self.regulation.value}_retention_check_failed",
                count=0,
                severity=Severity.MEDIUM,
                details=f"Could not parse date column '{self.date_column}'",
            )]

        if expired_count > 0:
            total_rows = df.height
            ratio = expired_count / total_rows if total_rows > 0 else 0

            issues.append(ValidationIssue(
                column=self.date_column,
                issue_type=f"{self.regulation.value}_retention_exceeded",
                count=expired_count,
                severity=Severity.HIGH if ratio > 0.1 else Severity.MEDIUM,
                details=(
                    f"Found {expired_count} records ({ratio:.1%}) exceeding "
                    f"{self.retention_days}-day retention period. "
                    "Consider implementing automated data purging."
                ),
                expected=f"All records within {self.retention_days} days",
                actual=f"{expired_count} records exceed retention period",
            ))

        return issues


class ConsentValidator(PrivacyValidator):
    """Validates consent tracking compliance.

    Checks that proper consent records exist for PII processing
    as required by GDPR Article 7.
    """

    name = "consent_tracking"

    def __init__(
        self,
        consent_column: str,
        pii_columns: list[str],
        valid_consent_values: list[str] | None = None,
        require_explicit: bool = True,
        **kwargs: Any,
    ):
        """Initialize consent validator.

        Args:
            consent_column: Column containing consent status
            pii_columns: Columns containing PII that require consent
            valid_consent_values: Values indicating valid consent
            require_explicit: Whether explicit consent is required
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.consent_column = consent_column
        self.pii_columns = pii_columns
        self.valid_consent_values = valid_consent_values or [
            "yes", "true", "1", "explicit", "granted", "accepted"
        ]
        self.require_explicit = require_explicit

    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        """Return empty list - consent validator uses explicit columns."""
        return []

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate consent tracking compliance.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        schema = lf.collect_schema()

        # Check if consent column exists
        if self.consent_column not in schema.names():
            return [ValidationIssue(
                column=self.consent_column,
                issue_type=f"{self.regulation.value}_consent_column_missing",
                count=0,
                severity=Severity.CRITICAL,
                details=(
                    f"Consent tracking column '{self.consent_column}' not found. "
                    "GDPR requires documented consent for PII processing."
                ),
            )]

        df = lf.collect()
        total_rows = df.height

        if total_rows == 0:
            return issues

        # Check for missing consent
        valid_values = [v.lower() for v in self.valid_consent_values]

        # Count records with PII but without valid consent
        for pii_col in self.pii_columns:
            if pii_col not in schema.names():
                continue

            # Records with PII data
            has_pii = df.filter(pl.col(pii_col).is_not_null())

            if has_pii.height == 0:
                continue

            # Records with PII but invalid/missing consent
            missing_consent = has_pii.filter(
                pl.col(self.consent_column).is_null() |
                ~pl.col(self.consent_column).cast(pl.Utf8).str.to_lowercase().is_in(valid_values)
            )

            missing_count = missing_consent.height

            if missing_count > 0:
                ratio = missing_count / has_pii.height

                issues.append(ValidationIssue(
                    column=pii_col,
                    issue_type=f"{self.regulation.value}_consent_missing",
                    count=missing_count,
                    severity=Severity.CRITICAL if ratio > 0.1 else Severity.HIGH,
                    details=(
                        f"Found {missing_count} records ({ratio:.1%}) with PII in "
                        f"'{pii_col}' but without valid consent in '{self.consent_column}'. "
                        "GDPR Article 7 requires demonstrable consent."
                    ),
                    expected="Valid consent for all PII records",
                    actual=f"{missing_count} records lack consent",
                ))

        return issues
