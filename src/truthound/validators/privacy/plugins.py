"""Privacy Regulation Plugin System.

This module provides a plugin architecture for extending privacy compliance
validation with custom regulations, regional adaptations, or industry-specific
requirements.

Key Features:
- Register custom privacy regulations
- Define industry-specific PII patterns
- Create jurisdiction-specific validators
- Extend existing regulations with custom rules

Example:
    from truthound.validators.privacy.plugins import (
        PrivacyRegulationPlugin,
        register_privacy_plugin,
        get_privacy_plugin,
    )

    # Define custom regulation plugin
    @register_privacy_plugin("popia")
    class POPIAPlugin(PrivacyRegulationPlugin):
        '''South Africa Protection of Personal Information Act.'''

        regulation_code = "popia"
        regulation_name = "Protection of Personal Information Act"
        jurisdiction = "South Africa"
        effective_date = "2021-07-01"

        def get_pii_definitions(self):
            return [
                PIIFieldDefinition(
                    name="sa_id_number",
                    pattern=re.compile(r"^\\d{13}$"),
                    column_hints=["id_number", "sa_id", "identity"],
                    category=PIICategory.GOVERNMENT_ID,
                ),
                # ... more definitions
            ]

    # Use the plugin
    plugin = get_privacy_plugin("popia")
    validator = plugin.create_validator()
    issues = validator.validate(df.lazy())
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Type
from datetime import date
import re

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.privacy.base import (
    PrivacyRegulation,
    PIICategory,
    PIIFieldDefinition,
    PrivacyFinding,
    PrivacyValidator,
)


# Plugin Registry
_PRIVACY_PLUGINS: dict[str, Type["PrivacyRegulationPlugin"]] = {}


def register_privacy_plugin(
    code: str,
) -> Callable[[Type["PrivacyRegulationPlugin"]], Type["PrivacyRegulationPlugin"]]:
    """Decorator to register a privacy regulation plugin.

    Args:
        code: Unique regulation code (e.g., "popia", "pdpa")

    Returns:
        Decorator function

    Example:
        @register_privacy_plugin("popia")
        class POPIAPlugin(PrivacyRegulationPlugin):
            regulation_code = "popia"
            # ...
    """
    def decorator(cls: Type[PrivacyRegulationPlugin]) -> Type[PrivacyRegulationPlugin]:
        if code in _PRIVACY_PLUGINS:
            raise ValueError(f"Privacy plugin '{code}' is already registered")
        _PRIVACY_PLUGINS[code] = cls
        return cls
    return decorator


def get_privacy_plugin(code: str) -> "PrivacyRegulationPlugin":
    """Get a registered privacy plugin by code.

    Args:
        code: Regulation code

    Returns:
        Plugin instance

    Raises:
        KeyError: If plugin is not registered
    """
    if code not in _PRIVACY_PLUGINS:
        available = list(_PRIVACY_PLUGINS.keys())
        raise KeyError(
            f"Privacy plugin '{code}' not found. Available: {available}"
        )
    return _PRIVACY_PLUGINS[code]()


def list_privacy_plugins() -> list[str]:
    """List all registered privacy plugins.

    Returns:
        List of plugin codes
    """
    return list(_PRIVACY_PLUGINS.keys())


def get_all_privacy_plugins() -> dict[str, "PrivacyRegulationPlugin"]:
    """Get all registered privacy plugins.

    Returns:
        Dict mapping code to plugin instance
    """
    return {code: cls() for code, cls in _PRIVACY_PLUGINS.items()}


@dataclass
class RegulationMetadata:
    """Metadata about a privacy regulation."""

    code: str
    name: str
    jurisdiction: str
    effective_date: date | None = None
    supersedes: list[str] = field(default_factory=list)
    url: str | None = None
    description: str = ""
    key_articles: list[str] = field(default_factory=list)


class PrivacyRegulationPlugin(ABC):
    """Base class for privacy regulation plugins.

    Implement this class to add support for a new privacy regulation.
    The plugin provides:
    - PII field definitions specific to the regulation
    - Compliance validators
    - Regulation metadata

    Example:
        class MyRegulationPlugin(PrivacyRegulationPlugin):
            regulation_code = "my_reg"
            regulation_name = "My Privacy Regulation"
            jurisdiction = "My Country"

            def get_pii_definitions(self) -> list[PIIFieldDefinition]:
                return [...]

            def get_special_category_definitions(self) -> list[PIIFieldDefinition]:
                return [...]
    """

    # Class attributes to be overridden
    regulation_code: str = ""
    regulation_name: str = ""
    jurisdiction: str = ""
    effective_date: str | None = None
    supersedes: list[str] = []
    regulation_url: str | None = None
    description: str = ""
    key_articles: list[str] = []

    @abstractmethod
    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        """Get PII field definitions for this regulation.

        Returns:
            List of PIIFieldDefinition for standard PII types
        """
        pass

    def get_special_category_definitions(self) -> list[PIIFieldDefinition]:
        """Get special/sensitive category definitions.

        Override this for regulations with special data categories
        (e.g., GDPR Article 9, CCPA sensitive PI).

        Returns:
            List of PIIFieldDefinition for sensitive data types
        """
        return []

    def get_all_definitions(self) -> list[PIIFieldDefinition]:
        """Get all PII definitions (standard + special).

        Returns:
            Combined list of all PII definitions
        """
        return self.get_pii_definitions() + self.get_special_category_definitions()

    def get_metadata(self) -> RegulationMetadata:
        """Get regulation metadata.

        Returns:
            RegulationMetadata instance
        """
        effective = None
        if self.effective_date:
            try:
                effective = date.fromisoformat(self.effective_date)
            except ValueError:
                pass

        return RegulationMetadata(
            code=self.regulation_code,
            name=self.regulation_name,
            jurisdiction=self.jurisdiction,
            effective_date=effective,
            supersedes=self.supersedes,
            url=self.regulation_url,
            description=self.description,
            key_articles=self.key_articles,
        )

    def create_validator(self, **kwargs: Any) -> "PluginBasedValidator":
        """Create a validator instance using this plugin.

        Args:
            **kwargs: Additional validator configuration

        Returns:
            PluginBasedValidator instance
        """
        return PluginBasedValidator(plugin=self, **kwargs)

    def validate_compliance(
        self, lf: pl.LazyFrame, **kwargs: Any
    ) -> list[ValidationIssue]:
        """Convenience method to validate compliance.

        Args:
            lf: Input LazyFrame
            **kwargs: Additional validator options

        Returns:
            List of validation issues
        """
        validator = self.create_validator(**kwargs)
        return validator.validate(lf)


class PluginBasedValidator(PrivacyValidator):
    """Validator that uses a privacy regulation plugin.

    This validator dynamically uses PII definitions from a plugin
    rather than hardcoded patterns.

    Example:
        # Using plugin directly
        plugin = get_privacy_plugin("hipaa_healthcare")
        validator = PluginBasedValidator(plugin=plugin)

        # Using regulation code (convenience)
        validator = PluginBasedValidator(regulation_code="hipaa_healthcare")

        # Validate
        issues = validator.validate(df.lazy())
    """

    name = "plugin_based_privacy"

    def __init__(
        self,
        plugin: PrivacyRegulationPlugin | None = None,
        regulation_code: str | None = None,
        include_special_categories: bool = True,
        **kwargs: Any,
    ):
        """Initialize plugin-based validator.

        Args:
            plugin: Privacy regulation plugin to use
            regulation_code: Alternatively, regulation code to look up plugin
            include_special_categories: Include special/sensitive categories
            **kwargs: Additional config passed to PrivacyValidator

        Raises:
            ValueError: If neither plugin nor regulation_code is provided
            KeyError: If regulation_code is not found in registry
        """
        super().__init__(**kwargs)

        # Resolve plugin
        if plugin is not None:
            self._plugin = plugin
        elif regulation_code is not None:
            self._plugin = get_privacy_plugin(regulation_code)
        else:
            raise ValueError(
                "Either 'plugin' or 'regulation_code' must be provided"
            )

        self._include_special_categories = include_special_categories

        # Set regulation from plugin
        try:
            self.regulation = PrivacyRegulation(self._plugin.regulation_code)
        except ValueError:
            # Custom regulation not in enum, use GDPR as base
            self.regulation = PrivacyRegulation.GDPR

    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        """Get PII definitions from the plugin."""
        if self._include_special_categories:
            return self._plugin.get_all_definitions()
        return self._plugin.get_pii_definitions()

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate using plugin definitions.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []
        pii_defs = self.get_pii_definitions()

        if not pii_defs:
            return issues

        df = lf.collect()
        schema = lf.collect_schema()

        # Determine columns to check
        if self.columns:
            columns_to_check = [c for c in self.columns if c in schema.names()]
        else:
            # Check all string-like columns
            columns_to_check = [
                name for name, dtype in schema.items()
                if dtype in (pl.Utf8, pl.String)
            ]

        # Check each column
        all_findings: list[PrivacyFinding] = []

        for column in columns_to_check:
            findings = self._detect_pii_in_column(df, column, pii_defs)
            all_findings.extend(findings)

        # Convert to validation issues
        issues = self._convert_findings_to_issues(all_findings)

        return issues


# =============================================================================
# Built-in Additional Regulation Plugins
# =============================================================================


@register_privacy_plugin("popia")
class POPIAPlugin(PrivacyRegulationPlugin):
    """South Africa Protection of Personal Information Act (POPIA).

    Effective: July 1, 2021
    Jurisdiction: South Africa
    """

    regulation_code = "popia"
    regulation_name = "Protection of Personal Information Act"
    jurisdiction = "South Africa"
    effective_date = "2021-07-01"
    regulation_url = "https://popia.co.za/"
    description = (
        "South African data protection law that regulates the processing "
        "of personal information by public and private bodies."
    )
    key_articles = [
        "Section 9 - Conditions for lawful processing",
        "Section 11 - Consent, justification and objection",
        "Section 26 - Special personal information",
    ]

    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        return [
            PIIFieldDefinition(
                name="South African ID Number",
                pattern=re.compile(r"^\d{13}$"),
                column_hints=["id_number", "sa_id", "identity", "id_no"],
                category=PIICategory.GOVERNMENT_ID,
                requires_consent=True,
                confidence_base=90,
            ),
            PIIFieldDefinition(
                name="South African Passport",
                pattern=re.compile(r"^[A-Z]\d{8}$"),
                column_hints=["passport", "travel_doc"],
                category=PIICategory.GOVERNMENT_ID,
                requires_consent=True,
                confidence_base=85,
            ),
            PIIFieldDefinition(
                name="South African Phone",
                pattern=re.compile(r"^(\+27|0)[1-9]\d{8}$"),
                column_hints=["phone", "mobile", "cell", "contact"],
                category=PIICategory.DIRECT_IDENTIFIER,
                requires_consent=True,
                confidence_base=80,
            ),
        ]

    def get_special_category_definitions(self) -> list[PIIFieldDefinition]:
        return [
            PIIFieldDefinition(
                name="Race or Ethnic Origin",
                pattern=None,
                column_hints=["race", "ethnic", "ethnicity", "population_group"],
                category=PIICategory.RACIAL_ETHNIC,
                is_special_category=True,
                requires_consent=True,
                confidence_base=85,
            ),
            PIIFieldDefinition(
                name="Religious Belief",
                pattern=None,
                column_hints=["religion", "religious", "faith", "belief"],
                category=PIICategory.RELIGIOUS,
                is_special_category=True,
                requires_consent=True,
                confidence_base=85,
            ),
            PIIFieldDefinition(
                name="Health Information",
                pattern=None,
                column_hints=["health", "medical", "diagnosis", "condition", "hiv"],
                category=PIICategory.HEALTH,
                is_special_category=True,
                requires_consent=True,
                confidence_base=85,
            ),
        ]


@register_privacy_plugin("pdpa_thailand")
class PDPAThailandPlugin(PrivacyRegulationPlugin):
    """Thailand Personal Data Protection Act (PDPA).

    Effective: June 1, 2022
    Jurisdiction: Thailand
    """

    regulation_code = "pdpa_thailand"
    regulation_name = "Personal Data Protection Act"
    jurisdiction = "Thailand"
    effective_date = "2022-06-01"
    description = (
        "Thailand's comprehensive data protection law modeled on GDPR, "
        "regulating the collection, use, and disclosure of personal data."
    )
    key_articles = [
        "Section 19 - Collection of personal data",
        "Section 24 - Consent",
        "Section 26 - Sensitive personal data",
    ]

    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        return [
            PIIFieldDefinition(
                name="Thai National ID",
                pattern=re.compile(r"^\d{13}$"),
                column_hints=["national_id", "id_card", "citizen_id", "thai_id"],
                category=PIICategory.GOVERNMENT_ID,
                requires_consent=True,
                confidence_base=90,
            ),
            PIIFieldDefinition(
                name="Thai Passport",
                pattern=re.compile(r"^[A-Z]{2}\d{7}$"),
                column_hints=["passport", "passport_no"],
                category=PIICategory.GOVERNMENT_ID,
                requires_consent=True,
                confidence_base=85,
            ),
            PIIFieldDefinition(
                name="Thai Phone Number",
                pattern=re.compile(r"^(\+66|0)[689]\d{8}$"),
                column_hints=["phone", "mobile", "tel"],
                category=PIICategory.DIRECT_IDENTIFIER,
                requires_consent=True,
                confidence_base=80,
            ),
        ]

    def get_special_category_definitions(self) -> list[PIIFieldDefinition]:
        return [
            PIIFieldDefinition(
                name="Racial/Ethnic Origin",
                pattern=None,
                column_hints=["race", "ethnic", "nationality"],
                category=PIICategory.RACIAL_ETHNIC,
                is_special_category=True,
                requires_consent=True,
                confidence_base=85,
            ),
            PIIFieldDefinition(
                name="Criminal Record",
                pattern=None,
                column_hints=["criminal", "conviction", "offense"],
                category=PIICategory.CRIMINAL,
                is_special_category=True,
                requires_consent=True,
                confidence_base=85,
            ),
        ]


@register_privacy_plugin("pdpb_india")
class PDPBIndiaPlugin(PrivacyRegulationPlugin):
    """India Digital Personal Data Protection Bill (PDPB).

    Status: Passed in 2023
    Jurisdiction: India
    """

    regulation_code = "pdpb_india"
    regulation_name = "Digital Personal Data Protection Act"
    jurisdiction = "India"
    effective_date = "2024-01-01"  # Approximate
    description = (
        "India's comprehensive data protection law establishing principles "
        "for processing digital personal data."
    )
    key_articles = [
        "Section 4 - Grounds for processing",
        "Section 6 - Consent",
        "Section 7 - Certain legitimate uses",
    ]

    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        return [
            PIIFieldDefinition(
                name="Aadhaar Number",
                pattern=re.compile(r"^\d{12}$"),
                column_hints=["aadhaar", "aadhar", "uid", "uidai"],
                category=PIICategory.GOVERNMENT_ID,
                requires_consent=True,
                confidence_base=95,
            ),
            PIIFieldDefinition(
                name="PAN Number",
                pattern=re.compile(r"^[A-Z]{5}\d{4}[A-Z]$"),
                column_hints=["pan", "pan_number", "pan_no", "tax_id"],
                category=PIICategory.GOVERNMENT_ID,
                requires_consent=True,
                confidence_base=90,
            ),
            PIIFieldDefinition(
                name="Indian Passport",
                pattern=re.compile(r"^[A-Z]\d{7}$"),
                column_hints=["passport", "passport_no"],
                category=PIICategory.GOVERNMENT_ID,
                requires_consent=True,
                confidence_base=85,
            ),
            PIIFieldDefinition(
                name="Indian Phone",
                pattern=re.compile(r"^(\+91|0)?[6-9]\d{9}$"),
                column_hints=["phone", "mobile", "contact"],
                category=PIICategory.DIRECT_IDENTIFIER,
                requires_consent=True,
                confidence_base=80,
            ),
        ]

    def get_special_category_definitions(self) -> list[PIIFieldDefinition]:
        return [
            PIIFieldDefinition(
                name="Health Data",
                pattern=None,
                column_hints=["health", "medical", "diagnosis", "prescription"],
                category=PIICategory.HEALTH,
                is_special_category=True,
                requires_consent=True,
                confidence_base=85,
            ),
            PIIFieldDefinition(
                name="Biometric Data",
                pattern=None,
                column_hints=["fingerprint", "biometric", "iris", "face_id"],
                category=PIICategory.BIOMETRIC,
                is_special_category=True,
                requires_consent=True,
                confidence_base=90,
            ),
        ]


@register_privacy_plugin("kvkk_turkey")
class KVKKTurkeyPlugin(PrivacyRegulationPlugin):
    """Turkey Personal Data Protection Law (KVKK).

    Effective: April 7, 2016
    Jurisdiction: Turkey
    """

    regulation_code = "kvkk_turkey"
    regulation_name = "Kisisel Verilerin Korunmasi Kanunu"
    jurisdiction = "Turkey"
    effective_date = "2016-04-07"
    description = (
        "Turkey's personal data protection law establishing rules for "
        "processing personal data of natural persons."
    )
    key_articles = [
        "Article 5 - Conditions for processing personal data",
        "Article 6 - Special categories of personal data",
        "Article 7 - Erasure, destruction, or anonymization",
    ]

    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        return [
            PIIFieldDefinition(
                name="Turkish ID Number (TC Kimlik)",
                pattern=re.compile(r"^\d{11}$"),
                column_hints=["tc_kimlik", "kimlik_no", "tc_no", "identity"],
                category=PIICategory.GOVERNMENT_ID,
                requires_consent=True,
                confidence_base=90,
            ),
            PIIFieldDefinition(
                name="Turkish Phone",
                pattern=re.compile(r"^(\+90|0)?5\d{9}$"),
                column_hints=["phone", "mobile", "telefon", "cep"],
                category=PIICategory.DIRECT_IDENTIFIER,
                requires_consent=True,
                confidence_base=80,
            ),
        ]

    def get_special_category_definitions(self) -> list[PIIFieldDefinition]:
        return [
            PIIFieldDefinition(
                name="Health Data",
                pattern=None,
                column_hints=["saglik", "health", "medical", "hastane"],
                category=PIICategory.HEALTH,
                is_special_category=True,
                requires_consent=True,
                confidence_base=85,
            ),
            PIIFieldDefinition(
                name="Religious Belief",
                pattern=None,
                column_hints=["din", "religion", "mezhep", "inanc"],
                category=PIICategory.RELIGIOUS,
                is_special_category=True,
                requires_consent=True,
                confidence_base=85,
            ),
        ]


# =============================================================================
# Industry-Specific Plugins
# =============================================================================


@register_privacy_plugin("hipaa_healthcare")
class HIPAAHealthcarePlugin(PrivacyRegulationPlugin):
    """US HIPAA for Healthcare Industry.

    Focus on Protected Health Information (PHI).
    """

    regulation_code = "hipaa_healthcare"
    regulation_name = "Health Insurance Portability and Accountability Act"
    jurisdiction = "United States"
    effective_date = "1996-08-21"
    description = (
        "US federal law that protects sensitive patient health information "
        "from being disclosed without consent."
    )
    key_articles = [
        "164.502 - Uses and disclosures of protected health information",
        "164.508 - Uses and disclosures for which authorization is required",
        "164.512 - Uses and disclosures for which authorization is not required",
    ]

    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        return [
            PIIFieldDefinition(
                name="Medical Record Number",
                pattern=re.compile(r"^MRN\d{6,10}$"),
                column_hints=["mrn", "medical_record", "patient_id", "chart_no"],
                category=PIICategory.HEALTH,
                requires_consent=True,
                confidence_base=95,
            ),
            PIIFieldDefinition(
                name="Health Plan ID",
                pattern=re.compile(r"^\d{9,11}$"),
                column_hints=["health_plan_id", "insurance_id", "member_id", "policy_no"],
                category=PIICategory.HEALTH,
                requires_consent=True,
                confidence_base=85,
            ),
            PIIFieldDefinition(
                name="Social Security Number",
                pattern=re.compile(r"^\d{3}-?\d{2}-?\d{4}$"),
                column_hints=["ssn", "social_security", "ss_number"],
                category=PIICategory.GOVERNMENT_ID,
                requires_consent=True,
                confidence_base=95,
            ),
        ]

    def get_special_category_definitions(self) -> list[PIIFieldDefinition]:
        """PHI-specific sensitive data."""
        return [
            PIIFieldDefinition(
                name="Diagnosis Code (ICD)",
                pattern=re.compile(r"^[A-Z]\d{2}(\.\d{1,4})?$"),
                column_hints=["icd_code", "diagnosis_code", "dx_code", "icd10"],
                category=PIICategory.HEALTH,
                is_special_category=True,
                requires_consent=True,
                confidence_base=90,
            ),
            PIIFieldDefinition(
                name="Procedure Code (CPT)",
                pattern=re.compile(r"^\d{5}$"),
                column_hints=["cpt_code", "procedure_code", "billing_code"],
                category=PIICategory.HEALTH,
                is_special_category=True,
                requires_consent=True,
                confidence_base=85,
            ),
            PIIFieldDefinition(
                name="Prescription Information",
                pattern=None,
                column_hints=["prescription", "medication", "rx", "drug_name", "dosage"],
                category=PIICategory.HEALTH,
                is_special_category=True,
                requires_consent=True,
                confidence_base=85,
            ),
            PIIFieldDefinition(
                name="Lab Results",
                pattern=None,
                column_hints=["lab_result", "test_result", "blood_test", "diagnosis"],
                category=PIICategory.HEALTH,
                is_special_category=True,
                requires_consent=True,
                confidence_base=85,
            ),
            PIIFieldDefinition(
                name="Mental Health Information",
                pattern=None,
                column_hints=["mental_health", "psychiatric", "therapy", "counseling"],
                category=PIICategory.HEALTH,
                is_special_category=True,
                requires_consent=True,
                confidence_base=90,
            ),
        ]


@register_privacy_plugin("pci_dss_financial")
class PCIDSSFinancialPlugin(PrivacyRegulationPlugin):
    """PCI-DSS for Financial/Payment Industry.

    Focus on cardholder data and payment information.
    """

    regulation_code = "pci_dss_financial"
    regulation_name = "Payment Card Industry Data Security Standard"
    jurisdiction = "Global"
    effective_date = "2004-12-15"
    description = (
        "Information security standard for organizations handling payment cards. "
        "Focused on protecting cardholder data."
    )
    key_articles = [
        "Requirement 3 - Protect stored cardholder data",
        "Requirement 4 - Encrypt transmission of cardholder data",
        "Requirement 7 - Restrict access to cardholder data",
    ]

    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        return [
            PIIFieldDefinition(
                name="Primary Account Number (PAN)",
                pattern=re.compile(r"^(?:\d{4}[-\s]?){3}\d{4}$|^\d{13,19}$"),
                column_hints=[
                    "card_number", "pan", "credit_card", "cc_number",
                    "account_number", "card_no"
                ],
                category=PIICategory.FINANCIAL,
                requires_consent=True,
                confidence_base=95,
            ),
            PIIFieldDefinition(
                name="Card Expiry Date",
                pattern=re.compile(r"^(0[1-9]|1[0-2])/?([0-9]{2}|[0-9]{4})$"),
                column_hints=["expiry", "expiration", "exp_date", "card_exp"],
                category=PIICategory.FINANCIAL,
                requires_consent=True,
                confidence_base=85,
            ),
            PIIFieldDefinition(
                name="CVV/CVC",
                pattern=re.compile(r"^\d{3,4}$"),
                column_hints=["cvv", "cvc", "security_code", "card_code"],
                category=PIICategory.FINANCIAL,
                requires_consent=True,
                confidence_base=90,
            ),
            PIIFieldDefinition(
                name="Bank Account Number",
                pattern=re.compile(r"^\d{8,17}$"),
                column_hints=["bank_account", "account_no", "acct_number"],
                category=PIICategory.FINANCIAL,
                requires_consent=True,
                confidence_base=80,
            ),
            PIIFieldDefinition(
                name="Routing Number",
                pattern=re.compile(r"^\d{9}$"),
                column_hints=["routing", "aba_number", "routing_no"],
                category=PIICategory.FINANCIAL,
                requires_consent=True,
                confidence_base=80,
            ),
        ]

    def get_special_category_definitions(self) -> list[PIIFieldDefinition]:
        return [
            PIIFieldDefinition(
                name="PIN Block",
                pattern=None,
                column_hints=["pin", "pin_block", "encrypted_pin"],
                category=PIICategory.FINANCIAL,
                is_special_category=True,
                requires_consent=True,
                confidence_base=95,
            ),
            PIIFieldDefinition(
                name="Magnetic Stripe Data",
                pattern=None,
                column_hints=["track_data", "magnetic_stripe", "track1", "track2"],
                category=PIICategory.FINANCIAL,
                is_special_category=True,
                requires_consent=True,
                confidence_base=95,
            ),
        ]
