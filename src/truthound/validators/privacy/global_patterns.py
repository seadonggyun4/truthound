"""Global privacy patterns for multiple jurisdictions.

This module provides PII patterns for international privacy regulations:
- LGPD (Brazil)
- PIPEDA (Canada)
- APPI (Japan)
- PDPA (Singapore, Thailand)
- POPIA (South Africa)

Each jurisdiction has specific PII definitions and requirements.
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


# Brazil LGPD patterns
LGPD_PII_DEFINITIONS: list[PIIFieldDefinition] = [
    PIIFieldDefinition(
        name="CPF (Brazilian ID)",
        pattern=re.compile(r"^\d{3}\.?\d{3}\.?\d{3}-?\d{2}$"),
        column_hints=["cpf", "cpf_number", "cadastro_pessoa_fisica"],
        category=PIICategory.GOVERNMENT_ID,
        regulations=[PrivacyRegulation.LGPD],
        requires_consent=True,
        confidence_base=95,
        description="Brazilian individual taxpayer ID (CPF)",
    ),
    PIIFieldDefinition(
        name="CNPJ (Brazilian Business ID)",
        pattern=re.compile(r"^\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}$"),
        column_hints=["cnpj", "cnpj_number", "cadastro_empresa"],
        category=PIICategory.GOVERNMENT_ID,
        regulations=[PrivacyRegulation.LGPD],
        requires_consent=True,
        confidence_base=95,
        description="Brazilian business taxpayer ID (CNPJ)",
    ),
    PIIFieldDefinition(
        name="RG (Brazilian ID Card)",
        pattern=re.compile(r"^\d{1,2}\.?\d{3}\.?\d{3}-?[0-9Xx]$"),
        column_hints=["rg", "identidade", "registro_geral"],
        category=PIICategory.GOVERNMENT_ID,
        regulations=[PrivacyRegulation.LGPD],
        requires_consent=True,
        confidence_base=85,
        description="Brazilian identity card number (RG)",
    ),
    PIIFieldDefinition(
        name="Brazilian Phone",
        pattern=re.compile(r"^\+?55\s?\(?\d{2}\)?\s?\d{4,5}-?\d{4}$"),
        column_hints=["telefone", "celular", "phone", "mobile"],
        category=PIICategory.DIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.LGPD],
        requires_consent=True,
        confidence_base=90,
        description="Brazilian phone number with country code",
    ),
    PIIFieldDefinition(
        name="CEP (Brazilian Postal Code)",
        pattern=re.compile(r"^\d{5}-?\d{3}$"),
        column_hints=["cep", "codigo_postal", "postal_code"],
        category=PIICategory.DIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.LGPD],
        requires_consent=True,
        confidence_base=85,
        description="Brazilian postal code (CEP)",
    ),
]

# Canada PIPEDA patterns
PIPEDA_PII_DEFINITIONS: list[PIIFieldDefinition] = [
    PIIFieldDefinition(
        name="SIN (Social Insurance Number)",
        pattern=re.compile(r"^\d{3}[\s-]?\d{3}[\s-]?\d{3}$"),
        column_hints=["sin", "social_insurance", "nas", "numero_assurance"],
        category=PIICategory.GOVERNMENT_ID,
        regulations=[PrivacyRegulation.PIPEDA],
        requires_consent=True,
        confidence_base=95,
        description="Canadian Social Insurance Number",
    ),
    PIIFieldDefinition(
        name="Canadian Driver's License",
        pattern=re.compile(r"^[A-Z]\d{4}-\d{5}-\d{5}$|^[A-Z]{2}\d{6}$"),
        column_hints=["drivers_license", "permis_conduire", "dl_number"],
        category=PIICategory.GOVERNMENT_ID,
        regulations=[PrivacyRegulation.PIPEDA],
        requires_consent=True,
        confidence_base=85,
        description="Canadian driver's license number",
    ),
    PIIFieldDefinition(
        name="Canadian Health Card",
        pattern=re.compile(r"^\d{10}[A-Z]{2}$|^\d{4}-?\d{3}-?\d{3}$"),
        column_hints=["health_card", "ohip", "carte_sante", "ramq"],
        category=PIICategory.HEALTH,
        regulations=[PrivacyRegulation.PIPEDA],
        requires_consent=True,
        is_special_category=True,
        confidence_base=90,
        description="Canadian provincial health card number",
    ),
    PIIFieldDefinition(
        name="Canadian Postal Code",
        pattern=re.compile(r"^[A-Z]\d[A-Z]\s?\d[A-Z]\d$"),
        column_hints=["postal_code", "code_postal", "zip"],
        category=PIICategory.DIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.PIPEDA],
        requires_consent=True,
        confidence_base=90,
        description="Canadian postal code (FSA LDU format)",
    ),
    PIIFieldDefinition(
        name="Canadian Phone",
        pattern=re.compile(r"^\+?1?\s?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$"),
        column_hints=["phone", "telephone", "mobile", "cell"],
        category=PIICategory.DIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.PIPEDA],
        requires_consent=True,
        confidence_base=85,
        description="Canadian phone number",
    ),
]

# Japan APPI patterns
APPI_PII_DEFINITIONS: list[PIIFieldDefinition] = [
    PIIFieldDefinition(
        name="My Number (Individual Number)",
        pattern=re.compile(r"^\d{12}$"),
        column_hints=["my_number", "マイナンバー", "kojin_bango", "individual_number"],
        category=PIICategory.GOVERNMENT_ID,
        regulations=[PrivacyRegulation.APPI],
        requires_consent=True,
        is_special_category=True,
        confidence_base=90,
        description="Japanese Individual Number (My Number)",
    ),
    PIIFieldDefinition(
        name="Japanese Postal Code",
        pattern=re.compile(r"^\d{3}-?\d{4}$"),
        column_hints=["postal_code", "郵便番号", "yubin_bango"],
        category=PIICategory.DIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.APPI],
        requires_consent=True,
        confidence_base=85,
        description="Japanese postal code (7 digits)",
    ),
    PIIFieldDefinition(
        name="Japanese Phone",
        pattern=re.compile(r"^0\d{1,4}-?\d{1,4}-?\d{4}$"),
        column_hints=["phone", "電話番号", "denwa", "tel"],
        category=PIICategory.DIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.APPI],
        requires_consent=True,
        confidence_base=85,
        description="Japanese phone number",
    ),
    PIIFieldDefinition(
        name="Japanese Driver's License",
        pattern=re.compile(r"^\d{12}$"),
        column_hints=["drivers_license", "運転免許", "unten_menkyo"],
        category=PIICategory.GOVERNMENT_ID,
        regulations=[PrivacyRegulation.APPI],
        requires_consent=True,
        confidence_base=80,
        description="Japanese driver's license number",
    ),
    PIIFieldDefinition(
        name="Japanese Name (Kanji)",
        pattern=re.compile(r"^[\u4e00-\u9faf\u3400-\u4dbf]{1,4}\s*[\u4e00-\u9faf\u3400-\u4dbf]{1,4}$"),
        column_hints=["name", "氏名", "shimei", "名前", "namae"],
        category=PIICategory.DIRECT_IDENTIFIER,
        regulations=[PrivacyRegulation.APPI],
        requires_consent=True,
        confidence_base=70,
        description="Japanese name in Kanji characters",
    ),
]

# Global common patterns (applicable across jurisdictions)
GLOBAL_PII_DEFINITIONS: list[PIIFieldDefinition] = [
    PIIFieldDefinition(
        name="Email Address",
        pattern=re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
        column_hints=["email", "e-mail", "mail", "correo", "電子メール", "メール"],
        category=PIICategory.DIRECT_IDENTIFIER,
        regulations=[
            PrivacyRegulation.GDPR, PrivacyRegulation.CCPA,
            PrivacyRegulation.LGPD, PrivacyRegulation.PIPEDA,
            PrivacyRegulation.APPI,
        ],
        requires_consent=True,
        confidence_base=95,
        description="Email address (universal PII)",
    ),
    PIIFieldDefinition(
        name="International Phone (E.164)",
        pattern=re.compile(r"^\+[1-9]\d{6,14}$"),
        column_hints=["phone", "telephone", "mobile", "cell", "tel"],
        category=PIICategory.DIRECT_IDENTIFIER,
        regulations=[
            PrivacyRegulation.GDPR, PrivacyRegulation.CCPA,
            PrivacyRegulation.LGPD, PrivacyRegulation.PIPEDA,
            PrivacyRegulation.APPI,
        ],
        requires_consent=True,
        confidence_base=90,
        description="International phone number (E.164 format)",
    ),
    PIIFieldDefinition(
        name="Passport Number (ICAO)",
        pattern=re.compile(r"^[A-Z0-9]{6,9}$"),
        column_hints=["passport", "travel_document", "pasaporte", "パスポート"],
        category=PIICategory.GOVERNMENT_ID,
        regulations=[
            PrivacyRegulation.GDPR, PrivacyRegulation.CCPA,
            PrivacyRegulation.LGPD, PrivacyRegulation.PIPEDA,
            PrivacyRegulation.APPI,
        ],
        requires_consent=True,
        confidence_base=85,
        description="International passport number (ICAO format)",
    ),
    PIIFieldDefinition(
        name="Credit Card Number",
        pattern=re.compile(r"^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$"),
        column_hints=["credit_card", "card_number", "cc", "tarjeta", "カード番号"],
        category=PIICategory.FINANCIAL,
        regulations=[
            PrivacyRegulation.GDPR, PrivacyRegulation.CCPA,
            PrivacyRegulation.LGPD, PrivacyRegulation.PIPEDA,
            PrivacyRegulation.APPI,
        ],
        requires_consent=True,
        confidence_base=95,
        description="Credit/debit card number (requires PCI-DSS)",
    ),
    PIIFieldDefinition(
        name="IP Address (IPv4)",
        pattern=re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$"),
        column_hints=["ip", "ip_address", "client_ip", "source_ip", "IPアドレス"],
        category=PIICategory.INDIRECT_IDENTIFIER,
        regulations=[
            PrivacyRegulation.GDPR, PrivacyRegulation.CCPA,
            PrivacyRegulation.LGPD, PrivacyRegulation.PIPEDA,
            PrivacyRegulation.APPI,
        ],
        requires_consent=True,
        confidence_base=95,
        description="IPv4 address (PII under most regulations)",
    ),
    PIIFieldDefinition(
        name="UUID/GUID",
        pattern=re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"),
        column_hints=["uuid", "guid", "device_id", "user_id", "tracking_id"],
        category=PIICategory.INDIRECT_IDENTIFIER,
        regulations=[
            PrivacyRegulation.GDPR, PrivacyRegulation.CCPA,
        ],
        requires_consent=True,
        confidence_base=80,
        description="Universal unique identifier (potential PII)",
    ),
    PIIFieldDefinition(
        name="Date of Birth",
        pattern=re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}$|^\d{2}[-/]\d{2}[-/]\d{4}$"),
        column_hints=[
            "dob", "birth", "birthday", "date_of_birth", "born",
            "nascimento", "生年月日", "誕生日",
        ],
        category=PIICategory.DIRECT_IDENTIFIER,
        regulations=[
            PrivacyRegulation.GDPR, PrivacyRegulation.CCPA,
            PrivacyRegulation.LGPD, PrivacyRegulation.PIPEDA,
            PrivacyRegulation.APPI,
        ],
        requires_consent=True,
        confidence_base=85,
        description="Date of birth (reveals age, identifies minors)",
    ),
    PIIFieldDefinition(
        name="Health/Medical Data",
        pattern=None,
        column_hints=[
            "health", "medical", "diagnosis", "treatment", "medication",
            "saúde", "médico", "健康", "医療", "santé",
        ],
        category=PIICategory.HEALTH,
        regulations=[
            PrivacyRegulation.GDPR, PrivacyRegulation.CCPA,
            PrivacyRegulation.LGPD, PrivacyRegulation.PIPEDA,
            PrivacyRegulation.APPI, PrivacyRegulation.HIPAA,
        ],
        requires_consent=True,
        is_special_category=True,
        confidence_base=90,
        description="Health data (special/sensitive category globally)",
    ),
    PIIFieldDefinition(
        name="Biometric Data",
        pattern=None,
        column_hints=[
            "biometric", "fingerprint", "face", "facial", "iris", "retina",
            "biométrico", "生体認証", "指紋",
        ],
        category=PIICategory.BIOMETRIC,
        regulations=[
            PrivacyRegulation.GDPR, PrivacyRegulation.CCPA,
            PrivacyRegulation.LGPD, PrivacyRegulation.PIPEDA,
            PrivacyRegulation.APPI,
        ],
        requires_consent=True,
        is_special_category=True,
        confidence_base=95,
        description="Biometric data (special/sensitive category globally)",
    ),
    PIIFieldDefinition(
        name="Racial/Ethnic Origin",
        pattern=None,
        column_hints=[
            "race", "ethnicity", "ethnic", "racial", "origin",
            "raça", "etnia", "人種", "民族",
        ],
        category=PIICategory.RACIAL_ETHNIC,
        regulations=[
            PrivacyRegulation.GDPR, PrivacyRegulation.CCPA,
            PrivacyRegulation.LGPD,
        ],
        requires_consent=True,
        is_special_category=True,
        confidence_base=95,
        description="Racial/ethnic data (special/sensitive category)",
    ),
    PIIFieldDefinition(
        name="Religious Belief",
        pattern=None,
        column_hints=[
            "religion", "religious", "faith", "belief",
            "religião", "fé", "宗教",
        ],
        category=PIICategory.RELIGIOUS,
        regulations=[
            PrivacyRegulation.GDPR, PrivacyRegulation.CCPA,
            PrivacyRegulation.LGPD,
        ],
        requires_consent=True,
        is_special_category=True,
        confidence_base=95,
        description="Religious data (special/sensitive category)",
    ),
]


class GlobalPrivacyValidator(PrivacyValidator):
    """Multi-jurisdiction privacy compliance validator.

    Validates against multiple privacy regulations simultaneously:
    - GDPR (EU)
    - CCPA/CPRA (California)
    - LGPD (Brazil)
    - PIPEDA (Canada)
    - APPI (Japan)

    Useful for organizations operating across multiple jurisdictions.
    """

    name = "global_privacy"
    regulation = PrivacyRegulation.GDPR  # Default, but checks all

    def __init__(
        self,
        regulations: list[PrivacyRegulation] | None = None,
        **kwargs: Any,
    ):
        """Initialize global privacy validator.

        Args:
            regulations: List of regulations to check (None = all)
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.target_regulations = regulations or [
            PrivacyRegulation.GDPR,
            PrivacyRegulation.CCPA,
            PrivacyRegulation.LGPD,
            PrivacyRegulation.PIPEDA,
            PrivacyRegulation.APPI,
        ]

    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        """Get combined PII definitions for all target regulations."""
        all_defs: list[PIIFieldDefinition] = []

        # Add global patterns
        all_defs.extend(GLOBAL_PII_DEFINITIONS)

        # Add regulation-specific patterns
        if PrivacyRegulation.LGPD in self.target_regulations:
            all_defs.extend(LGPD_PII_DEFINITIONS)

        if PrivacyRegulation.PIPEDA in self.target_regulations:
            all_defs.extend(PIPEDA_PII_DEFINITIONS)

        if PrivacyRegulation.APPI in self.target_regulations:
            all_defs.extend(APPI_PII_DEFINITIONS)

        return all_defs

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate against multiple privacy regulations.

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

        # Convert findings to issues with multi-regulation context
        for finding in all_findings:
            regs = ", ".join(r.value.upper() for r in self.target_regulations)
            issue = ValidationIssue(
                column=finding.column,
                issue_type=f"global_privacy_{finding.violation_type}",
                count=finding.count,
                severity=finding.severity,
                details=(
                    f"{finding.pii_type} ({finding.category.value}) detected with "
                    f"{finding.confidence}% confidence. "
                    f"Applicable regulations: {regs}. "
                    f"{finding.recommendation}"
                ),
                expected=f"No unprotected {finding.pii_type}",
                actual=f"Found {finding.count} potential instances",
                sample_values=finding.sample_values,
            )
            issues.append(issue)

        return issues


class LGPDComplianceValidator(PrivacyValidator):
    """Brazil LGPD compliance validator.

    Detects personal data as defined in Brazil's Lei Geral de
    Proteção de Dados (LGPD), which is similar to GDPR but with
    Brazil-specific identifiers like CPF and CNPJ.
    """

    name = "lgpd_compliance"
    regulation = PrivacyRegulation.LGPD

    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        """Get LGPD-specific PII definitions."""
        return LGPD_PII_DEFINITIONS + [
            d for d in GLOBAL_PII_DEFINITIONS
            if PrivacyRegulation.LGPD in d.regulations
        ]

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate LGPD compliance."""
        issues: list[ValidationIssue] = []
        pii_defs = self.get_pii_definitions()

        schema = lf.collect_schema()
        df = lf.collect()

        if df.height == 0:
            return issues

        if self.columns:
            columns = [c for c in self.columns if c in schema.names()]
        else:
            columns = [c for c in schema.names() if schema[c] in (pl.String, pl.Utf8)]

        all_findings: list[PrivacyFinding] = []
        for column in columns:
            findings = self._detect_pii_in_column(df, column, pii_defs)
            all_findings.extend(findings)

        issues = self._convert_findings_to_issues(all_findings)
        return issues


class PIPEDAComplianceValidator(PrivacyValidator):
    """Canada PIPEDA compliance validator.

    Detects personal information as defined in Canada's Personal
    Information Protection and Electronic Documents Act (PIPEDA).
    """

    name = "pipeda_compliance"
    regulation = PrivacyRegulation.PIPEDA

    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        """Get PIPEDA-specific PII definitions."""
        return PIPEDA_PII_DEFINITIONS + [
            d for d in GLOBAL_PII_DEFINITIONS
            if PrivacyRegulation.PIPEDA in d.regulations
        ]

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate PIPEDA compliance."""
        issues: list[ValidationIssue] = []
        pii_defs = self.get_pii_definitions()

        schema = lf.collect_schema()
        df = lf.collect()

        if df.height == 0:
            return issues

        if self.columns:
            columns = [c for c in self.columns if c in schema.names()]
        else:
            columns = [c for c in schema.names() if schema[c] in (pl.String, pl.Utf8)]

        all_findings: list[PrivacyFinding] = []
        for column in columns:
            findings = self._detect_pii_in_column(df, column, pii_defs)
            all_findings.extend(findings)

        issues = self._convert_findings_to_issues(all_findings)
        return issues


class APPIComplianceValidator(PrivacyValidator):
    """Japan APPI compliance validator.

    Detects personal information as defined in Japan's Act on
    Protection of Personal Information (APPI/個人情報保護法).
    """

    name = "appi_compliance"
    regulation = PrivacyRegulation.APPI

    def get_pii_definitions(self) -> list[PIIFieldDefinition]:
        """Get APPI-specific PII definitions."""
        return APPI_PII_DEFINITIONS + [
            d for d in GLOBAL_PII_DEFINITIONS
            if PrivacyRegulation.APPI in d.regulations
        ]

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate APPI compliance."""
        issues: list[ValidationIssue] = []
        pii_defs = self.get_pii_definitions()

        schema = lf.collect_schema()
        df = lf.collect()

        if df.height == 0:
            return issues

        if self.columns:
            columns = [c for c in self.columns if c in schema.names()]
        else:
            columns = [c for c in schema.names() if schema[c] in (pl.String, pl.Utf8)]

        all_findings: list[PrivacyFinding] = []
        for column in columns:
            findings = self._detect_pii_in_column(df, column, pii_defs)
            all_findings.extend(findings)

        issues = self._convert_findings_to_issues(all_findings)
        return issues
