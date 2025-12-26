"""Privacy compliance validators.

This module provides comprehensive validators for global privacy regulations:

**Supported Regulations:**
- GDPR (EU General Data Protection Regulation)
- CCPA/CPRA (California Consumer Privacy Act)
- LGPD (Brazil Lei Geral de Proteção de Dados)
- PIPEDA (Canada Personal Information Protection)
- APPI (Japan Act on Protection of Personal Information)

**Plugin-Based Regulations (Extensible):**
- POPIA (South Africa Protection of Personal Information Act)
- PDPA Thailand (Personal Data Protection Act)
- PDPB India (Personal Data Protection Bill)
- KVKK Turkey (Kişisel Verilerin Korunması Kanunu)
- HIPAA Healthcare (US Health Insurance Portability)
- PCI-DSS Financial (Payment Card Industry Data Security)

**Validator Categories:**

1. **PII Detection Validators**
   - GDPRComplianceValidator: EU personal data detection
   - CCPAComplianceValidator: California PI detection
   - LGPDComplianceValidator: Brazil personal data
   - PIPEDAComplianceValidator: Canada personal information
   - APPIComplianceValidator: Japan personal information
   - GlobalPrivacyValidator: Multi-jurisdiction detection

2. **Special Category Validators**
   - GDPRSpecialCategoryValidator: Article 9 sensitive data
   - CCPASensitiveInfoValidator: CPRA sensitive PI

3. **Compliance Validators**
   - DataRetentionValidator: Retention period compliance
   - ConsentValidator: Consent tracking validation
   - GDPRDataMinimizationValidator: Article 5(1)(c) compliance
   - GDPRRightToErasureValidator: Article 17 compliance
   - CCPADoNotSellValidator: Opt-out tracking
   - CCPAConsumerRightsValidator: Rights infrastructure

4. **Plugin-Based Validators**
   - PluginBasedValidator: Use any registered privacy plugin
   - POPIAPlugin: South Africa POPIA compliance
   - PDPAThailandPlugin: Thailand PDPA compliance
   - PDPBIndiaPlugin: India PDPB compliance
   - KVKKTurkeyPlugin: Turkey KVKK compliance
   - HIPAAHealthcarePlugin: US HIPAA healthcare compliance
   - PCIDSSFinancialPlugin: PCI-DSS payment card compliance

**Usage Example:**

    from truthound.validators.privacy import (
        GDPRComplianceValidator,
        CCPAComplianceValidator,
        GlobalPrivacyValidator,
    )

    # GDPR compliance check
    gdpr_validator = GDPRComplianceValidator()
    issues = gdpr_validator.validate(df.lazy())

    # CCPA compliance check
    ccpa_validator = CCPAComplianceValidator()
    issues = ccpa_validator.validate(df.lazy())

    # Multi-jurisdiction check
    global_validator = GlobalPrivacyValidator()
    issues = global_validator.validate(df.lazy())

**Plugin System Example:**

    from truthound.validators.privacy import (
        get_privacy_plugin,
        list_privacy_plugins,
        PluginBasedValidator,
    )

    # List available plugins
    plugins = list_privacy_plugins()
    # ['popia', 'pdpa_thailand', 'pdpb_india', 'kvkk_turkey', ...]

    # Use a plugin
    plugin = get_privacy_plugin("popia")
    validator = plugin.create_validator()
    issues = validator.validate(df.lazy())

    # Or directly
    validator = PluginBasedValidator(regulation_code="hipaa_healthcare")
    issues = validator.validate(df.lazy())

Validators:
    GDPRComplianceValidator: GDPR Article 4 personal data detection
    GDPRSpecialCategoryValidator: GDPR Article 9 special categories
    GDPRDataMinimizationValidator: GDPR Article 5(1)(c) data minimization
    GDPRRightToErasureValidator: GDPR Article 17 right to be forgotten
    CCPAComplianceValidator: CCPA 1798.140 personal information
    CCPASensitiveInfoValidator: CPRA 1798.140(ae) sensitive PI
    CCPADoNotSellValidator: CCPA 1798.120 do not sell
    CCPAConsumerRightsValidator: Consumer rights infrastructure
    DataRetentionValidator: Data retention period compliance
    ConsentValidator: Consent tracking compliance
    GlobalPrivacyValidator: Multi-regulation compliance
    LGPDComplianceValidator: Brazil LGPD compliance
    PIPEDAComplianceValidator: Canada PIPEDA compliance
    APPIComplianceValidator: Japan APPI compliance
    PluginBasedValidator: Plugin-based privacy compliance
"""

from truthound.validators.privacy.base import (
    # Enums
    PrivacyRegulation,
    PIICategory,
    ConsentStatus,
    LegalBasis,
    # Data classes
    PIIFieldDefinition,
    PrivacyFinding,
    # Base validators
    PrivacyValidator,
    DataRetentionValidator,
    ConsentValidator,
)

from truthound.validators.privacy.gdpr import (
    # PII Definitions
    GDPR_PII_DEFINITIONS,
    # Validators
    GDPRComplianceValidator,
    GDPRSpecialCategoryValidator,
    GDPRDataMinimizationValidator,
    GDPRRightToErasureValidator,
)

from truthound.validators.privacy.ccpa import (
    # PII Definitions
    CCPA_PII_DEFINITIONS,
    # Validators
    CCPAComplianceValidator,
    CCPASensitiveInfoValidator,
    CCPADoNotSellValidator,
    CCPAConsumerRightsValidator,
)

from truthound.validators.privacy.global_patterns import (
    # PII Definitions
    LGPD_PII_DEFINITIONS,
    PIPEDA_PII_DEFINITIONS,
    APPI_PII_DEFINITIONS,
    GLOBAL_PII_DEFINITIONS,
    # Validators
    GlobalPrivacyValidator,
    LGPDComplianceValidator,
    PIPEDAComplianceValidator,
    APPIComplianceValidator,
)

from truthound.validators.privacy.plugins import (
    # Plugin system
    PrivacyRegulationPlugin,
    PluginBasedValidator,
    register_privacy_plugin,
    get_privacy_plugin,
    list_privacy_plugins,
    # Regional plugins
    POPIAPlugin,
    PDPAThailandPlugin,
    PDPBIndiaPlugin,
    KVKKTurkeyPlugin,
    # Industry plugins
    HIPAAHealthcarePlugin,
    PCIDSSFinancialPlugin,
)

__all__ = [
    # Enums
    "PrivacyRegulation",
    "PIICategory",
    "ConsentStatus",
    "LegalBasis",
    # Data classes
    "PIIFieldDefinition",
    "PrivacyFinding",
    # Base validators
    "PrivacyValidator",
    "DataRetentionValidator",
    "ConsentValidator",
    # GDPR
    "GDPR_PII_DEFINITIONS",
    "GDPRComplianceValidator",
    "GDPRSpecialCategoryValidator",
    "GDPRDataMinimizationValidator",
    "GDPRRightToErasureValidator",
    # CCPA
    "CCPA_PII_DEFINITIONS",
    "CCPAComplianceValidator",
    "CCPASensitiveInfoValidator",
    "CCPADoNotSellValidator",
    "CCPAConsumerRightsValidator",
    # Global patterns
    "LGPD_PII_DEFINITIONS",
    "PIPEDA_PII_DEFINITIONS",
    "APPI_PII_DEFINITIONS",
    "GLOBAL_PII_DEFINITIONS",
    "GlobalPrivacyValidator",
    "LGPDComplianceValidator",
    "PIPEDAComplianceValidator",
    "APPIComplianceValidator",
    # Plugin system
    "PrivacyRegulationPlugin",
    "PluginBasedValidator",
    "register_privacy_plugin",
    "get_privacy_plugin",
    "list_privacy_plugins",
    # Regional plugins
    "POPIAPlugin",
    "PDPAThailandPlugin",
    "PDPBIndiaPlugin",
    "KVKKTurkeyPlugin",
    # Industry plugins
    "HIPAAHealthcarePlugin",
    "PCIDSSFinancialPlugin",
]
