"""Privacy compliance validators.

This module provides comprehensive validators for global privacy regulations:

**Supported Regulations:**
- GDPR (EU General Data Protection Regulation)
- CCPA/CPRA (California Consumer Privacy Act)
- LGPD (Brazil Lei Geral de Proteção de Dados)
- PIPEDA (Canada Personal Information Protection)
- APPI (Japan Act on Protection of Personal Information)

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
]
