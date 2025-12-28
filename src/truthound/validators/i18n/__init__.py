"""Internationalization (i18n) for Validator Error Messages.

Enterprise-grade i18n system with comprehensive features:

Core Features:
- Pre-defined error messages for all validator categories
- Support for 15+ languages (en, ko, ja, zh, de, fr, es, pt, it, ru, ar, he, fa)
- Automatic locale detection
- Consistent formatting with placeholders

Enterprise Features:
- CLDR plural rules support for all languages
- RTL language support (Arabic, Hebrew, Persian)
- Locale-aware number and date formatting
- Regional dialect support (en-US/en-GB, zh-CN/zh-TW, pt-BR/pt-PT)
- TMS integration (Crowdin, Lokalise, Phrase)
- Dynamic catalog loading with caching
- Context-based messages (formal, informal, technical)

Example:
    from truthound.validators.i18n import (
        get_validator_message,
        set_validator_locale,
        ValidatorMessageCode,
        pluralize,
        format_number,
        format_date,
    )

    # Set locale
    set_validator_locale("ko")

    # Get localized message
    msg = get_validator_message(
        ValidatorMessageCode.NULL_VALUES_FOUND,
        column="email",
        count=10,
    )
    # -> "'email' 컬럼에서 10개의 null 값이 발견되었습니다"

    # Pluralization
    msg = pluralize(
        count=5,
        forms={"one": "{count} file", "other": "{count} files"},
        locale="ru",
    )
    # -> "5 файлов"

    # Number formatting
    format_number(1234567.89, "de")  # "1.234.567,89"

    # Date formatting
    format_date(datetime.now(), "ko", DateStyle.LONG)  # "2024년 12월 28일"
"""

# Core messages
from truthound.validators.i18n.messages import (
    ValidatorMessageCode,
    ValidatorI18n,
    get_validator_message,
    set_validator_locale,
    get_validator_locale,
    format_issue_message,
)

# Catalogs
from truthound.validators.i18n.catalogs import (
    ValidatorMessageCatalog,
    CatalogBuilder,
    get_default_messages,
    get_korean_messages,
    get_japanese_messages,
    get_chinese_messages,
    get_german_messages,
    get_french_messages,
    get_spanish_messages,
    get_all_catalogs,
    get_supported_locales,
    create_custom_catalog,
)

# Protocols and types
from truthound.validators.i18n.protocols import (
    LocaleInfo,
    TextDirection,
    PluralCategory,
    NumberStyle,
    DateStyle,
    TimeStyle,
    MessageContext,
    FormattedNumber,
    FormattedDate,
    PluralizedMessage,
    ResolvedMessage,
)

# Plural rules
from truthound.validators.i18n.plural import (
    CLDRPluralRules,
    PluralOperands,
    get_plural_category,
    pluralize,
    get_plural_rules,
)

# BiDi support
from truthound.validators.i18n.bidi import (
    BiDiHandler,
    BiDiControl,
    BiDiConfig,
    BiDiStats,
    detect_direction,
    wrap_bidi,
    get_locale_direction,
    is_rtl_language,
    RTL_LANGUAGES,
)

# Number and date formatting
from truthound.validators.i18n.formatting import (
    LocaleNumberFormatter,
    LocaleDateFormatter,
    NumberSymbols,
    CurrencyInfo,
    DateTimePatterns,
    RelativeTimeData,
    format_number,
    format_currency,
    format_date,
    format_time,
    format_relative_time,
    get_number_symbols,
    get_currency_info,
    get_date_patterns,
)

# Dialects
from truthound.validators.i18n.dialects import (
    DialectDefinition,
    DialectRegistry,
    DialectResolver,
    get_dialect_registry,
    register_dialect,
    get_fallback_chain,
    create_dialect,
)

# TMS integration
from truthound.validators.i18n.tms import (
    TMSProvider,
    TMSConfig,
    TMSManager,
    BaseTMSProvider,
    CrowdinProvider,
    LokaliseProvider,
    PhraseProvider,
    TranslationStatus,
    WebhookEvent,
    create_provider,
    get_tms_manager,
)

# Dynamic loading
from truthound.validators.i18n.loader import (
    CatalogManager,
    ContextResolver,
    MessageComposer,
    LRUCache,
    FileSystemStorage,
    MemoryStorage,
    ContextualMessage,
    get_catalog_manager,
    get_context_resolver,
    resolve_message,
    resolve_plural_message,
)

# Extended catalogs
from truthound.validators.i18n.extended_catalogs import (
    get_portuguese_messages,
    get_portuguese_br_messages,
    get_portuguese_pt_messages,
    get_italian_messages,
    get_russian_messages,
    get_arabic_messages,
    get_hebrew_messages,
    get_persian_messages,
    get_all_extended_catalogs,
    get_extended_supported_locales,
)

__all__ = [
    # Core Messages
    "ValidatorMessageCode",
    "ValidatorI18n",
    "get_validator_message",
    "set_validator_locale",
    "get_validator_locale",
    "format_issue_message",

    # Catalogs
    "ValidatorMessageCatalog",
    "CatalogBuilder",
    "get_default_messages",
    "get_korean_messages",
    "get_japanese_messages",
    "get_chinese_messages",
    "get_german_messages",
    "get_french_messages",
    "get_spanish_messages",
    "get_all_catalogs",
    "get_supported_locales",
    "create_custom_catalog",

    # Extended Catalogs
    "get_portuguese_messages",
    "get_portuguese_br_messages",
    "get_portuguese_pt_messages",
    "get_italian_messages",
    "get_russian_messages",
    "get_arabic_messages",
    "get_hebrew_messages",
    "get_persian_messages",
    "get_all_extended_catalogs",
    "get_extended_supported_locales",

    # Protocols and Types
    "LocaleInfo",
    "TextDirection",
    "PluralCategory",
    "NumberStyle",
    "DateStyle",
    "TimeStyle",
    "MessageContext",
    "FormattedNumber",
    "FormattedDate",
    "PluralizedMessage",
    "ResolvedMessage",

    # Plural Rules
    "CLDRPluralRules",
    "PluralOperands",
    "get_plural_category",
    "pluralize",
    "get_plural_rules",

    # BiDi Support
    "BiDiHandler",
    "BiDiControl",
    "BiDiConfig",
    "BiDiStats",
    "detect_direction",
    "wrap_bidi",
    "get_locale_direction",
    "is_rtl_language",
    "RTL_LANGUAGES",

    # Number and Date Formatting
    "LocaleNumberFormatter",
    "LocaleDateFormatter",
    "NumberSymbols",
    "CurrencyInfo",
    "DateTimePatterns",
    "RelativeTimeData",
    "format_number",
    "format_currency",
    "format_date",
    "format_time",
    "format_relative_time",
    "get_number_symbols",
    "get_currency_info",
    "get_date_patterns",

    # Dialects
    "DialectDefinition",
    "DialectRegistry",
    "DialectResolver",
    "get_dialect_registry",
    "register_dialect",
    "get_fallback_chain",
    "create_dialect",

    # TMS Integration
    "TMSProvider",
    "TMSConfig",
    "TMSManager",
    "BaseTMSProvider",
    "CrowdinProvider",
    "LokaliseProvider",
    "PhraseProvider",
    "TranslationStatus",
    "WebhookEvent",
    "create_provider",
    "get_tms_manager",

    # Dynamic Loading
    "CatalogManager",
    "ContextResolver",
    "MessageComposer",
    "LRUCache",
    "FileSystemStorage",
    "MemoryStorage",
    "ContextualMessage",
    "get_catalog_manager",
    "get_context_resolver",
    "resolve_message",
    "resolve_plural_message",
]
