"""Internationalization (i18n) support for Data Docs reports.

This module provides multi-language report generation with:
- CLDR-compliant plural rules
- Locale-specific formatting
- Translation catalogs for 15+ languages
- RTL (right-to-left) support

Example:
    from truthound.datadocs.i18n import (
        ReportCatalog,
        get_catalog,
        format_number,
        pluralize,
    )

    # Get catalog for Korean
    catalog = get_catalog("ko")

    # Translate a message
    msg = catalog.get("report.title")  # "데이터 품질 보고서"

    # Format numbers
    formatted = format_number(1234567.89, "ko")  # "1,234,567.89"

    # Pluralize (selects correct plural form, does NOT translate)
    msg = pluralize(1, "file", "files", "en")  # "1 file" (ONE form)
    msg = pluralize(5, "file", "files", "en")  # "5 files" (OTHER form)
"""

from truthound.datadocs.i18n.catalog import (
    ReportCatalog,
    CatalogRegistry,
    get_catalog,
    register_catalog,
    get_supported_locales,
    create_catalog_builder,
)
from truthound.datadocs.i18n.plurals import (
    PluralCategory,
    PluralRules,
    get_plural_category,
    pluralize,
    pluralize_with_forms,
)
from truthound.datadocs.i18n.formatting import (
    NumberFormatter,
    DateFormatter,
    format_number,
    format_percentage,
    format_date,
    format_datetime,
    format_duration,
)
from truthound.datadocs.i18n.loader import (
    LocaleLoader,
    load_locale_from_file,
    load_locale_from_dict,
)

__all__ = [
    # Catalog
    "ReportCatalog",
    "CatalogRegistry",
    "get_catalog",
    "register_catalog",
    "get_supported_locales",
    "create_catalog_builder",
    # Plurals
    "PluralCategory",
    "PluralRules",
    "get_plural_category",
    "pluralize",
    "pluralize_with_forms",
    # Formatting
    "NumberFormatter",
    "DateFormatter",
    "format_number",
    "format_percentage",
    "format_date",
    "format_datetime",
    "format_duration",
    # Loader
    "LocaleLoader",
    "load_locale_from_file",
    "load_locale_from_dict",
]
