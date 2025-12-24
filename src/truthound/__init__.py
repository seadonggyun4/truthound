"""Truthound - Zero-configuration data quality toolkit powered by Polars."""

from truthound.api import check, mask, profile, scan
from truthound.decorators import validator
from truthound.drift import compare
from truthound.report import Report
from truthound.schema import Schema, learn

# Data sources and execution engines (Phase 5)
from truthound import datasources
from truthound import execution
from truthound.datasources import get_datasource, get_sql_datasource

__version__ = "0.1.0"
__all__ = [
    # Core API
    "check",
    "scan",
    "mask",
    "profile",
    "learn",
    "compare",
    "validator",
    "Report",
    "Schema",
    # Phase 5: Data sources
    "datasources",
    "execution",
    "get_datasource",
    "get_sql_datasource",
]
