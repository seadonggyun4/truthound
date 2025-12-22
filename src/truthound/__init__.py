"""Truthound - Zero-configuration data quality toolkit powered by Polars."""

from truthound.api import check, mask, profile, scan
from truthound.decorators import validator
from truthound.drift import compare
from truthound.report import Report
from truthound.schema import Schema, learn

__version__ = "0.1.0"
__all__ = [
    "check",
    "scan",
    "mask",
    "profile",
    "learn",
    "compare",
    "validator",
    "Report",
    "Schema",
]
