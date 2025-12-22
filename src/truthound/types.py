"""Type definitions for Truthound."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Union

import polars as pl

if TYPE_CHECKING:
    import pandas as pd

# Using Any for pandas DataFrame to avoid import issues
DataInput = Union[str, pl.DataFrame, pl.LazyFrame, dict, Any]


class Severity(str, Enum):
    """Severity levels for data quality issues."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __ge__(self, other: "Severity") -> bool:
        order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        return order.index(self) >= order.index(other)

    def __gt__(self, other: "Severity") -> bool:
        order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        return order.index(self) > order.index(other)

    def __le__(self, other: "Severity") -> bool:
        order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        return order.index(self) <= order.index(other)

    def __lt__(self, other: "Severity") -> bool:
        order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        return order.index(self) < order.index(other)


class PIIType(str, Enum):
    """Types of personally identifiable information."""

    EMAIL = "Email Address"
    PHONE = "Phone Number"
    SSN = "SSN"
    CREDIT_CARD = "Credit Card"
    IP_ADDRESS = "IP Address"
    DATE_OF_BIRTH = "Date of Birth"
    ADDRESS = "Physical Address"
    # Korean specific
    KOREAN_RRN = "Korean RRN"  # 주민등록번호
    KOREAN_PHONE = "Korean Phone"  # 한국 전화번호
    BANK_ACCOUNT = "Bank Account"  # 계좌번호
    PASSPORT = "Passport Number"  # 여권번호
