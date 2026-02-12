"""Type definitions for Truthound."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
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


# ============================================================================
# Result Format - Controls validation result detail level
# ============================================================================

# Ordered list for comparison operators (lower index = less detail)
_RESULT_FORMAT_ORDER: list[str] = ["boolean_only", "basic", "summary", "complete"]


class ResultFormat(str, Enum):
    """Controls the detail level of validation results.

    Each level includes all information from the previous level plus additional
    detail. Higher levels require more computation.

    Levels:
        BOOLEAN_ONLY: Only pass/fail flag. No counts, samples, or details.
                      Fastest — skips all unnecessary computation.
        BASIC:        + observed_value, unexpected_count, partial_unexpected_list.
                      Similar to current default behavior.
        SUMMARY:      + partial_unexpected_counts (value frequency), partial index list.
                      Default level — good balance of info and performance.
        COMPLETE:     + full unexpected_list, full index_list, unexpected_rows DataFrame,
                      debug query. Most detailed — use for debugging.
    """

    BOOLEAN_ONLY = "boolean_only"
    BASIC = "basic"
    SUMMARY = "summary"
    COMPLETE = "complete"

    def __ge__(self, other: "ResultFormat") -> bool:
        return _RESULT_FORMAT_ORDER.index(self.value) >= _RESULT_FORMAT_ORDER.index(other.value)

    def __gt__(self, other: "ResultFormat") -> bool:
        return _RESULT_FORMAT_ORDER.index(self.value) > _RESULT_FORMAT_ORDER.index(other.value)

    def __le__(self, other: "ResultFormat") -> bool:
        return _RESULT_FORMAT_ORDER.index(self.value) <= _RESULT_FORMAT_ORDER.index(other.value)

    def __lt__(self, other: "ResultFormat") -> bool:
        return _RESULT_FORMAT_ORDER.index(self.value) < _RESULT_FORMAT_ORDER.index(other.value)

    @classmethod
    def from_string(cls, value: str) -> "ResultFormat":
        """Create from string, case-insensitive."""
        try:
            return cls(value.lower().strip())
        except ValueError:
            valid = ", ".join(m.value for m in cls)
            raise ValueError(f"Invalid result_format: {value!r}. Must be one of: {valid}")


@dataclass(frozen=True)
class ResultFormatConfig:
    """Fine-grained control over result detail level.

    Extends ResultFormat with additional options that cannot be expressed
    by the enum alone (e.g., limiting the number of returned failure rows
    even at COMPLETE level).

    This is an immutable (frozen) dataclass for thread safety.
    """

    format: ResultFormat = ResultFormat.SUMMARY
    partial_unexpected_count: int = 20
    include_unexpected_rows: bool = False
    max_unexpected_rows: int = 1000
    include_unexpected_index: bool = False
    return_debug_query: bool = False

    def __post_init__(self) -> None:
        if self.partial_unexpected_count < 0:
            raise ValueError(
                f"partial_unexpected_count must be >= 0, got {self.partial_unexpected_count}"
            )
        if self.max_unexpected_rows < 1:
            raise ValueError(
                f"max_unexpected_rows must be >= 1, got {self.max_unexpected_rows}"
            )

    # -- Query methods: what to include at each level --

    def includes_observed_value(self) -> bool:
        """Whether to compute and include observed_value (BASIC+)."""
        return self.format >= ResultFormat.BASIC

    def includes_unexpected_samples(self) -> bool:
        """Whether to collect partial_unexpected_list samples (BASIC+)."""
        return self.format >= ResultFormat.BASIC

    def includes_unexpected_counts(self) -> bool:
        """Whether to collect value frequency counts (SUMMARY+)."""
        return self.format >= ResultFormat.SUMMARY

    def includes_full_results(self) -> bool:
        """Whether to collect full failure rows/indices (COMPLETE only)."""
        return self.format >= ResultFormat.COMPLETE

    # -- Factory helpers --

    def replace(self, **kwargs: Any) -> "ResultFormatConfig":
        """Create a new config with updated values (immutable update)."""
        from dataclasses import asdict
        current = asdict(self)
        current.update(kwargs)
        # Ensure format stays as enum
        if isinstance(current.get("format"), str):
            current["format"] = ResultFormat.from_string(current["format"])
        return ResultFormatConfig(**current)

    @classmethod
    def from_any(cls, value: "str | ResultFormat | ResultFormatConfig | None") -> "ResultFormatConfig":
        """Normalize any supported input into a ResultFormatConfig.

        Accepts:
            - None → default config (SUMMARY)
            - str → parsed as ResultFormat enum
            - ResultFormat → wrapped in default config
            - ResultFormatConfig → returned as-is
        """
        if value is None:
            return cls()
        if isinstance(value, ResultFormatConfig):
            return value
        if isinstance(value, ResultFormat):
            return cls(format=value)
        if isinstance(value, str):
            return cls(format=ResultFormat.from_string(value))
        raise TypeError(
            f"Cannot convert {type(value).__name__} to ResultFormatConfig. "
            f"Expected str, ResultFormat, ResultFormatConfig, or None."
        )


# ============================================================================
# Validation Detail - Structured validation result data
# ============================================================================


@dataclass
class ValidationDetail:
    """Structured detail for a single validation result.

    Mirrors GX's ExpectationValidationResult.result dictionary as a
    type-safe dataclass.  Fields are populated progressively depending
    on the active ``ResultFormat`` level:

    ========== =====================================================
    Level      Fields populated
    ========== =====================================================
    BOOLEAN_ONLY  element_count, missing_count
    BASIC         + observed_value, unexpected_count,
                  unexpected_percent, unexpected_percent_nonmissing,
                  partial_unexpected_list
    SUMMARY       + partial_unexpected_counts,
                  partial_unexpected_index_list
    COMPLETE      + unexpected_list, unexpected_index_list,
                  unexpected_rows, debug_query
    ========== =====================================================

    The ``to_dict()`` helper serialises only non-default fields so
    that lower detail levels produce compact output.
    """

    # -- Always populated (BOOLEAN_ONLY and above) --
    element_count: int = 0
    missing_count: int = 0

    # -- BASIC and above --
    observed_value: Any = None
    unexpected_count: int = 0
    unexpected_percent: float = 0.0
    unexpected_percent_nonmissing: float = 0.0
    partial_unexpected_list: list[Any] | None = None

    # -- SUMMARY and above --
    partial_unexpected_counts: list[dict[str, Any]] | None = None
    partial_unexpected_index_list: list[int] | None = None

    # -- COMPLETE only --
    unexpected_list: list[Any] | None = None
    unexpected_index_list: list[int] | None = None
    unexpected_rows: pl.DataFrame | None = None
    debug_query: str | None = None

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self, *, include_zeros: bool = False) -> dict[str, Any]:
        """Return a dict containing only non-default / non-None fields.

        Args:
            include_zeros: If ``True``, include int/float fields even
                when they are zero.  Defaults to ``False`` for compact
                output.
        """
        result: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            if not include_zeros and isinstance(value, (int, float)) and value == 0:
                continue
            if isinstance(value, pl.DataFrame):
                result[f.name] = value.to_dicts()
            else:
                result[f.name] = value
        return result

    # ------------------------------------------------------------------ #
    # Factory helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def from_aggregates(
        cls,
        *,
        element_count: int,
        missing_count: int = 0,
        unexpected_count: int = 0,
        observed_value: Any = None,
    ) -> "ValidationDetail":
        """Create a detail pre-filled with Phase-1 aggregate data."""
        total = element_count
        nonmissing = total - missing_count
        return cls(
            element_count=element_count,
            missing_count=missing_count,
            observed_value=observed_value,
            unexpected_count=unexpected_count,
            unexpected_percent=(
                (unexpected_count / total * 100) if total > 0 else 0.0
            ),
            unexpected_percent_nonmissing=(
                (unexpected_count / nonmissing * 100) if nonmissing > 0 else 0.0
            ),
        )


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
