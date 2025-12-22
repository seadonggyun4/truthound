"""Base classes for validators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import polars as pl

from truthound.types import Severity


@dataclass
class ValidationIssue:
    """Represents a single data quality issue found during validation."""

    column: str
    issue_type: str
    count: int
    severity: Severity
    details: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "column": self.column,
            "issue_type": self.issue_type,
            "count": self.count,
            "severity": self.severity.value,
            "details": self.details,
        }


class Validator(ABC):
    """Abstract base class for all validators."""

    name: str = "base"

    @abstractmethod
    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Run validation on the given LazyFrame.

        Args:
            lf: Polars LazyFrame to validate.

        Returns:
            List of validation issues found.
        """
        pass
