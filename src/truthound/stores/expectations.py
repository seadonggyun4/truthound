"""Expectation types for storage.

This module defines the data structures for expectations (validation rules)
that can be persisted and loaded from stores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


@dataclass
class Expectation:
    """A single validation expectation (rule).

    An expectation defines what conditions a column or dataset should meet.
    This is similar to Great Expectations' Expectation concept.

    Attributes:
        expectation_type: Type of expectation (e.g., "not_null", "in_range").
        column: Column this expectation applies to (None for table-level).
        kwargs: Parameters for the expectation.
        meta: Additional metadata about the expectation.
        enabled: Whether this expectation is active.
        mostly: Fraction of rows that must pass (0.0 to 1.0).
    """

    expectation_type: str
    column: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    mostly: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "expectation_type": self.expectation_type,
            "column": self.column,
            "kwargs": self.kwargs,
            "meta": self.meta,
            "enabled": self.enabled,
            "mostly": self.mostly,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Expectation":
        """Create from dictionary."""
        return cls(
            expectation_type=data["expectation_type"],
            column=data.get("column"),
            kwargs=data.get("kwargs", {}),
            meta=data.get("meta", {}),
            enabled=data.get("enabled", True),
            mostly=data.get("mostly", 1.0),
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        if self.column:
            return f"{self.expectation_type}(column={self.column})"
        return f"{self.expectation_type}"


@dataclass
class ExpectationSuite:
    """A collection of expectations for a data asset.

    An expectation suite groups related expectations together and provides
    metadata about when and how they should be applied.

    Attributes:
        name: Unique name for this suite.
        data_asset: Name of the data asset this suite validates.
        expectations: List of expectations in this suite.
        meta: Additional metadata about the suite.
        created_at: When the suite was created.
        updated_at: When the suite was last updated.
        tags: Key-value tags for filtering/grouping.
    """

    name: str
    data_asset: str
    expectations: list[Expectation] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Unique identifier for this suite."""
        return self.name

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "data_asset": self.data_asset,
            "expectations": [e.to_dict() for e in self.expectations],
            "meta": self.meta,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExpectationSuite":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            data_asset=data["data_asset"],
            expectations=[Expectation.from_dict(e) for e in data.get("expectations", [])],
            meta=data.get("meta", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
            tags=data.get("tags", {}),
        )

    def add_expectation(self, expectation: Expectation) -> None:
        """Add an expectation to the suite.

        Args:
            expectation: The expectation to add.
        """
        self.expectations.append(expectation)
        self.updated_at = datetime.now()

    def remove_expectation(self, index: int) -> Expectation:
        """Remove an expectation by index.

        Args:
            index: Index of the expectation to remove.

        Returns:
            The removed expectation.
        """
        expectation = self.expectations.pop(index)
        self.updated_at = datetime.now()
        return expectation

    def get_expectations_for_column(self, column: str) -> list[Expectation]:
        """Get all expectations for a specific column.

        Args:
            column: Column name to filter by.

        Returns:
            List of expectations for the column.
        """
        return [e for e in self.expectations if e.column == column]

    def get_table_expectations(self) -> list[Expectation]:
        """Get all table-level expectations (not column-specific).

        Returns:
            List of table-level expectations.
        """
        return [e for e in self.expectations if e.column is None]

    def get_enabled_expectations(self) -> list[Expectation]:
        """Get all enabled expectations.

        Returns:
            List of enabled expectations.
        """
        return [e for e in self.expectations if e.enabled]

    @property
    def expectation_count(self) -> int:
        """Get the total number of expectations."""
        return len(self.expectations)

    @property
    def enabled_count(self) -> int:
        """Get the number of enabled expectations."""
        return len(self.get_enabled_expectations())

    def summary(self) -> str:
        """Get a human-readable summary of the suite."""
        columns = {e.column for e in self.expectations if e.column}
        lines = [
            f"Expectation Suite: {self.name}",
            f"Data Asset: {self.data_asset}",
            f"Expectations: {self.expectation_count} ({self.enabled_count} enabled)",
            f"Columns: {len(columns)}",
            f"Updated: {self.updated_at.isoformat()}",
        ]
        return "\n".join(lines)

    @classmethod
    def create(
        cls,
        name: str,
        data_asset: str,
        expectations: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> "ExpectationSuite":
        """Create a new expectation suite with a convenient API.

        Args:
            name: Name for the suite.
            data_asset: Data asset this suite validates.
            expectations: Optional list of expectation dicts.
            **kwargs: Additional suite attributes.

        Returns:
            New ExpectationSuite instance.

        Example:
            >>> suite = ExpectationSuite.create(
            ...     name="customer_suite",
            ...     data_asset="customers.csv",
            ...     expectations=[
            ...         {"expectation_type": "not_null", "column": "email"},
            ...         {"expectation_type": "unique", "column": "id"},
            ...     ]
            ... )
        """
        suite = cls(name=name, data_asset=data_asset, **kwargs)
        if expectations:
            for exp_dict in expectations:
                suite.add_expectation(Expectation.from_dict(exp_dict))
        return suite
