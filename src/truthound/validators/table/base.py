"""Base classes for table metadata validators.

This module provides base classes for table-level validation.
"""

from abc import abstractmethod
from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator


class TableValidator(Validator):
    """Base class for table-level validators.

    Table validators check metadata and aggregate properties of entire tables,
    such as row counts, column counts, freshness, and schema matching.
    """

    name = "table_base"
    category = "table"

    def __init__(self, **kwargs: Any):
        """Initialize table validator.

        Args:
            **kwargs: Additional config passed to base Validator
        """
        super().__init__(**kwargs)

    @abstractmethod
    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate the table.

        Args:
            lf: LazyFrame to validate

        Returns:
            List of validation issues
        """
        pass


class TableSchemaValidator(Validator):
    """Base class for schema-focused table validators.

    Provides common functionality for validators that check table structure.
    """

    name = "table_schema_base"
    category = "table"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def _get_schema(self, lf: pl.LazyFrame) -> dict[str, pl.DataType]:
        """Get schema of the LazyFrame.

        Args:
            lf: LazyFrame to get schema from

        Returns:
            Dictionary of column name to data type
        """
        return lf.collect_schema()

    def _get_columns(self, lf: pl.LazyFrame) -> list[str]:
        """Get column names of the LazyFrame.

        Args:
            lf: LazyFrame to get columns from

        Returns:
            List of column names
        """
        return list(lf.collect_schema().keys())
