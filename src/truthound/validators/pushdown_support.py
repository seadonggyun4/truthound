"""Query Pushdown Support for Validators.

This module provides the integration layer between validators and the
query pushdown framework, enabling validators to execute server-side
for SQL data sources.

Features:
- PushdownCapable protocol for validators that support SQL pushdown
- Automatic SQL query generation from validators
- Mixed execution: pushdown what's possible, fallback for the rest
- Performance optimization through server-side execution

Example:
    >>> from truthound.validators.pushdown_support import (
    ...     PushdownValidationEngine,
    ...     pushdown_validator,
    ... )
    >>>
    >>> # Create engine for SQL data source
    >>> engine = PushdownValidationEngine(sql_datasource)
    >>>
    >>> # Execute validators with automatic pushdown
    >>> issues = engine.validate([null_validator, duplicate_validator])
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from truthound.validators.base import ValidationIssue, Validator

if TYPE_CHECKING:
    import polars as pl
    from truthound.datasources.sql.base import BaseSQLDataSource
    from truthound.execution.pushdown import QueryBuilder, SQLDialect


# =============================================================================
# Pushdown Capability
# =============================================================================


class PushdownLevel(Enum):
    """Level of pushdown support for a validator."""

    NONE = auto()  # No pushdown possible
    PARTIAL = auto()  # Some checks can be pushed down
    FULL = auto()  # All checks can be pushed down


@dataclass
class PushdownQuery:
    """A query that can be pushed down to the database.

    Attributes:
        sql: The SQL query string
        column: Column being validated
        check_type: Type of validation check
        params: Query parameters
    """

    sql: str
    column: str
    check_type: str
    params: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"PushdownQuery({self.check_type} on {self.column})"


@dataclass
class PushdownResult:
    """Result of a pushdown query execution.

    Attributes:
        column: Column that was validated
        check_type: Type of validation check
        value: The result value (e.g., count of nulls)
        total_rows: Total row count for ratio calculation
        metadata: Additional result metadata
    """

    column: str
    check_type: str
    value: Any
    total_rows: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Pushdown Protocol
# =============================================================================


@runtime_checkable
class PushdownCapable(Protocol):
    """Protocol for validators that support query pushdown.

    Validators implementing this protocol can have their validation
    logic executed server-side in SQL databases, reducing data transfer
    and improving performance.

    Example:
        class NullValidator(Validator, PushdownCapable):
            pushdown_level = PushdownLevel.FULL

            def get_pushdown_queries(
                self,
                table: str,
                columns: list[str],
                dialect: SQLDialect,
            ) -> list[PushdownQuery]:
                return [
                    PushdownQuery(
                        sql=f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL",
                        column=col,
                        check_type="null_count",
                    )
                    for col in columns
                ]

            def process_pushdown_results(
                self,
                results: list[PushdownResult],
            ) -> list[ValidationIssue]:
                issues = []
                for r in results:
                    if r.value > 0:
                        issues.append(ValidationIssue(...))
                return issues
    """

    pushdown_level: PushdownLevel

    def get_pushdown_queries(
        self,
        table: str,
        columns: list[str],
        dialect: "SQLDialect",
    ) -> list[PushdownQuery]:
        """Generate pushdown queries for this validator.

        Args:
            table: Fully qualified table name
            columns: Columns to validate
            dialect: SQL dialect for query generation

        Returns:
            List of queries to execute server-side
        """
        ...

    def process_pushdown_results(
        self,
        results: list[PushdownResult],
    ) -> list[ValidationIssue]:
        """Process pushdown query results into validation issues.

        Args:
            results: Results from pushdown queries

        Returns:
            List of validation issues found
        """
        ...


# =============================================================================
# Pushdown Validation Engine
# =============================================================================


class PushdownValidationEngine:
    """Engine for executing validators with query pushdown.

    This engine analyzes validators and determines which validation
    logic can be pushed down to the database for server-side execution.

    Usage:
        >>> engine = PushdownValidationEngine(sql_datasource)
        >>> issues = engine.validate(validators)

    The engine will:
    1. Identify pushdown-capable validators
    2. Generate optimized SQL queries
    3. Execute queries server-side
    4. Fall back to client-side for non-pushable validators
    """

    def __init__(
        self,
        datasource: "BaseSQLDataSource",
        enable_batching: bool = True,
        max_batch_size: int = 10,
    ) -> None:
        """Initialize pushdown validation engine.

        Args:
            datasource: SQL data source to validate
            enable_batching: Combine multiple queries for efficiency
            max_batch_size: Maximum queries per batch
        """
        self.datasource = datasource
        self.enable_batching = enable_batching
        self.max_batch_size = max_batch_size
        self._dialect = self._infer_dialect()

    def _infer_dialect(self) -> "SQLDialect":
        """Infer SQL dialect from data source."""
        from truthound.execution.pushdown import SQLDialect

        source_type = self.datasource.source_type.lower()
        dialect_map = {
            "postgresql": SQLDialect.POSTGRESQL,
            "postgres": SQLDialect.POSTGRESQL,
            "mysql": SQLDialect.MYSQL,
            "sqlite": SQLDialect.SQLITE,
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
            "oracle": SQLDialect.ORACLE,
            "sqlserver": SQLDialect.SQLSERVER,
            "mssql": SQLDialect.SQLSERVER,
        }
        return dialect_map.get(source_type, SQLDialect.GENERIC)

    def validate(
        self,
        validators: list[Validator],
        columns: list[str] | None = None,
    ) -> list[ValidationIssue]:
        """Execute validators with pushdown optimization.

        Args:
            validators: List of validators to execute
            columns: Optional column filter

        Returns:
            All validation issues found
        """
        all_issues: list[ValidationIssue] = []

        # Separate pushdown-capable and regular validators
        pushdown_validators: list[Validator] = []
        regular_validators: list[Validator] = []

        for v in validators:
            if isinstance(v, PushdownCapable) and v.pushdown_level != PushdownLevel.NONE:
                pushdown_validators.append(v)
            else:
                regular_validators.append(v)

        # Get table and column info
        table = self.datasource.full_table_name
        target_columns = columns or self.datasource.columns

        # Execute pushdown validators
        if pushdown_validators:
            pushdown_issues = self._execute_pushdown_validators(
                pushdown_validators,
                table,
                target_columns,
            )
            all_issues.extend(pushdown_issues)

        # Execute regular validators (client-side)
        if regular_validators:
            lf = self.datasource.to_polars_lazyframe()
            for v in regular_validators:
                try:
                    issues = v.validate(lf)
                    all_issues.extend(issues)
                except Exception as e:
                    import logging
                    logging.getLogger("truthound.pushdown").warning(
                        f"Validator {v.name} failed: {e}"
                    )

        return all_issues

    def _execute_pushdown_validators(
        self,
        validators: list[Validator],
        table: str,
        columns: list[str],
    ) -> list[ValidationIssue]:
        """Execute pushdown-capable validators."""
        all_issues: list[ValidationIssue] = []

        # Get total row count (needed for ratio calculations)
        total_rows = self.datasource.row_count or 0

        for validator in validators:
            if not isinstance(validator, PushdownCapable):
                continue

            try:
                # Generate queries
                queries = validator.get_pushdown_queries(
                    table,
                    columns,
                    self._dialect,
                )

                if not queries:
                    continue

                # Execute queries
                results = self._execute_queries(queries, total_rows)

                # Process results
                issues = validator.process_pushdown_results(results)
                all_issues.extend(issues)

            except Exception as e:
                import logging
                logging.getLogger("truthound.pushdown").warning(
                    f"Pushdown failed for {validator.name}, falling back: {e}"
                )
                # Fallback to client-side
                try:
                    lf = self.datasource.to_polars_lazyframe()
                    issues = validator.validate(lf)
                    all_issues.extend(issues)
                except Exception:
                    pass

        return all_issues

    def _execute_queries(
        self,
        queries: list[PushdownQuery],
        total_rows: int,
    ) -> list[PushdownResult]:
        """Execute pushdown queries and return results."""
        results: list[PushdownResult] = []

        if self.enable_batching and len(queries) > 1:
            # Batch execution
            results.extend(self._execute_batched(queries, total_rows))
        else:
            # Individual execution
            for query in queries:
                result = self._execute_single(query, total_rows)
                if result:
                    results.append(result)

        return results

    def _execute_single(
        self,
        query: PushdownQuery,
        total_rows: int,
    ) -> PushdownResult | None:
        """Execute a single pushdown query."""
        try:
            rows = self.datasource.execute_query(query.sql, query.params or None)
            if rows and len(rows) > 0:
                # Get first value from first row
                value = list(rows[0].values())[0] if rows[0] else None
                return PushdownResult(
                    column=query.column,
                    check_type=query.check_type,
                    value=value,
                    total_rows=total_rows,
                )
        except Exception:
            pass
        return None

    def _execute_batched(
        self,
        queries: list[PushdownQuery],
        total_rows: int,
    ) -> list[PushdownResult]:
        """Execute queries in batches using UNION ALL."""
        results: list[PushdownResult] = []

        # Group queries by compatibility (same structure)
        groups: dict[str, list[PushdownQuery]] = {}
        for q in queries:
            key = q.check_type
            if key not in groups:
                groups[key] = []
            groups[key].append(q)

        # Execute each group
        for check_type, group_queries in groups.items():
            # For simplicity, execute individually if complex
            # A more sophisticated version would build UNION ALL
            for query in group_queries:
                result = self._execute_single(query, total_rows)
                if result:
                    results.append(result)

        return results

    def explain(self, validators: list[Validator]) -> str:
        """Explain which validators can be pushed down.

        Args:
            validators: List of validators to analyze

        Returns:
            Human-readable explanation
        """
        lines = ["Pushdown Analysis:", "=" * 40]

        for v in validators:
            if isinstance(v, PushdownCapable):
                level = v.pushdown_level.name
                lines.append(f"  {v.name}: {level} pushdown")
            else:
                lines.append(f"  {v.name}: Client-side only")

        lines.append("=" * 40)
        return "\n".join(lines)


# =============================================================================
# Pushdown Mixin for Common Validators
# =============================================================================


class NullCheckPushdownMixin:
    """Mixin providing pushdown support for null checking.

    Usage:
        class NullValidator(Validator, NullCheckPushdownMixin):
            pushdown_level = PushdownLevel.FULL
    """

    pushdown_level = PushdownLevel.FULL

    def get_pushdown_queries(
        self,
        table: str,
        columns: list[str],
        dialect: "SQLDialect",
    ) -> list[PushdownQuery]:
        """Generate NULL count queries for each column."""
        queries = []
        for col in columns:
            quoted_col = self._quote_identifier(col, dialect)
            queries.append(
                PushdownQuery(
                    sql=f"SELECT COUNT(*) FROM {table} WHERE {quoted_col} IS NULL",
                    column=col,
                    check_type="null_count",
                )
            )
        return queries

    def _quote_identifier(self, identifier: str, dialect: "SQLDialect") -> str:
        """Quote an identifier for the given dialect."""
        from truthound.execution.pushdown import SQLDialect

        quote_char = {
            SQLDialect.MYSQL: "`",
            SQLDialect.BIGQUERY: "`",
            SQLDialect.SQLSERVER: "[",
        }.get(dialect, '"')

        if dialect == SQLDialect.SQLSERVER:
            return f"[{identifier}]"
        return f"{quote_char}{identifier}{quote_char}"


class DuplicateCheckPushdownMixin:
    """Mixin providing pushdown support for duplicate checking."""

    pushdown_level = PushdownLevel.FULL

    def get_pushdown_queries(
        self,
        table: str,
        columns: list[str],
        dialect: "SQLDialect",
    ) -> list[PushdownQuery]:
        """Generate duplicate count queries for each column."""
        queries = []
        for col in columns:
            quoted_col = self._quote_identifier(col, dialect)
            queries.append(
                PushdownQuery(
                    sql=f"""
                        SELECT COUNT(*) - COUNT(DISTINCT {quoted_col})
                        FROM {table}
                        WHERE {quoted_col} IS NOT NULL
                    """,
                    column=col,
                    check_type="duplicate_count",
                )
            )
        return queries

    def _quote_identifier(self, identifier: str, dialect: "SQLDialect") -> str:
        """Quote an identifier for the given dialect."""
        from truthound.execution.pushdown import SQLDialect

        quote_char = {
            SQLDialect.MYSQL: "`",
            SQLDialect.BIGQUERY: "`",
            SQLDialect.SQLSERVER: "[",
        }.get(dialect, '"')

        if dialect == SQLDialect.SQLSERVER:
            return f"[{identifier}]"
        return f"{quote_char}{identifier}{quote_char}"


class RangeCheckPushdownMixin:
    """Mixin providing pushdown support for range checking."""

    pushdown_level = PushdownLevel.FULL

    def get_pushdown_queries(
        self,
        table: str,
        columns: list[str],
        dialect: "SQLDialect",
    ) -> list[PushdownQuery]:
        """Generate min/max queries for numeric columns."""
        queries = []
        for col in columns:
            quoted_col = self._quote_identifier(col, dialect)
            queries.append(
                PushdownQuery(
                    sql=f"SELECT MIN({quoted_col}), MAX({quoted_col}) FROM {table}",
                    column=col,
                    check_type="range",
                )
            )
        return queries

    def _quote_identifier(self, identifier: str, dialect: "SQLDialect") -> str:
        """Quote an identifier for the given dialect."""
        from truthound.execution.pushdown import SQLDialect

        quote_char = {
            SQLDialect.MYSQL: "`",
            SQLDialect.BIGQUERY: "`",
            SQLDialect.SQLSERVER: "[",
        }.get(dialect, '"')

        if dialect == SQLDialect.SQLSERVER:
            return f"[{identifier}]"
        return f"{quote_char}{identifier}{quote_char}"


class StatsPushdownMixin:
    """Mixin providing pushdown support for statistical validation."""

    pushdown_level = PushdownLevel.FULL

    def get_pushdown_queries(
        self,
        table: str,
        columns: list[str],
        dialect: "SQLDialect",
    ) -> list[PushdownQuery]:
        """Generate statistics queries for numeric columns."""
        queries = []
        for col in columns:
            quoted_col = self._quote_identifier(col, dialect)
            queries.append(
                PushdownQuery(
                    sql=f"""
                        SELECT
                            COUNT({quoted_col}) as cnt,
                            AVG({quoted_col}) as mean,
                            MIN({quoted_col}) as min_val,
                            MAX({quoted_col}) as max_val,
                            SUM({quoted_col}) as total
                        FROM {table}
                    """,
                    column=col,
                    check_type="stats",
                )
            )
        return queries

    def _quote_identifier(self, identifier: str, dialect: "SQLDialect") -> str:
        """Quote an identifier for the given dialect."""
        from truthound.execution.pushdown import SQLDialect

        quote_char = {
            SQLDialect.MYSQL: "`",
            SQLDialect.BIGQUERY: "`",
            SQLDialect.SQLSERVER: "[",
        }.get(dialect, '"')

        if dialect == SQLDialect.SQLSERVER:
            return f"[{identifier}]"
        return f"{quote_char}{identifier}{quote_char}"


# =============================================================================
# Utility Functions
# =============================================================================


def supports_pushdown(validator: Validator) -> bool:
    """Check if a validator supports pushdown.

    Args:
        validator: Validator to check

    Returns:
        True if validator supports any level of pushdown
    """
    if not isinstance(validator, PushdownCapable):
        return False
    return validator.pushdown_level != PushdownLevel.NONE


def get_pushdown_level(validator: Validator) -> PushdownLevel:
    """Get the pushdown level for a validator.

    Args:
        validator: Validator to check

    Returns:
        Pushdown level (NONE if not pushdown-capable)
    """
    if isinstance(validator, PushdownCapable):
        return validator.pushdown_level
    return PushdownLevel.NONE


def estimate_pushdown_savings(
    validators: list[Validator],
    row_count: int,
) -> dict[str, Any]:
    """Estimate performance savings from pushdown.

    Args:
        validators: Validators to analyze
        row_count: Dataset row count

    Returns:
        Dictionary with savings estimates
    """
    pushdown_count = sum(1 for v in validators if supports_pushdown(v))
    regular_count = len(validators) - pushdown_count

    # Rough estimate: pushdown saves ~90% of data transfer
    estimated_transfer_reduction = pushdown_count * row_count * 0.9

    return {
        "total_validators": len(validators),
        "pushdown_validators": pushdown_count,
        "regular_validators": regular_count,
        "pushdown_ratio": pushdown_count / len(validators) if validators else 0,
        "estimated_rows_saved": int(estimated_transfer_reduction),
    }
