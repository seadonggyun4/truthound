"""Base classes for query-based validators.

Query validators allow validation using Polars SQL or expression-based queries,
providing flexibility similar to Great Expectations' expect_query_* series.

Design Principles:
1. Use Polars SQL context for SQL-like queries
2. Support both SQL strings and Polars expressions
3. Provide clear error messages with query context
4. Enable parameterized queries for reusability
5. SQL Injection protection through query validation

Security:
    This module implements SQL injection protection for Polars SQL queries:
    - Validates query structure before execution
    - Blocks dangerous SQL patterns (DDL, DCL, multiple statements)
    - Supports safe parameterized queries
    - Whitelist-based table name validation

    The security module (truthound.validators.security) provides:
    - Multi-level security policies (STRICT, STANDARD, PERMISSIVE)
    - Pluggable pattern registry for custom blocking rules
    - Fluent SecureSQLBuilder API
    - ParameterizedQuery for safe parameter substitution
    - QueryAuditLogger for security monitoring
"""

from abc import abstractmethod
from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator, ValidatorConfig
from truthound.validators.registry import register_validator

# Import from the comprehensive security module
from truthound.validators.security import (
    SQLSecurityError,
    SQLInjectionError,
    QueryValidationError,
    SQLQueryValidator,
    validate_sql_query,
    SecureSQLBuilder,
    ParameterizedQuery,
    WhitelistValidator,
    SchemaWhitelist,
    SecurityPolicy,
    SecurityLevel,
    SecureQueryMixin,
    QueryAuditLogger,
    AuditEntry,
)

# Backward compatibility alias
SQLValidationError = QueryValidationError


class QueryValidator(Validator):
    """Base class for query-based validators.

    Provides infrastructure for executing Polars SQL queries and validating results.
    Subclasses implement specific validation logic on query results.

    Security:
        SQL queries are validated before execution to prevent injection attacks.
        Set validate_sql=False to disable validation (not recommended).

    Example:
        class MyQueryValidator(QueryValidator):
            def validate_query_result(self, result: pl.DataFrame) -> list[ValidationIssue]:
                # Custom validation logic
                pass
    """

    name = "query_base"
    category = "query"

    def __init__(
        self,
        query: str | None = None,
        expression: pl.Expr | None = None,
        table_name: str = "data",
        validate_sql: bool = True,
        allowed_tables: list[str] | None = None,
        **kwargs: Any,
    ):
        """Initialize query validator.

        Args:
            query: SQL query string (uses Polars SQL context)
            expression: Polars expression as alternative to SQL
            table_name: Name to use for the table in SQL context
            validate_sql: Whether to validate SQL for security (default: True)
            allowed_tables: Whitelist of allowed table names for SQL validation
            **kwargs: Additional validator config
        """
        super().__init__(**kwargs)
        self.query = query
        self.expression = expression
        self.table_name = table_name
        self.validate_sql = validate_sql

        # Set up allowed tables (always include the main table)
        if allowed_tables:
            self.allowed_tables = list(allowed_tables)
            if table_name not in self.allowed_tables:
                self.allowed_tables.append(table_name)
        else:
            self.allowed_tables = [table_name]

        if query is None and expression is None:
            raise ValueError("Either 'query' or 'expression' must be provided")

        # Validate SQL query if provided
        if query and validate_sql:
            try:
                validate_sql_query(query, allowed_tables=self.allowed_tables)
            except SQLValidationError as e:
                raise ValueError(f"SQL query validation failed: {e}")

    def _execute_query(self, lf: pl.LazyFrame) -> pl.DataFrame:
        """Execute the query and return results.

        Args:
            lf: Input LazyFrame

        Returns:
            Query result as DataFrame
        """
        if self.query:
            # Use Polars SQL context
            ctx = pl.SQLContext()
            ctx.register(self.table_name, lf)
            return ctx.execute(self.query).collect()
        elif self.expression:
            # Use expression directly
            return lf.select(self.expression).collect()
        else:
            raise ValueError("No query or expression provided")

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Execute query and validate results."""
        try:
            result = self._execute_query(lf)
            return self.validate_query_result(result, lf)
        except Exception as e:
            # Return validation issue for query execution errors
            return [
                ValidationIssue(
                    column="_query",
                    issue_type="query_execution_error",
                    count=1,
                    severity=Severity.CRITICAL,
                    details=f"Query execution failed: {e}",
                    expected="Valid query execution",
                )
            ]

    @abstractmethod
    def validate_query_result(
        self, result: pl.DataFrame, original_lf: pl.LazyFrame
    ) -> list[ValidationIssue]:
        """Validate the query result.

        Args:
            result: Query execution result
            original_lf: Original LazyFrame for context

        Returns:
            List of validation issues
        """
        pass


class ExpressionValidator(Validator):
    """Base class for expression-based validators.

    Uses Polars expressions for flexible, composable validation logic.
    More performant than SQL for simple validations.
    """

    name = "expression_base"
    category = "query"

    def __init__(
        self,
        filter_expr: pl.Expr,
        description: str = "Expression validation",
        **kwargs: Any,
    ):
        """Initialize expression validator.

        Args:
            filter_expr: Polars expression that returns boolean series
                         True = valid, False = invalid
            description: Human-readable description of the validation
            **kwargs: Additional validator config
        """
        super().__init__(**kwargs)
        self.filter_expr = filter_expr
        self.description = description

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate using the expression."""
        issues: list[ValidationIssue] = []

        try:
            # Count rows that fail the expression (where expr is False)
            result = lf.select([
                pl.len().alias("_total"),
                (~self.filter_expr).sum().alias("_failures"),
            ]).collect()

            total = result["_total"][0]
            failures = result["_failures"][0]

            if failures > 0:
                if self._passes_mostly(failures, total):
                    return issues

                ratio = failures / total if total > 0 else 0

                issues.append(
                    ValidationIssue(
                        column="_expression",
                        issue_type="expression_validation_failed",
                        count=failures,
                        severity=self._calculate_severity(ratio),
                        details=f"{failures}/{total} rows failed: {self.description}",
                        expected=self.description,
                    )
                )
        except Exception as e:
            issues.append(
                ValidationIssue(
                    column="_expression",
                    issue_type="expression_execution_error",
                    count=1,
                    severity=Severity.CRITICAL,
                    details=f"Expression execution failed: {e}",
                    expected="Valid expression execution",
                )
            )

        return issues
