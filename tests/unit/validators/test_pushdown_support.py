"""Tests for pushdown support module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock, MagicMock

import pytest

from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.pushdown_support import (
    PushdownLevel,
    PushdownQuery,
    PushdownResult,
    PushdownCapable,
    PushdownValidationEngine,
    NullCheckPushdownMixin,
    DuplicateCheckPushdownMixin,
    RangeCheckPushdownMixin,
    StatsPushdownMixin,
    supports_pushdown,
    get_pushdown_level,
    estimate_pushdown_savings,
)
from truthound.types import Severity


# =============================================================================
# Fixtures
# =============================================================================


class MockValidator(Validator):
    """Simple mock validator for testing."""

    name = "mock"
    category = "test"

    def validate(self, lf):
        return []


class MockPushdownValidator(Validator, NullCheckPushdownMixin):
    """Mock validator with pushdown support."""

    name = "mock_pushdown"
    category = "test"
    pushdown_level = PushdownLevel.FULL

    def validate(self, lf):
        return []

    def process_pushdown_results(self, results):
        issues = []
        for r in results:
            if r.value and r.value > 0:
                issues.append(
                    ValidationIssue(
                        column=r.column,
                        issue_type="null_values",
                        count=r.value,
                        severity=Severity.MEDIUM,
                        details=f"Found {r.value} null values",
                    )
                )
        return issues


@pytest.fixture
def mock_sql_datasource():
    """Create a mock SQL data source."""
    datasource = Mock()
    datasource.source_type = "postgresql"
    datasource.full_table_name = "test_schema.test_table"
    datasource.columns = ["id", "name", "email", "age"]
    datasource.row_count = 1000

    def execute_query(sql, params=None):
        # Return mock results based on query type
        if "IS NULL" in sql:
            return [{"count": 5}]
        elif "DISTINCT" in sql:
            return [{"count": 3}]
        elif "MIN" in sql:
            return [{"min_val": 0, "max_val": 100}]
        elif "COUNT" in sql:
            return [{"cnt": 1000, "mean": 50.0}]
        return [{}]

    datasource.execute_query = execute_query
    return datasource


# =============================================================================
# Test PushdownQuery
# =============================================================================


class TestPushdownQuery:
    """Tests for PushdownQuery dataclass."""

    def test_create_basic(self):
        """Test creating a basic pushdown query."""
        query = PushdownQuery(
            sql="SELECT COUNT(*) FROM users WHERE name IS NULL",
            column="name",
            check_type="null_count",
        )
        assert query.sql == "SELECT COUNT(*) FROM users WHERE name IS NULL"
        assert query.column == "name"
        assert query.check_type == "null_count"
        assert query.params == {}

    def test_create_with_params(self):
        """Test creating query with parameters."""
        query = PushdownQuery(
            sql="SELECT COUNT(*) FROM users WHERE age > ?",
            column="age",
            check_type="range_violation",
            params={"min_age": 18},
        )
        assert query.params == {"min_age": 18}

    def test_str_representation(self):
        """Test string representation."""
        query = PushdownQuery(
            sql="SELECT 1",
            column="name",
            check_type="null_count",
        )
        assert "null_count" in str(query)
        assert "name" in str(query)


# =============================================================================
# Test PushdownResult
# =============================================================================


class TestPushdownResult:
    """Tests for PushdownResult dataclass."""

    def test_create_basic(self):
        """Test creating a basic pushdown result."""
        result = PushdownResult(
            column="name",
            check_type="null_count",
            value=10,
            total_rows=1000,
        )
        assert result.column == "name"
        assert result.check_type == "null_count"
        assert result.value == 10
        assert result.total_rows == 1000

    def test_with_metadata(self):
        """Test result with metadata."""
        result = PushdownResult(
            column="email",
            check_type="pattern_violation",
            value=5,
            metadata={"pattern": "^[a-z]+@.*$"},
        )
        assert result.metadata["pattern"] == "^[a-z]+@.*$"


# =============================================================================
# Test PushdownLevel
# =============================================================================


class TestPushdownLevel:
    """Tests for PushdownLevel enum."""

    def test_levels_exist(self):
        """Test all pushdown levels exist."""
        assert PushdownLevel.NONE
        assert PushdownLevel.PARTIAL
        assert PushdownLevel.FULL

    def test_ordering(self):
        """Test levels can be compared."""
        assert PushdownLevel.NONE.value < PushdownLevel.PARTIAL.value
        assert PushdownLevel.PARTIAL.value < PushdownLevel.FULL.value


# =============================================================================
# Test Pushdown Mixins
# =============================================================================


class TestNullCheckPushdownMixin:
    """Tests for NullCheckPushdownMixin."""

    def test_generate_queries(self):
        """Test generating null check queries."""
        from truthound.execution.pushdown import SQLDialect

        validator = MockPushdownValidator()
        queries = validator.get_pushdown_queries(
            table="users",
            columns=["name", "email"],
            dialect=SQLDialect.POSTGRESQL,
        )

        assert len(queries) == 2
        assert all(q.check_type == "null_count" for q in queries)
        assert any("name" in q.sql for q in queries)
        assert any("email" in q.sql for q in queries)

    def test_quote_identifier_postgresql(self):
        """Test identifier quoting for PostgreSQL."""
        from truthound.execution.pushdown import SQLDialect

        validator = MockPushdownValidator()
        quoted = validator._quote_identifier("column", SQLDialect.POSTGRESQL)
        assert quoted == '"column"'

    def test_quote_identifier_mysql(self):
        """Test identifier quoting for MySQL."""
        from truthound.execution.pushdown import SQLDialect

        validator = MockPushdownValidator()
        quoted = validator._quote_identifier("column", SQLDialect.MYSQL)
        assert quoted == "`column`"


class TestDuplicateCheckPushdownMixin:
    """Tests for DuplicateCheckPushdownMixin."""

    def test_generate_queries(self):
        """Test generating duplicate check queries."""
        from truthound.execution.pushdown import SQLDialect

        class DuplicateValidator(Validator, DuplicateCheckPushdownMixin):
            name = "dup"

            def validate(self, lf):
                return []

            def process_pushdown_results(self, results):
                return []

        validator = DuplicateValidator()
        queries = validator.get_pushdown_queries(
            table="users",
            columns=["email"],
            dialect=SQLDialect.POSTGRESQL,
        )

        assert len(queries) == 1
        assert "DISTINCT" in queries[0].sql
        assert queries[0].check_type == "duplicate_count"


# =============================================================================
# Test PushdownValidationEngine
# =============================================================================


class TestPushdownValidationEngine:
    """Tests for PushdownValidationEngine."""

    def test_create_engine(self, mock_sql_datasource):
        """Test creating pushdown validation engine."""
        engine = PushdownValidationEngine(mock_sql_datasource)
        assert engine.datasource == mock_sql_datasource

    def test_infer_dialect_postgresql(self, mock_sql_datasource):
        """Test dialect inference for PostgreSQL."""
        from truthound.execution.pushdown import SQLDialect

        mock_sql_datasource.source_type = "postgresql"
        engine = PushdownValidationEngine(mock_sql_datasource)
        assert engine._dialect == SQLDialect.POSTGRESQL

    def test_infer_dialect_mysql(self, mock_sql_datasource):
        """Test dialect inference for MySQL."""
        from truthound.execution.pushdown import SQLDialect

        mock_sql_datasource.source_type = "mysql"
        engine = PushdownValidationEngine(mock_sql_datasource)
        assert engine._dialect == SQLDialect.MYSQL

    def test_infer_dialect_bigquery(self, mock_sql_datasource):
        """Test dialect inference for BigQuery."""
        from truthound.execution.pushdown import SQLDialect

        mock_sql_datasource.source_type = "bigquery"
        engine = PushdownValidationEngine(mock_sql_datasource)
        assert engine._dialect == SQLDialect.BIGQUERY

    def test_validate_with_pushdown_validators(self, mock_sql_datasource):
        """Test validation with pushdown-capable validators."""
        engine = PushdownValidationEngine(mock_sql_datasource)
        validator = MockPushdownValidator()

        issues = engine.validate([validator])

        # Should have issues from pushdown execution
        assert isinstance(issues, list)

    def test_validate_with_regular_validators(self, mock_sql_datasource):
        """Test validation with non-pushdown validators falls back."""
        import polars as pl

        # Setup mock to return a valid LazyFrame
        mock_sql_datasource.to_polars_lazyframe = Mock(
            return_value=pl.DataFrame({"a": [1, 2, 3]}).lazy()
        )

        engine = PushdownValidationEngine(mock_sql_datasource)
        validator = MockValidator()

        issues = engine.validate([validator])

        assert isinstance(issues, list)

    def test_validate_mixed_validators(self, mock_sql_datasource):
        """Test validation with mixed pushdown and regular validators."""
        import polars as pl

        mock_sql_datasource.to_polars_lazyframe = Mock(
            return_value=pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]}).lazy()
        )

        engine = PushdownValidationEngine(mock_sql_datasource)

        pushdown_validator = MockPushdownValidator()
        regular_validator = MockValidator()

        issues = engine.validate([pushdown_validator, regular_validator])

        assert isinstance(issues, list)

    def test_explain(self, mock_sql_datasource):
        """Test explain method."""
        engine = PushdownValidationEngine(mock_sql_datasource)

        pushdown_validator = MockPushdownValidator()
        regular_validator = MockValidator()

        explanation = engine.explain([pushdown_validator, regular_validator])

        assert "Pushdown Analysis" in explanation
        assert "FULL" in explanation or "Client-side" in explanation


# =============================================================================
# Test Utility Functions
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_supports_pushdown_true(self):
        """Test supports_pushdown returns True for pushdown validators."""
        validator = MockPushdownValidator()
        assert supports_pushdown(validator) is True

    def test_supports_pushdown_false(self):
        """Test supports_pushdown returns False for regular validators."""
        validator = MockValidator()
        assert supports_pushdown(validator) is False

    def test_get_pushdown_level_full(self):
        """Test get_pushdown_level for full pushdown."""
        validator = MockPushdownValidator()
        assert get_pushdown_level(validator) == PushdownLevel.FULL

    def test_get_pushdown_level_none(self):
        """Test get_pushdown_level for non-pushdown validator."""
        validator = MockValidator()
        assert get_pushdown_level(validator) == PushdownLevel.NONE

    def test_estimate_pushdown_savings(self):
        """Test estimate_pushdown_savings."""
        pushdown_validator = MockPushdownValidator()
        regular_validator = MockValidator()

        savings = estimate_pushdown_savings(
            validators=[pushdown_validator, regular_validator],
            row_count=1_000_000,
        )

        assert savings["total_validators"] == 2
        assert savings["pushdown_validators"] == 1
        assert savings["regular_validators"] == 1
        assert savings["pushdown_ratio"] == 0.5
        assert savings["estimated_rows_saved"] > 0

    def test_estimate_pushdown_savings_all_pushdown(self):
        """Test estimate with all pushdown validators."""
        validators = [MockPushdownValidator(), MockPushdownValidator()]

        savings = estimate_pushdown_savings(validators, row_count=1_000_000)

        assert savings["pushdown_ratio"] == 1.0

    def test_estimate_pushdown_savings_empty(self):
        """Test estimate with no validators."""
        savings = estimate_pushdown_savings([], row_count=1000)

        assert savings["total_validators"] == 0
        assert savings["pushdown_ratio"] == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestPushdownIntegration:
    """Integration tests for pushdown support."""

    def test_end_to_end_null_check(self, mock_sql_datasource):
        """Test end-to-end null check with pushdown."""
        engine = PushdownValidationEngine(mock_sql_datasource)
        validator = MockPushdownValidator()

        issues = engine.validate([validator], columns=["name", "email"])

        # Should execute pushdown queries and return issues
        assert isinstance(issues, list)
        # If mock returns count=5, we should have issues
        for issue in issues:
            assert hasattr(issue, "column")
            assert hasattr(issue, "count")

    def test_fallback_on_pushdown_error(self, mock_sql_datasource):
        """Test fallback to client-side on pushdown error."""
        import polars as pl

        # Make pushdown fail
        mock_sql_datasource.execute_query = Mock(side_effect=Exception("DB Error"))
        mock_sql_datasource.to_polars_lazyframe = Mock(
            return_value=pl.DataFrame({"name": [None, "a", None]}).lazy()
        )

        engine = PushdownValidationEngine(mock_sql_datasource)
        validator = MockPushdownValidator()

        # Should not raise, should fallback
        issues = engine.validate([validator])
        assert isinstance(issues, list)
