"""Tests for Query-based validators."""

import polars as pl
import pytest

from truthound.types import Severity


# =============================================================================
# Query Result Validators Tests
# =============================================================================


class TestQueryResultValidators:
    """Tests for query result validators."""

    def test_query_returns_single_value_match(self):
        """Test query returning expected single value."""
        from truthound.validators import QueryReturnsSingleValueValidator

        df = pl.DataFrame({
            "status": ["active"] * 50 + ["inactive"] * 50,
        })

        validator = QueryReturnsSingleValueValidator(
            query="SELECT COUNT(*) as cnt FROM data WHERE status = 'active'",
            expected_value=50,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_query_returns_single_value_mismatch(self):
        """Test query returning unexpected value."""
        from truthound.validators import QueryReturnsSingleValueValidator

        df = pl.DataFrame({
            "status": ["active"] * 30 + ["inactive"] * 70,
        })

        validator = QueryReturnsSingleValueValidator(
            query="SELECT COUNT(*) as cnt FROM data WHERE status = 'active'",
            expected_value=50,
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].issue_type == "query_value_mismatch"

    def test_query_returns_single_value_with_tolerance(self):
        """Test query with numeric tolerance."""
        from truthound.validators import QueryReturnsSingleValueValidator

        df = pl.DataFrame({
            "price": [10.0, 20.0, 30.0],
        })

        validator = QueryReturnsSingleValueValidator(
            query="SELECT AVG(price) as avg_price FROM data",
            expected_value=20.0,
            tolerance=0.01,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_query_returns_no_rows_pass(self):
        """Test query returning no rows (expected)."""
        from truthound.validators import QueryReturnsNoRowsValidator

        df = pl.DataFrame({
            "amount": [10, 20, 30, 40],
        })

        validator = QueryReturnsNoRowsValidator(
            query="SELECT * FROM data WHERE amount < 0",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_query_returns_no_rows_fail(self):
        """Test query returning rows when none expected."""
        from truthound.validators import QueryReturnsNoRowsValidator

        df = pl.DataFrame({
            "amount": [10, -5, 30, -10],
        })

        validator = QueryReturnsNoRowsValidator(
            query="SELECT * FROM data WHERE amount < 0",
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 2

    def test_query_returns_rows_pass(self):
        """Test query returning rows within bounds."""
        from truthound.validators import QueryReturnsRowsValidator

        df = pl.DataFrame({
            "role": ["admin", "user", "user", "user"],
        })

        validator = QueryReturnsRowsValidator(
            query="SELECT * FROM data WHERE role = 'admin'",
            min_rows=1,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_query_returns_rows_below_min(self):
        """Test query returning too few rows."""
        from truthound.validators import QueryReturnsRowsValidator

        df = pl.DataFrame({
            "role": ["user", "user", "user"],
        })

        validator = QueryReturnsRowsValidator(
            query="SELECT * FROM data WHERE role = 'admin'",
            min_rows=1,
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].issue_type == "query_insufficient_rows"

    def test_query_result_matches_with_matcher(self):
        """Test query result with custom matcher."""
        from truthound.validators import QueryResultMatchesValidator

        df = pl.DataFrame({
            "price": [100, 200, 300],
        })

        validator = QueryResultMatchesValidator(
            query="SELECT AVG(price) as avg_price FROM data",
            matcher=lambda result: result["avg_price"][0] == 200,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0


# =============================================================================
# Query Row Count Validators Tests
# =============================================================================


class TestQueryRowCountValidators:
    """Tests for query row count validators."""

    def test_query_row_count_exact(self):
        """Test exact row count match."""
        from truthound.validators import QueryRowCountValidator

        df = pl.DataFrame({
            "status": ["active"] * 100,
        })

        validator = QueryRowCountValidator(
            query="SELECT * FROM data WHERE status = 'active'",
            expected_count=100,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_query_row_count_range(self):
        """Test row count within range."""
        from truthound.validators import QueryRowCountValidator

        df = pl.DataFrame({
            "value": list(range(75)),
        })

        validator = QueryRowCountValidator(
            query="SELECT * FROM data",
            min_count=50,
            max_count=100,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_query_row_count_ratio(self):
        """Test row count ratio validation."""
        from truthound.validators import QueryRowCountRatioValidator

        df = pl.DataFrame({
            "status": ["completed"] * 90 + ["error"] * 10,
        })

        validator = QueryRowCountRatioValidator(
            query="SELECT * FROM data WHERE status = 'completed'",
            min_ratio=0.85,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_query_row_count_ratio_fail(self):
        """Test row count ratio below minimum."""
        from truthound.validators import QueryRowCountRatioValidator

        df = pl.DataFrame({
            "status": ["completed"] * 70 + ["error"] * 30,
        })

        validator = QueryRowCountRatioValidator(
            query="SELECT * FROM data WHERE status = 'completed'",
            min_ratio=0.85,
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].issue_type == "query_ratio_below_min"

    def test_query_row_count_compare_equal(self):
        """Test comparing row counts between queries."""
        from truthound.validators import QueryRowCountCompareValidator

        df = pl.DataFrame({
            "type": ["order"] * 50 + ["order_item"] * 50,
        })

        validator = QueryRowCountCompareValidator(
            query="SELECT * FROM data WHERE type = 'order'",
            compare_query="SELECT * FROM data WHERE type = 'order_item'",
            relationship="equal",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0


# =============================================================================
# Query Column Validators Tests
# =============================================================================


class TestQueryColumnValidators:
    """Tests for query column validators."""

    def test_query_column_values_valid(self):
        """Test column values in expected set."""
        from truthound.validators import QueryColumnValuesValidator

        df = pl.DataFrame({
            "status": ["active", "inactive", "pending", "active"],
        })

        validator = QueryColumnValuesValidator(
            query="SELECT DISTINCT status FROM data",
            column="status",
            expected_values=["active", "inactive", "pending", "closed"],
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_query_column_values_unexpected(self):
        """Test unexpected column values."""
        from truthound.validators import QueryColumnValuesValidator

        df = pl.DataFrame({
            "status": ["active", "inactive", "unknown"],
        })

        validator = QueryColumnValuesValidator(
            query="SELECT DISTINCT status FROM data",
            column="status",
            expected_values=["active", "inactive", "pending"],
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert "unknown" in str(issues[0].sample_values)

    def test_query_column_unique(self):
        """Test column uniqueness in query result."""
        from truthound.validators import QueryColumnUniqueValidator

        df = pl.DataFrame({
            "category": ["A", "B", "C"],
            "count": [10, 20, 30],
        })

        validator = QueryColumnUniqueValidator(
            query="SELECT category, SUM(count) as total FROM data GROUP BY category",
            column="category",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_query_column_not_null(self):
        """Test no nulls in query column."""
        from truthound.validators import QueryColumnNotNullValidator

        df = pl.DataFrame({
            "category": ["A", "B", "C"],
            "value": [10, 20, 30],
        })

        validator = QueryColumnNotNullValidator(
            query="SELECT category, AVG(value) as avg_val FROM data GROUP BY category",
            column="avg_val",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0


# =============================================================================
# Query Aggregate Validators Tests
# =============================================================================


class TestQueryAggregateValidators:
    """Tests for query aggregate validators."""

    def test_query_aggregate_in_range(self):
        """Test aggregate value within range."""
        from truthound.validators import QueryAggregateValidator

        df = pl.DataFrame({
            "price": [10, 20, 30, 40, 50],
        })

        validator = QueryAggregateValidator(
            query="SELECT AVG(price) as avg_price FROM data",
            column="avg_price",
            min_value=20,
            max_value=40,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_query_aggregate_below_min(self):
        """Test aggregate value below minimum."""
        from truthound.validators import QueryAggregateValidator

        df = pl.DataFrame({
            "price": [5, 10, 15],
        })

        validator = QueryAggregateValidator(
            query="SELECT AVG(price) as avg_price FROM data",
            column="avg_price",
            min_value=20,
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].issue_type == "query_aggregate_below_min"

    def test_query_group_aggregate(self):
        """Test group-level aggregate validation."""
        from truthound.validators import QueryGroupAggregateValidator

        df = pl.DataFrame({
            "category": ["A", "A", "B", "B"],
            "price": [10, 20, 15, 25],
        })

        validator = QueryGroupAggregateValidator(
            query="SELECT category, AVG(price) as avg_price FROM data GROUP BY category",
            group_column="category",
            aggregate_column="avg_price",
            min_value=10,
            max_value=30,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_query_aggregate_compare(self):
        """Test comparing aggregates between queries."""
        from truthound.validators import QueryAggregateCompareValidator

        df = pl.DataFrame({
            "type": ["debit"] * 5 + ["credit"] * 5,
            "amount": [100] * 10,
        })

        validator = QueryAggregateCompareValidator(
            query="SELECT SUM(amount) as total FROM data WHERE type = 'debit'",
            compare_query="SELECT SUM(amount) as total FROM data WHERE type = 'credit'",
            column="total",
            relationship="equal",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0


# =============================================================================
# Expression Validators Tests
# =============================================================================


class TestExpressionValidators:
    """Tests for expression-based validators."""

    def test_custom_expression_pass(self):
        """Test custom expression validation pass."""
        from truthound.validators import CustomExpressionValidator

        df = pl.DataFrame({
            "quantity": [1, 2, 3],
            "price": [10, 20, 30],
        })

        validator = CustomExpressionValidator(
            filter_expr=(pl.col("quantity") > 0) & (pl.col("price") > 0),
            description="Quantity and price must be positive",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_custom_expression_fail(self):
        """Test custom expression validation fail."""
        from truthound.validators import CustomExpressionValidator

        df = pl.DataFrame({
            "quantity": [1, -2, 3],
            "price": [10, 20, -30],
        })

        validator = CustomExpressionValidator(
            filter_expr=(pl.col("quantity") > 0) & (pl.col("price") > 0),
            description="Quantity and price must be positive",
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 2

    def test_conditional_expression(self):
        """Test conditional expression validation."""
        from truthound.validators import ConditionalExpressionValidator

        df = pl.DataFrame({
            "status": ["shipped", "shipped", "pending"],
            "tracking": ["TRK001", "TRK002", None],
        })

        validator = ConditionalExpressionValidator(
            condition=pl.col("status") == "shipped",
            then_expr=pl.col("tracking").is_not_null(),
            description="Shipped orders must have tracking",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_conditional_expression_fail(self):
        """Test conditional expression failure."""
        from truthound.validators import ConditionalExpressionValidator

        df = pl.DataFrame({
            "status": ["shipped", "shipped", "pending"],
            "tracking": ["TRK001", None, None],
        })

        validator = ConditionalExpressionValidator(
            condition=pl.col("status") == "shipped",
            then_expr=pl.col("tracking").is_not_null(),
            description="Shipped orders must have tracking",
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 1

    def test_multi_condition_and(self):
        """Test multiple conditions with AND logic."""
        from truthound.validators import MultiConditionValidator

        df = pl.DataFrame({
            "age": [25, 30, 35],
            "status": ["active", "active", "active"],
        })

        validator = MultiConditionValidator(
            conditions=[
                (pl.col("age") >= 18, "Age 18+"),
                (pl.col("status") == "active", "Active status"),
            ],
            logic="and",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_multi_condition_or(self):
        """Test multiple conditions with OR logic."""
        from truthound.validators import MultiConditionValidator

        df = pl.DataFrame({
            "phone": ["123", None, None],
            "email": [None, "a@b.com", None],
        })

        validator = MultiConditionValidator(
            conditions=[
                (pl.col("phone").is_not_null(), "Phone provided"),
                (pl.col("email").is_not_null(), "Email provided"),
            ],
            logic="or",
        )
        issues = validator.validate(df.lazy())

        # Third row has neither phone nor email
        assert len(issues) == 1
        assert issues[0].count == 1

    def test_row_level_validator(self):
        """Test row-level Python function validation."""
        from truthound.validators import RowLevelValidator

        df = pl.DataFrame({
            "type": ["subscription", "one-time", "subscription"],
            "billing_cycle": ["monthly", None, "yearly"],
        })

        def validate_order(row):
            if row["type"] == "subscription":
                return row["billing_cycle"] in ["monthly", "yearly"]
            return True

        validator = RowLevelValidator(
            row_validator=validate_order,
            description="Subscription orders need billing cycle",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestQueryValidatorIntegration:
    """Integration tests for query validators."""

    def test_validator_registry(self):
        """Test that query validators are registered."""
        from truthound.validators import registry

        query_validators = [
            "query_returns_single_value",
            "query_returns_no_rows",
            "query_returns_rows",
            "query_result_matches",
            "query_row_count",
            "query_row_count_ratio",
            "query_row_count_compare",
            "query_column_values",
            "query_column_unique",
            "query_column_not_null",
            "query_aggregate",
            "query_group_aggregate",
            "query_aggregate_compare",
            "custom_expression",
            "conditional_expression",
            "multi_condition",
            "row_level",
        ]

        for name in query_validators:
            assert registry.get(name) is not None, f"Validator {name} not registered"

    def test_complex_business_rule(self):
        """Test complex business rule validation."""
        from truthound.validators import (
            QueryReturnsNoRowsValidator,
            ConditionalExpressionValidator,
        )

        # E-commerce order data
        df = pl.DataFrame({
            "order_id": [1, 2, 3, 4],
            "status": ["completed", "completed", "pending", "cancelled"],
            "payment_status": ["paid", "paid", "pending", "refunded"],
            "total": [100, 200, 150, 50],
            "items_count": [2, 3, 1, 1],
        })

        # Rule 1: No completed orders without payment
        v1 = QueryReturnsNoRowsValidator(
            query="SELECT * FROM data WHERE status = 'completed' AND payment_status != 'paid'",
        )
        issues1 = v1.validate(df.lazy())
        assert len(issues1) == 0

        # Rule 2: Cancelled orders must have refund
        v2 = ConditionalExpressionValidator(
            condition=pl.col("status") == "cancelled",
            then_expr=pl.col("payment_status") == "refunded",
            description="Cancelled orders must be refunded",
        )
        issues2 = v2.validate(df.lazy())
        assert len(issues2) == 0

    def test_query_error_handling(self):
        """Test graceful error handling for invalid queries."""
        from truthound.validators import QueryReturnsSingleValueValidator

        df = pl.DataFrame({
            "value": [1, 2, 3],
        })

        # Invalid SQL
        validator = QueryReturnsSingleValueValidator(
            query="SELECT invalid_syntax FROM",
            expected_value=1,
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].issue_type == "query_execution_error"
        assert issues[0].severity == Severity.CRITICAL
