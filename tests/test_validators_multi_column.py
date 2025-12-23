"""Tests for Multi-column compound validators."""

import polars as pl
import pytest

from truthound.types import Severity


# =============================================================================
# Arithmetic Validators Tests
# =============================================================================


class TestArithmeticValidators:
    """Tests for arithmetic validators."""

    def test_column_sum_with_result_column(self):
        """Test sum equals result column."""
        from truthound.validators import ColumnSumValidator

        df = pl.DataFrame({
            "a": [10, 20, 30],
            "b": [5, 10, 15],
            "c": [5, 10, 15],
            "total": [20, 40, 60],
        })

        validator = ColumnSumValidator(
            columns=["a", "b", "c"],
            result_column="total",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_column_sum_mismatch(self):
        """Test sum mismatch detection."""
        from truthound.validators import ColumnSumValidator

        df = pl.DataFrame({
            "a": [10, 20, 30],
            "b": [5, 10, 15],
            "total": [15, 50, 45],  # Second row: 20+10=30 != 50, Third row: 30+15=45 = 45 (correct)
        })

        validator = ColumnSumValidator(
            columns=["a", "b"],
            result_column="total",
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 1  # Only second row is wrong

    def test_column_sum_with_expected_value(self):
        """Test sum equals expected constant."""
        from truthound.validators import ColumnSumValidator

        df = pl.DataFrame({
            "q1": [25, 25, 25],
            "q2": [25, 25, 25],
            "q3": [25, 25, 25],
            "q4": [25, 25, 25],
        })

        validator = ColumnSumValidator(
            columns=["q1", "q2", "q3", "q4"],
            expected_value=100,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_column_sum_range(self):
        """Test sum within range."""
        from truthound.validators import ColumnSumValidator

        df = pl.DataFrame({
            "score_1": [30, 40, 50],
            "score_2": [40, 50, 60],
        })

        validator = ColumnSumValidator(
            columns=["score_1", "score_2"],
            min_value=50,
            max_value=120,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_column_product(self):
        """Test product validation."""
        from truthound.validators import ColumnProductValidator

        df = pl.DataFrame({
            "quantity": [2, 3, 4],
            "unit_price": [10, 20, 30],
            "total_price": [20, 60, 120],
        })

        validator = ColumnProductValidator(
            columns=["quantity", "unit_price"],
            result_column="total_price",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_column_difference(self):
        """Test difference validation."""
        from truthound.validators import ColumnDifferenceValidator

        df = pl.DataFrame({
            "gross": [100, 200, 300],
            "tax": [10, 20, 30],
            "net": [90, 180, 270],
        })

        validator = ColumnDifferenceValidator(
            columns=["gross", "tax"],
            result_column="net",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_column_ratio(self):
        """Test ratio validation."""
        from truthound.validators import ColumnRatioValidator

        df = pl.DataFrame({
            "profit": [20, 30, 40],
            "revenue": [100, 100, 100],
        })

        validator = ColumnRatioValidator(
            columns=["profit", "revenue"],
            min_value=0.1,
            max_value=0.5,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_column_ratio_zero_division(self):
        """Test ratio with zero denominator handling."""
        from truthound.validators import ColumnRatioValidator

        df = pl.DataFrame({
            "numerator": [10, 20, 30],
            "denominator": [2, 0, 3],  # Zero in second row
        })

        validator = ColumnRatioValidator(
            columns=["numerator", "denominator"],
            min_value=5,
            max_value=15,
            handle_zero_division="skip",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0  # Zero division skipped

    def test_column_percentage(self):
        """Test percentage validation."""
        from truthound.validators import ColumnPercentageValidator

        df = pl.DataFrame({
            "part": [25, 50, 75],
            "total": [100, 100, 100],
        })

        validator = ColumnPercentageValidator(
            columns=["part", "total"],
            min_value=0,
            max_value=100,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0


# =============================================================================
# Comparison Validators Tests
# =============================================================================


class TestComparisonValidators:
    """Tests for comparison validators."""

    def test_column_comparison_less_than(self):
        """Test less than comparison."""
        from truthound.validators import ColumnComparisonValidator

        df = pl.DataFrame({
            "start_date": [1, 2, 3],
            "end_date": [5, 6, 7],
        })

        validator = ColumnComparisonValidator(
            columns=["start_date", "end_date"],
            operator="<",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_column_comparison_fail(self):
        """Test comparison failure."""
        from truthound.validators import ColumnComparisonValidator

        df = pl.DataFrame({
            "price": [100, 200, 300],
            "max_price": [150, 150, 150],  # Second: 200 > 150, Third: 300 > 150
        })

        validator = ColumnComparisonValidator(
            columns=["price", "max_price"],
            operator="<=",
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 2  # Second and third rows exceed max

    def test_column_chain_comparison(self):
        """Test chain comparison."""
        from truthound.validators import ColumnChainComparisonValidator

        df = pl.DataFrame({
            "min_val": [0, 10, 20],
            "value": [50, 50, 50],
            "max_val": [100, 100, 100],
        })

        validator = ColumnChainComparisonValidator(
            columns=["min_val", "value", "max_val"],
            operators=["<=", "<="],
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_column_max(self):
        """Test max validation."""
        from truthound.validators import ColumnMaxValidator

        df = pl.DataFrame({
            "score_1": [80, 70, 90],
            "score_2": [75, 85, 80],
            "score_3": [70, 60, 95],
            "max_score": [80, 85, 95],
        })

        validator = ColumnMaxValidator(
            columns=["score_1", "score_2", "score_3"],
            result_column="max_score",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_column_min(self):
        """Test min validation."""
        from truthound.validators import ColumnMinValidator

        df = pl.DataFrame({
            "price_a": [100, 200, 300],
            "price_b": [150, 180, 250],
            "min_price": [100, 180, 250],
        })

        validator = ColumnMinValidator(
            columns=["price_a", "price_b"],
            result_column="min_price",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_column_mean(self):
        """Test mean validation."""
        from truthound.validators import ColumnMeanValidator

        df = pl.DataFrame({
            "score_1": [80, 70, 90],
            "score_2": [90, 80, 100],
            "avg_score": [85.0, 75.0, 95.0],
        })

        validator = ColumnMeanValidator(
            columns=["score_1", "score_2"],
            result_column="avg_score",
            tolerance=0.01,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0


# =============================================================================
# Consistency Validators Tests
# =============================================================================


class TestConsistencyValidators:
    """Tests for consistency validators."""

    def test_column_consistency_rules(self):
        """Test consistency rules."""
        from truthound.validators import ColumnConsistencyValidator

        df = pl.DataFrame({
            "status": ["shipped", "shipped", "pending"],
            "ship_date": ["2024-01-01", "2024-01-02", None],
        })

        validator = ColumnConsistencyValidator(
            rules=[
                {
                    "when": pl.col("status") == "shipped",
                    "then": pl.col("ship_date").is_not_null(),
                    "description": "Shipped orders need ship date",
                }
            ]
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_column_consistency_fail(self):
        """Test consistency rule failure."""
        from truthound.validators import ColumnConsistencyValidator

        df = pl.DataFrame({
            "status": ["shipped", "shipped", "pending"],
            "ship_date": ["2024-01-01", None, None],  # Second row missing
        })

        validator = ColumnConsistencyValidator(
            rules=[
                {
                    "when": pl.col("status") == "shipped",
                    "then": pl.col("ship_date").is_not_null(),
                    "description": "Shipped orders need ship date",
                }
            ]
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 1

    def test_column_mutual_exclusivity(self):
        """Test mutual exclusivity."""
        from truthound.validators import ColumnMutualExclusivityValidator

        df = pl.DataFrame({
            "credit_card": ["1234", None, None],
            "bank_transfer": [None, "ACC123", None],
            "paypal": [None, None, "PP001"],
        })

        validator = ColumnMutualExclusivityValidator(
            columns=["credit_card", "bank_transfer", "paypal"],
            allow_none=False,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_column_mutual_exclusivity_violation(self):
        """Test mutual exclusivity violation."""
        from truthound.validators import ColumnMutualExclusivityValidator

        df = pl.DataFrame({
            "credit_card": ["1234", "5678", None],
            "bank_transfer": [None, "ACC123", None],  # Second row has both
        })

        validator = ColumnMutualExclusivityValidator(
            columns=["credit_card", "bank_transfer"],
            allow_none=True,
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 1

    def test_column_coexistence(self):
        """Test column coexistence."""
        from truthound.validators import ColumnCoexistenceValidator

        df = pl.DataFrame({
            "street": ["123 Main", None, "456 Oak"],
            "city": ["NYC", None, "LA"],
            "zip": ["10001", None, "90001"],
        })

        validator = ColumnCoexistenceValidator(
            columns=["street", "city", "zip"],
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_column_coexistence_violation(self):
        """Test coexistence violation."""
        from truthound.validators import ColumnCoexistenceValidator

        df = pl.DataFrame({
            "street": ["123 Main", "456 Oak", None],
            "city": ["NYC", None, None],  # Second row partial
        })

        validator = ColumnCoexistenceValidator(
            columns=["street", "city"],
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 1

    def test_column_dependency(self):
        """Test column dependency."""
        from truthound.validators import ColumnDependencyValidator

        df = pl.DataFrame({
            "type": ["subscription", "one-time", "subscription"],
            "billing_cycle": ["monthly", None, "yearly"],
        })

        validator = ColumnDependencyValidator(
            condition_column="type",
            condition_value="subscription",
            required_columns=["billing_cycle"],
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_column_implication(self):
        """Test column implication."""
        from truthound.validators import ColumnImplicationValidator

        df = pl.DataFrame({
            "country": ["US", "UK", "JP"],
            "currency": ["USD", "GBP", "JPY"],
        })

        validator = ColumnImplicationValidator(
            antecedent_column="country",
            antecedent_value="US",
            consequent_column="currency",
            consequent_value="USD",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0


# =============================================================================
# Statistical Validators Tests
# =============================================================================


class TestStatisticalValidators:
    """Tests for statistical validators."""

    def test_column_correlation_positive(self):
        """Test positive correlation."""
        from truthound.validators import ColumnCorrelationValidator

        df = pl.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [2, 4, 5, 8, 10, 11, 14, 16, 18, 20],
        })

        validator = ColumnCorrelationValidator(
            column_a="x",
            column_b="y",
            min_correlation=0.9,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_column_correlation_too_low(self):
        """Test correlation below threshold."""
        from truthound.validators import ColumnCorrelationValidator

        df = pl.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [5, 2, 4, 1, 3],  # Low correlation
        })

        validator = ColumnCorrelationValidator(
            column_a="x",
            column_b="y",
            min_correlation=0.8,
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].issue_type == "correlation_too_low"

    def test_column_covariance(self):
        """Test covariance validation."""
        from truthound.validators import ColumnCovarianceValidator

        df = pl.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
        })

        validator = ColumnCovarianceValidator(
            column_a="x",
            column_b="y",
            min_value=0,  # Positive covariance expected
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_multi_column_variance(self):
        """Test variance across columns."""
        from truthound.validators import MultiColumnVarianceValidator

        df = pl.DataFrame({
            "score_1": [80, 85, 90, 75, 95],
            "score_2": [82, 88, 87, 78, 92],
            "score_3": [79, 83, 91, 74, 96],
        })

        validator = MultiColumnVarianceValidator(
            columns=["score_1", "score_2", "score_3"],
            max_variance_ratio=3.0,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestMultiColumnIntegration:
    """Integration tests for multi-column validators."""

    def test_validator_registry(self):
        """Test that multi-column validators are registered."""
        from truthound.validators import registry

        multi_column_validators = [
            "column_sum",
            "column_product",
            "column_difference",
            "column_ratio",
            "column_percentage",
            "column_comparison",
            "column_chain_comparison",
            "column_max",
            "column_min",
            "column_mean",
            "column_consistency",
            "column_mutual_exclusivity",
            "column_coexistence",
            "column_dependency",
            "column_implication",
            "column_correlation",
            "column_covariance",
            "multi_column_variance",
        ]

        for name in multi_column_validators:
            assert registry.get(name) is not None, f"Validator {name} not registered"

    def test_financial_calculation_validation(self):
        """Test financial calculation chain validation."""
        from truthound.validators import (
            ColumnProductValidator,
            ColumnSumValidator,
            ColumnDifferenceValidator,
        )

        # Invoice data
        df = pl.DataFrame({
            "quantity": [10, 5, 20],
            "unit_price": [100, 200, 50],
            "subtotal": [1000, 1000, 1000],
            "tax_rate": [0.1, 0.1, 0.1],
            "tax": [100, 100, 100],
            "total": [1100, 1100, 1100],
        })

        # Validate quantity * unit_price = subtotal
        v1 = ColumnProductValidator(
            columns=["quantity", "unit_price"],
            result_column="subtotal",
        )
        issues1 = v1.validate(df.lazy())
        assert len(issues1) == 0

        # Validate subtotal + tax = total
        v2 = ColumnSumValidator(
            columns=["subtotal", "tax"],
            result_column="total",
        )
        issues2 = v2.validate(df.lazy())
        assert len(issues2) == 0

    def test_complex_business_rules(self):
        """Test complex business rule validation."""
        from truthound.validators import (
            ColumnConsistencyValidator,
            ColumnDependencyValidator,
            ColumnImplicationValidator,
        )

        # E-commerce order data
        df = pl.DataFrame({
            "order_type": ["standard", "subscription", "subscription"],
            "billing_cycle": [None, "monthly", "yearly"],
            "status": ["shipped", "active", "active"],
            "ship_date": ["2024-01-15", None, None],
            "country": ["US", "US", "UK"],
            "currency": ["USD", "USD", "GBP"],
        })

        # If subscription, billing_cycle required
        v1 = ColumnDependencyValidator(
            condition_column="order_type",
            condition_value="subscription",
            required_columns=["billing_cycle"],
        )
        issues1 = v1.validate(df.lazy())
        assert len(issues1) == 0

        # If shipped, ship_date required
        v2 = ColumnConsistencyValidator(
            rules=[
                {
                    "when": pl.col("status") == "shipped",
                    "then": pl.col("ship_date").is_not_null(),
                    "description": "Shipped orders need ship date",
                }
            ]
        )
        issues2 = v2.validate(df.lazy())
        assert len(issues2) == 0

        # Country-currency implication
        v3 = ColumnImplicationValidator(
            antecedent_column="country",
            antecedent_value="US",
            consequent_column="currency",
            consequent_value="USD",
        )
        issues3 = v3.validate(df.lazy())
        assert len(issues3) == 0
