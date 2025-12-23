"""Tests for data profiling validators."""

import numpy as np
import polars as pl
import pytest

from truthound.validators.profiling import (
    CardinalityValidator,
    UniquenessRatioValidator,
    EntropyValidator,
    InformationGainValidator,
    ValueFrequencyValidator,
    DistributionShapeValidator,
)


class TestCardinalityValidator:
    """Tests for CardinalityValidator."""

    @pytest.fixture
    def high_cardinality_data(self):
        """Create data with high cardinality (unique values)."""
        return pl.LazyFrame({"id": list(range(100))})

    @pytest.fixture
    def low_cardinality_data(self):
        """Create data with low cardinality."""
        return pl.LazyFrame({"category": ["A", "B", "C"] * 100})

    def test_high_cardinality_passes(self, high_cardinality_data):
        """Test high cardinality data passes min check."""
        validator = CardinalityValidator(
            column="id",
            min_cardinality=0.9,
        )
        issues = validator.validate(high_cardinality_data)
        assert len(issues) == 0

    def test_low_cardinality_fails(self, high_cardinality_data):
        """Test high cardinality data fails max check."""
        validator = CardinalityValidator(
            column="id",
            max_cardinality=0.5,
        )
        issues = validator.validate(high_cardinality_data)
        assert len(issues) == 1
        assert issues[0].issue_type == "cardinality_too_high"

    def test_cardinality_too_low(self, low_cardinality_data):
        """Test low cardinality detection."""
        validator = CardinalityValidator(
            column="category",
            min_cardinality=0.5,
        )
        issues = validator.validate(low_cardinality_data)
        assert len(issues) == 1
        assert issues[0].issue_type == "cardinality_too_low"

    def test_unique_count_bounds(self, low_cardinality_data):
        """Test absolute unique count bounds."""
        validator = CardinalityValidator(
            column="category",
            min_unique_count=5,
        )
        issues = validator.validate(low_cardinality_data)
        assert len(issues) == 1
        assert issues[0].issue_type == "unique_count_too_low"


class TestUniquenessRatioValidator:
    """Tests for UniquenessRatioValidator."""

    @pytest.fixture
    def data_with_duplicates(self):
        """Create data with duplicates."""
        return pl.LazyFrame({
            "email": [
                "a@test.com",
                "b@test.com",
                "a@test.com",  # Duplicate
                "c@test.com",
                "b@test.com",  # Duplicate
            ]
        })

    @pytest.fixture
    def unique_data(self):
        """Create data with all unique values."""
        return pl.LazyFrame({"id": list(range(100))})

    def test_detects_low_uniqueness(self, data_with_duplicates):
        """Test detection of low uniqueness."""
        validator = UniquenessRatioValidator(
            column="email",
            min_uniqueness=0.9,
        )
        issues = validator.validate(data_with_duplicates)
        assert len(issues) == 1
        assert issues[0].issue_type == "uniqueness_too_low"

    def test_reports_top_duplicates(self, data_with_duplicates):
        """Test reporting of top duplicates."""
        validator = UniquenessRatioValidator(
            column="email",
            min_uniqueness=0.9,
            report_top_duplicates=3,
        )
        issues = validator.validate(data_with_duplicates)
        assert "a@test.com" in issues[0].details

    def test_passes_unique_data(self, unique_data):
        """Test unique data passes validation."""
        validator = UniquenessRatioValidator(
            column="id",
            min_uniqueness=0.99,
        )
        issues = validator.validate(unique_data)
        assert len(issues) == 0


class TestEntropyValidator:
    """Tests for EntropyValidator."""

    @pytest.fixture
    def constant_data(self):
        """Create data with constant value (zero entropy)."""
        return pl.LazyFrame({"status": ["active"] * 100})

    @pytest.fixture
    def uniform_data(self):
        """Create data with uniform distribution (high entropy)."""
        return pl.LazyFrame({
            "category": [f"cat_{i}" for i in range(100)]
        })

    @pytest.fixture
    def skewed_data(self):
        """Create data with skewed distribution."""
        return pl.LazyFrame({
            "category": ["A"] * 90 + ["B"] * 10
        })

    def test_constant_data_entropy(self, constant_data):
        """Test constant data has zero entropy."""
        validator = EntropyValidator(
            column="status",
            min_entropy=0.5,
        )
        issues = validator.validate(constant_data)
        assert len(issues) == 1
        assert issues[0].issue_type == "entropy_too_low"

    def test_high_entropy_data(self, uniform_data):
        """Test uniform data has high entropy."""
        validator = EntropyValidator(
            column="category",
            max_entropy=3.0,  # Very low for 100 unique values
        )
        issues = validator.validate(uniform_data)
        assert len(issues) == 1
        assert issues[0].issue_type == "entropy_too_high"

    def test_normalized_entropy(self, skewed_data):
        """Test normalized entropy validation."""
        validator = EntropyValidator(
            column="category",
            min_normalized_entropy=0.9,  # Require high normalized entropy
        )
        issues = validator.validate(skewed_data)
        assert len(issues) == 1
        assert issues[0].issue_type == "normalized_entropy_too_low"


class TestInformationGainValidator:
    """Tests for InformationGainValidator."""

    @pytest.fixture
    def correlated_data(self):
        """Create data with high correlation between columns."""
        return pl.LazyFrame({
            "country": ["US", "US", "UK", "UK", "DE", "DE"],
            "currency": ["USD", "USD", "GBP", "GBP", "EUR", "EUR"],
        })

    @pytest.fixture
    def independent_data(self):
        """Create data with independent columns."""
        return pl.LazyFrame({
            "category": ["A", "B", "C", "A", "B", "C"],
            "random": ["X", "Y", "X", "Z", "X", "Y"],
        })

    def test_high_information_gain(self, correlated_data):
        """Test detection of high information gain."""
        validator = InformationGainValidator(
            column="country",
            target_column="currency",
            min_information_gain=0.5,
        )
        issues = validator.validate(correlated_data)
        # Country perfectly predicts currency
        assert len(issues) == 0

    def test_low_information_gain(self, independent_data):
        """Test detection of low information gain."""
        validator = InformationGainValidator(
            column="category",
            target_column="random",
            min_information_gain=1.0,  # High threshold
        )
        issues = validator.validate(independent_data)
        assert len(issues) == 1
        assert issues[0].issue_type == "information_gain_too_low"


class TestValueFrequencyValidator:
    """Tests for ValueFrequencyValidator."""

    @pytest.fixture
    def dominant_value_data(self):
        """Create data with one dominant value."""
        return pl.LazyFrame({
            "status": ["active"] * 95 + ["inactive"] * 5
        })

    @pytest.fixture
    def uniform_frequency_data(self):
        """Create data with uniform value frequencies."""
        return pl.LazyFrame({
            "category": ["A", "B", "C", "D"] * 25
        })

    def test_detects_dominant_value(self, dominant_value_data):
        """Test detection of overly dominant value."""
        validator = ValueFrequencyValidator(
            column="status",
            max_top_frequency=0.8,
        )
        issues = validator.validate(dominant_value_data)
        assert len(issues) == 1
        assert issues[0].issue_type == "top_frequency_too_high"

    def test_detects_rare_values(self, dominant_value_data):
        """Test detection of rare values."""
        validator = ValueFrequencyValidator(
            column="status",
            min_bottom_frequency=0.1,
        )
        issues = validator.validate(dominant_value_data)
        assert len(issues) == 1
        assert issues[0].issue_type == "rare_values_detected"

    def test_expected_values_check(self, uniform_frequency_data):
        """Test expected values validation."""
        validator = ValueFrequencyValidator(
            column="category",
            expected_values=["A", "B", "C", "D", "E"],  # E is missing
        )
        issues = validator.validate(uniform_frequency_data)
        assert len(issues) == 1
        assert issues[0].issue_type == "expected_values_missing"

    def test_expected_frequencies(self, dominant_value_data):
        """Test expected frequency validation."""
        validator = ValueFrequencyValidator(
            column="status",
            expected_frequencies={"active": 0.5, "inactive": 0.5},
            frequency_tolerance=0.2,
        )
        issues = validator.validate(dominant_value_data)
        assert len(issues) == 1
        assert issues[0].issue_type == "frequency_deviation"

    def test_top_n_ratio(self, dominant_value_data):
        """Test top N ratio check."""
        validator = ValueFrequencyValidator(
            column="status",
            top_n_max_ratio=0.9,
            top_n=1,
        )
        issues = validator.validate(dominant_value_data)
        assert len(issues) == 1
        assert issues[0].issue_type == "top_n_ratio_too_high"


class TestDistributionShapeValidator:
    """Tests for DistributionShapeValidator."""

    @pytest.fixture
    def uniform_distribution(self):
        """Create data with uniform distribution."""
        return pl.LazyFrame({
            "category": ["A", "B", "C", "D"] * 25
        })

    @pytest.fixture
    def skewed_distribution(self):
        """Create data with highly skewed distribution."""
        return pl.LazyFrame({
            "category": ["A"] * 90 + ["B"] * 7 + ["C"] * 2 + ["D"] * 1
        })

    def test_uniform_distribution_passes(self, uniform_distribution):
        """Test uniform distribution passes check."""
        validator = DistributionShapeValidator(
            column="category",
            expected_shape="uniform",
            shape_tolerance=0.1,
        )
        issues = validator.validate(uniform_distribution)
        assert len(issues) == 0

    def test_non_uniform_detection(self, skewed_distribution):
        """Test detection of non-uniform distribution."""
        validator = DistributionShapeValidator(
            column="category",
            expected_shape="uniform",
            shape_tolerance=0.05,
        )
        issues = validator.validate(skewed_distribution)
        assert len(issues) == 1
        assert issues[0].issue_type == "non_uniform_distribution"

    def test_gini_coefficient(self, skewed_distribution):
        """Test Gini coefficient validation."""
        validator = DistributionShapeValidator(
            column="category",
            max_gini=0.3,  # Low Gini = more equal
        )
        issues = validator.validate(skewed_distribution)
        assert len(issues) == 1
        assert issues[0].issue_type == "gini_too_high"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe(self):
        """Test handling of empty data."""
        data = pl.LazyFrame({"col": []})
        validator = CardinalityValidator(column="col")
        issues = validator.validate(data)
        assert len(issues) == 0

    def test_single_value(self):
        """Test handling of single value."""
        data = pl.LazyFrame({"col": ["only_value"]})
        validator = EntropyValidator(column="col", min_entropy=0.0)
        issues = validator.validate(data)
        assert len(issues) == 0

    def test_null_handling(self):
        """Test null value handling."""
        data = pl.LazyFrame({"col": [None, "A", None, "B", "A"]})

        # Without nulls (default)
        validator = CardinalityValidator(
            column="col",
            include_nulls=False,
        )
        issues = validator.validate(data)
        # Should compute cardinality on non-null values only

        # With nulls
        validator_with_nulls = CardinalityValidator(
            column="col",
            include_nulls=True,
        )
        issues_with_nulls = validator_with_nulls.validate(data)
        # Both should handle gracefully
        assert isinstance(issues, list)
        assert isinstance(issues_with_nulls, list)
