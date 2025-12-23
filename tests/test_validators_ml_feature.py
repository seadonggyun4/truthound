"""Tests for ML feature validators."""

import pytest
import polars as pl
import numpy as np

from truthound.validators.ml_feature import (
    FeatureNullImpactValidator,
    FeatureScaleValidator,
    FeatureCorrelationMatrixValidator,
    TargetLeakageValidator,
    ScaleType,
)


class TestFeatureNullImpactValidator:
    """Tests for feature null impact validation."""

    def test_high_null_ratio_detection(self):
        """Test detection of high null ratios."""
        # Create DataFrame with high null ratio
        df = pl.DataFrame(
            {
                "feature1": [1.0, None, None, None, None, 6.0, None, None, None, 10.0],
                "feature2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            }
        )
        validator = FeatureNullImpactValidator(max_null_ratio=0.5)
        issues = validator.validate(df.lazy())

        assert len(issues) > 0
        assert any(i.issue_type == "high_null_ratio" for i in issues)

    def test_no_nulls_passes(self):
        """Test that data without nulls passes."""
        df = pl.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        validator = FeatureNullImpactValidator(max_null_ratio=0.5)
        issues = validator.validate(df.lazy())

        null_issues = [i for i in issues if "null" in i.issue_type.lower()]
        assert len(null_issues) == 0

    def test_null_correlation_detection(self):
        """Test detection of correlated null patterns."""
        # Create correlated null pattern
        n = 200
        nulls = np.random.choice([True, False], n, p=[0.3, 0.7])
        feature1 = [None if nulls[i] else float(i) for i in range(n)]
        feature2 = [None if nulls[i] else float(i * 2) for i in range(n)]
        feature3 = [float(i) for i in range(n)]

        df = pl.DataFrame(
            {"feature1": feature1, "feature2": feature2, "feature3": feature3}
        )

        validator = FeatureNullImpactValidator(
            detect_null_correlation=True, null_correlation_threshold=0.7
        )
        issues = validator.validate(df.lazy())

        # Should detect correlated null patterns
        assert any(i.issue_type == "null_pattern_correlation" for i in issues)

    def test_warning_level_nulls(self):
        """Test warning level null detection."""
        df = pl.DataFrame(
            {
                "feature1": [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "feature2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            }
        )
        validator = FeatureNullImpactValidator(
            max_null_ratio=0.5, warn_null_ratio=0.05
        )
        issues = validator.validate(df.lazy())

        assert any(i.issue_type == "elevated_null_ratio" for i in issues)


class TestFeatureScaleValidator:
    """Tests for feature scale validation."""

    def test_scale_inconsistency_detection(self):
        """Test detection of scale inconsistency."""
        df = pl.DataFrame(
            {
                "feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
                "feature2": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            }
        )
        validator = FeatureScaleValidator(max_scale_ratio=10.0)
        issues = validator.validate(df.lazy())

        assert len(issues) > 0
        assert any(i.issue_type == "scale_inconsistency" for i in issues)

    def test_consistent_scale_passes(self):
        """Test that consistent scales pass."""
        df = pl.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [2.0, 4.0, 6.0, 8.0, 10.0],
            }
        )
        validator = FeatureScaleValidator(max_scale_ratio=100.0)
        issues = validator.validate(df.lazy())

        scale_issues = [i for i in issues if i.issue_type == "scale_inconsistency"]
        assert len(scale_issues) == 0

    def test_extreme_magnitude_detection(self):
        """Test detection of extreme magnitudes."""
        df = pl.DataFrame(
            {
                "feature1": [1e10, 2e10, 3e10, 4e10, 5e10],
                "feature2": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        validator = FeatureScaleValidator(max_magnitude=1e6)
        issues = validator.validate(df.lazy())

        assert any(i.issue_type == "extreme_magnitude" for i in issues)

    def test_standard_scale_check(self):
        """Test checking for standard scaling."""
        # Not standard scaled
        df = pl.DataFrame(
            {
                "feature1": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        )
        validator = FeatureScaleValidator(check_standard_scale=True)
        issues = validator.validate(df.lazy())

        assert any(i.issue_type == "not_standard_scaled" for i in issues)

    def test_expected_scale_type(self):
        """Test expected scale type validation."""
        # Raw data (not normalized)
        df = pl.DataFrame(
            {
                "feature1": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        )
        validator = FeatureScaleValidator(expected_scale=ScaleType.STANDARD)
        issues = validator.validate(df.lazy())

        assert any(i.issue_type == "unexpected_scale_type" for i in issues)


class TestFeatureCorrelationMatrixValidator:
    """Tests for feature correlation validation."""

    def test_high_correlation_detection(self):
        """Test detection of highly correlated features."""
        n = 100
        x = np.random.randn(n)
        df = pl.DataFrame(
            {
                "feature1": x,
                "feature2": x + np.random.randn(n) * 0.01,  # Very correlated
                "feature3": np.random.randn(n),  # Independent
            }
        )
        validator = FeatureCorrelationMatrixValidator(max_correlation=0.9)
        issues = validator.validate(df.lazy())

        assert len(issues) > 0
        assert any(i.issue_type == "high_feature_correlation" for i in issues)

    def test_low_correlation_passes(self):
        """Test that uncorrelated features pass."""
        n = 100
        df = pl.DataFrame(
            {
                "feature1": np.random.randn(n),
                "feature2": np.random.randn(n),
                "feature3": np.random.randn(n),
            }
        )
        validator = FeatureCorrelationMatrixValidator(max_correlation=0.9)
        issues = validator.validate(df.lazy())

        high_corr = [i for i in issues if i.issue_type == "high_feature_correlation"]
        assert len(high_corr) == 0

    def test_warning_level_correlation(self):
        """Test warning level correlation detection."""
        n = 100
        x = np.random.randn(n)
        df = pl.DataFrame(
            {
                "feature1": x,
                "feature2": x + np.random.randn(n) * 0.5,  # Moderately correlated
            }
        )
        validator = FeatureCorrelationMatrixValidator(
            max_correlation=0.95, warn_correlation=0.5
        )
        issues = validator.validate(df.lazy())

        assert any(i.issue_type == "elevated_feature_correlation" for i in issues)

    def test_single_feature_no_error(self):
        """Test that single feature doesn't cause error."""
        df = pl.DataFrame({"feature1": [1.0, 2.0, 3.0, 4.0, 5.0]})
        validator = FeatureCorrelationMatrixValidator()
        issues = validator.validate(df.lazy())
        # Should not raise error with single feature
        assert isinstance(issues, list)


class TestTargetLeakageValidator:
    """Tests for target leakage validation."""

    def test_perfect_correlation_detection(self):
        """Test detection of perfect correlation with target."""
        n = 100
        target = np.random.randn(n)
        df = pl.DataFrame(
            {
                "feature1": target,  # Perfect leak
                "feature2": np.random.randn(n),
                "target": target,
            }
        )
        validator = TargetLeakageValidator(
            target_column="target", max_correlation=0.95
        )
        issues = validator.validate(df.lazy())

        assert len(issues) > 0
        assert any(i.issue_type == "target_leakage_detected" for i in issues)

    def test_no_leakage_passes(self):
        """Test that uncorrelated features pass."""
        n = 100
        df = pl.DataFrame(
            {
                "feature1": np.random.randn(n),
                "feature2": np.random.randn(n),
                "target": np.random.randn(n),
            }
        )
        validator = TargetLeakageValidator(
            target_column="target", max_correlation=0.95
        )
        issues = validator.validate(df.lazy())

        leakage = [i for i in issues if "leakage" in i.issue_type.lower()]
        assert len(leakage) == 0

    def test_warning_level_correlation(self):
        """Test warning level correlation detection."""
        n = 100
        target = np.random.randn(n)
        df = pl.DataFrame(
            {
                "feature1": target + np.random.randn(n) * 0.3,  # High but not perfect
                "target": target,
            }
        )
        validator = TargetLeakageValidator(
            target_column="target", max_correlation=0.99, warn_correlation=0.7
        )
        issues = validator.validate(df.lazy())

        assert any(i.issue_type == "potential_target_leakage" for i in issues)

    def test_missing_target_column(self):
        """Test handling of missing target column."""
        df = pl.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [1.0, 2.0, 3.0],
            }
        )
        validator = TargetLeakageValidator(target_column="target")
        issues = validator.validate(df.lazy())

        assert len(issues) > 0
        assert any(i.issue_type == "target_column_missing" for i in issues)

    def test_excludes_target_from_features(self):
        """Test that target is excluded from feature list."""
        n = 100
        target = np.random.randn(n)
        df = pl.DataFrame(
            {
                "feature1": np.random.randn(n),
                "target": target,
            }
        )
        validator = TargetLeakageValidator(target_column="target")
        issues = validator.validate(df.lazy())

        # Target correlation with itself should not be reported
        leakage = [i for i in issues if "leakage" in i.issue_type.lower()]
        assert len(leakage) == 0


class TestEdgeCases:
    """Test edge cases for ML feature validators."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pl.DataFrame({"feature1": [], "feature2": []}).cast(pl.Float64)
        validator = FeatureNullImpactValidator()
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_single_row(self):
        """Test with single row."""
        df = pl.DataFrame({"feature1": [1.0], "feature2": [2.0]})
        validator = FeatureCorrelationMatrixValidator(min_samples=1)
        issues = validator.validate(df.lazy())
        # Should not raise error
        assert isinstance(issues, list)

    def test_non_numeric_columns(self):
        """Test handling of non-numeric columns."""
        df = pl.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "category": ["a", "b", "c", "d", "e"],
            }
        )
        validator = FeatureScaleValidator()
        issues = validator.validate(df.lazy())
        # Should only analyze numeric columns
        assert isinstance(issues, list)

    def test_column_exclusion(self):
        """Test column exclusion."""
        df = pl.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "id": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        validator = FeatureNullImpactValidator(exclude_columns=["id"])
        issues = validator.validate(df.lazy())
        # id column should be excluded
        for issue in issues:
            assert "id" not in issue.column

    def test_specific_columns(self):
        """Test specifying specific columns."""
        df = pl.DataFrame(
            {
                "feature1": [1.0, None, None, None, 5.0],
                "feature2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature3": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        validator = FeatureNullImpactValidator(
            columns=["feature2", "feature3"], max_null_ratio=0.3
        )
        issues = validator.validate(df.lazy())
        # feature1 should be excluded
        null_issues = [i for i in issues if "null" in i.issue_type.lower()]
        assert len(null_issues) == 0
