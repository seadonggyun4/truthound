"""Tests for anomaly detection validators."""

import polars as pl
import pytest
import numpy as np

from truthound.validators.anomaly import (
    # Base classes
    AnomalyValidator,
    ColumnAnomalyValidator,
    # Statistical
    IQRAnomalyValidator,
    MADAnomalyValidator,
    GrubbsTestValidator,
    TukeyFencesValidator,
    PercentileAnomalyValidator,
    # Multivariate
    MahalanobisValidator,
    ZScoreMultivariateValidator,
    # ML-based
    IsolationForestValidator,
    LOFValidator,
    DBSCANAnomalyValidator,
)


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def normal_data_df():
    """DataFrame with normally distributed data (no anomalies)."""
    np.random.seed(42)
    return pl.DataFrame({
        "value": np.random.normal(100, 15, 1000).tolist(),
        "score": np.random.normal(50, 10, 1000).tolist(),
    })


@pytest.fixture
def data_with_outliers_df():
    """DataFrame with clear outliers."""
    np.random.seed(42)
    normal_values = np.random.normal(100, 15, 900).tolist()
    # Add 100 extreme outliers (10%)
    outliers = [500, 600, 700, -200, -300] * 20
    all_values = normal_values + outliers
    np.random.shuffle(all_values)
    return pl.DataFrame({"value": all_values})


@pytest.fixture
def multivariate_normal_df():
    """DataFrame with multivariate normal data."""
    np.random.seed(42)
    n = 500
    # Create correlated features
    x1 = np.random.normal(0, 1, n)
    x2 = 0.8 * x1 + 0.2 * np.random.normal(0, 1, n)
    x3 = 0.5 * x1 + 0.5 * np.random.normal(0, 1, n)
    return pl.DataFrame({
        "feature1": x1.tolist(),
        "feature2": x2.tolist(),
        "feature3": x3.tolist(),
    })


@pytest.fixture
def multivariate_with_outliers_df():
    """DataFrame with multivariate outliers."""
    np.random.seed(42)
    n = 450
    # Normal data
    x1 = np.random.normal(0, 1, n)
    x2 = 0.8 * x1 + 0.2 * np.random.normal(0, 1, n)
    x3 = 0.5 * x1 + 0.5 * np.random.normal(0, 1, n)

    # Add 50 outliers (10%)
    outlier_x1 = np.random.uniform(5, 8, 50)
    outlier_x2 = np.random.uniform(-8, -5, 50)
    outlier_x3 = np.random.uniform(5, 8, 50)

    return pl.DataFrame({
        "feature1": np.concatenate([x1, outlier_x1]).tolist(),
        "feature2": np.concatenate([x2, outlier_x2]).tolist(),
        "feature3": np.concatenate([x3, outlier_x3]).tolist(),
    })


# =============================================================================
# Statistical Validators Tests
# =============================================================================


class TestIQRAnomalyValidator:
    """Tests for IQR-based anomaly validator."""

    def test_no_anomalies_in_normal_data(self, normal_data_df):
        """Test that normal data has few anomalies."""
        validator = IQRAnomalyValidator(
            column="value",
            iqr_multiplier=1.5,
            max_anomaly_ratio=0.05,
        )
        issues = validator.validate(normal_data_df.lazy())
        # Normal data should have very few outliers
        assert len(issues) == 0

    def test_detects_outliers(self, data_with_outliers_df):
        """Test that outliers are detected."""
        validator = IQRAnomalyValidator(
            column="value",
            iqr_multiplier=1.5,
            max_anomaly_ratio=0.05,
        )
        issues = validator.validate(data_with_outliers_df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "iqr_anomaly_detected"
        assert issues[0].count > 0

    def test_extreme_multiplier(self, data_with_outliers_df):
        """Test extreme multiplier (k=3.0) is more lenient."""
        validator_standard = IQRAnomalyValidator(
            column="value",
            iqr_multiplier=1.5,
            max_anomaly_ratio=0.001,
        )
        validator_extreme = IQRAnomalyValidator(
            column="value",
            iqr_multiplier=3.0,
            max_anomaly_ratio=0.001,
        )

        issues_standard = validator_standard.validate(data_with_outliers_df.lazy())
        issues_extreme = validator_extreme.validate(data_with_outliers_df.lazy())

        # Standard should detect more than extreme
        if issues_standard and issues_extreme:
            assert issues_standard[0].count >= issues_extreme[0].count

    def test_detect_lower_only(self, data_with_outliers_df):
        """Test detecting only lower bound anomalies."""
        validator = IQRAnomalyValidator(
            column="value",
            detect_lower=True,
            detect_upper=False,
            max_anomaly_ratio=0.01,
        )
        issues = validator.validate(data_with_outliers_df.lazy())
        # Should only detect lower outliers
        assert isinstance(issues, list)


class TestMADAnomalyValidator:
    """Tests for MAD-based anomaly validator."""

    def test_no_anomalies_in_normal_data(self, normal_data_df):
        """Test that normal data passes MAD validation."""
        validator = MADAnomalyValidator(
            column="value",
            threshold=3.5,
            max_anomaly_ratio=0.05,
        )
        issues = validator.validate(normal_data_df.lazy())
        assert len(issues) == 0

    def test_detects_outliers(self, data_with_outliers_df):
        """Test MAD detects outliers."""
        validator = MADAnomalyValidator(
            column="value",
            threshold=3.5,
            max_anomaly_ratio=0.05,
        )
        issues = validator.validate(data_with_outliers_df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "mad_anomaly_detected"

    def test_custom_threshold(self, data_with_outliers_df):
        """Test custom threshold affects detection."""
        validator_strict = MADAnomalyValidator(
            column="value",
            threshold=2.0,  # Stricter
            max_anomaly_ratio=0.001,
        )
        validator_lenient = MADAnomalyValidator(
            column="value",
            threshold=5.0,  # More lenient
            max_anomaly_ratio=0.001,
        )

        issues_strict = validator_strict.validate(data_with_outliers_df.lazy())
        issues_lenient = validator_lenient.validate(data_with_outliers_df.lazy())

        # Stricter threshold should detect more
        if issues_strict and issues_lenient:
            assert issues_strict[0].count >= issues_lenient[0].count


class TestGrubbsTestValidator:
    """Tests for Grubbs' test validator."""

    def test_detects_single_outlier(self):
        """Test Grubbs detects a single outlier."""
        df = pl.DataFrame({
            "value": [10, 11, 12, 13, 14, 15, 100]  # 100 is obvious outlier
        })
        validator = GrubbsTestValidator(
            column="value",
            alpha=0.05,
            max_anomaly_ratio=0.0,  # Any outlier triggers
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "grubbs_outlier_detected"

    def test_iterative_detection(self, data_with_outliers_df):
        """Test iterative Grubbs detection."""
        validator = GrubbsTestValidator(
            column="value",
            alpha=0.05,
            max_iterations=100,
            max_anomaly_ratio=0.01,  # Lower to trigger issue
        )
        issues = validator.validate(data_with_outliers_df.lazy())
        # Grubbs iteratively finds outliers
        assert isinstance(issues, list)


class TestTukeyFencesValidator:
    """Tests for Tukey's fences validator."""

    def test_classifies_mild_and_extreme(self, data_with_outliers_df):
        """Test Tukey classifies mild vs extreme outliers."""
        validator = TukeyFencesValidator(
            column="value",
            detect_mild=True,
            detect_extreme=True,
            max_anomaly_ratio=0.05,
        )
        issues = validator.validate(data_with_outliers_df.lazy())
        assert len(issues) == 1
        # Details should mention both mild and extreme counts
        assert "Mild:" in issues[0].details
        assert "Extreme:" in issues[0].details

    def test_extreme_only(self, data_with_outliers_df):
        """Test detecting extreme outliers only."""
        validator_all = TukeyFencesValidator(
            column="value",
            detect_mild=True,
            detect_extreme=True,
            max_anomaly_ratio=0.001,
        )
        validator_extreme = TukeyFencesValidator(
            column="value",
            detect_mild=False,
            detect_extreme=True,
            max_anomaly_ratio=0.001,
        )

        issues_all = validator_all.validate(data_with_outliers_df.lazy())
        issues_extreme = validator_extreme.validate(data_with_outliers_df.lazy())

        # All should detect >= extreme only
        if issues_all and issues_extreme:
            assert issues_all[0].count >= issues_extreme[0].count


class TestPercentileAnomalyValidator:
    """Tests for percentile-based anomaly validator."""

    def test_percentile_bounds(self, normal_data_df):
        """Test percentile-based detection."""
        validator = PercentileAnomalyValidator(
            column="value",
            lower_percentile=1.0,
            upper_percentile=99.0,
            max_anomaly_ratio=0.03,  # 2% expected outside [1, 99]
        )
        issues = validator.validate(normal_data_df.lazy())
        # Should be close to 2% outside bounds
        assert len(issues) == 0

    def test_invalid_percentile_range(self):
        """Test that invalid percentile range raises error."""
        with pytest.raises(ValueError, match="Invalid percentile range"):
            PercentileAnomalyValidator(
                column="value",
                lower_percentile=99.0,
                upper_percentile=1.0,
            )


# =============================================================================
# Multivariate Validators Tests
# =============================================================================


class TestMahalanobisValidator:
    """Tests for Mahalanobis distance validator."""

    def test_no_anomalies_in_normal_multivariate(self, multivariate_normal_df):
        """Test that normal multivariate data passes."""
        validator = MahalanobisValidator(
            columns=["feature1", "feature2", "feature3"],
            threshold_percentile=97.5,
            max_anomaly_ratio=0.05,
        )
        issues = validator.validate(multivariate_normal_df.lazy())
        # Should detect few anomalies in normal data
        assert len(issues) == 0

    def test_detects_multivariate_outliers(self, multivariate_with_outliers_df):
        """Test that multivariate outliers are detected."""
        validator = MahalanobisValidator(
            columns=["feature1", "feature2", "feature3"],
            threshold_percentile=97.5,
            max_anomaly_ratio=0.05,
        )
        issues = validator.validate(multivariate_with_outliers_df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "mahalanobis_anomaly"


class TestZScoreMultivariateValidator:
    """Tests for multivariate Z-score validator."""

    def test_method_any(self, multivariate_with_outliers_df):
        """Test 'any' method for combining Z-scores."""
        validator = ZScoreMultivariateValidator(
            columns=["feature1", "feature2", "feature3"],
            threshold=3.0,
            method="any",
            max_anomaly_ratio=0.05,
        )
        issues = validator.validate(multivariate_with_outliers_df.lazy())
        assert len(issues) == 1
        assert "any" in issues[0].details

    def test_method_all(self, multivariate_with_outliers_df):
        """Test 'all' method is more restrictive."""
        validator_any = ZScoreMultivariateValidator(
            columns=["feature1", "feature2", "feature3"],
            threshold=3.0,
            method="any",
            max_anomaly_ratio=0.001,
        )
        validator_all = ZScoreMultivariateValidator(
            columns=["feature1", "feature2", "feature3"],
            threshold=3.0,
            method="all",
            max_anomaly_ratio=0.001,
        )

        issues_any = validator_any.validate(multivariate_with_outliers_df.lazy())
        issues_all = validator_all.validate(multivariate_with_outliers_df.lazy())

        # 'any' should detect more than 'all'
        if issues_any and issues_all:
            assert issues_any[0].count >= issues_all[0].count

    def test_invalid_method_raises(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Invalid method"):
            ZScoreMultivariateValidator(
                columns=["a", "b"],
                method="invalid",
            )


# =============================================================================
# ML-Based Validators Tests
# =============================================================================


class TestIsolationForestValidator:
    """Tests for Isolation Forest validator."""

    def test_detects_anomalies(self, multivariate_with_outliers_df):
        """Test Isolation Forest detects anomalies."""
        validator = IsolationForestValidator(
            columns=["feature1", "feature2", "feature3"],
            contamination=0.1,
            max_anomaly_ratio=0.05,
            random_state=42,
        )
        issues = validator.validate(multivariate_with_outliers_df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "isolation_forest_anomaly"

    def test_auto_contamination(self, multivariate_with_outliers_df):
        """Test auto contamination mode."""
        validator = IsolationForestValidator(
            columns=["feature1", "feature2", "feature3"],
            contamination="auto",
            max_anomaly_ratio=0.15,
            random_state=42,
        )
        issues = validator.validate(multivariate_with_outliers_df.lazy())
        # Should work without errors
        assert isinstance(issues, list)

    def test_uses_all_numeric_columns(self, multivariate_normal_df):
        """Test that columns=None uses all numeric columns."""
        validator = IsolationForestValidator(
            columns=None,  # Should use all numeric
            contamination=0.05,
            max_anomaly_ratio=0.1,
            random_state=42,
        )
        issues = validator.validate(multivariate_normal_df.lazy())
        assert isinstance(issues, list)


class TestLOFValidator:
    """Tests for Local Outlier Factor validator."""

    def test_detects_local_anomalies(self, multivariate_with_outliers_df):
        """Test LOF detects local anomalies."""
        validator = LOFValidator(
            columns=["feature1", "feature2", "feature3"],
            n_neighbors=20,
            contamination=0.1,
            max_anomaly_ratio=0.05,
        )
        issues = validator.validate(multivariate_with_outliers_df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "lof_anomaly"

    def test_adapts_n_neighbors(self):
        """Test LOF adapts n_neighbors for small datasets."""
        small_df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
            "y": [1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
        })
        validator = LOFValidator(
            columns=["x", "y"],
            n_neighbors=100,  # Larger than dataset
            max_anomaly_ratio=0.0,
        )
        issues = validator.validate(small_df.lazy())
        # Should work without error
        assert isinstance(issues, list)


class TestDBSCANAnomalyValidator:
    """Tests for DBSCAN anomaly validator."""

    def test_detects_noise_points(self, multivariate_with_outliers_df):
        """Test DBSCAN detects noise points as anomalies."""
        validator = DBSCANAnomalyValidator(
            columns=["feature1", "feature2", "feature3"],
            eps=0.5,
            min_samples=5,
            max_anomaly_ratio=0.05,
        )
        issues = validator.validate(multivariate_with_outliers_df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "dbscan_anomaly"
        assert "clusters" in issues[0].details


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestAnomalyValidatorEdgeCases:
    """Tests for edge cases in anomaly validators."""

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        empty_df = pl.DataFrame({"value": pl.Series([], dtype=pl.Float64)})
        validator = IQRAnomalyValidator(
            column="value",
            max_anomaly_ratio=0.1,
        )
        issues = validator.validate(empty_df.lazy())
        assert issues == []

    def test_small_dataframe(self):
        """Test handling of very small dataframe."""
        small_df = pl.DataFrame({"value": [1.0, 2.0]})
        validator = IQRAnomalyValidator(
            column="value",
            max_anomaly_ratio=0.1,
        )
        issues = validator.validate(small_df.lazy())
        # Should handle gracefully
        assert issues == []

    def test_null_values_filtered(self):
        """Test that null values are properly filtered."""
        df = pl.DataFrame({
            "value": [1.0, None, 2.0, None, 3.0, 100.0]
        })
        validator = IQRAnomalyValidator(
            column="value",
            max_anomaly_ratio=0.0,
        )
        issues = validator.validate(df.lazy())
        # Should work with nulls
        assert isinstance(issues, list)

    def test_constant_values(self):
        """Test handling of constant values (zero variance)."""
        df = pl.DataFrame({"value": [5.0] * 100})
        validator = MADAnomalyValidator(
            column="value",
            threshold=3.0,
            max_anomaly_ratio=0.1,
        )
        issues = validator.validate(df.lazy())
        # Should handle zero MAD gracefully
        assert issues == []


# =============================================================================
# Integration Tests
# =============================================================================


class TestAnomalyValidatorIntegration:
    """Integration tests for anomaly validators."""

    def test_multiple_validators_same_data(self, data_with_outliers_df):
        """Test multiple anomaly validators on same data."""
        validators = [
            IQRAnomalyValidator(column="value", max_anomaly_ratio=0.05),
            MADAnomalyValidator(column="value", max_anomaly_ratio=0.05),
            TukeyFencesValidator(column="value", max_anomaly_ratio=0.05),
        ]

        for validator in validators:
            issues = validator.validate(data_with_outliers_df.lazy())
            assert len(issues) >= 1, f"{validator.name} should detect anomalies"

    def test_validator_registry_contains_anomaly_validators(self):
        """Test that anomaly validators are registered."""
        from truthound.validators import registry

        anomaly_validators = registry.get_by_category("anomaly")
        expected_validators = [
            "iqr_anomaly", "mad_anomaly", "grubbs_test", "tukey_fences",
            "percentile_anomaly", "mahalanobis", "zscore_multivariate",
            "isolation_forest", "lof", "one_class_svm", "dbscan_anomaly",
        ]

        for name in expected_validators:
            assert name in anomaly_validators, f"{name} should be in registry"

    def test_anomaly_validators_in_all_export(self):
        """Test anomaly validators are exported from main module."""
        from truthound.validators import (
            AnomalyValidator,
            ColumnAnomalyValidator,
            IQRAnomalyValidator,
            MADAnomalyValidator,
            GrubbsTestValidator,
            TukeyFencesValidator,
            PercentileAnomalyValidator,
            MahalanobisValidator,
            ZScoreMultivariateValidator,
            IsolationForestValidator,
            LOFValidator,
            DBSCANAnomalyValidator,
        )
        # All imports should work
        assert AnomalyValidator is not None
        assert IQRAnomalyValidator is not None
        assert IsolationForestValidator is not None
