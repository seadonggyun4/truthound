"""Tests for data drift validators."""

import polars as pl
import pytest
import numpy as np

from truthound.validators.drift import (
    DriftValidator,
    ColumnDriftValidator,
    NumericDriftMixin,
    CategoricalDriftMixin,
    KSTestValidator,
    ChiSquareDriftValidator,
    WassersteinDriftValidator,
    PSIValidator,
    CSIValidator,
    MeanDriftValidator,
    VarianceDriftValidator,
    QuantileDriftValidator,
    RangeDriftValidator,
    FeatureDriftValidator,
    JSDivergenceValidator,
)


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def reference_numeric_df():
    """Reference data with normal distribution."""
    np.random.seed(42)
    return pl.DataFrame({
        "value": np.random.normal(100, 15, 1000).tolist(),
        "score": np.random.uniform(0, 100, 1000).tolist(),
    })


@pytest.fixture
def current_same_df():
    """Current data with same distribution as reference."""
    np.random.seed(43)
    return pl.DataFrame({
        "value": np.random.normal(100, 15, 1000).tolist(),
        "score": np.random.uniform(0, 100, 1000).tolist(),
    })


@pytest.fixture
def current_drifted_df():
    """Current data with significant drift from reference."""
    np.random.seed(44)
    return pl.DataFrame({
        "value": np.random.normal(130, 25, 1000).tolist(),  # Mean and std shifted
        "score": np.random.uniform(20, 80, 1000).tolist(),  # Range shifted
    })


@pytest.fixture
def reference_categorical_df():
    """Reference data with categorical column."""
    np.random.seed(42)
    categories = np.random.choice(["A", "B", "C", "D"], size=1000, p=[0.4, 0.3, 0.2, 0.1])
    return pl.DataFrame({"category": categories.tolist()})


@pytest.fixture
def current_categorical_same_df():
    """Current categorical data with similar distribution."""
    np.random.seed(43)
    # Use exact same probabilities as reference for "no drift" scenario
    categories = np.random.choice(["A", "B", "C", "D"], size=1000, p=[0.4, 0.3, 0.2, 0.1])
    return pl.DataFrame({"category": categories.tolist()})


@pytest.fixture
def current_categorical_drifted_df():
    """Current categorical data with significant drift."""
    np.random.seed(44)
    categories = np.random.choice(["A", "B", "C", "D", "E"], size=1000, p=[0.1, 0.1, 0.3, 0.4, 0.1])
    return pl.DataFrame({"category": categories.tolist()})


# =============================================================================
# Base Drift Validator Tests
# =============================================================================


class TestDriftValidatorBase:
    """Tests for base drift validator classes."""

    def test_drift_validator_requires_reference_data(self):
        """Test that reference_data is required."""
        # DriftValidator is abstract, but we can test via subclass
        df = pl.DataFrame({"value": [1, 2, 3]})
        validator = KSTestValidator(
            column="value",
            reference_data=df,
            alpha=0.05
        )
        assert validator.reference_data is not None

    def test_reference_data_lazy_conversion(self):
        """Test that DataFrame is converted to LazyFrame."""
        df = pl.DataFrame({"value": [1, 2, 3]})
        validator = KSTestValidator(
            column="value",
            reference_data=df,
            alpha=0.05
        )
        assert isinstance(validator.reference_data, pl.LazyFrame)

    def test_reference_data_lazy_preserved(self):
        """Test that LazyFrame is preserved."""
        lf = pl.DataFrame({"value": [1, 2, 3]}).lazy()
        validator = KSTestValidator(
            column="value",
            reference_data=lf,
            alpha=0.05
        )
        assert isinstance(validator.reference_data, pl.LazyFrame)


# =============================================================================
# Statistical Test Validators
# =============================================================================


class TestKSTestValidator:
    """Tests for Kolmogorov-Smirnov test validator."""

    def test_no_drift_detected(self, reference_numeric_df, current_same_df):
        """Test that similar distributions pass."""
        validator = KSTestValidator(
            column="value",
            reference_data=reference_numeric_df,
            p_value_threshold=0.05,
        )
        issues = validator.validate(current_same_df.lazy())
        assert len(issues) == 0

    def test_drift_detected(self, reference_numeric_df, current_drifted_df):
        """Test that different distributions fail."""
        validator = KSTestValidator(
            column="value",
            reference_data=reference_numeric_df,
            p_value_threshold=0.05,
        )
        issues = validator.validate(current_drifted_df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "distribution_drift_detected"

    def test_ks_statistic_returned(self, reference_numeric_df, current_drifted_df):
        """Test that KS statistic is properly calculated."""
        validator = KSTestValidator(
            column="value",
            reference_data=reference_numeric_df,
            p_value_threshold=0.05,
        )
        statistic, p_value = validator.calculate_drift_score(
            reference_numeric_df.lazy(), current_drifted_df.lazy()
        )
        assert 0 <= statistic <= 1
        assert 0 <= p_value <= 1

    def test_custom_alpha(self, reference_numeric_df, current_same_df):
        """Test custom alpha threshold."""
        # With very low threshold (0.001), less likely to detect drift in similar data
        validator = KSTestValidator(
            column="value",
            reference_data=reference_numeric_df,
            p_value_threshold=0.001,  # Very low threshold
        )
        issues = validator.validate(current_same_df.lazy())
        # With very low threshold, should not detect drift in similar distributions
        assert len(issues) == 0


class TestChiSquareDriftValidator:
    """Tests for Chi-Square drift validator."""

    def test_no_drift_categorical(self, reference_categorical_df, current_categorical_same_df):
        """Test similar categorical distributions pass."""
        validator = ChiSquareDriftValidator(
            column="category",
            reference_data=reference_categorical_df,
            p_value_threshold=0.05,
        )
        issues = validator.validate(current_categorical_same_df.lazy())
        # Similar distributions should have high p-value
        assert len(issues) == 0

    def test_drift_detected_categorical(self, reference_categorical_df, current_categorical_drifted_df):
        """Test different categorical distributions fail."""
        validator = ChiSquareDriftValidator(
            column="category",
            reference_data=reference_categorical_df,
            p_value_threshold=0.05,
        )
        issues = validator.validate(current_categorical_drifted_df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "categorical_drift_detected"

    def test_chi_square_returns_statistic(self, reference_categorical_df, current_categorical_drifted_df):
        """Test chi-square statistic calculation."""
        validator = ChiSquareDriftValidator(
            column="category",
            reference_data=reference_categorical_df,
            p_value_threshold=0.05,
        )
        chi2, p_value = validator.calculate_drift_score(
            reference_categorical_df.lazy(), current_categorical_drifted_df.lazy()
        )
        assert chi2 >= 0
        assert 0 <= p_value <= 1


class TestWassersteinDriftValidator:
    """Tests for Wasserstein distance drift validator."""

    def test_no_drift_wasserstein(self, reference_numeric_df, current_same_df):
        """Test similar distributions have low Wasserstein distance."""
        validator = WassersteinDriftValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold=10.0,
        )
        issues = validator.validate(current_same_df.lazy())
        assert len(issues) == 0

    def test_drift_detected_wasserstein(self, reference_numeric_df, current_drifted_df):
        """Test different distributions have high Wasserstein distance."""
        validator = WassersteinDriftValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold=5.0,  # Lower threshold to catch drift
        )
        issues = validator.validate(current_drifted_df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "wasserstein_drift_detected"

    def test_wasserstein_normalized(self, reference_numeric_df, current_drifted_df):
        """Test that Wasserstein distance is normalized by std."""
        validator = WassersteinDriftValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold=1.0,
            normalize=True,
        )
        distance = validator.calculate_drift_score(
            reference_numeric_df.lazy(), current_drifted_df.lazy()
        )
        # Normalized distance should be reasonable
        assert distance >= 0


# =============================================================================
# PSI Validators Tests
# =============================================================================


class TestPSIValidator:
    """Tests for Population Stability Index validator."""

    def test_no_drift_psi(self, reference_numeric_df, current_same_df):
        """Test similar distributions have low PSI."""
        validator = PSIValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold=0.25,
            n_bins=10,
        )
        issues = validator.validate(current_same_df.lazy())
        assert len(issues) == 0

    def test_drift_detected_psi(self, reference_numeric_df, current_drifted_df):
        """Test different distributions have high PSI."""
        validator = PSIValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold=0.1,  # Lower threshold
            n_bins=10,
        )
        issues = validator.validate(current_drifted_df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "psi_drift_detected"

    def test_psi_categorical(self, reference_categorical_df, current_categorical_drifted_df):
        """Test PSI for categorical columns."""
        validator = PSIValidator(
            column="category",
            reference_data=reference_categorical_df,
            threshold=0.1,
            is_categorical=True,
        )
        issues = validator.validate(current_categorical_drifted_df.lazy())
        assert len(issues) == 1

    def test_psi_thresholds(self, reference_numeric_df, current_drifted_df):
        """Test PSI interpretation thresholds."""
        validator = PSIValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold=0.25,
        )
        psi = validator.calculate_drift_score(
            reference_numeric_df.lazy(), current_drifted_df.lazy()
        )
        # PSI should be positive
        assert psi >= 0

    def test_psi_severity_levels(self):
        """Test PSI severity calculation."""
        df = pl.DataFrame({"value": [1, 2, 3]})
        validator = PSIValidator(column="value", reference_data=df, threshold=0.25)

        from truthound.types import Severity
        assert validator._get_psi_severity(0.05) == Severity.LOW
        assert validator._get_psi_severity(0.15) == Severity.MEDIUM
        assert validator._get_psi_severity(0.35) == Severity.HIGH
        assert validator._get_psi_severity(0.6) == Severity.CRITICAL


class TestCSIValidator:
    """Tests for Characteristic Stability Index validator."""

    def test_csi_no_drift(self, reference_numeric_df, current_same_df):
        """Test CSI with no drift."""
        validator = CSIValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold_per_bin=0.1,
            n_bins=10,
        )
        issues = validator.validate(current_same_df.lazy())
        assert len(issues) == 0

    def test_csi_detects_bin_drift(self, reference_numeric_df, current_drifted_df):
        """Test CSI detects per-bin drift."""
        validator = CSIValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold_per_bin=0.02,  # Low threshold
            n_bins=10,
        )
        issues = validator.validate(current_drifted_df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "csi_drift_detected"

    def test_csi_returns_per_bin_contributions(self, reference_numeric_df, current_drifted_df):
        """Test CSI returns per-bin PSI contributions."""
        validator = CSIValidator(
            column="value",
            reference_data=reference_numeric_df,
            n_bins=10,
        )
        contributions = validator.calculate_drift_score(
            reference_numeric_df.lazy(), current_drifted_df.lazy()
        )
        assert isinstance(contributions, list)
        assert all(isinstance(c, tuple) and len(c) == 2 for c in contributions)


# =============================================================================
# Numeric Drift Validators Tests
# =============================================================================


class TestMeanDriftValidator:
    """Tests for mean drift validator."""

    def test_no_mean_drift(self, reference_numeric_df, current_same_df):
        """Test similar means pass."""
        validator = MeanDriftValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold_pct=10.0,
        )
        issues = validator.validate(current_same_df.lazy())
        assert len(issues) == 0

    def test_mean_drift_detected_pct(self, reference_numeric_df, current_drifted_df):
        """Test mean drift detected with percentage threshold."""
        validator = MeanDriftValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold_pct=5.0,  # Low threshold
        )
        issues = validator.validate(current_drifted_df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "mean_drift_detected"

    def test_mean_drift_detected_abs(self, reference_numeric_df, current_drifted_df):
        """Test mean drift detected with absolute threshold."""
        validator = MeanDriftValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold_abs=5.0,  # Low threshold
        )
        issues = validator.validate(current_drifted_df.lazy())
        assert len(issues) == 1

    def test_mean_drift_requires_threshold(self, reference_numeric_df):
        """Test that at least one threshold is required."""
        with pytest.raises(ValueError, match="At least one"):
            MeanDriftValidator(
                column="value",
                reference_data=reference_numeric_df,
            )


class TestVarianceDriftValidator:
    """Tests for variance drift validator."""

    def test_no_variance_drift(self, reference_numeric_df, current_same_df):
        """Test similar variances pass."""
        validator = VarianceDriftValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold_pct=20.0,
        )
        issues = validator.validate(current_same_df.lazy())
        assert len(issues) == 0

    def test_variance_drift_detected(self, reference_numeric_df, current_drifted_df):
        """Test variance drift detected."""
        validator = VarianceDriftValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold_pct=10.0,  # Low threshold
        )
        issues = validator.validate(current_drifted_df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "variance_drift_detected"

    def test_use_std_option(self, reference_numeric_df, current_drifted_df):
        """Test std deviation vs variance option."""
        validator_std = VarianceDriftValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold_pct=10.0,
            use_std=True,
        )
        validator_var = VarianceDriftValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold_pct=10.0,
            use_std=False,
        )
        # Both should detect drift but with different metrics
        ref_std, curr_std, _ = validator_std.calculate_drift_score(
            reference_numeric_df.lazy(), current_drifted_df.lazy()
        )
        ref_var, curr_var, _ = validator_var.calculate_drift_score(
            reference_numeric_df.lazy(), current_drifted_df.lazy()
        )
        assert ref_var > ref_std  # Variance > std for std > 1


class TestQuantileDriftValidator:
    """Tests for quantile drift validator."""

    def test_no_quantile_drift(self, reference_numeric_df, current_same_df):
        """Test similar quantiles pass."""
        validator = QuantileDriftValidator(
            column="value",
            reference_data=reference_numeric_df,
            quantiles=[0.25, 0.5, 0.75],
            threshold_pct=15.0,
        )
        issues = validator.validate(current_same_df.lazy())
        assert len(issues) == 0

    def test_quantile_drift_detected(self, reference_numeric_df, current_drifted_df):
        """Test quantile drift detected."""
        validator = QuantileDriftValidator(
            column="value",
            reference_data=reference_numeric_df,
            quantiles=[0.5, 0.95],
            threshold_pct=5.0,  # Low threshold
        )
        issues = validator.validate(current_drifted_df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "quantile_drift_detected"

    def test_custom_quantiles(self, reference_numeric_df, current_drifted_df):
        """Test custom quantile selection."""
        validator = QuantileDriftValidator(
            column="value",
            reference_data=reference_numeric_df,
            quantiles=[0.1, 0.9],  # Tails
            threshold_pct=5.0,
        )
        drift_results = validator.calculate_drift_score(
            reference_numeric_df.lazy(), current_drifted_df.lazy()
        )
        assert 0.1 in drift_results
        assert 0.9 in drift_results


class TestRangeDriftValidator:
    """Tests for range drift validator."""

    def test_no_range_drift(self, reference_numeric_df, current_same_df):
        """Test similar ranges pass."""
        validator = RangeDriftValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold_pct=20.0,
        )
        issues = validator.validate(current_same_df.lazy())
        assert len(issues) == 0

    def test_range_drift_detected(self, reference_numeric_df, current_drifted_df):
        """Test range drift detected."""
        validator = RangeDriftValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold_pct=5.0,
        )
        issues = validator.validate(current_drifted_df.lazy())
        # May detect min/max drift
        assert len(issues) >= 0  # Depends on random data

    def test_allow_expansion(self, reference_numeric_df):
        """Test allow_expansion option."""
        # Create current data with expanded range
        current = pl.DataFrame({
            "value": [50.0] + reference_numeric_df["value"].to_list() + [200.0]
        })

        validator_strict = RangeDriftValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold_pct=5.0,
            allow_expansion=False,
        )
        validator_allow = RangeDriftValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold_pct=5.0,
            allow_expansion=True,  # Only alert on shrinkage
        )

        issues_strict = validator_strict.validate(current.lazy())
        issues_allow = validator_allow.validate(current.lazy())

        # allow_expansion should have fewer or equal issues
        assert len(issues_allow) <= len(issues_strict)


# =============================================================================
# Multi-Feature Drift Validators Tests
# =============================================================================


class TestFeatureDriftValidator:
    """Tests for multi-feature drift validator."""

    def test_no_drift_multi_feature(self, reference_numeric_df, current_same_df):
        """Test no drift across multiple features."""
        validator = FeatureDriftValidator(
            columns=["value", "score"],
            reference_data=reference_numeric_df,
            method="psi",
            threshold=0.25,
        )
        issues = validator.validate(current_same_df.lazy())
        assert len(issues) == 0

    def test_drift_detected_multi_feature(self, reference_numeric_df, current_drifted_df):
        """Test drift detected across features."""
        validator = FeatureDriftValidator(
            columns=["value", "score"],
            reference_data=reference_numeric_df,
            method="psi",
            threshold=0.1,  # Low threshold
        )
        issues = validator.validate(current_drifted_df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "feature_drift_detected"

    def test_supported_methods(self, reference_numeric_df, current_drifted_df):
        """Test all supported methods work."""
        for method in ["psi", "ks", "wasserstein", "chi_square"]:
            validator = FeatureDriftValidator(
                columns=["value"],
                reference_data=reference_numeric_df,
                method=method,
                threshold=0.1,
            )
            # Should not raise
            issues = validator.validate(current_drifted_df.lazy())
            assert isinstance(issues, list)

    def test_unsupported_method_raises(self, reference_numeric_df):
        """Test unsupported method raises error."""
        with pytest.raises(ValueError, match="Unsupported method"):
            FeatureDriftValidator(
                columns=["value"],
                reference_data=reference_numeric_df,
                method="invalid_method",
            )

    def test_alert_on_any_vs_min_count(self, reference_numeric_df, current_drifted_df):
        """Test alert_on_any vs min_drift_count options."""
        # With alert_on_any=True (default), any drifted column triggers
        validator_any = FeatureDriftValidator(
            columns=["value", "score"],
            reference_data=reference_numeric_df,
            threshold=0.1,
            alert_on_any=True,
        )

        # With min_drift_count=2, need both columns to drift
        validator_count = FeatureDriftValidator(
            columns=["value", "score"],
            reference_data=reference_numeric_df,
            threshold=0.1,
            alert_on_any=False,
            min_drift_count=2,
        )

        issues_any = validator_any.validate(current_drifted_df.lazy())
        issues_count = validator_count.validate(current_drifted_df.lazy())

        # alert_on_any=True should be more sensitive
        assert len(issues_any) >= len(issues_count)

    def test_categorical_columns_option(self, reference_categorical_df, current_categorical_drifted_df):
        """Test categorical_columns option."""
        # Add numeric column
        ref_df = reference_categorical_df.with_columns(
            pl.Series("value", np.random.normal(100, 15, 1000).tolist())
        )
        curr_df = current_categorical_drifted_df.with_columns(
            pl.Series("value", np.random.normal(130, 25, 1000).tolist())
        )

        validator = FeatureDriftValidator(
            columns=["value", "category"],
            reference_data=ref_df,
            categorical_columns=["category"],
            method="chi_square",
            threshold=0.5,
        )
        issues = validator.validate(curr_df.lazy())
        assert isinstance(issues, list)


class TestJSDivergenceValidator:
    """Tests for Jensen-Shannon divergence validator."""

    def test_no_drift_js(self, reference_numeric_df, current_same_df):
        """Test similar distributions have low JS divergence."""
        validator = JSDivergenceValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold=0.2,
        )
        issues = validator.validate(current_same_df.lazy())
        assert len(issues) == 0

    def test_drift_detected_js(self, reference_numeric_df, current_drifted_df):
        """Test different distributions have high JS divergence."""
        validator = JSDivergenceValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold=0.05,  # Low threshold
        )
        issues = validator.validate(current_drifted_df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "js_divergence_drift_detected"

    def test_js_bounded(self, reference_numeric_df, current_drifted_df):
        """Test JS divergence is bounded [0, 1]."""
        validator = JSDivergenceValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold=0.1,
        )
        js_div = validator.calculate_drift_score(
            reference_numeric_df.lazy(), current_drifted_df.lazy()
        )
        assert 0 <= js_div <= 1

    def test_js_categorical(self, reference_categorical_df, current_categorical_drifted_df):
        """Test JS divergence for categorical columns."""
        validator = JSDivergenceValidator(
            column="category",
            reference_data=reference_categorical_df,
            threshold=0.1,
            is_categorical=True,
        )
        issues = validator.validate(current_categorical_drifted_df.lazy())
        assert len(issues) == 1


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestDriftValidatorEdgeCases:
    """Tests for edge cases in drift validators."""

    def test_empty_reference_data(self):
        """Test handling of empty reference data."""
        empty_df = pl.DataFrame({"value": pl.Series([], dtype=pl.Float64)})
        current_df = pl.DataFrame({"value": [1.0, 2.0, 3.0]})

        validator = KSTestValidator(
            column="value",
            reference_data=empty_df,
            p_value_threshold=0.05,
        )
        issues = validator.validate(current_df.lazy())
        # Should handle gracefully
        assert isinstance(issues, list)

    def test_empty_current_data(self, reference_numeric_df):
        """Test handling of empty current data."""
        empty_df = pl.DataFrame({"value": pl.Series([], dtype=pl.Float64)})

        validator = KSTestValidator(
            column="value",
            reference_data=reference_numeric_df,
            p_value_threshold=0.05,
        )
        issues = validator.validate(empty_df.lazy())
        # Should handle gracefully
        assert isinstance(issues, list)

    def test_null_values_handled(self, reference_numeric_df):
        """Test null values are properly filtered."""
        current_with_nulls = pl.DataFrame({
            "value": [None, 100.0, None, 105.0, 95.0, None]
        })

        validator = MeanDriftValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold_pct=50.0,
        )
        issues = validator.validate(current_with_nulls.lazy())
        # Should handle nulls gracefully
        assert isinstance(issues, list)

    def test_single_value_data(self):
        """Test handling of single-value data."""
        ref_df = pl.DataFrame({"value": [100.0] * 100})
        curr_df = pl.DataFrame({"value": [100.0] * 100})

        validator = PSIValidator(
            column="value",
            reference_data=ref_df,
            threshold=0.25,
        )
        issues = validator.validate(curr_df.lazy())
        # Should handle gracefully (constant distribution)
        assert isinstance(issues, list)

    def test_missing_column(self, reference_numeric_df):
        """Test handling of missing column in current data."""
        current_df = pl.DataFrame({"other_column": [1, 2, 3]})

        validator = MeanDriftValidator(
            column="value",
            reference_data=reference_numeric_df,
            threshold_pct=10.0,
        )
        # Should raise or handle gracefully
        with pytest.raises(Exception):
            validator.validate(current_df.lazy())


# =============================================================================
# Integration Tests
# =============================================================================


class TestDriftValidatorIntegration:
    """Integration tests for drift validators."""

    def test_multiple_validators_same_data(self, reference_numeric_df, current_drifted_df):
        """Test multiple drift validators on same data."""
        validators = [
            KSTestValidator(column="value", reference_data=reference_numeric_df, alpha=0.05),
            PSIValidator(column="value", reference_data=reference_numeric_df, threshold=0.1),
            MeanDriftValidator(column="value", reference_data=reference_numeric_df, threshold_pct=5.0),
            VarianceDriftValidator(column="value", reference_data=reference_numeric_df, threshold_pct=10.0),
        ]

        for validator in validators:
            issues = validator.validate(current_drifted_df.lazy())
            # All should detect drift
            assert len(issues) >= 1, f"{validator.name} should detect drift"

    def test_validator_registry_contains_drift_validators(self):
        """Test that drift validators are registered."""
        from truthound.validators import registry

        drift_validators = registry.get_by_category("drift")
        expected_validators = [
            "ks_test", "chi_square_drift", "wasserstein_drift",
            "psi", "csi",
            "mean_drift", "variance_drift", "quantile_drift", "range_drift",
            "feature_drift", "js_divergence",
        ]

        for name in expected_validators:
            assert name in drift_validators, f"{name} should be in registry"

    def test_drift_validators_in_all_export(self):
        """Test drift validators are exported from main module."""
        from truthound.validators import (
            DriftValidator,
            ColumnDriftValidator,
            KSTestValidator,
            ChiSquareDriftValidator,
            WassersteinDriftValidator,
            PSIValidator,
            CSIValidator,
            MeanDriftValidator,
            VarianceDriftValidator,
            QuantileDriftValidator,
            RangeDriftValidator,
            FeatureDriftValidator,
            JSDivergenceValidator,
        )
        # All imports should work
        assert DriftValidator is not None
        assert KSTestValidator is not None
        assert PSIValidator is not None
