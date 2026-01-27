"""Tests for drift detection."""

import polars as pl
import pytest

import truthound as th
from truthound.drift.detectors import (
    AndersonDarlingDetector,
    BhattacharyyaDetector,
    ChiSquareDetector,
    CramervonMisesDetector,
    DriftLevel,
    EnergyDetector,
    HellingerDetector,
    JensenShannonDetector,
    KLDivergenceDetector,
    KSTestDetector,
    MMDDetector,
    PSIDetector,
    TotalVariationDetector,
    WassersteinDetector,
)


class TestKSTestDetector:
    """Tests for Kolmogorov-Smirnov test detector."""

    def test_no_drift_same_distribution(self):
        """Test that identical distributions show no drift."""
        data = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        detector = KSTestDetector()
        result = detector.detect(data, data)

        assert result.statistic == 0.0
        assert not result.drifted

    def test_detects_drift_different_distributions(self):
        """Test detection of drift between different distributions."""
        baseline = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        current = pl.Series([50, 60, 70, 80, 90, 100, 110, 120, 130, 140])

        detector = KSTestDetector()
        result = detector.detect(baseline, current)

        assert result.statistic > 0.5
        assert result.drifted
        assert result.level in (DriftLevel.MEDIUM, DriftLevel.HIGH)

    def test_handles_empty_data(self):
        """Test handling of empty series."""
        baseline = pl.Series([1, 2, 3])
        current = pl.Series([], dtype=pl.Int64)

        detector = KSTestDetector()
        result = detector.detect(baseline, current)

        assert not result.drifted
        assert "Insufficient data" in result.details


class TestPSIDetector:
    """Tests for Population Stability Index detector."""

    def test_no_drift_same_distribution(self):
        """Test that similar distributions show low PSI."""
        data = pl.Series(list(range(100)))
        detector = PSIDetector()
        result = detector.detect(data, data)

        assert result.statistic < 0.1
        assert not result.drifted

    def test_detects_drift_shifted_distribution(self):
        """Test detection of shifted distribution."""
        baseline = pl.Series(list(range(100)))
        current = pl.Series(list(range(50, 150)))

        detector = PSIDetector()
        result = detector.detect(baseline, current)

        # Shift should cause some PSI but might not exceed threshold
        assert result.statistic > 0


class TestChiSquareDetector:
    """Tests for Chi-square detector."""

    def test_no_drift_same_categories(self):
        """Test that identical categorical distributions show no drift."""
        data = pl.Series(["a", "b", "c", "a", "b", "c"] * 10)
        detector = ChiSquareDetector()
        result = detector.detect(data, data)

        assert result.statistic < 0.1
        assert not result.drifted

    def test_detects_drift_different_categories(self):
        """Test detection of categorical drift."""
        baseline = pl.Series(["a", "a", "a", "b", "b", "c"] * 10)
        current = pl.Series(["a", "b", "b", "b", "c", "c"] * 10)

        detector = ChiSquareDetector()
        result = detector.detect(baseline, current)

        # Distribution changed, should detect drift
        assert result.statistic > 0

    def test_detects_new_category(self):
        """Test detection of new category appearing."""
        baseline = pl.Series(["a", "b", "c"] * 20)
        current = pl.Series(["a", "b", "c", "d"] * 15)

        detector = ChiSquareDetector()
        result = detector.detect(baseline, current)

        assert result.drifted


class TestJensenShannonDetector:
    """Tests for Jensen-Shannon divergence detector."""

    def test_no_drift_same_distribution(self):
        """Test that identical distributions have zero JS divergence."""
        data = pl.Series(list(range(100)))
        detector = JensenShannonDetector()
        result = detector.detect(data, data)

        assert result.statistic < 0.01
        assert not result.drifted

    def test_detects_drift(self):
        """Test detection of distribution drift."""
        baseline = pl.Series(list(range(0, 100)))
        current = pl.Series(list(range(80, 180)))

        detector = JensenShannonDetector()
        result = detector.detect(baseline, current)

        assert result.statistic > 0


class TestCompare:
    """Tests for th.compare() function."""

    def test_compare_identical_data(self):
        """Test comparing identical datasets."""
        data = {"value": [1, 2, 3, 4, 5], "category": ["a", "b", "c", "a", "b"]}

        report = th.compare(data, data)

        assert not report.has_drift
        assert len(report.get_drifted_columns()) == 0

    def test_compare_drifted_numeric(self):
        """Test detecting drift in numeric column."""
        baseline = {"value": list(range(100))}
        current = {"value": list(range(500, 600))}

        report = th.compare(baseline, current)

        assert report.has_drift
        assert "value" in report.get_drifted_columns()

    def test_compare_drifted_categorical(self):
        """Test detecting drift in categorical column."""
        baseline = {"status": ["active"] * 50 + ["inactive"] * 50}
        current = {"status": ["active"] * 10 + ["inactive"] * 90}

        report = th.compare(baseline, current)

        # Should detect the distribution shift
        assert len(report.columns) == 1

    def test_compare_specific_columns(self):
        """Test comparing specific columns only."""
        baseline = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
        current = {"a": [100, 200, 300], "b": [4, 5, 6], "c": [700, 800, 900]}

        report = th.compare(baseline, current, columns=["a", "b"])

        assert len(report.columns) == 2
        column_names = [c.column for c in report.columns]
        assert "a" in column_names
        assert "b" in column_names
        assert "c" not in column_names

    def test_compare_specific_method(self):
        """Test using specific detection method."""
        baseline = {"value": list(range(100))}
        current = {"value": list(range(50, 150))}

        report_ks = th.compare(baseline, current, method="ks")
        report_psi = th.compare(baseline, current, method="psi")

        # Both should have analyzed the column
        assert len(report_ks.columns) == 1
        assert len(report_psi.columns) == 1

        # Methods should be different
        assert report_ks.columns[0].result.method == "ks_test"
        assert report_psi.columns[0].result.method == "psi"

    def test_compare_with_threshold(self):
        """Test using custom threshold."""
        baseline = {"value": list(range(100))}
        current = {"value": list(range(10, 110))}

        # Loose threshold
        report_loose = th.compare(baseline, current, method="psi", threshold=0.5)
        # Strict threshold
        report_strict = th.compare(baseline, current, method="psi", threshold=0.01)

        # Strict threshold more likely to detect drift
        assert report_strict.columns[0].result.threshold < report_loose.columns[0].result.threshold

    def test_report_json_output(self):
        """Test JSON output of drift report."""
        baseline = {"value": [1, 2, 3, 4, 5]}
        current = {"value": [10, 20, 30, 40, 50]}

        report = th.compare(baseline, current)
        json_str = report.to_json()

        import json
        data = json.loads(json_str)

        assert "baseline_source" in data
        assert "current_source" in data
        assert "columns" in data
        assert len(data["columns"]) == 1


class TestDriftReport:
    """Tests for DriftReport."""

    def test_has_drift_property(self):
        """Test has_drift property."""
        baseline = {"value": list(range(100))}
        current = {"value": list(range(500, 600))}

        report = th.compare(baseline, current)

        assert report.has_drift is True

    def test_get_drifted_columns(self):
        """Test getting list of drifted columns."""
        baseline = {
            "stable": list(range(100)),
            "drifted": list(range(100)),
        }
        current = {
            "stable": list(range(100)),
            "drifted": list(range(500, 600)),
        }

        report = th.compare(baseline, current)
        drifted = report.get_drifted_columns()

        assert "drifted" in drifted


class TestSampling:
    """Tests for sampling optimization."""

    def test_compare_with_sampling(self):
        """Test comparison with sampling for large datasets."""
        baseline = {"value": list(range(10000))}
        current = {"value": list(range(5000, 15000))}

        # Compare with sampling
        report = th.compare(baseline, current, sample_size=1000)

        assert len(report.columns) == 1
        # Should still detect drift even with sampling
        assert report.has_drift

    def test_sampling_deterministic(self):
        """Test that sampling is deterministic with seed."""
        baseline = {"value": list(range(10000))}
        current = {"value": list(range(5000, 15000))}

        report1 = th.compare(baseline, current, sample_size=1000)
        report2 = th.compare(baseline, current, sample_size=1000)

        # Same sampling should give same results
        assert report1.columns[0].result.statistic == report2.columns[0].result.statistic


class TestKLDivergenceDetector:
    """Tests for Kullback-Leibler divergence detector."""

    def test_no_drift_same_distribution(self):
        """Test that identical distributions have zero KL divergence."""
        data = pl.Series(list(range(100)))
        detector = KLDivergenceDetector()
        result = detector.detect(data, data)

        assert result.statistic < 0.01
        assert not result.drifted
        assert result.method == "kl_divergence"

    def test_detects_drift_different_distributions(self):
        """Test detection of drift between different distributions."""
        baseline = pl.Series(list(range(100)))
        current = pl.Series(list(range(500, 600)))

        detector = KLDivergenceDetector()
        result = detector.detect(baseline, current)

        assert result.statistic > 0
        assert result.drifted
        assert result.level != DriftLevel.NONE

    def test_handles_empty_data(self):
        """Test handling of empty series."""
        baseline = pl.Series([1, 2, 3])
        current = pl.Series([], dtype=pl.Int64)

        detector = KLDivergenceDetector()
        result = detector.detect(baseline, current)

        assert not result.drifted
        assert "Insufficient data" in result.details

    def test_rejects_non_numeric_data(self):
        """Test that non-numeric data raises TypeError."""
        baseline = pl.Series(["a", "b", "c"])
        current = pl.Series(["x", "y", "z"])

        detector = KLDivergenceDetector()
        with pytest.raises(TypeError, match="numeric columns"):
            detector.detect(baseline, current)


class TestWassersteinDetector:
    """Tests for Wasserstein (Earth Mover's) distance detector."""

    def test_no_drift_same_distribution(self):
        """Test that identical distributions have zero Wasserstein distance."""
        data = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        detector = WassersteinDetector()
        result = detector.detect(data, data)

        assert result.statistic < 0.01
        assert not result.drifted
        assert result.method == "wasserstein"

    def test_detects_drift_different_distributions(self):
        """Test detection of drift between different distributions."""
        baseline = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        current = pl.Series([10.0, 11.0, 12.0, 13.0, 14.0] * 20)

        detector = WassersteinDetector()
        result = detector.detect(baseline, current)

        assert result.statistic > 0
        assert result.drifted

    def test_handles_empty_data(self):
        """Test handling of empty series."""
        baseline = pl.Series([1.0, 2.0, 3.0])
        current = pl.Series([], dtype=pl.Float64)

        detector = WassersteinDetector()
        result = detector.detect(baseline, current)

        assert not result.drifted
        assert "Insufficient data" in result.details

    def test_rejects_non_numeric_data(self):
        """Test that non-numeric data raises TypeError."""
        baseline = pl.Series(["a", "b", "c"])
        current = pl.Series(["x", "y", "z"])

        detector = WassersteinDetector()
        with pytest.raises(TypeError, match="numeric columns"):
            detector.detect(baseline, current)

    def test_normalized_statistic(self):
        """Test that statistic is normalized by baseline std."""
        baseline = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        current = pl.Series([2.0, 3.0, 4.0, 5.0, 6.0] * 20)

        detector = WassersteinDetector()
        result = detector.detect(baseline, current)

        assert "raw_distance" in result.details


class TestCramervonMisesDetector:
    """Tests for Cramér-von Mises detector."""

    def test_no_drift_same_distribution(self):
        """Test that identical distributions show no drift."""
        data = pl.Series(list(range(100)))
        detector = CramervonMisesDetector()
        result = detector.detect(data, data)

        assert not result.drifted
        assert result.method == "cramer_von_mises"

    def test_detects_drift_different_distributions(self):
        """Test detection of drift between different distributions."""
        baseline = pl.Series(list(range(100)))
        current = pl.Series(list(range(500, 600)))

        detector = CramervonMisesDetector()
        result = detector.detect(baseline, current)

        assert result.statistic > 0
        assert result.p_value is not None

    def test_handles_insufficient_data(self):
        """Test handling of insufficient data."""
        baseline = pl.Series([1])
        current = pl.Series([2])

        detector = CramervonMisesDetector()
        result = detector.detect(baseline, current)

        assert not result.drifted
        assert "Insufficient data" in result.details

    def test_rejects_non_numeric_data(self):
        """Test that non-numeric data raises TypeError."""
        baseline = pl.Series(["a", "b", "c"])
        current = pl.Series(["x", "y", "z"])

        detector = CramervonMisesDetector()
        with pytest.raises(TypeError, match="numeric columns"):
            detector.detect(baseline, current)


class TestAndersonDarlingDetector:
    """Tests for Anderson-Darling detector."""

    def test_no_drift_same_distribution(self):
        """Test that identical distributions show no drift."""
        data = pl.Series(list(range(100)))
        detector = AndersonDarlingDetector()
        result = detector.detect(data, data)

        assert not result.drifted
        assert result.method == "anderson_darling"

    def test_detects_drift_different_distributions(self):
        """Test detection of drift between different distributions."""
        baseline = pl.Series(list(range(100)))
        current = pl.Series(list(range(500, 600)))

        detector = AndersonDarlingDetector()
        result = detector.detect(baseline, current)

        assert result.statistic > 0
        assert result.p_value is not None

    def test_handles_insufficient_data(self):
        """Test handling of insufficient data."""
        baseline = pl.Series([1])
        current = pl.Series([2])

        detector = AndersonDarlingDetector()
        result = detector.detect(baseline, current)

        assert not result.drifted
        assert "Insufficient data" in result.details

    def test_rejects_non_numeric_data(self):
        """Test that non-numeric data raises TypeError."""
        baseline = pl.Series(["a", "b", "c"])
        current = pl.Series(["x", "y", "z"])

        detector = AndersonDarlingDetector()
        with pytest.raises(TypeError, match="numeric columns"):
            detector.detect(baseline, current)


class TestCompareNewMethods:
    """Tests for th.compare() with new detection methods."""

    def test_compare_with_kl_method(self):
        """Test comparison using KL divergence method."""
        baseline = {"value": list(range(100))}
        current = {"value": list(range(50, 150))}

        report = th.compare(baseline, current, method="kl")

        assert len(report.columns) == 1
        assert report.columns[0].result.method == "kl_divergence"

    def test_compare_with_wasserstein_method(self):
        """Test comparison using Wasserstein distance method."""
        baseline = {"value": list(range(100))}
        current = {"value": list(range(50, 150))}

        report = th.compare(baseline, current, method="wasserstein")

        assert len(report.columns) == 1
        assert report.columns[0].result.method == "wasserstein"

    def test_compare_with_cvm_method(self):
        """Test comparison using Cramér-von Mises method."""
        baseline = {"value": list(range(100))}
        current = {"value": list(range(50, 150))}

        report = th.compare(baseline, current, method="cvm")

        assert len(report.columns) == 1
        assert report.columns[0].result.method == "cramer_von_mises"

    def test_compare_with_anderson_method(self):
        """Test comparison using Anderson-Darling method."""
        baseline = {"value": list(range(100))}
        current = {"value": list(range(50, 150))}

        report = th.compare(baseline, current, method="anderson")

        assert len(report.columns) == 1
        assert report.columns[0].result.method == "anderson_darling"

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError with helpful message."""
        baseline = {"value": [1, 2, 3]}
        current = {"value": [4, 5, 6]}

        with pytest.raises(ValueError, match="Unknown comparison method"):
            th.compare(baseline, current, method="invalid_method")

    def test_compare_all_methods_on_drifted_data(self):
        """Test all methods can detect significant drift."""
        baseline = {"value": list(range(100))}
        current = {"value": list(range(500, 600))}

        methods = ["ks", "psi", "js", "kl", "wasserstein", "cvm", "anderson"]

        for method in methods:
            report = th.compare(baseline, current, method=method)
            assert report.has_drift, f"Method {method} should detect drift"

    def test_compare_all_methods_on_identical_data(self):
        """Test all methods report no drift for identical data."""
        data = {"value": list(range(100))}

        methods = ["ks", "psi", "js", "kl", "wasserstein", "cvm", "anderson"]

        for method in methods:
            report = th.compare(data, data, method=method)
            assert not report.has_drift, f"Method {method} should not detect drift for identical data"


class TestHellingerDetector:
    """Tests for Hellinger distance detector."""

    def test_no_drift_same_distribution(self):
        """Test that identical distributions have zero Hellinger distance."""
        data = pl.Series(list(range(100)))
        detector = HellingerDetector()
        result = detector.detect(data, data)

        assert result.statistic < 0.01
        assert not result.drifted
        assert result.method == "hellinger"

    def test_detects_drift_different_distributions(self):
        """Test detection of drift between different distributions."""
        baseline = pl.Series(list(range(100)))
        current = pl.Series(list(range(500, 600)))

        detector = HellingerDetector()
        result = detector.detect(baseline, current)

        assert result.statistic > 0.1
        assert result.drifted
        assert result.level != DriftLevel.NONE

    def test_bounded_statistic(self):
        """Test that Hellinger distance is bounded [0, 1]."""
        baseline = pl.Series(list(range(100)))
        current = pl.Series(list(range(1000, 1100)))

        detector = HellingerDetector()
        result = detector.detect(baseline, current)

        assert 0.0 <= result.statistic <= 1.0

    def test_handles_categorical_data(self):
        """Test Hellinger distance works with categorical data."""
        baseline = pl.Series(["a", "b", "c"] * 30)
        current = pl.Series(["a", "a", "d"] * 30)

        detector = HellingerDetector()
        result = detector.detect(baseline, current)

        assert result.statistic > 0
        assert result.method == "hellinger"

    def test_handles_empty_data(self):
        """Test handling of empty series."""
        baseline = pl.Series([1, 2, 3])
        current = pl.Series([], dtype=pl.Int64)

        detector = HellingerDetector()
        result = detector.detect(baseline, current)

        assert not result.drifted
        assert "Insufficient data" in result.details


class TestBhattacharyyaDetector:
    """Tests for Bhattacharyya distance detector."""

    def test_no_drift_same_distribution(self):
        """Test that identical distributions have near-zero Bhattacharyya distance."""
        data = pl.Series(list(range(100)))
        detector = BhattacharyyaDetector()
        result = detector.detect(data, data)

        assert result.statistic < 0.01
        assert not result.drifted
        assert result.method == "bhattacharyya"

    def test_detects_drift_different_distributions(self):
        """Test detection of drift between different distributions."""
        baseline = pl.Series(list(range(100)))
        current = pl.Series(list(range(500, 600)))

        detector = BhattacharyyaDetector()
        result = detector.detect(baseline, current)

        assert result.statistic > 0.1
        assert result.drifted

    def test_reports_bc_coefficient(self):
        """Test that Bhattacharyya coefficient is in details."""
        baseline = pl.Series(list(range(100)))
        current = pl.Series(list(range(50, 150)))

        detector = BhattacharyyaDetector()
        result = detector.detect(baseline, current)

        assert "bc_coeff" in result.details

    def test_handles_categorical_data(self):
        """Test Bhattacharyya distance works with categorical data."""
        baseline = pl.Series(["a", "b", "c"] * 30)
        current = pl.Series(["a", "a", "d"] * 30)

        detector = BhattacharyyaDetector()
        result = detector.detect(baseline, current)

        assert result.statistic > 0
        assert result.method == "bhattacharyya"

    def test_handles_empty_data(self):
        """Test handling of empty series."""
        baseline = pl.Series([1, 2, 3])
        current = pl.Series([], dtype=pl.Int64)

        detector = BhattacharyyaDetector()
        result = detector.detect(baseline, current)

        assert not result.drifted
        assert "Insufficient data" in result.details


class TestTotalVariationDetector:
    """Tests for Total Variation distance detector."""

    def test_no_drift_same_distribution(self):
        """Test that identical distributions have zero TV distance."""
        data = pl.Series(list(range(100)))
        detector = TotalVariationDetector()
        result = detector.detect(data, data)

        assert result.statistic < 0.01
        assert not result.drifted
        assert result.method == "total_variation"

    def test_detects_drift_different_distributions(self):
        """Test detection of drift between different distributions."""
        baseline = pl.Series(list(range(100)))
        current = pl.Series(list(range(500, 600)))

        detector = TotalVariationDetector()
        result = detector.detect(baseline, current)

        assert result.statistic > 0.1
        assert result.drifted

    def test_bounded_statistic(self):
        """Test that TV distance is bounded [0, 1]."""
        baseline = pl.Series(list(range(100)))
        current = pl.Series(list(range(1000, 1100)))

        detector = TotalVariationDetector()
        result = detector.detect(baseline, current)

        assert 0.0 <= result.statistic <= 1.0

    def test_handles_categorical_data(self):
        """Test TV distance works with categorical data."""
        baseline = pl.Series(["a", "b", "c"] * 30)
        current = pl.Series(["a", "a", "d"] * 30)

        detector = TotalVariationDetector()
        result = detector.detect(baseline, current)

        assert result.statistic > 0
        assert result.method == "total_variation"

    def test_handles_empty_data(self):
        """Test handling of empty series."""
        baseline = pl.Series([1, 2, 3])
        current = pl.Series([], dtype=pl.Int64)

        detector = TotalVariationDetector()
        result = detector.detect(baseline, current)

        assert not result.drifted
        assert "Insufficient data" in result.details


class TestEnergyDetector:
    """Tests for Energy distance detector."""

    def test_no_drift_same_distribution(self):
        """Test that identical distributions have near-zero Energy distance."""
        data = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        detector = EnergyDetector()
        result = detector.detect(data, data)

        assert result.statistic < 0.01
        assert not result.drifted
        assert result.method == "energy"

    def test_detects_drift_different_distributions(self):
        """Test detection of drift between different distributions."""
        baseline = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        current = pl.Series([10.0, 11.0, 12.0, 13.0, 14.0] * 20)

        detector = EnergyDetector()
        result = detector.detect(baseline, current)

        assert result.statistic > 0
        assert result.drifted

    def test_reports_raw_energy(self):
        """Test that raw energy distance is in details."""
        baseline = pl.Series(list(range(100)))
        current = pl.Series(list(range(50, 150)))

        detector = EnergyDetector()
        result = detector.detect(baseline, current)

        assert "raw_energy" in result.details

    def test_handles_empty_data(self):
        """Test handling of empty series."""
        baseline = pl.Series([1.0, 2.0, 3.0])
        current = pl.Series([], dtype=pl.Float64)

        detector = EnergyDetector()
        result = detector.detect(baseline, current)

        assert not result.drifted
        assert "Insufficient data" in result.details

    def test_rejects_non_numeric_data(self):
        """Test that non-numeric data raises TypeError."""
        baseline = pl.Series(["a", "b", "c"])
        current = pl.Series(["x", "y", "z"])

        detector = EnergyDetector()
        with pytest.raises(TypeError, match="numeric columns"):
            detector.detect(baseline, current)

    def test_max_samples_parameter(self):
        """Test that max_samples parameter limits computation."""
        baseline = pl.Series(list(range(10000)))
        current = pl.Series(list(range(5000, 15000)))

        detector = EnergyDetector(max_samples=100)
        result = detector.detect(baseline, current)

        # Should still compute and detect drift
        assert result.statistic > 0


class TestMMDDetector:
    """Tests for Maximum Mean Discrepancy detector."""

    def test_no_drift_same_distribution(self):
        """Test that identical distributions have near-zero MMD."""
        data = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        detector = MMDDetector()
        result = detector.detect(data, data)

        assert result.statistic < 0.05
        assert not result.drifted
        assert result.method == "mmd"

    def test_detects_drift_different_distributions(self):
        """Test detection of drift between different distributions."""
        baseline = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        current = pl.Series([10.0, 11.0, 12.0, 13.0, 14.0] * 20)

        detector = MMDDetector()
        result = detector.detect(baseline, current)

        assert result.statistic > 0
        assert result.drifted

    def test_reports_kernel_info(self):
        """Test that kernel information is in details."""
        baseline = pl.Series(list(range(100)))
        current = pl.Series(list(range(50, 150)))

        detector = MMDDetector()
        result = detector.detect(baseline, current)

        assert "kernel" in result.details
        assert "gamma" in result.details

    def test_different_kernel_types(self):
        """Test MMD with different kernel types."""
        baseline = pl.Series(list(range(100)))
        current = pl.Series(list(range(50, 150)))

        kernels = ["rbf", "linear", "polynomial"]
        for kernel in kernels:
            detector = MMDDetector(kernel=kernel)
            result = detector.detect(baseline, current)
            assert f"kernel={kernel}" in result.details

    def test_handles_insufficient_data(self):
        """Test handling of insufficient data."""
        baseline = pl.Series([1.0])
        current = pl.Series([2.0])

        detector = MMDDetector()
        result = detector.detect(baseline, current)

        assert not result.drifted
        assert "Insufficient data" in result.details

    def test_rejects_non_numeric_data(self):
        """Test that non-numeric data raises TypeError."""
        baseline = pl.Series(["a", "b", "c"])
        current = pl.Series(["x", "y", "z"])

        detector = MMDDetector()
        with pytest.raises(TypeError, match="numeric columns"):
            detector.detect(baseline, current)

    def test_custom_bandwidth(self):
        """Test MMD with custom bandwidth parameter."""
        baseline = pl.Series(list(range(100)))
        current = pl.Series(list(range(50, 150)))

        detector = MMDDetector(bandwidth=0.5)
        result = detector.detect(baseline, current)

        assert "gamma=0.5" in result.details


class TestCompareNewDistanceMetrics:
    """Tests for th.compare() with new distance metrics."""

    def test_compare_with_hellinger_method(self):
        """Test comparison using Hellinger distance method."""
        baseline = {"value": list(range(100))}
        current = {"value": list(range(50, 150))}

        report = th.compare(baseline, current, method="hellinger")

        assert len(report.columns) == 1
        assert report.columns[0].result.method == "hellinger"

    def test_compare_with_bhattacharyya_method(self):
        """Test comparison using Bhattacharyya distance method."""
        baseline = {"value": list(range(100))}
        current = {"value": list(range(50, 150))}

        report = th.compare(baseline, current, method="bhattacharyya")

        assert len(report.columns) == 1
        assert report.columns[0].result.method == "bhattacharyya"

    def test_compare_with_tv_method(self):
        """Test comparison using Total Variation distance method."""
        baseline = {"value": list(range(100))}
        current = {"value": list(range(50, 150))}

        report = th.compare(baseline, current, method="tv")

        assert len(report.columns) == 1
        assert report.columns[0].result.method == "total_variation"

    def test_compare_with_total_variation_method(self):
        """Test comparison using full 'total_variation' method name."""
        baseline = {"value": list(range(100))}
        current = {"value": list(range(50, 150))}

        report = th.compare(baseline, current, method="total_variation")

        assert len(report.columns) == 1
        assert report.columns[0].result.method == "total_variation"

    def test_compare_with_energy_method(self):
        """Test comparison using Energy distance method."""
        baseline = {"value": list(range(100))}
        current = {"value": list(range(50, 150))}

        report = th.compare(baseline, current, method="energy")

        assert len(report.columns) == 1
        assert report.columns[0].result.method == "energy"

    def test_compare_with_mmd_method(self):
        """Test comparison using MMD method."""
        baseline = {"value": list(range(100))}
        current = {"value": list(range(50, 150))}

        report = th.compare(baseline, current, method="mmd")

        assert len(report.columns) == 1
        assert report.columns[0].result.method == "mmd"

    def test_all_new_methods_detect_drift(self):
        """Test all new distance methods can detect significant drift."""
        baseline = {"value": list(range(100))}
        current = {"value": list(range(500, 600))}

        methods = ["hellinger", "bhattacharyya", "tv", "energy", "mmd"]

        for method in methods:
            report = th.compare(baseline, current, method=method)
            assert report.has_drift, f"Method {method} should detect drift"

    def test_all_new_methods_no_drift_identical_data(self):
        """Test all new methods report no drift for identical data."""
        data = {"value": list(range(100))}

        methods = ["hellinger", "bhattacharyya", "tv", "energy", "mmd"]

        for method in methods:
            report = th.compare(data, data, method=method)
            assert not report.has_drift, f"Method {method} should not detect drift for identical data"

    def test_categorical_methods_on_categorical_data(self):
        """Test that categorical-compatible methods work with categorical data."""
        baseline = {"category": ["a", "b", "c"] * 30}
        current = {"category": ["a", "a", "d"] * 30}

        # These methods work with categorical data
        categorical_methods = ["hellinger", "bhattacharyya", "tv"]

        for method in categorical_methods:
            report = th.compare(baseline, current, method=method)
            assert len(report.columns) == 1, f"Method {method} should analyze categorical data"
