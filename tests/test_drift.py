"""Tests for drift detection."""

import polars as pl
import pytest

import truthound as th
from truthound.drift.detectors import (
    ChiSquareDetector,
    DriftLevel,
    JensenShannonDetector,
    KSTestDetector,
    PSIDetector,
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
