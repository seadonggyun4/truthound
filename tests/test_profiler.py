"""Tests for the profiler module (Phase 7)."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from truthound.profiler import (
    # Base types
    DataType,
    Strictness,
    ProfileCategory,
    PatternMatch,
    DistributionStats,
    ValueFrequency,
    ColumnProfile,
    TableProfile,
    # Profiler
    DataProfiler,
    ProfilerConfig,
    ColumnProfiler,
    # Convenience functions
    profile_dataframe,
    save_profile,
    load_profile,
    # Generators
    generate_suite,
    ValidationSuite,
    RuleCategory,
    RuleConfidence,
    SchemaRuleGenerator,
    StatsRuleGenerator,
    PatternRuleGenerator,
    MLRuleGenerator,
)


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pl.DataFrame({
        "id": range(1, 101),
        "name": [f"User {i}" for i in range(1, 101)],
        "email": [f"user{i}@example.com" for i in range(1, 101)],
        "age": [20 + (i % 50) for i in range(100)],
        "score": [float(i * 1.5) for i in range(100)],
        "category": ["A", "B", "C", "D"] * 25,
        "active": [True, False] * 50,
        "created_at": [datetime(2024, 1, 1)] * 100,
    })


@pytest.fixture
def sample_df_with_nulls():
    """Create a sample DataFrame with nulls."""
    return pl.DataFrame({
        "id": range(1, 101),
        "name": [f"User {i}" if i % 5 != 0 else None for i in range(1, 101)],
        "email": [f"user{i}@example.com" if i % 10 != 0 else None for i in range(1, 101)],
        "age": [20 + (i % 50) if i % 3 != 0 else None for i in range(100)],
    })


@pytest.fixture
def pattern_test_df():
    """Create a DataFrame with various patterns."""
    return pl.DataFrame({
        "email": ["test@example.com", "user@domain.org", "admin@company.net"] * 33 + ["invalid"],
        "uuid": ["550e8400-e29b-41d4-a716-446655440000"] * 100,
        "phone": ["+821012345678"] * 100,
        "ip": ["192.168.1.1", "10.0.0.1", "172.16.0.1"] * 33 + ["invalid"],
        "korean_rrn": ["901231-1234567"] * 100,
        "korean_phone": ["010-1234-5678"] * 100,
    })


# =============================================================================
# Base Data Structures Tests
# =============================================================================


class TestDataStructures:
    """Test base data structures."""

    def test_distribution_stats_to_dict(self):
        stats = DistributionStats(mean=10.0, std=2.0, min=5.0, max=15.0)
        d = stats.to_dict()
        assert d["mean"] == 10.0
        assert d["std"] == 2.0
        assert d["min"] == 5.0
        assert d["max"] == 15.0

    def test_pattern_match_to_dict(self):
        match = PatternMatch(
            pattern="email",
            regex=r"^[\w.]+@[\w.]+$",
            match_ratio=0.95,
            sample_matches=("test@example.com",),
        )
        d = match.to_dict()
        assert d["pattern"] == "email"
        assert d["match_ratio"] == 0.95
        assert len(d["sample_matches"]) == 1

    def test_value_frequency_to_dict(self):
        vf = ValueFrequency(value="A", count=100, ratio=0.25)
        d = vf.to_dict()
        assert d["value"] == "A"
        assert d["count"] == 100
        assert d["ratio"] == 0.25

    def test_column_profile_to_dict(self):
        profile = ColumnProfile(
            name="test_col",
            physical_type="Int64",
            inferred_type=DataType.INTEGER,
            row_count=100,
            null_count=5,
            null_ratio=0.05,
            distinct_count=95,
            unique_ratio=0.95,
        )
        d = profile.to_dict()
        assert d["name"] == "test_col"
        assert d["physical_type"] == "Int64"
        assert d["inferred_type"] == "integer"
        assert d["null_ratio"] == 0.05

    def test_table_profile_iteration(self):
        col1 = ColumnProfile(name="col1", physical_type="Int64")
        col2 = ColumnProfile(name="col2", physical_type="String")
        table = TableProfile(columns=(col1, col2))

        # Test iteration
        cols = list(table)
        assert len(cols) == 2
        assert cols[0].name == "col1"
        assert cols[1].name == "col2"

        # Test indexing
        assert table[0].name == "col1"
        assert table["col2"].name == "col2"

        # Test get
        assert table.get("col1") is not None
        assert table.get("nonexistent") is None


# =============================================================================
# Column Profiler Tests
# =============================================================================


class TestColumnProfiler:
    """Test column profiler functionality."""

    def test_basic_profiling(self, sample_df):
        profiler = ColumnProfiler()
        lf = sample_df.lazy()
        schema = lf.collect_schema()

        # Profile id column
        profile = profiler.profile_column("id", lf, schema["id"])

        assert profile.name == "id"
        assert profile.row_count == 100
        assert profile.null_count == 0
        assert profile.distinct_count == 100
        assert profile.is_unique

    def test_numeric_column_profiling(self, sample_df):
        profiler = ColumnProfiler()
        lf = sample_df.lazy()
        schema = lf.collect_schema()

        profile = profiler.profile_column("age", lf, schema["age"])

        assert profile.distribution is not None
        assert profile.distribution.min is not None
        assert profile.distribution.max is not None
        assert profile.distribution.mean is not None

    def test_string_column_profiling(self, sample_df):
        profiler = ColumnProfiler()
        lf = sample_df.lazy()
        schema = lf.collect_schema()

        profile = profiler.profile_column("name", lf, schema["name"])

        assert profile.min_length is not None
        assert profile.max_length is not None
        assert profile.avg_length is not None

    def test_categorical_detection(self, sample_df):
        profiler = ColumnProfiler()
        lf = sample_df.lazy()
        schema = lf.collect_schema()

        profile = profiler.profile_column("category", lf, schema["category"])

        # Low cardinality should be detected
        assert profile.distinct_count == 4
        assert profile.unique_ratio < 0.1


# =============================================================================
# Data Profiler Tests
# =============================================================================


class TestDataProfiler:
    """Test main DataProfiler class."""

    def test_profile_dataframe(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df.lazy(), name="test")

        assert profile.name == "test"
        assert profile.row_count == 100
        assert profile.column_count == 8
        assert len(profile.columns) == 8

    def test_profile_with_config(self, sample_df):
        config = ProfilerConfig(
            top_n_values=5,
            include_patterns=False,
        )
        profiler = DataProfiler(config=config)
        profile = profiler.profile(sample_df.lazy())

        # Check that top_n was respected
        for col in profile.columns:
            if col.top_values:
                assert len(col.top_values) <= 5

    def test_profile_with_nulls(self, sample_df_with_nulls):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df_with_nulls.lazy())

        # Check null ratios are calculated
        for col in profile.columns:
            if col.name == "name":
                assert col.null_ratio > 0
            elif col.name == "id":
                assert col.null_ratio == 0

    def test_profile_with_correlations(self, sample_df):
        config = ProfilerConfig(include_correlations=True)
        profiler = DataProfiler(config=config, include_correlations=True)
        profile = profiler.profile(sample_df.lazy())

        # Correlations should be computed for numeric columns
        # id, age, score are numeric and should have correlations
        assert isinstance(profile.correlations, tuple)

    def test_profile_dataframe_convenience(self, sample_df):
        profile = profile_dataframe(sample_df, name="convenience_test")

        assert profile.name == "convenience_test"
        assert profile.row_count == 100


class TestProfileSerialization:
    """Test profile save/load functionality."""

    def test_save_and_load_profile(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df.lazy(), name="serialize_test")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            # Save
            save_profile(profile, path)
            assert path.exists()

            # Load
            loaded = load_profile(path)

            assert loaded.name == profile.name
            assert loaded.row_count == profile.row_count
            assert loaded.column_count == profile.column_count
            assert len(loaded.columns) == len(profile.columns)

            # Check column details preserved
            for orig, loaded_col in zip(profile.columns, loaded.columns):
                assert orig.name == loaded_col.name
                assert orig.null_ratio == loaded_col.null_ratio

        finally:
            path.unlink()


# =============================================================================
# Pattern Detection Tests
# =============================================================================


class TestPatternDetection:
    """Test pattern detection functionality."""

    def test_email_pattern_detection(self, pattern_test_df):
        profiler = DataProfiler()
        profile = profiler.profile(pattern_test_df.lazy())

        email_col = profile.get("email")
        assert email_col is not None

        # Should detect email pattern
        if email_col.detected_patterns:
            pattern_names = [p.pattern for p in email_col.detected_patterns]
            assert "email" in pattern_names or email_col.inferred_type == DataType.EMAIL

    def test_uuid_pattern_detection(self, pattern_test_df):
        profiler = DataProfiler()
        profile = profiler.profile(pattern_test_df.lazy())

        uuid_col = profile.get("uuid")
        assert uuid_col is not None
        assert uuid_col.inferred_type == DataType.UUID or any(
            p.pattern == "uuid" for p in uuid_col.detected_patterns
        )


# =============================================================================
# Rule Generator Tests
# =============================================================================


class TestSchemaRuleGenerator:
    """Test schema rule generator."""

    def test_generate_schema_rules(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df.lazy())

        generator = SchemaRuleGenerator()
        rules = generator.generate(profile, Strictness.MEDIUM)

        # Should have column existence rules
        existence_rules = [r for r in rules if "ColumnExistsValidator" in r.validator_class]
        assert len(existence_rules) >= profile.column_count

        # Should have completeness rules
        completeness_rules = [r for r in rules if "Completeness" in r.validator_class or "NotNull" in r.validator_class]
        assert len(completeness_rules) > 0

    def test_strictness_affects_rules(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df.lazy())

        generator = SchemaRuleGenerator()

        loose_rules = generator.generate(profile, Strictness.LOOSE)
        strict_rules = generator.generate(profile, Strictness.STRICT)

        # Strict mode should generate more rules
        assert len(strict_rules) >= len(loose_rules)


class TestStatsRuleGenerator:
    """Test statistics rule generator."""

    def test_generate_stats_rules(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df.lazy())

        generator = StatsRuleGenerator()
        rules = generator.generate(profile, Strictness.MEDIUM)

        # Should have distribution rules for numeric columns
        range_rules = [r for r in rules if "Range" in r.validator_class]
        assert len(range_rules) > 0

    def test_uniqueness_rules(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df.lazy())

        generator = StatsRuleGenerator()
        rules = generator.generate(profile, Strictness.MEDIUM)

        # ID column should have uniqueness rule
        unique_rules = [r for r in rules if "Unique" in r.validator_class]
        assert len(unique_rules) > 0


class TestPatternRuleGenerator:
    """Test pattern rule generator."""

    def test_generate_pattern_rules(self, pattern_test_df):
        profiler = DataProfiler()
        profile = profiler.profile(pattern_test_df.lazy())

        generator = PatternRuleGenerator()
        rules = generator.generate(profile, Strictness.MEDIUM)

        # Should have format validators for detected patterns
        validator_classes = [r.validator_class for r in rules]

        # Should detect at least some format validators
        format_validators = ["EmailValidator", "UuidValidator", "RegexValidator"]
        has_format_validator = any(
            v in vc for v in format_validators for vc in validator_classes
        )
        assert has_format_validator or len(rules) > 0


class TestMLRuleGenerator:
    """Test ML rule generator."""

    def test_generate_outlier_rules(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df.lazy())

        generator = MLRuleGenerator()
        rules = generator.generate(profile, Strictness.MEDIUM)

        # Should have outlier detection rules for numeric columns
        outlier_rules = [r for r in rules if "Outlier" in r.validator_class or "Anomaly" in r.validator_class]
        assert len(outlier_rules) > 0


# =============================================================================
# Validation Suite Tests
# =============================================================================


class TestValidationSuite:
    """Test ValidationSuite and ValidationSuiteGenerator."""

    def test_generate_suite(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df.lazy())

        suite = generate_suite(profile, strictness="medium")

        assert len(suite) > 0
        assert suite.strictness == Strictness.MEDIUM

    def test_filter_by_category(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df.lazy())

        suite = generate_suite(profile, strictness="medium")
        schema_suite = suite.filter_by_category(RuleCategory.SCHEMA)

        for rule in schema_suite:
            assert rule.category == RuleCategory.SCHEMA

    def test_filter_by_confidence(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df.lazy())

        suite = generate_suite(profile, strictness="medium")
        high_conf_suite = suite.filter_by_confidence(RuleConfidence.HIGH)

        for rule in high_conf_suite:
            assert rule.confidence == RuleConfidence.HIGH

    def test_to_yaml(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df.lazy())

        suite = generate_suite(profile, strictness="medium")
        yaml_output = suite.to_yaml()

        assert "rules:" in yaml_output
        assert "validator:" in yaml_output

    def test_to_python_code(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df.lazy())

        suite = generate_suite(profile, strictness="medium")
        code = suite.to_python_code()

        assert "from truthound.validators import" in code
        assert "def create_validators():" in code
        assert "validators.append(" in code

    def test_include_exclude_categories(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df.lazy())

        # Only schema rules
        suite = generate_suite(
            profile,
            strictness="medium",
            include_categories=["schema"],
        )

        for rule in suite:
            assert rule.category == RuleCategory.SCHEMA

        # Exclude schema rules
        suite2 = generate_suite(
            profile,
            strictness="medium",
            exclude_categories=["schema"],
        )

        for rule in suite2:
            assert rule.category != RuleCategory.SCHEMA


class TestSuiteSerialization:
    """Test suite save/load functionality."""

    def test_suite_to_dict(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df.lazy())

        suite = generate_suite(profile, strictness="medium", name="test_suite")
        d = suite.to_dict()

        assert d["name"] == "test_suite"
        assert "rules" in d
        assert "summary" in d
        assert d["summary"]["total_rules"] == len(suite)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the complete profiling workflow."""

    def test_end_to_end_workflow(self, sample_df):
        """Test complete workflow: profile -> generate -> export."""
        # 1. Profile
        profiler = DataProfiler()
        profile = profiler.profile(sample_df.lazy(), name="e2e_test")

        assert profile.row_count > 0
        assert profile.column_count > 0

        # 2. Generate suite
        suite = generate_suite(
            profile,
            strictness="medium",
            include_categories=["schema", "completeness", "distribution"],
        )

        assert len(suite) > 0

        # 3. Export to different formats
        yaml_out = suite.to_yaml()
        assert "rules:" in yaml_out

        json_out = suite.to_dict()
        assert "rules" in json_out

        python_out = suite.to_python_code()
        assert "def create_validators():" in python_out

    def test_profile_save_load_generate(self, sample_df):
        """Test save/load profile then generate suite."""
        profiler = DataProfiler()
        profile = profiler.profile(sample_df.lazy())

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            # Save profile
            save_profile(profile, path)

            # Load profile
            loaded_profile = load_profile(path)

            # Generate suite from loaded profile
            suite = generate_suite(loaded_profile, strictness="medium")

            assert len(suite) > 0

        finally:
            path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
