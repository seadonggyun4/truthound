"""Tests for Quality Reporter filters."""

import pytest
from dataclasses import dataclass
from typing import Any

from truthound.profiler.quality import (
    QualityLevel,
    QualityMetrics,
    RuleType,
    RuleQualityScore,
)
from truthound.reporters.quality.filters import (
    QualityFilter,
    LevelFilter,
    MetricFilter,
    ConfidenceFilter,
    ColumnFilter,
    RuleTypeFilter,
    RecommendationFilter,
    CustomFilter,
    CompositeFilter,
    NotFilter,
    AllOfFilter,
    AnyOfFilter,
    BaseQualityFilter,
    QUALITY_LEVEL_ORDER,
    get_level_value,
)
from truthound.reporters.quality.config import QualityFilterConfig


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_metrics_excellent() -> QualityMetrics:
    """Create excellent quality metrics."""
    return QualityMetrics(
        precision=0.96,
        recall=0.95,
        f1_score=0.955,
        accuracy=0.97,
        confidence=0.9,
        quality_level=QualityLevel.EXCELLENT,
    )


@pytest.fixture
def sample_metrics_good() -> QualityMetrics:
    """Create good quality metrics."""
    return QualityMetrics(
        precision=0.88,
        recall=0.86,
        f1_score=0.87,
        accuracy=0.89,
        confidence=0.85,
        quality_level=QualityLevel.GOOD,
    )


@pytest.fixture
def sample_metrics_acceptable() -> QualityMetrics:
    """Create acceptable quality metrics."""
    return QualityMetrics(
        precision=0.75,
        recall=0.72,
        f1_score=0.735,
        accuracy=0.78,
        confidence=0.7,
        quality_level=QualityLevel.ACCEPTABLE,
    )


@pytest.fixture
def sample_metrics_poor() -> QualityMetrics:
    """Create poor quality metrics."""
    return QualityMetrics(
        precision=0.55,
        recall=0.52,
        f1_score=0.534,
        accuracy=0.6,
        confidence=0.5,
        quality_level=QualityLevel.POOR,
    )


@pytest.fixture
def sample_metrics_unacceptable() -> QualityMetrics:
    """Create unacceptable quality metrics."""
    return QualityMetrics(
        precision=0.35,
        recall=0.32,
        f1_score=0.334,
        accuracy=0.4,
        confidence=0.3,
        quality_level=QualityLevel.UNACCEPTABLE,
    )


@pytest.fixture
def sample_scores(
    sample_metrics_excellent,
    sample_metrics_good,
    sample_metrics_acceptable,
    sample_metrics_poor,
    sample_metrics_unacceptable,
) -> list[RuleQualityScore]:
    """Create a list of sample quality scores."""
    return [
        RuleQualityScore(
            rule_name="email_format",
            rule_type=RuleType.PATTERN,
            column="email",
            metrics=sample_metrics_excellent,
            recommendation="Excellent quality. Safe to use.",
            should_use=True,
        ),
        RuleQualityScore(
            rule_name="age_range",
            rule_type=RuleType.RANGE,
            column="age",
            metrics=sample_metrics_good,
            recommendation="Good quality. Recommended for use.",
            should_use=True,
        ),
        RuleQualityScore(
            rule_name="name_not_null",
            rule_type=RuleType.COMPLETENESS,
            column="name",
            metrics=sample_metrics_acceptable,
            recommendation="Acceptable quality. Monitor for issues.",
            should_use=True,
        ),
        RuleQualityScore(
            rule_name="phone_pattern",
            rule_type=RuleType.PATTERN,
            column="phone",
            metrics=sample_metrics_poor,
            recommendation="Poor quality. Consider improvements.",
            should_use=False,
        ),
        RuleQualityScore(
            rule_name="status_enum",
            rule_type=RuleType.CUSTOM,
            column="status",
            metrics=sample_metrics_unacceptable,
            recommendation="Unacceptable quality. Do not use.",
            should_use=False,
        ),
    ]


# =============================================================================
# Test Utility Functions
# =============================================================================


class TestGetLevelValue:
    """Tests for get_level_value function."""

    def test_excellent_level(self):
        """Test excellent level value."""
        assert get_level_value("excellent") == 4

    def test_good_level(self):
        """Test good level value."""
        assert get_level_value("good") == 3

    def test_acceptable_level(self):
        """Test acceptable level value."""
        assert get_level_value("acceptable") == 2

    def test_poor_level(self):
        """Test poor level value."""
        assert get_level_value("poor") == 1

    def test_unacceptable_level(self):
        """Test unacceptable level value."""
        assert get_level_value("unacceptable") == 0

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert get_level_value("EXCELLENT") == 4
        assert get_level_value("Good") == 3

    def test_unknown_level(self):
        """Test unknown level returns -1."""
        assert get_level_value("unknown") == -1


# =============================================================================
# Test Level Filter
# =============================================================================


class TestLevelFilter:
    """Tests for LevelFilter."""

    def test_exact_level_match(self, sample_scores):
        """Test filtering by exact level."""
        filter_obj = LevelFilter(exact_level="good")
        result = filter_obj.apply(sample_scores)
        assert len(result) == 1
        assert result[0].rule_name == "age_range"

    def test_min_level(self, sample_scores):
        """Test filtering by minimum level."""
        filter_obj = LevelFilter(min_level="good")
        result = filter_obj.apply(sample_scores)
        assert len(result) == 2
        assert all(s.metrics.quality_level.value.lower() in ["excellent", "good"] for s in result)

    def test_max_level(self, sample_scores):
        """Test filtering by maximum level."""
        filter_obj = LevelFilter(max_level="acceptable")
        result = filter_obj.apply(sample_scores)
        assert len(result) == 3
        assert all(
            s.metrics.quality_level.value.lower() in ["acceptable", "poor", "unacceptable"]
            for s in result
        )

    def test_level_range(self, sample_scores):
        """Test filtering by level range."""
        filter_obj = LevelFilter(min_level="acceptable", max_level="good")
        result = filter_obj.apply(sample_scores)
        assert len(result) == 2
        assert all(
            s.metrics.quality_level.value.lower() in ["acceptable", "good"]
            for s in result
        )

    def test_no_filter(self, sample_scores):
        """Test no filtering applied."""
        filter_obj = LevelFilter()
        result = filter_obj.apply(sample_scores)
        assert len(result) == 5

    def test_filter_name(self):
        """Test filter name generation."""
        assert "level=good" in LevelFilter(exact_level="good").name
        assert "level>=acceptable" in LevelFilter(min_level="acceptable").name
        assert "level[poor..good]" in LevelFilter(min_level="poor", max_level="good").name


# =============================================================================
# Test Metric Filter
# =============================================================================


class TestMetricFilter:
    """Tests for MetricFilter."""

    def test_f1_score_gte(self, sample_scores):
        """Test filtering by F1 score >= threshold."""
        filter_obj = MetricFilter("f1_score", ">=", 0.85)
        result = filter_obj.apply(sample_scores)
        assert len(result) == 2
        assert all(s.metrics.f1_score >= 0.85 for s in result)

    def test_f1_score_gt(self, sample_scores):
        """Test filtering by F1 score > threshold."""
        filter_obj = MetricFilter("f1_score", ">", 0.9)
        result = filter_obj.apply(sample_scores)
        assert len(result) == 1
        assert result[0].rule_name == "email_format"

    def test_precision_lte(self, sample_scores):
        """Test filtering by precision <= threshold."""
        filter_obj = MetricFilter("precision", "<=", 0.6)
        result = filter_obj.apply(sample_scores)
        assert len(result) == 2

    def test_recall_lt(self, sample_scores):
        """Test filtering by recall < threshold."""
        filter_obj = MetricFilter("recall", "<", 0.5)
        result = filter_obj.apply(sample_scores)
        assert len(result) == 1
        assert result[0].rule_name == "status_enum"

    def test_f1_alias(self, sample_scores):
        """Test 'f1' alias for 'f1_score'."""
        filter_obj = MetricFilter("f1", ">=", 0.9)
        result = filter_obj.apply(sample_scores)
        assert len(result) == 1

    def test_invalid_metric(self):
        """Test invalid metric name raises error."""
        with pytest.raises(ValueError, match="Invalid metric"):
            MetricFilter("invalid_metric", ">=", 0.5)

    def test_invalid_operator(self):
        """Test invalid operator raises error."""
        with pytest.raises(ValueError, match="Invalid operator"):
            MetricFilter("f1_score", "??", 0.5)

    def test_filter_name(self):
        """Test filter name generation."""
        filter_obj = MetricFilter("f1_score", ">=", 0.85)
        assert "f1_score>=0.85" in filter_obj.name


# =============================================================================
# Test Confidence Filter
# =============================================================================


class TestConfidenceFilter:
    """Tests for ConfidenceFilter."""

    def test_min_confidence(self, sample_scores):
        """Test filtering by minimum confidence."""
        filter_obj = ConfidenceFilter(min_value=0.8)
        result = filter_obj.apply(sample_scores)
        assert len(result) == 2
        assert all(s.metrics.confidence >= 0.8 for s in result)

    def test_max_confidence(self, sample_scores):
        """Test filtering by maximum confidence."""
        filter_obj = ConfidenceFilter(max_value=0.6)
        result = filter_obj.apply(sample_scores)
        assert len(result) == 2
        assert all(s.metrics.confidence <= 0.6 for s in result)

    def test_confidence_range(self, sample_scores):
        """Test filtering by confidence range."""
        filter_obj = ConfidenceFilter(min_value=0.5, max_value=0.8)
        result = filter_obj.apply(sample_scores)
        assert len(result) == 2

    def test_no_filter(self, sample_scores):
        """Test no filtering applied."""
        filter_obj = ConfidenceFilter()
        result = filter_obj.apply(sample_scores)
        assert len(result) == 5


# =============================================================================
# Test Column Filter
# =============================================================================


class TestColumnFilter:
    """Tests for ColumnFilter."""

    def test_include_columns(self, sample_scores):
        """Test filtering by included columns."""
        filter_obj = ColumnFilter(include=["email", "age"])
        result = filter_obj.apply(sample_scores)
        assert len(result) == 2
        assert all(s.column in ["email", "age"] for s in result)

    def test_exclude_columns(self, sample_scores):
        """Test filtering by excluded columns."""
        filter_obj = ColumnFilter(exclude=["email", "age"])
        result = filter_obj.apply(sample_scores)
        assert len(result) == 3
        assert all(s.column not in ["email", "age"] for s in result)

    def test_pattern_match(self, sample_scores):
        """Test filtering by column pattern."""
        filter_obj = ColumnFilter(pattern=r"^(email|age)$")
        result = filter_obj.apply(sample_scores)
        assert len(result) == 2


# =============================================================================
# Test Rule Type Filter
# =============================================================================


class TestRuleTypeFilter:
    """Tests for RuleTypeFilter."""

    def test_include_rule_types(self, sample_scores):
        """Test filtering by included rule types."""
        filter_obj = RuleTypeFilter(include=["pattern"])
        result = filter_obj.apply(sample_scores)
        assert len(result) == 2
        assert all(s.rule_type == RuleType.PATTERN for s in result)

    def test_exclude_rule_types(self, sample_scores):
        """Test filtering by excluded rule types."""
        filter_obj = RuleTypeFilter(exclude=["pattern", "custom"])
        result = filter_obj.apply(sample_scores)
        assert len(result) == 2


# =============================================================================
# Test Recommendation Filter
# =============================================================================


class TestRecommendationFilter:
    """Tests for RecommendationFilter."""

    def test_should_use_true(self, sample_scores):
        """Test filtering for should_use=True."""
        filter_obj = RecommendationFilter(should_use=True)
        result = filter_obj.apply(sample_scores)
        assert len(result) == 3
        assert all(s.should_use for s in result)

    def test_should_use_false(self, sample_scores):
        """Test filtering for should_use=False."""
        filter_obj = RecommendationFilter(should_use=False)
        result = filter_obj.apply(sample_scores)
        assert len(result) == 2
        assert all(not s.should_use for s in result)

    def test_contains_text(self, sample_scores):
        """Test filtering by recommendation text."""
        filter_obj = RecommendationFilter(contains="excellent")
        result = filter_obj.apply(sample_scores)
        assert len(result) == 1
        assert result[0].rule_name == "email_format"


# =============================================================================
# Test Custom Filter
# =============================================================================


class TestCustomFilter:
    """Tests for CustomFilter."""

    def test_custom_predicate(self, sample_scores):
        """Test filtering with custom predicate."""
        filter_obj = CustomFilter(
            predicate=lambda s: s.rule_name.startswith("email"),
            name="email_rules",
        )
        result = filter_obj.apply(sample_scores)
        assert len(result) == 1
        assert result[0].rule_name == "email_format"


# =============================================================================
# Test Composite Filters
# =============================================================================


class TestCompositeFilter:
    """Tests for CompositeFilter and operators."""

    def test_and_combination(self, sample_scores):
        """Test AND combination of filters."""
        level_filter = LevelFilter(min_level="good")
        confidence_filter = ConfidenceFilter(min_value=0.85)
        combined = level_filter.and_(confidence_filter)

        result = combined.apply(sample_scores)
        assert len(result) == 2
        assert all(s.metrics.quality_level.value.lower() in ["excellent", "good"] for s in result)
        assert all(s.metrics.confidence >= 0.85 for s in result)

    def test_or_combination(self, sample_scores):
        """Test OR combination of filters."""
        excellent_filter = LevelFilter(exact_level="excellent")
        poor_filter = LevelFilter(exact_level="poor")
        combined = excellent_filter.or_(poor_filter)

        result = combined.apply(sample_scores)
        assert len(result) == 2

    def test_not_filter(self, sample_scores):
        """Test NOT filter."""
        level_filter = LevelFilter(exact_level="excellent")
        negated = level_filter.not_()

        result = negated.apply(sample_scores)
        assert len(result) == 4
        assert all(s.metrics.quality_level != QualityLevel.EXCELLENT for s in result)

    def test_operator_and(self, sample_scores):
        """Test & operator."""
        f1 = LevelFilter(min_level="good")
        f2 = ConfidenceFilter(min_value=0.85)
        combined = f1 & f2

        result = combined.apply(sample_scores)
        assert len(result) == 2

    def test_operator_or(self, sample_scores):
        """Test | operator."""
        f1 = LevelFilter(exact_level="excellent")
        f2 = LevelFilter(exact_level="poor")
        combined = f1 | f2

        result = combined.apply(sample_scores)
        assert len(result) == 2

    def test_operator_not(self, sample_scores):
        """Test ~ operator."""
        f = LevelFilter(exact_level="excellent")
        negated = ~f

        result = negated.apply(sample_scores)
        assert len(result) == 4

    def test_all_of(self, sample_scores):
        """Test AllOfFilter."""
        filters = [
            LevelFilter(min_level="acceptable"),
            RecommendationFilter(should_use=True),
        ]
        combined = AllOfFilter(filters)

        result = combined.apply(sample_scores)
        assert len(result) == 3

    def test_any_of(self, sample_scores):
        """Test AnyOfFilter."""
        filters = [
            LevelFilter(exact_level="excellent"),
            LevelFilter(exact_level="unacceptable"),
        ]
        combined = AnyOfFilter(filters)

        result = combined.apply(sample_scores)
        assert len(result) == 2


# =============================================================================
# Test QualityFilter Factory
# =============================================================================


class TestQualityFilterFactory:
    """Tests for QualityFilter factory class."""

    def test_by_level(self, sample_scores):
        """Test by_level factory method."""
        filter_obj = QualityFilter.by_level("good")
        result = filter_obj.apply(sample_scores)
        assert len(result) == 1

    def test_by_level_with_min(self, sample_scores):
        """Test by_level with min_level."""
        filter_obj = QualityFilter.by_level(min_level="acceptable")
        result = filter_obj.apply(sample_scores)
        assert len(result) == 3

    def test_by_metric(self, sample_scores):
        """Test by_metric factory method."""
        filter_obj = QualityFilter.by_metric("f1_score", ">=", 0.7)
        result = filter_obj.apply(sample_scores)
        assert len(result) == 3

    def test_by_confidence(self, sample_scores):
        """Test by_confidence factory method."""
        filter_obj = QualityFilter.by_confidence(min_value=0.7)
        result = filter_obj.apply(sample_scores)
        assert len(result) == 3

    def test_by_column(self, sample_scores):
        """Test by_column factory method."""
        filter_obj = QualityFilter.by_column(include=["email", "age"])
        result = filter_obj.apply(sample_scores)
        assert len(result) == 2

    def test_by_rule_type(self, sample_scores):
        """Test by_rule_type factory method."""
        filter_obj = QualityFilter.by_rule_type(include=["pattern"])
        result = filter_obj.apply(sample_scores)
        assert len(result) == 2

    def test_by_recommendation(self, sample_scores):
        """Test by_recommendation factory method."""
        filter_obj = QualityFilter.by_recommendation(should_use=True)
        result = filter_obj.apply(sample_scores)
        assert len(result) == 3

    def test_custom(self, sample_scores):
        """Test custom factory method."""
        filter_obj = QualityFilter.custom(lambda s: len(s.rule_name) > 10)
        result = filter_obj.apply(sample_scores)
        # email_format (12), name_not_null (13), phone_pattern (13), status_enum (11) = 4 rules
        assert len(result) == 4

    def test_all_of(self, sample_scores):
        """Test all_of factory method."""
        filter_obj = QualityFilter.all_of(
            QualityFilter.by_level(min_level="good"),
            QualityFilter.by_confidence(min_value=0.8),
        )
        result = filter_obj.apply(sample_scores)
        assert len(result) == 2

    def test_any_of(self, sample_scores):
        """Test any_of factory method."""
        filter_obj = QualityFilter.any_of(
            QualityFilter.by_level("excellent"),
            QualityFilter.by_level("poor"),
        )
        result = filter_obj.apply(sample_scores)
        assert len(result) == 2

    def test_none(self, sample_scores):
        """Test none factory method."""
        filter_obj = QualityFilter.none()
        result = filter_obj.apply(sample_scores)
        assert len(result) == 0

    def test_all(self, sample_scores):
        """Test all factory method."""
        filter_obj = QualityFilter.all()
        result = filter_obj.apply(sample_scores)
        assert len(result) == 5

    def test_from_config(self, sample_scores):
        """Test from_config factory method."""
        config = QualityFilterConfig(
            min_level="acceptable",
            min_f1=0.5,
            should_use_only=True,
        )
        filter_obj = QualityFilter.from_config(config)
        result = filter_obj.apply(sample_scores)
        assert len(result) == 3
        assert all(s.should_use for s in result)

    def test_from_config_empty(self, sample_scores):
        """Test from_config with empty config."""
        config = QualityFilterConfig()
        filter_obj = QualityFilter.from_config(config)
        result = filter_obj.apply(sample_scores)
        assert len(result) == 5


# =============================================================================
# Test Complex Filter Chains
# =============================================================================


class TestComplexFilterChains:
    """Tests for complex filter chains."""

    def test_triple_and_chain(self, sample_scores):
        """Test chaining three filters with AND."""
        filter_obj = (
            QualityFilter.by_level(min_level="acceptable")
            .and_(QualityFilter.by_confidence(min_value=0.6))
            .and_(QualityFilter.by_recommendation(should_use=True))
        )
        result = filter_obj.apply(sample_scores)
        assert len(result) == 3

    def test_mixed_and_or(self, sample_scores):
        """Test mixing AND and OR."""
        # (excellent OR good) AND should_use
        filter_obj = (
            QualityFilter.by_level("excellent")
            .or_(QualityFilter.by_level("good"))
        ).and_(QualityFilter.by_recommendation(should_use=True))

        result = filter_obj.apply(sample_scores)
        assert len(result) == 2

    def test_not_in_chain(self, sample_scores):
        """Test NOT in a chain."""
        # acceptable OR better, but NOT excellent
        filter_obj = (
            QualityFilter.by_level(min_level="acceptable")
            .and_(QualityFilter.by_level("excellent").not_())
        )
        result = filter_obj.apply(sample_scores)
        assert len(result) == 2
        assert all(s.metrics.quality_level != QualityLevel.EXCELLENT for s in result)
