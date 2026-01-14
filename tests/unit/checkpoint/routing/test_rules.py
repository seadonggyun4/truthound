"""Tests for built-in routing rules."""

from __future__ import annotations

from datetime import datetime, time

import pytest

from truthound.checkpoint.routing.base import RouteContext
from truthound.checkpoint.routing.rules import (
    AlwaysRule,
    DataAssetRule,
    ErrorRule,
    IssueCountRule,
    MetadataRule,
    NeverRule,
    PassRateRule,
    SeverityRule,
    StatusRule,
    TagRule,
    TimeWindowRule,
)


def create_context(**kwargs) -> RouteContext:
    """Create test context with defaults."""
    defaults = {
        "checkpoint_name": "test",
        "run_id": "run_123",
        "status": "failure",
        "data_asset": "sales_data_v2.csv",
        "run_time": datetime(2024, 1, 15, 14, 30),  # Monday, 2:30 PM
        "total_issues": 10,
        "critical_issues": 2,
        "high_issues": 3,
        "medium_issues": 3,
        "low_issues": 2,
        "info_issues": 0,
        "pass_rate": 85.0,
        "tags": {"env": "prod", "team": "data"},
        "metadata": {
            "region": "us-east-1",
            "priority": 5,
            "owners": ["alice", "bob"],
            "nested": {"key": "value"},
        },
        "error": None,
    }
    defaults.update(kwargs)
    return RouteContext(**defaults)


class TestAlwaysNeverRules:
    """Tests for AlwaysRule and NeverRule."""

    def test_always_rule(self):
        """Test AlwaysRule always matches."""
        rule = AlwaysRule()
        ctx = create_context()

        assert rule.evaluate(ctx) is True
        assert rule.description == "Always matches"

    def test_never_rule(self):
        """Test NeverRule never matches."""
        rule = NeverRule()
        ctx = create_context()

        assert rule.evaluate(ctx) is False
        assert rule.description == "Never matches"


class TestSeverityRule:
    """Tests for SeverityRule."""

    def test_min_severity_critical(self):
        """Test matching critical issues."""
        rule = SeverityRule(min_severity="critical")
        ctx = create_context(critical_issues=1)

        assert rule.evaluate(ctx) is True

    def test_min_severity_high(self):
        """Test matching high issues only (without max_severity)."""
        rule = SeverityRule(min_severity="high")

        # Has high issues
        ctx1 = create_context(critical_issues=0, high_issues=2)
        assert rule.evaluate(ctx1) is True

        # No high issues (only critical - not counted when min=high)
        ctx2 = create_context(critical_issues=1, high_issues=0)
        assert rule.evaluate(ctx2) is False

        # No high issues
        ctx3 = create_context(critical_issues=0, high_issues=0)
        assert rule.evaluate(ctx3) is False

    def test_severity_at_or_above(self):
        """Test matching severity at or above a threshold (high or more severe)."""
        # To match high OR critical, specify the range
        rule = SeverityRule(min_severity="critical", max_severity="high")

        # Has critical issues
        ctx1 = create_context(critical_issues=1, high_issues=0)
        assert rule.evaluate(ctx1) is True

        # Has high issues
        ctx2 = create_context(critical_issues=0, high_issues=2)
        assert rule.evaluate(ctx2) is True

        # No high or critical
        ctx3 = create_context(critical_issues=0, high_issues=0)
        assert rule.evaluate(ctx3) is False

    def test_severity_range(self):
        """Test matching within severity range."""
        rule = SeverityRule(min_severity="medium", max_severity="medium")
        ctx = create_context(
            critical_issues=10, high_issues=10, medium_issues=1, low_issues=10
        )

        # Should only count medium issues
        assert rule.evaluate(ctx) is True

    def test_min_count(self):
        """Test minimum count requirement."""
        rule = SeverityRule(min_severity="critical", min_count=3)

        ctx1 = create_context(critical_issues=2)
        assert rule.evaluate(ctx1) is False

        ctx2 = create_context(critical_issues=3)
        assert rule.evaluate(ctx2) is True

    def test_exact_count(self):
        """Test exact count matching."""
        # min_severity="high" only counts high issues
        rule = SeverityRule(min_severity="high", exact_count=3)

        ctx1 = create_context(critical_issues=2, high_issues=3)
        assert rule.evaluate(ctx1) is True  # Only high=3 counted

        ctx2 = create_context(critical_issues=3, high_issues=4)
        assert rule.evaluate(ctx2) is False  # high=4, not exactly 3

    def test_exact_count_with_range(self):
        """Test exact count with severity range."""
        # Count both critical and high
        rule = SeverityRule(min_severity="critical", max_severity="high", exact_count=5)

        ctx1 = create_context(critical_issues=2, high_issues=3)
        assert rule.evaluate(ctx1) is True  # 2 + 3 = 5

        ctx2 = create_context(critical_issues=3, high_issues=3)
        assert rule.evaluate(ctx2) is False  # 3 + 3 = 6

    def test_invalid_severity_raises(self):
        """Test invalid severity raises error."""
        with pytest.raises(ValueError, match="Invalid min_severity"):
            SeverityRule(min_severity="invalid")

    def test_description(self):
        """Test rule description."""
        rule1 = SeverityRule(min_severity="critical")
        assert "critical" in rule1.description

        rule2 = SeverityRule(min_severity="high", max_severity="medium")
        assert "high" in rule2.description and "medium" in rule2.description


class TestIssueCountRule:
    """Tests for IssueCountRule."""

    def test_min_issues(self):
        """Test minimum issues."""
        rule = IssueCountRule(min_issues=5)

        assert rule.evaluate(create_context(total_issues=10)) is True
        assert rule.evaluate(create_context(total_issues=4)) is False

    def test_max_issues(self):
        """Test maximum issues."""
        rule = IssueCountRule(min_issues=0, max_issues=5)

        assert rule.evaluate(create_context(total_issues=3)) is True
        assert rule.evaluate(create_context(total_issues=10)) is False

    def test_issue_range(self):
        """Test issue count range."""
        rule = IssueCountRule(min_issues=5, max_issues=10)

        assert rule.evaluate(create_context(total_issues=7)) is True
        assert rule.evaluate(create_context(total_issues=3)) is False
        assert rule.evaluate(create_context(total_issues=15)) is False

    def test_count_type(self):
        """Test different count types."""
        rule = IssueCountRule(min_issues=2, count_type="critical")

        ctx1 = create_context(total_issues=100, critical_issues=1)
        assert rule.evaluate(ctx1) is False

        ctx2 = create_context(total_issues=5, critical_issues=3)
        assert rule.evaluate(ctx2) is True


class TestStatusRule:
    """Tests for StatusRule."""

    def test_single_status(self):
        """Test matching single status."""
        rule = StatusRule(statuses=["failure"])

        assert rule.evaluate(create_context(status="failure")) is True
        assert rule.evaluate(create_context(status="success")) is False

    def test_multiple_statuses(self):
        """Test matching multiple statuses."""
        rule = StatusRule(statuses=["failure", "error"])

        assert rule.evaluate(create_context(status="failure")) is True
        assert rule.evaluate(create_context(status="error")) is True
        assert rule.evaluate(create_context(status="success")) is False

    def test_negate(self):
        """Test negated status matching."""
        rule = StatusRule(statuses=["success"], negate=True)

        assert rule.evaluate(create_context(status="failure")) is True
        assert rule.evaluate(create_context(status="success")) is False

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        rule = StatusRule(statuses=["FAILURE"])

        assert rule.evaluate(create_context(status="failure")) is True


class TestTagRule:
    """Tests for TagRule."""

    def test_tag_exists(self):
        """Test tag existence check."""
        rule = TagRule(tags={"env": None})

        ctx1 = create_context(tags={"env": "prod"})
        assert rule.evaluate(ctx1) is True

        ctx2 = create_context(tags={})
        assert rule.evaluate(ctx2) is False

    def test_tag_value(self):
        """Test tag value matching."""
        rule = TagRule(tags={"env": "prod"})

        ctx1 = create_context(tags={"env": "prod"})
        assert rule.evaluate(ctx1) is True

        ctx2 = create_context(tags={"env": "staging"})
        assert rule.evaluate(ctx2) is False

    def test_multiple_tags_all(self):
        """Test matching all tags."""
        rule = TagRule(tags={"env": "prod", "team": "data"}, match_all=True)

        ctx1 = create_context(tags={"env": "prod", "team": "data"})
        assert rule.evaluate(ctx1) is True

        ctx2 = create_context(tags={"env": "prod"})
        assert rule.evaluate(ctx2) is False

    def test_multiple_tags_any(self):
        """Test matching any tag."""
        rule = TagRule(tags={"env": "prod", "team": "data"}, match_all=False)

        ctx1 = create_context(tags={"env": "prod"})
        assert rule.evaluate(ctx1) is True

        ctx2 = create_context(tags={"env": "staging"})
        assert rule.evaluate(ctx2) is False

    def test_negate(self):
        """Test negated tag matching."""
        rule = TagRule(tags={"env": "prod"}, negate=True)

        ctx1 = create_context(tags={"env": "staging"})
        assert rule.evaluate(ctx1) is True

        ctx2 = create_context(tags={"env": "prod"})
        assert rule.evaluate(ctx2) is False


class TestDataAssetRule:
    """Tests for DataAssetRule."""

    def test_glob_pattern(self):
        """Test glob-style pattern matching."""
        rule = DataAssetRule(pattern="sales_*")

        ctx1 = create_context(data_asset="sales_data.csv")
        assert rule.evaluate(ctx1) is True

        ctx2 = create_context(data_asset="users_data.csv")
        assert rule.evaluate(ctx2) is False

    def test_glob_wildcard(self):
        """Test wildcard matching."""
        rule = DataAssetRule(pattern="*_v2*")

        ctx1 = create_context(data_asset="sales_v2.csv")
        assert rule.evaluate(ctx1) is True

        ctx2 = create_context(data_asset="sales_v1.csv")
        assert rule.evaluate(ctx2) is False

    def test_regex_pattern(self):
        """Test regex pattern matching."""
        rule = DataAssetRule(pattern=r"^prod_.*_v\d+$", is_regex=True)

        ctx1 = create_context(data_asset="prod_sales_v2")
        assert rule.evaluate(ctx1) is True

        ctx2 = create_context(data_asset="dev_sales_v2")
        assert rule.evaluate(ctx2) is False

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        rule = DataAssetRule(pattern="SALES*", case_sensitive=False)

        ctx = create_context(data_asset="sales_data.csv")
        assert rule.evaluate(ctx) is True


class TestMetadataRule:
    """Tests for MetadataRule."""

    def test_simple_key(self):
        """Test simple key matching."""
        rule = MetadataRule(key_path="region", expected_value="us-east-1")
        ctx = create_context()

        assert rule.evaluate(ctx) is True

    def test_nested_key(self):
        """Test nested key path."""
        rule = MetadataRule(key_path="nested.key", expected_value="value")
        ctx = create_context()

        assert rule.evaluate(ctx) is True

    def test_exists_comparator(self):
        """Test exists comparator."""
        rule = MetadataRule(key_path="region", comparator="exists")
        ctx = create_context()

        assert rule.evaluate(ctx) is True

    def test_numeric_comparators(self):
        """Test numeric comparators."""
        ctx = create_context()

        assert MetadataRule(key_path="priority", expected_value=5, comparator="eq").evaluate(ctx) is True
        assert MetadataRule(key_path="priority", expected_value=4, comparator="gt").evaluate(ctx) is True
        assert MetadataRule(key_path="priority", expected_value=5, comparator="gte").evaluate(ctx) is True
        assert MetadataRule(key_path="priority", expected_value=6, comparator="lt").evaluate(ctx) is True

    def test_contains_comparator(self):
        """Test contains comparator."""
        ctx = create_context()

        rule = MetadataRule(key_path="owners", expected_value="alice", comparator="contains")
        assert rule.evaluate(ctx) is True

    def test_regex_comparator(self):
        """Test regex comparator."""
        ctx = create_context()

        rule = MetadataRule(key_path="region", expected_value=r"us-.*-1", comparator="regex")
        assert rule.evaluate(ctx) is True

    def test_missing_key(self):
        """Test missing key returns False."""
        rule = MetadataRule(key_path="nonexistent", expected_value="value")
        ctx = create_context()

        assert rule.evaluate(ctx) is False


class TestTimeWindowRule:
    """Tests for TimeWindowRule."""

    def test_time_in_window(self):
        """Test time within window."""
        rule = TimeWindowRule(start_time="09:00", end_time="17:00")

        # 2:30 PM is within 9 AM - 5 PM
        ctx = create_context(run_time=datetime(2024, 1, 15, 14, 30))
        assert rule.evaluate(ctx) is True

    def test_time_outside_window(self):
        """Test time outside window."""
        rule = TimeWindowRule(start_time="09:00", end_time="17:00")

        # 8 PM is outside 9 AM - 5 PM
        ctx = create_context(run_time=datetime(2024, 1, 15, 20, 0))
        assert rule.evaluate(ctx) is False

    def test_overnight_window(self):
        """Test overnight window (wraps midnight)."""
        rule = TimeWindowRule(start_time="22:00", end_time="06:00")

        # 11 PM is within 10 PM - 6 AM
        ctx1 = create_context(run_time=datetime(2024, 1, 15, 23, 0))
        assert rule.evaluate(ctx1) is True

        # 3 AM is within 10 PM - 6 AM
        ctx2 = create_context(run_time=datetime(2024, 1, 15, 3, 0))
        assert rule.evaluate(ctx2) is True

        # 12 PM is outside 10 PM - 6 AM
        ctx3 = create_context(run_time=datetime(2024, 1, 15, 12, 0))
        assert rule.evaluate(ctx3) is False

    def test_days_of_week(self):
        """Test day of week filtering."""
        # Mon-Fri only
        rule = TimeWindowRule(
            start_time="00:00",
            end_time="23:59",
            days_of_week=[0, 1, 2, 3, 4],
        )

        # Monday
        ctx1 = create_context(run_time=datetime(2024, 1, 15, 12, 0))
        assert rule.evaluate(ctx1) is True

        # Saturday
        ctx2 = create_context(run_time=datetime(2024, 1, 20, 12, 0))
        assert rule.evaluate(ctx2) is False


class TestPassRateRule:
    """Tests for PassRateRule."""

    def test_below_threshold(self):
        """Test pass rate below threshold."""
        rule = PassRateRule(max_rate=90.0)

        ctx1 = create_context(pass_rate=85.0)
        assert rule.evaluate(ctx1) is True

        ctx2 = create_context(pass_rate=95.0)
        assert rule.evaluate(ctx2) is False

    def test_above_threshold(self):
        """Test pass rate above threshold."""
        rule = PassRateRule(min_rate=80.0)

        ctx1 = create_context(pass_rate=85.0)
        assert rule.evaluate(ctx1) is True

        ctx2 = create_context(pass_rate=75.0)
        assert rule.evaluate(ctx2) is False

    def test_within_range(self):
        """Test pass rate within range."""
        rule = PassRateRule(min_rate=80.0, max_rate=90.0)

        ctx1 = create_context(pass_rate=85.0)
        assert rule.evaluate(ctx1) is True

        ctx2 = create_context(pass_rate=75.0)
        assert rule.evaluate(ctx2) is False

        ctx3 = create_context(pass_rate=95.0)
        assert rule.evaluate(ctx3) is False


class TestErrorRule:
    """Tests for ErrorRule."""

    def test_any_error(self):
        """Test matching any error."""
        rule = ErrorRule()

        ctx1 = create_context(error="Something went wrong")
        assert rule.evaluate(ctx1) is True

        ctx2 = create_context(error=None)
        assert rule.evaluate(ctx2) is False

    def test_error_pattern(self):
        """Test error pattern matching."""
        rule = ErrorRule(pattern=r"timeout|timed out")

        ctx1 = create_context(error="Connection timeout")
        assert rule.evaluate(ctx1) is True

        ctx2 = create_context(error="Invalid data format")
        assert rule.evaluate(ctx2) is False

    def test_no_error(self):
        """Test matching when no error."""
        rule = ErrorRule(negate=True)

        ctx1 = create_context(error=None)
        assert rule.evaluate(ctx1) is True

        ctx2 = create_context(error="Error!")
        assert rule.evaluate(ctx2) is False
