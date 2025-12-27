"""Tests for the Reporter SDK mixins module."""

import json
import pytest
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from truthound.reporters.sdk.mixins import (
    FormattingMixin,
    AggregationMixin,
    FilteringMixin,
    SerializationMixin,
    TemplatingMixin,
    StreamingMixin,
)


# Mock ValidatorResult for testing
@dataclass
class MockValidatorResult:
    """Mock validator result for testing mixins."""

    validator_name: str = "test_validator"
    column: Optional[str] = None
    success: bool = True
    severity: Optional[str] = "error"
    message: str = ""


@dataclass
class MockValidationResult:
    """Mock validation result for testing mixins."""

    results: list = None

    def __post_init__(self):
        if self.results is None:
            self.results = []


class FormattingTest(FormattingMixin):
    """Test class using FormattingMixin."""

    pass


class AggregationTest(AggregationMixin):
    """Test class using AggregationMixin."""

    pass


class FilteringTest(FilteringMixin):
    """Test class using FilteringMixin."""

    pass


class SerializationTest(SerializationMixin):
    """Test class using SerializationMixin."""

    pass


class TemplatingTest(TemplatingMixin):
    """Test class using TemplatingMixin."""

    pass


class StreamingTest(StreamingMixin):
    """Test class using StreamingMixin."""

    pass


def _check_jinja2_available():
    """Check if jinja2 is available."""
    try:
        import jinja2

        return True
    except ImportError:
        return False


class TestFormattingMixin:
    """Tests for FormattingMixin."""

    def setup_method(self):
        self.formatter = FormattingTest()

    def test_format_as_table_basic(self):
        """Test basic table formatting."""
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        result = self.formatter.format_as_table(data)

        assert "name" in result
        assert "age" in result
        assert "Alice" in result
        assert "Bob" in result

    def test_format_as_table_empty(self):
        """Test table formatting with empty data."""
        result = self.formatter.format_as_table([])
        assert result == ""

    def test_format_as_table_with_columns(self):
        """Test table formatting with specific columns."""
        data = [
            {"name": "Alice", "age": 30, "email": "alice@test.com"},
        ]
        result = self.formatter.format_as_table(data, columns=["name", "email"])

        assert "name" in result
        assert "email" in result
        assert "Alice" in result
        # age should not appear since we only specified name and email
        assert "30" not in result

    def test_format_as_table_markdown(self):
        """Test markdown table formatting."""
        data = [{"name": "Alice", "age": 30}]
        result = self.formatter.format_as_table(data, style="markdown")

        assert "|" in result
        assert "---" in result

    def test_format_as_table_grid(self):
        """Test grid table formatting."""
        data = [{"name": "Alice", "age": 30}]
        result = self.formatter.format_as_table(data, style="grid")

        assert "╔" in result or "║" in result

    def test_format_as_table_simple(self):
        """Test simple table formatting."""
        data = [{"name": "Alice", "age": 30}]
        result = self.formatter.format_as_table(data, style="simple")

        assert "name" in result
        assert "Alice" in result

    def test_truncate(self):
        """Test string truncation."""
        text = "This is a long string that needs truncation"
        result = self.formatter.truncate(text, max_length=20)

        assert len(result) <= 20
        assert result.endswith("...")

    def test_truncate_short_text(self):
        """Test truncation of short text."""
        text = "Short"
        result = self.formatter.truncate(text, max_length=20)
        assert result == "Short"

    def test_truncate_custom_suffix(self):
        """Test truncation with custom suffix."""
        text = "Long text here"
        result = self.formatter.truncate(text, max_length=10, suffix=">>")
        assert result.endswith(">>")

    def test_indent(self):
        """Test text indentation."""
        text = "Line 1\nLine 2\nLine 3"
        result = self.formatter.indent(text, prefix="    ")

        for line in result.split("\n"):
            assert line.startswith("    ")

    def test_indent_custom_prefix(self):
        """Test indentation with custom prefix."""
        text = "Line 1\nLine 2"
        result = self.formatter.indent(text, prefix=">>> ")

        assert result.startswith(">>> ")
        assert "\n>>> " in result

    def test_wrap(self):
        """Test text wrapping."""
        text = "This is a long line that should be wrapped at a certain width"
        result = self.formatter.wrap(text, width=20)

        for line in result.split("\n"):
            assert len(line) <= 20

    def test_format_number(self):
        """Test number formatting."""
        assert self.formatter.format_number(1234567) == "1,234,567"
        assert self.formatter.format_number(1234567.89, precision=2) == "1,234,567.89"

    def test_format_number_custom_separator(self):
        """Test number formatting with custom separator."""
        result = self.formatter.format_number(1234567, thousands_sep=" ")
        assert result == "1 234 567"

    def test_format_percentage(self):
        """Test percentage formatting."""
        assert self.formatter.format_percentage(0.5) == "50.0%"
        assert self.formatter.format_percentage(0.333, precision=2) == "33.30%"
        assert self.formatter.format_percentage(1.0) == "100.0%"

    def test_format_percentage_with_sign(self):
        """Test percentage with sign."""
        result = self.formatter.format_percentage(0.5, include_sign=True)
        assert result == "+50.0%"

    def test_format_bytes(self):
        """Test bytes formatting."""
        assert "B" in self.formatter.format_bytes(100)
        assert "KB" in self.formatter.format_bytes(1024)
        assert "MB" in self.formatter.format_bytes(1024 * 1024)
        assert "GB" in self.formatter.format_bytes(1024 * 1024 * 1024)
        assert "TB" in self.formatter.format_bytes(1024 * 1024 * 1024 * 1024)

    def test_format_datetime(self):
        """Test datetime formatting."""
        dt = datetime(2024, 1, 15, 10, 30, 45)
        result = self.formatter.format_datetime(dt)
        assert "2024" in result

    def test_format_datetime_custom_format(self):
        """Test datetime with custom format."""
        dt = datetime(2024, 1, 15)
        result = self.formatter.format_datetime(dt, format="%Y/%m/%d")
        assert result == "2024/01/15"

    def test_format_duration(self):
        """Test duration formatting."""
        # Milliseconds
        result = self.formatter.format_duration(0.5)
        assert "ms" in result

        # Seconds only
        result = self.formatter.format_duration(45.5)
        assert "s" in result

        # Minutes and seconds
        result = self.formatter.format_duration(125.0)
        assert "m" in result

        # Hours
        result = self.formatter.format_duration(3700.0)
        assert "h" in result

    def test_format_relative_time(self):
        """Test relative time formatting."""
        now = datetime.now()

        # Just now
        result = self.formatter.format_relative_time(now)
        assert "now" in result.lower()

        # Minutes ago
        past = now - timedelta(minutes=5)
        result = self.formatter.format_relative_time(past)
        assert "minute" in result.lower()

        # Hours ago
        past = now - timedelta(hours=3)
        result = self.formatter.format_relative_time(past)
        assert "hour" in result.lower()


class TestAggregationMixin:
    """Tests for AggregationMixin."""

    def setup_method(self):
        self.aggregator = AggregationTest()
        self.sample_results = [
            MockValidatorResult(validator_name="not_null", column="id", severity="error", success=True),
            MockValidatorResult(validator_name="unique", column="id", severity="error", success=True),
            MockValidatorResult(validator_name="format", column="email", severity="warning", success=False),
            MockValidatorResult(validator_name="not_null", column="email", severity="error", success=True),
            MockValidatorResult(validator_name="range", column="age", severity="info", success=False),
        ]

    def test_group_by_column(self):
        """Test grouping by column."""
        grouped = self.aggregator.group_by_column(self.sample_results)

        assert "id" in grouped
        assert "email" in grouped
        assert "age" in grouped
        assert len(grouped["id"]) == 2
        assert len(grouped["email"]) == 2

    def test_group_by_severity(self):
        """Test grouping by severity."""
        grouped = self.aggregator.group_by_severity(self.sample_results)

        assert "error" in grouped
        assert "warning" in grouped
        assert "info" in grouped

    def test_group_by_validator(self):
        """Test grouping by validator."""
        grouped = self.aggregator.group_by_validator(self.sample_results)

        assert "not_null" in grouped
        assert "unique" in grouped
        assert "format" in grouped
        assert len(grouped["not_null"]) == 2

    def test_group_by_custom_key(self):
        """Test grouping by custom key function."""
        grouped = self.aggregator.group_by(
            self.sample_results,
            key=lambda r: "passed" if r.success else "failed",
        )

        assert "passed" in grouped
        assert "failed" in grouped

    def test_count_by_severity(self):
        """Test counting by severity."""
        counts = self.aggregator.count_by_severity(self.sample_results)

        assert counts["error"] == 3
        assert counts["warning"] == 1
        assert counts["info"] == 1

    def test_count_by_column(self):
        """Test counting by column."""
        counts = self.aggregator.count_by_column(self.sample_results)

        assert counts["id"] == 2
        assert counts["email"] == 2
        assert counts["age"] == 1

    def test_get_summary_stats(self):
        """Test getting summary statistics."""
        result = MockValidationResult(results=self.sample_results)
        stats = self.aggregator.get_summary_stats(result)

        assert stats["total_validators"] == 5
        assert stats["passed"] == 3
        assert stats["failed"] == 2
        assert "pass_rate" in stats


class TestFilteringMixin:
    """Tests for FilteringMixin."""

    def setup_method(self):
        self.filterer = FilteringTest()
        # Use severities from SEVERITY_ORDER: critical, high, medium, low, info
        self.sample_results = [
            MockValidatorResult(validator_name="not_null", column="id", severity="critical", success=True),
            MockValidatorResult(validator_name="format", column="email", severity="medium", success=False),
            MockValidatorResult(validator_name="range", column="age", severity="info", success=False),
            MockValidatorResult(validator_name="length", column="name", severity="high", success=True),
        ]

    def test_filter_by_severity_include(self):
        """Test filtering by severity with include list."""
        high_sev = self.filterer.filter_by_severity(
            self.sample_results,
            include_severities=["critical", "high"],
        )
        assert all(r.severity in ["critical", "high"] for r in high_sev)
        assert len(high_sev) == 2

    def test_filter_by_min_severity(self):
        """Test filtering by minimum severity.

        min_severity="medium" means include medium and more severe (high, critical).
        The severity level order is: critical=0, high=1, medium=2, low=3, info=4
        So min_severity filters out lower severity (higher numbers).
        """
        results = self.filterer.filter_by_severity(
            self.sample_results,
            min_severity="medium",
        )
        # Should filter out "info" and "low" severities
        severities = [r.severity for r in results]
        assert "info" not in severities
        assert len(results) == 3  # 1 critical + 1 high + 1 medium

    def test_filter_by_column_include(self):
        """Test filtering by column with include list."""
        id_results = self.filterer.filter_by_column(
            self.sample_results,
            include_columns=["id"],
        )
        assert all(r.column == "id" for r in id_results)

    def test_filter_by_columns_multiple(self):
        """Test filtering by multiple columns."""
        results = self.filterer.filter_by_column(
            self.sample_results,
            include_columns=["id", "email"],
        )
        assert all(r.column in ["id", "email"] for r in results)

    def test_filter_by_validator_include(self):
        """Test filtering by validator with include list."""
        results = self.filterer.filter_by_validator(
            self.sample_results,
            include_validators=["not_null"],
        )
        assert all(r.validator_name == "not_null" for r in results)

    def test_filter_failed(self):
        """Test filtering failed results."""
        failed = self.filterer.filter_failed(self.sample_results)
        assert all(not r.success for r in failed)
        assert len(failed) == 2

    def test_filter_passed(self):
        """Test filtering passed results."""
        passed = self.filterer.filter_passed(self.sample_results)
        assert all(r.success for r in passed)
        assert len(passed) == 2

    def test_sort_by_severity(self):
        """Test sorting by severity.

        Severity levels: critical=0, high=1, medium=2, low=3, info=4
        ascending=False (default) means reverse=True in sorted(),
        so highest level numbers come first (info=4, then medium=2, etc.)
        """
        sorted_results = self.filterer.sort_by_severity(self.sample_results)
        levels = [{"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}.get(r.severity, 5)
                  for r in sorted_results]
        # With reverse=True, should be descending order of level numbers
        assert levels == sorted(levels, reverse=True)

    def test_sort_by_severity_ascending(self):
        """Test sorting by severity ascending."""
        sorted_results = self.filterer.sort_by_severity(
            self.sample_results,
            ascending=True,
        )
        # ascending=True means reverse=False, so lowest level numbers first (critical=0)
        levels = [{"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}.get(r.severity, 5)
                  for r in sorted_results]
        assert levels == sorted(levels)

    def test_sort_by_column(self):
        """Test sorting by column name."""
        sorted_results = self.filterer.sort_by_column(self.sample_results)
        columns = [r.column for r in sorted_results]
        assert columns == sorted(columns)

    def test_limit(self):
        """Test limiting results."""
        limited = self.filterer.limit(self.sample_results, count=2)
        assert len(limited) == 2


class TestSerializationMixin:
    """Tests for SerializationMixin."""

    def setup_method(self):
        self.serializer = SerializationTest()
        self.sample_data = {
            "name": "Test Report",
            "results": [
                {"column": "id", "passed": True},
                {"column": "email", "passed": False},
            ],
        }

    def test_to_json(self):
        """Test JSON serialization."""
        result = self.serializer.to_json(self.sample_data)
        parsed = json.loads(result)
        assert parsed["name"] == "Test Report"

    def test_to_json_with_indent(self):
        """Test JSON with indentation."""
        result = self.serializer.to_json(self.sample_data, indent=2)
        assert "\n" in result
        assert "  " in result

    def test_to_json_compact(self):
        """Test compact JSON."""
        result = self.serializer.to_json(self.sample_data, indent=None)
        assert "\n" not in result

    def test_to_json_date_handling(self):
        """Test JSON date serialization."""
        data = {"timestamp": datetime(2024, 1, 15, 10, 30)}
        result = self.serializer.to_json(data)
        parsed = json.loads(result)
        assert "2024" in parsed["timestamp"]

    def test_to_csv(self):
        """Test CSV serialization."""
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        result = self.serializer.to_csv(data)

        assert "name" in result
        assert "age" in result
        assert "Alice" in result
        assert "Bob" in result

    def test_to_csv_custom_delimiter(self):
        """Test CSV with custom delimiter."""
        data = [{"a": 1, "b": 2}]
        result = self.serializer.to_csv(data, delimiter=";")
        assert ";" in result

    def test_to_csv_custom_columns(self):
        """Test CSV with custom columns."""
        data = [{"a": 1, "b": 2, "c": 3}]
        result = self.serializer.to_csv(data, columns=["a", "b"])
        lines = result.strip().split("\n")
        header = lines[0]
        assert "a" in header
        assert "b" in header
        # c should not be in header
        assert "c" not in header.split(",")

    def test_to_xml_element(self):
        """Test XML element creation."""
        elem = self.serializer.to_xml_element(
            tag="result",
            value="content",
            attributes={"id": "1"},
        )

        assert isinstance(elem, str)
        assert "<result" in elem
        assert 'id="1"' in elem
        assert "content" in elem

    def test_to_xml_element_self_closing(self):
        """Test self-closing XML element."""
        elem = self.serializer.to_xml_element(tag="empty")
        assert "<empty/>" in elem

    def test_to_xml_element_with_children(self):
        """Test XML element with children."""
        child1 = self.serializer.to_xml_element("child1", value="c1")
        child2 = self.serializer.to_xml_element("child2", value="c2")

        parent = self.serializer.to_xml_element(
            "parent",
            children=[child1, child2],
        )

        assert "<parent>" in parent
        assert "</parent>" in parent
        assert "<child1>" in parent
        assert "<child2>" in parent


class TestTemplatingMixin:
    """Tests for TemplatingMixin."""

    def setup_method(self):
        self.templater = TemplatingTest()

    def test_interpolate(self):
        """Test f-string style interpolation."""
        template = "Status: {status}, Count: {count}"
        result = self.templater.interpolate(template, {"status": "OK", "count": 10})
        assert result == "Status: OK, Count: 10"

    def test_interpolate_missing_key(self):
        """Test interpolation with missing key raises error."""
        template = "Hello, {name}!"
        with pytest.raises(KeyError):
            self.templater.interpolate(template, {})

    @pytest.mark.skipif(
        not _check_jinja2_available(),
        reason="jinja2 not installed",
    )
    def test_render_template(self):
        """Test Jinja2 template rendering."""
        template = "Hello, {{ name }}!"
        result = self.templater.render_template(template, {"name": "World"})
        assert result == "Hello, World!"

    @pytest.mark.skipif(
        not _check_jinja2_available(),
        reason="jinja2 not installed",
    )
    def test_render_template_multiple_vars(self):
        """Test template with multiple variables."""
        template = "{{ name }} has {{ count }} items"
        result = self.templater.render_template(
            template,
            {"name": "Alice", "count": 5},
        )
        assert result == "Alice has 5 items"


class TestStreamingMixin:
    """Tests for StreamingMixin."""

    def setup_method(self):
        self.streamer = StreamingTest()
        self.sample_results = [
            MockValidatorResult(validator_name="v1", column="id", success=True),
            MockValidatorResult(validator_name="v2", column="email", success=False),
            MockValidatorResult(validator_name="v3", column="age", success=True),
        ]

    def test_stream_results(self):
        """Test streaming results in chunks."""
        results = list(range(10))  # Create simple list
        stream = self.streamer.stream_results(results, chunk_size=3)

        batches = list(stream)
        assert len(batches) == 4  # 3 + 3 + 3 + 1

    def test_stream_lines(self):
        """Test streaming formatted lines."""

        def formatter(r):
            return f"Column: {r.column}"

        lines = list(self.streamer.stream_lines(self.sample_results, formatter))
        assert len(lines) == 3
        assert "Column: id" in lines

    def test_render_streaming(self):
        """Test streaming render."""

        def formatter(item):
            return f"Item: {item.column}"

        output = self.streamer.render_streaming(self.sample_results, formatter)

        assert "Item: id" in output
        assert "Item: email" in output
        assert "Item: age" in output


class TestMixinCombination:
    """Tests for combining multiple mixins."""

    def test_combined_mixins(self):
        """Test using multiple mixins together."""

        class CombinedReporter(FormattingMixin, FilteringMixin, SerializationMixin):
            pass

        reporter = CombinedReporter()

        results = [
            MockValidatorResult(validator_name="v1", column="id", severity="info", success=True),
            MockValidatorResult(validator_name="v2", column="email", severity="error", success=False),
        ]

        # Filter
        failed = reporter.filter_failed(results)
        assert len(failed) == 1

        # Format as table (use dict representation)
        table_data = [{"column": r.column, "success": r.success} for r in failed]
        table = reporter.format_as_table(table_data)
        assert "email" in table

        # Serialize
        json_output = reporter.to_json(table_data)
        assert "email" in json_output
