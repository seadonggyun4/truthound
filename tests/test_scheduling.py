"""Tests for the scheduling module.

Comprehensive test suite for cron expression parsing, matching,
next-run calculation, and builder functionality.
"""

import pytest
from datetime import datetime, timedelta

from truthound.scheduling import (
    # Core
    CronExpression,
    CronField,
    CronFieldType,
    # Parser
    CronParser,
    CronParseError,
    # Builder
    CronBuilder,
    # Iterator
    CronIterator,
    # Validation
    validate_expression,
    is_valid_expression,
    # Presets
    YEARLY,
    ANNUALLY,
    MONTHLY,
    WEEKLY,
    DAILY,
    MIDNIGHT,
    HOURLY,
    EVERY_MINUTE,
    EVERY_SECOND,
    WEEKDAYS_9AM,
    WEEKDAYS_6PM,
    FIRST_OF_MONTH,
    LAST_OF_MONTH,
)
from truthound.scheduling.presets import (
    BUSINESS_START,
    BUSINESS_END,
    BUSINESS_HOURS_15MIN,
    BUSINESS_HOURS_HOURLY,
    LAST_FRIDAY,
    FIRST_MONDAY,
    EVERY_5_MIN,
    EVERY_15_MIN,
    EVERY_30_MIN,
    EVERY_2_HOURS,
    EVERY_4_HOURS,
    EVERY_6_HOURS,
    TWICE_DAILY,
    THREE_TIMES_DAILY,
    WEEKENDS_NOON,
    NIGHTLY_2AM,
    NIGHTLY_3AM,
    SUNDAY_MAINTENANCE,
    QUARTERLY,
    END_OF_QUARTER,
    get_preset,
    list_presets,
    PRESETS,
)
from truthound.scheduling.cron import (
    FieldConstraints,
    FIELD_CONSTRAINTS,
    MatchContext,
)


# =============================================================================
# CronParseError Tests
# =============================================================================


class TestCronParseError:
    """Tests for CronParseError exception."""

    def test_error_message(self):
        """Test error message formatting."""
        error = CronParseError("Invalid expression", "* * * * * *", 5)
        assert "Invalid expression" in str(error)
        assert error.expression == "* * * * * *"
        assert error.position == 5

    def test_error_without_position(self):
        """Test error without position info."""
        error = CronParseError("Simple error")
        assert error.expression == ""
        assert error.position == -1


# =============================================================================
# FieldConstraints Tests
# =============================================================================


class TestFieldConstraints:
    """Tests for field constraints."""

    def test_second_constraints(self):
        """Test second field constraints."""
        c = FIELD_CONSTRAINTS[CronFieldType.SECOND]
        assert c.min_value == 0
        assert c.max_value == 59
        assert not c.supports_l
        assert not c.supports_w

    def test_minute_constraints(self):
        """Test minute field constraints."""
        c = FIELD_CONSTRAINTS[CronFieldType.MINUTE]
        assert c.min_value == 0
        assert c.max_value == 59

    def test_hour_constraints(self):
        """Test hour field constraints."""
        c = FIELD_CONSTRAINTS[CronFieldType.HOUR]
        assert c.min_value == 0
        assert c.max_value == 23

    def test_day_of_month_constraints(self):
        """Test day of month field constraints."""
        c = FIELD_CONSTRAINTS[CronFieldType.DAY_OF_MONTH]
        assert c.min_value == 1
        assert c.max_value == 31
        assert c.supports_l
        assert c.supports_w
        assert c.supports_question

    def test_month_constraints(self):
        """Test month field constraints."""
        c = FIELD_CONSTRAINTS[CronFieldType.MONTH]
        assert c.min_value == 1
        assert c.max_value == 12
        assert c.names == {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
            "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
            "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
        }

    def test_day_of_week_constraints(self):
        """Test day of week field constraints."""
        c = FIELD_CONSTRAINTS[CronFieldType.DAY_OF_WEEK]
        assert c.min_value == 0
        assert c.max_value == 6
        assert c.names == {
            "SUN": 0, "MON": 1, "TUE": 2, "WED": 3,
            "THU": 4, "FRI": 5, "SAT": 6,
        }
        assert c.supports_l
        assert c.supports_hash
        assert c.supports_question

    def test_year_constraints(self):
        """Test year field constraints."""
        c = FIELD_CONSTRAINTS[CronFieldType.YEAR]
        assert c.min_value == 1970
        assert c.max_value == 2099


# =============================================================================
# CronParser Basic Tests
# =============================================================================


class TestCronParserBasic:
    """Tests for basic cron expression parsing."""

    def test_parse_standard_expression(self):
        """Test parsing standard 5-field expression."""
        parser = CronParser("0 9 * * *")
        fields = parser.parse()
        assert len(fields) == 5

    def test_parse_extended_6_field(self):
        """Test parsing 6-field expression with seconds."""
        parser = CronParser("0 0 9 * * *")
        fields = parser.parse()
        assert len(fields) == 6
        assert fields[0].field_type == CronFieldType.SECOND

    def test_parse_extended_7_field(self):
        """Test parsing 7-field expression with years."""
        parser = CronParser("0 0 9 * * * 2024")
        fields = parser.parse()
        assert len(fields) == 7
        assert fields[6].field_type == CronFieldType.YEAR

    def test_parse_invalid_field_count(self):
        """Test parsing with invalid number of fields."""
        parser = CronParser("0 9 *")
        with pytest.raises(CronParseError) as exc:
            parser.parse()
        assert "Invalid number of fields: 3" in str(exc.value)

    def test_parse_wildcard(self):
        """Test parsing wildcard (*)."""
        parser = CronParser("* * * * *")
        fields = parser.parse()
        assert all(f.is_any for f in fields)

    def test_parse_single_value(self):
        """Test parsing single values."""
        parser = CronParser("30 9 15 6 3")
        fields = parser.parse()
        assert 30 in fields[0].values
        assert 9 in fields[1].values
        assert 15 in fields[2].values
        assert 6 in fields[3].values
        assert 3 in fields[4].values


# =============================================================================
# CronParser Range Tests
# =============================================================================


class TestCronParserRange:
    """Tests for range parsing."""

    def test_parse_simple_range(self):
        """Test parsing simple range (1-5)."""
        expr = CronExpression.parse("0 9-17 * * *")
        hour_field = expr.get_field(CronFieldType.HOUR)
        assert hour_field is not None
        assert hour_field.values == frozenset(range(9, 18))

    def test_parse_range_with_names(self):
        """Test parsing range with named values."""
        expr = CronExpression.parse("0 9 * * MON-FRI")
        dow_field = expr.get_field(CronFieldType.DAY_OF_WEEK)
        assert dow_field is not None
        assert dow_field.values == frozenset([1, 2, 3, 4, 5])

    def test_parse_wraparound_range(self):
        """Test parsing wraparound range (FRI-MON)."""
        expr = CronExpression.parse("0 9 * * FRI-MON")
        dow_field = expr.get_field(CronFieldType.DAY_OF_WEEK)
        assert dow_field is not None
        assert dow_field.values == frozenset([0, 1, 5, 6])  # FRI, SAT, SUN, MON

    def test_parse_month_range_with_names(self):
        """Test parsing month range with names."""
        expr = CronExpression.parse("0 9 * JAN-MAR *")
        month_field = expr.get_field(CronFieldType.MONTH)
        assert month_field is not None
        assert month_field.values == frozenset([1, 2, 3])


# =============================================================================
# CronParser Step Tests
# =============================================================================


class TestCronParserStep:
    """Tests for step parsing."""

    def test_parse_step_from_start(self):
        """Test parsing step from start (*/15)."""
        expr = CronExpression.parse("*/15 * * * *")
        minute_field = expr.get_field(CronFieldType.MINUTE)
        assert minute_field is not None
        assert minute_field.values == frozenset([0, 15, 30, 45])

    def test_parse_step_from_value(self):
        """Test parsing step from specific value (5/10)."""
        expr = CronExpression.parse("5/10 * * * *")
        minute_field = expr.get_field(CronFieldType.MINUTE)
        assert minute_field is not None
        assert minute_field.values == frozenset([5, 15, 25, 35, 45, 55])

    def test_parse_step_in_range(self):
        """Test parsing step in range (10-30/5)."""
        expr = CronExpression.parse("10-30/5 * * * *")
        minute_field = expr.get_field(CronFieldType.MINUTE)
        assert minute_field is not None
        assert minute_field.values == frozenset([10, 15, 20, 25, 30])

    def test_parse_hour_step(self):
        """Test parsing hour step (*/2)."""
        expr = CronExpression.parse("0 */2 * * *")
        hour_field = expr.get_field(CronFieldType.HOUR)
        assert hour_field is not None
        assert hour_field.values == frozenset([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])

    def test_invalid_step_zero(self):
        """Test that step of 0 raises error."""
        with pytest.raises(CronParseError):
            CronExpression.parse("*/0 * * * *")

    def test_invalid_step_negative(self):
        """Test that negative step raises error."""
        with pytest.raises(CronParseError):
            CronExpression.parse("*/-5 * * * *")


# =============================================================================
# CronParser List Tests
# =============================================================================


class TestCronParserList:
    """Tests for list parsing."""

    def test_parse_simple_list(self):
        """Test parsing simple list (1,3,5)."""
        expr = CronExpression.parse("0 9,12,18 * * *")
        hour_field = expr.get_field(CronFieldType.HOUR)
        assert hour_field is not None
        assert hour_field.values == frozenset([9, 12, 18])

    def test_parse_mixed_list(self):
        """Test parsing list with ranges and values."""
        expr = CronExpression.parse("0 9-11,14,16-18 * * *")
        hour_field = expr.get_field(CronFieldType.HOUR)
        assert hour_field is not None
        assert hour_field.values == frozenset([9, 10, 11, 14, 16, 17, 18])

    def test_parse_weekday_list(self):
        """Test parsing weekday list."""
        expr = CronExpression.parse("0 9 * * MON,WED,FRI")
        dow_field = expr.get_field(CronFieldType.DAY_OF_WEEK)
        assert dow_field is not None
        assert dow_field.values == frozenset([1, 3, 5])


# =============================================================================
# CronParser Special Characters Tests
# =============================================================================


class TestCronParserSpecialChars:
    """Tests for special character parsing."""

    def test_parse_question_mark(self):
        """Test parsing ? (no specific value)."""
        expr = CronExpression.parse("0 9 ? * MON")
        dom_field = expr.get_field(CronFieldType.DAY_OF_MONTH)
        assert dom_field is not None
        assert dom_field.is_any

    def test_parse_last_day_of_month(self):
        """Test parsing L in day of month."""
        expr = CronExpression.parse("0 9 L * *")
        dom_field = expr.get_field(CronFieldType.DAY_OF_MONTH)
        assert dom_field is not None
        # L field has special matching behavior

    def test_parse_last_weekday(self):
        """Test parsing nL (last Friday)."""
        expr = CronExpression.parse("0 9 * * 5L")
        dow_field = expr.get_field(CronFieldType.DAY_OF_WEEK)
        assert dow_field is not None
        assert 5 in dow_field.values  # Friday

    def test_parse_nearest_weekday(self):
        """Test parsing W (nearest weekday)."""
        expr = CronExpression.parse("0 9 15W * ?")
        dom_field = expr.get_field(CronFieldType.DAY_OF_MONTH)
        assert dom_field is not None

    def test_parse_nth_weekday(self):
        """Test parsing # (nth weekday)."""
        expr = CronExpression.parse("0 9 * * 1#2")  # Second Monday
        dow_field = expr.get_field(CronFieldType.DAY_OF_WEEK)
        assert dow_field is not None

    def test_invalid_question_in_minute(self):
        """Test that ? in minute field raises error."""
        with pytest.raises(CronParseError):
            CronExpression.parse("? 9 * * *")

    def test_invalid_l_in_hour(self):
        """Test that L in hour field raises error."""
        with pytest.raises(CronParseError):
            CronExpression.parse("0 L * * *")

    def test_invalid_hash_in_day(self):
        """Test that # in day of month field raises error."""
        with pytest.raises(CronParseError):
            CronExpression.parse("0 9 1#2 * *")


# =============================================================================
# CronParser Alias Tests
# =============================================================================


class TestCronParserAlias:
    """Tests for predefined expression aliases."""

    def test_yearly_alias(self):
        """Test @yearly alias."""
        expr = CronExpression.parse("@yearly")
        # Should match Jan 1 at midnight
        assert expr.matches(datetime(2024, 1, 1, 0, 0))
        assert not expr.matches(datetime(2024, 2, 1, 0, 0))

    def test_annually_alias(self):
        """Test @annually alias."""
        expr = CronExpression.parse("@annually")
        # Same as yearly
        assert expr.matches(datetime(2024, 1, 1, 0, 0))
        assert not expr.matches(datetime(2024, 6, 15, 0, 0))

    def test_monthly_alias(self):
        """Test @monthly alias."""
        expr = CronExpression.parse("@monthly")
        # Should match first of every month at midnight
        assert expr.matches(datetime(2024, 1, 1, 0, 0))
        assert expr.matches(datetime(2024, 6, 1, 0, 0))
        assert not expr.matches(datetime(2024, 6, 2, 0, 0))

    def test_weekly_alias(self):
        """Test @weekly alias."""
        expr = CronExpression.parse("@weekly")
        # Should match Sunday at midnight
        # Jan 7, 2024 is Sunday
        assert expr.matches(datetime(2024, 1, 7, 0, 0))
        assert not expr.matches(datetime(2024, 1, 8, 0, 0))

    def test_daily_alias(self):
        """Test @daily alias."""
        expr = CronExpression.parse("@daily")
        # Should match every day at midnight
        assert expr.matches(datetime(2024, 1, 15, 0, 0))
        assert not expr.matches(datetime(2024, 1, 15, 1, 0))

    def test_midnight_alias(self):
        """Test @midnight alias."""
        expr = CronExpression.parse("@midnight")
        # Same as daily
        assert expr.matches(datetime(2024, 1, 15, 0, 0))
        assert not expr.matches(datetime(2024, 1, 15, 12, 0))

    def test_hourly_alias(self):
        """Test @hourly alias."""
        expr = CronExpression.parse("@hourly")
        # Should match every hour at minute 0
        assert expr.matches(datetime(2024, 1, 15, 9, 0))
        assert expr.matches(datetime(2024, 1, 15, 10, 0))
        assert not expr.matches(datetime(2024, 1, 15, 10, 30))

    def test_every_minute_alias(self):
        """Test @every_minute alias."""
        expr = CronExpression.parse("@every_minute")
        # Should match every minute
        assert expr.matches(datetime(2024, 1, 15, 9, 30))
        assert expr.matches(datetime(2024, 1, 15, 9, 31))

    def test_every_second_alias(self):
        """Test @every_second alias."""
        expr = CronExpression.parse("@every_second")
        assert expr.has_seconds


# =============================================================================
# CronExpression Matching Tests
# =============================================================================


class TestCronExpressionMatching:
    """Tests for cron expression matching."""

    def test_match_exact_time(self):
        """Test matching exact time."""
        expr = CronExpression.parse("30 9 * * *")
        assert expr.matches(datetime(2024, 1, 15, 9, 30))
        assert not expr.matches(datetime(2024, 1, 15, 9, 31))
        assert not expr.matches(datetime(2024, 1, 15, 10, 30))

    def test_match_wildcard(self):
        """Test matching with wildcards."""
        expr = CronExpression.parse("* * * * *")
        assert expr.matches(datetime(2024, 1, 15, 9, 30))
        assert expr.matches(datetime(2024, 6, 20, 14, 45))

    def test_match_range(self):
        """Test matching with range."""
        expr = CronExpression.parse("0 9-17 * * *")
        assert expr.matches(datetime(2024, 1, 15, 9, 0))
        assert expr.matches(datetime(2024, 1, 15, 12, 0))
        assert expr.matches(datetime(2024, 1, 15, 17, 0))
        assert not expr.matches(datetime(2024, 1, 15, 8, 0))
        assert not expr.matches(datetime(2024, 1, 15, 18, 0))

    def test_match_step(self):
        """Test matching with step."""
        expr = CronExpression.parse("*/15 * * * *")
        assert expr.matches(datetime(2024, 1, 15, 9, 0))
        assert expr.matches(datetime(2024, 1, 15, 9, 15))
        assert expr.matches(datetime(2024, 1, 15, 9, 30))
        assert expr.matches(datetime(2024, 1, 15, 9, 45))
        assert not expr.matches(datetime(2024, 1, 15, 9, 10))

    def test_match_weekday(self):
        """Test matching specific weekdays."""
        expr = CronExpression.parse("0 9 * * MON-FRI")
        # Monday Jan 15, 2024
        assert expr.matches(datetime(2024, 1, 15, 9, 0))
        # Sunday Jan 14, 2024
        assert not expr.matches(datetime(2024, 1, 14, 9, 0))
        # Saturday Jan 13, 2024
        assert not expr.matches(datetime(2024, 1, 13, 9, 0))

    def test_match_specific_month(self):
        """Test matching specific month."""
        expr = CronExpression.parse("0 9 1 JAN *")
        assert expr.matches(datetime(2024, 1, 1, 9, 0))
        assert not expr.matches(datetime(2024, 2, 1, 9, 0))

    def test_match_with_seconds(self):
        """Test matching with seconds field."""
        expr = CronExpression.parse("30 0 9 * * *")
        dt = datetime(2024, 1, 15, 9, 0, 30)
        assert expr.matches(dt)
        dt2 = datetime(2024, 1, 15, 9, 0, 0)
        assert not expr.matches(dt2)


# =============================================================================
# CronExpression Last Day Tests
# =============================================================================


class TestCronExpressionLastDay:
    """Tests for L (last) modifier matching."""

    def test_match_last_day_of_january(self):
        """Test matching last day of January (31)."""
        expr = CronExpression.parse("0 9 L * *")
        assert expr.matches(datetime(2024, 1, 31, 9, 0))
        assert not expr.matches(datetime(2024, 1, 30, 9, 0))

    def test_match_last_day_of_february(self):
        """Test matching last day of February."""
        expr = CronExpression.parse("0 9 L * *")
        # Leap year 2024
        assert expr.matches(datetime(2024, 2, 29, 9, 0))
        assert not expr.matches(datetime(2024, 2, 28, 9, 0))

    def test_match_last_day_of_april(self):
        """Test matching last day of April (30)."""
        expr = CronExpression.parse("0 9 L * *")
        assert expr.matches(datetime(2024, 4, 30, 9, 0))
        assert not expr.matches(datetime(2024, 4, 29, 9, 0))


# =============================================================================
# CronExpression Next Run Tests
# =============================================================================


class TestCronExpressionNextRun:
    """Tests for next run calculation."""

    def test_next_simple(self):
        """Test simple next run calculation."""
        expr = CronExpression.parse("0 9 * * *")
        after = datetime(2024, 1, 15, 8, 0)
        next_run = expr.next(after)
        assert next_run is not None
        assert next_run == datetime(2024, 1, 15, 9, 0)

    def test_next_same_hour(self):
        """Test next run within same hour."""
        expr = CronExpression.parse("30 * * * *")
        after = datetime(2024, 1, 15, 9, 0)
        next_run = expr.next(after)
        assert next_run is not None
        assert next_run == datetime(2024, 1, 15, 9, 30)

    def test_next_crosses_day(self):
        """Test next run crossing day boundary."""
        expr = CronExpression.parse("0 9 * * *")
        after = datetime(2024, 1, 15, 10, 0)
        next_run = expr.next(after)
        assert next_run is not None
        assert next_run == datetime(2024, 1, 16, 9, 0)

    def test_next_crosses_month(self):
        """Test next run crossing month boundary."""
        expr = CronExpression.parse("0 9 1 * *")
        after = datetime(2024, 1, 15, 10, 0)
        next_run = expr.next(after)
        assert next_run is not None
        assert next_run == datetime(2024, 2, 1, 9, 0)

    def test_next_crosses_year(self):
        """Test next run crossing year boundary."""
        expr = CronExpression.parse("0 0 1 1 *")
        after = datetime(2024, 6, 15, 10, 0)
        next_run = expr.next(after)
        assert next_run is not None
        assert next_run == datetime(2025, 1, 1, 0, 0)

    def test_next_specific_weekday(self):
        """Test next run on specific weekday."""
        expr = CronExpression.parse("0 9 * * MON")
        after = datetime(2024, 1, 16, 10, 0)  # Tuesday
        next_run = expr.next(after)
        assert next_run is not None
        # Next Monday is Jan 22
        assert next_run.weekday() == 0  # Monday

    def test_next_with_seconds(self):
        """Test next run with seconds field."""
        expr = CronExpression.parse("*/10 * * * * *")
        after = datetime(2024, 1, 15, 9, 0, 5)
        next_run = expr.next(after)
        assert next_run is not None
        assert next_run.second == 10

    def test_next_n(self):
        """Test getting next n runs."""
        expr = CronExpression.parse("0 * * * *")
        after = datetime(2024, 1, 15, 9, 0)
        next_runs = expr.next_n(5, after)
        assert len(next_runs) == 5
        assert next_runs[0] == datetime(2024, 1, 15, 10, 0)
        assert next_runs[1] == datetime(2024, 1, 15, 11, 0)
        assert next_runs[4] == datetime(2024, 1, 15, 14, 0)


# =============================================================================
# CronIterator Tests
# =============================================================================


class TestCronIterator:
    """Tests for CronIterator."""

    def test_iterator_basic(self):
        """Test basic iterator functionality."""
        expr = CronExpression.parse("0 * * * *")
        after = datetime(2024, 1, 15, 9, 0)
        iterator = expr.iter(after, limit=3)

        results = list(iterator)
        assert len(results) == 3
        assert results[0] == datetime(2024, 1, 15, 10, 0)
        assert results[1] == datetime(2024, 1, 15, 11, 0)
        assert results[2] == datetime(2024, 1, 15, 12, 0)

    def test_iterator_no_limit(self):
        """Test iterator without limit (should still work with manual break)."""
        expr = CronExpression.parse("0 0 1 * *")
        after = datetime(2024, 1, 1, 0, 0)
        iterator = expr.iter(after)

        count = 0
        for dt in iterator:
            count += 1
            if count >= 5:
                break

        assert count == 5

    def test_iterator_with_limit(self):
        """Test iterator respects limit."""
        expr = CronExpression.parse("* * * * *")
        iterator = CronIterator(expr, datetime(2024, 1, 1, 0, 0), limit=10)

        results = list(iterator)
        assert len(results) == 10


# =============================================================================
# CronBuilder Tests
# =============================================================================


class TestCronBuilder:
    """Tests for CronBuilder."""

    def test_builder_at_minute(self):
        """Test setting specific minutes."""
        expr = CronBuilder().at_minute(0, 30).build()
        assert "0,30" in expr.expression

    def test_builder_every_n_minutes(self):
        """Test every n minutes."""
        expr = CronBuilder().every_n_minutes(15).build()
        assert "*/15" in expr.expression

    def test_builder_at_hour(self):
        """Test setting specific hours."""
        expr = CronBuilder().at_hour(9, 17).build()
        assert "9,17" in expr.expression

    def test_builder_every_n_hours(self):
        """Test every n hours."""
        expr = CronBuilder().every_n_hours(2).build()
        assert "*/2" in expr.expression

    def test_builder_on_day(self):
        """Test setting specific days."""
        expr = CronBuilder().on_day(1, 15).build()
        assert "1,15" in expr.expression

    def test_builder_on_last_day(self):
        """Test last day of month."""
        expr = CronBuilder().on_last_day().build()
        assert " L " in expr.expression

    def test_builder_on_weekday_nearest(self):
        """Test weekday nearest to day."""
        expr = CronBuilder().on_weekday_nearest(15).build()
        assert "15W" in expr.expression

    def test_builder_in_month(self):
        """Test setting specific months."""
        expr = CronBuilder().in_month(1, 6, 12).build()
        assert "1,6,12" in expr.expression

    def test_builder_on_weekdays(self):
        """Test weekdays (MON-FRI)."""
        expr = CronBuilder().on_weekdays().build()
        assert "MON-FRI" in expr.expression

    def test_builder_on_weekends(self):
        """Test weekends (SAT,SUN)."""
        expr = CronBuilder().on_weekends().build()
        assert "SAT,SUN" in expr.expression

    def test_builder_daily_at(self):
        """Test daily at specific time."""
        expr = CronBuilder().daily_at(9, 30).build()
        minute_field = expr.get_field(CronFieldType.MINUTE)
        hour_field = expr.get_field(CronFieldType.HOUR)
        assert minute_field is not None and 30 in minute_field.values
        assert hour_field is not None and 9 in hour_field.values

    def test_builder_hourly_at(self):
        """Test hourly at specific minute."""
        expr = CronBuilder().hourly_at(30).build()
        minute_field = expr.get_field(CronFieldType.MINUTE)
        assert minute_field is not None and 30 in minute_field.values

    def test_builder_with_seconds(self):
        """Test builder with seconds."""
        expr = CronBuilder().with_seconds().at_second(0, 30).build()
        assert expr.has_seconds

    def test_builder_every_n_seconds(self):
        """Test every n seconds."""
        expr = CronBuilder().every_n_seconds(10).build()
        assert expr.has_seconds
        assert "*/10" in expr.expression

    def test_builder_complex_expression(self):
        """Test building complex expression."""
        expr = (
            CronBuilder()
            .at_minute(0)
            .at_hour(9, 12, 18)
            .on_weekdays()
            .build()
        )
        # Should run at 9am, 12pm, 6pm on weekdays
        assert expr.matches(datetime(2024, 1, 15, 9, 0))  # Monday 9am
        assert expr.matches(datetime(2024, 1, 15, 12, 0))  # Monday 12pm
        assert expr.matches(datetime(2024, 1, 15, 18, 0))  # Monday 6pm
        assert not expr.matches(datetime(2024, 1, 14, 9, 0))  # Sunday


# =============================================================================
# CronExpression Factory Tests
# =============================================================================


class TestCronExpressionFactory:
    """Tests for CronExpression factory methods."""

    def test_yearly_factory(self):
        """Test yearly() factory."""
        expr = CronExpression.yearly()
        assert expr.matches(datetime(2024, 1, 1, 0, 0))
        assert not expr.matches(datetime(2024, 6, 1, 0, 0))

    def test_monthly_factory(self):
        """Test monthly() factory."""
        expr = CronExpression.monthly()
        assert expr.matches(datetime(2024, 1, 1, 0, 0))
        assert expr.matches(datetime(2024, 2, 1, 0, 0))
        assert not expr.matches(datetime(2024, 1, 2, 0, 0))

    def test_weekly_factory(self):
        """Test weekly() factory."""
        expr = CronExpression.weekly()
        # Jan 7, 2024 is Sunday
        assert expr.matches(datetime(2024, 1, 7, 0, 0))
        assert not expr.matches(datetime(2024, 1, 8, 0, 0))

    def test_daily_factory(self):
        """Test daily() factory."""
        expr = CronExpression.daily()
        assert expr.matches(datetime(2024, 1, 15, 0, 0))
        assert not expr.matches(datetime(2024, 1, 15, 1, 0))

    def test_hourly_factory(self):
        """Test hourly() factory."""
        expr = CronExpression.hourly()
        assert expr.matches(datetime(2024, 1, 15, 9, 0))
        assert expr.matches(datetime(2024, 1, 15, 10, 0))
        assert not expr.matches(datetime(2024, 1, 15, 10, 30))

    def test_every_n_minutes_factory(self):
        """Test every_n_minutes() factory."""
        expr = CronExpression.every_n_minutes(15)
        assert expr.matches(datetime(2024, 1, 15, 9, 0))
        assert expr.matches(datetime(2024, 1, 15, 9, 15))
        assert expr.matches(datetime(2024, 1, 15, 9, 30))
        assert not expr.matches(datetime(2024, 1, 15, 9, 10))

    def test_every_n_hours_factory(self):
        """Test every_n_hours() factory."""
        expr = CronExpression.every_n_hours(2)
        assert expr.matches(datetime(2024, 1, 15, 0, 0))
        assert expr.matches(datetime(2024, 1, 15, 2, 0))
        assert expr.matches(datetime(2024, 1, 15, 4, 0))
        assert not expr.matches(datetime(2024, 1, 15, 1, 0))


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
    """Tests for expression validation."""

    def test_validate_valid_expression(self):
        """Test validating valid expression."""
        errors = validate_expression("0 9 * * *")
        assert len(errors) == 0

    def test_validate_invalid_expression(self):
        """Test validating invalid expression."""
        errors = validate_expression("invalid")
        assert len(errors) > 0

    def test_validate_out_of_range(self):
        """Test validating out of range values."""
        errors = validate_expression("60 * * * *")  # Minute out of range
        assert len(errors) > 0

    def test_is_valid_expression(self):
        """Test is_valid_expression function."""
        assert is_valid_expression("0 9 * * *")
        assert is_valid_expression("*/15 * * * *")
        assert is_valid_expression("@daily")
        assert not is_valid_expression("invalid")
        assert not is_valid_expression("* * *")  # Wrong number of fields


# =============================================================================
# Presets Tests
# =============================================================================


class TestPresets:
    """Tests for predefined cron expressions."""

    def test_yearly_preset(self):
        """Test YEARLY preset."""
        assert YEARLY.matches(datetime(2024, 1, 1, 0, 0))
        assert ANNUALLY.matches(datetime(2024, 1, 1, 0, 0))

    def test_monthly_preset(self):
        """Test MONTHLY preset."""
        assert MONTHLY.matches(datetime(2024, 6, 1, 0, 0))

    def test_weekly_preset(self):
        """Test WEEKLY preset."""
        # Sunday Jan 7, 2024
        assert WEEKLY.matches(datetime(2024, 1, 7, 0, 0))

    def test_daily_preset(self):
        """Test DAILY preset."""
        assert DAILY.matches(datetime(2024, 1, 15, 0, 0))
        assert MIDNIGHT.matches(datetime(2024, 1, 15, 0, 0))

    def test_hourly_preset(self):
        """Test HOURLY preset."""
        assert HOURLY.matches(datetime(2024, 1, 15, 9, 0))

    def test_every_minute_preset(self):
        """Test EVERY_MINUTE preset."""
        assert EVERY_MINUTE.matches(datetime(2024, 1, 15, 9, 30))
        assert EVERY_MINUTE.matches(datetime(2024, 1, 15, 9, 31))

    def test_every_second_preset(self):
        """Test EVERY_SECOND preset."""
        assert EVERY_SECOND.has_seconds

    def test_weekdays_9am_preset(self):
        """Test WEEKDAYS_9AM preset."""
        # Monday Jan 15, 2024
        assert WEEKDAYS_9AM.matches(datetime(2024, 1, 15, 9, 0))
        # Sunday
        assert not WEEKDAYS_9AM.matches(datetime(2024, 1, 14, 9, 0))

    def test_weekdays_6pm_preset(self):
        """Test WEEKDAYS_6PM preset."""
        # Friday Jan 19, 2024
        assert WEEKDAYS_6PM.matches(datetime(2024, 1, 19, 18, 0))

    def test_business_start_preset(self):
        """Test BUSINESS_START preset."""
        # 8 AM on Monday
        assert BUSINESS_START.matches(datetime(2024, 1, 15, 8, 0))

    def test_business_end_preset(self):
        """Test BUSINESS_END preset."""
        # 5 PM on Friday
        assert BUSINESS_END.matches(datetime(2024, 1, 19, 17, 0))

    def test_first_of_month_preset(self):
        """Test FIRST_OF_MONTH preset."""
        assert FIRST_OF_MONTH.matches(datetime(2024, 6, 1, 6, 0))

    def test_data_pipeline_presets(self):
        """Test data pipeline presets."""
        assert EVERY_5_MIN.matches(datetime(2024, 1, 15, 9, 5))
        assert EVERY_15_MIN.matches(datetime(2024, 1, 15, 9, 15))
        assert EVERY_30_MIN.matches(datetime(2024, 1, 15, 9, 30))
        assert EVERY_2_HOURS.matches(datetime(2024, 1, 15, 2, 0))
        assert EVERY_4_HOURS.matches(datetime(2024, 1, 15, 4, 0))
        assert EVERY_6_HOURS.matches(datetime(2024, 1, 15, 6, 0))
        assert TWICE_DAILY.matches(datetime(2024, 1, 15, 0, 0))
        assert TWICE_DAILY.matches(datetime(2024, 1, 15, 12, 0))
        assert THREE_TIMES_DAILY.matches(datetime(2024, 1, 15, 8, 0))
        assert THREE_TIMES_DAILY.matches(datetime(2024, 1, 15, 12, 0))
        assert THREE_TIMES_DAILY.matches(datetime(2024, 1, 15, 18, 0))

    def test_off_hours_presets(self):
        """Test off-hours presets."""
        # Sunday at noon
        assert WEEKENDS_NOON.matches(datetime(2024, 1, 14, 12, 0))
        assert NIGHTLY_2AM.matches(datetime(2024, 1, 15, 2, 0))
        assert NIGHTLY_3AM.matches(datetime(2024, 1, 15, 3, 0))
        # Sunday at 3 AM
        assert SUNDAY_MAINTENANCE.matches(datetime(2024, 1, 14, 3, 0))

    def test_quarterly_presets(self):
        """Test quarterly presets."""
        # First day of Q1
        assert QUARTERLY.matches(datetime(2024, 1, 1, 0, 0))
        # First day of Q2
        assert QUARTERLY.matches(datetime(2024, 4, 1, 0, 0))


# =============================================================================
# Preset Registry Tests
# =============================================================================


class TestPresetRegistry:
    """Tests for preset registry functions."""

    def test_get_preset_by_name(self):
        """Test getting preset by name."""
        expr = get_preset("daily")
        assert expr is not None
        assert expr == DAILY

    def test_get_preset_case_insensitive(self):
        """Test case-insensitive preset lookup."""
        assert get_preset("DAILY") == DAILY
        assert get_preset("Daily") == DAILY
        assert get_preset("dAiLy") == DAILY

    def test_get_preset_with_dashes(self):
        """Test preset lookup with dashes."""
        assert get_preset("every-minute") == EVERY_MINUTE
        assert get_preset("weekdays-9am") == WEEKDAYS_9AM

    def test_get_preset_not_found(self):
        """Test preset lookup for non-existent preset."""
        assert get_preset("nonexistent") is None

    def test_list_presets(self):
        """Test listing all presets."""
        names = list_presets()
        assert "daily" in names
        assert "hourly" in names
        assert "weekly" in names
        assert "monthly" in names
        assert "yearly" in names
        assert len(names) > 20

    def test_presets_dict(self):
        """Test PRESETS dictionary."""
        assert "daily" in PRESETS
        assert PRESETS["daily"] == DAILY
        assert len(PRESETS) > 20


# =============================================================================
# CronField Tests
# =============================================================================


class TestCronField:
    """Tests for CronField."""

    def test_field_repr(self):
        """Test field string representation."""
        expr = CronExpression.parse("30 9 * * *")
        minute_field = expr.get_field(CronFieldType.MINUTE)
        assert minute_field is not None
        assert "MINUTE" in repr(minute_field)

    def test_field_has_special(self):
        """Test has_special property."""
        expr = CronExpression.parse("0 9 L * *")
        dom_field = expr.get_field(CronFieldType.DAY_OF_MONTH)
        assert dom_field is not None
        assert dom_field.has_special

    def test_field_is_any(self):
        """Test is_any property."""
        expr = CronExpression.parse("* * * * *")
        minute_field = expr.get_field(CronFieldType.MINUTE)
        assert minute_field is not None
        assert minute_field.is_any


# =============================================================================
# CronExpression Properties Tests
# =============================================================================


class TestCronExpressionProperties:
    """Tests for CronExpression properties."""

    def test_expression_property(self):
        """Test expression property."""
        expr = CronExpression.parse("0 9 * * *")
        assert expr.expression == "0 9 * * *"

    def test_fields_property(self):
        """Test fields property."""
        expr = CronExpression.parse("0 9 * * *")
        assert len(expr.fields) == 5

    def test_has_seconds_property(self):
        """Test has_seconds property."""
        expr5 = CronExpression.parse("0 9 * * *")
        expr6 = CronExpression.parse("0 0 9 * * *")
        assert not expr5.has_seconds
        assert expr6.has_seconds

    def test_get_field(self):
        """Test get_field method."""
        expr = CronExpression.parse("30 9 15 6 1")
        assert expr.get_field(CronFieldType.MINUTE) is not None
        assert expr.get_field(CronFieldType.HOUR) is not None
        assert expr.get_field(CronFieldType.SECOND) is None

    def test_repr(self):
        """Test __repr__ method."""
        expr = CronExpression.parse("0 9 * * *")
        assert "CronExpression" in repr(expr)
        assert "0 9 * * *" in repr(expr)

    def test_str(self):
        """Test __str__ method."""
        expr = CronExpression.parse("0 9 * * *")
        assert str(expr) == "0 9 * * *"

    def test_equality(self):
        """Test equality comparison."""
        expr1 = CronExpression.parse("0 9 * * *")
        expr2 = CronExpression.parse("0 9 * * *")
        expr3 = CronExpression.parse("0 10 * * *")
        assert expr1 == expr2
        assert expr1 != expr3
        assert expr1 != "not an expression"

    def test_hash(self):
        """Test hash method."""
        expr1 = CronExpression.parse("0 9 * * *")
        expr2 = CronExpression.parse("0 9 * * *")
        assert hash(expr1) == hash(expr2)
        # Can be used in sets/dicts
        s = {expr1, expr2}
        assert len(s) == 1


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_leap_year_february(self):
        """Test February 29 in leap year."""
        expr = CronExpression.parse("0 9 29 2 *")
        assert expr.matches(datetime(2024, 2, 29, 9, 0))

    def test_month_boundary(self):
        """Test month boundary handling."""
        expr = CronExpression.parse("0 0 1 * *")
        after = datetime(2024, 1, 31, 12, 0)
        next_run = expr.next(after)
        assert next_run is not None
        assert next_run.month == 2
        assert next_run.day == 1

    def test_year_boundary(self):
        """Test year boundary handling."""
        expr = CronExpression.parse("0 0 1 1 *")
        after = datetime(2024, 12, 31, 12, 0)
        next_run = expr.next(after)
        assert next_run is not None
        assert next_run.year == 2025
        assert next_run.month == 1

    def test_dst_transition(self):
        """Test expression still works around DST (basic sanity check)."""
        expr = CronExpression.parse("0 2 * * *")
        # Just ensure it doesn't crash
        after = datetime(2024, 3, 10, 0, 0)  # Near US DST transition
        next_run = expr.next(after)
        assert next_run is not None

    def test_empty_values_after_range_parse(self):
        """Test that range parsing produces correct values."""
        expr = CronExpression.parse("0 9-9 * * *")  # Single value range
        hour_field = expr.get_field(CronFieldType.HOUR)
        assert hour_field is not None
        assert hour_field.values == frozenset([9])

    def test_whitespace_handling(self):
        """Test handling of extra whitespace."""
        expr = CronExpression.parse("  0   9   *   *   *  ")
        assert expr.matches(datetime(2024, 1, 15, 9, 0))


# =============================================================================
# MatchContext Tests
# =============================================================================


class TestMatchContext:
    """Tests for MatchContext."""

    def test_match_context_creation(self):
        """Test creating match context."""
        ctx = MatchContext(year=2024, month=1, day=15)
        assert ctx.year == 2024
        assert ctx.month == 1
        assert ctx.day == 15


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_business_hours_schedule(self):
        """Test creating business hours schedule."""
        expr = (
            CronBuilder()
            .at_minute(0, 30)
            .at_hour(9, 10, 11, 12, 13, 14, 15, 16, 17)
            .on_weekdays()
            .build()
        )

        # Monday 9:30am
        assert expr.matches(datetime(2024, 1, 15, 9, 30))
        # Monday 5:00pm
        assert expr.matches(datetime(2024, 1, 15, 17, 0))
        # Sunday
        assert not expr.matches(datetime(2024, 1, 14, 9, 0))
        # 8am (too early)
        assert not expr.matches(datetime(2024, 1, 15, 8, 0))
        # 6pm (too late)
        assert not expr.matches(datetime(2024, 1, 15, 18, 0))

    def test_end_of_month_report(self):
        """Test end of month report schedule."""
        expr = CronExpression.parse("0 18 L * *")

        # Last day of January at 6pm
        assert expr.matches(datetime(2024, 1, 31, 18, 0))
        # Last day of February (leap year) at 6pm
        assert expr.matches(datetime(2024, 2, 29, 18, 0))
        # Last day of April at 6pm
        assert expr.matches(datetime(2024, 4, 30, 18, 0))

    def test_iterate_and_collect(self):
        """Test iterating and collecting results."""
        expr = CronExpression.parse("0 9 * * MON-FRI")
        after = datetime(2024, 1, 15, 0, 0)  # Monday

        runs = list(expr.iter(after, limit=5))

        assert len(runs) == 5
        # All should be weekdays at 9am
        for run in runs:
            assert run.hour == 9
            assert run.minute == 0
            assert run.weekday() < 5  # Mon-Fri

    def test_complex_validation_scenario(self):
        """Test complex validation scenario."""
        # Valid expressions
        assert is_valid_expression("0 9 * * MON-FRI")
        assert is_valid_expression("*/15 9-17 * * 1-5")
        assert is_valid_expression("0 0 1 1 *")
        assert is_valid_expression("30 6 L * *")

        # Invalid expressions
        assert not is_valid_expression("invalid cron")
        assert not is_valid_expression("* *")
        assert not is_valid_expression("60 * * * *")
        assert not is_valid_expression("* 24 * * *")
