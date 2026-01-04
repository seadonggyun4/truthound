"""Tests for IntervalTrigger with interval_seconds parameter."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from truthound.profiler.scheduling.triggers import IntervalTrigger


class TestIntervalTriggerSeconds:
    """Test IntervalTrigger interval_seconds parameter."""

    def test_interval_seconds_takes_precedence(self):
        """Test that interval_seconds overrides component parameters."""
        trigger = IntervalTrigger(
            days=10,
            hours=10,
            minutes=10,
            seconds=10,
            interval_seconds=3600,  # 1 hour - should override
        )

        assert trigger.interval == timedelta(hours=1)
        assert trigger.interval.total_seconds() == 3600

    def test_interval_seconds_only(self):
        """Test specifying only interval_seconds."""
        trigger = IntervalTrigger(interval_seconds=7200)  # 2 hours

        assert trigger.interval == timedelta(hours=2)
        assert trigger.interval.total_seconds() == 7200

    def test_interval_seconds_fractional(self):
        """Test fractional interval_seconds."""
        trigger = IntervalTrigger(interval_seconds=1.5)  # 1.5 seconds

        assert trigger.interval == timedelta(seconds=1.5)
        assert trigger.interval.total_seconds() == 1.5

    def test_component_parameters_when_no_interval_seconds(self):
        """Test component parameters work when interval_seconds is None."""
        trigger = IntervalTrigger(
            days=1,
            hours=6,
            minutes=30,
        )

        expected = timedelta(days=1, hours=6, minutes=30)
        assert trigger.interval == expected

    def test_should_run_with_interval_seconds(self):
        """Test should_run works with interval_seconds."""
        trigger = IntervalTrigger(interval_seconds=60)  # 1 minute

        now = datetime.now()
        last_run = now - timedelta(seconds=30)  # 30 seconds ago

        # Should not run yet (only 30 seconds elapsed)
        assert trigger.should_run(last_run, {}) is False

        # Should run (more than 60 seconds elapsed)
        last_run = now - timedelta(seconds=90)
        assert trigger.should_run(last_run, {}) is True

    def test_get_next_run_time_with_interval_seconds(self):
        """Test get_next_run_time works with interval_seconds."""
        trigger = IntervalTrigger(interval_seconds=3600)  # 1 hour

        now = datetime.now()
        next_run = trigger.get_next_run_time(now)

        expected = now + timedelta(hours=1)
        assert next_run == expected

    def test_zero_interval_seconds(self):
        """Test zero interval_seconds."""
        trigger = IntervalTrigger(interval_seconds=0)

        assert trigger.interval == timedelta(seconds=0)

    def test_large_interval_seconds(self):
        """Test large interval_seconds (1 week)."""
        trigger = IntervalTrigger(interval_seconds=604800)  # 7 days

        assert trigger.interval == timedelta(weeks=1)
        assert trigger.interval.total_seconds() == 604800


class TestIntervalTriggerBackwardCompatibility:
    """Test backward compatibility with existing usage."""

    def test_days_only(self):
        """Test existing days-only usage still works."""
        trigger = IntervalTrigger(days=1)

        assert trigger.interval == timedelta(days=1)

    def test_hours_only(self):
        """Test existing hours-only usage still works."""
        trigger = IntervalTrigger(hours=6)

        assert trigger.interval == timedelta(hours=6)

    def test_minutes_only(self):
        """Test existing minutes-only usage still works."""
        trigger = IntervalTrigger(minutes=30)

        assert trigger.interval == timedelta(minutes=30)

    def test_mixed_components(self):
        """Test existing mixed component usage still works."""
        trigger = IntervalTrigger(days=1, hours=12)

        expected = timedelta(days=1, hours=12)
        assert trigger.interval == expected

    def test_default_no_interval(self):
        """Test default trigger with no interval."""
        trigger = IntervalTrigger()

        assert trigger.interval == timedelta()
        assert trigger.interval.total_seconds() == 0


class TestIntervalTriggerConversions:
    """Test common interval conversions."""

    def test_one_hour(self):
        """Test 1 hour interval in multiple ways."""
        trigger_seconds = IntervalTrigger(interval_seconds=3600)
        trigger_hours = IntervalTrigger(hours=1)
        trigger_minutes = IntervalTrigger(minutes=60)

        assert trigger_seconds.interval == trigger_hours.interval
        assert trigger_hours.interval == trigger_minutes.interval
        assert trigger_seconds.interval.total_seconds() == 3600

    def test_one_day(self):
        """Test 1 day interval in multiple ways."""
        trigger_seconds = IntervalTrigger(interval_seconds=86400)
        trigger_days = IntervalTrigger(days=1)
        trigger_hours = IntervalTrigger(hours=24)

        assert trigger_seconds.interval == trigger_days.interval
        assert trigger_days.interval == trigger_hours.interval
        assert trigger_seconds.interval.total_seconds() == 86400

    def test_thirty_minutes(self):
        """Test 30 minute interval."""
        trigger_seconds = IntervalTrigger(interval_seconds=1800)
        trigger_minutes = IntervalTrigger(minutes=30)

        assert trigger_seconds.interval == trigger_minutes.interval
        assert trigger_seconds.interval.total_seconds() == 1800
