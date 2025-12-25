"""Predefined cron expression presets.

This module provides commonly used cron expressions as constants
for easy reuse and readability.

Usage:
    >>> from truthound.scheduling.presets import DAILY, WEEKDAYS_9AM
    >>>
    >>> # Use directly
    >>> next_run = DAILY.next()
    >>>
    >>> # Or in checkpoint configuration
    >>> checkpoint = Checkpoint(
    ...     name="daily_validation",
    ...     schedule=DAILY,
    ... )
"""

from truthound.scheduling.cron import CronExpression


# =============================================================================
# Standard Intervals
# =============================================================================

# Every year on January 1st at midnight
YEARLY = CronExpression.parse("0 0 1 1 *")
ANNUALLY = YEARLY

# First day of every month at midnight
MONTHLY = CronExpression.parse("0 0 1 * *")

# Every Sunday at midnight
WEEKLY = CronExpression.parse("0 0 * * 0")

# Every day at midnight
DAILY = CronExpression.parse("0 0 * * *")
MIDNIGHT = DAILY

# Every hour at minute 0
HOURLY = CronExpression.parse("0 * * * *")

# Every minute
EVERY_MINUTE = CronExpression.parse("* * * * *")

# Every second (6-field cron)
EVERY_SECOND = CronExpression.parse("* * * * * *")


# =============================================================================
# Business Schedule Presets
# =============================================================================

# Weekdays (Monday-Friday) at 9 AM
WEEKDAYS_9AM = CronExpression.parse("0 9 * * 1-5")

# Weekdays (Monday-Friday) at 6 PM
WEEKDAYS_6PM = CronExpression.parse("0 18 * * 1-5")

# Weekdays at start of business (8 AM)
BUSINESS_START = CronExpression.parse("0 8 * * 1-5")

# Weekdays at end of business (5 PM)
BUSINESS_END = CronExpression.parse("0 17 * * 1-5")

# Every 15 minutes during business hours (9 AM - 5 PM, weekdays)
BUSINESS_HOURS_15MIN = CronExpression.parse("*/15 9-17 * * 1-5")

# Every hour during business hours
BUSINESS_HOURS_HOURLY = CronExpression.parse("0 9-17 * * 1-5")


# =============================================================================
# Month Boundary Presets
# =============================================================================

# First day of month at 6 AM
FIRST_OF_MONTH = CronExpression.parse("0 6 1 * *")

# Last day of month at 6 AM
LAST_OF_MONTH = CronExpression.parse("0 6 L * *")

# First Monday of month
FIRST_MONDAY = CronExpression.parse("0 9 * * 1#1")

# Last Friday of month
LAST_FRIDAY = CronExpression.parse("0 17 * * 5L")


# =============================================================================
# Data Pipeline Presets
# =============================================================================

# Every 5 minutes (common for near-real-time)
EVERY_5_MIN = CronExpression.parse("*/5 * * * *")

# Every 15 minutes
EVERY_15_MIN = CronExpression.parse("*/15 * * * *")

# Every 30 minutes
EVERY_30_MIN = CronExpression.parse("*/30 * * * *")

# Every 2 hours
EVERY_2_HOURS = CronExpression.parse("0 */2 * * *")

# Every 4 hours
EVERY_4_HOURS = CronExpression.parse("0 */4 * * *")

# Every 6 hours
EVERY_6_HOURS = CronExpression.parse("0 */6 * * *")

# Every 12 hours (twice daily)
TWICE_DAILY = CronExpression.parse("0 0,12 * * *")

# Three times daily (morning, noon, evening)
THREE_TIMES_DAILY = CronExpression.parse("0 8,12,18 * * *")


# =============================================================================
# Weekend/Off-hours Presets
# =============================================================================

# Weekends only at noon
WEEKENDS_NOON = CronExpression.parse("0 12 * * 0,6")

# Every night at 2 AM (common for batch jobs)
NIGHTLY_2AM = CronExpression.parse("0 2 * * *")

# Every night at 3 AM
NIGHTLY_3AM = CronExpression.parse("0 3 * * *")

# Sunday at 3 AM (weekly maintenance window)
SUNDAY_MAINTENANCE = CronExpression.parse("0 3 * * 0")


# =============================================================================
# Quarter Presets
# =============================================================================

# First day of each quarter
QUARTERLY = CronExpression.parse("0 0 1 1,4,7,10 *")

# Last day of each quarter
END_OF_QUARTER = CronExpression.parse("0 0 L 3,6,9,12 *")


# =============================================================================
# Preset Registry
# =============================================================================

PRESETS: dict[str, CronExpression] = {
    # Standard
    "yearly": YEARLY,
    "annually": ANNUALLY,
    "monthly": MONTHLY,
    "weekly": WEEKLY,
    "daily": DAILY,
    "midnight": MIDNIGHT,
    "hourly": HOURLY,
    "every_minute": EVERY_MINUTE,
    "every_second": EVERY_SECOND,
    # Business
    "weekdays_9am": WEEKDAYS_9AM,
    "weekdays_6pm": WEEKDAYS_6PM,
    "business_start": BUSINESS_START,
    "business_end": BUSINESS_END,
    "business_hours_15min": BUSINESS_HOURS_15MIN,
    "business_hours_hourly": BUSINESS_HOURS_HOURLY,
    # Month boundaries
    "first_of_month": FIRST_OF_MONTH,
    "last_of_month": LAST_OF_MONTH,
    "first_monday": FIRST_MONDAY,
    "last_friday": LAST_FRIDAY,
    # Data pipeline
    "every_5_min": EVERY_5_MIN,
    "every_15_min": EVERY_15_MIN,
    "every_30_min": EVERY_30_MIN,
    "every_2_hours": EVERY_2_HOURS,
    "every_4_hours": EVERY_4_HOURS,
    "every_6_hours": EVERY_6_HOURS,
    "twice_daily": TWICE_DAILY,
    "three_times_daily": THREE_TIMES_DAILY,
    # Off-hours
    "weekends_noon": WEEKENDS_NOON,
    "nightly_2am": NIGHTLY_2AM,
    "nightly_3am": NIGHTLY_3AM,
    "sunday_maintenance": SUNDAY_MAINTENANCE,
    # Quarter
    "quarterly": QUARTERLY,
    "end_of_quarter": END_OF_QUARTER,
}


def get_preset(name: str) -> CronExpression | None:
    """Get a preset cron expression by name.

    Args:
        name: Preset name (case-insensitive).

    Returns:
        CronExpression or None if not found.
    """
    return PRESETS.get(name.lower().replace("-", "_"))


def list_presets() -> list[str]:
    """List all available preset names.

    Returns:
        List of preset names.
    """
    return list(PRESETS.keys())
