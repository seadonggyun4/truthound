"""Scheduling module for Truthound.

This module provides a comprehensive cron expression parser and scheduler
with support for standard and extended cron syntax.

Features:
    - Standard 5-field cron (minute, hour, day, month, weekday)
    - Extended 6-field cron with seconds
    - Extended 7-field cron with years
    - Special characters: *, /, -, ,, L, W, #, ?
    - Named months and weekdays
    - Predefined expressions (@yearly, @monthly, @weekly, etc.)
    - Efficient next-run calculation
    - Expression builder and validator
    - Timezone support

Syntax Reference:
    Field         Values          Special Characters
    ─────────────────────────────────────────────────
    Second        0-59            * / , -
    Minute        0-59            * / , -
    Hour          0-23            * / , -
    Day of Month  1-31            * / , - L W
    Month         1-12 or JAN-DEC * / , -
    Day of Week   0-6 or SUN-SAT  * / , - L #
    Year          1970-2099       * / , -

Special Characters:
    *   Any value
    ,   List separator (1,3,5)
    -   Range (1-5)
    /   Step (*/15 = every 15)
    L   Last (L in day-of-month = last day)
    W   Nearest weekday (15W = nearest weekday to 15th)
    #   Nth weekday (2#3 = third Monday)
    ?   No specific value (day-of-month or day-of-week)

Usage:
    >>> from truthound.scheduling import CronExpression, CronParser
    >>>
    >>> # Parse and get next run times
    >>> expr = CronExpression.parse("0 9 * * MON-FRI")
    >>> next_run = expr.next()
    >>> next_5 = expr.next_n(5)
    >>>
    >>> # Use expression builder
    >>> expr = (CronExpression.builder()
    ...     .every_day()
    ...     .at_hour(9)
    ...     .at_minute(0)
    ...     .build())
    >>>
    >>> # Predefined expressions
    >>> daily = CronExpression.daily()
    >>> hourly = CronExpression.hourly()
"""

from truthound.scheduling.cron import (
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
)

from truthound.scheduling.presets import (
    # Predefined expressions
    YEARLY,
    ANNUALLY,
    MONTHLY,
    WEEKLY,
    DAILY,
    MIDNIGHT,
    HOURLY,
    EVERY_MINUTE,
    EVERY_SECOND,
    # Business schedule presets
    WEEKDAYS_9AM,
    WEEKDAYS_6PM,
    FIRST_OF_MONTH,
    LAST_OF_MONTH,
)

__all__ = [
    # Core
    "CronExpression",
    "CronField",
    "CronFieldType",
    # Parser
    "CronParser",
    "CronParseError",
    # Builder
    "CronBuilder",
    # Iterator
    "CronIterator",
    # Validation
    "validate_expression",
    "is_valid_expression",
    # Presets
    "YEARLY",
    "ANNUALLY",
    "MONTHLY",
    "WEEKLY",
    "DAILY",
    "MIDNIGHT",
    "HOURLY",
    "EVERY_MINUTE",
    "EVERY_SECOND",
    "WEEKDAYS_9AM",
    "WEEKDAYS_6PM",
    "FIRST_OF_MONTH",
    "LAST_OF_MONTH",
]
