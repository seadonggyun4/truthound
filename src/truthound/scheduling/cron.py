"""Optimized cron expression parser and scheduler.

This module provides a high-performance cron expression parser with support
for standard and extended syntax, efficient next-run calculation, and
comprehensive validation.

Design Principles:
    1. Immutable expressions: Thread-safe by design
    2. Lazy evaluation: Parse once, evaluate many times
    3. Efficient iteration: O(1) memory for next-run calculation
    4. Extensible: Easy to add new field types or syntax
"""

from __future__ import annotations

import calendar
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import cached_property
from typing import Any, Iterator, Set, FrozenSet


# =============================================================================
# Exceptions
# =============================================================================


class CronParseError(ValueError):
    """Raised when cron expression parsing fails."""

    def __init__(self, message: str, expression: str = "", position: int = -1) -> None:
        self.expression = expression
        self.position = position
        super().__init__(message)


# =============================================================================
# Field Types
# =============================================================================


class CronFieldType(Enum):
    """Types of cron fields."""

    SECOND = auto()
    MINUTE = auto()
    HOUR = auto()
    DAY_OF_MONTH = auto()
    MONTH = auto()
    DAY_OF_WEEK = auto()
    YEAR = auto()


@dataclass(frozen=True)
class FieldConstraints:
    """Constraints for a cron field."""

    min_value: int
    max_value: int
    names: dict[str, int] = field(default_factory=dict)
    supports_l: bool = False
    supports_w: bool = False
    supports_hash: bool = False
    supports_question: bool = False


# Field constraint definitions
FIELD_CONSTRAINTS: dict[CronFieldType, FieldConstraints] = {
    CronFieldType.SECOND: FieldConstraints(0, 59),
    CronFieldType.MINUTE: FieldConstraints(0, 59),
    CronFieldType.HOUR: FieldConstraints(0, 23),
    CronFieldType.DAY_OF_MONTH: FieldConstraints(
        1, 31,
        supports_l=True,
        supports_w=True,
        supports_question=True,
    ),
    CronFieldType.MONTH: FieldConstraints(
        1, 12,
        names={
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
            "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
            "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
        },
    ),
    CronFieldType.DAY_OF_WEEK: FieldConstraints(
        0, 6,
        names={
            "SUN": 0, "MON": 1, "TUE": 2, "WED": 3,
            "THU": 4, "FRI": 5, "SAT": 6,
        },
        supports_l=True,
        supports_hash=True,
        supports_question=True,
    ),
    CronFieldType.YEAR: FieldConstraints(1970, 2099),
}


# =============================================================================
# Cron Field
# =============================================================================


class CronField:
    """Represents a parsed cron field.

    A CronField contains the set of valid values for a particular field
    (minute, hour, etc.) along with any special modifiers (L, W, #).

    Attributes:
        field_type: The type of field (MINUTE, HOUR, etc.)
        values: Frozen set of valid integer values
        is_any: True if field matches any value (*)
        last: True if L modifier is present
        weekday_nearest: Day number if W modifier is present
        nth_weekday: Tuple of (weekday, nth) if # modifier is present
    """

    __slots__ = (
        "_field_type",
        "_values",
        "_is_any",
        "_last",
        "_weekday_nearest",
        "_nth_weekday",
        "_original",
    )

    def __init__(
        self,
        field_type: CronFieldType,
        values: FrozenSet[int],
        *,
        is_any: bool = False,
        last: bool = False,
        weekday_nearest: int | None = None,
        nth_weekday: tuple[int, int] | None = None,
        original: str = "",
    ) -> None:
        """Initialize cron field.

        Args:
            field_type: Type of this field.
            values: Set of valid values.
            is_any: Matches any value.
            last: L modifier present.
            weekday_nearest: W modifier value.
            nth_weekday: (weekday, nth) for # modifier.
            original: Original expression string.
        """
        self._field_type = field_type
        self._values = values
        self._is_any = is_any
        self._last = last
        self._weekday_nearest = weekday_nearest
        self._nth_weekday = nth_weekday
        self._original = original

    @property
    def field_type(self) -> CronFieldType:
        return self._field_type

    @property
    def values(self) -> FrozenSet[int]:
        return self._values

    @property
    def is_any(self) -> bool:
        return self._is_any

    @property
    def has_special(self) -> bool:
        """Check if field has special modifiers."""
        return self._last or self._weekday_nearest is not None or self._nth_weekday is not None

    def matches(self, value: int, context: "MatchContext | None" = None) -> bool:
        """Check if a value matches this field.

        Args:
            value: Value to check.
            context: Optional context for special modifiers.

        Returns:
            True if value matches.
        """
        if self._is_any:
            return True

        # Handle special modifiers
        if context:
            if self._last:
                if self._field_type == CronFieldType.DAY_OF_MONTH:
                    last_day = calendar.monthrange(context.year, context.month)[1]
                    return value == last_day
                elif self._field_type == CronFieldType.DAY_OF_WEEK:
                    # Last occurrence of weekday in month
                    return self._is_last_weekday(value, context)

            if self._weekday_nearest is not None:
                return self._is_nearest_weekday(value, context)

            if self._nth_weekday is not None:
                return self._is_nth_weekday(value, context)

        return value in self._values

    def _is_last_weekday(self, value: int, context: "MatchContext") -> bool:
        """Check if value is last occurrence of weekday in month."""
        if not self._values:
            return False

        target_weekday = next(iter(self._values))
        last_day = calendar.monthrange(context.year, context.month)[1]

        # Find last occurrence of target weekday
        for day in range(last_day, 0, -1):
            dt = datetime(context.year, context.month, day)
            if dt.weekday() == (target_weekday - 1) % 7:  # Convert to Python weekday
                return context.day == day

        return False

    def _is_nearest_weekday(self, value: int, context: "MatchContext") -> bool:
        """Check if value is nearest weekday to specified day."""
        target_day = self._weekday_nearest
        if target_day is None:
            return False

        dt = datetime(context.year, context.month, target_day)
        weekday = dt.weekday()

        if weekday == 5:  # Saturday -> Friday
            nearest = target_day - 1
        elif weekday == 6:  # Sunday -> Monday
            nearest = target_day + 1
        else:
            nearest = target_day

        # Handle month boundaries
        last_day = calendar.monthrange(context.year, context.month)[1]
        nearest = max(1, min(nearest, last_day))

        return context.day == nearest

    def _is_nth_weekday(self, value: int, context: "MatchContext") -> bool:
        """Check if value is nth occurrence of weekday in month."""
        if self._nth_weekday is None:
            return False

        target_weekday, nth = self._nth_weekday

        # Find nth occurrence
        count = 0
        for day in range(1, calendar.monthrange(context.year, context.month)[1] + 1):
            dt = datetime(context.year, context.month, day)
            if dt.weekday() == (target_weekday - 1) % 7:
                count += 1
                if count == nth:
                    return context.day == day

        return False

    def __repr__(self) -> str:
        return f"CronField({self._field_type.name}, {self._original!r})"


@dataclass
class MatchContext:
    """Context for matching special cron modifiers."""

    year: int
    month: int
    day: int


# =============================================================================
# Cron Parser
# =============================================================================


class CronParser:
    """Parser for cron expressions.

    Supports:
        - Standard 5-field cron (minute hour day month weekday)
        - Extended 6-field cron (second minute hour day month weekday)
        - Extended 7-field cron (second minute hour day month weekday year)
        - Special expressions (@yearly, @monthly, etc.)
    """

    # Predefined expression aliases
    ALIASES: dict[str, str] = {
        "@yearly": "0 0 1 1 *",
        "@annually": "0 0 1 1 *",
        "@monthly": "0 0 1 * *",
        "@weekly": "0 0 * * 0",
        "@daily": "0 0 * * *",
        "@midnight": "0 0 * * *",
        "@hourly": "0 * * * *",
        "@every_minute": "* * * * *",
        "@every_second": "* * * * * *",  # 6-field
    }

    def __init__(self, expression: str) -> None:
        """Initialize parser with expression.

        Args:
            expression: Cron expression string.
        """
        self._original = expression.strip()
        self._expression = self._resolve_alias(self._original)
        self._fields: list[CronField] = []

    def _resolve_alias(self, expression: str) -> str:
        """Resolve predefined aliases."""
        lower = expression.lower()
        if lower in self.ALIASES:
            return self.ALIASES[lower]
        return expression

    def parse(self) -> list[CronField]:
        """Parse the cron expression.

        Returns:
            List of CronField objects.

        Raises:
            CronParseError: If expression is invalid.
        """
        parts = self._expression.split()

        if len(parts) == 5:
            # Standard: minute hour day month weekday
            field_types = [
                CronFieldType.MINUTE,
                CronFieldType.HOUR,
                CronFieldType.DAY_OF_MONTH,
                CronFieldType.MONTH,
                CronFieldType.DAY_OF_WEEK,
            ]
        elif len(parts) == 6:
            # Extended: second minute hour day month weekday
            field_types = [
                CronFieldType.SECOND,
                CronFieldType.MINUTE,
                CronFieldType.HOUR,
                CronFieldType.DAY_OF_MONTH,
                CronFieldType.MONTH,
                CronFieldType.DAY_OF_WEEK,
            ]
        elif len(parts) == 7:
            # Extended with year: second minute hour day month weekday year
            field_types = [
                CronFieldType.SECOND,
                CronFieldType.MINUTE,
                CronFieldType.HOUR,
                CronFieldType.DAY_OF_MONTH,
                CronFieldType.MONTH,
                CronFieldType.DAY_OF_WEEK,
                CronFieldType.YEAR,
            ]
        else:
            raise CronParseError(
                f"Invalid number of fields: {len(parts)}. "
                "Expected 5, 6, or 7 fields.",
                self._original,
            )

        self._fields = [
            self._parse_field(part, field_type)
            for part, field_type in zip(parts, field_types)
        ]

        return self._fields

    def _parse_field(self, part: str, field_type: CronFieldType) -> CronField:
        """Parse a single cron field.

        Args:
            part: Field expression string.
            field_type: Type of this field.

        Returns:
            Parsed CronField.
        """
        constraints = FIELD_CONSTRAINTS[field_type]
        original = part

        # Handle ? (no specific value)
        if part == "?":
            if not constraints.supports_question:
                raise CronParseError(
                    f"? not supported for {field_type.name}",
                    self._original,
                )
            return CronField(field_type, frozenset(), is_any=True, original=original)

        # Handle * (any value)
        if part == "*":
            return CronField(field_type, frozenset(), is_any=True, original=original)

        # Handle L (last)
        if "L" in part.upper():
            return self._parse_last(part, field_type, constraints, original)

        # Handle W (weekday nearest) - only if it ends with W and is numeric+W
        if part.upper().endswith("W") and constraints.supports_w:
            # Check if it's actually a W modifier (number + W or LW)
            prefix = part.upper()[:-1]
            if prefix.isdigit() or prefix == "L":
                return self._parse_weekday(part, field_type, constraints, original)

        # Handle # (nth weekday)
        if "#" in part:
            return self._parse_nth(part, field_type, constraints, original)

        # Parse normal expression (ranges, lists, steps)
        values = self._parse_values(part, field_type, constraints)

        return CronField(field_type, frozenset(values), original=original)

    def _parse_last(
        self,
        part: str,
        field_type: CronFieldType,
        constraints: FieldConstraints,
        original: str,
    ) -> CronField:
        """Parse L (last) modifier."""
        if not constraints.supports_l:
            raise CronParseError(
                f"L not supported for {field_type.name}",
                self._original,
            )

        part_upper = part.upper()

        if part_upper == "L":
            # Just L - last day of month or last day of week
            return CronField(field_type, frozenset(), last=True, original=original)

        # Handle nL (e.g., 5L = last Friday)
        if part_upper.endswith("L"):
            weekday_str = part_upper[:-1]
            weekday = self._resolve_value(weekday_str, constraints)
            return CronField(
                field_type,
                frozenset([weekday]),
                last=True,
                original=original,
            )

        raise CronParseError(f"Invalid L expression: {part}", self._original)

    def _parse_weekday(
        self,
        part: str,
        field_type: CronFieldType,
        constraints: FieldConstraints,
        original: str,
    ) -> CronField:
        """Parse W (nearest weekday) modifier."""
        if not constraints.supports_w:
            raise CronParseError(
                f"W not supported for {field_type.name}",
                self._original,
            )

        part_upper = part.upper()

        if part_upper == "LW":
            # Last weekday of month
            return CronField(field_type, frozenset(), last=True, original=original)

        # Handle nW (e.g., 15W = nearest weekday to 15th)
        if part_upper.endswith("W"):
            day_str = part_upper[:-1]
            day = int(day_str)
            if day < constraints.min_value or day > constraints.max_value:
                raise CronParseError(
                    f"Day {day} out of range for W",
                    self._original,
                )
            return CronField(
                field_type,
                frozenset(),
                weekday_nearest=day,
                original=original,
            )

        raise CronParseError(f"Invalid W expression: {part}", self._original)

    def _parse_nth(
        self,
        part: str,
        field_type: CronFieldType,
        constraints: FieldConstraints,
        original: str,
    ) -> CronField:
        """Parse # (nth weekday) modifier."""
        if not constraints.supports_hash:
            raise CronParseError(
                f"# not supported for {field_type.name}",
                self._original,
            )

        parts = part.split("#")
        if len(parts) != 2:
            raise CronParseError(f"Invalid # expression: {part}", self._original)

        weekday = self._resolve_value(parts[0], constraints)
        nth = int(parts[1])

        if nth < 1 or nth > 5:
            raise CronParseError(f"Invalid nth value: {nth}", self._original)

        return CronField(
            field_type,
            frozenset([weekday]),
            nth_weekday=(weekday, nth),
            original=original,
        )

    def _parse_values(
        self,
        part: str,
        field_type: CronFieldType,
        constraints: FieldConstraints,
    ) -> set[int]:
        """Parse normal cron values (ranges, lists, steps)."""
        values: set[int] = set()

        # Handle comma-separated list
        for segment in part.split(","):
            segment = segment.strip()

            # Handle step (*/n or n-m/s)
            if "/" in segment:
                values.update(self._parse_step(segment, constraints))
            # Handle range (n-m)
            elif "-" in segment:
                values.update(self._parse_range(segment, constraints))
            # Handle single value
            else:
                value = self._resolve_value(segment, constraints)
                values.add(value)

        return values

    def _parse_step(
        self,
        segment: str,
        constraints: FieldConstraints,
    ) -> set[int]:
        """Parse step expression (*/n or n-m/s)."""
        parts = segment.split("/")
        if len(parts) != 2:
            raise CronParseError(f"Invalid step: {segment}", self._original)

        step = int(parts[1])
        if step <= 0:
            raise CronParseError(f"Step must be positive: {step}", self._original)

        base = parts[0]

        if base == "*":
            start = constraints.min_value
            end = constraints.max_value
        elif "-" in base:
            range_parts = base.split("-")
            start = self._resolve_value(range_parts[0], constraints)
            end = self._resolve_value(range_parts[1], constraints)
        else:
            start = self._resolve_value(base, constraints)
            end = constraints.max_value

        return set(range(start, end + 1, step))

    def _parse_range(
        self,
        segment: str,
        constraints: FieldConstraints,
    ) -> set[int]:
        """Parse range expression (n-m)."""
        parts = segment.split("-")
        if len(parts) != 2:
            raise CronParseError(f"Invalid range: {segment}", self._original)

        start = self._resolve_value(parts[0], constraints)
        end = self._resolve_value(parts[1], constraints)

        if start > end:
            # Handle wraparound (e.g., FRI-MON)
            values = set(range(start, constraints.max_value + 1))
            values.update(range(constraints.min_value, end + 1))
            return values

        return set(range(start, end + 1))

    def _resolve_value(self, value: str, constraints: FieldConstraints) -> int:
        """Resolve a value (number or name) to integer."""
        value = value.strip().upper()

        # Check named values
        if value in constraints.names:
            return constraints.names[value]

        # Parse as integer
        try:
            num = int(value)
            if num < constraints.min_value or num > constraints.max_value:
                raise CronParseError(
                    f"Value {num} out of range "
                    f"[{constraints.min_value}-{constraints.max_value}]",
                    self._original,
                )
            return num
        except ValueError:
            raise CronParseError(
                f"Invalid value: {value}",
                self._original,
            )


# =============================================================================
# Cron Expression
# =============================================================================


class CronExpression:
    """Parsed cron expression with efficient next-run calculation.

    CronExpression is immutable and thread-safe. It can be used to:
    - Check if a datetime matches the expression
    - Calculate the next matching datetime
    - Iterate over matching datetimes

    Example:
        >>> expr = CronExpression.parse("0 9 * * MON-FRI")
        >>> expr.matches(datetime(2024, 1, 15, 9, 0))  # True (Monday)
        >>> expr.next()  # Next matching datetime
        >>> list(expr.iter(limit=5))  # Next 5 matching datetimes
    """

    __slots__ = (
        "_expression",
        "_fields",
        "_has_seconds",
        "_has_years",
        "_field_map",
    )

    def __init__(self, expression: str, fields: list[CronField]) -> None:
        """Initialize cron expression.

        Args:
            expression: Original expression string.
            fields: Parsed cron fields.
        """
        self._expression = expression
        self._fields = tuple(fields)
        self._has_seconds = len(fields) >= 6
        self._has_years = len(fields) >= 7

        # Build field map for quick access
        self._field_map: dict[CronFieldType, CronField] = {
            f.field_type: f for f in fields
        }

    @classmethod
    def parse(cls, expression: str) -> "CronExpression":
        """Parse a cron expression.

        Args:
            expression: Cron expression string.

        Returns:
            Parsed CronExpression.

        Raises:
            CronParseError: If expression is invalid.
        """
        parser = CronParser(expression)
        fields = parser.parse()
        return cls(expression, fields)

    @classmethod
    def builder(cls) -> "CronBuilder":
        """Create a cron expression builder.

        Returns:
            CronBuilder for fluent construction.
        """
        return CronBuilder()

    # Predefined expression factories
    @classmethod
    def yearly(cls) -> "CronExpression":
        return cls.parse("0 0 1 1 *")

    @classmethod
    def monthly(cls) -> "CronExpression":
        return cls.parse("0 0 1 * *")

    @classmethod
    def weekly(cls) -> "CronExpression":
        return cls.parse("0 0 * * 0")

    @classmethod
    def daily(cls) -> "CronExpression":
        return cls.parse("0 0 * * *")

    @classmethod
    def hourly(cls) -> "CronExpression":
        return cls.parse("0 * * * *")

    @classmethod
    def every_n_minutes(cls, n: int) -> "CronExpression":
        return cls.parse(f"*/{n} * * * *")

    @classmethod
    def every_n_hours(cls, n: int) -> "CronExpression":
        return cls.parse(f"0 */{n} * * *")

    @property
    def expression(self) -> str:
        """Get original expression string."""
        return self._expression

    @property
    def fields(self) -> tuple[CronField, ...]:
        """Get parsed fields."""
        return self._fields

    @property
    def has_seconds(self) -> bool:
        """Check if expression includes seconds."""
        return self._has_seconds

    def get_field(self, field_type: CronFieldType) -> CronField | None:
        """Get a specific field by type."""
        return self._field_map.get(field_type)

    def matches(self, dt: datetime) -> bool:
        """Check if a datetime matches this expression.

        Args:
            dt: Datetime to check.

        Returns:
            True if datetime matches.
        """
        context = MatchContext(year=dt.year, month=dt.month, day=dt.day)

        # Check each field
        for field in self._fields:
            if field.field_type == CronFieldType.SECOND:
                if not field.matches(dt.second, context):
                    return False
            elif field.field_type == CronFieldType.MINUTE:
                if not field.matches(dt.minute, context):
                    return False
            elif field.field_type == CronFieldType.HOUR:
                if not field.matches(dt.hour, context):
                    return False
            elif field.field_type == CronFieldType.DAY_OF_MONTH:
                if not field.matches(dt.day, context):
                    return False
            elif field.field_type == CronFieldType.MONTH:
                if not field.matches(dt.month, context):
                    return False
            elif field.field_type == CronFieldType.DAY_OF_WEEK:
                # Python weekday: Monday=0, Sunday=6
                # Cron weekday: Sunday=0, Saturday=6
                cron_weekday = (dt.weekday() + 1) % 7
                if not field.matches(cron_weekday, context):
                    return False
            elif field.field_type == CronFieldType.YEAR:
                if not field.matches(dt.year, context):
                    return False

        return True

    def next(self, after: datetime | None = None) -> datetime | None:
        """Get next matching datetime.

        Args:
            after: Start searching after this datetime (default: now).

        Returns:
            Next matching datetime, or None if none found within 4 years.
        """
        if after is None:
            after = datetime.now()

        # Start from next second/minute
        if self._has_seconds:
            current = after.replace(microsecond=0) + timedelta(seconds=1)
        else:
            current = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

        # Search limit: 4 years
        end = current + timedelta(days=365 * 4)

        while current < end:
            if self.matches(current):
                return current

            # Advance to next candidate
            current = self._advance(current)

        return None

    def next_n(self, n: int, after: datetime | None = None) -> list[datetime]:
        """Get next n matching datetimes.

        Args:
            n: Number of matches to find.
            after: Start searching after this datetime.

        Returns:
            List of matching datetimes.
        """
        results = []
        current = after

        for _ in range(n):
            next_dt = self.next(current)
            if next_dt is None:
                break
            results.append(next_dt)
            current = next_dt

        return results

    def iter(
        self,
        after: datetime | None = None,
        limit: int | None = None,
    ) -> "CronIterator":
        """Create iterator over matching datetimes.

        Args:
            after: Start after this datetime.
            limit: Maximum number of matches.

        Returns:
            CronIterator.
        """
        return CronIterator(self, after, limit)

    def _advance(self, current: datetime) -> datetime:
        """Advance to next candidate datetime.

        Uses field constraints to skip non-matching times.
        """
        # Check month
        month_field = self._field_map.get(CronFieldType.MONTH)
        if month_field and not month_field.is_any:
            if current.month not in month_field.values:
                # Skip to next valid month
                for m in sorted(month_field.values):
                    if m > current.month:
                        return current.replace(month=m, day=1, hour=0, minute=0, second=0)
                # Wrap to next year
                return current.replace(
                    year=current.year + 1,
                    month=min(month_field.values),
                    day=1, hour=0, minute=0, second=0,
                )

        # Check day of month
        dom_field = self._field_map.get(CronFieldType.DAY_OF_MONTH)
        if dom_field and not dom_field.is_any and not dom_field.has_special:
            if current.day not in dom_field.values:
                # Skip to next valid day
                for d in sorted(dom_field.values):
                    if d > current.day:
                        try:
                            return current.replace(day=d, hour=0, minute=0, second=0)
                        except ValueError:
                            pass  # Day doesn't exist in this month
                # Wrap to next month
                if current.month == 12:
                    return current.replace(
                        year=current.year + 1,
                        month=1, day=1, hour=0, minute=0, second=0,
                    )
                return current.replace(
                    month=current.month + 1,
                    day=1, hour=0, minute=0, second=0,
                )

        # Default: advance by smallest unit
        if self._has_seconds:
            return current + timedelta(seconds=1)
        return current + timedelta(minutes=1)

    def __repr__(self) -> str:
        return f"CronExpression({self._expression!r})"

    def __str__(self) -> str:
        return self._expression

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CronExpression):
            return self._expression == other._expression
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._expression)


# =============================================================================
# Cron Iterator
# =============================================================================


class CronIterator(Iterator[datetime]):
    """Iterator over matching datetimes.

    Efficiently iterates without storing all matches in memory.
    """

    def __init__(
        self,
        expression: CronExpression,
        after: datetime | None = None,
        limit: int | None = None,
    ) -> None:
        """Initialize iterator.

        Args:
            expression: Cron expression.
            after: Start after this datetime.
            limit: Maximum matches.
        """
        self._expression = expression
        self._current = after
        self._limit = limit
        self._count = 0

    def __iter__(self) -> "CronIterator":
        return self

    def __next__(self) -> datetime:
        if self._limit is not None and self._count >= self._limit:
            raise StopIteration

        next_dt = self._expression.next(self._current)
        if next_dt is None:
            raise StopIteration

        self._current = next_dt
        self._count += 1

        return next_dt


# =============================================================================
# Cron Builder
# =============================================================================


class CronBuilder:
    """Fluent builder for cron expressions.

    Example:
        >>> expr = (CronBuilder()
        ...     .at_minute(0, 30)
        ...     .at_hour(9, 17)
        ...     .on_weekdays()
        ...     .build())
    """

    def __init__(self) -> None:
        """Initialize builder with defaults (every minute)."""
        self._second: str = "0"
        self._minute: str = "*"
        self._hour: str = "*"
        self._day_of_month: str = "*"
        self._month: str = "*"
        self._day_of_week: str = "*"
        self._include_seconds: bool = False

    def with_seconds(self) -> "CronBuilder":
        """Include seconds field in expression."""
        self._include_seconds = True
        return self

    def at_second(self, *seconds: int) -> "CronBuilder":
        """Set specific seconds."""
        self._include_seconds = True
        self._second = ",".join(str(s) for s in seconds)
        return self

    def every_n_seconds(self, n: int) -> "CronBuilder":
        """Run every n seconds."""
        self._include_seconds = True
        self._second = f"*/{n}"
        return self

    def at_minute(self, *minutes: int) -> "CronBuilder":
        """Set specific minutes."""
        self._minute = ",".join(str(m) for m in minutes)
        return self

    def every_n_minutes(self, n: int) -> "CronBuilder":
        """Run every n minutes."""
        self._minute = f"*/{n}"
        return self

    def at_hour(self, *hours: int) -> "CronBuilder":
        """Set specific hours."""
        self._hour = ",".join(str(h) for h in hours)
        return self

    def every_n_hours(self, n: int) -> "CronBuilder":
        """Run every n hours."""
        self._hour = f"*/{n}"
        return self

    def on_day(self, *days: int) -> "CronBuilder":
        """Set specific days of month."""
        self._day_of_month = ",".join(str(d) for d in days)
        return self

    def on_last_day(self) -> "CronBuilder":
        """Run on last day of month."""
        self._day_of_month = "L"
        return self

    def on_weekday_nearest(self, day: int) -> "CronBuilder":
        """Run on weekday nearest to specified day."""
        self._day_of_month = f"{day}W"
        return self

    def in_month(self, *months: int | str) -> "CronBuilder":
        """Set specific months."""
        self._month = ",".join(str(m).upper() for m in months)
        return self

    def on_weekday(self, *weekdays: int | str) -> "CronBuilder":
        """Set specific weekdays (0=SUN, 6=SAT)."""
        self._day_of_week = ",".join(str(w).upper() for w in weekdays)
        return self

    def on_weekdays(self) -> "CronBuilder":
        """Run Monday through Friday."""
        self._day_of_week = "MON-FRI"
        return self

    def on_weekends(self) -> "CronBuilder":
        """Run Saturday and Sunday."""
        self._day_of_week = "SAT,SUN"
        return self

    def every_day(self) -> "CronBuilder":
        """Run every day."""
        self._day_of_month = "*"
        self._day_of_week = "*"
        return self

    def daily_at(self, hour: int, minute: int = 0) -> "CronBuilder":
        """Run daily at specific time."""
        self._minute = str(minute)
        self._hour = str(hour)
        return self

    def hourly_at(self, minute: int) -> "CronBuilder":
        """Run hourly at specific minute."""
        self._minute = str(minute)
        return self

    def build(self) -> CronExpression:
        """Build the cron expression.

        Returns:
            Parsed CronExpression.
        """
        if self._include_seconds:
            expr = (
                f"{self._second} {self._minute} {self._hour} "
                f"{self._day_of_month} {self._month} {self._day_of_week}"
            )
        else:
            expr = (
                f"{self._minute} {self._hour} "
                f"{self._day_of_month} {self._month} {self._day_of_week}"
            )

        return CronExpression.parse(expr)


# =============================================================================
# Validation Functions
# =============================================================================


def validate_expression(expression: str) -> list[str]:
    """Validate a cron expression.

    Args:
        expression: Cron expression to validate.

    Returns:
        List of validation errors (empty if valid).
    """
    errors = []

    try:
        CronExpression.parse(expression)
    except CronParseError as e:
        errors.append(str(e))

    return errors


def is_valid_expression(expression: str) -> bool:
    """Check if a cron expression is valid.

    Args:
        expression: Cron expression to check.

    Returns:
        True if valid.
    """
    try:
        CronExpression.parse(expression)
        return True
    except CronParseError:
        return False
