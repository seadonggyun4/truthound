"""Built-in routing rules for common use cases.

This module provides ready-to-use routing rules:
- SeverityRule: Route based on issue severity levels
- IssueCountRule: Route based on number of issues
- StatusRule: Route based on checkpoint status
- TagRule: Route based on tag presence/values
- DataAssetRule: Route based on data asset name patterns
- MetadataRule: Route based on metadata values
- TimeWindowRule: Route based on time of day/week
- AlwaysRule / NeverRule: Constant rules for testing

Example:
    >>> from truthound.checkpoint.routing.rules import SeverityRule, IssueCountRule
    >>> from truthound.checkpoint.routing import AllOf
    >>>
    >>> # Route when there are critical issues in production
    >>> rule = AllOf([
    ...     SeverityRule(min_severity="critical"),
    ...     TagRule(tags={"env": "prod"}),
    ... ])
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from truthound.checkpoint.routing.base import RouteContext


# Severity order for comparison
SEVERITY_ORDER: dict[str, int] = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
}


@dataclass
class AlwaysRule:
    """A rule that always matches.

    Useful for default routes or testing.

    Example:
        >>> rule = AlwaysRule()
        >>> rule.evaluate(any_context)
        True
    """

    def evaluate(self, context: "RouteContext") -> bool:
        """Always returns True."""
        return True

    @property
    def description(self) -> str:
        """Get rule description."""
        return "Always matches"


@dataclass
class NeverRule:
    """A rule that never matches.

    Useful for disabling routes without removing them.

    Example:
        >>> rule = NeverRule()
        >>> rule.evaluate(any_context)
        False
    """

    def evaluate(self, context: "RouteContext") -> bool:
        """Always returns False."""
        return False

    @property
    def description(self) -> str:
        """Get rule description."""
        return "Never matches"


@dataclass
class SeverityRule:
    """Route based on issue severity levels.

    Matches when issues of the specified severity or higher exist.

    Attributes:
        min_severity: Minimum severity to match (critical, high, medium, low, info)
        max_severity: Maximum severity to match (optional)
        exact_count: Match only when exactly this many issues of the severity exist
        min_count: Minimum number of issues at the severity level

    Example:
        >>> # Match when there are critical issues
        >>> rule = SeverityRule(min_severity="critical")
        >>>
        >>> # Match when there are at least 5 high or critical issues
        >>> rule = SeverityRule(min_severity="high", min_count=5)
        >>>
        >>> # Match for exactly medium severity issues
        >>> rule = SeverityRule(min_severity="medium", max_severity="medium")
    """

    min_severity: str = "high"
    max_severity: str | None = None
    exact_count: int | None = None
    min_count: int = 1

    def __post_init__(self) -> None:
        """Validate severity values."""
        self.min_severity = self.min_severity.lower()
        if self.max_severity:
            self.max_severity = self.max_severity.lower()

        if self.min_severity not in SEVERITY_ORDER:
            raise ValueError(
                f"Invalid min_severity: {self.min_severity}. "
                f"Must be one of: {list(SEVERITY_ORDER.keys())}"
            )

        if self.max_severity and self.max_severity not in SEVERITY_ORDER:
            raise ValueError(
                f"Invalid max_severity: {self.max_severity}. "
                f"Must be one of: {list(SEVERITY_ORDER.keys())}"
            )

    def _count_issues_in_range(self, context: "RouteContext") -> int:
        """Count issues within the severity range.

        min_severity specifies the minimum severity level.
        Higher severity = lower order number (critical=0 is highest).
        So we count severities where order <= min_order (more severe or equal).
        If max_severity is specified, we only count within that range.
        """
        min_order = SEVERITY_ORDER[self.min_severity]
        # If no max_severity, default to same as min (only that severity level)
        # To get "min_severity or higher", set max_severity=None means only min_severity
        max_order = (
            SEVERITY_ORDER[self.max_severity]
            if self.max_severity
            else min_order
        )

        # Ensure min_order <= max_order (critical=0 < high=1 < medium=2)
        # If user specifies min=high, max=critical, swap them
        if min_order > max_order:
            min_order, max_order = max_order, min_order

        count = 0
        severity_counts = {
            "critical": context.critical_issues,
            "high": context.high_issues,
            "medium": context.medium_issues,
            "low": context.low_issues,
            "info": context.info_issues,
        }

        for severity, order in SEVERITY_ORDER.items():
            if min_order <= order <= max_order:
                count += severity_counts[severity]

        return count

    def evaluate(self, context: "RouteContext") -> bool:
        """Evaluate if issues match the severity criteria.

        Args:
            context: The routing context.

        Returns:
            True if issues match the criteria.
        """
        count = self._count_issues_in_range(context)

        if self.exact_count is not None:
            return count == self.exact_count

        return count >= self.min_count

    @property
    def description(self) -> str:
        """Get rule description."""
        parts = [f"severity >= {self.min_severity}"]

        if self.max_severity:
            parts[0] = f"{self.min_severity} <= severity <= {self.max_severity}"

        if self.exact_count is not None:
            parts.append(f"exactly {self.exact_count} issues")
        elif self.min_count > 1:
            parts.append(f"at least {self.min_count} issues")

        return ", ".join(parts)


@dataclass
class IssueCountRule:
    """Route based on the number of issues.

    Matches when the total issue count falls within the specified range.

    Attributes:
        min_issues: Minimum number of issues (inclusive)
        max_issues: Maximum number of issues (inclusive, optional)
        count_type: What to count (total, critical, high, medium, low, info)

    Example:
        >>> # Match when there are at least 10 issues
        >>> rule = IssueCountRule(min_issues=10)
        >>>
        >>> # Match when there are between 5 and 20 issues
        >>> rule = IssueCountRule(min_issues=5, max_issues=20)
        >>>
        >>> # Match when there are at least 3 critical issues
        >>> rule = IssueCountRule(min_issues=3, count_type="critical")
    """

    min_issues: int = 0
    max_issues: int | None = None
    count_type: Literal[
        "total", "critical", "high", "medium", "low", "info"
    ] = "total"

    def _get_count(self, context: "RouteContext") -> int:
        """Get the issue count based on count_type."""
        counts = {
            "total": context.total_issues,
            "critical": context.critical_issues,
            "high": context.high_issues,
            "medium": context.medium_issues,
            "low": context.low_issues,
            "info": context.info_issues,
        }
        return counts[self.count_type]

    def evaluate(self, context: "RouteContext") -> bool:
        """Evaluate if issue count matches the criteria.

        Args:
            context: The routing context.

        Returns:
            True if issue count is within range.
        """
        count = self._get_count(context)

        if count < self.min_issues:
            return False

        if self.max_issues is not None and count > self.max_issues:
            return False

        return True

    @property
    def description(self) -> str:
        """Get rule description."""
        type_desc = f"{self.count_type} issues" if self.count_type != "total" else "issues"

        if self.max_issues is None:
            return f"{type_desc} >= {self.min_issues}"

        return f"{self.min_issues} <= {type_desc} <= {self.max_issues}"


@dataclass
class StatusRule:
    """Route based on checkpoint status.

    Matches when the checkpoint status is one of the specified values.

    Attributes:
        statuses: List of statuses to match
        negate: If True, match when status is NOT in the list

    Example:
        >>> # Match on failure or error
        >>> rule = StatusRule(statuses=["failure", "error"])
        >>>
        >>> # Match on anything except success
        >>> rule = StatusRule(statuses=["success"], negate=True)
    """

    statuses: list[str] = field(default_factory=lambda: ["failure", "error"])
    negate: bool = False

    def __post_init__(self) -> None:
        """Normalize status values."""
        self.statuses = [s.lower() for s in self.statuses]

    def evaluate(self, context: "RouteContext") -> bool:
        """Evaluate if status matches.

        Args:
            context: The routing context.

        Returns:
            True if status matches (or doesn't match if negate=True).
        """
        status = context.status.lower()
        matches = status in self.statuses

        return not matches if self.negate else matches

    @property
    def description(self) -> str:
        """Get rule description."""
        if self.negate:
            return f"status not in {self.statuses}"
        return f"status in {self.statuses}"


@dataclass
class TagRule:
    """Route based on tag presence or values.

    Matches when specified tags exist and optionally match values.

    Attributes:
        tags: Dictionary of tag name to expected value (None = just check existence)
        match_all: If True, all tags must match; if False, any tag matches
        negate: If True, match when tags DON'T match

    Example:
        >>> # Match when env=prod
        >>> rule = TagRule(tags={"env": "prod"})
        >>>
        >>> # Match when either env=prod OR team=data
        >>> rule = TagRule(tags={"env": "prod", "team": "data"}, match_all=False)
        >>>
        >>> # Match when 'critical' tag exists (any value)
        >>> rule = TagRule(tags={"critical": None})
    """

    tags: dict[str, str | None] = field(default_factory=dict)
    match_all: bool = True
    negate: bool = False

    def _tag_matches(
        self, context_tags: dict[str, str], tag: str, value: str | None
    ) -> bool:
        """Check if a single tag matches."""
        if tag not in context_tags:
            return False

        if value is None:
            return True

        return context_tags[tag] == value

    def evaluate(self, context: "RouteContext") -> bool:
        """Evaluate if tags match.

        Args:
            context: The routing context.

        Returns:
            True if tags match the criteria.
        """
        if not self.tags:
            return True

        matches = [
            self._tag_matches(context.tags, tag, value)
            for tag, value in self.tags.items()
        ]

        if self.match_all:
            result = all(matches)
        else:
            result = any(matches)

        return not result if self.negate else result

    @property
    def description(self) -> str:
        """Get rule description."""
        op = "all" if self.match_all else "any"
        neg = "NOT " if self.negate else ""
        return f"{neg}{op} tags match: {self.tags}"


@dataclass
class DataAssetRule:
    """Route based on data asset name patterns.

    Matches when the data asset name matches a pattern.

    Attributes:
        pattern: Regex pattern or glob-like pattern to match
        is_regex: If True, treat pattern as regex; if False, use glob-style
        case_sensitive: Whether matching is case-sensitive

    Example:
        >>> # Match any asset starting with "sales_"
        >>> rule = DataAssetRule(pattern="sales_*")
        >>>
        >>> # Match using regex
        >>> rule = DataAssetRule(pattern=r"^prod_.*_v\d+$", is_regex=True)
        >>>
        >>> # Match case-insensitive
        >>> rule = DataAssetRule(pattern="USERS*", case_sensitive=False)
    """

    pattern: str = "*"
    is_regex: bool = False
    case_sensitive: bool = True
    _compiled: re.Pattern[str] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Compile the pattern."""
        if self.is_regex:
            regex_pattern = self.pattern
        else:
            # Convert glob to regex
            regex_pattern = self._glob_to_regex(self.pattern)

        flags = 0 if self.case_sensitive else re.IGNORECASE
        self._compiled = re.compile(regex_pattern, flags)

    def _glob_to_regex(self, glob: str) -> str:
        """Convert glob pattern to regex."""
        # Escape special regex characters except * and ?
        result = ""
        for char in glob:
            if char == "*":
                result += ".*"
            elif char == "?":
                result += "."
            elif char in r"\[]{}()^$.|+":
                result += "\\" + char
            else:
                result += char

        return f"^{result}$"

    def evaluate(self, context: "RouteContext") -> bool:
        """Evaluate if data asset name matches.

        Args:
            context: The routing context.

        Returns:
            True if data asset matches the pattern.
        """
        if self._compiled is None:
            return False

        return bool(self._compiled.match(context.data_asset))

    @property
    def description(self) -> str:
        """Get rule description."""
        pattern_type = "regex" if self.is_regex else "glob"
        return f"data_asset matches {pattern_type}: {self.pattern}"


@dataclass
class MetadataRule:
    """Route based on metadata values.

    Matches when metadata contains specified key-value pairs.

    Attributes:
        key_path: Dot-separated path to the metadata value (e.g., "config.env")
        expected_value: Expected value (None = just check existence)
        comparator: Comparison operator (eq, ne, gt, gte, lt, lte, contains, regex)

    Example:
        >>> # Match when metadata.region == "us-east-1"
        >>> rule = MetadataRule(key_path="region", expected_value="us-east-1")
        >>>
        >>> # Match when metadata.priority > 5
        >>> rule = MetadataRule(key_path="priority", expected_value=5, comparator="gt")
        >>>
        >>> # Match when metadata.owners contains "data-team"
        >>> rule = MetadataRule(key_path="owners", expected_value="data-team", comparator="contains")
    """

    key_path: str
    expected_value: Any = None
    comparator: Literal[
        "eq", "ne", "gt", "gte", "lt", "lte", "contains", "regex", "exists"
    ] = "eq"

    def _get_value(self, metadata: dict[str, Any]) -> tuple[bool, Any]:
        """Get value from metadata using dot-separated path.

        Returns:
            Tuple of (found, value).
        """
        parts = self.key_path.split(".")
        current = metadata

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False, None

        return True, current

    def _compare(self, actual: Any, expected: Any) -> bool:
        """Compare values using the specified comparator."""
        if self.comparator == "eq":
            return actual == expected
        elif self.comparator == "ne":
            return actual != expected
        elif self.comparator == "gt":
            return actual > expected
        elif self.comparator == "gte":
            return actual >= expected
        elif self.comparator == "lt":
            return actual < expected
        elif self.comparator == "lte":
            return actual <= expected
        elif self.comparator == "contains":
            if isinstance(actual, str):
                return expected in actual
            elif isinstance(actual, (list, tuple, set)):
                return expected in actual
            elif isinstance(actual, dict):
                return expected in actual
            return False
        elif self.comparator == "regex":
            if isinstance(actual, str) and isinstance(expected, str):
                return bool(re.search(expected, actual))
            return False
        elif self.comparator == "exists":
            return True  # If we got here, the key exists

        return False

    def evaluate(self, context: "RouteContext") -> bool:
        """Evaluate if metadata matches.

        Args:
            context: The routing context.

        Returns:
            True if metadata matches the criteria.
        """
        found, actual = self._get_value(context.metadata)

        if not found:
            return False

        if self.comparator == "exists":
            return True

        return self._compare(actual, self.expected_value)

    @property
    def description(self) -> str:
        """Get rule description."""
        if self.comparator == "exists":
            return f"metadata.{self.key_path} exists"

        op_map = {
            "eq": "==",
            "ne": "!=",
            "gt": ">",
            "gte": ">=",
            "lt": "<",
            "lte": "<=",
            "contains": "contains",
            "regex": "~=",
        }
        op = op_map.get(self.comparator, self.comparator)
        return f"metadata.{self.key_path} {op} {self.expected_value!r}"


@dataclass
class TimeWindowRule:
    """Route based on time of day or day of week.

    Matches when the current time falls within the specified window.
    Useful for routing to different channels during/outside business hours.

    Attributes:
        start_time: Start of time window (HH:MM format)
        end_time: End of time window (HH:MM format)
        days_of_week: List of days (0=Monday, 6=Sunday), None = all days
        timezone: Timezone name (e.g., "UTC", "America/New_York")
        use_run_time: If True, use checkpoint run_time instead of current time

    Example:
        >>> # Match during business hours (9 AM - 5 PM, Mon-Fri)
        >>> rule = TimeWindowRule(
        ...     start_time="09:00",
        ...     end_time="17:00",
        ...     days_of_week=[0, 1, 2, 3, 4],  # Mon-Fri
        ... )
        >>>
        >>> # Match outside business hours
        >>> rule = TimeWindowRule(
        ...     start_time="17:00",
        ...     end_time="09:00",  # Wraps around midnight
        ... )
    """

    start_time: str = "00:00"
    end_time: str = "23:59"
    days_of_week: list[int] | None = None
    timezone: str = "UTC"
    use_run_time: bool = True
    _start: time = field(default=None, repr=False)  # type: ignore
    _end: time = field(default=None, repr=False)  # type: ignore

    def __post_init__(self) -> None:
        """Parse time strings."""
        self._start = self._parse_time(self.start_time)
        self._end = self._parse_time(self.end_time)

    def _parse_time(self, time_str: str) -> time:
        """Parse time string to time object."""
        parts = time_str.split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        second = int(parts[2]) if len(parts) > 2 else 0
        return time(hour, minute, second)

    def _get_current_time(self, context: "RouteContext") -> datetime:
        """Get the current time, optionally with timezone conversion."""
        if self.use_run_time:
            dt = context.run_time
        else:
            dt = datetime.now()

        # Convert to specified timezone if pytz is available
        try:
            import pytz

            tz = pytz.timezone(self.timezone)
            if dt.tzinfo is None:
                dt = pytz.UTC.localize(dt)
            dt = dt.astimezone(tz)
        except ImportError:
            pass  # Use naive datetime

        return dt

    def _time_in_window(self, check_time: time) -> bool:
        """Check if time is within the window.

        Handles wrapping around midnight.
        """
        if self._start <= self._end:
            return self._start <= check_time <= self._end
        else:
            # Window wraps around midnight
            return check_time >= self._start or check_time <= self._end

    def evaluate(self, context: "RouteContext") -> bool:
        """Evaluate if current time is within the window.

        Args:
            context: The routing context.

        Returns:
            True if time is within the window.
        """
        dt = self._get_current_time(context)

        # Check day of week
        if self.days_of_week is not None:
            if dt.weekday() not in self.days_of_week:
                return False

        # Check time
        return self._time_in_window(dt.time())

    @property
    def description(self) -> str:
        """Get rule description."""
        parts = [f"{self.start_time}-{self.end_time}"]

        if self.days_of_week:
            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            days = [day_names[d] for d in self.days_of_week]
            parts.append(f"({', '.join(days)})")

        if self.timezone != "UTC":
            parts.append(f"[{self.timezone}]")

        return f"time window: {' '.join(parts)}"


@dataclass
class PassRateRule:
    """Route based on validation pass rate.

    Matches when the pass rate falls within the specified range.

    Attributes:
        min_rate: Minimum pass rate (0-100)
        max_rate: Maximum pass rate (0-100)

    Example:
        >>> # Match when pass rate is below 90%
        >>> rule = PassRateRule(max_rate=90.0)
        >>>
        >>> # Match when pass rate is between 50% and 80%
        >>> rule = PassRateRule(min_rate=50.0, max_rate=80.0)
    """

    min_rate: float = 0.0
    max_rate: float = 100.0

    def evaluate(self, context: "RouteContext") -> bool:
        """Evaluate if pass rate is within range.

        Args:
            context: The routing context.

        Returns:
            True if pass rate is within range.
        """
        return self.min_rate <= context.pass_rate <= self.max_rate

    @property
    def description(self) -> str:
        """Get rule description."""
        if self.min_rate == 0.0:
            return f"pass_rate <= {self.max_rate}%"
        if self.max_rate == 100.0:
            return f"pass_rate >= {self.min_rate}%"
        return f"{self.min_rate}% <= pass_rate <= {self.max_rate}%"


@dataclass
class ErrorRule:
    """Route based on error presence or pattern.

    Matches when an error occurred and optionally matches a pattern.

    Attributes:
        pattern: Regex pattern to match error message (None = any error)
        negate: If True, match when there is NO error

    Example:
        >>> # Match on any error
        >>> rule = ErrorRule()
        >>>
        >>> # Match on timeout errors
        >>> rule = ErrorRule(pattern=r"timeout|timed out")
        >>>
        >>> # Match when there is no error
        >>> rule = ErrorRule(negate=True)
    """

    pattern: str | None = None
    negate: bool = False
    _compiled: re.Pattern[str] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Compile pattern if provided."""
        if self.pattern:
            self._compiled = re.compile(self.pattern, re.IGNORECASE)

    def evaluate(self, context: "RouteContext") -> bool:
        """Evaluate if error matches.

        Args:
            context: The routing context.

        Returns:
            True if error matches criteria.
        """
        has_error = context.error is not None

        if self.negate:
            return not has_error

        if not has_error:
            return False

        if self._compiled is None:
            return True

        return bool(self._compiled.search(context.error))

    @property
    def description(self) -> str:
        """Get rule description."""
        if self.negate:
            return "no error"
        if self.pattern:
            return f"error matches: {self.pattern}"
        return "has error"
