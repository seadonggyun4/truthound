"""Report generation for validation results."""

import heapq
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator

from rich.console import Console
from rich.table import Table

from truthound.types import ResultFormat, Severity
from truthound.validators.base import ExceptionInfo, ValidationIssue

# Severity ordering for heap (lower value = higher priority)
_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
}


@dataclass
class ExceptionSummary:
    """Aggregate summary of exceptions encountered during validation.

    Collects exception statistics from ``ValidationIssue.exception_info``
    fields, providing a quick overview of validator failures, retry
    attempts, and failure categories.
    """

    total_exceptions: int = 0
    total_retries: int = 0
    exceptions_by_type: dict[str, int] = field(default_factory=dict)
    exceptions_by_category: dict[str, int] = field(default_factory=dict)
    exceptions_by_validator: dict[str, int] = field(default_factory=dict)
    retryable_count: int = 0

    @classmethod
    def from_issues(cls, issues: list[ValidationIssue]) -> "ExceptionSummary":
        """Build summary from a list of ValidationIssues."""
        summary = cls()
        type_counter: Counter[str] = Counter()
        cat_counter: Counter[str] = Counter()
        val_counter: Counter[str] = Counter()

        for issue in issues:
            exc = issue.exception_info
            if exc is None or not exc.raised_exception:
                continue

            summary.total_exceptions += 1
            summary.total_retries += exc.retry_count

            if exc.exception_type:
                type_counter[exc.exception_type] += 1
            if exc.failure_category and exc.failure_category != "unknown":
                cat_counter[exc.failure_category] += 1
            if exc.validator_name:
                val_counter[exc.validator_name] += 1
            if exc.is_retryable:
                summary.retryable_count += 1

        summary.exceptions_by_type = dict(type_counter)
        summary.exceptions_by_category = dict(cat_counter)
        summary.exceptions_by_validator = dict(val_counter)
        return summary

    @property
    def has_exceptions(self) -> bool:
        return self.total_exceptions > 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "total_exceptions": self.total_exceptions,
        }
        if self.total_retries > 0:
            d["total_retries"] = self.total_retries
        if self.retryable_count > 0:
            d["retryable_count"] = self.retryable_count
        if self.exceptions_by_type:
            d["exceptions_by_type"] = self.exceptions_by_type
        if self.exceptions_by_category:
            d["exceptions_by_category"] = self.exceptions_by_category
        if self.exceptions_by_validator:
            d["exceptions_by_validator"] = self.exceptions_by_validator
        return d


@dataclass
class ReportStatistics:
    """Aggregate statistics computed from validation issues.

    Mirrors GX's ``ExpectationSuiteValidationResult.statistics``.  An
    instance is lazily computed by :meth:`Report._compute_statistics` and
    cached on ``Report.statistics``.
    """

    total_validations: int = 0
    successful_validations: int = 0
    unsuccessful_validations: int = 0
    success_percent: float = 0.0

    issues_by_severity: dict[str, int] = field(default_factory=dict)
    issues_by_column: dict[str, int] = field(default_factory=dict)
    issues_by_validator: dict[str, int] = field(default_factory=dict)
    issues_by_type: dict[str, int] = field(default_factory=dict)

    most_problematic_columns: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "total_validations": self.total_validations,
            "successful_validations": self.successful_validations,
            "unsuccessful_validations": self.unsuccessful_validations,
            "success_percent": round(self.success_percent, 2),
            "issues_by_severity": self.issues_by_severity,
            "issues_by_column": self.issues_by_column,
            "issues_by_validator": self.issues_by_validator,
            "issues_by_type": self.issues_by_type,
            "most_problematic_columns": [
                {"column": col, "issue_count": cnt}
                for col, cnt in self.most_problematic_columns
            ],
        }


@dataclass
class Report:
    """Validation report containing all issues found.

    Issues are maintained in a heap structure for efficient severity-based
    operations. The heap enables O(1) access to the most severe issue and
    O(k log n) retrieval of top-k issues by severity.

    PHASE 2 additions:
        ``statistics``  — aggregate breakdown by severity / column / validator.
        ``success``     — overall pass/fail (True if no HIGH+ issues).
    """

    issues: list[ValidationIssue] = field(default_factory=list)
    source: str = "unknown"
    row_count: int = 0
    column_count: int = 0
    result_format: ResultFormat = ResultFormat.SUMMARY

    # PHASE 2 fields
    statistics: ReportStatistics | None = field(default=None, repr=False, compare=False)
    success: bool = field(default=True, repr=False, compare=False)

    # PHASE 5 fields
    exception_summary: ExceptionSummary | None = field(default=None, repr=False, compare=False)

    _issues_heap: list[tuple[int, int, ValidationIssue]] = field(
        default_factory=list, repr=False, compare=False
    )
    _heap_counter: int = field(default=0, repr=False, compare=False)
    _sorted_cache: list[ValidationIssue] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Initialize heap from existing issues if any."""
        if self.issues:
            self._rebuild_heap()
        self._compute_statistics()

    def _rebuild_heap(self) -> None:
        """Rebuild heap from current issues list."""
        self._issues_heap = [
            (_SEVERITY_ORDER[issue.severity], i, issue)
            for i, issue in enumerate(self.issues)
        ]
        heapq.heapify(self._issues_heap)  # O(n) heapify instead of n*log(n) pushes
        self._heap_counter = len(self.issues)
        self._sorted_cache = None

    def _invalidate_cache(self) -> None:
        """Invalidate sorted cache when issues change."""
        self._sorted_cache = None

    def _compute_statistics(self) -> None:
        """(Re)compute aggregate statistics from the current issue list."""
        stats = ReportStatistics()
        sev_counter: Counter[str] = Counter()
        col_counter: Counter[str] = Counter()
        val_counter: Counter[str] = Counter()
        type_counter: Counter[str] = Counter()

        for issue in self.issues:
            sev_counter[issue.severity.value] += 1
            col_counter[issue.column] += 1
            type_counter[issue.issue_type] += 1
            if issue.validator_name:
                val_counter[issue.validator_name] += 1

        stats.unsuccessful_validations = len(self.issues)
        stats.issues_by_severity = dict(sev_counter)
        stats.issues_by_column = dict(col_counter)
        stats.issues_by_validator = dict(val_counter)
        stats.issues_by_type = dict(type_counter)
        stats.most_problematic_columns = col_counter.most_common(10)

        self.statistics = stats
        self.success = not self.has_high

        # PHASE 5: exception summary
        exc_summary = ExceptionSummary.from_issues(self.issues)
        self.exception_summary = exc_summary if exc_summary.has_exceptions else None

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue maintaining heap property.

        Args:
            issue: The validation issue to add.
        """
        self.issues.append(issue)
        heapq.heappush(
            self._issues_heap,
            (_SEVERITY_ORDER[issue.severity], self._heap_counter, issue),
        )
        self._heap_counter += 1
        self._invalidate_cache()
        self._compute_statistics()

    def add_issues(self, issues: list[ValidationIssue]) -> None:
        """Add multiple issues efficiently.

        Args:
            issues: List of validation issues to add.
        """
        for issue in issues:
            self.issues.append(issue)
            self._issues_heap.append(
                (_SEVERITY_ORDER[issue.severity], self._heap_counter, issue)
            )
            self._heap_counter += 1
        heapq.heapify(self._issues_heap)  # Re-heapify once after all additions
        self._invalidate_cache()
        self._compute_statistics()

    def get_sorted_issues(self) -> list[ValidationIssue]:
        """Get issues sorted by severity (highest first).

        Uses cached result if available for repeated access.

        Returns:
            List of issues sorted by severity.
        """
        if self._sorted_cache is not None:
            return self._sorted_cache

        if not self._issues_heap and self.issues:
            self._rebuild_heap()

        # Use Timsort for full sort (faster than heap extraction for full list)
        self._sorted_cache = sorted(
            self.issues, key=lambda x: _SEVERITY_ORDER[x.severity]
        )
        return self._sorted_cache

    def get_top_issues(self, k: int) -> list[ValidationIssue]:
        """Get top k issues by severity efficiently.

        Uses heap for O(k log n) performance, better than full sort for small k.

        Args:
            k: Number of top issues to retrieve.

        Returns:
            List of top k issues sorted by severity.
        """
        if not self._issues_heap and self.issues:
            self._rebuild_heap()

        if k >= len(self._issues_heap):
            return self.get_sorted_issues()

        # Use nsmallest for efficient top-k (severity 0 = highest priority)
        return [
            issue for _, _, issue in heapq.nsmallest(k, self._issues_heap)
        ]

    def get_most_severe(self) -> ValidationIssue | None:
        """Get the most severe issue in O(1) time.

        Returns:
            The most severe issue, or None if no issues exist.
        """
        if not self._issues_heap and self.issues:
            self._rebuild_heap()

        if not self._issues_heap:
            return None

        return self._issues_heap[0][2]  # Heap root is always min (most severe)

    def iter_by_severity(self) -> Iterator[ValidationIssue]:
        """Iterate through issues in severity order.

        Uses cached sorted list if available.

        Yields:
            Issues in descending severity order.
        """
        yield from self.get_sorted_issues()

    def __str__(self) -> str:
        """Return a formatted string representation using Rich."""
        console = Console(force_terminal=True, width=80)
        with console.capture() as capture:
            self._print_to_console(console)
        return capture.get()

    def _print_to_console(self, console: Console) -> None:
        """Print the report to a Rich console.

        Output adapts to result_format:
            BOOLEAN_ONLY: Minimal pass/fail per column (no details)
            BASIC:        Standard table (column, issue, count, severity)
            SUMMARY:      Standard table + details column
            COMPLETE:     Standard table + details + sample values
        """
        console.print()
        console.print("[bold]Truthound Report[/bold]")
        console.print("━" * 52)

        if not self.issues:
            console.print("[green]✓ No issues found[/green]")
            console.print()
            return

        table = Table(show_header=True, header_style="bold")
        table.add_column("Column", style="cyan")
        table.add_column("Issue", style="white")
        table.add_column("Count", justify="right")
        table.add_column("Severity", justify="center")

        # Add details column for BASIC+ formats
        show_details = self.result_format >= ResultFormat.BASIC
        if show_details:
            table.add_column("Details", style="dim")

        # Use pre-sorted heap for O(n) iteration instead of O(n log n) sort
        for issue in self.iter_by_severity():
            severity_style = self._get_severity_style(issue.severity)
            row = [
                issue.column,
                issue.issue_type,
                f"{issue.count:,}",
                f"[{severity_style}]{issue.severity.value}[/{severity_style}]",
            ]
            if show_details:
                row.append(str(issue.details or ""))
            table.add_row(*row)

        console.print(table)
        console.print()

        # Summary
        unique_columns = len({i.column for i in self.issues if i.column != "*"})
        console.print(f"Summary: {len(self.issues)} issues found in {unique_columns} columns")

        # PHASE 5: Exception summary
        if self.exception_summary is not None and self.exception_summary.has_exceptions:
            es = self.exception_summary
            console.print()
            console.print(f"[yellow]Exceptions: {es.total_exceptions} validator(s) raised errors[/yellow]")
            if es.total_retries > 0:
                console.print(f"  Retries: {es.total_retries}")
            if es.exceptions_by_category:
                cats = ", ".join(f"{k}={v}" for k, v in es.exceptions_by_category.items())
                console.print(f"  Categories: {cats}")

        console.print()

    def _get_severity_style(self, severity: Severity) -> str:
        """Get Rich style for severity level."""
        return {
            Severity.CRITICAL: "bold red",
            Severity.HIGH: "red",
            Severity.MEDIUM: "yellow",
            Severity.LOW: "dim",
        }[severity]

    def print(self) -> None:
        """Print the report to stdout."""
        console = Console()
        self._print_to_console(console)

    def to_dict(self) -> dict:
        """Convert report to dictionary for JSON serialization.

        Output varies by result_format:
            BOOLEAN_ONLY: {success, issue_count} only
            BASIC+:       full issue details + statistics
        """
        base: dict[str, Any] = {
            "source": self.source,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "issue_count": len(self.issues),
            "result_format": self.result_format.value,
            "success": self.success,
        }

        if self.result_format == ResultFormat.BOOLEAN_ONLY:
            # Minimal per-issue info: column, issue_type, count, severity
            base["issues"] = [
                {
                    "column": issue.column,
                    "issue_type": issue.issue_type,
                    "count": issue.count,
                    "severity": issue.severity.value,
                    "success": issue.success,
                }
                for issue in self.issues
            ]
        else:
            base["issues"] = [issue.to_dict() for issue in self.issues]

        if self.statistics is not None:
            base["statistics"] = self.statistics.to_dict()

        if self.exception_summary is not None:
            base["exception_summary"] = self.exception_summary.to_dict()

        return base

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def filter_by_severity(self, min_severity: Severity) -> "Report":
        """Return a new report with only issues at or above the given severity.

        Args:
            min_severity: Minimum severity level to include.

        Returns:
            New Report with filtered issues.
        """
        filtered_issues = [i for i in self.issues if i.severity >= min_severity]
        return Report(
            issues=filtered_issues,
            source=self.source,
            row_count=self.row_count,
            column_count=self.column_count,
            result_format=self.result_format,
        )

    @property
    def has_issues(self) -> bool:
        """Check if the report contains any issues."""
        return len(self.issues) > 0

    @property
    def has_critical(self) -> bool:
        """Check if the report contains critical issues."""
        return any(i.severity == Severity.CRITICAL for i in self.issues)

    @property
    def has_high(self) -> bool:
        """Check if the report contains high severity issues."""
        return any(i.severity >= Severity.HIGH for i in self.issues)


@dataclass
class PIIReport:
    """Report for PII scan results."""

    findings: list[dict] = field(default_factory=list)
    source: str = "unknown"
    row_count: int = 0

    def __str__(self) -> str:
        """Return a formatted string representation using Rich."""
        console = Console(force_terminal=True, width=80)
        with console.capture() as capture:
            self._print_to_console(console)
        return capture.get()

    def _print_to_console(self, console: Console) -> None:
        """Print the PII report to a Rich console."""
        console.print()
        console.print("[bold]Truthound PII Scan[/bold]")
        console.print("━" * 52)

        if not self.findings:
            console.print("[green]✓ No PII detected[/green]")
            console.print()
            return

        table = Table(show_header=True, header_style="bold")
        table.add_column("Column", style="cyan")
        table.add_column("PII Type", style="white")
        table.add_column("Count", justify="right")
        table.add_column("Confidence", justify="center")

        for finding in self.findings:
            table.add_row(
                finding["column"],
                finding["pii_type"],
                f"{finding['count']:,}",
                f"{finding['confidence']}%",
            )

        console.print(table)
        console.print()

        console.print(f"[yellow]Warning: Found {len(self.findings)} columns with potential PII[/yellow]")
        console.print()

    def print(self) -> None:
        """Print the report to stdout."""
        console = Console()
        self._print_to_console(console)

    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "source": self.source,
            "row_count": self.row_count,
            "finding_count": len(self.findings),
            "findings": self.findings,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @property
    def has_pii(self) -> bool:
        """Check if PII was detected."""
        return len(self.findings) > 0


@dataclass
class ProfileReport:
    """Report for dataset profiling."""

    source: str = "unknown"
    row_count: int = 0
    column_count: int = 0
    size_bytes: int = 0
    columns: list[dict] = field(default_factory=list)

    def __str__(self) -> str:
        """Return a formatted string representation using Rich."""
        console = Console(force_terminal=True, width=80)
        with console.capture() as capture:
            self._print_to_console(console)
        return capture.get()

    def _print_to_console(self, console: Console) -> None:
        """Print the profile report to a Rich console."""
        console.print()
        console.print("[bold]Truthound Profile[/bold]")
        console.print("━" * 52)

        size_str = self._format_size(self.size_bytes)
        console.print(f"Dataset: {self.source}")
        console.print(f"Rows: {self.row_count:,} | Columns: {self.column_count} | Size: {size_str}")
        console.print()

        if not self.columns:
            return

        table = Table(show_header=True, header_style="bold")
        table.add_column("Column", style="cyan")
        table.add_column("Type", style="white")
        table.add_column("Nulls", justify="right")
        table.add_column("Unique", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")

        for col in self.columns:
            table.add_row(
                col["name"],
                col["dtype"],
                col["null_pct"],
                col["unique_pct"],
                col.get("min", "-"),
                col.get("max", "-"),
            )

        console.print(table)
        console.print()

    def _format_size(self, size_bytes: int) -> str:
        """Format byte size to human readable string."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    def print(self) -> None:
        """Print the report to stdout."""
        console = Console()
        self._print_to_console(console)

    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "source": self.source,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "size_bytes": self.size_bytes,
            "columns": self.columns,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
