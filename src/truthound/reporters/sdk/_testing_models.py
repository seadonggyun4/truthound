"""Mock data builders and fixtures for reporter SDK tests."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class Severity(Enum):
    """Severity levels for mock validation results."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MockValidatorResult:
    """Mock validator result for testing."""

    validator_name: str
    column: str | None = None
    passed: bool = True
    message: str = ""
    severity: Severity = Severity.ERROR
    details: dict[str, Any] = field(default_factory=dict)
    row_count: int = 100
    failed_count: int = 0
    execution_time_ms: float = 10.0
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if not self.message:
            status = "passed" if self.passed else "failed"
            self.message = f"Validation {status} for {self.column or 'data'}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "validator_name": self.validator_name,
            "column": self.column,
            "passed": self.passed,
            "message": self.message,
            "severity": self.severity.value,
            "details": self.details,
            "row_count": self.row_count,
            "failed_count": self.failed_count,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class MockValidationResult:
    """Mock validation result for testing reporters."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data_asset: str = "test_data.csv"
    validator_results: list[MockValidatorResult] = field(default_factory=list)
    success: bool = True
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    row_count: int = 1000
    columns: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.started_at is None:
            self.started_at = datetime.now() - timedelta(seconds=5)
        if self.completed_at is None:
            self.completed_at = datetime.now()
        if not self.columns:
            self.columns = ["id", "name", "email", "age", "status"]
        if not self.validator_results:
            self._generate_default_results()

    def _generate_default_results(self) -> None:
        """Populate default validator results."""
        self.validator_results = [
            MockValidatorResult(
                validator_name="not_null",
                column="id",
                passed=True,
                row_count=self.row_count,
            ),
            MockValidatorResult(
                validator_name="unique",
                column="id",
                passed=True,
                row_count=self.row_count,
            ),
            MockValidatorResult(
                validator_name="email_format",
                column="email",
                passed=self.success,
                failed_count=0 if self.success else 5,
                row_count=self.row_count,
            ),
        ]

    @property
    def passed_count(self) -> int:
        return sum(1 for result in self.validator_results if result.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for result in self.validator_results if not result.passed)

    @property
    def total_count(self) -> int:
        return len(self.validator_results)

    @property
    def success_rate(self) -> float:
        if self.total_count == 0:
            return 100.0
        return (self.passed_count / self.total_count) * 100

    @property
    def duration_ms(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "run_id": self.run_id,
            "data_asset": self.data_asset,
            "success": self.success,
            "validator_results": [result.to_dict() for result in self.validator_results],
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "total_count": self.total_count,
            "success_rate": self.success_rate,
            "row_count": self.row_count,
            "columns": self.columns,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


class MockResultBuilder:
    """Fluent builder for creating mock validation results."""

    def __init__(self) -> None:
        self._run_id = str(uuid.uuid4())
        self._data_asset = "test_data.csv"
        self._validator_results: list[MockValidatorResult] = []
        self._metadata: dict[str, Any] = {}
        self._row_count = 1000
        self._columns: list[str] = []
        self._started_at: datetime | None = None
        self._completed_at: datetime | None = None

    def with_run_id(self, run_id: str) -> MockResultBuilder:
        self._run_id = run_id
        return self

    def with_data_asset(self, data_asset: str) -> MockResultBuilder:
        self._data_asset = data_asset
        return self

    def with_row_count(self, count: int) -> MockResultBuilder:
        self._row_count = count
        return self

    def with_columns(self, columns: list[str]) -> MockResultBuilder:
        self._columns = columns
        return self

    def with_metadata(self, metadata: dict[str, Any]) -> MockResultBuilder:
        self._metadata = metadata
        return self

    def with_timestamps(
        self,
        started_at: datetime,
        completed_at: datetime | None = None,
    ) -> MockResultBuilder:
        self._started_at = started_at
        self._completed_at = completed_at or datetime.now()
        return self

    def add_result(self, result: MockValidatorResult) -> MockResultBuilder:
        self._validator_results.append(result)
        return self

    def add_passed(
        self,
        validator_name: str,
        column: str | None = None,
        **kwargs: Any,
    ) -> MockResultBuilder:
        self._validator_results.append(
            MockValidatorResult(
                validator_name=validator_name,
                column=column,
                passed=True,
                row_count=self._row_count,
                **kwargs,
            )
        )
        return self

    def add_failed(
        self,
        validator_name: str,
        column: str | None = None,
        failed_count: int = 1,
        severity: Severity = Severity.ERROR,
        **kwargs: Any,
    ) -> MockResultBuilder:
        self._validator_results.append(
            MockValidatorResult(
                validator_name=validator_name,
                column=column,
                passed=False,
                failed_count=failed_count,
                severity=severity,
                row_count=self._row_count,
                **kwargs,
            )
        )
        return self

    def add_warning(
        self,
        validator_name: str,
        column: str | None = None,
        message: str = "",
        **kwargs: Any,
    ) -> MockResultBuilder:
        self._validator_results.append(
            MockValidatorResult(
                validator_name=validator_name,
                column=column,
                passed=True,
                severity=Severity.WARNING,
                message=message,
                row_count=self._row_count,
                **kwargs,
            )
        )
        return self

    def _build_many_validators(self) -> MockValidationResult:
        """Build a result with many validators for stress-style fixtures."""
        for index in range(100):
            if index % 5 == 0:
                self.add_failed(f"validator_{index:03d}", column=f"col_{index % 10}")
            else:
                self.add_passed(f"validator_{index:03d}", column=f"col_{index % 10}")
        return self.build()

    def build(self) -> MockValidationResult:
        success = all(result.passed for result in self._validator_results)
        return MockValidationResult(
            run_id=self._run_id,
            data_asset=self._data_asset,
            validator_results=self._validator_results,
            success=success,
            started_at=self._started_at,
            completed_at=self._completed_at,
            metadata=self._metadata,
            row_count=self._row_count,
            columns=self._columns or ["id", "name", "email", "age", "status"],
        )


def create_mock_result(
    passed: int = 5,
    failed: int = 0,
    warnings: int = 0,
    row_count: int = 1000,
    data_asset: str = "test_data.csv",
) -> MockValidationResult:
    """Create a mock validation result with specified counts."""
    builder = MockResultBuilder().with_data_asset(data_asset).with_row_count(row_count)
    validators = ["not_null", "unique", "range", "format", "regex", "length"]
    columns = ["id", "name", "email", "age", "status", "created_at"]

    for index in range(passed):
        builder.add_passed(
            validators[index % len(validators)],
            column=columns[index % len(columns)],
        )

    for index in range(failed):
        builder.add_failed(
            validators[index % len(validators)],
            column=columns[index % len(columns)],
            failed_count=max(1, row_count // 100),
        )

    for index in range(warnings):
        column = columns[index % len(columns)]
        builder.add_warning(
            validators[index % len(validators)],
            column=column,
            message=f"Warning: potential issue in {column}",
        )

    return builder.build()


def create_mock_results(
    count: int = 5,
    success_rate: float = 0.8,
    row_count: int = 1000,
) -> list[MockValidationResult]:
    """Create multiple mock validation results."""
    import random

    results: list[MockValidationResult] = []
    for index in range(count):
        is_success = random.random() < success_rate
        passed = random.randint(3, 10)
        failed = 0 if is_success else random.randint(1, 3)
        results.append(
            create_mock_result(
                passed=passed,
                failed=failed,
                row_count=row_count,
                data_asset=f"data_{index:03d}.csv",
            )
        )
    return results


def create_mock_validator_result(
    validator_name: str = "not_null",
    column: str | None = "id",
    passed: bool = True,
    **kwargs: Any,
) -> MockValidatorResult:
    """Create a single mock validator result."""
    return MockValidatorResult(
        validator_name=validator_name,
        column=column,
        passed=passed,
        **kwargs,
    )


def create_sample_data() -> dict[str, list[MockValidationResult]]:
    """Create sample data for common reporter test scenarios."""
    return {
        "all_passed": [create_mock_result(passed=10, failed=0)],
        "all_failed": [create_mock_result(passed=0, failed=10)],
        "mixed": [create_mock_result(passed=7, failed=3)],
        "empty": [create_mock_result(passed=0, failed=0)],
        "single_passed": [create_mock_result(passed=1, failed=0)],
        "single_failed": [create_mock_result(passed=0, failed=1)],
        "large_dataset": [create_mock_result(passed=50, failed=10, row_count=1_000_000)],
        "multiple_runs": create_mock_results(count=5, success_rate=0.8),
    }


def create_edge_case_data() -> dict[str, MockValidationResult]:
    """Create edge-case data for reporter boundary tests."""
    return {
        "empty_validators": MockResultBuilder().build(),
        "zero_rows": MockResultBuilder().with_row_count(0).add_passed("not_null", column="id").build(),
        "unicode_data_asset": MockResultBuilder()
        .with_data_asset("데이터_파일_🔥.csv")
        .add_passed("not_null", column="이름")
        .build(),
        "special_chars": MockResultBuilder()
        .with_data_asset('file"with<special>chars&.csv')
        .add_failed(
            "format",
            column="col<with>chars",
            message='Error: "value" is <invalid>',
        )
        .build(),
        "long_column_names": MockResultBuilder()
        .add_passed(
            "not_null",
            column="this_is_a_very_long_column_name_that_might_cause_formatting_issues_in_some_reporters",
        )
        .build(),
        "many_validators": MockResultBuilder().with_row_count(1000)._build_many_validators(),
    }


def create_stress_test_data(
    num_validators: int = 1000,
    num_rows: int = 10_000_000,
) -> MockValidationResult:
    """Create stress test data for reporter performance testing."""
    import random

    builder = MockResultBuilder().with_row_count(num_rows).with_data_asset("stress_test.csv")
    validators = ["not_null", "unique", "range", "format", "regex", "length", "custom"]
    columns = [f"col_{index}" for index in range(50)]

    for _index in range(num_validators):
        if random.random() < 0.9:
            builder.add_passed(
                random.choice(validators),
                column=random.choice(columns),
            )
        else:
            builder.add_failed(
                random.choice(validators),
                column=random.choice(columns),
                failed_count=random.randint(1, num_rows // 1000),
            )

    return builder.build()


__all__ = [
    "MockResultBuilder",
    "MockValidationResult",
    "MockValidatorResult",
    "Severity",
    "create_edge_case_data",
    "create_mock_result",
    "create_mock_results",
    "create_mock_validator_result",
    "create_sample_data",
    "create_stress_test_data",
]
