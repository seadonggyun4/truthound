"""Fuzz testing support for validators.

This module provides fuzz testing capabilities:
- Random data generation for validators
- Property-based testing integration
- Edge case discovery
- Crash detection and reporting

Example:
    from truthound.validators.sdk.enterprise.fuzzing import (
        FuzzRunner,
        FuzzConfig,
        run_fuzz_tests,
    )

    # Run fuzz tests
    runner = FuzzRunner(config)
    results = runner.fuzz(MyValidator)

    # Check for issues
    for result in results:
        if not result.success:
            print(f"Found issue: {result.error}")
"""

from __future__ import annotations

import logging
import random
import string
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Any, Callable, Generator, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FuzzStrategy(Enum):
    """Fuzzing strategies."""

    RANDOM = auto()          # Pure random data
    BOUNDARY = auto()        # Boundary values
    MUTATION = auto()        # Mutate valid data
    DICTIONARY = auto()      # Use dictionary of known problematic values
    STRUCTURE_AWARE = auto() # Schema-aware fuzzing


@dataclass(frozen=True)
class FuzzConfig:
    """Configuration for fuzz testing.

    Attributes:
        strategy: Fuzzing strategy to use
        iterations: Number of test iterations
        seed: Random seed for reproducibility
        max_rows: Maximum rows per test case
        max_columns: Maximum columns per test case
        timeout_seconds: Maximum time per test
        include_nulls: Include null values
        include_edge_cases: Include numeric edge cases
        include_unicode: Include unicode strings
        mutation_rate: Rate of mutations (0.0-1.0)
    """

    strategy: FuzzStrategy = FuzzStrategy.RANDOM
    iterations: int = 100
    seed: int | None = None
    max_rows: int = 1000
    max_columns: int = 20
    timeout_seconds: float = 10.0
    include_nulls: bool = True
    include_edge_cases: bool = True
    include_unicode: bool = True
    mutation_rate: float = 0.1

    @classmethod
    def quick(cls) -> "FuzzConfig":
        """Quick fuzz configuration."""
        return cls(
            iterations=10,
            max_rows=100,
            timeout_seconds=5.0,
        )

    @classmethod
    def thorough(cls) -> "FuzzConfig":
        """Thorough fuzz configuration."""
        return cls(
            iterations=1000,
            max_rows=10000,
            timeout_seconds=30.0,
        )


@dataclass
class FuzzResult:
    """Result of a single fuzz test.

    Attributes:
        iteration: Test iteration number
        success: Whether test passed
        error: Error message if failed
        exception: Exception if raised
        duration_seconds: Test duration
        data_shape: Shape of test data (rows, cols)
        seed_used: Random seed for this iteration
    """

    iteration: int
    success: bool
    error: str | None = None
    exception: Exception | None = None
    duration_seconds: float = 0.0
    data_shape: tuple[int, int] = (0, 0)
    seed_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "iteration": self.iteration,
            "success": self.success,
            "error": self.error,
            "exception_type": type(self.exception).__name__ if self.exception else None,
            "duration_seconds": self.duration_seconds,
            "data_shape": self.data_shape,
            "seed_used": self.seed_used,
        }


@dataclass
class FuzzReport:
    """Aggregated fuzz testing report.

    Attributes:
        total_iterations: Total tests run
        passed: Number of tests passed
        failed: Number of tests failed
        errors: List of error results
        total_duration_seconds: Total test time
        started_at: When testing started
        finished_at: When testing finished
    """

    total_iterations: int = 0
    passed: int = 0
    failed: int = 0
    errors: list[FuzzResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_iterations == 0:
            return 0.0
        return self.passed / self.total_iterations

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_iterations": self.total_iterations,
            "passed": self.passed,
            "failed": self.failed,
            "success_rate": self.success_rate,
            "total_duration_seconds": self.total_duration_seconds,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "errors": [e.to_dict() for e in self.errors],
        }


class DataGenerator(ABC):
    """Abstract base class for fuzz data generators."""

    @abstractmethod
    def generate(
        self,
        rows: int,
        columns: int,
        seed: int | None = None,
    ) -> Any:
        """Generate random data.

        Args:
            rows: Number of rows
            columns: Number of columns
            seed: Random seed

        Returns:
            Generated data (polars DataFrame)
        """
        pass


class RandomDataGenerator(DataGenerator):
    """Generates random test data for validators."""

    def __init__(self, config: FuzzConfig):
        """Initialize generator.

        Args:
            config: Fuzz configuration
        """
        self.config = config

        # Edge case values
        self._numeric_edge_cases = [
            0, -0, 1, -1,
            float("inf"), float("-inf"), float("nan"),
            2**31 - 1, -(2**31), 2**63 - 1, -(2**63),
            1e-300, 1e300, -1e-300, -1e300,
            0.1 + 0.2 - 0.3,  # Floating point precision
        ]

        self._string_edge_cases = [
            "",
            " ",
            "\t",
            "\n",
            "\r\n",
            "null",
            "NULL",
            "None",
            "undefined",
            "NaN",
            "inf",
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../etc/passwd",
            "\x00",  # Null byte
            "\xff" * 100,  # High bytes
            "a" * 10000,  # Long string
        ]

        self._unicode_strings = [
            "Hello 世界",
            "مرحبا",
            "שלום",
            "🎉🚀💻",
            "Ñoño",
            "日本語テスト",
            "\u200b",  # Zero-width space
            "\ufeff",  # BOM
        ]

    def _random_string(self, length: int = 10) -> str:
        """Generate random string."""
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))

    def _random_numeric(self) -> float | int:
        """Generate random numeric value."""
        choice = random.random()
        if choice < 0.3:
            return random.randint(-1000000, 1000000)
        elif choice < 0.6:
            return random.uniform(-1e10, 1e10)
        elif choice < 0.8 and self.config.include_edge_cases:
            return random.choice(self._numeric_edge_cases)
        else:
            return random.gauss(0, 100)

    def _random_value(self, dtype: str) -> Any:
        """Generate random value of specified type."""
        if self.config.include_nulls and random.random() < 0.05:
            return None

        if dtype == "int":
            if self.config.include_edge_cases and random.random() < 0.1:
                int_edge_cases = [
                    int(n) for n in self._numeric_edge_cases
                    if isinstance(n, int) or (isinstance(n, float) and n.is_integer() and abs(n) < 2**63)
                ]
                if int_edge_cases:
                    return random.choice(int_edge_cases)
            return random.randint(-2**31, 2**31 - 1)

        elif dtype == "float":
            return self._random_numeric()

        elif dtype == "str":
            choice = random.random()
            if self.config.include_edge_cases and choice < 0.1:
                return random.choice(self._string_edge_cases)
            elif self.config.include_unicode and choice < 0.2:
                return random.choice(self._unicode_strings)
            else:
                return self._random_string(random.randint(1, 50))

        elif dtype == "bool":
            return random.choice([True, False])

        elif dtype == "date":
            days = random.randint(-36500, 36500)  # ~100 years range
            return datetime.now(timezone.utc) + timedelta(days=days)

        else:
            return self._random_string()

    def generate(
        self,
        rows: int,
        columns: int,
        seed: int | None = None,
    ) -> Any:
        """Generate random DataFrame.

        Args:
            rows: Number of rows
            columns: Number of columns
            seed: Random seed

        Returns:
            Polars LazyFrame
        """
        try:
            import polars as pl
        except ImportError:
            raise ImportError("polars is required for fuzz testing")

        if seed is not None:
            random.seed(seed)

        # Generate column names and types
        dtypes = ["int", "float", "str", "bool", "date"]
        data: dict[str, list[Any]] = {}

        for i in range(min(columns, self.config.max_columns)):
            col_name = f"col_{i}"
            col_type = random.choice(dtypes)
            data[col_name] = [
                self._random_value(col_type)
                for _ in range(min(rows, self.config.max_rows))
            ]

        return pl.LazyFrame(data)


class BoundaryDataGenerator(DataGenerator):
    """Generates boundary value test cases."""

    def generate(
        self,
        rows: int,
        columns: int,
        seed: int | None = None,
    ) -> Any:
        """Generate boundary value test data."""
        try:
            import polars as pl
        except ImportError:
            raise ImportError("polars is required for fuzz testing")

        if seed is not None:
            random.seed(seed)

        # Boundary values for different types
        int_boundaries = [
            0, 1, -1, 2**31 - 1, -2**31, 2**63 - 1, -2**63,
        ]

        float_boundaries = [
            0.0, -0.0, 1.0, -1.0,
            float("inf"), float("-inf"), float("nan"),
            1e-300, 1e300,
            2.2250738585072014e-308,  # Min normal
            1.7976931348623157e+308,  # Max
        ]

        str_boundaries = [
            "", " ", "a", "a" * 1000,
            "\x00", "\xff",
        ]

        data = {
            "int_col": int_boundaries[:rows] if rows <= len(int_boundaries)
                else int_boundaries * (rows // len(int_boundaries) + 1),
            "float_col": float_boundaries[:rows] if rows <= len(float_boundaries)
                else float_boundaries * (rows // len(float_boundaries) + 1),
            "str_col": str_boundaries[:rows] if rows <= len(str_boundaries)
                else str_boundaries * (rows // len(str_boundaries) + 1),
        }

        # Trim to requested rows
        for key in data:
            data[key] = data[key][:rows]

        return pl.LazyFrame(data)


class PropertyBasedTester:
    """Property-based testing for validators.

    Tests that certain properties hold for all inputs:
    - Validator should not crash on any input
    - Output should always be a list
    - Issues should have required fields
    """

    def __init__(self, validator_class: type):
        """Initialize tester.

        Args:
            validator_class: Validator class to test
        """
        self.validator_class = validator_class

    def test_no_crash(self, data: Any) -> bool:
        """Test that validator doesn't crash."""
        try:
            validator = self.validator_class()
            result = validator.validate(data)
            return True
        except Exception:
            return False

    def test_returns_list(self, data: Any) -> bool:
        """Test that validator returns a list."""
        try:
            validator = self.validator_class()
            result = validator.validate(data)
            return isinstance(result, list)
        except Exception:
            return False

    def test_issues_have_fields(self, data: Any) -> bool:
        """Test that issues have required fields."""
        try:
            validator = self.validator_class()
            issues = validator.validate(data)

            for issue in issues:
                if not hasattr(issue, "column"):
                    return False
                if not hasattr(issue, "issue_type"):
                    return False
                if not hasattr(issue, "severity"):
                    return False

            return True
        except Exception:
            return False

    def run_all(self, data: Any) -> dict[str, bool]:
        """Run all property tests.

        Args:
            data: Test data

        Returns:
            Dictionary of test name to result
        """
        return {
            "no_crash": self.test_no_crash(data),
            "returns_list": self.test_returns_list(data),
            "issues_have_fields": self.test_issues_have_fields(data),
        }


class FuzzRunner:
    """Runs fuzz tests against validators."""

    def __init__(self, config: FuzzConfig | None = None):
        """Initialize runner.

        Args:
            config: Fuzz configuration
        """
        self.config = config or FuzzConfig()
        self._generator = self._create_generator()

    def _create_generator(self) -> DataGenerator:
        """Create data generator based on strategy."""
        if self.config.strategy == FuzzStrategy.BOUNDARY:
            return BoundaryDataGenerator()
        else:
            return RandomDataGenerator(self.config)

    def fuzz(
        self,
        validator_class: type,
        on_result: Callable[[FuzzResult], None] | None = None,
    ) -> FuzzReport:
        """Run fuzz tests against validator.

        Args:
            validator_class: Validator class to test
            on_result: Callback for each result

        Returns:
            FuzzReport with all results
        """
        report = FuzzReport(started_at=datetime.now(timezone.utc))

        # Set seed if specified
        base_seed = self.config.seed or int(time.time())

        for i in range(self.config.iterations):
            iteration_seed = base_seed + i
            random.seed(iteration_seed)

            # Generate random data dimensions
            rows = random.randint(1, self.config.max_rows)
            columns = random.randint(1, self.config.max_columns)

            start_time = time.perf_counter()

            try:
                # Generate data
                data = self._generator.generate(rows, columns, iteration_seed)

                # Run validator
                validator = validator_class()
                result = validator.validate(data)

                # Check result
                duration = time.perf_counter() - start_time

                if duration > self.config.timeout_seconds:
                    fuzz_result = FuzzResult(
                        iteration=i,
                        success=False,
                        error=f"Timeout after {duration:.2f}s",
                        duration_seconds=duration,
                        data_shape=(rows, columns),
                        seed_used=iteration_seed,
                    )
                    report.failed += 1
                    report.errors.append(fuzz_result)
                else:
                    fuzz_result = FuzzResult(
                        iteration=i,
                        success=True,
                        duration_seconds=duration,
                        data_shape=(rows, columns),
                        seed_used=iteration_seed,
                    )
                    report.passed += 1

            except Exception as e:
                duration = time.perf_counter() - start_time
                fuzz_result = FuzzResult(
                    iteration=i,
                    success=False,
                    error=str(e),
                    exception=e,
                    duration_seconds=duration,
                    data_shape=(rows, columns),
                    seed_used=iteration_seed,
                )
                report.failed += 1
                report.errors.append(fuzz_result)

            report.total_iterations += 1
            report.total_duration_seconds += fuzz_result.duration_seconds

            if on_result:
                on_result(fuzz_result)

        report.finished_at = datetime.now(timezone.utc)
        return report

    def fuzz_with_properties(
        self,
        validator_class: type,
    ) -> dict[str, FuzzReport]:
        """Run fuzz tests with property checking.

        Args:
            validator_class: Validator to test

        Returns:
            Dictionary of property name to fuzz report
        """
        tester = PropertyBasedTester(validator_class)
        reports: dict[str, FuzzReport] = {}

        # Test each property
        for prop_name, prop_test in [
            ("no_crash", tester.test_no_crash),
            ("returns_list", tester.test_returns_list),
            ("issues_have_fields", tester.test_issues_have_fields),
        ]:
            report = FuzzReport(started_at=datetime.now(timezone.utc))

            base_seed = self.config.seed or int(time.time())

            for i in range(self.config.iterations):
                iteration_seed = base_seed + i
                random.seed(iteration_seed)

                rows = random.randint(1, self.config.max_rows)
                columns = random.randint(1, self.config.max_columns)

                start_time = time.perf_counter()

                try:
                    data = self._generator.generate(rows, columns, iteration_seed)
                    passed = prop_test(data)
                    duration = time.perf_counter() - start_time

                    if passed:
                        report.passed += 1
                    else:
                        report.failed += 1
                        report.errors.append(FuzzResult(
                            iteration=i,
                            success=False,
                            error=f"Property '{prop_name}' violated",
                            duration_seconds=duration,
                            data_shape=(rows, columns),
                            seed_used=iteration_seed,
                        ))

                except Exception as e:
                    duration = time.perf_counter() - start_time
                    report.failed += 1
                    report.errors.append(FuzzResult(
                        iteration=i,
                        success=False,
                        error=str(e),
                        exception=e,
                        duration_seconds=duration,
                        data_shape=(rows, columns),
                        seed_used=iteration_seed,
                    ))

                report.total_iterations += 1
                report.total_duration_seconds += duration

            report.finished_at = datetime.now(timezone.utc)
            reports[prop_name] = report

        return reports


def run_fuzz_tests(
    validator_class: type,
    iterations: int = 100,
    seed: int | None = None,
) -> FuzzReport:
    """Run fuzz tests on a validator.

    Args:
        validator_class: Validator to test
        iterations: Number of iterations
        seed: Random seed

    Returns:
        FuzzReport
    """
    config = FuzzConfig(iterations=iterations, seed=seed)
    runner = FuzzRunner(config)
    return runner.fuzz(validator_class)
