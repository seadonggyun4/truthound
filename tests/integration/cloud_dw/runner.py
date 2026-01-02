"""Integration test runner for Cloud DW tests.

This module provides a test runner that orchestrates integration tests
across multiple cloud data warehouse backends.

Features:
    - Parallel backend execution
    - Test result aggregation
    - Cost tracking and limits
    - CI/CD integration
    - Report generation
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

from tests.integration.cloud_dw.base import (
    CloudDWTestBackend,
    CloudDWTestCase,
    IntegrationTestConfig,
    TestCategory,
    TestDataset,
    TestMetrics,
    TestTable,
)
from tests.integration.cloud_dw.fixtures import (
    SQLDialect,
    StandardTestData,
    TestDataGenerator,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Test Result Types
# =============================================================================


class TestStatus(Enum):
    """Status of a test execution."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """Result of a single test execution.

    Attributes:
        test_name: Name of the test.
        backend_name: Backend that ran the test.
        status: Test status.
        duration_seconds: Test duration.
        message: Optional result message.
        error: Error details if failed.
        metrics: Test metrics.
    """

    test_name: str
    backend_name: str
    status: TestStatus
    duration_seconds: float = 0.0
    message: str | None = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "backend_name": self.backend_name,
            "status": self.status.value,
            "duration_seconds": self.duration_seconds,
            "message": self.message,
            "error": self.error,
            "metrics": self.metrics,
        }


@dataclass
class TestSuiteResult:
    """Result of a test suite execution.

    Attributes:
        suite_name: Name of the test suite.
        results: Individual test results.
        start_time: When the suite started.
        end_time: When the suite ended.
        total_cost_usd: Total estimated cost.
    """

    suite_name: str
    results: list[TestResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None
    total_cost_usd: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.SKIPPED)

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.ERROR)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    @property
    def duration_seconds(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    def add_result(self, result: TestResult) -> None:
        """Add a test result."""
        self.results.append(result)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suite_name": self.suite_name,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "skipped": self.skipped,
                "errors": self.errors,
                "success_rate": self.success_rate,
                "duration_seconds": self.duration_seconds,
                "total_cost_usd": self.total_cost_usd,
            },
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "results": [r.to_dict() for r in self.results],
        }

    def to_junit_xml(self) -> str:
        """Convert to JUnit XML format for CI integration."""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<testsuite name="{self.suite_name}" '
            f'tests="{self.total}" '
            f'failures="{self.failed}" '
            f'errors="{self.errors}" '
            f'skipped="{self.skipped}" '
            f'time="{self.duration_seconds:.3f}">',
        ]

        for result in self.results:
            classname = f"cloud_dw.{result.backend_name}"
            lines.append(
                f'  <testcase classname="{classname}" '
                f'name="{result.test_name}" '
                f'time="{result.duration_seconds:.3f}">'
            )

            if result.status == TestStatus.FAILED:
                lines.append(
                    f'    <failure message="{result.message or "Test failed"}">'
                    f'{result.error or ""}</failure>'
                )
            elif result.status == TestStatus.ERROR:
                lines.append(
                    f'    <error message="{result.message or "Test error"}">'
                    f'{result.error or ""}</error>'
                )
            elif result.status == TestStatus.SKIPPED:
                lines.append(
                    f'    <skipped message="{result.message or "Test skipped"}"/>'
                )

            lines.append("  </testcase>")

        lines.append("</testsuite>")
        return "\n".join(lines)


# =============================================================================
# Test Suite Definition
# =============================================================================


@dataclass
class TestSuite:
    """Definition of a test suite.

    A test suite groups related tests together with shared configuration.

    Attributes:
        name: Suite name.
        tests: List of test cases.
        categories: Categories to include.
        backends: Backends to run against.
        config: Test configuration.
    """

    name: str
    tests: list[CloudDWTestCase] = field(default_factory=list)
    categories: list[TestCategory] | None = None
    backends: list[str] | None = None
    config: IntegrationTestConfig | None = None

    def add_test(self, test: CloudDWTestCase) -> "TestSuite":
        """Add a test case to the suite."""
        self.tests.append(test)
        return self

    def filter_by_category(self, *categories: TestCategory) -> "TestSuite":
        """Filter tests by category."""
        self.categories = list(categories)
        return self

    def filter_by_backend(self, *backends: str) -> "TestSuite":
        """Filter backends to run against."""
        self.backends = list(backends)
        return self


# =============================================================================
# Integration Test Runner
# =============================================================================


class IntegrationTestRunner:
    """Runner for cloud DW integration tests.

    This class orchestrates the execution of integration tests across
    multiple cloud data warehouse backends.

    Features:
        - Parallel backend execution
        - Test data setup and teardown
        - Cost tracking
        - Result aggregation
        - CI/CD integration

    Example:
        >>> runner = IntegrationTestRunner()
        >>> runner.add_backend("bigquery")
        >>> runner.add_backend("snowflake")
        >>>
        >>> suite = TestSuite("validation_tests")
        >>> suite.add_test(NullDetectionTest())
        >>> suite.add_test(DuplicateDetectionTest())
        >>>
        >>> result = runner.run(suite)
        >>> print(result.to_junit_xml())
    """

    def __init__(
        self,
        config: IntegrationTestConfig | None = None,
    ) -> None:
        """Initialize the runner.

        Args:
            config: Test configuration.
        """
        self.config = config or IntegrationTestConfig.from_env()
        self._backends: dict[str, CloudDWTestBackend] = {}
        self._results: list[TestSuiteResult] = []

    def add_backend(
        self,
        name: str,
        backend: CloudDWTestBackend | None = None,
        **kwargs: Any,
    ) -> "IntegrationTestRunner":
        """Add a backend for testing.

        Args:
            name: Backend name.
            backend: Backend instance (created from registry if not provided).
            **kwargs: Credential overrides for registry creation.

        Returns:
            Self for chaining.
        """
        if backend is not None:
            self._backends[name] = backend
        else:
            from tests.integration.cloud_dw.backends import get_backend
            self._backends[name] = get_backend(name, self.config, **kwargs)

        return self

    def add_available_backends(self) -> "IntegrationTestRunner":
        """Add all available backends.

        Returns:
            Self for chaining.
        """
        from tests.integration.cloud_dw.backends import get_available_backends

        for name in get_available_backends():
            try:
                self.add_backend(name)
                logger.info(f"Added backend: {name}")
            except Exception as e:
                logger.warning(f"Could not add backend {name}: {e}")

        return self

    def run(
        self,
        suite: TestSuite,
        parallel_backends: bool = False,
    ) -> TestSuiteResult:
        """Run a test suite.

        Args:
            suite: Test suite to run.
            parallel_backends: If True, run backends in parallel.

        Returns:
            Test suite result.
        """
        result = TestSuiteResult(suite_name=suite.name)

        # Filter backends
        backends_to_run = self._filter_backends(suite.backends)
        if not backends_to_run:
            logger.warning("No backends available to run tests")
            result.end_time = datetime.utcnow()
            return result

        # Filter tests
        tests_to_run = self._filter_tests(suite.tests, suite.categories)
        if not tests_to_run:
            logger.warning("No tests to run after filtering")
            result.end_time = datetime.utcnow()
            return result

        logger.info(
            f"Running suite '{suite.name}': "
            f"{len(tests_to_run)} tests × {len(backends_to_run)} backends"
        )

        # Run tests
        if parallel_backends and len(backends_to_run) > 1:
            self._run_parallel(backends_to_run, tests_to_run, result)
        else:
            self._run_sequential(backends_to_run, tests_to_run, result)

        result.end_time = datetime.utcnow()

        # Calculate total cost
        for backend in backends_to_run.values():
            result.total_cost_usd += backend.metrics.total_cost_usd

        self._results.append(result)
        return result

    def _filter_backends(
        self,
        backend_names: list[str] | None,
    ) -> dict[str, CloudDWTestBackend]:
        """Filter backends by name."""
        if backend_names is None:
            return self._backends

        return {
            name: backend
            for name, backend in self._backends.items()
            if name in backend_names
        }

    def _filter_tests(
        self,
        tests: list[CloudDWTestCase],
        categories: list[TestCategory] | None,
    ) -> list[CloudDWTestCase]:
        """Filter tests by category."""
        if categories is None:
            return tests

        return [
            test for test in tests
            if test.category in categories
        ]

    def _run_sequential(
        self,
        backends: dict[str, CloudDWTestBackend],
        tests: list[CloudDWTestCase],
        result: TestSuiteResult,
    ) -> None:
        """Run tests sequentially across backends."""
        for backend_name, backend in backends.items():
            self._run_backend_tests(backend_name, backend, tests, result)

    def _run_parallel(
        self,
        backends: dict[str, CloudDWTestBackend],
        tests: list[CloudDWTestCase],
        result: TestSuiteResult,
    ) -> None:
        """Run tests in parallel across backends."""
        with ThreadPoolExecutor(max_workers=self.config.parallel_tests) as executor:
            futures = {
                executor.submit(
                    self._run_backend_tests,
                    backend_name,
                    backend,
                    tests,
                    result,
                ): backend_name
                for backend_name, backend in backends.items()
            }

            for future in as_completed(futures):
                backend_name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Backend {backend_name} failed: {e}")

    def _run_backend_tests(
        self,
        backend_name: str,
        backend: CloudDWTestBackend,
        tests: list[CloudDWTestCase],
        result: TestSuiteResult,
    ) -> None:
        """Run all tests against a single backend."""
        logger.info(f"Running tests on {backend_name}")

        # Get dialect for this backend
        dialect = self._get_dialect(backend_name)
        generator = TestDataGenerator(dialect)

        try:
            with backend:
                # Create test dataset
                dataset = backend.create_test_dataset(suffix=result.suite_name)

                for test in tests:
                    test_result = self._run_single_test(
                        backend_name,
                        backend,
                        test,
                        dataset,
                        generator,
                    )
                    result.add_result(test_result)

        except Exception as e:
            # Backend-level failure
            logger.error(f"Backend {backend_name} error: {e}")
            for test in tests:
                result.add_result(TestResult(
                    test_name=test.__class__.__name__,
                    backend_name=backend_name,
                    status=TestStatus.ERROR,
                    error=str(e),
                ))

    def _run_single_test(
        self,
        backend_name: str,
        backend: CloudDWTestBackend,
        test: CloudDWTestCase,
        dataset: TestDataset,
        generator: TestDataGenerator,
    ) -> TestResult:
        """Run a single test case."""
        test_name = test.__class__.__name__
        start_time = time.time()

        try:
            # Create test table if needed
            table = None
            if test.requires_data:
                # Determine which data type to use
                data_type = test.data_types[0] if test.data_types else None
                if data_type:
                    # Map data type to generator dataset name
                    dataset_name = self._data_type_to_dataset(data_type)
                    schema = generator.get_schema(dataset_name)
                    data = generator.generate(dataset_name)

                    table = backend.create_test_table(
                        dataset,
                        f"test_{test_name.lower()}",
                        schema,
                        data,
                    )

            # Run the test
            success = test.run(backend, dataset, table)

            duration = time.time() - start_time

            return TestResult(
                test_name=test_name,
                backend_name=backend_name,
                status=TestStatus.PASSED if success else TestStatus.FAILED,
                duration_seconds=duration,
                metrics=backend.metrics.to_dict(),
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(f"Test {test_name} failed on {backend_name}")

            return TestResult(
                test_name=test_name,
                backend_name=backend_name,
                status=TestStatus.ERROR,
                duration_seconds=duration,
                error=str(e),
            )

    def _get_dialect(self, backend_name: str) -> SQLDialect:
        """Get SQL dialect for a backend."""
        dialect_map = {
            "bigquery": SQLDialect.BIGQUERY,
            "snowflake": SQLDialect.SNOWFLAKE,
            "redshift": SQLDialect.REDSHIFT,
            "databricks": SQLDialect.DATABRICKS,
        }
        return dialect_map.get(backend_name, SQLDialect.BIGQUERY)

    def _data_type_to_dataset(self, data_type: Any) -> str:
        """Map test data type to dataset name."""
        from tests.integration.cloud_dw.base import TestDataType

        mapping = {
            TestDataType.BASIC: "users",
            TestDataType.TEMPORAL: "transactions",
            TestDataType.NULLS: "nulls",
            TestDataType.EDGE_CASES: "edge_cases",
            TestDataType.UNICODE: "unicode",
        }
        return mapping.get(data_type, "users")

    def get_results(self) -> list[TestSuiteResult]:
        """Get all test results."""
        return self._results

    def save_results(
        self,
        path: str | Path,
        format: str = "json",
    ) -> None:
        """Save test results to file.

        Args:
            path: Output file path.
            format: Output format ("json" or "junit").
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(path, "w") as f:
                json.dump(
                    [r.to_dict() for r in self._results],
                    f,
                    indent=2,
                    default=str,
                )
        elif format == "junit":
            with open(path, "w") as f:
                for result in self._results:
                    f.write(result.to_junit_xml())
                    f.write("\n")
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Results saved to {path}")

    def print_summary(self) -> None:
        """Print a summary of all results."""
        print("\n" + "=" * 60)
        print("INTEGRATION TEST SUMMARY")
        print("=" * 60)

        total_passed = 0
        total_failed = 0
        total_skipped = 0
        total_errors = 0
        total_cost = 0.0

        for result in self._results:
            print(f"\n{result.suite_name}:")
            print(f"  Passed:  {result.passed}")
            print(f"  Failed:  {result.failed}")
            print(f"  Skipped: {result.skipped}")
            print(f"  Errors:  {result.errors}")
            print(f"  Duration: {result.duration_seconds:.2f}s")
            print(f"  Cost:    ${result.total_cost_usd:.4f}")

            total_passed += result.passed
            total_failed += result.failed
            total_skipped += result.skipped
            total_errors += result.errors
            total_cost += result.total_cost_usd

        print("\n" + "-" * 60)
        print("TOTAL:")
        print(f"  Passed:  {total_passed}")
        print(f"  Failed:  {total_failed}")
        print(f"  Skipped: {total_skipped}")
        print(f"  Errors:  {total_errors}")
        print(f"  Cost:    ${total_cost:.4f}")
        print("=" * 60 + "\n")


# =============================================================================
# CI/CD Integration
# =============================================================================


def detect_ci_environment() -> dict[str, Any]:
    """Detect the CI/CD environment.

    Returns:
        Dictionary with CI environment information.
    """
    ci_vars = {
        "github_actions": {
            "detect": "GITHUB_ACTIONS",
            "vars": ["GITHUB_RUN_ID", "GITHUB_WORKFLOW", "GITHUB_SHA"],
        },
        "gitlab_ci": {
            "detect": "GITLAB_CI",
            "vars": ["CI_JOB_ID", "CI_PIPELINE_ID", "CI_COMMIT_SHA"],
        },
        "jenkins": {
            "detect": "JENKINS_URL",
            "vars": ["BUILD_ID", "JOB_NAME", "GIT_COMMIT"],
        },
        "circleci": {
            "detect": "CIRCLECI",
            "vars": ["CIRCLE_BUILD_NUM", "CIRCLE_WORKFLOW_ID", "CIRCLE_SHA1"],
        },
        "azure_devops": {
            "detect": "TF_BUILD",
            "vars": ["BUILD_BUILDID", "BUILD_DEFINITIONNAME", "BUILD_SOURCEVERSION"],
        },
    }

    for ci_name, config in ci_vars.items():
        if os.getenv(config["detect"]):
            return {
                "ci": ci_name,
                "is_ci": True,
                **{var: os.getenv(var) for var in config["vars"]},
            }

    return {"ci": None, "is_ci": False}


def create_ci_runner() -> IntegrationTestRunner:
    """Create a runner configured for CI/CD.

    Returns:
        Configured IntegrationTestRunner.
    """
    ci_env = detect_ci_environment()

    # Create config with CI-appropriate settings
    config = IntegrationTestConfig(
        dry_run=os.getenv("TRUTHOUND_TEST_DRY_RUN", "false").lower() == "true",
        max_cost_usd=float(os.getenv("TRUTHOUND_TEST_MAX_COST_USD", "5.0")),
        timeout_seconds=int(os.getenv("TRUTHOUND_TEST_TIMEOUT", "600")),
        cleanup_on_failure=True,
        collect_metrics=True,
        log_queries=ci_env.get("is_ci", False),  # Log queries in CI
    )

    runner = IntegrationTestRunner(config)

    # Add available backends
    runner.add_available_backends()

    return runner
