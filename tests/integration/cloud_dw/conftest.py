"""Pytest configuration for Cloud DW integration tests.

This module provides pytest fixtures, markers, and hooks for
running integration tests against cloud data warehouses.

Usage:
    # Run all integration tests
    pytest tests/integration/cloud_dw/ -v

    # Run specific platform tests
    pytest tests/integration/cloud_dw/ -v -m bigquery

    # Run with dry-run mode
    pytest tests/integration/cloud_dw/ -v --dry-run

    # Skip expensive tests
    pytest tests/integration/cloud_dw/ -v --skip-expensive

Configuration:
    Tests are configured via environment variables:
    - BIGQUERY_PROJECT: GCP project for BigQuery
    - SNOWFLAKE_ACCOUNT: Snowflake account
    - REDSHIFT_HOST: Redshift cluster endpoint
    - DATABRICKS_HOST: Databricks workspace URL
    - TRUTHOUND_TEST_DRY_RUN: Enable dry-run mode
    - TRUTHOUND_TEST_MAX_COST_USD: Maximum cost limit
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Generator

import pytest

from tests.integration.cloud_dw.base import (
    ConnectionStatus,
    IntegrationTestConfig,
    TestDataset,
)
from tests.integration.cloud_dw.fixtures import (
    SQLDialect,
    StandardTestData,
    TestDataGenerator,
)
from tests.integration.cloud_dw.runner import (
    IntegrationTestRunner,
    TestSuiteResult,
    detect_ci_environment,
)

if TYPE_CHECKING:
    from tests.integration.cloud_dw.base import CloudDWTestBackend

logger = logging.getLogger(__name__)


# =============================================================================
# Pytest Markers
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers."""
    # Platform markers
    config.addinivalue_line(
        "markers",
        "bigquery: marks test to run only on BigQuery",
    )
    config.addinivalue_line(
        "markers",
        "snowflake: marks test to run only on Snowflake",
    )
    config.addinivalue_line(
        "markers",
        "redshift: marks test to run only on Redshift",
    )
    config.addinivalue_line(
        "markers",
        "databricks: marks test to run only on Databricks",
    )

    # Feature markers
    config.addinivalue_line(
        "markers",
        "expensive: marks test as expensive (high cost or long duration)",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks test as slow (> 60 seconds)",
    )
    config.addinivalue_line(
        "markers",
        "requires_data: marks test as requiring test data setup",
    )

    # Category markers
    config.addinivalue_line(
        "markers",
        "connection: marks test as testing connection functionality",
    )
    config.addinivalue_line(
        "markers",
        "schema: marks test as testing schema inference",
    )
    config.addinivalue_line(
        "markers",
        "validation: marks test as testing data validation",
    )
    config.addinivalue_line(
        "markers",
        "performance: marks test as a performance benchmark",
    )
    config.addinivalue_line(
        "markers",
        "truthound: marks test as testing Truthound validator integration",
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options.

    Note: --skip-expensive is defined in the parent conftest.py
    (tests/integration/conftest.py) to avoid duplication.
    """
    parser.addoption(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run tests in dry-run mode (validate queries without executing)",
    )
    parser.addoption(
        "--max-cost",
        type=float,
        default=None,
        help="Maximum cost limit in USD for the test run",
    )
    parser.addoption(
        "--backend",
        action="append",
        default=None,
        help="Specific backend(s) to test (can be specified multiple times)",
    )
    parser.addoption(
        "--parallel-backends",
        action="store_true",
        default=False,
        help="Run backends in parallel",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Modify collected tests based on options and markers."""
    # Skip expensive tests if requested
    if config.getoption("--skip-expensive"):
        skip_expensive = pytest.mark.skip(reason="--skip-expensive option set")
        for item in items:
            if "expensive" in item.keywords:
                item.add_marker(skip_expensive)

    # Filter by backend if specified
    backends = config.getoption("--backend")
    if backends:
        skip_other = pytest.mark.skip(reason=f"Only running backends: {backends}")
        for item in items:
            item_backends = []
            for marker in ["bigquery", "snowflake", "redshift", "databricks"]:
                if marker in item.keywords:
                    item_backends.append(marker)

            # If test has backend markers and none match, skip
            if item_backends and not any(b in backends for b in item_backends):
                item.add_marker(skip_other)


# =============================================================================
# Fixtures - Configuration
# =============================================================================


@pytest.fixture(scope="session")
def integration_config(request: pytest.FixtureRequest) -> IntegrationTestConfig:
    """Create integration test configuration.

    Configuration is sourced from:
    1. Command line options
    2. Environment variables
    3. Default values
    """
    config = IntegrationTestConfig.from_env()

    # Override with command line options
    if request.config.getoption("--dry-run"):
        config.dry_run = True

    max_cost = request.config.getoption("--max-cost")
    if max_cost is not None:
        config.max_cost_usd = max_cost

    return config


@pytest.fixture(scope="session")
def ci_environment() -> dict[str, Any]:
    """Detect CI/CD environment."""
    return detect_ci_environment()


# =============================================================================
# Fixtures - Backend Availability
# =============================================================================


@pytest.fixture(scope="session")
def available_backends() -> list[str]:
    """Get list of available backends."""
    from tests.integration.cloud_dw.backends import get_available_backends

    return get_available_backends()


@pytest.fixture(scope="session")
def backend_availability_report() -> dict[str, dict[str, Any]]:
    """Get detailed availability report."""
    from tests.integration.cloud_dw.backends import get_backend_availability_report

    return get_backend_availability_report()


# =============================================================================
# Fixtures - BigQuery
# =============================================================================


@pytest.fixture(scope="session")
def bigquery_available() -> bool:
    """Check if BigQuery is available."""
    from tests.integration.cloud_dw.backends import get_available_backends

    return "bigquery" in get_available_backends()


@pytest.fixture(scope="session")
def bigquery_backend(
    integration_config: IntegrationTestConfig,
    bigquery_available: bool,
) -> Generator["CloudDWTestBackend | None", None, None]:
    """Create BigQuery test backend."""
    if not bigquery_available:
        pytest.skip("BigQuery not available")
        yield None
        return

    from tests.integration.cloud_dw.backends import get_backend

    backend = get_backend("bigquery", integration_config)
    backend.connect()

    yield backend

    backend.cleanup_test_data(force=True)
    backend.disconnect()


@pytest.fixture
def bigquery_dataset(
    bigquery_backend: "CloudDWTestBackend",
) -> Generator[TestDataset, None, None]:
    """Create a test dataset for BigQuery."""
    if bigquery_backend is None:
        pytest.skip("BigQuery backend not available")

    dataset = bigquery_backend.create_test_dataset()
    yield dataset

    # Cleanup is handled by backend fixture


@pytest.fixture
def bigquery_generator() -> TestDataGenerator:
    """Create data generator for BigQuery."""
    return TestDataGenerator(SQLDialect.BIGQUERY)


# =============================================================================
# Fixtures - Snowflake
# =============================================================================


@pytest.fixture(scope="session")
def snowflake_available() -> bool:
    """Check if Snowflake is available."""
    from tests.integration.cloud_dw.backends import get_available_backends

    return "snowflake" in get_available_backends()


@pytest.fixture(scope="session")
def snowflake_backend(
    integration_config: IntegrationTestConfig,
    snowflake_available: bool,
) -> Generator["CloudDWTestBackend | None", None, None]:
    """Create Snowflake test backend."""
    if not snowflake_available:
        pytest.skip("Snowflake not available")
        yield None
        return

    from tests.integration.cloud_dw.backends import get_backend

    backend = get_backend("snowflake", integration_config)
    backend.connect()

    yield backend

    backend.cleanup_test_data(force=True)
    backend.disconnect()


@pytest.fixture
def snowflake_dataset(
    snowflake_backend: "CloudDWTestBackend",
) -> Generator[TestDataset, None, None]:
    """Create a test dataset for Snowflake."""
    if snowflake_backend is None:
        pytest.skip("Snowflake backend not available")

    dataset = snowflake_backend.create_test_dataset()
    yield dataset


@pytest.fixture
def snowflake_generator() -> TestDataGenerator:
    """Create data generator for Snowflake."""
    return TestDataGenerator(SQLDialect.SNOWFLAKE)


# =============================================================================
# Fixtures - Redshift
# =============================================================================


@pytest.fixture(scope="session")
def redshift_available() -> bool:
    """Check if Redshift is available."""
    from tests.integration.cloud_dw.backends import get_available_backends

    return "redshift" in get_available_backends()


@pytest.fixture(scope="session")
def redshift_backend(
    integration_config: IntegrationTestConfig,
    redshift_available: bool,
) -> Generator["CloudDWTestBackend | None", None, None]:
    """Create Redshift test backend."""
    if not redshift_available:
        pytest.skip("Redshift not available")
        yield None
        return

    from tests.integration.cloud_dw.backends import get_backend

    backend = get_backend("redshift", integration_config)
    backend.connect()

    yield backend

    backend.cleanup_test_data(force=True)
    backend.disconnect()


@pytest.fixture
def redshift_dataset(
    redshift_backend: "CloudDWTestBackend",
) -> Generator[TestDataset, None, None]:
    """Create a test dataset for Redshift."""
    if redshift_backend is None:
        pytest.skip("Redshift backend not available")

    dataset = redshift_backend.create_test_dataset()
    yield dataset


@pytest.fixture
def redshift_generator() -> TestDataGenerator:
    """Create data generator for Redshift."""
    return TestDataGenerator(SQLDialect.REDSHIFT)


# =============================================================================
# Fixtures - Databricks
# =============================================================================


@pytest.fixture(scope="session")
def databricks_available() -> bool:
    """Check if Databricks is available."""
    from tests.integration.cloud_dw.backends import get_available_backends

    return "databricks" in get_available_backends()


@pytest.fixture(scope="session")
def databricks_backend(
    integration_config: IntegrationTestConfig,
    databricks_available: bool,
) -> Generator["CloudDWTestBackend | None", None, None]:
    """Create Databricks test backend."""
    if not databricks_available:
        pytest.skip("Databricks not available")
        yield None
        return

    from tests.integration.cloud_dw.backends import get_backend

    backend = get_backend("databricks", integration_config)
    backend.connect()

    yield backend

    backend.cleanup_test_data(force=True)
    backend.disconnect()


@pytest.fixture
def databricks_dataset(
    databricks_backend: "CloudDWTestBackend",
) -> Generator[TestDataset, None, None]:
    """Create a test dataset for Databricks."""
    if databricks_backend is None:
        pytest.skip("Databricks backend not available")

    dataset = databricks_backend.create_test_dataset()
    yield dataset


@pytest.fixture
def databricks_generator() -> TestDataGenerator:
    """Create data generator for Databricks."""
    return TestDataGenerator(SQLDialect.DATABRICKS)


# =============================================================================
# Fixtures - Generic (Multi-Backend)
# =============================================================================


@pytest.fixture(params=["bigquery", "snowflake", "redshift", "databricks"])
def any_backend(
    request: pytest.FixtureRequest,
    integration_config: IntegrationTestConfig,
    available_backends: list[str],
) -> Generator["CloudDWTestBackend | None", None, None]:
    """Parameterized fixture for all backends.

    This allows running the same test against all available backends.
    """
    backend_name = request.param

    if backend_name not in available_backends:
        pytest.skip(f"{backend_name} not available")
        yield None
        return

    from tests.integration.cloud_dw.backends import get_backend

    backend = get_backend(backend_name, integration_config)
    backend.connect()

    yield backend

    backend.cleanup_test_data(force=True)
    backend.disconnect()


@pytest.fixture
def any_dataset(
    any_backend: "CloudDWTestBackend",
) -> Generator[TestDataset, None, None]:
    """Create a test dataset for any backend."""
    if any_backend is None:
        pytest.skip("Backend not available")

    dataset = any_backend.create_test_dataset()
    yield dataset


# =============================================================================
# Fixtures - Test Data
# =============================================================================


@pytest.fixture
def users_data() -> list[dict[str, Any]]:
    """Generate standard users test data."""
    return StandardTestData.users_data(n=100)


@pytest.fixture
def products_data() -> list[dict[str, Any]]:
    """Generate standard products test data."""
    return StandardTestData.products_data(n=50)


@pytest.fixture
def transactions_data() -> list[dict[str, Any]]:
    """Generate standard transactions test data."""
    return StandardTestData.transactions_data(n=200)


@pytest.fixture
def nulls_data() -> list[dict[str, Any]]:
    """Generate data with null patterns."""
    return StandardTestData.nulls_data(n=100, null_ratio=0.2)


@pytest.fixture
def edge_cases_data() -> list[dict[str, Any]]:
    """Generate edge case test data."""
    return StandardTestData.edge_cases_data()


@pytest.fixture
def unicode_data() -> list[dict[str, Any]]:
    """Generate Unicode test data."""
    return StandardTestData.unicode_data()


# =============================================================================
# Fixtures - Test Runner
# =============================================================================


@pytest.fixture(scope="session")
def integration_runner(
    integration_config: IntegrationTestConfig,
) -> IntegrationTestRunner:
    """Create an integration test runner."""
    return IntegrationTestRunner(integration_config)


# =============================================================================
# Hooks
# =============================================================================


def pytest_terminal_summary(
    terminalreporter: Any,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Add integration test summary to terminal output."""
    # Check if we ran integration tests
    stats = terminalreporter.stats
    if not any(
        "integration" in str(getattr(item, "fspath", ""))
        for items in stats.values()
        for item in items
        if hasattr(item, "fspath")
    ):
        return

    terminalreporter.write_sep("=", "Integration Test Summary")

    # Report available backends
    try:
        from tests.integration.cloud_dw.backends import get_backend_availability_report

        report = get_backend_availability_report()
        terminalreporter.write_line("\nBackend Availability:")
        for name, status in report.items():
            symbol = "✓" if status["available"] else "✗"
            reason = "" if status["available"] else f" ({status.get('missing_env_vars', [])})"
            terminalreporter.write_line(f"  {symbol} {name}{reason}")
    except Exception:
        pass
