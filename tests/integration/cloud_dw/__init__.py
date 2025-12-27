"""Cloud Data Warehouse Integration Tests.

This package provides a comprehensive, extensible framework for testing
Truthound's integration with various cloud data warehouses.

Supported Platforms:
    - Google BigQuery
    - Snowflake
    - AWS Redshift
    - Databricks
    - Oracle
    - SQL Server

Architecture:
    - Abstract test base classes for consistent testing across platforms
    - Backend-agnostic test fixtures and data generators
    - CI/CD integration with secret management
    - Parallel test execution support
    - Cost-aware test execution (dry-run mode)

Usage:
    # Run all integration tests
    pytest tests/integration/cloud_dw/ -v

    # Run specific platform tests
    pytest tests/integration/cloud_dw/ -v -m bigquery

    # Run with specific credentials
    BIGQUERY_PROJECT=my-project pytest tests/integration/cloud_dw/ -v -m bigquery

    # Dry-run mode (validates queries without execution)
    pytest tests/integration/cloud_dw/ -v --dry-run

Configuration:
    Tests are configured via environment variables or pytest.ini.
    See conftest.py for available options.
"""

from tests.integration.cloud_dw.base import (
    CloudDWTestBackend,
    IntegrationTestConfig,
    TestDataset,
    ConnectionStatus,
)
from tests.integration.cloud_dw.fixtures import (
    TestDataGenerator,
    StandardTestData,
)
from tests.integration.cloud_dw.runner import (
    IntegrationTestRunner,
    TestResult,
    TestSuite,
)

__all__ = [
    # Base classes
    "CloudDWTestBackend",
    "IntegrationTestConfig",
    "TestDataset",
    "ConnectionStatus",
    # Fixtures
    "TestDataGenerator",
    "StandardTestData",
    # Runner
    "IntegrationTestRunner",
    "TestResult",
    "TestSuite",
]
