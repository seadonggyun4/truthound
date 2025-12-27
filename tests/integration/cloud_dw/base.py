"""Base classes for Cloud DW integration tests.

This module provides the abstract base classes and protocols that define
the interface for cloud data warehouse integration testing.

Design Principles:
    - Backend-agnostic: Same tests run against all platforms
    - Extensible: Easy to add new platforms
    - Safe: Cost controls and dry-run mode
    - Observable: Comprehensive metrics and logging
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterator,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    import polars as pl
    from truthound.datasources.sql.base import BaseSQLDataSource

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class ConnectionStatus(Enum):
    """Status of a cloud DW connection."""

    CONNECTED = auto()
    DISCONNECTED = auto()
    FAILED = auto()
    TIMEOUT = auto()
    AUTH_ERROR = auto()
    UNAVAILABLE = auto()


class TestDataType(Enum):
    """Types of test data for validation tests."""

    BASIC = "basic"  # Simple types: int, string, float, bool
    TEMPORAL = "temporal"  # Dates, timestamps, intervals
    COMPLEX = "complex"  # Arrays, structs, JSON
    NULLS = "nulls"  # Various null patterns
    EDGE_CASES = "edge_cases"  # Empty strings, min/max values
    UNICODE = "unicode"  # International characters
    LARGE_SCALE = "large_scale"  # Performance testing data


class TestCategory(Enum):
    """Categories of integration tests."""

    CONNECTION = "connection"  # Connection and auth tests
    SCHEMA = "schema"  # Schema inference tests
    QUERY = "query"  # Query execution tests
    VALIDATION = "validation"  # Data validation tests
    PERFORMANCE = "performance"  # Performance benchmarks
    COST = "cost"  # Cost estimation tests
    SECURITY = "security"  # Security-related tests


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class IntegrationTestConfig:
    """Configuration for integration tests.

    This configuration controls how integration tests behave,
    including timeouts, cost limits, and test data settings.

    Attributes:
        dry_run: If True, validate queries without executing them.
        max_cost_usd: Maximum allowed cost per test in USD.
        max_bytes: Maximum bytes to process per test.
        timeout_seconds: Query timeout in seconds.
        retry_attempts: Number of retry attempts for transient failures.
        retry_delay_seconds: Delay between retries.
        cleanup_on_failure: Whether to clean up test data on failure.
        parallel_tests: Maximum parallel test executions.
        log_queries: Whether to log executed queries.
        collect_metrics: Whether to collect performance metrics.
    """

    dry_run: bool = False
    max_cost_usd: float = 1.0
    max_bytes: int = 10 * 1024 * 1024 * 1024  # 10 GB
    timeout_seconds: int = 300
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    cleanup_on_failure: bool = True
    parallel_tests: int = 4
    log_queries: bool = True
    collect_metrics: bool = True

    # Test data settings
    test_dataset_prefix: str = "truthound_test"
    test_table_prefix: str = "test_"
    auto_cleanup: bool = True
    cleanup_after_hours: int = 24

    @classmethod
    def from_env(cls) -> "IntegrationTestConfig":
        """Create configuration from environment variables."""
        import os

        return cls(
            dry_run=os.getenv("TRUTHOUND_TEST_DRY_RUN", "false").lower() == "true",
            max_cost_usd=float(os.getenv("TRUTHOUND_TEST_MAX_COST_USD", "1.0")),
            max_bytes=int(os.getenv("TRUTHOUND_TEST_MAX_BYTES", str(10 * 1024**3))),
            timeout_seconds=int(os.getenv("TRUTHOUND_TEST_TIMEOUT", "300")),
            retry_attempts=int(os.getenv("TRUTHOUND_TEST_RETRIES", "3")),
            cleanup_on_failure=os.getenv("TRUTHOUND_TEST_CLEANUP", "true").lower()
            == "true",
            parallel_tests=int(os.getenv("TRUTHOUND_TEST_PARALLEL", "4")),
            log_queries=os.getenv("TRUTHOUND_TEST_LOG_QUERIES", "true").lower()
            == "true",
            collect_metrics=os.getenv("TRUTHOUND_TEST_METRICS", "true").lower()
            == "true",
        )


@dataclass
class TestMetrics:
    """Metrics collected during test execution.

    Attributes:
        query_count: Number of queries executed.
        total_bytes_processed: Total bytes processed.
        total_cost_usd: Estimated total cost in USD.
        total_duration_seconds: Total test duration.
        queries: List of query execution details.
    """

    query_count: int = 0
    total_bytes_processed: int = 0
    total_cost_usd: float = 0.0
    total_duration_seconds: float = 0.0
    queries: list[dict[str, Any]] = field(default_factory=list)

    def add_query(
        self,
        query: str,
        duration_seconds: float,
        bytes_processed: int = 0,
        cost_usd: float = 0.0,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Record a query execution."""
        self.query_count += 1
        self.total_bytes_processed += bytes_processed
        self.total_cost_usd += cost_usd
        self.total_duration_seconds += duration_seconds
        self.queries.append(
            {
                "query": query[:500],  # Truncate long queries
                "duration_seconds": duration_seconds,
                "bytes_processed": bytes_processed,
                "cost_usd": cost_usd,
                "success": success,
                "error": error,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "query_count": self.query_count,
            "total_bytes_processed": self.total_bytes_processed,
            "total_cost_usd": self.total_cost_usd,
            "total_duration_seconds": self.total_duration_seconds,
            "avg_query_duration": (
                self.total_duration_seconds / self.query_count
                if self.query_count > 0
                else 0
            ),
            "queries": self.queries,
        }


# =============================================================================
# Test Dataset
# =============================================================================


@dataclass
class TestDataset:
    """Represents a test dataset in a cloud DW.

    This class manages test data lifecycle including creation,
    population, and cleanup.

    Attributes:
        name: Dataset/schema name.
        tables: Dictionary of table names to their schemas.
        created_at: When the dataset was created.
        owner: Who created the dataset (for cleanup).
        metadata: Additional metadata.
    """

    name: str
    tables: dict[str, dict[str, str]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    owner: str = "integration_test"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def age_hours(self) -> float:
        """Get the age of this dataset in hours."""
        delta = datetime.utcnow() - self.created_at
        return delta.total_seconds() / 3600

    def is_stale(self, max_hours: int = 24) -> bool:
        """Check if this dataset is stale and should be cleaned up."""
        return self.age_hours > max_hours


@dataclass
class TestTable:
    """Represents a test table with schema and data.

    Attributes:
        name: Table name.
        schema: Column definitions {name: type}.
        row_count: Expected row count.
        data_type: Type of test data.
        created_at: When the table was created.
    """

    name: str
    schema: dict[str, str]
    row_count: int = 0
    data_type: TestDataType = TestDataType.BASIC
    created_at: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Credentials Protocol
# =============================================================================


@runtime_checkable
class CredentialsProtocol(Protocol):
    """Protocol for cloud DW credentials."""

    def validate(self) -> bool:
        """Validate that credentials are properly configured."""
        ...

    def get_connection_params(self) -> dict[str, Any]:
        """Get connection parameters."""
        ...

    @property
    def is_service_account(self) -> bool:
        """Check if using service account credentials."""
        ...


@dataclass
class BaseCredentials:
    """Base class for cloud DW credentials.

    Subclasses should implement platform-specific credential handling.
    """

    def validate(self) -> bool:
        """Validate credentials are present and properly formatted."""
        return True

    def get_connection_params(self) -> dict[str, Any]:
        """Get connection parameters dictionary."""
        return {}

    @property
    def is_service_account(self) -> bool:
        """Check if using service account credentials."""
        return False

    def mask_sensitive(self) -> dict[str, Any]:
        """Get credentials with sensitive values masked."""
        params = self.get_connection_params()
        masked = {}
        sensitive_keys = {"password", "secret", "token", "key", "credentials"}
        for key, value in params.items():
            if any(s in key.lower() for s in sensitive_keys):
                masked[key] = "***MASKED***"
            else:
                masked[key] = value
        return masked


# =============================================================================
# Abstract Test Backend
# =============================================================================


ConfigT = TypeVar("ConfigT", bound=BaseCredentials)


class CloudDWTestBackend(ABC, Generic[ConfigT]):
    """Abstract base class for cloud DW test backends.

    This class defines the interface that all cloud DW test backends
    must implement. It provides:
    - Connection management
    - Test data lifecycle management
    - Query execution with cost controls
    - Metrics collection

    Subclasses must implement platform-specific methods.

    Example:
        >>> class BigQueryTestBackend(CloudDWTestBackend[BigQueryCredentials]):
        ...     platform_name = "bigquery"
        ...
        ...     def _create_connection(self) -> Any:
        ...         from google.cloud import bigquery
        ...         return bigquery.Client(project=self.credentials.project)
    """

    # Class-level attributes
    platform_name: ClassVar[str] = "unknown"
    supports_dry_run: ClassVar[bool] = False
    supports_cost_estimation: ClassVar[bool] = False
    default_quote_char: ClassVar[str] = '"'

    def __init__(
        self,
        credentials: ConfigT,
        config: IntegrationTestConfig | None = None,
    ) -> None:
        """Initialize the test backend.

        Args:
            credentials: Platform-specific credentials.
            config: Test configuration. Uses defaults if not provided.
        """
        self.credentials = credentials
        self.config = config or IntegrationTestConfig()
        self.metrics = TestMetrics()
        self._connection: Any = None
        self._status = ConnectionStatus.DISCONNECTED
        self._datasets: list[TestDataset] = []

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    @property
    def status(self) -> ConnectionStatus:
        """Get current connection status."""
        return self._status

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._status == ConnectionStatus.CONNECTED

    def connect(self) -> bool:
        """Establish connection to the cloud DW.

        Returns:
            True if connection successful, False otherwise.
        """
        if self.is_connected:
            return True

        try:
            if not self.credentials.validate():
                self._status = ConnectionStatus.AUTH_ERROR
                logger.error(f"[{self.platform_name}] Invalid credentials")
                return False

            self._connection = self._create_connection()
            self._status = ConnectionStatus.CONNECTED
            logger.info(f"[{self.platform_name}] Connected successfully")
            return True

        except TimeoutError:
            self._status = ConnectionStatus.TIMEOUT
            logger.error(f"[{self.platform_name}] Connection timeout")
            return False
        except Exception as e:
            self._status = ConnectionStatus.FAILED
            logger.error(f"[{self.platform_name}] Connection failed: {e}")
            return False

    def disconnect(self) -> None:
        """Close connection to the cloud DW."""
        if self._connection is not None:
            try:
                self._close_connection()
            except Exception as e:
                logger.warning(f"[{self.platform_name}] Error closing connection: {e}")
            finally:
                self._connection = None
                self._status = ConnectionStatus.DISCONNECTED
                logger.info(f"[{self.platform_name}] Disconnected")

    @contextmanager
    def connection(self) -> Iterator[Any]:
        """Context manager for connection.

        Yields:
            The active connection object.

        Example:
            >>> with backend.connection() as conn:
            ...     result = conn.execute(query)
        """
        was_connected = self.is_connected
        try:
            if not was_connected:
                self.connect()
            yield self._connection
        finally:
            if not was_connected:
                self.disconnect()

    @abstractmethod
    def _create_connection(self) -> Any:
        """Create platform-specific connection.

        Returns:
            The connection object.
        """
        ...

    def _close_connection(self) -> None:
        """Close platform-specific connection.

        Override if special cleanup is needed.
        """
        if hasattr(self._connection, "close"):
            self._connection.close()

    # -------------------------------------------------------------------------
    # Query Execution
    # -------------------------------------------------------------------------

    def execute_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        dry_run: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a query with cost controls.

        Args:
            query: SQL query to execute.
            params: Query parameters for parameterized queries.
            dry_run: Override config dry_run setting.

        Returns:
            List of result rows as dictionaries.

        Raises:
            ValueError: If cost limits would be exceeded.
            RuntimeError: If query execution fails.
        """
        use_dry_run = dry_run if dry_run is not None else self.config.dry_run

        if self.config.log_queries:
            logger.debug(f"[{self.platform_name}] Query: {query[:200]}...")

        # Check cost before executing
        if self.supports_cost_estimation and not use_dry_run:
            estimate = self._estimate_query_cost(query)
            if estimate:
                if estimate.get("bytes", 0) > self.config.max_bytes:
                    raise ValueError(
                        f"Query would process {estimate['bytes']:,} bytes, "
                        f"exceeding limit of {self.config.max_bytes:,}"
                    )
                if estimate.get("cost_usd", 0) > self.config.max_cost_usd:
                    raise ValueError(
                        f"Query would cost ${estimate['cost_usd']:.4f}, "
                        f"exceeding limit of ${self.config.max_cost_usd:.2f}"
                    )

        start_time = time.time()
        try:
            if use_dry_run and self.supports_dry_run:
                result = self._dry_run_query(query, params)
            else:
                result = self._execute_query(query, params)

            duration = time.time() - start_time

            # Record metrics
            if self.config.collect_metrics:
                bytes_processed = self._get_last_query_bytes()
                cost_usd = self._get_last_query_cost()
                self.metrics.add_query(
                    query=query,
                    duration_seconds=duration,
                    bytes_processed=bytes_processed,
                    cost_usd=cost_usd,
                    success=True,
                )

            return result

        except Exception as e:
            duration = time.time() - start_time
            if self.config.collect_metrics:
                self.metrics.add_query(
                    query=query,
                    duration_seconds=duration,
                    success=False,
                    error=str(e),
                )
            raise

    @abstractmethod
    def _execute_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute query on the platform.

        Args:
            query: SQL query to execute.
            params: Query parameters.

        Returns:
            List of result rows.
        """
        ...

    def _dry_run_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform a dry run of the query.

        Override for platforms that support dry run.

        Returns:
            Empty list (dry run doesn't return data).
        """
        return []

    def _estimate_query_cost(self, query: str) -> dict[str, Any] | None:
        """Estimate query cost before execution.

        Override for platforms that support cost estimation.

        Returns:
            Dictionary with 'bytes' and 'cost_usd' keys, or None.
        """
        return None

    def _get_last_query_bytes(self) -> int:
        """Get bytes processed by the last query.

        Override for platforms that track this.
        """
        return 0

    def _get_last_query_cost(self) -> float:
        """Get estimated cost of the last query in USD.

        Override for platforms that track this.
        """
        return 0.0

    # -------------------------------------------------------------------------
    # Test Data Management
    # -------------------------------------------------------------------------

    def create_test_dataset(
        self,
        suffix: str | None = None,
    ) -> TestDataset:
        """Create a test dataset/schema.

        Args:
            suffix: Optional suffix for the dataset name.

        Returns:
            The created TestDataset.
        """
        import uuid

        name = f"{self.config.test_dataset_prefix}"
        if suffix:
            name = f"{name}_{suffix}"
        name = f"{name}_{uuid.uuid4().hex[:8]}"

        dataset = TestDataset(name=name)
        self._create_dataset(name)
        self._datasets.append(dataset)

        logger.info(f"[{self.platform_name}] Created test dataset: {name}")
        return dataset

    def create_test_table(
        self,
        dataset: TestDataset,
        table_name: str,
        schema: dict[str, str],
        data: list[dict[str, Any]] | None = None,
    ) -> TestTable:
        """Create a test table with optional data.

        Args:
            dataset: The dataset to create the table in.
            table_name: Name for the table.
            schema: Column definitions {name: type}.
            data: Optional data to insert.

        Returns:
            The created TestTable.
        """
        full_name = f"{self.config.test_table_prefix}{table_name}"
        self._create_table(dataset.name, full_name, schema)

        table = TestTable(
            name=full_name,
            schema=schema,
            row_count=len(data) if data else 0,
        )
        dataset.tables[full_name] = schema

        if data:
            self._insert_data(dataset.name, full_name, data)

        logger.info(
            f"[{self.platform_name}] Created test table: "
            f"{dataset.name}.{full_name} ({table.row_count} rows)"
        )
        return table

    def cleanup_test_data(self, force: bool = False) -> int:
        """Clean up all test datasets.

        Args:
            force: If True, clean up even if cleanup_on_failure is False.

        Returns:
            Number of datasets cleaned up.
        """
        if not force and not self.config.auto_cleanup:
            return 0

        cleaned = 0
        for dataset in self._datasets[:]:  # Copy list for safe iteration
            try:
                self._drop_dataset(dataset.name)
                self._datasets.remove(dataset)
                cleaned += 1
                logger.info(f"[{self.platform_name}] Cleaned up dataset: {dataset.name}")
            except Exception as e:
                logger.warning(
                    f"[{self.platform_name}] Failed to clean up {dataset.name}: {e}"
                )

        return cleaned

    def cleanup_stale_datasets(self, max_hours: int | None = None) -> int:
        """Clean up stale test datasets.

        Args:
            max_hours: Override default max age for stale datasets.

        Returns:
            Number of datasets cleaned up.
        """
        max_age = max_hours or self.config.cleanup_after_hours
        stale_datasets = self._find_stale_datasets(max_age)

        cleaned = 0
        for dataset_name in stale_datasets:
            try:
                self._drop_dataset(dataset_name)
                cleaned += 1
                logger.info(
                    f"[{self.platform_name}] Cleaned up stale dataset: {dataset_name}"
                )
            except Exception as e:
                logger.warning(
                    f"[{self.platform_name}] Failed to clean up {dataset_name}: {e}"
                )

        return cleaned

    @abstractmethod
    def _create_dataset(self, name: str) -> None:
        """Create a dataset/schema on the platform."""
        ...

    @abstractmethod
    def _drop_dataset(self, name: str) -> None:
        """Drop a dataset/schema from the platform."""
        ...

    @abstractmethod
    def _create_table(
        self,
        dataset: str,
        table: str,
        schema: dict[str, str],
    ) -> None:
        """Create a table on the platform."""
        ...

    @abstractmethod
    def _insert_data(
        self,
        dataset: str,
        table: str,
        data: list[dict[str, Any]],
    ) -> None:
        """Insert data into a table."""
        ...

    def _find_stale_datasets(self, max_hours: int) -> list[str]:
        """Find stale test datasets to clean up.

        Override for platforms that support dataset metadata queries.
        """
        return []

    # -------------------------------------------------------------------------
    # Schema Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_table_schema(
        self,
        dataset: str,
        table: str,
    ) -> dict[str, str]:
        """Get the schema of a table.

        Args:
            dataset: Dataset/schema name.
            table: Table name.

        Returns:
            Dictionary of column names to types.
        """
        ...

    @abstractmethod
    def get_row_count(
        self,
        dataset: str,
        table: str,
    ) -> int:
        """Get the row count of a table.

        Args:
            dataset: Dataset/schema name.
            table: Table name.

        Returns:
            Number of rows in the table.
        """
        ...

    # -------------------------------------------------------------------------
    # Truthound Integration
    # -------------------------------------------------------------------------

    @abstractmethod
    def create_datasource(
        self,
        dataset: str,
        table: str,
    ) -> "BaseSQLDataSource":
        """Create a Truthound DataSource for the table.

        Args:
            dataset: Dataset/schema name.
            table: Table name.

        Returns:
            A configured Truthound DataSource.
        """
        ...

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def quote_identifier(self, identifier: str) -> str:
        """Quote an identifier for the platform.

        Args:
            identifier: The identifier to quote.

        Returns:
            Quoted identifier.
        """
        escaped = identifier.replace(
            self.default_quote_char,
            self.default_quote_char + self.default_quote_char,
        )
        return f"{self.default_quote_char}{escaped}{self.default_quote_char}"

    def get_full_table_name(self, dataset: str, table: str) -> str:
        """Get the fully qualified table name.

        Args:
            dataset: Dataset/schema name.
            table: Table name.

        Returns:
            Fully qualified table name.
        """
        return f"{self.quote_identifier(dataset)}.{self.quote_identifier(table)}"

    def __enter__(self) -> "CloudDWTestBackend[ConfigT]":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        try:
            if self.config.cleanup_on_failure or exc_type is None:
                self.cleanup_test_data()
        finally:
            self.disconnect()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"platform={self.platform_name!r}, "
            f"status={self._status.name})"
        )


# =============================================================================
# Test Case Base Class
# =============================================================================


class CloudDWTestCase(ABC):
    """Base class for cloud DW test cases.

    This class provides a standard interface for writing integration tests
    that work across all supported cloud DW platforms.

    Example:
        >>> class TestDataValidation(CloudDWTestCase):
        ...     category = TestCategory.VALIDATION
        ...
        ...     def test_null_detection(self, backend, test_table):
        ...         source = backend.create_datasource(
        ...             test_table.dataset,
        ...             test_table.name
        ...         )
        ...         result = th.check(source, validators=[NotNullValidator("id")])
        ...         assert result.success
    """

    category: ClassVar[TestCategory] = TestCategory.VALIDATION
    requires_data: ClassVar[bool] = True
    data_types: ClassVar[list[TestDataType]] = [TestDataType.BASIC]

    @abstractmethod
    def run(
        self,
        backend: CloudDWTestBackend,
        dataset: TestDataset,
        table: TestTable | None = None,
    ) -> bool:
        """Run the test case.

        Args:
            backend: The test backend to use.
            dataset: The test dataset.
            table: The test table (if requires_data is True).

        Returns:
            True if test passed, False otherwise.
        """
        ...
