# Custom Data Source Development

This document covers how to create custom data sources in Truthound.

## Overview

Custom data sources allow you to integrate Truthound with any data backend. All data sources implement the `DataSourceProtocol` and typically extend `BaseDataSource` or `AsyncBaseDataSource`.

## Core Concepts

### DataSourceProtocol

The protocol that all data sources must implement:

```python
from truthound.datasources import DataSourceProtocol

class DataSourceProtocol(Protocol):
    name: str                              # Data source name
    source_type: str                       # Type identifier
    schema: dict[str, ColumnType]          # Column name to type mapping
    columns: list[str]                     # Column names
    row_count: int | None                  # Row count (if available)
    capabilities: set[DataSourceCapability]  # Supported capabilities

    def get_execution_engine(self) -> ExecutionEngineProtocol: ...
    def sample(self, n: int, seed: int) -> DataSourceProtocol: ...
    def validate_connection(self) -> bool: ...
```

### ColumnType

Unified type representation across all backends:

```python
from truthound.datasources import ColumnType

class ColumnType(Enum):
    # Numeric
    INTEGER = "integer"
    FLOAT = "float"
    DECIMAL = "decimal"

    # String
    STRING = "string"
    TEXT = "text"

    # Date/Time
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    DURATION = "duration"

    # Other
    BOOLEAN = "boolean"
    BINARY = "binary"
    LIST = "list"
    STRUCT = "struct"
    JSON = "json"
    NULL = "null"
    UNKNOWN = "unknown"
```

### DataSourceCapability

Capabilities that data sources can declare:

```python
from truthound.datasources import DataSourceCapability

class DataSourceCapability(Enum):
    LAZY_EVALUATION = "lazy_evaluation"    # Deferred execution
    SQL_PUSHDOWN = "sql_pushdown"          # Execute on server
    SAMPLING = "sampling"                  # Supports sampling
    STREAMING = "streaming"                # Streaming support
    SCHEMA_INFERENCE = "schema_inference"  # Auto-detect schema
    ROW_COUNT = "row_count"                # Efficient row count
```

## Creating a Sync Data Source

### Step 1: Define Configuration

```python
from dataclasses import dataclass
from truthound.datasources import DataSourceConfig

@dataclass
class MyDataSourceConfig(DataSourceConfig):
    """Configuration for my custom data source."""

    # Custom parameters
    api_url: str = "http://localhost:8080"
    api_key: str | None = None
    timeout: float = 30.0

    # Inherited from DataSourceConfig:
    # name: str | None = None
    # max_rows: int = 10_000_000
    # max_memory_mb: int = 4096
    # sample_size: int = 100_000
    # sample_seed: int | None = 42
    # cache_schema: bool = True
    # strict_types: bool = False
    # metadata: dict[str, Any] = field(default_factory=dict)
```

### Step 2: Implement the Data Source

```python
from typing import Any
import polars as pl
from truthound.datasources import (
    BaseDataSource,
    ColumnType,
    DataSourceCapability,
)
from truthound.execution import PolarsExecutionEngine

class MyDataSource(BaseDataSource[MyDataSourceConfig]):
    """Custom data source implementation."""

    source_type = "my_source"

    def __init__(
        self,
        api_url: str = "http://localhost:8080",
        api_key: str | None = None,
        config: MyDataSourceConfig | None = None,
    ) -> None:
        """Initialize the data source."""
        if config is None:
            config = MyDataSourceConfig(
                api_url=api_url,
                api_key=api_key,
            )
        super().__init__(config)
        self._data: pl.DataFrame | None = None

    @classmethod
    def _default_config(cls) -> MyDataSourceConfig:
        """Return default configuration."""
        return MyDataSourceConfig()

    @property
    def capabilities(self) -> set[DataSourceCapability]:
        """Declare supported capabilities."""
        return {
            DataSourceCapability.SCHEMA_INFERENCE,
            DataSourceCapability.SAMPLING,
        }

    @property
    def schema(self) -> dict[str, ColumnType]:
        """Get schema from cached data or fetch it."""
        if self._cached_schema is not None:
            return self._cached_schema

        # Fetch data if not cached
        if self._data is None:
            self._data = self._fetch_data()

        # Convert Polars schema to ColumnType
        from truthound.datasources.base import polars_to_column_type

        self._cached_schema = {
            col: polars_to_column_type(dtype)
            for col, dtype in self._data.schema.items()
        }
        return self._cached_schema

    @property
    def row_count(self) -> int | None:
        """Get row count."""
        if self._data is not None:
            return len(self._data)
        return self._cached_row_count

    def get_execution_engine(self):
        """Get execution engine."""
        lf = self.to_polars_lazyframe()
        return PolarsExecutionEngine(lf)

    def sample(self, n: int = 1000, seed: int | None = None) -> "MyDataSource":
        """Create a sampled data source."""
        if self._data is None:
            self._data = self._fetch_data()

        sampled_data = self._data.sample(n=min(n, len(self._data)), seed=seed)

        # Create new source with sampled data
        new_source = MyDataSource(config=self._config)
        new_source._data = sampled_data
        new_source._config.name = f"{self.name}_sample"
        return new_source

    def to_polars_lazyframe(self) -> pl.LazyFrame:
        """Convert to Polars LazyFrame."""
        if self._data is None:
            self._data = self._fetch_data()
        return self._data.lazy()

    def _fetch_data(self) -> pl.DataFrame:
        """Fetch data from the API."""
        import requests

        headers = {}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        response = requests.get(
            f"{self._config.api_url}/data",
            headers=headers,
            timeout=self._config.timeout,
        )
        response.raise_for_status()

        # Parse JSON response to DataFrame
        data = response.json()
        return pl.DataFrame(data)

    def validate_connection(self) -> bool:
        """Validate API connection."""
        import requests

        try:
            response = requests.get(
                f"{self._config.api_url}/health",
                timeout=self._config.timeout,
            )
            return response.status_code == 200
        except Exception:
            return False
```

### Step 3: Use the Data Source

```python
import truthound as th
from my_module import MyDataSource

# Create source
source = MyDataSource(
    api_url="http://api.example.com",
    api_key="secret",
)

# Use with validation API
report = th.check(
    source=source,
    validators=["null", "duplicate"],
)

# Or access directly
print(f"Schema: {source.schema}")
print(f"Rows: {source.row_count}")
lf = source.to_polars_lazyframe()
```

## Creating an Async Data Source

For I/O-bound operations, use async data sources.

### Step 1: Define Configuration

```python
from dataclasses import dataclass
from truthound.datasources import AsyncDataSourceConfig

@dataclass
class MyAsyncDataSourceConfig(AsyncDataSourceConfig):
    """Configuration for async data source."""

    api_url: str = "http://localhost:8080"
    api_key: str | None = None

    # Inherited from AsyncDataSourceConfig:
    # max_concurrent_requests: int = 10
    # connection_timeout: float = 30.0
    # query_timeout: float = 300.0
    # pool_size: int = 5
    # retry_attempts: int = 3
    # retry_delay: float = 1.0
    # retry_backoff: float = 2.0
```

### Step 2: Implement the Async Data Source

```python
from typing import Any
import polars as pl
from truthound.datasources import (
    AsyncBaseDataSource,
    ColumnType,
    DataSourceCapability,
)

class MyAsyncDataSource(AsyncBaseDataSource[MyAsyncDataSourceConfig]):
    """Async custom data source."""

    source_type = "my_async_source"

    def __init__(
        self,
        api_url: str = "http://localhost:8080",
        api_key: str | None = None,
        config: MyAsyncDataSourceConfig | None = None,
    ) -> None:
        if config is None:
            config = MyAsyncDataSourceConfig(
                api_url=api_url,
                api_key=api_key,
            )
        super().__init__(config)
        self._client: Any = None
        self._data: list[dict] | None = None

    @classmethod
    def _default_config(cls) -> MyAsyncDataSourceConfig:
        return MyAsyncDataSourceConfig()

    @property
    def capabilities(self) -> set[DataSourceCapability]:
        return {
            DataSourceCapability.SCHEMA_INFERENCE,
            DataSourceCapability.SAMPLING,
        }

    async def connect_async(self) -> None:
        """Establish async connection."""
        if self._is_connected:
            return

        import aiohttp
        self._client = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._config.connection_timeout),
        )
        self._is_connected = True

    async def disconnect_async(self) -> None:
        """Close async connection."""
        if not self._is_connected:
            return

        if self._client:
            await self._client.close()
            self._client = None
        self._is_connected = False

    async def validate_connection_async(self) -> bool:
        """Validate connection asynchronously."""
        try:
            if not self._is_connected:
                await self.connect_async()

            async with self._client.get(
                f"{self._config.api_url}/health"
            ) as response:
                return response.status == 200
        except Exception:
            return False

    async def get_schema_async(self) -> dict[str, ColumnType]:
        """Get schema asynchronously."""
        if self._cached_schema is not None:
            return self._cached_schema

        # Fetch sample data for schema inference
        data = await self._fetch_data_async(limit=100)

        # Infer schema from data
        from truthound.datasources.nosql.base import DocumentSchemaInferrer
        inferrer = DocumentSchemaInferrer()
        self._cached_schema = inferrer.infer_from_documents(data)

        return self._cached_schema

    async def get_row_count_async(self) -> int | None:
        """Get row count asynchronously."""
        if not self._is_connected:
            await self.connect_async()

        headers = self._get_headers()
        async with self._client.get(
            f"{self._config.api_url}/count",
            headers=headers,
        ) as response:
            data = await response.json()
            return data.get("count")

    async def sample_async(
        self, n: int = 1000, seed: int | None = None
    ) -> "MyAsyncDataSource":
        """Create sampled async data source."""
        data = await self._fetch_data_async(limit=n)

        new_source = MyAsyncDataSource(config=self._config)
        new_source._data = data
        new_source._config.name = f"{self.name}_sample"
        return new_source

    async def to_polars_lazyframe_async(self) -> pl.LazyFrame:
        """Convert to Polars LazyFrame asynchronously."""
        if self._data is None:
            self._data = await self._fetch_data_async()

        return pl.DataFrame(self._data).lazy()

    async def _fetch_data_async(self, limit: int | None = None) -> list[dict]:
        """Fetch data from API."""
        if not self._is_connected:
            await self.connect_async()

        headers = self._get_headers()
        params = {}
        if limit:
            params["limit"] = limit
        elif self._config.max_rows:
            params["limit"] = self._config.max_rows

        async with self._client.get(
            f"{self._config.api_url}/data",
            headers=headers,
            params=params,
        ) as response:
            return await response.json()

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers

    # Context manager support
    async def __aenter__(self) -> "MyAsyncDataSource":
        await self.connect_async()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect_async()
```

### Step 3: Use the Async Data Source

```python
import asyncio
import truthound as th
from my_module import MyAsyncDataSource

async def main():
    source = MyAsyncDataSource(
        api_url="http://api.example.com",
        api_key="secret",
    )

    async with source:
        # Get schema
        schema = await source.get_schema_async()
        print(f"Schema: {schema}")

        # Get LazyFrame
        lf = await source.to_polars_lazyframe_async()
        df = lf.collect()

        # Validate
        report = th.check(
            df,
            validators=["null", "duplicate"],
        )
        print(f"Issues: {len(report.issues)}")

asyncio.run(main())
```

## Extending SQL Data Sources

For SQL databases, extend `BaseSQLDataSource`:

```python
from dataclasses import dataclass
from typing import Any
from truthound.datasources.sql import (
    BaseSQLDataSource,
    SQLDataSourceConfig,
)

@dataclass
class MyDBConfig(SQLDataSourceConfig):
    """Configuration for custom SQL database."""

    host: str = "localhost"
    port: int = 5555
    database: str = "mydb"
    user: str = "user"
    password: str = ""

class MyDBDataSource(BaseSQLDataSource):
    """Custom SQL database data source."""

    source_type = "mydb"

    def __init__(
        self,
        table: str,
        host: str = "localhost",
        port: int = 5555,
        database: str = "mydb",
        user: str = "user",
        password: str = "",
        config: MyDBConfig | None = None,
    ) -> None:
        if config is None:
            config = MyDBConfig(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
            )
        super().__init__(table=table, config=config)

    @classmethod
    def _default_config(cls) -> MyDBConfig:
        return MyDBConfig()

    def _create_connection(self) -> Any:
        """Create database connection."""
        import mydb_driver  # Your database driver

        return mydb_driver.connect(
            host=self._config.host,
            port=self._config.port,
            database=self._config.database,
            user=self._config.user,
            password=self._config.password,
        )

    def _fetch_schema(self) -> list[tuple[str, str]]:
        """Fetch schema from database."""
        query = f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{self._table}'
            ORDER BY ordinal_position
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return cursor.fetchall()

    def _get_row_count_query(self) -> str:
        """Get row count query."""
        return f"SELECT COUNT(*) FROM {self._quote_identifier(self._table)}"

    def _quote_identifier(self, identifier: str) -> str:
        """Quote identifier for SQL."""
        escaped = identifier.replace('"', '""')
        return f'"{escaped}"'

    @property
    def full_table_name(self) -> str:
        """Get fully qualified table name."""
        return f'"{self._config.database}"."{self._table}"'
```

## Type Conversion Utilities

Use built-in utilities for type conversion:

```python
from truthound.datasources.base import (
    polars_to_column_type,
    pandas_dtype_to_column_type,
    sql_type_to_column_type,
)

# Polars dtype to ColumnType
import polars as pl
col_type = polars_to_column_type(pl.Int64)  # ColumnType.INTEGER

# Pandas dtype to ColumnType
import pandas as pd
col_type = pandas_dtype_to_column_type(pd.Series([1, 2, 3]).dtype)  # ColumnType.INTEGER

# SQL type string to ColumnType
col_type = sql_type_to_column_type("VARCHAR(255)")  # ColumnType.STRING
col_type = sql_type_to_column_type("TIMESTAMP")     # ColumnType.DATETIME
```

## Adapters

Convert between sync and async sources:

```python
from truthound.datasources import (
    adapt_to_async,
    adapt_to_sync,
    is_async_source,
    is_sync_source,
)

# Wrap sync source for async usage
sync_source = MyDataSource(api_url="http://example.com")
async_source = adapt_to_async(sync_source)

async with async_source:
    lf = await async_source.to_polars_lazyframe_async()

# Wrap async source for sync usage
async_source = MyAsyncDataSource(api_url="http://example.com")
sync_source = adapt_to_sync(async_source)

with sync_source:
    lf = sync_source.to_polars_lazyframe()

# Check source type
print(is_async_source(async_source))  # True
print(is_sync_source(sync_source))    # True
```

## Error Handling

Define custom exceptions:

```python
from truthound.datasources.base import (
    DataSourceError,
    DataSourceConnectionError,
    DataSourceSizeError,
    DataSourceSchemaError,
)

class MyDataSourceError(DataSourceError):
    """Base error for my data source."""
    pass

class MyConnectionError(DataSourceConnectionError):
    """Connection error for my data source."""

    def __init__(self, message: str, api_url: str | None = None) -> None:
        self.api_url = api_url
        super().__init__(source_type="my_source", message=message)
```

## Registration with Factory

Register your data source with the factory system:

```python
from truthound.datasources import get_datasource

# The factory uses type detection
# For custom types, check and create manually:

def get_my_datasource(data: Any, **kwargs):
    """Create data source with custom type detection."""
    if isinstance(data, str) and data.startswith("mydb://"):
        return MyDataSource.from_connection_string(data, **kwargs)
    return get_datasource(data, **kwargs)
```

## Best Practices

1. **Implement all abstract methods** - `schema`, `get_execution_engine`, `sample`, `to_polars_lazyframe`
2. **Cache schema** - Use `_cached_schema` to avoid repeated fetches
3. **Respect size limits** - Check `max_rows` and `max_memory_mb`
4. **Declare capabilities** - Override `capabilities` property accurately
5. **Support context managers** - Implement `__enter__`/`__exit__` for resource cleanup
6. **Use type conversion utilities** - Leverage built-in converters for consistent types
7. **Handle errors gracefully** - Use appropriate exception types
8. **Document configuration** - Add docstrings to config classes
