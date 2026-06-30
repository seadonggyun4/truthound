# Data Sources 아키텍처

핵심 개념과 경계에서 Truthound, Phase을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 테이블 of Contents

1. [개요](#overview)
2. [아키텍처 Layers](#architecture-layers)
3. 핵심 개념과 경계에서 Design, Patterns을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
4. 핵심 개념과 경계에서 Type, System을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
5. 핵심 개념과 경계에서 Enterprise, Features을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
6. 핵심 개념과 경계에서 Extensibility, Guide을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
7. 핵심 개념과 경계에서 Testing, Strategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
8. 핵심 개념과 경계에서 Quality, Assessment을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

핵심 개념과 경계에서 Truthound, DataSource, ExecutionEngine을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Key Principles

| 핵심 개념과 경계에서 Principle을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Implementation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|----------------|
| 핵심 개념과 경계에서 Separation, Concerns을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 DataSource, ExecutionEngine을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 Duck, Typing, Safety을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 `@runtime_checkable`, Protocol을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 Lazy, Evaluation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 Backend, Agnostic을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Validators을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Supported Data Sources

```
In-Memory:   Polars, Pandas, Spark, Dict
File:        CSV, JSON, Parquet, NDJSON
RDBMS:       PostgreSQL, MySQL, SQLite, Oracle, SQL Server
Cloud DW:    BigQuery, Snowflake, Redshift, Databricks
```

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 아키텍처 Layers

### Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Protocol Layer                                │
│  DataSourceProtocol, ExecutionEngineProtocol, SQLDataSourceProtocol │
│                    (Duck Typing + Type Safety)                       │
├─────────────────────────────────────────────────────────────────────┤
│                     Abstract Base Layer                              │
│         BaseDataSource[ConfigT], BaseExecutionEngine[ConfigT]        │
│               (Common Functionality + Generic Types)                 │
├─────────────────────────────────────────────────────────────────────┤
│                    Specialized Base Layer                            │
│              BaseSQLDataSource, CloudDWDataSource                    │
│          (SQL-specific: Connection Pool, Query Building)             │
├─────────────────────────────────────────────────────────────────────┤
│                    Concrete Implementations                          │
│  PolarsDataSource, PandasDataSource, PostgreSQLDataSource, etc.     │
│                  (Backend-specific logic)                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

#### 1. Protocol Layer (`_protocols.py`)

핵심 개념과 경계에서 `Protocol`, Defines, Python, Protocol을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
@runtime_checkable
class DataSourceProtocol(Protocol):
    """Protocol defining the interface for all data sources."""

    @property
    def name(self) -> str: ...

    @property
    def schema(self) -> dict[str, ColumnType]: ...

    def get_execution_engine(self) -> ExecutionEngineProtocol: ...

    def sample(self, n: int = 1000, seed: int | None = None) -> DataSourceProtocol: ...
```

핵심 개념과 경계에서 Benefits을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 핵심 개념과 경계에서 Enables을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 핵심 개념과 경계에서 IDE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

#### 2. Abstract Base Layer (`base.py`)

핵심 개념과 경계에서 Provides, ABC, Generic을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
ConfigT = TypeVar("ConfigT", bound=DataSourceConfig)

class BaseDataSource(ABC, Generic[ConfigT]):
    """Abstract base class for all data sources."""

    source_type: str = "base"

    def __init__(self, config: ConfigT | None = None) -> None:
        self._config = config or self._default_config()
        self._cached_schema: dict[str, ColumnType] | None = None
```

핵심 개념과 경계에서 Provided, Utilities을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 핵심 개념과 경계에서 `check_size_limits()`, `needs_sampling()`, Size을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 컬럼 type helpers (`get_numeric_columns()`, `get_string_columns()`)
- 핵심 개념과 경계에서 Context을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 스키마 caching

#### 3. Specialized Base Layer (`sql/base.py`, `sql/cloud_base.py`)

핵심 개념과 경계에서 SQL, Adds을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
class BaseSQLDataSource(BaseDataSource[SQLDataSourceConfig]):
    """Abstract base for SQL-based data sources."""

    # Connection pooling
    _pool: SQLConnectionPool | None = None

    # SQL query builders
    def build_count_query(self, condition: str | None = None) -> str
    def build_distinct_count_query(self, column: str) -> str
    def build_null_count_query(self, column: str) -> str
```

#### 4. Concrete Implementations

핵심 개념과 경계에서 Backend-specific을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
class PostgreSQLDataSource(BaseSQLDataSource):
    source_type = "postgresql"

    def _create_connection(self) -> Any:
        import psycopg2
        return psycopg2.connect(...)

    def _quote_identifier(self, identifier: str) -> str:
        return f'"{identifier}"'
```

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Design Patterns

### 1. Strategy Pattern

핵심 개념과 경계에서 ExecutionEngines을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
# Same interface, different implementations
polars_engine: ExecutionEngineProtocol = PolarsExecutionEngine(lf)
pandas_engine: ExecutionEngineProtocol = PandasExecutionEngine(df)
sql_engine: ExecutionEngineProtocol = SQLExecutionEngine(source)

# All support the same operations
for engine in [polars_engine, pandas_engine, sql_engine]:
    print(engine.count_rows())
    print(engine.count_nulls("column"))
```

### 2. Factory Pattern

핵심 개념과 경계에서 Factory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
def get_datasource(data: Any, **kwargs) -> DataSourceProtocol:
    """Auto-detect and create appropriate data source."""

    if _is_polars_dataframe(data):
        return PolarsDataSource(data, ...)
    if _is_pandas_dataframe(data):
        return PandasDataSource(data, ...)
    if _is_spark_dataframe(data):
        return SparkDataSource(data, ...)
    # ... more detection logic
```

### 3. Abstract Factory Pattern

핵심 개념과 경계에서 DataSources, ExecutionEngines을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
class PolarsDataSource(BaseDataSource):
    def get_execution_engine(self) -> PolarsExecutionEngine:
        return PolarsExecutionEngine(self._lf)

class PostgreSQLDataSource(BaseSQLDataSource):
    def get_execution_engine(self) -> SQLExecutionEngine:
        return SQLExecutionEngine(self)
```

### 4. Object Pool Pattern

핵심 개념과 경계에서 SQL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
class SQLConnectionPool:
    """Thread-safe connection pool for SQL databases."""

    def __init__(self, connection_factory, size=5, timeout=30.0):
        self._pool: Queue = Queue(maxsize=size)
        self._lock = Lock()
        self._factory = connection_factory

    @contextmanager
    def acquire(self) -> Iterator[Any]:
        """Acquire a connection from the pool."""
        conn = self._pool.get(timeout=self._timeout)
        try:
            yield conn
        finally:
            self._pool.put(conn)
```

### 5. Template Method Pattern

핵심 개념과 경계에서 Base을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
class BaseSQLDataSource(ABC):
    # Template method - uses abstract methods
    def execute_query(self, query: str) -> list[dict]:
        with self._get_connection() as conn:  # Uses _create_connection()
            cursor = conn.cursor()
            cursor.execute(query)
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    # Abstract methods - implemented by subclasses
    @abstractmethod
    def _create_connection(self) -> Any: ...

    @abstractmethod
    def _quote_identifier(self, identifier: str) -> str: ...
```

### 6. 어댑터 Pattern

핵심 개념과 경계에서 Polars, Converts을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
class PandasDataSource(BaseDataSource):
    def to_polars_lazyframe(self) -> pl.LazyFrame:
        """Adapt Pandas DataFrame to Polars LazyFrame."""
        return pl.from_pandas(self._df).lazy()

class SparkDataSource(BaseDataSource):
    def to_polars_lazyframe(self) -> pl.LazyFrame:
        """Adapt Spark DataFrame to Polars LazyFrame."""
        pandas_df = self._spark_df.toPandas()
        return pl.from_pandas(pandas_df).lazy()
```

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Type System

### Unified 컬럼 Types

핵심 개념과 경계에서 `ColumnType`, ColumnType을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
class ColumnType(Enum):
    # Numeric
    INTEGER = "integer"
    FLOAT = "float"
    DECIMAL = "decimal"

    # String
    STRING = "string"
    TEXT = "text"

    # Temporal
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

### Type Conversion Functions

```python
# Polars -> ColumnType
def polars_to_column_type(polars_dtype: Any) -> ColumnType:
    dtype_name = type(polars_dtype).__name__
    if dtype_name in ("Int8", "Int16", "Int32", "Int64"):
        return ColumnType.INTEGER
    # ...

# Pandas -> ColumnType
def pandas_dtype_to_column_type(pandas_dtype: Any) -> ColumnType:
    dtype_str = str(pandas_dtype).lower()
    if "int" in dtype_str:
        return ColumnType.INTEGER
    # ...

# SQL -> ColumnType
def sql_type_to_column_type(sql_type: str) -> ColumnType:
    sql_upper = sql_type.upper()
    if any(t in sql_upper for t in ("INT", "SERIAL", "BIGINT")):
        return ColumnType.INTEGER
    # ...
```

### Generic 설정 Types

```python
ConfigT = TypeVar("ConfigT", bound=DataSourceConfig)

class BaseDataSource(ABC, Generic[ConfigT]):
    def __init__(self, config: ConfigT | None = None):
        self._config: ConfigT = config or self._default_config()
```

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

```python
class BigQueryDataSource(CloudDWDataSource):
    _config: BigQueryConfig  # Type narrowing

    def query(self):
        # IDE knows _config has BigQuery-specific fields
        if self._config.maximum_bytes_billed:
            ...
```

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Enterprise Features

### 1. Connection Pooling

핵심 개념과 경계에서 Thread-safe을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
config = SQLDataSourceConfig(
    pool_size=5,           # Max connections
    pool_timeout=30.0,     # Acquire timeout
    query_timeout=300.0,   # Query timeout
)
```

### 2. Cloud DW Cost Control

핵심 개념과 경계에서 BigQuery을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
class BigQueryDataSource(CloudDWDataSource):
    def _get_cost_estimate(self, query: str) -> dict[str, Any]:
        """Estimate query cost using dry run."""
        job_config = bigquery.QueryJobConfig(dry_run=True)
        query_job = client.query(query, job_config=job_config)

        bytes_processed = query_job.total_bytes_processed
        cost_per_byte = 5.0 / (1024**4)  # $5 per TB

        return {
            "bytes_processed": bytes_processed,
            "estimated_cost_usd": bytes_processed * cost_per_byte,
        }

# Usage with limits
source.execute_with_cost_check(
    "SELECT * FROM large_table",
    max_bytes=1_000_000_000,  # 1GB limit
    max_cost_usd=1.0,          # $1 limit
)
```

### 3. Multiple Authentication Methods

#### Snowflake
```python
# Password
SnowflakeDataSource(user="...", password="...")

# SSO
SnowflakeDataSource(authenticator="externalbrowser")

# Key-pair
SnowflakeDataSource(private_key_path="/path/to/key.pem")
```

#### BigQuery
```python
# Service account file
BigQueryDataSource(credentials_path="/path/to/sa.json")

# Service account dict
BigQueryDataSource(credentials_dict={...})

# Application Default Credentials
BigQueryDataSource(project="my-project")  # Uses ADC
```

### 4. SQL Dialect Handling

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

| 데이터베이스 | 핵심 개념과 경계에서 Quote, Character을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Example을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-----------------|---------|
| 핵심 개념과 경계에서 PostgreSQL, SQL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 `"`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 `"column_name"`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 MySQL, SQL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 ` `을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 ` `, ` `을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 SQL, Server을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 `[]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 `[column_name]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 BigQuery을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 ` `을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 ` `, ` `을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 Oracle을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 `"`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 `"COLUMN_NAME"`, COLUMN_NAME을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### 5. Graceful Dependency Management

핵심 개념과 경계에서 Optional을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
# sql/__init__.py
try:
    from truthound.datasources.sql.bigquery import BigQueryDataSource
except ImportError:
    BigQueryDataSource = None  # type: ignore

def check_source_available(source_type: str) -> bool:
    """Check if a specific SQL source type is available."""
    sources = get_available_sources()
    return sources.get(source_type) is not None
```

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Extensibility Guide

### Adding a New Data 소스

핵심 개념과 경계에서 SQL, CockroachDB을(를) 다루는 항목입니다:

```python
# src/truthound/datasources/sql/cockroachdb.py

from truthound.datasources.sql.base import BaseSQLDataSource, SQLDataSourceConfig

class CockroachDBDataSource(BaseSQLDataSource):
    source_type = "cockroachdb"

    def __init__(self, table: str, host: str, database: str, **kwargs):
        super().__init__(table=table, config=SQLDataSourceConfig(**kwargs))
        self._host = host
        self._database = database

    def _create_connection(self) -> Any:
        import psycopg2
        return psycopg2.connect(
            host=self._host,
            database=self._database,
            sslmode="require",
        )

    def _get_table_schema_query(self) -> str:
        return f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{self._table}'
        """

    def _get_row_count_query(self) -> str:
        return f"SELECT COUNT(*) FROM {self.full_table_name}"

    def _quote_identifier(self, identifier: str) -> str:
        return f'"{identifier}"'  # PostgreSQL-compatible
```

### Required Abstract Methods

| 핵심 개념과 경계에서 Method을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Purpose을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|---------|
| 핵심 개념과 경계에서 `_create_connection()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Create 데이터베이스 connection |
| 핵심 개념과 경계에서 `_get_table_schema_query()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | SQL to retrieve 컬럼 info |
| 핵심 개념과 경계에서 `_get_row_count_query()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 SQL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 `_quote_identifier()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Quote을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Adding a New Execution 엔진

```python
class DuckDBExecutionEngine(BaseExecutionEngine):
    engine_type = "duckdb"

    def __init__(self, connection):
        super().__init__()
        self._conn = connection

    def count_rows(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM data").fetchone()[0]

    def count_nulls(self, column: str) -> int:
        return self._conn.execute(
            f"SELECT COUNT(*) FROM data WHERE {column} IS NULL"
        ).fetchone()[0]

    # ... implement remaining abstract methods
```

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Testing Strategy

### Test Categories

| 핵심 개념과 경계에서 Category을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 파일 | 핵심 개념과 경계에서 Tests을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Coverage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|------|-------|----------|
| 핵심 개념과 경계에서 Core, DataSources을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 `test_datasources.py`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Polars, Pandas, 파일, Dict |
| 핵심 개념과 경계에서 Execution, Engines을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 `test_execution_engines.py`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Polars, Pandas을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 SQL, DataSources을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 `test_datasources_sql.py`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 PostgreSQL, SQLite, MySQL, SQL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 Enterprise을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 `test_datasources_enterprise.py`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Cloud을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Testing Approach

#### 1. Unit Tests
핵심 개념과 경계에서 DataSource, Engine을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
def test_polars_source_count_rows(sample_polars_df):
    source = PolarsDataSource(sample_polars_df)
    engine = source.get_execution_engine()
    assert engine.count_rows() == 5
```

#### 2. Cross-엔진 Comparison
핵심 개념과 경계에서 Polars, Verify, Pandas을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
class TestEngineComparison:
    def test_count_rows_match(self, polars_engine, pandas_engine):
        assert polars_engine.count_rows() == pandas_engine.count_rows()

    def test_stats_match(self, polars_engine, pandas_engine):
        polars_stats = polars_engine.get_stats("salary")
        pandas_stats = pandas_engine.get_stats("salary")
        assert polars_stats["mean"] == pytest.approx(pandas_stats["mean"])
```

#### 3. Mock Tests for Cloud DW
핵심 개념과 경계에서 Test을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
class MockCloudSource(CloudDWDataSource):
    def _get_cost_estimate(self, query):
        return {"bytes_processed": 1000000, "estimated_cost_usd": 5.0}

def test_cost_limit_exceeded():
    source = MockCloudSource()
    with pytest.raises(DataSourceError, match="exceeds limit"):
        source.execute_with_cost_check(query, max_cost_usd=1.0)
```

#### 4. Factory Tests
핵심 개념과 경계에서 Verify을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
def test_detect_polars_dataframe(sample_polars_df):
    source = get_datasource(sample_polars_df)
    assert isinstance(source, PolarsDataSource)

def test_detect_sql_connection_string():
    source_type = detect_datasource_type("postgresql://user@host/db")
    assert source_type == "postgresql"
```

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Quality Assessment

### 메트릭 Summary

| 핵심 개념과 경계에서 Metric을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Score을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Notes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------|-------|
| 핵심 개념과 경계에서 Abstraction, Quality을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Clean, Protocol, ABC을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 Extensibility을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 New을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 Maintainability을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Consistent을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 Test, Coverage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Spark을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 Documentation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Comprehensive을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 Overall을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Enterprise-ready을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### SOLID Principles Compliance

| 핵심 개념과 경계에서 Principle을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Status을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Implementation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|--------|----------------|
| 핵심 개념과 경계에서 Responsibility을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Pass을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 DataSource, ExecutionEngine을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 Closed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Pass을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 New을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 Substitution을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Pass을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Protocol을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 Segregation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Pass을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 SQL, Separate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 핵심 개념과 경계에서 Inversion을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Pass을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 핵심 개념과 경계에서 Validators, Protocol을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Strengths

1. 핵심 개념과 경계에서 Phase을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. 핵심 개념과 경계에서 Enterprise, Connection, IAM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. 핵심 개념과 경계에서 Type, Generic을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
4. 핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
5. 핵심 개념과 경계에서 Graceful을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Minor Improvement Areas

1. 핵심 개념과 경계에서 SQL, Injection, Some을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. 핵심 개념과 경계에서 Spark, Limited을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. 핵심 개념과 경계에서 SQL, Not을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 파일 Structure

```
src/truthound/
├── datasources/
│   ├── __init__.py           # Public exports
│   ├── _protocols.py         # Protocol definitions
│   ├── base.py               # BaseDataSource, utilities
│   ├── factory.py            # get_datasource(), auto-detection
│   ├── polars_source.py      # Polars, File, Dict sources
│   ├── pandas_source.py      # Pandas source
│   ├── spark_source.py       # Spark source
│   └── sql/
│       ├── __init__.py       # SQL exports, availability checks
│       ├── base.py           # BaseSQLDataSource, connection pool
│       ├── cloud_base.py     # CloudDWDataSource, cost control
│       ├── sqlite.py         # SQLite
│       ├── postgresql.py     # PostgreSQL
│       ├── mysql.py          # MySQL
│       ├── oracle.py         # Oracle
│       ├── sqlserver.py      # SQL Server
│       ├── bigquery.py       # BigQuery
│       ├── snowflake.py      # Snowflake
│       ├── redshift.py       # Redshift
│       └── databricks.py     # Databricks
│
├── execution/
│   ├── __init__.py           # Public exports
│   ├── _protocols.py         # ExecutionEngineProtocol
│   ├── base.py               # BaseExecutionEngine
│   ├── polars_engine.py      # Polars engine
│   ├── pandas_engine.py      # Pandas engine
│   └── sql_engine.py         # SQL engine with pushdown
```

핵심 개념과 경계에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 레퍼런스s

- 핵심 개념과 경계에서 Data, Sources, Guide, Usage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 핵심 개념과 경계에서 Architecture, Overview, Overall을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 핵심 개념과 경계에서 Python, Protocols, PEP을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 핵심 개념과 경계에서 Polars, Documentation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
