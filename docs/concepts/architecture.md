# Architecture Overview

This document provides a comprehensive overview of Truthound's internal architecture, design principles, and system structure.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [System Architecture](#2-system-architecture)
3. [Core Components](#3-core-components)
4. [Data Flow](#4-data-flow)
5. [Validator Framework](#5-validator-framework)
6. [Execution Model](#6-execution-model)
7. [Extension Points](#7-extension-points)
8. [Phase Overview](#8-phase-overview)
9. [Performance Architecture](#9-performance-architecture)
10. [Testing Architecture](#10-testing-architecture)

---

## 1. Design Philosophy

### Core Principles

| Principle | Description |
|-----------|-------------|
| **Zero Configuration** | Immediate usability with sensible defaults; no boilerplate required |
| **Performance First** | Polars LazyFrame architecture for efficient memory usage and computation |
| **Type Safety** | Strong typing throughout with comprehensive runtime validation |
| **Extensibility** | Modular architecture supporting custom validators, sources, and reporters |
| **Composability** | Components designed for combination and reuse |
| **Observability** | Rich output formats and detailed diagnostics |

### Architectural Constraints

1. **Polars Native**: All core operations implemented using Polars for consistent performance
2. **Lazy Evaluation**: Deferred computation until results are required
3. **Immutability**: Data structures are immutable where possible
4. **Protocol-Based**: Components interact through well-defined protocols
5. **Fail-Fast**: Validation errors raised immediately with clear context

---

## 2. System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Interface                                   │
│  ┌──────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │         Python API               │  │             CLI                 │  │
│  │  th.check() th.scan() th.compare │  │  truthound check data.csv       │  │
│  └─────────────────┬────────────────┘  └───────────────┬─────────────────┘  │
└────────────────────┼───────────────────────────────────┼────────────────────┘
                     │                                   │
                     └─────────────────┬─────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Input Layer                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        Data Source Factory                           │    │
│  │  Polars │ Pandas │ Spark │ SQL │ BigQuery │ Snowflake │ Files      │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Core Engine                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Schema    │  │  Validator  │  │    Drift    │  │     PII     │        │
│  │  Inference  │  │   Engine    │  │   Engine    │  │   Scanner   │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │                 │
│         └────────────────┼────────────────┼────────────────┘                 │
│                          ▼                ▼                                   │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                    Polars LazyFrame Processing                     │      │
│  └───────────────────────────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────────────────────┬────┘
                                                                         │
                                                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Output Layer                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                          Reporter Factory                            │    │
│  │     Console │ JSON │ HTML │ Markdown │ JUnit │ Stores (S3/GCS/DB)  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
src/truthound/
├── __init__.py              # Public API exports
├── cli.py                   # CLI interface (Typer)
├── types.py                 # Type definitions
├── core/                    # Core validation logic
├── validators/              # 289 validator implementations across 28 categories
│   ├── base.py              # Validator base classes
│   ├── schema/              # Schema validators (10)
│   ├── completeness/        # Completeness validators (5)
│   ├── uniqueness/          # Uniqueness validators (6)
│   ├── distribution/        # Distribution validators (7)
│   ├── string/              # String validators (9)
│   ├── datetime/            # Datetime validators (6)
│   ├── aggregate/           # Aggregate validators (5)
│   ├── multi_column/        # Multi-column validators (5)
│   ├── query/               # Query validators (6)
│   ├── table/               # Table validators (6)
│   ├── geospatial/          # Geospatial validators (5)
│   ├── drift/               # Drift validators (5)
│   ├── anomaly/             # Anomaly validators (4)
│   ├── privacy/             # Privacy validators (5)
│   ├── business_rule/       # Business validators (3)
│   ├── localization/        # Localization validators (4)
│   ├── ml_feature/          # ML feature validators (5)
│   ├── profiling/           # Profiling validators (4)
│   ├── referential/         # Referential validators (5)
│   ├── timeseries/          # Time series validators (6)
│   ├── cross_table/         # Cross-table validators (2)
│   ├── streaming/           # Streaming validators (5)
│   ├── memory/              # Memory validators (4)
│   ├── optimization/        # Optimization validators (6)
│   ├── i18n/                # i18n validators (10)
│   ├── timeout/             # Timeout validators (4)
│   ├── security/            # Security validators (2)
│   └── sdk/                 # SDK validators (4)
├── datasources/             # Data source adapters
│   ├── base.py              # DataSource protocol
│   ├── polars_source.py     # Polars adapter
│   ├── pandas_source.py     # Pandas adapter
│   ├── spark_source.py      # Spark adapter
│   └── sql/                 # SQL adapters (SQLite, PostgreSQL, BigQuery, etc.)
├── execution/               # Execution engines
├── profiler/                # Auto-profiling system
├── checkpoint/              # Checkpoint & CI/CD system
├── stores/                  # Result storage backends
├── reporters/               # Output formatters
├── datadocs/                # HTML report generation
├── plugins/                 # Plugin architecture
├── ml/                      # ML module (anomaly, drift, rule learning)
├── lineage/                 # Data lineage tracking
└── realtime/                # Streaming validation
```

---

## 3. Core Components

### 3.1 DataSource

The `DataSource` abstraction provides a unified interface for accessing data from various backends.

```python
from truthound.datasources import BaseDataSource

class DataSource(Protocol):
    """Protocol for data source implementations."""

    @property
    def schema(self) -> dict[str, ColumnType]:
        """Return column name to type mapping."""
        ...

    def to_lazyframe(self) -> pl.LazyFrame:
        """Convert to Polars LazyFrame for processing."""
        ...

    def get_execution_engine(self) -> ExecutionEngine:
        """Return execution engine for this source."""
        ...

    def needs_sampling(self) -> bool:
        """Check if data exceeds size limits."""
        ...

    def sample(self, n: int) -> DataSource:
        """Return sampled data source."""
        ...
```

### 3.2 ExecutionEngine

The `ExecutionEngine` handles actual validation operations with backend-specific optimizations.

```python
from truthound.execution import ExecutionEngine

class ExecutionEngine(Protocol):
    """Protocol for execution engine implementations."""

    def count_rows(self) -> int:
        """Return total row count."""
        ...

    def count_nulls(self, column: str) -> int:
        """Return null count for column."""
        ...

    def count_distinct(self, column: str) -> int:
        """Return distinct value count."""
        ...

    def get_stats(self, column: str) -> dict:
        """Return statistical summary for column."""
        ...

    def count_matching(self, condition: str) -> int:
        """Return count matching condition."""
        ...
```

### 3.3 Validator

All validators inherit from the `Validator` base class and implement the validation protocol.

```python
from truthound.validators.base import Validator

class Validator(ABC):
    """Base class for all validators."""

    name: str                    # Unique validator name
    category: str                # Validator category
    description: str             # Human-readable description

    @abstractmethod
    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Execute validation and return issues."""
        ...

    @abstractmethod
    def get_config(self) -> dict:
        """Return validator configuration."""
        ...
```

### 3.4 ValidationIssue

```python
@dataclass
class ValidationIssue:
    """Represents a single validation issue."""

    validator: str       # Validator that found the issue
    column: str | None   # Affected column (if applicable)
    severity: Severity   # low, medium, high, critical
    message: str         # Human-readable message
    details: dict        # Additional context
    row_indices: list[int] | None  # Affected rows (if available)
```

### 3.5 Reporter

Reporters transform validation results into various output formats.

```python
from truthound.reporters.base import ValidationReporter

class ValidationReporter(Protocol[C]):
    """Protocol for reporter implementations."""

    name: str                    # Reporter name
    file_extension: str          # Output file extension

    def render(self, data: ValidationResult) -> str:
        """Render result to string format."""
        ...

    def save(self, data: ValidationResult, path: Path) -> None:
        """Save result to file."""
        ...
```

### 3.6 Store

Stores persist validation results and expectations.

```python
from truthound.stores import get_store, ValidationResult

# Create store instance
store = get_store("filesystem", base_path=".truthound/results")

# Save result
result = ValidationResult.from_report(report, "customers.csv")
run_id = store.save(result)

# Retrieve result
result = store.get(run_id)
```

**Available Backends**:

| Backend | Package | Description |
|---------|---------|-------------|
| `filesystem` | (built-in) | Local filesystem storage |
| `memory` | (built-in) | In-memory storage for testing |
| `s3` | boto3 | AWS S3 storage |
| `gcs` | google-cloud-storage | Google Cloud Storage |
| `database` | sqlalchemy | SQL database storage |

---

## 4. Data Flow

### Validation Flow

```
Input                    Processing                     Output
─────                    ──────────                     ──────

Data Source         ┌─────────────────┐
(CSV, Parquet,  ───►│  Input Adapter  │
 DataFrame, SQL)    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  LazyFrame      │  (Polars lazy evaluation)
                    └────────┬────────┘
                             │
                    ┌────────┼────────┐
                    ▼        ▼        ▼
              ┌──────────┬──────────┬──────────┐
              │ Schema   │ Pattern  │ Statist- │
              │ Valid-   │ Valid-   │ ical     │  (Parallel execution)
              │ ators    │ ators    │ Valid.   │
              └────┬─────┴────┬─────┴────┬─────┘
                   │          │          │
                   └──────────┼──────────┘
                              ▼
                    ┌─────────────────┐
                    │ Issue Collector │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐            ┌──────────┐
                    │ ValidationResult│───────────►│ Reporter │───► Output
                    └─────────────────┘            └──────────┘
```

### Drift Detection Flow

```
Baseline Data           Current Data
     │                       │
     ▼                       ▼
┌─────────────┐        ┌─────────────┐
│  LazyFrame  │        │  LazyFrame  │
└──────┬──────┘        └──────┬──────┘
       │                      │
       └──────────┬───────────┘
                  ▼
         ┌─────────────────┐
         │ Column Sampling │  (Optional)
         └────────┬────────┘
                  │
         ┌────────┼────────┐
         ▼        ▼        ▼
   ┌──────────┬──────────┬──────────┐
   │  KS Test │ PSI      │ Chi-Sq   │  (Method selection)
   └────┬─────┴────┬─────┴────┬─────┘
        │          │          │
        └──────────┼──────────┘
                   ▼
          ┌─────────────────┐
          │  Drift Report   │
          └─────────────────┘
```

---

## 5. Validator Framework

### Validator Categories

Validators are organized into 28 categories based on their validation focus:

| Category | Count | Focus |
|----------|-------|-------|
| Schema | 15 | Column structure, types, relationships |
| Completeness | 12 | Null detection, required fields |
| Uniqueness | 17 | Duplicates, primary keys |
| Distribution | 15 | Range, outliers, statistics |
| String | 19 | Patterns, formats, encoding |
| Datetime | 10 | Format, range, sequence |
| Aggregate | 8 | Statistical constraints |
| Cross-table | 5 | Multi-table relationships |
| Multi-column | 21 | Column comparisons |
| Query | 20 | Expression-based validation |
| Table | 18 | Metadata, freshness |
| Geospatial | 13 | Coordinates, boundaries |
| Drift | 14 | Distribution changes |
| Anomaly | 18 | Outlier detection |
| Privacy | 16 | PII detection, GDPR/CCPA |
| Business Rule | 8 | Business rules (Luhn, IBAN) |
| Localization | 9 | Regional formats |
| ML Feature | 5 | Feature quality |
| Profiling | 7 | Data characteristics |
| Referential | 14 | Foreign key integrity |
| Time Series | 14 | Temporal patterns |
| Streaming | 12 | Stream validation |
| Memory | 8 | Memory-efficient validation |
| Optimization | 15 | DAG execution, profiling |
| SDK | 80 | Custom validator development |
| Security | 3 | ReDoS protection, SQL injection |
| i18n | 3 | Internationalized error messages |
| Timeout | - | Distributed timeout handling |

### Validator Registration

Validators are automatically registered using decorators:

```python
from truthound.validators.base import register_validator

@register_validator("null_check")
class NullCheckValidator(Validator):
    """Check for null values in columns."""

    name = "null_check"
    category = "completeness"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        # Implementation
        ...
```

### Validator Discovery

```python
from truthound.validators import get_validator, list_validators

# Get specific validator
validator = get_validator("null_check")

# List all validators
all_validators = list_validators()

# List by category
completeness_validators = list_validators(category="completeness")
```

---

## 6. Execution Model

### Lazy Evaluation

Truthound leverages Polars' lazy evaluation for efficient processing:

1. **Plan Construction**: Validation operations build a query plan
2. **Optimization**: Polars optimizes the plan (predicate pushdown, projection)
3. **Execution**: Plan executed only when results are collected

```python
# Query plan is built but not executed
lf = pl.scan_csv("large_file.csv")

# Validators add operations to plan
validator.validate(lf)

# Execution happens on collect()
issues = validator.validate(lf)  # Executes optimized plan
```

### Parallel Execution

Multiple validators can execute concurrently:

```python
from concurrent.futures import ThreadPoolExecutor

def run_validators(lf: pl.LazyFrame, validators: list[Validator]) -> list[ValidationIssue]:
    issues = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(v.validate, lf) for v in validators]
        for future in futures:
            issues.extend(future.result())
    return issues
```

### Memory Management

```python
from truthound.datasources.base import DataSourceConfig

config = DataSourceConfig(
    max_rows=10_000_000,      # Maximum rows before sampling
    max_memory_mb=4096,        # Memory threshold
    sample_size=100_000,       # Default sample size
    sample_seed=42,            # Reproducible sampling
)
```

---

## 7. Extension Points

### 7.1 Custom Validators

```python
from truthound.validators.base import Validator, register_validator

@register_validator("custom_check")
class CustomValidator(Validator):
    name = "custom_check"
    category = "custom"

    def __init__(self, column: str, threshold: float):
        self.column = column
        self.threshold = threshold

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        # Custom validation logic
        ...
```

### 7.2 Custom Data Sources

```python
from truthound.datasources import BaseDataSource, register_source

@register_source("custom")
class CustomDataSource(BaseDataSource):
    source_type = "custom"

    def to_lazyframe(self) -> pl.LazyFrame:
        # Convert custom format to LazyFrame
        ...
```

### 7.3 Custom Reporters

```python
from truthound.reporters import ValidationReporter, register_reporter

@register_reporter("xml")
class XMLReporter(ValidationReporter):
    name = "xml"
    file_extension = ".xml"

    def render(self, data: ValidationResult) -> str:
        # Render to XML format
        ...
```

### 7.4 Plugin System

The plugin architecture enables external extensions:

```python
from truthound.plugins import ValidatorPlugin, register_plugin

@register_plugin
class MyValidatorPlugin(ValidatorPlugin):
    def get_validators(self) -> list[type]:
        return [MyCustomValidator1, MyCustomValidator2]
```

See [Plugin Architecture](PLUGINS.md) for comprehensive plugin documentation.

---

## 8. Phase Overview

Truthound's development follows a phased approach:

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | Complete | Core validation engine with LazyFrame architecture |
| **Phase 2** | Complete | Advanced validators (aggregate, cross-table, drift, anomaly, privacy) |
| **Phase 3** | Complete | Extensibility (referential, time series, business, localization, ML) |
| **Phase 4** | Complete | Storage backends and reporters infrastructure |
| **Phase 5** | Complete | Multi-data source support (BigQuery, Snowflake, Databricks, etc.) |
| **Phase 6** | Complete | Checkpoint orchestration and CI/CD integration |
| **Phase 7** | Complete | Auto-profiling and rule generation |
| **Phase 8** | Complete | Data Docs (HTML report generation) |
| **Phase 9** | Complete | Plugin architecture |
| **Phase 10** | Complete | Advanced features (ML, Lineage, Realtime) |

### Feature Distribution

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         Truthound Feature Map                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1-3: Core Engine                                                     │
│  ├── 289 Validators across 28 categories                                   │
│  ├── Schema inference and learning                                          │
│  ├── Pattern detection (email, phone, credit card, etc.)                   │
│  └── Statistical validation (range, distribution, outliers)                │
│                                                                             │
│  Phase 4: Infrastructure                                                    │
│  ├── Storage backends (Filesystem, S3, GCS, Database)                      │
│  └── Reporters (Console, JSON, HTML, Markdown, JUnit)                      │
│                                                                             │
│  Phase 5: Multi-Source                                                      │
│  ├── DataFrame (Polars, Pandas, Spark)                                     │
│  ├── SQL (PostgreSQL, MySQL, SQLite)                                       │
│  └── Cloud DW (BigQuery, Snowflake, Redshift, Databricks)                 │
│                                                                             │
│  Phase 6: CI/CD                                                             │
│  ├── Checkpoint orchestration                                               │
│  ├── 12 CI platform support                                                 │
│  ├── Async execution                                                        │
│  └── Transaction management (Saga pattern)                                  │
│                                                                             │
│  Phase 7: Auto-Profiling                                                    │
│  ├── Statistical profiling                                                  │
│  ├── Pattern detection                                                      │
│  └── Rule generation                                                        │
│                                                                             │
│  Phase 8: Data Docs                                                         │
│  ├── HTML report generation                                                 │
│  ├── 5 themes, 4 chart libraries                                           │
│  └── Interactive dashboard (optional)                                       │
│                                                                             │
│  Phase 9: Plugin Architecture                                               │
│  ├── Validator plugins                                                      │
│  ├── Reporter plugins                                                       │
│  ├── DataSource plugins                                                     │
│  └── Hook system                                                            │
│                                                                             │
│  Phase 10: Advanced                                                         │
│  ├── ML Module (anomaly detection, drift, rule learning)                   │
│  ├── Lineage Module (graph, tracking, impact analysis)                     │
│  └── Realtime Module (streaming, incremental, checkpointing)              │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Performance Architecture

### Expression-Based Validator Architecture

Truthound implements an expression-based architecture that allows multiple validators to execute in a single `collect()` call.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Expression-Based Batch Execution                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    Validator 1          Validator 2          Validator 3                     │
│  (NullValidator)    (RangeValidator)    (CompletenessRatio)                 │
│         │                  │                    │                            │
│         ▼                  ▼                    ▼                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │
│  │ get_valida- │    │ get_valida- │    │ get_valida- │                      │
│  │ tion_exprs  │    │ tion_exprs  │    │ tion_exprs  │                      │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                      │
│         │                  │                  │                              │
│         └──────────────────┼──────────────────┘                              │
│                            ▼                                                 │
│                 ┌─────────────────────┐                                      │
│                 │  Expression Batch   │                                      │
│                 │     Executor        │                                      │
│                 └──────────┬──────────┘                                      │
│                            │                                                 │
│                            ▼                                                 │
│                 ┌─────────────────────┐                                      │
│                 │  lf.select([...])   │  ◄─── Single collect() call         │
│                 │     .collect()      │                                      │
│                 └──────────┬──────────┘                                      │
│                            │                                                 │
│                            ▼                                                 │
│                 ┌─────────────────────┐                                      │
│                 │   ValidationIssue   │                                      │
│                 │       Results       │                                      │
│                 └─────────────────────┘                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Components** (`src/truthound/validators/base.py`):

| Component | Description |
|-----------|-------------|
| `ValidationExpressionSpec` | Defines validation expression (count_expr, non_null_expr, severity thresholds) |
| `ExpressionValidatorMixin` | Mixin for single-validator expression-based execution |
| `ExpressionBatchExecutor` | Batches multiple validators into single collect() |

### Lazy Loading Architecture

The validator registry uses lazy loading to minimize startup time.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Lazy Loading Validator Registry                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    Application Start                                                         │
│          │                                                                   │
│          ▼                                                                   │
│    ┌─────────────────┐                                                       │
│    │ VALIDATOR_      │ ◄─── 200+ validators mapped to module paths          │
│    │ IMPORT_MAP      │      (not loaded yet)                                │
│    └────────┬────────┘                                                       │
│             │                                                                │
│             ▼                                                                │
│    ┌─────────────────┐                                                       │
│    │ get_validator() │ ◄─── User requests specific validator                │
│    └────────┬────────┘                                                       │
│             │                                                                │
│             ▼                                                                │
│    ┌─────────────────┐                                                       │
│    │ LazyValidator-  │ ◄─── On-demand import                                │
│    │ Loader          │                                                       │
│    └────────┬────────┘                                                       │
│             │                                                                │
│             ▼                                                                │
│    ┌─────────────────┐                                                       │
│    │ ValidatorImport │ ◄─── Metrics tracking (success/failure/timing)       │
│    │ Metrics         │                                                       │
│    └─────────────────┘                                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation** (`src/truthound/validators/_lazy.py`):
- `VALIDATOR_IMPORT_MAP`: 200+ validators mapped to their module paths
- `CATEGORY_MODULES`: 28 category modules for bulk loading
- `ValidatorImportMetrics`: Tracks import success/failure counts and timing

### Native Polars Optimizations

All data operations use native Polars expressions without Python callbacks.

| Operation | Pattern | File |
|-----------|---------|------|
| Masking (redact) | `pl.when/then/otherwise`, `str.replace_all()` | `maskers.py` |
| Masking (hash) | `pl.col().hash().cast(pl.String)` | `maskers.py` |
| Statistics | Single `select()` with all aggregations | `schema.py` |
| Validation | `count_expr`, `non_null_expr` expressions | `validators/base.py` |

### Cache Optimization

Cache fingerprinting uses xxhash for ~10x faster hashing.

```python
# Implementation in cache.py
def _fast_hash(content: str) -> str:
    if _HAS_XXHASH:
        return xxhash.xxh64(content.encode()).hexdigest()[:16]
    return hashlib.sha256(content.encode()).hexdigest()[:16]
```

### Report Optimization

Validation reports use heap-based sorting for O(1) most-severe-issue access.

```python
# Implementation in report.py
_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}

def add_issue(self, issue: ValidationIssue) -> None:
    heapq.heappush(
        self._issues_heap,
        (_SEVERITY_ORDER[issue.severity], self._heap_counter, issue),
    )
```

### Performance Summary

| Optimization | Location | Effect |
|-------------|----------|--------|
| Expression Batch Executor | `validators/base.py` | Multiple validators, single collect() |
| Lazy Loading Registry | `validators/_lazy.py` | 200+ validator on-demand loading |
| xxhash Cache | `cache.py` | ~10x faster fingerprinting |
| Native Polars Masking | `maskers.py` | No map_elements callbacks |
| Heap-Based Sorting | `report.py` | O(1) severity access |
| Batched Statistics | `schema.py` | Single select() for all stats |
| Streaming Mode | `maskers.py` | `engine="streaming"` for >1M rows |

---

## 10. Testing Architecture

### Design Patterns

#### Protocol-Based Optional Dependencies

Optional dependencies (boto3, sqlalchemy, jinja2) use Protocol definitions for type safety:

```python
# In _protocols.py
@runtime_checkable
class S3ClientProtocol(Protocol):
    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> dict[str, Any]: ...
    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]: ...
```

#### Mock-Based Testing

Optional dependencies are tested using comprehensive mocks:

```python
class MockS3Client:
    """In-memory S3 client mock."""

    def __init__(self) -> None:
        self._buckets: dict[str, dict[str, MockS3Object]] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ...) -> dict[str, Any]:
        self._buckets[Bucket][Key] = MockS3Object(body=Body)
        return {"ETag": "mock-etag"}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        obj = self._buckets[Bucket].get(Key)
        if not obj:
            raise MockS3ClientError("NoSuchKey", "Object not found")
        return {"Body": io.BytesIO(obj.body)}
```

### Test Categories

Note: Test counts change as the codebase evolves. Run `pytest --collect-only` for current counts.

| Category | Description |
|----------|-------------|
| Unit Tests | Core functionality |
| Validator Tests | Validator implementations |
| Integration Tests | End-to-end workflows |
| Mock Backend Tests | Optional dependency testing |
| E2E Tests | Complete pipeline tests |

---

## See Also

- [Getting Started](GETTING_STARTED.md) — Quick start guide
- [Validators Reference](VALIDATORS.md) — Complete validator documentation
- [Data Sources](DATASOURCES.md) — Data source adapters
- [Storage Backends](STORES.md) — Result persistence
- [Plugin Architecture](PLUGINS.md) — Extension system
- [API Reference](API_REFERENCE.md) — Complete API documentation
