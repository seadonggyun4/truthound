# Truthound Architecture

This document provides a comprehensive overview of Truthound's internal architecture, design patterns, and extension points.

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Components](#2-core-components)
3. [Design Patterns](#3-design-patterns)
4. [Module Structure](#4-module-structure)
5. [Type System](#5-type-system)
6. [Extension Points](#6-extension-points)
7. [Testing Architecture](#7-testing-architecture)

---

## 1. System Overview

Truthound follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Interface Layer                            │
├─────────────────────────────────────────────────────────────────────────────┤
│   Python API (th.check, th.scan, th.compare)   │   CLI (truthound check)   │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Input Adapter Layer                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  pandas.DataFrame │ polars.DataFrame │ polars.LazyFrame │ dict │ CSV/JSON  │
│                              ↓                                               │
│                   Unified Polars LazyFrame                                   │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Validation Engine                                  │
├────────────────┬────────────────┬────────────────┬─────────────────────────┤
│   Validators   │ Drift Detectors│   PII Scanners │   Anomaly Detectors     │
│   (239 total)  │   (11 types)   │   (8 patterns) │     (13 methods)        │
└────────────────┴────────────────┴────────────────┴─────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Results & Storage Layer                             │
├─────────────────────────────────────────────────────────────────────────────┤
│    ValidationResult    │    ResultStore (5 backends)    │    Expectations   │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Output Layer                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│     Console (Rich)     │     JSON     │     HTML     │     Markdown         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 Design Principles

- **Lazy Evaluation**: Polars LazyFrame enables query optimization and memory-efficient processing
- **Single Collection Pattern**: Validators minimize `collect()` calls to reduce computational overhead
- **Protocol-Based Typing**: Optional dependencies use Protocol definitions for type safety without runtime imports
- **Factory Pattern**: Extensible creation of stores and reporters via registry-based factories
- **Modular Extensibility**: Base classes and mixins enable rapid development of specialized validators

---

## 2. Core Components

### 2.1 Validators (`src/truthound/validators/`)

Validators are the core building blocks for data quality checks.

```python
from truthound.validators.base import Validator

class MyValidator(Validator):
    name = "my_validator"

    def validate(self, lf: pl.LazyFrame) -> ValidatorResult:
        # Validation logic here
        pass
```

**Categories (21 total, 239 validators)**:
- Schema (14): Column existence, types, referential integrity
- Completeness (7): Null checks, completeness ratios
- Uniqueness (13): Duplicate detection, primary keys
- Distribution (15): Range, outlier, statistical tests
- String (17): Regex, email, phone, JSON schema
- Datetime (10): Format, range, freshness
- Aggregate (8): Mean, median, sum constraints
- Multi-column (16): Column comparisons, conditional rules
- Drift (11): KS test, PSI, Jensen-Shannon
- Anomaly (13): Isolation Forest, LOF, Mahalanobis
- And more...

### 2.2 Stores (`src/truthound/stores/`)

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

### 2.3 Reporters (`src/truthound/reporters/`)

Reporters generate output in various formats.

```python
from truthound.reporters import get_reporter

# Create reporter
reporter = get_reporter("html", title="Validation Report")

# Generate output
html = reporter.render(validation_result)
reporter.write(validation_result, "report.html")
```

**Available Formats**:
| Format | Package | Description |
|--------|---------|-------------|
| `json` | (built-in) | JSON format output |
| `console` | rich | Terminal output with formatting |
| `markdown` | (built-in) | Markdown format output |
| `html` | jinja2 | HTML pages with styling |

---

## 3. Design Patterns

### 3.1 Factory Pattern with Registry

Stores and reporters use a registry-based factory pattern for extensibility:

```python
# Registration via decorator
@register_store("my_backend")
class MyStore(BaseStore):
    pass

# Usage via factory
store = get_store("my_backend", **config)
```

### 3.2 Protocol-Based Optional Dependencies

Optional dependencies (boto3, sqlalchemy, jinja2) use Protocol definitions for type safety:

```python
# In _protocols.py
@runtime_checkable
class S3ClientProtocol(Protocol):
    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> dict[str, Any]: ...
    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]: ...

# In implementation
if TYPE_CHECKING:
    from ._protocols import S3ClientProtocol

class S3Store(ValidationStore):
    def __init__(self):
        self._client: S3ClientProtocol | None = None
```

This approach:
- Avoids `Any` type usage
- Provides IDE autocompletion
- Enables mock-based testing without real dependencies
- Has zero runtime cost (TYPE_CHECKING block)

### 3.3 Lazy Import Pattern

Optional packages are imported lazily to avoid ImportError:

```python
# Lazy import with flag
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

def _require_boto3() -> None:
    if not HAS_BOTO3:
        raise ImportError(
            "boto3 is required for S3Store. "
            "Install with: pip install truthound[s3]"
        )
```

### 3.4 Configuration Dataclasses

Each component uses typed configuration:

```python
@dataclass
class S3Config(StoreConfig):
    bucket: str = ""
    prefix: str = ""
    use_compression: bool = False
    compression_level: int = 6
```

---

## 4. Module Structure

```
src/truthound/
├── __init__.py              # Public API (th.check, th.scan, etc.)
├── _api.py                  # API implementation
├── _inputs.py               # Input adapter layer
│
├── validators/              # Validation logic
│   ├── __init__.py
│   ├── base.py              # Base Validator class
│   ├── schema.py            # Schema validators
│   ├── completeness.py      # Null/completeness validators
│   ├── uniqueness.py        # Uniqueness validators
│   ├── distribution.py      # Distribution validators
│   ├── string.py            # String validators
│   ├── datetime_validators.py
│   ├── aggregate.py
│   ├── multi_column.py
│   ├── drift.py             # Drift detection
│   ├── anomaly.py           # Anomaly detection
│   └── ...
│
├── stores/                  # Result persistence
│   ├── __init__.py
│   ├── base.py              # BaseStore, StoreConfig
│   ├── factory.py           # get_store(), register_store()
│   ├── results.py           # ValidationResult, ValidatorResult
│   ├── expectations.py      # Expectation, ExpectationSuite
│   └── backends/
│       ├── __init__.py
│       ├── _protocols.py    # Protocol definitions
│       ├── filesystem.py    # FileSystemStore
│       ├── memory.py        # MemoryStore
│       ├── s3.py            # S3Store
│       ├── gcs.py           # GCSStore
│       └── database.py      # DatabaseStore
│
├── reporters/               # Report generation
│   ├── __init__.py
│   ├── base.py              # BaseReporter, ReporterConfig
│   ├── factory.py           # get_reporter(), register_reporter()
│   ├── _protocols.py        # Jinja2 Protocol definitions
│   ├── json_reporter.py
│   ├── console_reporter.py
│   ├── markdown_reporter.py
│   └── html_reporter.py
│
└── schema/                  # Schema inference & caching
    ├── __init__.py
    ├── inference.py
    └── cache.py
```

---

## 5. Type System

### 5.1 Generic Types

Stores and reporters use generic types for type safety:

```python
T = TypeVar("T")  # Item type
C = TypeVar("C", bound="StoreConfig")  # Config type

class BaseStore(Generic[T, C], ABC):
    _config: C

    def save(self, item: T) -> str: ...
    def get(self, item_id: str) -> T: ...
```

### 5.2 Protocol Definitions

Optional dependency interfaces are defined as Protocols:

```python
# stores/backends/_protocols.py
@runtime_checkable
class S3ClientProtocol(Protocol):
    def head_bucket(self, *, Bucket: str) -> dict[str, Any]: ...
    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]: ...
    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ...) -> dict[str, Any]: ...
    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]: ...
    def delete_object(self, *, Bucket: str, Key: str) -> dict[str, Any]: ...
    def list_objects_v2(self, *, Bucket: str, Prefix: str = "") -> dict[str, Any]: ...

@runtime_checkable
class GCSClientProtocol(Protocol):
    def bucket(self, bucket_name: str) -> "GCSBucketProtocol": ...

@runtime_checkable
class SQLEngineProtocol(Protocol):
    def connect(self) -> Any: ...
    def dispose(self) -> None: ...
```

### 5.3 Result Types

```python
@dataclass
class ValidationResult:
    run_id: str
    run_time: datetime
    data_asset: str
    status: ResultStatus
    results: list[ValidatorResult]
    statistics: ResultStatistics
    tags: dict[str, str] = field(default_factory=dict)

@dataclass
class ValidatorResult:
    validator_name: str
    success: bool
    column: str | None = None
    issue_type: str | None = None
    count: int | None = None
    severity: str | None = None
    message: str | None = None
```

---

## 6. Extension Points

### 6.1 Custom Validators

```python
from truthound.validators.base import Validator, ValidatorResult

class CustomBusinessValidator(Validator):
    name = "custom_business"

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def validate(self, lf: pl.LazyFrame) -> ValidatorResult:
        # Your validation logic
        passed = self._check_business_rule(lf)
        return ValidatorResult(
            validator_name=self.name,
            success=passed,
            message="Business rule validation"
        )
```

### 6.2 Custom Store Backend

```python
from truthound.stores import register_store, BaseStore, StoreConfig

@dataclass
class RedisConfig(StoreConfig):
    host: str = "localhost"
    port: int = 6379
    db: int = 0

@register_store("redis")
class RedisStore(BaseStore[ValidationResult, RedisConfig]):
    def save(self, item: ValidationResult) -> str:
        # Implementation
        pass

    def get(self, item_id: str) -> ValidationResult:
        # Implementation
        pass
```

### 6.3 Custom Reporter

```python
from truthound.reporters import register_reporter, BaseReporter, ReporterConfig

@dataclass
class SlackReporterConfig(ReporterConfig):
    webhook_url: str = ""
    channel: str = "#data-quality"

@register_reporter("slack")
class SlackReporter(BaseReporter[SlackReporterConfig]):
    def render(self, data: ValidationResult) -> str:
        # Build Slack message
        pass

    def write(self, data: ValidationResult, path: str | Path) -> None:
        # Send to Slack
        pass
```

---

## 7. Testing Architecture

### 7.1 Mock-Based Testing

Optional dependencies are tested using comprehensive mocks:

```
tests/
├── mocks/
│   ├── __init__.py
│   ├── cloud_mocks.py      # MockS3Client, MockGCSClient
│   ├── database_mocks.py   # MockSQLEngine, MockSession
│   └── reporter_mocks.py   # MockJinja2Template
├── test_stores_optional_backends.py
└── test_reporters_optional.py
```

Mock implementations mirror real behavior:

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

### 7.2 Test Categories

| Category | Count | Description |
|----------|-------|-------------|
| Unit Tests | 106 | Core functionality |
| Validator Tests | 473 | All 239 validators |
| Integration Tests | 138 | End-to-end workflows |
| Mock Backend Tests | 31 | Optional dependency testing |
| **Total** | **748+** | All passing |

---

## Summary

Truthound's architecture prioritizes:

1. **Performance**: Polars LazyFrame for efficient data processing
2. **Type Safety**: Protocol-based optional dependencies, generics
3. **Extensibility**: Registry-based factories, plugin architecture
4. **Testability**: Comprehensive mocks for optional dependencies
5. **Modularity**: Clear separation between validators, stores, reporters

For specific component documentation, see:
- [Stores Documentation](STORES.md)
- [Reporters Documentation](REPORTERS.md)
- [Validator Reference](VALIDATORS.md)
