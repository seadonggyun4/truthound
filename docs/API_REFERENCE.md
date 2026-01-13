# API Reference

This document provides a complete reference for the Truthound Python API.

---

## Table of Contents

1. [Core Functions](#1-core-functions)
2. [Schema Module](#2-schema-module)
3. [Validator Interface](#3-validator-interface)
4. [Data Sources](#4-data-sources)
5. [Storage Backends](#5-storage-backends)
6. [Reporters](#6-reporters)
7. [Profiler](#7-profiler)
8. [Data Docs](#8-data-docs)
9. [ML Module](#9-ml-module)
10. [CLI Reference](#10-cli-reference)
11. [Performance APIs](#11-performance-apis)

---

## 1. Core Functions

### th.check()

Validates data against rules and returns a validation report.

```python
def check(
    data: DataInput = None,
    source: BaseDataSource | None = None,
    *,
    schema: Schema | str | Path | None = None,
    validators: list[str | Validator] | None = None,
    columns: list[str] | None = None,
    min_severity: str | Severity = "low",
    strict: bool = False,
    auto_schema: bool = False,
    parallel: bool = False,
    max_workers: int | None = None,
    pushdown: bool | None = None,
) -> ValidationReport:
    """
    Validate data against rules.

    Args:
        data: Input data (DataFrame, file path, dict, etc.)
        source: DataSource instance for SQL databases, Spark, etc.
                If provided, data argument is ignored.
        schema: Schema to validate against
        validators: Specific validators to run
        columns: Columns to validate (None = all)
        min_severity: Minimum severity to report
        strict: Raise exception on failures
        auto_schema: Enable automatic schema caching
        parallel: Use DAG-based parallel execution
        max_workers: Max threads for parallel execution
        pushdown: Enable query pushdown for SQL sources

    Returns:
        ValidationReport with issues and statistics
    """
```

**Supported Input Types**:
- `polars.DataFrame`
- `polars.LazyFrame`
- `pandas.DataFrame`
- `str` or `Path` (CSV, Parquet, JSON files)
- `dict` (column name to values mapping)
- `BaseDataSource` (via `source` parameter)

### th.learn()

Learns a schema from data.

```python
def learn(
    data: DataInput,
    *,
    infer_constraints: bool = True,
    categorical_threshold: int = 20,
    sample_size: int | None = None,
) -> Schema:
    """
    Learn schema from data.

    Args:
        data: Input data
        infer_constraints: Infer min/max, allowed values
        categorical_threshold: Max unique values for categorical detection
        sample_size: Sample size for large datasets

    Returns:
        Inferred Schema object
    """
```

### th.compare()

Compares two datasets for drift.

```python
def compare(
    baseline: DataInput,
    current: DataInput,
    *,
    method: str = "auto",
    columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
    sample_size: int | None = None,
    sample_seed: int | None = None,
    threshold: float | None = None,
    correction: str = "bh",
) -> DriftReport:
    """
    Compare datasets for distribution drift.

    Args:
        baseline: Baseline/reference data
        current: Current data to compare
        method: Detection method ("auto", "ks", "psi", "chi2", "js", etc.)
        columns: Columns to compare (None = all)
        exclude_columns: Columns to exclude
        sample_size: Sample size for large datasets
        sample_seed: Random seed for reproducibility
        threshold: Custom threshold for drift detection
        correction: Multiple testing correction ("bh", "bonferroni", "none")

    Returns:
        DriftReport with per-column results
    """
```

**Available Methods**:

| Method | Description |
|--------|-------------|
| `auto` | Automatic selection based on data type |
| `ks` | Kolmogorov-Smirnov test |
| `psi` | Population Stability Index |
| `chi2` | Chi-square test |
| `js` | Jensen-Shannon divergence |
| `kl` | Kullback-Leibler divergence |
| `wasserstein` | Wasserstein distance |
| `cvm` | Cramer-von Mises test |
| `anderson` | Anderson-Darling test |

### th.scan()

Scans data for PII.

```python
def scan(
    data: DataInput = None,
    source: BaseDataSource | None = None,
    *,
    columns: list[str] | None = None,
    regulations: list[str] | None = None,
    min_confidence: float = 0.8,
) -> PIIReport:
    """
    Scan data for personally identifiable information.

    Args:
        data: Input data
        source: DataSource instance for SQL databases, Spark, etc.
                If provided, data argument is ignored.
        columns: Columns to scan (None = all)
        regulations: Regulations to check ("gdpr", "ccpa", "lgpd")
        min_confidence: Minimum confidence threshold

    Returns:
        PIIReport with findings and recommendations
    """
```

**Example with DataSource**:
```python
from truthound.datasources import get_sql_datasource

source = get_sql_datasource("mydb.db", table="users")
pii_report = th.scan(source=source)
```

### th.mask()

Masks sensitive data.

```python
def mask(
    data: pl.DataFrame = None,
    source: BaseDataSource | None = None,
    columns: list[str] | None = None,
    strategy: str = "redact",
    *,
    strict: bool = False,
) -> pl.DataFrame:
    """
    Mask sensitive data.

    Args:
        data: Input DataFrame
        source: DataSource instance for SQL databases, Spark, etc.
                If provided, data argument is ignored.
        columns: Columns to mask (None = auto-detect PII)
        strategy: Masking strategy ("redact", "hash", "fake")
        strict: If True, raise ValueError for non-existent columns.
                If False (default), emit warning and skip missing columns.

    Returns:
        DataFrame with masked values

    Raises:
        ValueError: If strict=True and a specified column doesn't exist.

    Warnings:
        MaskingWarning: When a column does not exist (only if strict=False).
    """
```

**Examples**:
```python
import truthound as th

# Basic masking (auto-detect PII)
masked_df = th.mask("data.csv")

# Mask specific columns
masked_df = th.mask(df, columns=["email", "phone"])

# Use hash strategy
masked_df = th.mask(df, strategy="hash")

# Strict mode - fail if columns don't exist
masked_df = th.mask(df, columns=["email"], strict=True)

# Non-strict mode (default) - warn and skip missing columns
masked_df = th.mask(df, columns=["email", "nonexistent"])
# Warning: Column 'nonexistent' not found in data. Skipping.
```

**Example with DataSource**:
```python
from truthound.datasources import get_sql_datasource

source = get_sql_datasource("mydb.db", table="users")
masked_df = th.mask(source=source, strategy="hash")
```

### th.profile()

Profiles data for statistical characteristics.

```python
def profile(
    data: DataInput = None,
    source: BaseDataSource | None = None,
    *,
    sample_size: int | None = None,
) -> DataProfile:
    """
    Generate statistical profile of data.

    Args:
        data: Input data
        source: DataSource instance for SQL databases, Spark, etc.
                If provided, data argument is ignored.
        sample_size: Sample size for large datasets

    Returns:
        DataProfile with column statistics
    """
```

**Example with DataSource**:
```python
from truthound.datasources import get_sql_datasource

source = get_sql_datasource("mydb.db", table="users")
profile = th.profile(source=source)
print(f"Rows: {profile.row_count}, Columns: {len(profile.columns)}")
```

---

## 2. Schema Module

### Schema Class

```python
from truthound.schema import Schema, ColumnSchema

class Schema:
    """Schema definition for data validation."""

    name: str | None
    version: str | None
    columns: list[ColumnSchema]

    @classmethod
    def from_yaml(cls, path: str | Path) -> Schema:
        """Load schema from YAML file."""

    @classmethod
    def from_dict(cls, data: dict) -> Schema:
        """Create schema from dictionary."""

    def save(self, path: str | Path) -> None:
        """Save schema to YAML file."""

    def to_dict(self) -> dict:
        """Convert to dictionary."""

    def validate(self, df: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate DataFrame against schema."""
```

### ColumnSchema Class

```python
@dataclass
class ColumnSchema:
    """Schema definition for a single column."""

    name: str
    dtype: str
    nullable: bool = True
    unique: bool = False
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list | None = None
    patterns: list[str] | None = None
    description: str | None = None
```

---

## 3. Validator Interface

### Base Validator

```python
from truthound.validators.base import Validator, ValidationIssue, Severity

class Validator(ABC):
    """Base class for all validators."""

    name: str           # Validator name
    category: str       # Validator category
    description: str    # Human-readable description

    @abstractmethod
    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Execute validation."""

    def get_config(self) -> dict:
        """Return configuration."""
```

### ValidationIssue

```python
@dataclass
class ValidationIssue:
    """Represents a validation issue."""

    validator: str
    column: str | None
    severity: Severity
    message: str
    details: dict = field(default_factory=dict)
    row_indices: list[int] | None = None
```

### Severity Enum

```python
class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

### Registering Validators

```python
from truthound.validators.base import register_validator

@register_validator("my_validator")
class MyValidator(Validator):
    name = "my_validator"
    category = "custom"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        ...
```

### Listing Validators

```python
from truthound.validators import list_validators, get_validator

# List all validators
validators = list_validators()

# List by category
schema_validators = list_validators(category="schema")

# Get specific validator
validator = get_validator("null_check")
```

---

## 4. Data Sources

### DataSource Protocol

```python
from truthound.datasources import BaseDataSource

class DataSource(Protocol):
    """Protocol for data source implementations."""

    @property
    def schema(self) -> dict[str, ColumnType]:
        """Column name to type mapping."""

    def to_lazyframe(self) -> pl.LazyFrame:
        """Convert to LazyFrame."""

    def get_execution_engine(self) -> ExecutionEngine:
        """Return execution engine."""

    def needs_sampling(self) -> bool:
        """Check if sampling needed."""

    def sample(self, n: int) -> DataSource:
        """Return sampled source."""
```

### Factory Functions

```python
from truthound.datasources import get_datasource, get_sql_datasource

# Auto-detect source type
source = get_datasource(data)

# SQL sources
source = get_sql_datasource("postgresql://...", table="users")
```

### Available Sources

```python
from truthound.datasources import (
    PolarsDataSource,
    PandasDataSource,
    FileDataSource,
    SparkDataSource,
)

from truthound.datasources.sql import (
    SQLiteDataSource,
    PostgreSQLDataSource,
    MySQLDataSource,
    BigQueryDataSource,
    SnowflakeDataSource,
    RedshiftDataSource,
    DatabricksDataSource,
    OracleDataSource,
    SQLServerDataSource,
)
```

---

## 5. Storage Backends

### Store Interface

```python
from truthound.stores import get_store, BaseStore

# Create store
store = get_store("filesystem", base_path=".truthound/results")
store.initialize()

# Save result
run_id = store.save(result)

# Retrieve result
result = store.get(run_id)

# Query results
results = store.query(StoreQuery(status="failure", limit=10))
```

### StoreQuery

```python
from truthound.stores.base import StoreQuery

query = StoreQuery(
    data_asset="customers.csv",
    status="failure",
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now(),
    order_by="run_time",
    ascending=False,
    limit=10,
    offset=0,
)
```

### Available Backends

| Backend | Configuration |
|---------|---------------|
| `filesystem` | `base_path`, `namespace`, `use_compression` |
| `memory` | (none) |
| `s3` | `bucket`, `prefix`, `region_name`, `compression` |
| `gcs` | `bucket`, `prefix`, `project`, `credentials_path` |
| `database` | `connection_url`, `pool_size`, `echo` |

---

## 6. Reporters

### Reporter Interface

```python
from truthound.reporters import get_reporter

# Create reporter
reporter = get_reporter("html", title="Validation Report")

# Render to string
html = reporter.render(result)

# Save to file
reporter.save(result, "report.html")
```

### Available Reporters

| Reporter | Output | Options |
|----------|--------|---------|
| `json` | JSON | `indent`, `include_details` |
| `console` | Terminal | `show_summary`, `max_issues` |
| `markdown` | Markdown | `include_toc` |
| `html` | HTML | `title`, `theme`, `chart_library` |
| `junit` | JUnit XML | (CI/CD integration) |

---

## 7. Profiler

### AutoProfiler

```python
from truthound.profiler import AutoProfiler

profiler = AutoProfiler()

# Profile data
profile = profiler.profile("data.csv")

# Generate rules
rules = profiler.generate_rules(profile)

# Save rules
rules.save("rules.yaml")
```

### DataProfile

```python
@dataclass
class DataProfile:
    """Statistical profile of a dataset."""

    row_count: int
    column_count: int
    memory_usage_mb: float
    columns: list[ColumnProfile]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
```

### ColumnProfile

```python
@dataclass
class ColumnProfile:
    """Statistical profile of a column."""

    name: str
    dtype: str
    null_count: int
    null_ratio: float
    unique_count: int
    unique_ratio: float
    mean: float | None
    std: float | None
    min: Any | None
    max: Any | None
    q25: float | None
    median: float | None
    q75: float | None
    skewness: float | None
    kurtosis: float | None
    entropy: float | None
    patterns: list[str]
```

---

## 8. Data Docs

### HTML Report Generation

```python
from truthound.datadocs import generate_html_report

html = generate_html_report(
    profile=profile.to_dict(),
    title="Data Quality Report",
    theme="professional",
    chart_library="plotly",
    output_path="report.html",
)
```

### Available Themes

| Theme | Description |
|-------|-------------|
| `default` | Clean, minimal design |
| `professional` | Corporate style |
| `dark` | Dark mode |
| `colorful` | Vibrant colors |
| `minimal` | Minimal styling |

### Available Chart Libraries

| Library | Description |
|---------|-------------|
| `plotly` | Interactive charts |
| `chartjs` | Lightweight charts |
| `echarts` | Rich visualizations |
| `vega` | Grammar of graphics |

---

## 9. ML Module

### Anomaly Detection

```python
from truthound.ml import (
    ZScoreAnomalyDetector,
    IQRAnomalyDetector,
    IsolationForestDetector,
    LOFDetector,
    MahalanobisDetector,
    EnsembleAnomalyDetector,
)

# Create detector
detector = IsolationForestDetector(contamination=0.1)

# Fit on training data
detector.fit(train_df)

# Detect anomalies
result = detector.detect(test_df)

# Get anomaly indices
indices = result.get_anomaly_indices()
```

### Drift Detection

```python
from truthound.ml import (
    KSDriftDetector,
    PSIDriftDetector,
    ChiSquareDriftDetector,
    JSDriftDetector,
)

detector = PSIDriftDetector(threshold=0.25)
result = detector.detect(baseline, current, column="feature")
```

### Rule Learning

```python
from truthound.ml import RuleLearner

learner = RuleLearner()
rules = learner.learn(df)
rules.save("learned_rules.yaml")
```

---

## 10. CLI Reference

### truthound check

```bash
truthound check DATA [OPTIONS]

Arguments:
  DATA                    Path to data file

Options:
  --schema PATH           Schema file path
  --validators TEXT       Comma-separated validator names
  --columns TEXT          Comma-separated column names
  --min-severity TEXT     Minimum severity (low, medium, high, critical)
  --strict               Exit with code 1 on failures
  --format TEXT           Output format (console, json, markdown)
  -o, --output PATH       Output file path
```

### truthound compare

```bash
truthound compare BASELINE CURRENT [OPTIONS]

Arguments:
  BASELINE               Baseline data file
  CURRENT                Current data file

Options:
  --method TEXT           Detection method (auto, ks, psi, chi2, js)
  --columns TEXT          Comma-separated column names
  --sample-size INT       Sample size for large datasets
  --threshold FLOAT       Drift threshold
  --format TEXT           Output format
  -o, --output PATH       Output file path
```

### truthound scan

```bash
truthound scan DATA [OPTIONS]

Arguments:
  DATA                    Path to data file

Options:
  --columns TEXT          Comma-separated column names
  --format TEXT           Output format
  -o, --output PATH       Output file path
```

### truthound mask

```bash
truthound mask DATA [OPTIONS]

Arguments:
  DATA                    Path to data file

Options:
  --columns TEXT          Comma-separated column names to mask
  --strategy, -s TEXT     Masking strategy (redact, hash, fake) [default: redact]
  --strict                Fail if specified columns don't exist (default: warn and skip)
  -o, --output PATH       Output file path [required]

Examples:
  truthound mask data.csv -o masked.csv
  truthound mask data.csv -o masked.csv --columns email,phone
  truthound mask data.csv -o masked.csv --strategy hash
  truthound mask data.csv -o masked.csv --columns email --strict
```

### truthound auto-profile

```bash
truthound auto-profile DATA [OPTIONS]

Arguments:
  DATA                    Path to data file

Options:
  -o, --output PATH       Output file path (default: profile.json)
  --sample-size INT       Sample size for large datasets
```

### truthound generate-suite

```bash
truthound generate-suite PROFILE [OPTIONS]

Arguments:
  PROFILE                 Profile JSON file

Options:
  -o, --output PATH       Output file path (default: rules.yaml)
  --categories TEXT       Comma-separated categories
```

### truthound docs generate

```bash
truthound docs generate PROFILE [OPTIONS]

Arguments:
  PROFILE                 Profile or validation result JSON

Options:
  -o, --output PATH       Output file path
  --theme TEXT            Theme name (default, professional, dark)
  --title TEXT            Report title
```

### truthound docs themes

```bash
truthound docs themes

# Lists available themes
```

### truthound list-formats

```bash
truthound list-formats

# Lists available output formats
```

### truthound list-categories

```bash
truthound list-categories

# Lists validator categories
```

---

## 11. Performance APIs

### Expression-Based Batch Execution

For advanced performance optimization, validators can be batched into a single `collect()` call.

```python
from truthound.validators.base import (
    ExpressionBatchExecutor,
    ValidationExpressionSpec,
    ExpressionValidatorMixin,
)
from truthound.validators.completeness.null import NullValidator
from truthound.validators.distribution.range import RangeValidator

# Batch multiple validators (single collect())
executor = ExpressionBatchExecutor()
executor.add_validator(NullValidator())
executor.add_validator(RangeValidator(min_value=0, max_value=100))
all_issues = executor.execute(lf)  # Single collect() for all validators
```

**ExpressionBatchExecutor**

```python
class ExpressionBatchExecutor:
    """Batches multiple expression-based validators into single collect()."""

    def add_validator(
        self,
        validator: Validator,
        columns: list[str] | None = None,
    ) -> None:
        """Add validator with optional column filter."""

    def execute(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Execute all validators with single collect()."""
```

**Supported Expression-Based Validators**:
- `NullValidator`, `NotNullValidator`, `CompletenessRatioValidator` (completeness)
- `BetweenValidator`, `RangeValidator`, `PositiveValidator`, `NonNegativeValidator` (range)

### Lazy Loading Registry

Validators are loaded on-demand to minimize startup time.

```python
from truthound.validators._lazy import (
    VALIDATOR_IMPORT_MAP,
    CATEGORY_MODULES,
    ValidatorImportMetrics,
    get_import_metrics,
)

# Get import metrics
metrics = get_import_metrics()
print(f"Loaded: {metrics.loaded_count}")
print(f"Failed: {metrics.failed_count}")
print(f"Total time: {metrics.total_import_time_ms:.2f}ms")
```

### Masking Performance

Data masking uses native Polars expressions and streaming mode for large datasets.

```python
import truthound as th

# Automatic streaming for large datasets (>1M rows)
masked_df = th.mask(large_df, strategy="hash")

# Masking strategies use native Polars:
# - "redact": pl.when/then/otherwise chains
# - "hash": Polars native hash() function (xxhash3)
# - "fake": Hash-based deterministic generation
```

---

## See Also

- [Getting Started](GETTING_STARTED.md) — Quick start guide
- [Examples](EXAMPLES.md) — Usage examples
- [Validators Reference](VALIDATORS.md) — Complete validator documentation
- [Plugin Architecture](PLUGINS.md) — Creating custom plugins
- [Performance Guide](PERFORMANCE.md) — Detailed performance optimization
