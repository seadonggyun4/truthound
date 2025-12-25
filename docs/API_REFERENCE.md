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

---

## 1. Core Functions

### th.check()

Validates data against rules and returns a validation report.

```python
def check(
    data: DataInput,
    *,
    schema: Schema | str | Path | None = None,
    validators: list[str | Validator] | None = None,
    columns: list[str] | None = None,
    min_severity: str | Severity = "low",
    strict: bool = False,
    auto_schema: bool = False,
    rules: dict | None = None,
) -> ValidationReport:
    """
    Validate data against rules.

    Args:
        data: Input data (DataFrame, file path, dict, etc.)
        schema: Schema to validate against
        validators: Specific validators to run
        columns: Columns to validate (None = all)
        min_severity: Minimum severity to report
        strict: Raise exception on failures
        auto_schema: Enable automatic schema caching
        rules: Rule dictionary

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
    data: DataInput,
    *,
    columns: list[str] | None = None,
    regulations: list[str] | None = None,
    min_confidence: float = 0.8,
) -> PIIReport:
    """
    Scan data for personally identifiable information.

    Args:
        data: Input data
        columns: Columns to scan (None = all)
        regulations: Regulations to check ("gdpr", "ccpa", "lgpd")
        min_confidence: Minimum confidence threshold

    Returns:
        PIIReport with findings and recommendations
    """
```

### th.mask()

Masks sensitive data.

```python
def mask(
    data: pl.DataFrame,
    *,
    strategy: str = "redact",
    columns: list[str] | None = None,
    strategies: dict[str, str] | None = None,
    pii_types: list[str] | None = None,
) -> pl.DataFrame:
    """
    Mask sensitive data.

    Args:
        data: Input DataFrame
        strategy: Default masking strategy ("redact", "hash", "fake")
        columns: Columns to mask (None = auto-detect)
        strategies: Per-column strategy mapping
        pii_types: PII types to mask

    Returns:
        DataFrame with masked values
    """
```

### th.profile()

Profiles data for statistical characteristics.

```python
def profile(
    data: DataInput,
    *,
    sample_size: int | None = None,
) -> DataProfile:
    """
    Generate statistical profile of data.

    Args:
        data: Input data
        sample_size: Sample size for large datasets

    Returns:
        DataProfile with column statistics
    """
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

## See Also

- [Getting Started](GETTING_STARTED.md) — Quick start guide
- [Examples](EXAMPLES.md) — Usage examples
- [Validators Reference](VALIDATORS.md) — Complete validator documentation
- [Plugin Architecture](PLUGINS.md) — Creating custom plugins
