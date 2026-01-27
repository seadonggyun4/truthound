# Core Functions

Main entry points for Truthound data quality operations.

## th.read()

Reads data from various sources and returns as Polars DataFrame. Convenience wrapper for `datasources.get_datasource()`.

### Signature

```python
def read(
    data: Any,
    sample_size: int | None = None,
    **kwargs: Any,
) -> pl.DataFrame:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `Any` | Required | File path, DataFrame, dict, or DataSource |
| `sample_size` | `int` | `None` | Optional sample size for large datasets |
| `**kwargs` | `Any` | - | Additional args passed to `get_datasource()` |

### Supported Input Types

| Type | Description | Example |
|------|-------------|---------|
| `str` | File path | `th.read("data.csv")` |
| `pl.DataFrame` | Polars DataFrame | `th.read(df)` |
| `pl.LazyFrame` | Polars LazyFrame | `th.read(lf)` |
| `dict` (data) | Column-oriented data | `th.read({"a": [1,2,3], "b": ["x","y","z"]})` |
| `dict` (config) | Config with "path" key | `th.read({"path": "data.csv", "delimiter": ";"})` |
| `BaseDataSource` | DataSource instance | `th.read(source)` |

### Returns

`pl.DataFrame` - Polars DataFrame containing the loaded data.

### Examples

```python
import truthound as th

# Read from file path
df = th.read("data.csv")
df = th.read("data.parquet")
df = th.read("data.json")

# Read from raw data dict
df = th.read({"a": [1, 2, 3], "b": ["x", "y", "z"]})

# Read with config dict
df = th.read({"path": "data.csv", "delimiter": ";"})

# With sampling for large datasets
df = th.read("large_data.csv", sample_size=10000)

# Read from Polars DataFrame/LazyFrame (passthrough)
df = th.read(pl.DataFrame({"x": [1, 2, 3]}))

# With additional options
df = th.read("data.csv", has_header=False)
```

---

## th.check()

Validates data against rules and returns a validation report.

### Signature

```python
def check(
    data: Any = None,
    source: BaseDataSource | None = None,
    validators: list[str | Validator] | None = None,
    validator_config: dict[str, dict[str, Any]] | None = None,
    min_severity: str | Severity | None = None,
    schema: str | Path | Schema | None = None,
    auto_schema: bool = False,
    use_engine: bool = False,
    parallel: bool = False,
    max_workers: int | None = None,
    pushdown: bool | None = None,
) -> Report:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `Any` | `None` | Input data (DataFrame, file path, dict) |
| `source` | `BaseDataSource` | `None` | DataSource for databases (overrides `data`) |
| `validators` | `list[str \| Validator]` | `None` | Specific validators to run (None = all) |
| `validator_config` | `dict` | `None` | Configuration for validators |
| `min_severity` | `str \| Severity` | `None` | Minimum severity to include |
| `schema` | `Schema \| str \| Path` | `None` | Schema to validate against |
| `auto_schema` | `bool` | `False` | Auto-learn and cache schema |
| `use_engine` | `bool` | `False` | Use execution engine (experimental) |
| `parallel` | `bool` | `False` | Use DAG-based parallel execution |
| `max_workers` | `int` | `None` | Max threads for parallel execution |
| `pushdown` | `bool` | `None` | Enable query pushdown for SQL sources |

### Returns

`Report` - Report containing validation issues and statistics.

### Examples

```python
import truthound as th

# Basic validation
report = th.check("data.csv")
print(f"Issues: {len(report.issues)}")

# With specific validators
report = th.check("data.csv", validators=["null", "duplicate", "range"])

# With validator configuration
report = th.check(
    "data.csv",
    validators=["regex"],
    validator_config={"regex": {"patterns": {"email": r"^[\w.+-]+@[\w-]+\.[\w.-]+$"}}}
)

# With schema validation
schema = th.learn("baseline.csv")
report = th.check("data.csv", schema=schema)

# With DataSource
from truthound.datasources.sql import PostgreSQLDataSource
source = PostgreSQLDataSource(table="users", host="localhost", database="mydb")
report = th.check(source=source)

# Filter by severity
report = th.check("data.csv", min_severity="medium")

# Parallel execution for large datasets
report = th.check("data.csv", parallel=True, max_workers=4)

# Query pushdown for SQL sources
report = th.check(source=source, pushdown=True)
```

---

## th.learn()

Learns a schema from data.

### Signature

```python
def learn(
    data: Any = None,
    source: BaseDataSource | None = None,
    infer_constraints: bool = True,
    categorical_threshold: int = 20,
) -> Schema:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `Any` | `None` | Input data (file path, DataFrame, dict) |
| `source` | `BaseDataSource` | `None` | DataSource for databases (overrides `data`) |
| `infer_constraints` | `bool` | `True` | Infer min/max, allowed values, etc. |
| `categorical_threshold` | `int` | `20` | Max unique values to treat as categorical |

### Returns

`Schema` - Inferred schema object with column definitions.

### Examples

```python
import truthound as th

# Basic schema learning
schema = th.learn("baseline.csv")

# Without constraint inference (types only)
schema = th.learn("data.csv", infer_constraints=False)

# Custom categorical threshold
schema = th.learn("data.csv", categorical_threshold=50)

# Save schema to file
schema.save("schema.yaml")

# Use for validation
report = th.check("new_data.csv", schema=schema)

# From database using DataSource
from truthound.datasources.sql import SQLiteDataSource
source = SQLiteDataSource(database="mydb.db", table="users")
schema = th.learn(source=source)

# PostgreSQL example
from truthound.datasources.sql import PostgreSQLDataSource
source = PostgreSQLDataSource(table="users", host="localhost", database="mydb")
schema = th.learn(source=source)
```

---

## th.scan()

Scans data for personally identifiable information (PII).

### Signature

```python
def scan(
    data: Any = None,
    source: BaseDataSource | None = None,
) -> PIIReport:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `Any` | `None` | Input data (file path, DataFrame, dict) |
| `source` | `BaseDataSource` | `None` | DataSource for databases |

### Returns

`PIIReport` - Report with PII findings.

### Examples

```python
import truthound as th

# Basic PII scan
pii_report = th.scan("customers.csv")

# findings is a list of dicts
for finding in pii_report.findings:
    print(f"{finding['column']}: {finding['pii_type']} ({finding['confidence']}%)")

# Check if PII was detected
if pii_report.has_pii:
    print(f"Found {len(pii_report.findings)} columns with potential PII")

# Print formatted report
pii_report.print()

# From database
from truthound.datasources.sql import PostgreSQLDataSource
source = PostgreSQLDataSource(table="users", host="localhost", database="mydb")
pii_report = th.scan(source=source)
```

---

## th.mask()

Masks sensitive data in a DataFrame.

### Signature

```python
def mask(
    data: Any = None,
    source: BaseDataSource | None = None,
    columns: list[str] | None = None,
    strategy: str = "redact",
    *,
    strict: bool = False,
) -> pl.DataFrame:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `Any` | `None` | Input data (file path, DataFrame, dict) |
| `source` | `BaseDataSource` | `None` | DataSource for databases |
| `columns` | `list[str]` | `None` | Columns to mask (auto-detect if None) |
| `strategy` | `str` | `"redact"` | Masking strategy |
| `strict` | `bool` | `False` | Raise error if columns don't exist |

### Strategies

| Strategy | Description | Example |
|----------|-------------|---------|
| `redact` | Replace with asterisks | `john@example.com` → `****` |
| `hash` | xxhash3 hash (16 chars) | `john@example.com` → `a8b9c0d1e2f3g4h5` |
| `fake` | Hash-based deterministic fake | `john@example.com` → `user_a8b9c0d1@fake.com` |

### Returns

`pl.DataFrame` - DataFrame with masked values.

### Examples

```python
import truthound as th
import polars as pl

df = pl.read_csv("customers.csv")

# Auto-detect and mask PII
masked = th.mask(df)

# Specific columns
masked = th.mask(df, columns=["email", "phone", "ssn"])

# Hash strategy (deterministic)
masked = th.mask(df, strategy="hash")

# Fake data (for testing)
masked = th.mask(df, strategy="fake")

# Strict mode - fails if column doesn't exist
masked = th.mask(df, columns=["email"], strict=True)

# From file path
masked = th.mask("customers.csv", columns=["email"])
```

---

## th.profile()

Generates a statistical profile of the data.

### Signature

```python
def profile(
    data: Any = None,
    source: BaseDataSource | None = None,
) -> ProfileReport:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `Any` | `None` | Input data (file path, DataFrame, dict) |
| `source` | `BaseDataSource` | `None` | DataSource for databases |

### Returns

`ProfileReport` - Profile with column statistics.

### Examples

```python
import truthound as th

profile = th.profile("data.csv")

print(f"Rows: {profile.row_count}")
print(f"Columns: {profile.column_count}")
print(f"Size: {profile.size_bytes} bytes")

# columns is a list of dicts with column statistics
for col in profile.columns:
    print(f"{col['name']}:")
    print(f"  Type: {col['dtype']}")
    print(f"  Nulls: {col['null_pct']}")
    print(f"  Unique: {col['unique_pct']}")
    if 'min' in col:
        print(f"  Min: {col['min']}")
        print(f"  Max: {col['max']}")

# Print formatted report
profile.print()

# Export to JSON
json_output = profile.to_json(indent=2)
```

---

## th.compare()

Compares two datasets for data drift.

### Signature

```python
def compare(
    baseline: Any,
    current: Any,
    columns: list[str] | None = None,
    method: str = "auto",
    threshold: float | None = None,
    sample_size: int | None = None,
) -> DriftReport:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `baseline` | `Any` | Required | Baseline (reference) data |
| `current` | `Any` | Required | Current data to compare |
| `columns` | `list[str]` | `None` | Columns to compare (None = all common) |
| `method` | `str` | `"auto"` | Detection method |
| `threshold` | `float` | `None` | Custom drift threshold |
| `sample_size` | `int` | `None` | Sample size for large datasets |

### Methods

**Statistical Tests (p-value based):**

| Method | Description | Column Type | Best For |
|--------|-------------|-------------|----------|
| `auto` | Automatic selection based on dtype | Any | General use (recommended) |
| `ks` | Kolmogorov-Smirnov test | Numeric only | Continuous numeric |
| `psi` | Population Stability Index | Numeric only | ML monitoring |
| `chi2` | Chi-squared test | Categorical | Categorical |
| `cvm` | Cramér-von Mises test | Numeric only | Tail sensitivity |
| `anderson` | Anderson-Darling test | Numeric only | Extreme values |

**Divergence Metrics:**

| Method | Description | Column Type | Best For |
|--------|-------------|-------------|----------|
| `js` | Jensen-Shannon divergence | Any | Any distribution |
| `kl` | Kullback-Leibler divergence | Numeric only | Information theory |

**Distance Metrics:**

| Method | Description | Column Type | Best For |
|--------|-------------|-------------|----------|
| `wasserstein` | Wasserstein (Earth Mover's) distance | Numeric only | Intuitive distance |
| `hellinger` | Hellinger distance | Any | Bounded metric [0,1] |
| `bhattacharyya` | Bhattacharyya distance | Any | Classification bounds |
| `tv` | Total Variation distance | Any | Max probability diff |
| `energy` | Energy distance | Numeric only | Location/scale changes |
| `mmd` | Maximum Mean Discrepancy | Numeric only | High-dimensional data |

> **Important:** Methods `ks`, `psi`, `kl`, `wasserstein`, `cvm`, `anderson`, `energy`, and `mmd` only work with numeric columns.
> For non-numeric columns, use `method="auto"`, `method="chi2"`, `method="js"`, `method="hellinger"`, `method="bhattacharyya"`, or `method="tv"`.

### Returns

`DriftReport` - Report with per-column drift results.

### Examples

```python
import truthound as th

# Basic comparison
drift = th.compare("baseline.csv", "current.csv")

if drift.has_high_drift:
    print("Significant drift detected!")
    for col_drift in drift.columns:
        if col_drift.result.drifted:
            print(f"  {col_drift.column}: {col_drift.result.method} = {col_drift.result.statistic:.4f}")

# Specific method (psi requires numeric columns)
drift = th.compare("train.csv", "prod.csv", method="psi", columns=["age", "income", "score"])

# KL divergence
drift = th.compare("baseline.csv", "current.csv", method="kl", columns=["age", "income"])

# Wasserstein distance (normalized)
drift = th.compare("baseline.csv", "current.csv", method="wasserstein", columns=["age", "income"])

# Cramér-von Mises (sensitive to tails)
drift = th.compare("baseline.csv", "current.csv", method="cvm", columns=["age", "income"])

# Anderson-Darling (most sensitive to tail differences)
drift = th.compare("baseline.csv", "current.csv", method="anderson", columns=["age", "income"])

# With custom threshold
drift = th.compare("old.csv", "new.csv", threshold=0.1)

# For large datasets, use sampling
drift = th.compare("big_train.csv", "big_prod.csv", sample_size=10000)

# Compare specific columns
drift = th.compare("baseline.csv", "current.csv", columns=["age", "income", "score"])

# Get drifted column names
drifted_cols = drift.get_drifted_columns()
print(f"Drifted columns: {drifted_cols}")
```

---

## DriftReport

Report returned by `th.compare()` with per-column drift analysis.

### Definition

```python
from truthound.drift.report import DriftReport, ColumnDrift
from truthound.drift.detectors import DriftResult, DriftLevel

@dataclass
class DriftReport:
    """Complete drift detection report."""

    baseline_source: str
    current_source: str
    baseline_rows: int
    current_rows: int
    columns: list[ColumnDrift]
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `has_drift` | `bool` | True if any column has drift |
| `has_high_drift` | `bool` | True if any column has HIGH drift |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_drifted_columns()` | `list[str]` | Column names with detected drift |
| `to_dict()` | `dict` | Convert to dictionary |
| `to_json(indent)` | `str` | Convert to JSON string |
| `print()` | `None` | Print formatted report to stdout |

### ColumnDrift

```python
@dataclass
class ColumnDrift:
    """Drift information for a single column."""

    column: str              # Column name
    dtype: str               # Data type
    result: DriftResult      # Drift detection result
    baseline_stats: dict     # Baseline statistics
    current_stats: dict      # Current statistics
```

### DriftResult

```python
@dataclass
class DriftResult:
    """Result of a drift detection test."""

    statistic: float         # Test statistic value
    p_value: float | None    # P-value (if applicable)
    threshold: float         # Threshold used for detection
    drifted: bool            # True if drift detected
    level: DriftLevel        # Drift severity level
    method: str              # Detection method used
    details: str | None      # Additional details
```

### DriftLevel Enum

```python
from truthound.drift.detectors import DriftLevel

class DriftLevel(str, Enum):
    NONE = "none"      # No drift
    LOW = "low"        # Minor drift
    MEDIUM = "medium"  # Moderate drift
    HIGH = "high"      # Significant drift
```

### Usage Examples

```python
import truthound as th

drift = th.compare("baseline.csv", "current.csv")

# Check overall drift
print(f"Has drift: {drift.has_drift}")
print(f"Has high drift: {drift.has_high_drift}")

# Iterate columns
for col in drift.columns:
    r = col.result
    print(f"{col.column} ({col.dtype}):")
    print(f"  Method: {r.method}")
    print(f"  Statistic: {r.statistic:.4f}")
    print(f"  Level: {r.level.value}")
    print(f"  Drifted: {r.drifted}")

# Export
json_output = drift.to_json(indent=2)
dict_output = drift.to_dict()
```

---

## Type Definitions

### DataInput

```python
from truthound.types import DataInput

DataInput = Union[
    str,           # File path
    pl.DataFrame,  # Polars DataFrame
    pl.LazyFrame,  # Polars LazyFrame
    pd.DataFrame,  # pandas DataFrame (via Any)
    dict,          # Dictionary
]
```

### Severity

```python
from truthound.types import Severity

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Severity supports comparison operators (custom __ge__, __gt__, __le__, __lt__)
Severity.HIGH > Severity.LOW      # True
Severity.MEDIUM >= Severity.LOW   # True
Severity.CRITICAL > Severity.HIGH # True
Severity.LOW < Severity.MEDIUM    # True
```

!!! info "No INFO Severity Level"
    Truthound does not have an `INFO` severity level. The lowest level is `LOW`.

## See Also

- [Schema](schema.md) - Schema classes
- [Validators](validators.md) - Validator interface
- [Data Sources](datasources.md) - Database connections
- [CLI Reference](../cli/index.md) - Command-line equivalents
