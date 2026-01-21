# Basic Profiling

This document describes the basic usage of the Truthound Profiler.

## DataProfiler

`DataProfiler` is the core class for profiling entire datasets.

### Basic Usage

```python
from truthound.profiler import DataProfiler, ProfilerConfig
import polars as pl

# Configuration
config = ProfilerConfig(
    sample_size=50000,
    include_patterns=True,
    include_correlations=False,
    top_n_values=10,
)

# Create profiler
profiler = DataProfiler(config)

# Profile LazyFrame
lf = pl.scan_csv("data.csv")
profile = profiler.profile(lf)

# View results
print(f"Rows: {profile.row_count}")
print(f"Columns: {profile.column_count}")
```

### TableProfile Structure

```python
@dataclass(frozen=True)
class TableProfile:
    """Table profile result (immutable object)"""

    # Table information
    name: str = ""
    row_count: int = 0
    column_count: int = 0

    # Memory estimation
    estimated_memory_bytes: int = 0

    # Column profiles (immutable tuple)
    columns: tuple[ColumnProfile, ...] = field(default_factory=tuple)

    # Table-level metrics
    duplicate_row_count: int = 0
    duplicate_row_ratio: float = 0.0

    # Correlation matrix
    correlations: tuple[tuple[str, str, float], ...] = field(default_factory=tuple)

    # Metadata
    source: str = ""
    profiled_at: datetime = field(default_factory=datetime.now)
    profile_duration_ms: float = 0.0
```

### Key Methods

```python
# Convert to dictionary
profile_dict = profile.to_dict()

# Retrieve specific column profile
col_profile = profile.get("email")

# List of column names
names = profile.column_names
```

## ColumnProfiler

`ColumnProfiler` analyzes individual columns.

### Basic Usage

```python
from truthound.profiler import ColumnProfiler

column_profiler = ColumnProfiler()
col_profile = column_profiler.profile(lf, "email")

print(f"Type: {col_profile.inferred_type}")
print(f"Null ratio: {col_profile.null_ratio:.2%}")
print(f"Unique ratio: {col_profile.unique_ratio:.2%}")
```

### ColumnProfile Structure

```python
@dataclass(frozen=True)
class ColumnProfile:
    """Column profile result (immutable object)"""

    # Basic information
    name: str
    physical_type: str  # Polars data type (string)
    inferred_type: DataType = DataType.UNKNOWN  # Inferred logical type

    # Completeness
    row_count: int = 0
    null_count: int = 0
    null_ratio: float = 0.0
    empty_string_count: int = 0

    # Uniqueness
    distinct_count: int = 0
    unique_ratio: float = 0.0
    is_unique: bool = False
    is_constant: bool = False

    # Statistical distribution (numeric columns)
    distribution: DistributionStats | None = None
    # DistributionStats fields: mean, std, min, max, median, q1, q3, skewness, kurtosis

    # Top/bottom values (ValueFrequency tuple)
    top_values: tuple[ValueFrequency, ...] = field(default_factory=tuple)
    bottom_values: tuple[ValueFrequency, ...] = field(default_factory=tuple)
    # ValueFrequency fields: value, count, ratio

    # Length statistics (string columns)
    min_length: int | None = None
    max_length: int | None = None
    avg_length: float | None = None

    # Pattern analysis (string columns)
    detected_patterns: tuple[PatternMatch, ...] = field(default_factory=tuple)
    # PatternMatch fields: pattern, regex, match_ratio, sample_matches

    # Temporal analysis (datetime columns)
    min_date: datetime | None = None
    max_date: datetime | None = None
    date_gaps: int = 0

    # Suggested validator list
    suggested_validators: tuple[str, ...] = field(default_factory=tuple)

    # Metadata
    profiled_at: datetime = field(default_factory=datetime.now)
    profile_duration_ms: float = 0.0
```

## DataType (Inferred Types)

```python
class DataType(str, Enum):
    """Inferred logical data type"""

    # Basic types
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"
    DURATION = "duration"

    # Semantic types (pattern detection)
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    UUID = "uuid"
    IP_ADDRESS = "ip_address"
    JSON = "json"

    # Identifiers
    CATEGORICAL = "categorical"
    IDENTIFIER = "identifier"

    # Numeric subtypes
    CURRENCY = "currency"
    PERCENTAGE = "percentage"

    # Korean-specific
    KOREAN_RRN = "korean_rrn"
    KOREAN_PHONE = "korean_phone"
    KOREAN_BUSINESS_NUMBER = "korean_business_number"

    # Unknown
    UNKNOWN = "unknown"
```

## ProfilerConfig Options

```python
@dataclass
class ProfilerConfig:
    """Profiling configuration"""

    # Sampling
    sample_size: int | None = None  # None = use all data
    random_seed: int = 42

    # Analysis options
    include_patterns: bool = True      # Enable pattern detection
    include_correlations: bool = False # Calculate correlations
    include_distributions: bool = True # Include distribution statistics

    # Performance tuning
    top_n_values: int = 10             # Number of top/bottom values
    pattern_sample_size: int = 1000    # Sample size for pattern matching
    correlation_threshold: float = 0.7 # Correlation threshold

    # Pattern detection
    min_pattern_match_ratio: float = 0.8  # Minimum pattern match ratio

    # Parallel processing
    n_jobs: int = 1
```

## CLI Usage

```bash
# Basic profiling
th profile data.csv

# JSON output
th profile data.csv -o profile.json

# Apply sampling
th profile data.csv --sample-size 10000

# Disable pattern detection
th profile data.csv --no-patterns

# Streaming mode (large files)
th profile large_file.csv --streaming --chunk-size 100000
```

## Supported File Formats

| Format | Extension | Support |
|--------|-----------|---------|
| CSV | `.csv` | ✅ |
| Parquet | `.parquet` | ✅ |
| JSON | `.json`, `.jsonl` | ✅ |
| Excel | `.xlsx`, `.xls` | ✅ |
| Arrow | `.arrow`, `.feather` | ✅ |

```python
# Profile various formats
profile = profiler.profile_file("data.parquet")
profile = profiler.profile_file("data.json")
profile = profiler.profile_file("data.xlsx")
```

## Result Serialization

```python
import json

# Save to JSON
with open("profile.json", "w") as f:
    json.dump(profile.to_dict(), f, indent=2, default=str)

# Load from JSON (manual restoration)
with open("profile.json") as f:
    data = json.load(f)
# TableProfile is frozen=True, so manual deserialization is required
```

## Next Steps

- [Sampling Strategies](sampling.md) - Processing large datasets
- [Pattern Matching](patterns.md) - Automatic detection of emails, phone numbers, etc.
- [Rule Generation](rule-generation.md) - Automatic validation rule generation
