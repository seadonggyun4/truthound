# Pattern Matching

This document describes the Polars native pattern matching system.

## Overview

The pattern matching system implemented in `src/truthound/profiler/native_patterns.py` uses Polars' vectorized `str.contains()` operations for high-performance pattern detection.

## PatternSpec

A dataclass for defining pattern specifications.

```python
@dataclass
class PatternSpec:
    """Pattern specification definition"""

    name: str                      # Pattern name (e.g., "email", "phone")
    regex: str                     # Regular expression pattern
    data_type: DataType            # Data type to infer upon matching
    priority: int = 0              # Priority (higher values match first)
    examples: list[str] = field(default_factory=list)  # Example values
    description: str = ""          # Pattern description
    category: str = "general"      # Pattern category
```

## PatternBuilder

Pattern definition using a fluent API.

```python
from truthound.profiler.native_patterns import PatternBuilder

# Create pattern using fluent style
pattern = (
    PatternBuilder("korean_mobile")
    .regex(r"^01[0-9]-?[0-9]{3,4}-?[0-9]{4}$")
    .data_type(DataType.KOREAN_PHONE)
    .priority(100)
    .examples(["010-1234-5678", "01012345678"])
    .description("Korean mobile phone number")
    .category("korean")
    .build()
)
```

## NativePatternMatcher

A pattern matcher using native Polars operations.

```python
from truthound.profiler.native_patterns import NativePatternMatcher

# Create matcher
matcher = NativePatternMatcher()

# Match patterns in column
results = matcher.match(lf, "email_column")

for result in results:
    print(f"Pattern: {result.pattern_name}")
    print(f"Match ratio: {result.match_ratio:.2%}")
    print(f"Data type: {result.data_type}")
```

### Internal Implementation

```python
class NativePatternMatcher:
    """Polars native pattern matcher"""

    def match(self, lf: pl.LazyFrame, column: str) -> list[PatternMatch]:
        """
        Perform high-performance pattern matching using
        Polars' vectorized str.contains()
        """
        col = pl.col(column)

        for pattern in self._patterns:
            # Native Polars operation (no map_elements)
            match_expr = col.str.contains(pattern.regex)
            match_count = match_expr.sum()
            # ...
```

## Built-in Patterns

### General Patterns

| Pattern Name | Data Type | Description |
|--------------|-----------|-------------|
| `email` | `EMAIL` | Email addresses |
| `url` | `URL` | URL/URI |
| `uuid` | `UUID` | UUID (v1-v5) |
| `ip_address` | `IP_ADDRESS` | IPv4/IPv6 |
| `phone` | `PHONE` | International phone numbers |
| `date_iso` | `DATE` | ISO 8601 dates |
| `datetime_iso` | `DATETIME` | ISO 8601 datetime |
| `json` | `JSON` | JSON objects/arrays |
| `currency` | `CURRENCY` | Currency amounts |
| `percentage` | `PERCENTAGE` | Percentages |

### Korean-Specific Patterns

| Pattern Name | Data Type | Description |
|--------------|-----------|-------------|
| `korean_rrn` | `KOREAN_RRN` | Resident registration number |
| `korean_phone` | `KOREAN_PHONE` | Korean phone numbers |
| `korean_mobile` | `KOREAN_PHONE` | Korean mobile phone numbers |
| `korean_business_number` | `KOREAN_BUSINESS_NUMBER` | Business registration number |

## Pattern Registry

```python
from truthound.profiler.native_patterns import PatternRegistry

# Retrieve default patterns
email_pattern = PatternRegistry.get("email")

# Retrieve patterns by category
korean_patterns = PatternRegistry.get_by_category("korean")

# Register custom patterns
PatternRegistry.register(
    PatternSpec(
        name="custom_id",
        regex=r"^[A-Z]{2}\d{6}$",
        data_type=DataType.IDENTIFIER,
        priority=50,
        examples=["AB123456"],
        description="Company-specific ID format",
    )
)

# Remove pattern
PatternRegistry.unregister("custom_id")
```

## PatternMatch Result

```python
@dataclass
class PatternMatch:
    """Pattern matching result"""

    pattern_name: str       # Matched pattern name
    regex: str              # Regular expression used
    data_type: DataType     # Inferred data type
    match_count: int        # Number of matched rows
    total_count: int        # Total rows (excluding null)
    match_ratio: float      # Match ratio (0.0-1.0)
    confidence: float       # Confidence level
    sample_matches: list[str]  # Sample matched values
```

## Priority-Based Matching

When multiple patterns match, results are returned based on priority.

```python
# Priority example
patterns = [
    PatternSpec("korean_mobile", ..., priority=100),  # Checked first
    PatternSpec("phone", ..., priority=50),           # General phone number
    PatternSpec("numeric", ..., priority=10),         # Numeric
]

# Korean mobile numbers match before general phone numbers
```

## Performance Optimization

### Vectorized Operations

```python
# Internal implementation - no Python callbacks
def _count_matches(self, lf: pl.LazyFrame, column: str, pattern: str) -> int:
    return (
        lf.select(
            pl.col(column)
            .str.contains(pattern)  # Polars native
            .sum()
        )
        .collect()
        .item()
    )
```

### Combining with Sampling

```python
from truthound.profiler.native_patterns import NativePatternMatcher
from truthound.profiler.sampling import Sampler, SamplingConfig

# Sample from large data then perform pattern matching
sampler = Sampler(SamplingConfig(max_rows=10_000))
sampled_result = sampler.sample(lf)

matcher = NativePatternMatcher()
patterns = matcher.match(sampled_result.data.lazy(), "email")
```

## CLI Usage

```bash
# Profile with pattern detection
th profile data.csv --include-patterns

# Disable pattern detection
th profile data.csv --no-patterns

# Pattern detection for specific columns only
th profile data.csv --pattern-columns email,phone
```

## Custom Pattern Files

Custom patterns can be defined in YAML format.

```yaml
# custom_patterns.yaml
patterns:
  - name: employee_id
    regex: "^EMP\\d{5}$"
    data_type: identifier
    priority: 80
    examples:
      - EMP00001
      - EMP12345
    description: Employee ID

  - name: product_sku
    regex: "^[A-Z]{3}-\\d{4}-[A-Z]$"
    data_type: identifier
    priority: 70
    examples:
      - ABC-1234-X
    description: Product SKU
```

```python
from truthound.profiler.native_patterns import load_patterns_from_yaml

# Load custom patterns
patterns = load_patterns_from_yaml("custom_patterns.yaml")
PatternRegistry.register_all(patterns)
```

## Next Steps

- [Rule Generation](rule-generation.md) - Generate validation rules from detected patterns
- [ML Inference](ml-inference.md) - ML-based type inference
