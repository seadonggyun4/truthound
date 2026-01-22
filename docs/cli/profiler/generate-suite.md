# truthound generate-suite

Generate a validation suite from a profile file. This command creates validation rules based on data characteristics.

## Synopsis

```bash
truthound generate-suite <profile_file> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `profile_file` | Yes | Path to the profile file (JSON or YAML from auto-profile) |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | None | Output file path |
| `--format` | `-f` | `yaml` | Output format (yaml, json, python, toml, checkpoint) |
| `--strictness` | `-s` | `medium` | Rule strictness (loose, medium, strict) |
| `--include` | `-i` | All | Rule categories to include. Categories: schema, completeness, uniqueness, format, distribution, pattern, temporal, relationship, anomaly |
| `--exclude` | `-e` | None | Rule categories to exclude. Categories: schema, completeness, uniqueness, format, distribution, pattern, temporal, relationship, anomaly |
| `--min-confidence` | | None | Minimum confidence level (low, medium, high) |
| `--name` | `-n` | Auto | Suite name |
| `--preset` | `-p` | None | Use preset configuration |
| `--config` | `-c` | None | Additional configuration file |
| `--group-by-category` | | `false` | Group rules by category |
| `--code-style` | | `functional` | Python code style (functional, class_based, declarative) |

## Description

The `generate-suite` command creates validation rules from a data profile:

1. **Schema Rules**: Data type validation, nullable checks
2. **Completeness Rules**: Null ratio thresholds
3. **Uniqueness Rules**: Duplicate detection
4. **Range Rules**: Min/max value bounds
5. **Format Rules**: Pattern matching for detected formats
6. **Consistency Rules**: Cross-column validation

## Examples

### Basic Generation

```bash
truthound generate-suite profile.json -o suite.yaml
```

Output file (`suite.yaml`):
```yaml
name: data_validation_suite
version: "1.0"
generated_at: "2024-01-15T10:30:00Z"

validators:
  - type: not_null
    columns: [id, created_at]
    severity: high

  - type: unique
    columns: [id]
    severity: critical

  - type: range
    column: age
    min_value: 18
    max_value: 85
    severity: medium

  - type: pattern
    column: email
    pattern: email
    severity: high

  - type: allowed_values
    column: status
    values: [active, inactive, pending]
    severity: medium
```

### Strictness Levels

```bash
# Loose: Relaxed thresholds, fewer rules
truthound generate-suite profile.json -o suite.yaml --strictness loose

# Medium (default): Balanced rules
truthound generate-suite profile.json -o suite.yaml --strictness medium

# Strict: Tight thresholds, comprehensive rules
truthound generate-suite profile.json -o suite.yaml --strictness strict
```

### Using Presets

```bash
# Production-ready rules
truthound generate-suite profile.json -o suite.yaml --preset production

# CI/CD optimized
truthound generate-suite profile.json -o checkpoint.yaml --preset ci_cd

# Development friendly
truthound generate-suite profile.json -o validators.py --preset development
```

### Include/Exclude Categories

```bash
# Only schema and completeness rules
truthound generate-suite profile.json -o suite.yaml --include schema,completeness

# Exclude format rules
truthound generate-suite profile.json -o suite.yaml --exclude format
```

### Confidence Filtering

```bash
# Only high-confidence rules
truthound generate-suite profile.json -o suite.yaml --min-confidence high

# Include medium and high confidence
truthound generate-suite profile.json -o suite.yaml --min-confidence medium
```

### Output Formats

#### YAML Output (default)

```bash
truthound generate-suite profile.json -o suite.yaml --format yaml
```

#### JSON Output

```bash
truthound generate-suite profile.json -o suite.json --format json
```

#### Python Output

```bash
truthound generate-suite profile.json -o validators.py --format python
```

Output file (`validators.py`):
```python
import truthound as th
from truthound.validators import NotNullValidator, UniqueValidator, RangeValidator

# Generated validation suite
validators = [
    NotNullValidator(columns=["id", "created_at"]),
    UniqueValidator(columns=["id"]),
    RangeValidator(column="age", min_value=18, max_value=85),
]

def validate(data):
    return th.check(data, validators=validators)
```

#### TOML Output

```bash
truthound generate-suite profile.json -o suite.toml --format toml
```

#### Checkpoint Output

```bash
truthound generate-suite profile.json -o checkpoint.yaml --format checkpoint
```

Output file (`checkpoint.yaml`):
```yaml
name: data_checkpoint
version: "1.0"
schedule: daily

data_assets:
  - name: data
    path: data/*.csv

expectations:
  - asset: data
    validators:
      - type: not_null
        columns: [id]
      - type: unique
        columns: [id]
```

### Python Code Styles

```bash
# Functional style (default)
truthound generate-suite profile.json -o validators.py --format python --code-style functional

# Class-based style
truthound generate-suite profile.json -o validators.py --format python --code-style class_based

# Declarative style
truthound generate-suite profile.json -o validators.py --format python --code-style declarative
```

### Group by Category

```bash
truthound generate-suite profile.json -o suite.yaml --group-by-category
```

Output:
```yaml
name: data_validation_suite

completeness:
  - type: not_null
    columns: [id, created_at]

uniqueness:
  - type: unique
    columns: [id]

range:
  - type: range
    column: age
    min_value: 18
    max_value: 85
```

### Custom Suite Name

```bash
truthound generate-suite profile.json -o suite.yaml --name "Customer Data Validation"
```

## Available Presets

| Preset | Strictness | Confidence | Format | Description |
|--------|------------|------------|--------|-------------|
| `default` | medium | medium | yaml | Balanced settings |
| `strict` | strict | high | yaml | Tight validation |
| `loose` | loose | low | yaml | Relaxed validation |
| `minimal` | medium | high | yaml | Essential rules only |
| `comprehensive` | strict | low | yaml | All possible rules |
| `schema_only` | medium | high | yaml | Schema rules only |
| `format_only` | medium | medium | yaml | Format rules only |
| `ci_cd` | medium | medium | checkpoint | CI/CD optimized |
| `development` | loose | medium | python | Dev-friendly code |
| `production` | strict | high | yaml | Production-ready |

## Rule Categories

| Category | Rules Generated | Description |
|----------|-----------------|-------------|
| `schema` | dtype, nullable | Data type validation |
| `completeness` | not_null, completeness_ratio | Null value checks |
| `uniqueness` | unique, no_duplicates | Duplicate detection |
| `format` | pattern, email, phone | Format validation |
| `distribution` | range, min, max, between | Numeric bounds |
| `pattern` | regex patterns | Pattern detection |
| `temporal` | date range, freshness | Time-based checks |
| `relationship` | cross_column, referential | Multi-column checks |
| `anomaly` | outlier detection | Statistical anomalies |

## Use Cases

### 1. Data Pipeline Validation

```bash
# Generate rules for data pipeline
truthound auto-profile source_data.csv -o profile.json --format json
truthound generate-suite profile.json -o pipeline_rules.yaml --preset ci_cd
```

### 2. Model Training Data

```bash
# Strict validation for ML training data
truthound generate-suite training_profile.json -o ml_rules.yaml --preset strict
```

### 3. Development Testing

```bash
# Generate Python validators for tests
truthound generate-suite profile.json -o test_validators.py --preset development --code-style class_based
```

### 4. Production Monitoring

```bash
# Production-ready validation suite
truthound generate-suite profile.json -o production_suite.yaml --preset production
```

## Related Commands

- [`auto-profile`](auto-profile.md) - Generate profile for suite generation
- [`quick-suite`](quick-suite.md) - Profile + generate in one step
- [`check`](../core/check.md) - Run validation with generated suite

## See Also

- [Profiler Guide](../../guides/profiler.md)
- [CI/CD Integration](../../guides/ci-cd.md)
- [Custom Validators](../../tutorials/custom-validator.md)
