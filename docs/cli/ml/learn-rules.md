# truthound ml learn-rules

Learn validation rules from data using machine learning analysis.

## Synopsis

```bash
truthound ml learn-rules <file> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `file` | Yes | Path to the data file (CSV, JSON, Parquet, NDJSON, JSONL) |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `learned_rules.json` | Output file path |
| `--strictness` | `-s` | `medium` | Rule strictness (loose, medium, strict) |
| `--min-confidence` | | `0.9` | Minimum confidence threshold (0.0-1.0) |
| `--max-rules` | | `100` | Maximum number of rules to generate |

## Description

The `ml learn-rules` command automatically generates validation rules from data:

1. **Analyzes** data patterns and distributions
2. **Infers** constraints and relationships
3. **Generates** validation rules with confidence scores
4. **Outputs** rules in usable format

## Learned Rule Types

| Rule Type | Description | Example |
|-----------|-------------|---------|
| `not_null` | Column should not have nulls | `email` has 0% nulls |
| `unique` | Column should be unique | `id` has 100% unique values |
| `range` | Numeric bounds | `age` between 0 and 120 |
| `pattern` | String format | `email` matches email pattern |
| `allowed_values` | Categorical values | `status` in [active, inactive] |
| `dtype` | Data type | `price` is Float64 |
| `correlation` | Column relationships | `total = quantity * price` |

## Examples

### Basic Rule Learning

```bash
truthound ml learn-rules data.csv
```

Output:
```
Learning Validation Rules
=========================
File: data.csv
Rows: 10,000
Columns: 8

Analyzing patterns...
Generating rules...

Learned Rules: 15

Rules by Category
─────────────────────────────────────────────────────────
Category        Count     Confidence Range
─────────────────────────────────────────────────────────
completeness    3         0.95 - 1.00
uniqueness      2         0.99 - 1.00
range           4         0.92 - 0.98
format          3         0.94 - 0.99
allowed_values  3         0.97 - 1.00
─────────────────────────────────────────────────────────

Top Rules (by confidence):
  1. [1.00] id: unique
  2. [1.00] created_at: not_null
  3. [0.99] email: pattern(email)
  4. [0.98] age: range(0, 120)
  5. [0.97] status: allowed_values([active, inactive, pending])

Output: learned_rules.json
```

### Custom Output Path

```bash
truthound ml learn-rules data.csv -o validation_rules.json
```

### Strictness Levels

```bash
# Loose: Fewer rules, higher tolerance
truthound ml learn-rules data.csv --strictness loose

# Medium (default): Balanced rules
truthound ml learn-rules data.csv --strictness medium

# Strict: More rules, tighter constraints
truthound ml learn-rules data.csv --strictness strict
```

### Confidence Threshold

```bash
# Only high-confidence rules (>= 95%)
truthound ml learn-rules data.csv --min-confidence 0.95

# Include lower-confidence rules
truthound ml learn-rules data.csv --min-confidence 0.8
```

### Limit Number of Rules

```bash
# Generate at most 50 rules
truthound ml learn-rules data.csv --max-rules 50

# Generate comprehensive ruleset
truthound ml learn-rules data.csv --max-rules 200
```

## Output Format

### JSON Output (default)

```json
{
  "source_file": "data.csv",
  "generated_at": "2024-01-15T10:30:00Z",
  "strictness": "medium",
  "min_confidence": 0.9,
  "summary": {
    "total_rules": 15,
    "avg_confidence": 0.96
  },
  "rules": [
    {
      "id": "rule_001",
      "type": "not_null",
      "column": "id",
      "confidence": 1.0,
      "severity": "critical",
      "evidence": {
        "null_count": 0,
        "null_ratio": 0.0
      }
    },
    {
      "id": "rule_002",
      "type": "unique",
      "column": "id",
      "confidence": 1.0,
      "severity": "critical",
      "evidence": {
        "unique_count": 10000,
        "unique_ratio": 1.0
      }
    },
    {
      "id": "rule_003",
      "type": "range",
      "column": "age",
      "confidence": 0.98,
      "severity": "high",
      "parameters": {
        "min_value": 0,
        "max_value": 120
      },
      "evidence": {
        "actual_min": 18,
        "actual_max": 85,
        "buffer_applied": true
      }
    },
    {
      "id": "rule_004",
      "type": "pattern",
      "column": "email",
      "confidence": 0.99,
      "severity": "high",
      "parameters": {
        "pattern": "email"
      },
      "evidence": {
        "match_ratio": 0.99,
        "sample_matches": ["john@example.com", "jane@test.org"]
      }
    },
    {
      "id": "rule_005",
      "type": "allowed_values",
      "column": "status",
      "confidence": 0.97,
      "severity": "medium",
      "parameters": {
        "values": ["active", "inactive", "pending"]
      },
      "evidence": {
        "observed_values": ["active", "inactive", "pending"],
        "value_counts": {
          "active": 6000,
          "inactive": 3500,
          "pending": 500
        }
      }
    }
  ]
}
```

## Strictness Levels

| Level | Description | Rule Generation |
|-------|-------------|-----------------|
| `loose` | Permissive rules | Wider ranges, fewer constraints |
| `medium` | Balanced rules | Reasonable buffers applied |
| `strict` | Tight rules | Close to observed data |

### Example: Range Rule by Strictness

For data with `age` values 18-85:

| Strictness | Generated Range | Buffer |
|------------|-----------------|--------|
| loose | 0-150 | ±100% |
| medium | 0-120 | ±40% |
| strict | 15-90 | ±10% |

## Use Cases

### 1. Bootstrap Validation

```bash
# Generate rules from reference data
truthound ml learn-rules reference_data.csv -o rules.json --strictness medium

# Use rules for validation
truthound check new_data.csv --rules rules.json
```

### 2. Schema Discovery

```bash
# Discover schema from unknown data
truthound ml learn-rules unknown_data.csv -o schema_rules.json --strictness loose
```

### 3. Continuous Rule Refinement

```bash
# Learn rules from production data periodically
truthound ml learn-rules weekly_data.csv -o rules_$(date +%Y%m%d).json --min-confidence 0.95
```

### 4. CI/CD Integration

```yaml
# GitHub Actions
- name: Learn and Validate
  run: |
    # Learn rules from baseline
    truthound ml learn-rules baseline.csv -o rules.json --strictness medium

    # Validate new data against learned rules
    truthound check new_data.csv --rules rules.json --strict
```

### 5. Documentation Generation

```bash
# Generate rules with high confidence for documentation
truthound ml learn-rules production_data.csv -o data_contract.json --strictness strict --min-confidence 0.98
```

## Comparison: `ml learn-rules` vs `generate-suite`

| Feature | `ml learn-rules` | `generate-suite` |
|---------|------------------|------------------|
| Input | Data file | Profile file |
| Approach | ML-based learning | Profile-based generation |
| Speed | Slower (analyzes data) | Faster (uses profile) |
| Output | JSON | YAML, JSON, Python, TOML |
| Customization | strictness, confidence | Many presets, categories |

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Error (invalid arguments, file not found, or other error) |

## Related Commands

- [`generate-suite`](../profiler/generate-suite.md) - Generate rules from profile
- [`quick-suite`](../profiler/quick-suite.md) - Profile + generate in one step
- [`check`](../core/check.md) - Validate with rules

## See Also

- [Profiler Guide](../../guides/profiler.md)
- [Custom Validators](../../tutorials/custom-validator.md)
- [Advanced Features](../../concepts/advanced.md)
