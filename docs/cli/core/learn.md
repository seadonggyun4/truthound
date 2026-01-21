# truthound learn

Learn schema from a data file. This command analyzes your data and generates a YAML schema file with inferred types and constraints.

## Synopsis

```bash
truthound learn <file> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `file` | Yes | Path to the data file (CSV, JSON, Parquet, NDJSON, JSONL) |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `schema.yaml` | Output schema file path |
| `--no-constraints` | | `false` | Don't infer constraints from data |

## Description

The `learn` command performs automatic schema inference by analyzing your data:

1. **Data Type Detection**: Identifies column types (Int64, Float64, String, Date, etc.)
2. **Constraint Inference**: Detects value ranges, allowed values, and patterns
3. **Nullability Detection**: Determines which columns allow null values
4. **Uniqueness Detection**: Identifies potential primary key columns

The generated schema can be used with [`truthound check`](check.md) to validate new data.

## Examples

### Basic Usage

Learn schema with default output:

```bash
truthound learn data.csv
```

Output:
```
Schema saved to schema.yaml
  Columns: 5
  Rows: 1,000
```

### Custom Output Path

Specify a custom output file:

```bash
truthound learn data.parquet -o my_schema.yaml
```

### Without Constraint Inference

Learn only data types without inferring min/max, allowed values, etc.:

```bash
truthound learn data.csv --no-constraints
```

This is useful when you want to define constraints manually.

### From Different File Formats

```bash
# From CSV
truthound learn users.csv

# From Parquet
truthound learn transactions.parquet

# From JSON
truthound learn events.json

# From NDJSON/JSONL
truthound learn logs.ndjson
```

## Output Format

The generated schema is a YAML file:

```yaml
name: data
version: "1.0"
columns:
  - name: id
    dtype: Int64
    nullable: false
    unique: true

  - name: email
    dtype: String
    nullable: true
    patterns:
      - email

  - name: age
    dtype: Int64
    nullable: true
    min_value: 0
    max_value: 120

  - name: status
    dtype: String
    nullable: false
    allowed_values:
      - active
      - inactive
      - pending

  - name: created_at
    dtype: Date
    nullable: false
```

## Schema Fields

| Field | Description |
|-------|-------------|
| `name` | Column name |
| `dtype` | Data type (Int64, Float64, String, Date, Datetime, Boolean) |
| `nullable` | Whether null values are allowed |
| `unique` | Whether values must be unique |
| `min_value` | Minimum value (numeric columns) |
| `max_value` | Maximum value (numeric columns) |
| `allowed_values` | List of valid values (categorical columns) |
| `patterns` | Data patterns (email, phone, url, etc.) |

## Use Cases

### 1. Schema-Based Validation Pipeline

```bash
# Step 1: Learn schema from reference data
truthound learn reference_data.csv -o schema.yaml

# Step 2: Validate new data against schema
truthound check new_data.csv --schema schema.yaml --strict
```

### 2. CI/CD Integration

```yaml
# .github/workflows/data-quality.yml
jobs:
  validate:
    steps:
      - name: Learn baseline schema
        run: truthound learn baseline/data.csv -o schema.yaml

      - name: Validate production data
        run: truthound check production/data.csv --schema schema.yaml --strict
```

### 3. Manual Schema Refinement

```bash
# Generate base schema without constraints
truthound learn data.csv --no-constraints -o base_schema.yaml

# Edit manually to add business rules
# Then use for validation
truthound check new_data.csv --schema base_schema.yaml
```

## Related Commands

- [`check`](check.md) - Validate data against schema
- [`profile`](profile.md) - Generate detailed data profile

## See Also

- [Python API: th.learn()](../../python-api/core-functions.md#thlearn)
- [Schema Configuration Guide](../../guides/configuration.md)
