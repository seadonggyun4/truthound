# truthound checkpoint validate

Validate a checkpoint configuration file. This command checks the configuration for errors before running.

## Synopsis

```bash
truthound checkpoint validate <config_file> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `config_file` | Yes | Path to the configuration file (YAML/JSON) |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--strict` | `-s` | `false` | Strict validation (verify file existence) |

## Description

The `checkpoint validate` command verifies configuration files:

1. **Syntax Check**: Validates YAML/JSON syntax
2. **Schema Validation**: Checks required fields and types
3. **Reference Check**: Validates cross-references (with `--strict`)
4. **File Existence**: Verifies data files exist (with `--strict`)

## Examples

### Basic Validation

```bash
truthound checkpoint validate truthound.yaml
```

Output (valid):
```
Validating truthound.yaml...

[OK]   Checkpoint 'daily_data_validation'
[OK]   Checkpoint 'hourly_metrics_check'

Validation passed: 2 checkpoint(s) are valid.
(Use --strict to also validate file existence)
```

Output (invalid):
```
Validating truthound.yaml...

[FAIL] Checkpoint 'daily_data_validation':
       - Invalid min_severity 'super_high'. Must be one of: critical, high, low, medium

Validation failed: 1 error(s) found in 1 checkpoint(s).
```

### Strict Validation

Check that all referenced files exist:

```bash
truthound checkpoint validate truthound.yaml --strict
```

Output (with missing files):
```
Validating truthound.yaml...

[FAIL] Checkpoint 'daily_data_validation':
       - Data source file not found: data/production.csv
[FAIL] Checkpoint 'hourly_metrics_check':
       - Data source file not found: data/metrics.parquet

Validation failed: 2 error(s) found in 2 checkpoint(s).
```

### JSON Configuration

Validate JSON configuration:

```bash
truthound checkpoint validate truthound.json
```

## Validation Rules

### Required Fields

| Field | Required | Description |
|-------|----------|-------------|
| `checkpoints` | Yes | At least one checkpoint (list) |
| `name` | Yes | Unique checkpoint identifier |
| `data_source` | Yes | Path to data file |
| `validators` | No | List of validators (optional) |

### Checkpoint Schema

```yaml
checkpoints:
- name: my_checkpoint       # Required: unique identifier
  data_source: data.csv     # Required: data file path
  validators:               # List of validator names
  - 'null'
  - duplicate
  validator_config:         # Optional: validator configurations
    range:
      columns:
        age:
          min_value: 0
          max_value: 150
  min_severity: medium      # Optional: minimum severity level
```

### Valid Validator Types

| Category | Types |
|----------|-------|
| Completeness | `null`, `not_null`, `completeness_ratio` |
| Uniqueness | `duplicate`, `unique`, `no_duplicates` |
| Range | `range`, `min`, `max`, `between` |
| Format | `regex`, `pattern`, `email`, `phone`, `url` |
| Consistency | `allowed_values`, `foreign_key` |
| Schema | `dtype`, `schema` |

### Severity Levels

| Level | Description |
|-------|-------------|
| `low` | Informational |
| `medium` | Warning |
| `high` | Error |
| `critical` | Critical failure |

## Error Messages

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Unknown validator type` | Typo in validator name | Check spelling |
| `Missing required field` | Required field not provided | Add the field |
| `Invalid severity` | Unrecognized severity level | Use low/medium/high/critical |
| `File not found` | Data file doesn't exist | Check path (with --strict) |
| `Invalid YAML syntax` | YAML parsing error | Fix indentation/syntax |

### Warnings

| Warning | Description |
|---------|-------------|
| `Environment variable` | Variable will be resolved at runtime |
| `Deprecated field` | Field is deprecated, use alternative |
| `Empty validators` | Checkpoint has no validators |

## Use Cases

### 1. Pre-Commit Hook

Validate configuration before committing:

```bash
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: validate-truthound-config
        name: Validate Truthound Config
        entry: truthound checkpoint validate truthound.yaml --strict
        language: system
        files: truthound\.ya?ml$
```

### 2. CI/CD Pipeline

```yaml
# GitHub Actions
- name: Validate Configuration
  run: truthound checkpoint validate truthound.yaml --strict

- name: Run Checkpoint
  run: truthound checkpoint run daily_data_validation --strict
```

### 3. Development Workflow

```bash
# Edit configuration
vim truthound.yaml

# Validate before running
truthound checkpoint validate truthound.yaml --strict && \
  truthound checkpoint run daily_data_validation
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Configuration is valid |
| 1 | Error (validation errors found, file not found, unreadable, or other error) |

## Related Commands

- [`checkpoint run`](run.md) - Run a checkpoint
- [`checkpoint list`](list.md) - List checkpoints
- [`checkpoint init`](init.md) - Initialize configuration

## See Also

- [Configuration Reference](../../guides/configuration.md)
- [CI/CD Integration Guide](../../guides/ci-cd.md)
