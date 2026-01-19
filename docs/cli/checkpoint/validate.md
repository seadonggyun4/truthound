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
Configuration Valid
===================
File: truthound.yaml
Checkpoints: 3
  - daily_validation (2 assets, 5 validators)
  - weekly_drift_check (2 assets, 1 validator)
  - monthly_audit (4 assets, 12 validators)

Status: ✓ Valid
```

Output (invalid):
```
Configuration Invalid
=====================
File: truthound.yaml

Errors:
  - Line 15: Unknown validator type 'not_nul' (did you mean 'not_null'?)
  - Line 23: Missing required field 'columns' for validator 'unique'
  - Line 31: Invalid severity 'super_high' (allowed: low, medium, high, critical)

Status: ✗ Invalid (3 errors)
```

### Strict Validation

Check that all referenced files exist:

```bash
truthound checkpoint validate truthound.yaml --strict
```

Output (with missing files):
```
Configuration Invalid
=====================
File: truthound.yaml

Errors:
  - data/customers.csv: File not found
  - data/orders.csv: File not found

Warnings:
  - Slack webhook URL uses environment variable ${SLACK_WEBHOOK_URL}
    (will be resolved at runtime)

Status: ✗ Invalid (2 errors, 1 warning)
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
| `checkpoints` | Yes | At least one checkpoint |
| `data_assets` | Yes | At least one data asset per checkpoint |
| `path` | Yes | Path for each data asset |
| `validators` | No | Validators list (optional) |

### Validator Schema

```yaml
validators:
  - type: not_null          # Required: validator type
    columns: [id, email]    # Required for column-based validators
    severity: high          # Optional: low, medium, high, critical
    message: "Custom msg"   # Optional: custom error message
```

### Valid Validator Types

| Category | Types |
|----------|-------|
| Completeness | `not_null`, `completeness_ratio` |
| Uniqueness | `unique`, `no_duplicates` |
| Range | `range`, `min`, `max`, `between` |
| Format | `pattern`, `email`, `phone`, `url` |
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
  run: truthound checkpoint run daily_validation --strict
```

### 3. Development Workflow

```bash
# Edit configuration
vim truthound.yaml

# Validate before running
truthound checkpoint validate truthound.yaml --strict && \
  truthound checkpoint run daily_validation
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Configuration is valid |
| 1 | Validation errors found |
| 2 | File not found or unreadable |

## Related Commands

- [`checkpoint run`](run.md) - Run a checkpoint
- [`checkpoint list`](list.md) - List checkpoints
- [`checkpoint init`](init.md) - Initialize configuration

## See Also

- [Configuration Reference](../../guides/configuration.md)
- [CI/CD Integration Guide](../../guides/ci-cd.md)
