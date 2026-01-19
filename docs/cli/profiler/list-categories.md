# list-categories

List available rule categories for suite generation.

## Synopsis

```bash
truthound list-categories
```

## Description

Displays all supported rule categories with descriptions. Categories can be used to include or exclude specific types of validation rules during suite generation.

## Available Categories

| Category | Description |
|----------|-------------|
| `schema` | Column existence, types, and structure |
| `completeness` | Null values and data completeness |
| `uniqueness` | Unique constraints and cardinality |
| `format` | Data format validation (email, phone, etc.) |
| `distribution` | Statistical distribution checks |
| `pattern` | Regex pattern matching |
| `temporal` | Date/time validation |
| `relationship` | Cross-column relationships |
| `anomaly` | Anomaly detection rules |

## Example

```bash
$ truthound list-categories
Available rule categories:

  schema         - Column existence, types, and structure
  completeness   - Null values and data completeness
  uniqueness     - Unique constraints and cardinality
  format         - Data format validation (email, phone, etc.)
  distribution   - Statistical distribution checks
  pattern        - Regex pattern matching
  temporal       - Date/time validation
  relationship   - Cross-column relationships
  anomaly        - Anomaly detection rules
```

## Usage with generate-suite

```bash
# Include only schema and completeness rules
truthound generate-suite profile.json --include schema,completeness

# Exclude anomaly detection rules
truthound generate-suite profile.json --exclude anomaly

# Generate only format validation rules
truthound generate-suite profile.json --include format
```

## Related Commands

- [`generate-suite`](generate-suite.md) - Generate validation rules from profile
- [`quick-suite`](quick-suite.md) - Profile and generate rules in one step
- [`list-formats`](list-formats.md) - List available output formats
- [`list-presets`](list-presets.md) - List available configuration presets
