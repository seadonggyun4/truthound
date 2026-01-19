# list-presets

List available configuration presets for suite generation.

## Synopsis

```bash
truthound list-presets
```

## Description

Displays all supported presets with descriptions for the `generate-suite` and `quick-suite` commands. Presets provide pre-configured settings for different use cases.

## Available Presets

| Preset | Description |
|--------|-------------|
| `default` | Balanced settings (medium strictness, all categories) |
| `strict` | Strict validation rules with high confidence |
| `loose` | Relaxed validation for flexible data |
| `minimal` | Only high-confidence schema rules |
| `comprehensive` | All generators with detailed output |
| `schema_only` | Schema and completeness rules only |
| `format_only` | Format and pattern rules only |
| `ci_cd` | Optimized for CI/CD pipelines (checkpoint format) |
| `development` | Development-friendly (Python code output) |
| `production` | Production-ready (strict, high confidence) |

## Example

```bash
$ truthound list-presets
Available configuration presets:

  default          - Balanced settings (medium strictness, all categories)
  strict           - Strict validation rules with high confidence
  loose            - Relaxed validation for flexible data
  minimal          - Only high-confidence schema rules
  comprehensive    - All generators with detailed output
  schema_only      - Schema and completeness rules only
  format_only      - Format and pattern rules only
  ci_cd            - Optimized for CI/CD pipelines (checkpoint format)
  development      - Development-friendly (Python code output)
  production       - Production-ready (strict, high confidence)
```

## Usage with generate-suite

```bash
# Use the strict preset
truthound generate-suite profile.json --preset strict

# Use the ci_cd preset for checkpoint output
truthound generate-suite profile.json --preset ci_cd
```

## Related Commands

- [`generate-suite`](generate-suite.md) - Generate validation rules from profile
- [`quick-suite`](quick-suite.md) - Profile and generate rules in one step
- [`list-formats`](list-formats.md) - List available output formats
- [`list-categories`](list-categories.md) - List available rule categories
