# list-formats

List available output formats for suite generation.

## Synopsis

```bash
truthound list-formats
```

## Description

Displays all supported output formats with descriptions for the `generate-suite` and `quick-suite` commands.

## Available Formats

| Format | Description |
|--------|-------------|
| `yaml` | Human-readable YAML format (default) |
| `json` | Machine-readable JSON format |
| `python` | Executable Python code with validators |
| `toml` | TOML configuration format |
| `checkpoint` | Truthound checkpoint format for CI/CD |

## Example

```bash
$ truthound list-formats
Available output formats:

  yaml         - Human-readable YAML format (default)
  json         - Machine-readable JSON format
  python       - Executable Python code with validators
  toml         - TOML configuration format
  checkpoint   - Truthound checkpoint format for CI/CD
```

## Related Commands

- [`generate-suite`](generate-suite.md) - Generate validation rules from profile
- [`quick-suite`](quick-suite.md) - Profile and generate rules in one step
- [`list-presets`](list-presets.md) - List available configuration presets
- [`list-categories`](list-categories.md) - List available rule categories
