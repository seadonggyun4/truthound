# truthound plugins info

Show detailed information about a specific plugin.

## Synopsis

```bash
truthound plugins info <NAME> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Plugin name |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--json` | | `False` | Output in JSON format |

## Description

The `plugin info` command displays detailed information about a plugin:

1. **Shows** name, version, type, and state
2. **Displays** description, author, and homepage
3. **Lists** compatibility requirements
4. **Shows** dependencies (plugins and Python packages)
5. **Displays** tags and metadata

## Examples

### Basic Info

```bash
truthound plugins info my-validator
```

Output:
```
╭─────────────── Plugin: my-validator ───────────────╮
│ Name: my-validator                                  │
│ Version: 0.1.0                                      │
│ Type: validator                                     │
│ Description: Custom validators for data quality    │
│ Author: John Doe                                    │
│ Homepage: https://github.com/johndoe/my-validator  │
│ License: MIT                                        │
│                                                     │
│ Compatibility:                                      │
│   Min Truthound: 1.0.0                              │
│   Max Truthound: Any                                │
│                                                     │
│ Dependencies:                                       │
│   Plugins: None                                     │
│   Python: pandas>=2.0                               │
│                                                     │
│ Tags: validation, custom                            │
╰─────────────────────────────────────────────────────╯
```

### JSON Output

```bash
truthound plugins info my-validator --json
```

Output:
```json
{
  "name": "my-validator",
  "version": "0.1.0",
  "type": "validator",
  "state": "active",
  "description": "Custom validators for data quality",
  "author": "John Doe",
  "homepage": "https://github.com/johndoe/my-validator",
  "license": "MIT",
  "min_truthound_version": "1.0.0",
  "max_truthound_version": null,
  "dependencies": [],
  "python_dependencies": ["pandas>=2.0"],
  "tags": ["validation", "custom"]
}
```

## Information Fields

| Field | Description |
|-------|-------------|
| Name | Plugin identifier |
| Version | Plugin version (semver) |
| Type | Plugin type (validator/reporter/hook/etc.) |
| State | Current state (discovered/loaded/active/etc.) |
| Description | Plugin description |
| Author | Plugin author |
| Homepage | Project homepage URL |
| License | License type |
| Min Truthound | Minimum compatible Truthound version |
| Max Truthound | Maximum compatible Truthound version |
| Plugin Dependencies | Required plugins |
| Python Dependencies | Required Python packages |
| Tags | Plugin tags for categorization |

## Use Cases

### 1. Check Plugin Compatibility

```bash
# Verify plugin is compatible with your Truthound version
truthound plugins info my-validator
```

### 2. View Dependencies

```bash
# Check what dependencies a plugin needs
truthound plugins info custom-reporter
```

### 3. Export Plugin Info

```bash
# Export to JSON for documentation
truthound plugins info my-validator --json > plugin-info.json
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Error (plugin not found, instantiation failed) |

## Related Commands

- [`plugin list`](list.md) - List all plugins
- [`plugin load`](load.md) - Load a plugin
- [`plugin enable`](enable.md) - Enable a plugin

## See Also

- [Plugin Commands Overview](index.md)
- [Plugin System](../../concepts/plugins.md)
