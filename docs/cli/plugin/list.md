# truthound plugins list

List all discovered plugins.

## Synopsis

```bash
truthound plugins list [OPTIONS]
```

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--type` | `-t` | None | Filter by plugin type (validator/reporter/hook/datasource/action/custom) |
| `--state` | `-s` | None | Filter by state (discovered/loading/loaded/active/inactive/error/unloading) |
| `--verbose` | `-v` | `False` | Show detailed information (description, author) |
| `--json` | | `False` | Output in JSON format |

## Description

The `plugin list` command displays all discovered plugins:

1. **Shows** plugin name, version, type, and state
2. **Filters** by type or state
3. **Displays** detailed info in verbose mode

## Plugin Types

| Type | Description |
|------|-------------|
| `validator` | Custom validator plugins |
| `reporter` | Custom reporter plugins |
| `hook` | Event hook plugins |
| `datasource` | Data source connection plugins |
| `action` | Notification/action plugins |
| `custom` | General-purpose plugins |

## Plugin States

| State | Description | Color |
|-------|-------------|-------|
| `discovered` | Found (not loaded) | Yellow |
| `loading` | Loading in progress | Cyan |
| `loaded` | Load complete | Blue |
| `active` | Activated | Green |
| `inactive` | Deactivated | Gray |
| `error` | Error occurred | Red |
| `unloading` | Unloading in progress | Yellow |

## Examples

### List All Plugins

```bash
truthound plugins list
```

Output:
```
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Name            ┃ Version ┃ Type      ┃ State      ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ my-validator    │ 0.1.0   │ validator │ active     │
│ custom-reporter │ 1.0.0   │ reporter  │ loaded     │
│ audit-hook      │ 0.2.0   │ hook      │ discovered │
└─────────────────┴─────────┴───────────┴────────────┘
```

### Filter by Type

```bash
truthound plugins list --type validator
```

Output:
```
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
┃ Name            ┃ Version ┃ Type      ┃ State  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━┩
│ my-validator    │ 0.1.0   │ validator │ active │
│ email-validator │ 0.2.0   │ validator │ loaded │
└─────────────────┴─────────┴───────────┴────────┘
```

### Filter by State

```bash
truthound plugins list --state active
```

### Verbose Output

```bash
truthound plugins list --verbose
```

Output:
```
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Name            ┃ Version ┃ Type      ┃ State  ┃ Description               ┃ Author   ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ my-validator    │ 0.1.0   │ validator │ active │ Custom data validators    │ John Doe │
│ custom-reporter │ 1.0.0   │ reporter  │ loaded │ Custom report formats     │ Jane Doe │
└─────────────────┴─────────┴───────────┴────────┴───────────────────────────┴──────────┘
```

### JSON Output

```bash
truthound plugins list --json
```

Output:
```json
[
  {
    "name": "my-validator",
    "version": "0.1.0",
    "type": "validator",
    "state": "active",
    "description": "Custom data validators",
    "author": "John Doe"
  }
]
```

### Combined Filters

```bash
truthound plugins list --type validator --state active --verbose
```

## Use Cases

### 1. Check Plugin Status

```bash
# See all plugins and their states
truthound plugins list

# Check which plugins are active
truthound plugins list --state active
```

### 2. Find Specific Plugin Types

```bash
# Find all validator plugins
truthound plugins list --type validator

# Find all reporter plugins
truthound plugins list --type reporter
```

### 3. Export Plugin List

```bash
# Export to JSON for automation
truthound plugins list --json > plugins.json
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Error |

## Related Commands

- [`plugin info`](info.md) - Show plugin details
- [`plugin load`](load.md) - Load a plugin
- [`plugin enable`](enable.md) - Enable a plugin

## See Also

- [Plugin Commands Overview](index.md)
- [Plugin System](../../concepts/plugins.md)
