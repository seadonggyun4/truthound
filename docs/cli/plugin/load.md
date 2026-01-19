# truthound plugin load

Load a discovered plugin.

## Synopsis

```bash
truthound plugin load <NAME> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Plugin name to load |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--activate/--no-activate` | | `--activate` | Activate after loading |

## Description

The `plugin load` command loads a discovered plugin:

1. **Loads** the plugin into memory
2. **Activates** the plugin by default
3. **Validates** dependencies and compatibility

## State Transitions

```
discovered → loading → loaded → active
```

With `--no-activate`:
```
discovered → loading → loaded
```

## Examples

### Load and Activate (Default)

```bash
truthound plugin load my-validator
```

Output:
```
Loading plugin: my-validator...
Plugin 'my-validator' loaded successfully.
Plugin 'my-validator' activated.
```

### Load Without Activating

```bash
truthound plugin load my-validator --no-activate
```

Output:
```
Loading plugin: my-validator...
Plugin 'my-validator' loaded successfully.
```

### Verify After Loading

```bash
# Load the plugin
truthound plugin load my-validator

# Verify it's active
truthound plugin list --state active
```

## Use Cases

### 1. Load Plugin for Testing

```bash
# Load plugin
truthound plugin load my-validator

# Test with validation
truthound check data.csv --validators my-validator
```

### 2. Load Without Activation

```bash
# Load but don't activate yet
truthound plugin load my-validator --no-activate

# Inspect the loaded plugin
truthound plugin info my-validator

# Activate when ready
truthound plugin enable my-validator
```

### 3. Load Multiple Plugins

```bash
# Load several plugins
truthound plugin load validator-a
truthound plugin load validator-b
truthound plugin load custom-reporter
```

## Error Handling

### Plugin Not Found

```bash
truthound plugin load unknown-plugin
```

Output:
```
Error: Plugin 'unknown-plugin' not found.
```

### Dependency Error

```bash
truthound plugin load my-validator
```

Output:
```
Error: Plugin 'my-validator' requires 'pandas>=2.0' which is not installed.
```

### Compatibility Error

```bash
truthound plugin load my-validator
```

Output:
```
Error: Plugin 'my-validator' requires Truthound >=2.0.0 (current: 1.0.0).
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Plugin not found |
| 2 | Load error (dependencies, compatibility) |

## Related Commands

- [`plugin list`](list.md) - List all plugins
- [`plugin unload`](unload.md) - Unload a plugin
- [`plugin enable`](enable.md) - Enable a plugin

## See Also

- [Plugin Commands Overview](index.md)
- [Plugin System](../../concepts/plugins.md)
