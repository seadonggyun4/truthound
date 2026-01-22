# truthound plugins enable

Enable (activate) a loaded plugin.

## Synopsis

```bash
truthound plugins enable <NAME>
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Plugin name to enable |

## Description

The `plugin enable` command activates a loaded plugin:

1. **Loads** the plugin automatically if not already loaded
2. **Activates** the plugin for use
3. **Makes** it available for validation
4. **Changes** state from loaded/inactive to active

## State Transitions

```
loaded → active
inactive → active
```

## Examples

### Enable Plugin

```bash
truthound plugins enable my-validator
```

Output:
```
Enabled plugin: my-validator
```

### Enable After Load Without Activate

```bash
# Load without activating
truthound plugins load my-validator --no-activate

# Enable when ready
truthound plugins enable my-validator
```

### Re-enable Disabled Plugin

```bash
# Plugin was disabled
truthound plugins enable my-validator
```

Output:
```
Enabled plugin: my-validator
```

### Verify After Enable

```bash
# Enable the plugin
truthound plugins enable my-validator

# Verify it's active
truthound plugins list --state active
```

## Use Cases

### 1. Activate After Inspection

```bash
# Load without activating
truthound plugins load my-validator --no-activate

# Inspect plugin
truthound plugins info my-validator

# Enable if satisfied
truthound plugins enable my-validator
```

### 2. Re-enable Temporarily Disabled Plugin

```bash
# Previously disabled for testing
truthound plugins enable my-validator

# Use in validation
truthound check data.csv --validators my-validator
```

### 3. Selective Plugin Activation

```bash
# Load multiple plugins
truthound plugins load validator-a --no-activate
truthound plugins load validator-b --no-activate
truthound plugins load validator-c --no-activate

# Enable only what's needed
truthound plugins enable validator-a
truthound plugins enable validator-c
```

## Error Handling

### Plugin Not Found

```bash
truthound plugins enable unknown-plugin
```

Output:
```
Error enabling plugin: Plugin 'unknown-plugin' not found.
```

> **Note**: If the plugin is discovered but not loaded, the `enable` command will automatically load it first before enabling.

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Error (plugin not found, enable failed) |

## Related Commands

- [`plugin disable`](disable.md) - Disable a plugin
- [`plugin load`](load.md) - Load a plugin
- [`plugin list`](list.md) - List all plugins

## See Also

- [Plugin Commands Overview](index.md)
- [Plugin System](../../concepts/plugins.md)
