# truthound plugin enable

Enable (activate) a loaded plugin.

## Synopsis

```bash
truthound plugin enable <NAME>
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Plugin name to enable |

## Description

The `plugin enable` command activates a loaded plugin:

1. **Activates** the plugin for use
2. **Makes** it available for validation
3. **Changes** state from loaded/inactive to active

## State Transitions

```
loaded → active
inactive → active
```

## Examples

### Enable Plugin

```bash
truthound plugin enable my-validator
```

Output:
```
Enabling plugin: my-validator...
Plugin 'my-validator' enabled successfully.
```

### Enable After Load Without Activate

```bash
# Load without activating
truthound plugin load my-validator --no-activate

# Enable when ready
truthound plugin enable my-validator
```

### Re-enable Disabled Plugin

```bash
# Plugin was disabled
truthound plugin enable my-validator
```

Output:
```
Enabling plugin: my-validator...
Plugin 'my-validator' enabled successfully.
```

### Verify After Enable

```bash
# Enable the plugin
truthound plugin enable my-validator

# Verify it's active
truthound plugin list --state active
```

## Use Cases

### 1. Activate After Inspection

```bash
# Load without activating
truthound plugin load my-validator --no-activate

# Inspect plugin
truthound plugin info my-validator

# Enable if satisfied
truthound plugin enable my-validator
```

### 2. Re-enable Temporarily Disabled Plugin

```bash
# Previously disabled for testing
truthound plugin enable my-validator

# Use in validation
truthound check data.csv --validators my-validator
```

### 3. Selective Plugin Activation

```bash
# Load multiple plugins
truthound plugin load validator-a --no-activate
truthound plugin load validator-b --no-activate
truthound plugin load validator-c --no-activate

# Enable only what's needed
truthound plugin enable validator-a
truthound plugin enable validator-c
```

## Error Handling

### Plugin Not Loaded

```bash
truthound plugin enable my-validator
```

Output:
```
Error: Plugin 'my-validator' is not loaded. Use 'plugin load' first.
```

### Plugin Already Active

```bash
truthound plugin enable my-validator
```

Output:
```
Plugin 'my-validator' is already active.
```

### Plugin Not Found

```bash
truthound plugin enable unknown-plugin
```

Output:
```
Error: Plugin 'unknown-plugin' not found.
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Plugin not found or not loaded |
| 2 | Enable error |

## Related Commands

- [`plugin disable`](disable.md) - Disable a plugin
- [`plugin load`](load.md) - Load a plugin
- [`plugin list`](list.md) - List all plugins

## See Also

- [Plugin Commands Overview](index.md)
- [Plugin System](../../concepts/plugins.md)
