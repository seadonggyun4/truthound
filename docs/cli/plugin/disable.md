# truthound plugins disable

Disable (deactivate) an active plugin.

## Synopsis

```bash
truthound plugins disable <NAME>
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Plugin name to disable |

## Description

The `plugin disable` command deactivates an active plugin:

1. **Deactivates** the plugin
2. **Keeps** it loaded but inactive
3. **Preserves** configuration for re-enabling

## State Transitions

```
active → inactive
```

## Examples

### Disable Plugin

```bash
truthound plugins disable my-validator
```

Output:
```
Disabled plugin: my-validator
```

### Verify After Disable

```bash
# Disable the plugin
truthound plugins disable my-validator

# Verify state changed
truthound plugins list
```

Output:
```
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Name            ┃ Version ┃ Type      ┃ State    ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
│ my-validator    │ 0.1.0   │ validator │ inactive │
└─────────────────┴─────────┴───────────┴──────────┘
```

### Disable and Re-enable

```bash
# Disable temporarily
truthound plugins disable my-validator

# Re-enable when needed
truthound plugins enable my-validator
```

## Use Cases

### 1. Temporary Deactivation

```bash
# Disable for testing without the plugin
truthound plugins disable my-validator

# Run validation without the plugin
truthound check data.csv

# Re-enable when done
truthound plugins enable my-validator
```

### 2. Troubleshooting

```bash
# Disable suspected problematic plugin
truthound plugins disable buggy-plugin

# Check if issue persists
truthound check data.csv

# If issue resolved, plugin was the cause
```

### 3. Selective Validation

```bash
# Disable plugins not needed for this run
truthound plugins disable extra-validator

# Run validation with fewer plugins
truthound check data.csv

# Re-enable after
truthound plugins enable extra-validator
```

### 4. A/B Testing

```bash
# Test with plugin
truthound check data.csv --output with-plugin.json

# Disable plugin
truthound plugins disable my-validator

# Test without plugin
truthound check data.csv --output without-plugin.json

# Compare results
```

## Error Handling

### Plugin Not Found or Error

```bash
truthound plugins disable unknown-plugin
```

Output:
```
Error disabling plugin: Plugin 'unknown-plugin' not found.
```

## Difference from Unload

| Action | disable | unload |
|--------|---------|--------|
| State after | inactive | discovered |
| Loaded in memory | Yes | No |
| Quick re-enable | Yes | No (requires load) |
| Use case | Temporary | Cleanup |

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Error (plugin not found, disable failed) |

## Related Commands

- [`plugin enable`](enable.md) - Enable a plugin
- [`plugin unload`](unload.md) - Unload a plugin
- [`plugin list`](list.md) - List all plugins

## See Also

- [Plugin Commands Overview](index.md)
- [Plugin System](../../concepts/plugins.md)
