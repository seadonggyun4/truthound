# truthound plugins unload

Unload a loaded plugin.

## Synopsis

```bash
truthound plugins unload <NAME>
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Plugin name to unload |

## Description

The `plugin unload` command unloads a loaded plugin:

1. **Deactivates** the plugin if active
2. **Unloads** from memory
3. **Returns** to discovered state

## State Transitions

```
active → inactive → unloading → discovered
loaded → unloading → discovered
```

## Examples

### Unload Plugin

```bash
truthound plugins unload my-validator
```

Output:
```
Unloaded plugin: my-validator
```

### Unload Active Plugin

```bash
# Plugin is currently active
truthound plugins unload my-validator
```

Output:
```
Unloaded plugin: my-validator
```

### Verify After Unload

```bash
# Unload the plugin
truthound plugins unload my-validator

# Verify state changed to discovered
truthound plugins list
```

Output:
```
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Name            ┃ Version ┃ Type      ┃ State      ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ my-validator    │ 0.1.0   │ validator │ discovered │
└─────────────────┴─────────┴───────────┴────────────┘
```

## Use Cases

### 1. Cleanup After Testing

```bash
# Done testing plugin
truthound plugins unload my-validator
```

### 2. Reload Plugin (Update)

```bash
# Unload current version
truthound plugins unload my-validator

# Install update
pip install --upgrade truthound-plugin-my-validator

# Reload
truthound plugins load my-validator
```

### 3. Troubleshooting

```bash
# Unload problematic plugin
truthound plugins unload buggy-plugin

# Check validation works without it
truthound check data.csv
```

## Error Handling

### Plugin Not Found or Error

```bash
truthound plugins unload unknown-plugin
```

Output:
```
Error unloading plugin: Plugin 'unknown-plugin' not found.
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Error (plugin not found, unload failed) |

## Related Commands

- [`plugin load`](load.md) - Load a plugin
- [`plugin disable`](disable.md) - Disable a plugin
- [`plugin list`](list.md) - List all plugins

## See Also

- [Plugin Commands Overview](index.md)
- [Plugin System](../../concepts/plugins.md)
