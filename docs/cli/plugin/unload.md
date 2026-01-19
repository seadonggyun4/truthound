# truthound plugin unload

Unload a loaded plugin.

## Synopsis

```bash
truthound plugin unload <NAME>
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
truthound plugin unload my-validator
```

Output:
```
Unloading plugin: my-validator...
Plugin 'my-validator' unloaded successfully.
```

### Unload Active Plugin

```bash
# Plugin is currently active
truthound plugin unload my-validator
```

Output:
```
Deactivating plugin: my-validator...
Unloading plugin: my-validator...
Plugin 'my-validator' unloaded successfully.
```

### Verify After Unload

```bash
# Unload the plugin
truthound plugin unload my-validator

# Verify state changed to discovered
truthound plugin list
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
truthound plugin unload my-validator
```

### 2. Reload Plugin (Update)

```bash
# Unload current version
truthound plugin unload my-validator

# Install update
pip install --upgrade truthound-plugin-my-validator

# Reload
truthound plugin load my-validator
```

### 3. Troubleshooting

```bash
# Unload problematic plugin
truthound plugin unload buggy-plugin

# Check validation works without it
truthound check data.csv
```

## Error Handling

### Plugin Not Loaded

```bash
truthound plugin unload my-validator
```

Output:
```
Error: Plugin 'my-validator' is not loaded.
```

### Plugin Not Found

```bash
truthound plugin unload unknown-plugin
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
| 2 | Unload error |

## Related Commands

- [`plugin load`](load.md) - Load a plugin
- [`plugin disable`](disable.md) - Disable a plugin
- [`plugin list`](list.md) - List all plugins

## See Also

- [Plugin Commands Overview](index.md)
- [Plugin System](../../concepts/plugins.md)
