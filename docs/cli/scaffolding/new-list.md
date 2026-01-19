# truthound new list

List available scaffold types.

## Synopsis

```bash
truthound new list [OPTIONS]
```

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--verbose` | `-v` | `False` | Show detailed descriptions |

## Description

The `new list` command displays all available scaffold types that can be generated using the `new` command:

1. **Shows** scaffold type names
2. **Displays** brief descriptions
3. **Lists** available options (with verbose mode)

## Examples

### Basic List

```bash
truthound new list
```

Output:
```
Available Scaffolds:
  validator  - Create custom validator
  reporter   - Create custom reporter
  plugin     - Create plugin package
```

### Verbose List

```bash
truthound new list --verbose
```

Output:
```
Available Scaffolds:

validator
  Create custom validator with boilerplate code
  Templates: basic, column, pattern, range, comparison, composite, full
  Options: --template, --author, --description, --category, --tests, --docs

reporter
  Create custom reporter with boilerplate code
  Templates: basic, full
  Options: --template, --author, --description, --extension, --content-type

plugin
  Create plugin package with multiple components
  Types: validator, reporter, hook, datasource, action, full
  Options: --type, --author, --description, --tests, --min-version, --python
```

## Scaffold Types

| Type | Description | Use Case |
|------|-------------|----------|
| `validator` | Custom validation logic | Data quality checks |
| `reporter` | Custom output formats | Report generation |
| `plugin` | Packaged extensions | Distributable components |

## Use Cases

### 1. Discover Available Options

```bash
# See what can be scaffolded
truthound new list

# Get details about each scaffold type
truthound new list -v
```

### 2. Before Creating a Component

```bash
# Check available scaffold types
truthound new list

# Then create the appropriate type
truthound new validator my_check
```

### 3. Documentation Reference

```bash
# Quick reference for available scaffolds
truthound new list --verbose
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Error |

## Related Commands

- [`new validator`](new-validator.md) - Create custom validator
- [`new reporter`](new-reporter.md) - Create custom reporter
- [`new plugin`](new-plugin.md) - Create plugin package
- [`new templates`](new-templates.md) - List available templates

## See Also

- [Scaffolding Overview](index.md)
- [Plugin System](../../concepts/plugins.md)
