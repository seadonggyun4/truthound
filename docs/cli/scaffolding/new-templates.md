# truthound new templates

List available templates for a scaffold type.

## Synopsis

```bash
truthound new templates <SCAFFOLD_TYPE>
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `SCAFFOLD_TYPE` | Yes | Scaffold type (validator, reporter, plugin) |

## Description

The `new templates` command displays available templates for a specific scaffold type:

1. **Lists** template names and descriptions
2. **Shows** template-specific features
3. **Helps** choose the right template

## Examples

### Validator Templates

```bash
truthound new templates validator
```

Output:
```
Available Templates for 'validator':

  basic       Minimal validator structure
  column      Column-level validator with target column support
  pattern     Regex pattern matching validator
  range       Numeric range validator with min/max bounds
  comparison  Cross-column comparison validator
  composite   Multiple validator combination
  full        All features included (production-ready)
```

### Reporter Templates

```bash
truthound new templates reporter
```

Output:
```
Available Templates for 'reporter':

  basic  Minimal reporter structure
  full   All features (filtering, sorting, configuration)
```

### Plugin Templates

```bash
truthound new templates plugin
```

Output:
```
Available Plugin Types:

  validator   Plugin with custom validators
  reporter    Plugin with custom reporters
  hook        Plugin with event hooks
  datasource  Plugin with data source connectors
  action      Plugin with checkpoint actions
  full        Plugin with all components
```

## Template Details

### Validator Templates

| Template | Description | Best For |
|----------|-------------|----------|
| `basic` | Minimal structure | Simple checks |
| `column` | Column-level validation | Target column validation |
| `pattern` | Regex pattern matching | Format validation |
| `range` | Numeric range checking | Bounds validation |
| `comparison` | Cross-column comparison | Column relationships |
| `composite` | Multiple validators | Complex rules |
| `full` | All features | Production validators |

### Reporter Templates

| Template | Description | Best For |
|----------|-------------|----------|
| `basic` | Minimal structure | Simple outputs |
| `full` | All features | Production reporters |

### Plugin Types

| Type | Description | Components |
|------|-------------|------------|
| `validator` | Validator plugin | Validator classes |
| `reporter` | Reporter plugin | Reporter classes |
| `hook` | Hook plugin | Hook handlers |
| `datasource` | DataSource plugin | DataSource classes |
| `action` | Action plugin | Action classes |
| `full` | Full plugin | All components |

## Workflow

```bash
# 1. List available scaffolds
truthound new list

# 2. Check available templates for your chosen scaffold type
truthound new templates validator

# 3. Create using a specific template
truthound new validator email_check --template pattern
```

## Use Cases

### 1. Choose Validator Template

```bash
# See what validator templates are available
truthound new templates validator

# Choose pattern template for email validation
truthound new validator email_format --template pattern --pattern "^[a-z@.]+$"
```

### 2. Choose Reporter Template

```bash
# See reporter templates
truthound new templates reporter

# Choose full template for production reporter
truthound new reporter detailed_report --template full
```

### 3. Choose Plugin Type

```bash
# See plugin types
truthound new templates plugin

# Choose validator type for validator-only plugin
truthound new plugin my_validators --type validator
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Error |
| 2 | Invalid scaffold type |

## Related Commands

- [`new validator`](new-validator.md) - Create custom validator
- [`new reporter`](new-reporter.md) - Create custom reporter
- [`new plugin`](new-plugin.md) - Create plugin package
- [`new list`](new-list.md) - List scaffold types

## See Also

- [Scaffolding Overview](index.md)
- [Custom Validator Tutorial](../../tutorials/custom-validator.md)
