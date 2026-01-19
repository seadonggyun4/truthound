# Scaffolding Commands

Code generation commands for creating custom validators, reporters, and plugins.

## Overview

| Command | Description | Primary Use Case |
|---------|-------------|------------------|
| [`new validator`](new-validator.md) | Create custom validator | Validation logic |
| [`new reporter`](new-reporter.md) | Create custom reporter | Output formatting |
| [`new plugin`](new-plugin.md) | Create plugin package | Extension development |
| [`new list`](new-list.md) | List scaffold types | Discovery |
| [`new templates`](new-templates.md) | List templates | Template selection |

## What is Scaffolding?

Scaffolding generates boilerplate code for extending Truthound:

- **Validators** - Custom data quality checks
- **Reporters** - Custom output formats
- **Plugins** - Packaged extensions with multiple components

## Quick Examples

### Create a Validator

```bash
# Basic validator
truthound new validator my_validator

# Pattern validator with regex
truthound new validator email_format --template pattern --pattern "^[a-z@.]+$"

# Range validator
truthound new validator percentage --template range --min 0 --max 100
```

### Create a Reporter

```bash
# Basic reporter
truthound new reporter my_reporter

# JSON reporter
truthound new reporter json_export --extension .json --content-type application/json
```

### Create a Plugin

```bash
# Validator plugin
truthound new plugin my_validators --type validator

# Full plugin with all components
truthound new plugin enterprise --type full
```

### Discover Options

```bash
# List available scaffolds
truthound new list --verbose

# List validator templates
truthound new templates validator
```

## Workflow

```mermaid
graph LR
    A[new list] --> B[Choose Type]
    B --> C[new templates]
    C --> D[Choose Template]
    D --> E[new validator/reporter/plugin]
    E --> F[Generated Code]
    F --> G[Customize]
    G --> H[Test & Use]
```

## Generated Structure

### Validator

```
my_validator/
├── __init__.py
├── my_validator.py      # Validator implementation
└── tests/
    └── test_my_validator.py
```

### Reporter

```
my_reporter/
├── __init__.py
├── my_reporter.py       # Reporter implementation
└── tests/
    └── test_my_reporter.py
```

### Plugin

```
my_plugin/
├── pyproject.toml       # Package configuration
├── README.md
├── src/
│   └── my_plugin/
│       ├── __init__.py
│       ├── validators/  # (if type=validator)
│       ├── reporters/   # (if type=reporter)
│       └── hooks/       # (if type=hook)
└── tests/
```

## Common Options

| Option | Description | Available In |
|--------|-------------|--------------|
| `--output, -o` | Output directory | All |
| `--author, -a` | Author name | All |
| `--description, -d` | Description | All |
| `--tests/--no-tests` | Generate tests | All |
| `--docs/--no-docs` | Generate docs | validator, reporter |

## Use Cases

### 1. Custom Validation Logic

```bash
# Create validator for business rules
truthound new validator customer_age \
  --template range \
  --min 18 \
  --max 120 \
  --description "Validate customer age range"
```

### 2. Custom Output Format

```bash
# Create XML reporter
truthound new reporter xml_export \
  --template full \
  --extension .xml \
  --content-type application/xml
```

### 3. Reusable Plugin Package

```bash
# Create distributable plugin
truthound new plugin company_validators \
  --type validator \
  --author "Data Team" \
  --min-version 1.0.0
```

### 4. Integration Development

```bash
# Create datasource connector
truthound new plugin custom_db \
  --type datasource \
  --description "Custom database connector"
```

## Command Reference

- [new validator](new-validator.md) - Create custom validator
- [new reporter](new-reporter.md) - Create custom reporter
- [new plugin](new-plugin.md) - Create plugin package
- [new list](new-list.md) - List scaffold types
- [new templates](new-templates.md) - List available templates

## See Also

- [Custom Validator Tutorial](../../tutorials/custom-validator.md)
- [Plugin System](../../concepts/plugins.md)
- [Validators Guide](../../guides/validators.md)
