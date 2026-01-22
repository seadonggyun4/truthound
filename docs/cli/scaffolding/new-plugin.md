# truthound new plugin

Create a Truthound plugin package with boilerplate code.

## Synopsis

```bash
truthound new plugin <name> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `name` | Yes | Plugin name (snake_case) |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `.` | Output directory |
| `--type` | `-t` | `validator` | Plugin type |
| `--author` | `-a` | None | Author name |
| `--description` | `-d` | None | Plugin description |
| `--tests/--no-tests` | | `--tests` | Generate test code |
| `--install/--no-install` | `-i` | `--no-install` | Auto-install plugin after generation |
| `--min-version` | | `0.1.0` | Minimum Truthound version |
| `--python` | | `3.10` | Minimum Python version |

## Description

The `new plugin` command generates a complete plugin package:

1. **Creates** package structure with `pyproject.toml`
2. **Generates** component scaffolds based on type
3. **Sets up** tests and documentation
4. **Configures** entry points for auto-discovery
5. **Installs** plugin in editable mode (with `--install` flag)

## Plugin Types

| Type | Description | Components |
|------|-------------|------------|
| `validator` | Custom validators | Validator classes |
| `reporter` | Custom reporters | Reporter classes |
| `hook` | Event hooks | Hook handlers |
| `datasource` | Data source connectors | DataSource classes |
| `action` | Checkpoint actions | Action classes |
| `full` | All components | Everything |

## Examples

### Quick Start (Recommended)

Create and install a plugin in one command:

```bash
# Create and immediately install
truthound new plugin my_validators --install

# Short form
truthound new plugin my_validators -i
```

Output:
```
Creating plugin 'my_validators'...

Successfully generated 6 files:
  ./truthound-plugin-my_validators/pyproject.toml
  ./truthound-plugin-my_validators/README.md
  ./truthound-plugin-my_validators/my_validators/__init__.py
  ./truthound-plugin-my_validators/my_validators/plugin.py
  ./truthound-plugin-my_validators/tests/__init__.py
  ./truthound-plugin-my_validators/tests/test_plugin.py

Installing plugin from ./truthound-plugin-my_validators...
✓ Plugin installed successfully!

Next steps:
  1. Edit ./truthound-plugin-my_validators/my_validators/plugin.py to implement your plugin
  2. truthound plugins list
```

### Validator Plugin

```bash
truthound new plugin my_validators --type validator
```

Generated structure:
```
truthound-plugin-my_validators/
├── pyproject.toml
├── README.md
├── my_validators/
│   ├── __init__.py
│   └── validators/
│       ├── __init__.py
│       └── sample_validator.py
└── tests/
    └── test_plugin.py
```

Generated `pyproject.toml`:
```toml
[project]
name = "truthound-my-validators"
version = "0.1.0"
description = "Custom validators for Truthound"
requires-python = ">=3.10"
dependencies = [
    "truthound>=0.1.0",
]

[project.entry-points."truthound.plugins"]
my_validators = "my_validators:MyValidatorsPlugin"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

### Reporter Plugin

```bash
truthound new plugin custom_reports --type reporter
```

Generated structure:
```
custom_reports/
├── pyproject.toml
├── README.md
├── src/
│   └── custom_reports/
│       ├── __init__.py
│       └── reporters/
│           ├── __init__.py
│           └── sample_reporter.py
└── tests/
    └── test_reporters.py
```

### Hook Plugin

```bash
truthound new plugin my_hooks --type hook
```

Generated structure:
```
my_hooks/
├── pyproject.toml
├── README.md
├── src/
│   └── my_hooks/
│       ├── __init__.py
│       └── hooks/
│           ├── __init__.py
│           └── sample_hook.py
└── tests/
    └── test_hooks.py
```

Generated hook code:
```python
from truthound.plugins.hooks import HookRegistry, HookType

@HookRegistry.register(HookType.PRE_VALIDATION)
def pre_validation_hook(context):
    """Called before validation starts."""
    print(f"Starting validation on {context.file_path}")

@HookRegistry.register(HookType.POST_VALIDATION)
def post_validation_hook(context, report):
    """Called after validation completes."""
    print(f"Validation complete: {len(report.issues)} issues found")
```

### DataSource Plugin

```bash
truthound new plugin custom_db --type datasource --description "Custom database connector"
```

Generated datasource code:
```python
from truthound.datasources.base import DataSource
import polars as pl

class CustomDbDataSource(DataSource):
    """Custom database data source."""

    name = "custom_db"

    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def read(self) -> pl.LazyFrame:
        """Read data from custom database."""
        # Implement connection and data reading
        pass

    def get_schema(self) -> dict:
        """Get schema information."""
        pass
```

### Action Plugin

```bash
truthound new plugin my_actions --type action
```

Generated action code:
```python
from truthound.checkpoint.actions import Action, ActionResult

class MyCustomAction(Action):
    """Custom checkpoint action."""

    name = "my_action"

    def execute(self, report, context) -> ActionResult:
        """Execute the action."""
        # Implement action logic
        return ActionResult(success=True, message="Action completed")
```

### Full Plugin

```bash
truthound new plugin enterprise \
  --type full \
  --author "Company Inc." \
  --description "Enterprise validation suite" \
  --install
```

Generated structure:
```
enterprise/
├── pyproject.toml
├── README.md
├── src/
│   └── enterprise/
│       ├── __init__.py
│       ├── validators/
│       │   ├── __init__.py
│       │   └── sample_validator.py
│       ├── reporters/
│       │   ├── __init__.py
│       │   └── sample_reporter.py
│       ├── hooks/
│       │   ├── __init__.py
│       │   └── sample_hook.py
│       ├── datasources/
│       │   ├── __init__.py
│       │   └── sample_datasource.py
│       └── actions/
│           ├── __init__.py
│           └── sample_action.py
└── tests/
    ├── test_validators.py
    ├── test_reporters.py
    ├── test_hooks.py
    ├── test_datasources.py
    └── test_actions.py
```

### Version Requirements

```bash
truthound new plugin my_plugin --min-version 1.0.0 --python 3.11
```

Generated `pyproject.toml`:
```toml
[project]
requires-python = ">=3.11"
dependencies = [
    "truthound>=1.0.0",
]
```

### Custom Output Directory

```bash
truthound new plugin my_plugin -o ./plugins/
```

Creates plugin in `./plugins/my_plugin/` directory.

## Plugin Type Comparison

| Type | Validators | Reporters | Hooks | DataSources | Actions |
|------|------------|-----------|-------|-------------|---------|
| validator | Yes | - | - | - | - |
| reporter | - | Yes | - | - | - |
| hook | - | - | Yes | - | - |
| datasource | - | - | - | Yes | - |
| action | - | - | - | - | Yes |
| full | Yes | Yes | Yes | Yes | Yes |

## Entry Points

Plugins use entry points for auto-discovery:

```toml
# Validators
[project.entry-points."truthound.validators"]
my_validators = "my_plugin.validators:register"

# Reporters
[project.entry-points."truthound.reporters"]
my_reporters = "my_plugin.reporters:register"

# Hooks
[project.entry-points."truthound.hooks"]
my_hooks = "my_plugin.hooks:register"

# DataSources
[project.entry-points."truthound.datasources"]
my_datasources = "my_plugin.datasources:register"

# Actions
[project.entry-points."truthound.actions"]
my_actions = "my_plugin.actions:register"
```

## Use Cases

### 1. Company-Specific Validators

```bash
truthound new plugin company_validators \
  --type validator \
  --author "Data Team" \
  --min-version 1.0.0
```

### 2. Custom Output Formats

```bash
truthound new plugin custom_formats \
  --type reporter \
  --description "Company-specific report formats"
```

### 3. Integration Hooks

```bash
truthound new plugin ci_integration \
  --type hook \
  --description "CI/CD pipeline integration hooks"
```

### 4. Custom Data Sources

```bash
truthound new plugin internal_db \
  --type datasource \
  --description "Internal database connectors"
```

### 5. Notification Actions

```bash
truthound new plugin notifications \
  --type action \
  --description "Custom notification channels"
```

## Installing the Plugin

### Option 1: Auto-install (Recommended)

Use the `--install` flag to automatically install after generation:

```bash
truthound new plugin my_plugin --install
```

This runs `pip install -e .` automatically after creating the plugin files.

### Option 2: Manual Install

If you didn't use `--install`, install manually:

```bash
cd truthound-plugin-my_plugin
pip install -e .
```

### Verify Installation

```bash
truthound plugins list
```

### Build and Publish

```bash
cd truthound-plugin-my_plugin
pip install build
python -m build
pip install dist/*.whl

# Publish to PyPI
pip install twine
twine upload dist/*
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Generation error |
| 2 | Invalid arguments |

## Related Commands

- [`new validator`](new-validator.md) - Create single validator
- [`new reporter`](new-reporter.md) - Create single reporter
- [`new list`](new-list.md) - List scaffold types

## See Also

- [Scaffolding Overview](index.md)
- [Plugin System](../../concepts/plugins.md)
