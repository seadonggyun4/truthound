# truthound plugins create

Create a new plugin template.

## Synopsis

```bash
truthound plugins create <NAME> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Plugin name (lowercase letters, numbers, hyphens, underscores; must start with lowercase letter) |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `.` | Output directory |
| `--type` | `-t` | `validator` | Plugin type (validator, reporter, hook, or any other value for custom) |
| `--author` | | None | Author name |

## Description

The `plugin create` command generates a new plugin template:

1. **Creates** package structure with `pyproject.toml`
2. **Generates** plugin implementation file
3. **Sets up** entry points for auto-discovery
4. **Creates** README documentation

## Plugin Types

| Type | Description | Generated Code |
|------|-------------|----------------|
| `validator` | Custom validator plugin | Validator class |
| `reporter` | Custom reporter plugin | Reporter class |
| `hook` | Event hook plugin | Hook functions |
| `custom` | General-purpose plugin | Base plugin class |

## Examples

### Create Validator Plugin

```bash
truthound plugins create my-validator
```

Generated structure:
```
truthound-plugin-my-validator/
├── my_validator/
│   ├── __init__.py
│   └── plugin.py
├── pyproject.toml
└── README.md
```

### Create Reporter Plugin

```bash
truthound plugins create my-reporter --type reporter
```

### Create Hook Plugin

```bash
truthound plugins create audit-hook --type hook
```

### With Author Info

```bash
truthound plugins create my-validator --author "John Doe"
```

### Custom Output Directory

```bash
truthound plugins create my-validator --output ./plugins
```

Creates in `./plugins/truthound-plugin-my-validator/`.

## Generated Files

### pyproject.toml

```toml
[project]
name = "truthound-plugin-my-validator"
version = "0.1.0"
description = "Custom validator plugin for Truthound"
requires-python = ">=3.10"
dependencies = [
    "truthound>=0.1.0",
]

[project.entry-points."truthound.plugins"]
my-validator = "my_validator.plugin:MyValidatorPlugin"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

### Validator Plugin (plugin.py)

```python
from truthound.plugins import Plugin, PluginType
from truthound.validators.base import Validator
from truthound.validators.report import ValidationIssue

class MyValidatorPlugin(Plugin):
    """Custom validator plugin."""

    name = "my-validator"
    version = "0.1.0"
    type = PluginType.VALIDATOR

    def get_validators(self):
        return [MyValidator()]

class MyValidator(Validator):
    """Custom validator implementation."""

    name = "my_validator"
    severity = "MEDIUM"

    def validate(self, df, columns=None):
        issues = []
        # Add validation logic here
        return issues
```

### Reporter Plugin (plugin.py)

```python
from truthound.plugins import Plugin, PluginType
from truthound.reporters.base import Reporter

class MyReporterPlugin(Plugin):
    """Custom reporter plugin."""

    name = "my-reporter"
    version = "0.1.0"
    type = PluginType.REPORTER

    def get_reporters(self):
        return [MyReporter()]

class MyReporter(Reporter):
    """Custom reporter implementation."""

    name = "my_reporter"
    extension = ".txt"
    content_type = "text/plain"

    def render(self, report):
        # Add rendering logic here
        return str(report)
```

### Hook Plugin (plugin.py)

```python
from truthound.plugins import Plugin, PluginType
from truthound.plugins.hooks import HookType

class AuditHookPlugin(Plugin):
    """Event hook plugin."""

    name = "audit-hook"
    version = "0.1.0"
    type = PluginType.HOOK

    def get_hooks(self):
        return {
            HookType.BEFORE_VALIDATION.value: self.on_validation_start,
            HookType.AFTER_VALIDATION.value: self.on_validation_complete,
        }

    def on_validation_start(self, context):
        """Called before validation starts."""
        print(f"Starting validation on {context.file_path}")

    def on_validation_complete(self, context, report):
        """Called after validation completes."""
        print(f"Validation complete: {len(report.issues)} issues found")
```

## Development Workflow

```bash
# 1. Create plugin template
truthound plugins create my-validator --type validator

# 2. Navigate to plugin directory
cd truthound-plugin-my-validator

# 3. Edit plugin implementation
# Edit my_validator/plugin.py

# 4. Install in development mode
pip install -e .

# 5. Verify plugin is discovered
truthound plugins list

# 6. Load and test
truthound plugins load my-validator
truthound check data.csv --validators my-validator

# 7. Build for distribution
pip install build
python -m build

# 8. Publish (optional)
pip install twine
twine upload dist/*
```

## Use Cases

### 1. Company-Specific Validators

```bash
truthound plugins create company-validators \
  --type validator \
  --author "Data Team"
```

### 2. Custom Report Format

```bash
truthound plugins create slack-reporter \
  --type reporter \
  --author "DevOps Team"
```

### 3. Audit Logging Hook

```bash
truthound plugins create compliance-audit \
  --type hook \
  --author "Compliance Team"
```

### 4. Multi-Purpose Plugin

```bash
truthound plugins create enterprise-suite \
  --type custom \
  --author "Enterprise Team"
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Invalid plugin name (must start with lowercase letter, contain only lowercase letters, numbers, hyphens, underscores) |

## Related Commands

- [`plugin list`](list.md) - List plugins
- [`plugin load`](load.md) - Load plugin
- [`new plugin`](../scaffolding/new-plugin.md) - Alternative scaffolding command

## See Also

- [Plugin Commands Overview](index.md)
- [Plugin System](../../concepts/plugins.md)
- [Scaffolding Commands](../scaffolding/index.md)
