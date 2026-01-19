# truthound plugin create

Create a new plugin template.

## Synopsis

```bash
truthound plugin create <NAME> [OPTIONS]
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Plugin name (snake_case) |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `.` | Output directory |
| `--type` | `-t` | `validator` | Plugin type (validator/reporter/hook/custom) |
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
truthound plugin create my-validator
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
truthound plugin create my-reporter --type reporter
```

### Create Hook Plugin

```bash
truthound plugin create audit-hook --type hook
```

### With Author Info

```bash
truthound plugin create my-validator --author "John Doe"
```

### Custom Output Directory

```bash
truthound plugin create my-validator --output ./plugins
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
    "truthound>=1.0.0",
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
            HookType.PRE_VALIDATION: self.pre_validation,
            HookType.POST_VALIDATION: self.post_validation,
        }

    def pre_validation(self, context):
        """Called before validation starts."""
        print(f"Starting validation on {context.file_path}")

    def post_validation(self, context, report):
        """Called after validation completes."""
        print(f"Validation complete: {len(report.issues)} issues found")
```

## Development Workflow

```bash
# 1. Create plugin template
truthound plugin create my-validator --type validator

# 2. Navigate to plugin directory
cd truthound-plugin-my-validator

# 3. Edit plugin implementation
# Edit my_validator/plugin.py

# 4. Install in development mode
pip install -e .

# 5. Verify plugin is discovered
truthound plugin list

# 6. Load and test
truthound plugin load my-validator
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
truthound plugin create company-validators \
  --type validator \
  --author "Data Team"
```

### 2. Custom Report Format

```bash
truthound plugin create slack-reporter \
  --type reporter \
  --author "DevOps Team"
```

### 3. Audit Logging Hook

```bash
truthound plugin create compliance-audit \
  --type hook \
  --author "Compliance Team"
```

### 4. Multi-Purpose Plugin

```bash
truthound plugin create enterprise-suite \
  --type custom \
  --author "Enterprise Team"
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Generation error |
| 2 | Invalid arguments |

## Related Commands

- [`plugin list`](list.md) - List plugins
- [`plugin load`](load.md) - Load plugin
- [`new plugin`](../scaffolding/new-plugin.md) - Alternative scaffolding command

## See Also

- [Plugin Commands Overview](index.md)
- [Plugin System](../../concepts/plugins.md)
- [Scaffolding Commands](../scaffolding/index.md)
