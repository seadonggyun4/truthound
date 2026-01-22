# Phase 9: Plugin Architecture

Truthound's plugin architecture is designed for extensibility and maintainability. External packages can extend validators, reporters, datasources, and more.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Plugin Types](#plugin-types)
- [Creating Plugins](#creating-plugins)
- [Hook System](#hook-system)
- [CLI Commands](#cli-commands)
- [Advanced Usage](#advanced-usage)

## Overview

Key components of the plugin architecture:

- **PluginManager**: Plugin lifecycle management (discovery, load, activate, unload)
- **PluginRegistry**: Plugin registration and lookup
- **HookManager**: Event-based extension system
- **PluginDiscovery**: Automatic plugin discovery (Entry points, directory scanning)

## Quick Start

### Using Plugins

```python
from truthound.plugins import PluginManager, get_plugin_manager

# Use global manager
manager = get_plugin_manager()

# Discover plugins
manager.discover_plugins()

# Load specific plugin
manager.load_plugin("my-validator-plugin")

# Load all plugins
manager.load_all()

# Check active plugins
for plugin in manager.get_active_plugins():
    print(f"{plugin.name} v{plugin.version}")
```

### Managing Plugins via CLI

```bash
# List discovered plugins
truthound plugins list

# Plugin details
truthound plugins info my-plugin

# Load plugin
truthound plugins load my-plugin

# Unload plugin
truthound plugins unload my-plugin

# Enable/disable plugin
truthound plugins enable my-plugin
truthound plugins disable my-plugin

# Create new plugin template
truthound plugins create my-new-plugin --type validator
```

#### CLI Command Behavior

| Command | Description | Prerequisites |
|---------|-------------|---------------|
| `list` | Shows all discovered plugins | None |
| `info` | Displays plugin metadata | None |
| `load` | Loads and optionally activates a plugin | Plugin must be discovered |
| `unload` | Unloads a loaded plugin | Plugin must be loaded |
| `enable` | Enables a plugin (loads if necessary) | Plugin must be discovered |
| `disable` | Disables a plugin (loads if necessary) | Plugin must be discovered |

**Important Notes:**

- The `unload` command only works on plugins that are currently loaded. Attempting to unload a plugin that has not been loaded will result in an error:
  ```
  Plugin 'my-plugin' is not loaded.
  ```
- The `enable` and `disable` commands will automatically load the plugin if it is not already loaded.
- Each CLI invocation creates a new plugin manager instance, so plugin state is not persisted between commands unless explicitly saved.

## Plugin Types

### 1. ValidatorPlugin

Adds custom validation rules.

```python
from truthound.plugins import ValidatorPlugin, PluginInfo, PluginType
from truthound.validators.base import Validator, ValidationIssue
from truthound.types import Severity
import polars as pl

class MyValidator(Validator):
    name = "my_validator"
    category = "custom"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues = []
        # Implement validation logic
        return issues

class MyValidatorPlugin(ValidatorPlugin):
    def _get_plugin_name(self) -> str:
        return "my-validator-plugin"

    def _get_plugin_version(self) -> str:
        return "1.0.0"

    def _get_description(self) -> str:
        return "Custom validators for my use case"

    def get_validators(self) -> list[type]:
        return [MyValidator]
```

### 2. ReporterPlugin

Adds new output formats.

```python
from truthound.plugins import ReporterPlugin
from truthound.reporters.base import ValidationReporter, ReporterConfig
from truthound.core import ValidationResult

class XMLReporter(ValidationReporter[ReporterConfig]):
    name = "xml"
    file_extension = ".xml"

    def render(self, data: ValidationResult) -> str:
        # XML rendering logic
        return "<report>...</report>"

class XMLReporterPlugin(ReporterPlugin):
    def _get_plugin_name(self) -> str:
        return "xml-reporter"

    def get_reporters(self) -> dict[str, type]:
        return {"xml": XMLReporter}
```

### 3. HookPlugin

Registers hooks that respond to events.

```python
from truthound.plugins import HookPlugin, HookType
from typing import Any, Callable

class NotifierPlugin(HookPlugin):
    def _get_plugin_name(self) -> str:
        return "notifier"

    def get_hooks(self) -> dict[str, Callable]:
        return {
            HookType.AFTER_VALIDATION.value: self._on_validation_complete,
            HookType.ON_ERROR.value: self._on_error,
        }

    def _on_validation_complete(self, datasource, result, issues, **kwargs):
        if issues:
            print(f"Found {len(issues)} issues!")

    def _on_error(self, error, context, **kwargs):
        print(f"Error occurred: {error}")
```

### 4. DataSourcePlugin

Adds new data source types.

```python
from truthound.plugins import DataSourcePlugin
from truthound.datasources.base import BaseDataSource

class MongoDataSource(BaseDataSource):
    source_type = "mongodb"
    # Implementation...

class MongoPlugin(DataSourcePlugin):
    def _get_plugin_name(self) -> str:
        return "mongodb-source"

    def get_datasource_types(self) -> dict[str, type]:
        return {"mongodb": MongoDataSource}
```

## Creating Plugins

### Directory Structure

```
truthound-plugin-myfeature/
├── myfeature/
│   ├── __init__.py
│   └── plugin.py
├── pyproject.toml
└── README.md
```

### pyproject.toml Configuration

```toml
[project]
name = "truthound-plugin-myfeature"
version = "0.1.0"
dependencies = ["truthound>=0.1.0"]

[project.entry-points."truthound.plugins"]
myfeature = "myfeature:MyFeaturePlugin"
```

### Creating Templates via CLI

```bash
# Create validator plugin + auto install (recommended)
truthound new plugin my_validator --type validator --install

# Create reporter plugin + auto install
truthound new plugin my_reporter --type reporter --install

# Create hook plugin + auto install
truthound new plugin my_notifier --type hook --install

# Create without install (manual installation required)
truthound new plugin my_validator --type validator
cd truthound-plugin-my_validator && pip install -e .
```

> **Tip**: Using the `--install` (`-i`) flag automatically runs `pip install -e .` after plugin creation, making it immediately available for use.

## Hook System

### Available Hook Types

| Hook | Description | Handler Signature |
|------|-------------|-------------------|
| `before_validation` | Before validation starts | `(datasource, validators, **kwargs)` |
| `after_validation` | After validation completes | `(datasource, result, issues, **kwargs)` |
| `on_issue_found` | When issue is found | `(issue, validator, **kwargs)` |
| `before_profile` | Before profiling starts | `(datasource, config, **kwargs)` |
| `after_profile` | After profiling completes | `(datasource, profile, **kwargs)` |
| `on_report_generate` | When report is generated | `(report, format, **kwargs)` |
| `on_error` | When error occurs | `(error, context, **kwargs)` |
| `on_plugin_load` | When plugin is loaded | `(plugin, manager)` |
| `on_plugin_unload` | When plugin is unloaded | `(plugin, manager)` |

### Using Decorators

```python
from truthound.plugins import before_validation, after_validation, on_error

@before_validation(priority=50)  # Lower priority executes first
def log_start(datasource, validators, **kwargs):
    print(f"Validating {datasource} with {len(validators)} validators")

@after_validation()
def log_complete(datasource, result, issues, **kwargs):
    print(f"Found {len(issues)} issues")

@on_error()
def handle_error(error, context, **kwargs):
    print(f"Error: {error}")
```

### Using HookManager Directly

```python
from truthound.plugins import HookManager, HookType

hooks = HookManager()

# Register hook
hooks.register(
    HookType.BEFORE_VALIDATION,
    my_handler,
    priority=100,
    source="my-plugin"
)

# Trigger hook
results = hooks.trigger(
    HookType.BEFORE_VALIDATION,
    datasource=source,
    validators=["null", "range"]
)

# Disable hooks from specific source
hooks.disable(source="my-plugin")
```

## CLI Commands

### Plugin Creation

```bash
# Create plugin + auto install (recommended)
truthound new plugin my_validator --type validator --install

# Using short options
truthound new plugin my_validator -t validator -i

# Create with all options
truthound new plugin enterprise \
    --type full \
    --author "Your Name" \
    --description "Enterprise validators" \
    --install \
    --output ./my-plugins/
```

### Plugin Management

```bash
# List plugins (with details)
truthound plugins list --verbose

# JSON output
truthound plugins list --json

# Filter by type
truthound plugins list --type validator

# Filter by state
truthound plugins list --state active

# Plugin info
truthound plugins info my-plugin --json

# Load plugin
truthound plugins load my-plugin --activate

# Unload plugin
truthound plugins unload my-plugin

# Enable/disable plugin
truthound plugins enable my-plugin
truthound plugins disable my-plugin
```

## Advanced Usage

### Plugin Configuration

```python
from truthound.plugins import PluginManager, PluginConfig

manager = PluginManager()

# Per-plugin configuration
config = PluginConfig(
    enabled=True,
    priority=50,  # Load order (lower = earlier)
    settings={
        "api_key": "...",
        "timeout": 30,
    },
    auto_load=True,
)

manager.set_plugin_config("my-plugin", config)
manager.load_plugin("my-plugin")
```

### Dependency Management

```python
from truthound.plugins import Plugin, PluginInfo, PluginType

class DependentPlugin(Plugin):
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="dependent-plugin",
            version="1.0.0",
            plugin_type=PluginType.CUSTOM,
            dependencies=("base-plugin",),  # Depends on other plugin
            python_dependencies=("requests", "jinja2"),  # Python package dependencies
        )
```

### Version Compatibility

```python
PluginInfo(
    name="my-plugin",
    version="1.0.0",
    plugin_type=PluginType.VALIDATOR,
    min_truthound_version="0.5.0",
    max_truthound_version="2.0.0",
)
```

### Context Manager Usage

```python
from truthound.plugins import PluginManager

with PluginManager() as manager:
    manager.discover_plugins()
    manager.load_all()
    # Perform operations...
# All plugins automatically unloaded
```

### Loading Plugins from Directory

```python
from truthound.plugins import PluginManager
from pathlib import Path

manager = PluginManager()
manager.add_plugin_directory(Path("./my-plugins"))
manager.discover_plugins()
```

## Example Plugins

Truthound includes reference example plugins:

```python
from truthound.plugins.examples import (
    CustomValidatorPlugin,  # Custom business rule validator
    SlackNotifierPlugin,    # Slack notification hook
    XMLReporterPlugin,      # XML reporter
)
```

See the `truthound/plugins/examples/` directory for detailed implementations.

## API Reference

### Core Classes

- `Plugin[ConfigT]`: Plugin base class
- `PluginConfig`: Plugin configuration
- `PluginInfo`: Plugin metadata
- `PluginType`: Plugin type enum
- `PluginState`: Plugin state enum

### Specialized Base Classes

- `ValidatorPlugin`: Validator plugin base class
- `ReporterPlugin`: Reporter plugin base class
- `DataSourcePlugin`: DataSource plugin base class
- `HookPlugin`: Hook plugin base class

### Management

- `PluginManager`: Plugin lifecycle management
- `PluginRegistry`: Plugin registration/lookup
- `PluginDiscovery`: Plugin discovery
- `HookManager`: Hook registration/execution

### Exceptions

- `PluginError`: Base plugin error
- `PluginLoadError`: Load failure
- `PluginNotFoundError`: Plugin not found
- `PluginDependencyError`: Dependency not satisfied
- `PluginCompatibilityError`: Version incompatibility

## Enterprise Features

Advanced plugin features are included for enterprise environments:

### Enterprise Plugin Manager

```python
from truthound.plugins import create_enterprise_manager

# Create enterprise manager with security level
manager = create_enterprise_manager(
    security_level="enterprise",  # "development", "standard", "enterprise", "strict"
    require_signature=True,       # Require plugin signature
    enable_hot_reload=True,       # Enable hot reload
)

# Load plugin
plugin = await manager.load("my-plugin")

# Execute in sandbox
result = await manager.execute_in_sandbox("my-plugin", my_function, arg1, arg2)
```

### Security Sandbox

Execute plugins in an isolated environment to enhance system security:

```python
from truthound.plugins import (
    SandboxFactory,
    IsolationLevel,
    SecurityPolicyPresets,
)

# Create sandbox by isolation level
sandbox = SandboxFactory().create(IsolationLevel.PROCESS)

# Use security policy preset
policy = SecurityPolicyPresets.ENTERPRISE.to_policy()
```

### Code Signing

Verify plugin integrity and origin:

```python
from pathlib import Path
from truthound.plugins import (
    SigningServiceImpl,
    SignatureAlgorithm,
    TrustStoreImpl,
    TrustLevel,
    create_verification_chain,
)

# Sign plugin
service = SigningServiceImpl(
    algorithm=SignatureAlgorithm.HMAC_SHA256,
    signer_id="my-org",
)
signature = service.sign(
    plugin_path=Path("my_plugin/"),
    private_key=b"secret_key",
)

# Configure trust store
trust_store = TrustStoreImpl()
trust_store.set_signer_trust("my-org", TrustLevel.TRUSTED)

# Verify signature
chain = create_verification_chain(trust_store=trust_store)
result = chain.verify(plugin_path, signature, context={})
```

### Hot Reload

Reload plugins without restarting the application:

```python
from truthound.plugins import HotReloadManager, ReloadStrategy, LifecycleManager

lifecycle = LifecycleManager()
reload_manager = HotReloadManager(
    lifecycle,
    default_strategy=ReloadStrategy.GRACEFUL,
)

# Start watching plugin
await reload_manager.watch(
    plugin_id="my-plugin",
    plugin_path=Path("plugins/my-plugin/"),
    auto_reload=True,
)

# Manual reload
result = await reload_manager.reload("my-plugin")
```

### Version Constraints

Supports semantic version constraints:

```python
from truthound.plugins import parse_constraint

# Various version constraint expressions
constraint = parse_constraint("^1.2.3")  # >=1.2.3 && <2.0.0
constraint = parse_constraint("~1.2.3")  # >=1.2.3 && <1.3.0
constraint = parse_constraint(">=1.0.0,<2.0.0")  # Range specification

# Check version compatibility
is_compatible = constraint.is_satisfied_by("1.5.0")
```

### Dependency Graph

Automatic plugin dependency management:

```python
from truthound.plugins import DependencyGraph, DependencyType

graph = DependencyGraph()
graph.add_node("plugin-c", "1.0.0")
graph.add_node("plugin-b", "1.0.0",
    dependencies={"plugin-c": DependencyType.REQUIRED})
graph.add_node("plugin-a", "1.0.0",
    dependencies={"plugin-b": DependencyType.REQUIRED})

# Determine load order
load_order = graph.get_load_order()
# -> ['plugin-c', 'plugin-b', 'plugin-a']

# Detect circular dependencies
cycles = graph.detect_cycles()
```

For detailed Enterprise features, refer to `.claude/docs/phase-09-plugins.md`.
