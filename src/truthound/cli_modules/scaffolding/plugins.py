"""Plugin scaffold generator.

This module provides scaffolding for creating Truthound plugins with
various template variants:
    - validator: Plugin that provides custom validators
    - reporter: Plugin that provides custom reporters
    - hook: Plugin that provides event hooks
    - datasource: Plugin that provides data source connectors
    - action: Plugin that provides checkpoint actions
    - full: Full-featured plugin with all components
"""

from __future__ import annotations

from typing import Any, ClassVar

from truthound.cli_modules.scaffolding.base import (
    BaseScaffold,
    ScaffoldConfig,
    ScaffoldResult,
    register_scaffold,
)


@register_scaffold(
    name="plugin",
    description="Generate a Truthound plugin",
    aliases=("plug", "p"),
)
class PluginScaffold(BaseScaffold):
    """Scaffold generator for plugins.

    Creates complete plugin packages with pyproject.toml, README, and tests.
    """

    name: ClassVar[str] = "plugin"
    description: ClassVar[str] = "Generate a Truthound plugin"
    aliases: ClassVar[tuple[str, ...]] = ("plug", "p")

    TEMPLATE_VARIANTS: ClassVar[tuple[str, ...]] = (
        "validator",
        "reporter",
        "hook",
        "datasource",
        "action",
        "full",
    )

    def get_options(self) -> dict[str, Any]:
        """Get plugin-specific options."""
        return {
            "plugin_type": {
                "type": "str",
                "default": "validator",
                "description": "Type of plugin (validator, reporter, hook, datasource, action)",
                "choices": ["validator", "reporter", "hook", "datasource", "action"],
            },
            "min_truthound_version": {
                "type": "str",
                "default": "0.1.0",
                "description": "Minimum Truthound version required",
            },
            "python_version": {
                "type": "str",
                "default": "3.10",
                "description": "Minimum Python version",
            },
        }

    def _generate_files(self, config: ScaffoldConfig, result: ScaffoldResult) -> None:
        """Generate plugin files based on variant."""
        variant = config.template_variant
        pkg_name = config.name.replace("-", "_")

        # Generate plugin package structure
        self._generate_pyproject(config, result)
        self._generate_readme(config, result)
        self._generate_package_init(config, result, pkg_name)

        # Generate plugin implementation based on type
        if variant == "validator":
            self._generate_validator_plugin(config, result, pkg_name)
        elif variant == "reporter":
            self._generate_reporter_plugin(config, result, pkg_name)
        elif variant == "hook":
            self._generate_hook_plugin(config, result, pkg_name)
        elif variant == "datasource":
            self._generate_datasource_plugin(config, result, pkg_name)
        elif variant == "action":
            self._generate_action_plugin(config, result, pkg_name)
        elif variant == "full":
            self._generate_full_plugin(config, result, pkg_name)
        else:
            self._generate_validator_plugin(config, result, pkg_name)

        # Generate tests
        if config.include_tests:
            self._generate_tests(config, result, pkg_name)

    def _generate_pyproject(self, config: ScaffoldConfig, result: ScaffoldResult) -> None:
        """Generate pyproject.toml."""
        pkg_name = config.name.replace("-", "_")
        min_python = config.extra.get("python_version", "3.10")
        min_truthound = config.extra.get("min_truthound_version", "0.1.0")

        content = f'''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "truthound-plugin-{config.name}"
version = "{config.version}"
description = "{config.description or f'A {config.template_variant} plugin for Truthound'}"
readme = "README.md"
license = {{text = "{config.license_type}"}}
authors = [
    {{name = "{config.author or 'Your Name'}", email = "your@email.com"}}
]
requires-python = ">={min_python}"
dependencies = [
    "truthound>={min_truthound}",
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Quality Assurance",
]
keywords = ["truthound", "data-quality", "plugin"]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

[project.urls]
Homepage = "https://github.com/{config.author or 'yourusername'}/truthound-plugin-{config.name}"
Repository = "https://github.com/{config.author or 'yourusername'}/truthound-plugin-{config.name}"
Issues = "https://github.com/{config.author or 'yourusername'}/truthound-plugin-{config.name}/issues"

[project.entry-points."truthound.plugins"]
{config.name} = "{pkg_name}:{config.class_name}Plugin"

[tool.setuptools.packages.find]
where = ["."]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "B", "A", "C4", "SIM"]
ignore = ["E501"]

[tool.mypy]
python_version = "{min_python}"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
'''
        result.add_file("pyproject.toml", content)

    def _generate_readme(self, config: ScaffoldConfig, result: ScaffoldResult) -> None:
        """Generate README.md."""
        content = f'''# Truthound Plugin: {config.name}

{config.description or f'A {config.template_variant} plugin for Truthound.'}

## Installation

```bash
pip install truthound-plugin-{config.name}
```

Or install from source:

```bash
git clone https://github.com/{config.author or 'yourusername'}/truthound-plugin-{config.name}
cd truthound-plugin-{config.name}
pip install -e .
```

## Usage

The plugin is automatically discovered by Truthound:

```python
from truthound.plugins import get_plugin_manager

# Get plugin manager
manager = get_plugin_manager()

# Discover and load plugins
manager.discover_plugins()
manager.load_plugin("{config.name}")

# List loaded plugins
print(manager.registry.list_names())
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/{config.author or 'yourusername'}/truthound-plugin-{config.name}
cd truthound-plugin-{config.name}

# Install in development mode
pip install -e ".[dev]"
```

### Testing

```bash
pytest
```

### Linting

```bash
ruff check .
ruff format .
mypy .
```

## License

{config.license_type}

---

*Version: {config.version}*
*Author: {config.author or 'Unknown'}*
'''
        result.add_file("README.md", content)

    def _generate_package_init(
        self, config: ScaffoldConfig, result: ScaffoldResult, pkg_name: str
    ) -> None:
        """Generate package __init__.py."""
        content = f'''"""Truthound plugin: {config.name}

{config.description or f'A {config.template_variant} plugin for Truthound.'}
"""

from {pkg_name}.plugin import {config.class_name}Plugin

__version__ = "{config.version}"
__all__ = ["{config.class_name}Plugin"]
'''
        result.add_file(f"{pkg_name}/__init__.py", content)

    def _generate_validator_plugin(
        self, config: ScaffoldConfig, result: ScaffoldResult, pkg_name: str
    ) -> None:
        """Generate validator plugin template."""
        content = f'''{self._get_header(config)}

from __future__ import annotations

from typing import Any

import polars as pl

from truthound.plugins import ValidatorPlugin, PluginInfo, PluginType
from truthound.validators.base import Validator, ValidatorConfig, ValidationIssue
from truthound.types import Severity


class {config.class_name}Validator(Validator):
    """Custom validator provided by {config.name} plugin.

    TODO: Implement your validation logic here.

    Example:
        >>> validator = {config.class_name}Validator()
        >>> issues = validator.validate(lf)
    """

    name = "{config.name}"
    category = "custom"
    default_severity = Severity.MEDIUM

    def __init__(
        self,
        config: ValidatorConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the validator.

        Args:
            config: Optional validator configuration.
            **kwargs: Additional arguments.
        """
        super().__init__(config, **kwargs)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Perform validation.

        Args:
            lf: Polars LazyFrame to validate.

        Returns:
            List of validation issues found.
        """
        issues: list[ValidationIssue] = []
        total_rows = lf.select(pl.len()).collect().item()

        if total_rows == 0:
            return issues

        # TODO: Implement your validation logic
        # Example:
        # for col in self._get_target_columns(lf):
        #     violations = lf.filter(pl.col(col).is_null()).collect()
        #     if violations.height > 0:
        #         issues.append(ValidationIssue(
        #             column=col,
        #             issue_type=self.name,
        #             count=violations.height,
        #             severity=self.default_severity,
        #             details=f"Found {{violations.height}} issues",
        #         ))

        return issues


class {config.class_name}Plugin(ValidatorPlugin):
    """Plugin that provides custom validators.

    This plugin is automatically discovered and can be loaded using:

        manager.load_plugin("{config.name}")
    """

    def _get_plugin_name(self) -> str:
        return "{config.name}"

    def _get_plugin_version(self) -> str:
        return "{config.version}"

    def _get_description(self) -> str:
        return "{config.description or 'Custom validators for Truthound'}"

    def _get_author(self) -> str:
        return "{config.author or ''}"

    def get_validators(self) -> list[type[Validator]]:
        """Return validator classes to register.

        Returns:
            List of validator classes.
        """
        return [{config.class_name}Validator]
'''
        result.add_file(f"{pkg_name}/plugin.py", content)

    def _generate_reporter_plugin(
        self, config: ScaffoldConfig, result: ScaffoldResult, pkg_name: str
    ) -> None:
        """Generate reporter plugin template."""
        content = f'''{self._get_header(config)}

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from truthound.plugins import ReporterPlugin, PluginInfo, PluginType
from truthound.reporters.base import ValidationReporter, ReporterConfig

if TYPE_CHECKING:
    from truthound.stores.results import ValidationResult


@dataclass
class {config.class_name}ReporterConfig(ReporterConfig):
    """Configuration for {config.class_name} reporter."""

    include_passed: bool = False


class {config.class_name}Reporter(ValidationReporter[{config.class_name}ReporterConfig]):
    """Custom reporter provided by {config.name} plugin.

    Example:
        >>> reporter = {config.class_name}Reporter()
        >>> output = reporter.render(validation_result)
    """

    name = "{config.name}"
    file_extension = ".txt"
    content_type = "text/plain"

    @classmethod
    def _default_config(cls) -> {config.class_name}ReporterConfig:
        return {config.class_name}ReporterConfig()

    def render(self, data: "ValidationResult") -> str:
        """Render validation result to string.

        Args:
            data: Validation result to render.

        Returns:
            Rendered string.
        """
        lines = [
            f"Validation Report: {{data.data_asset}}",
            f"Status: {{data.status.value}}",
            f"Total Issues: {{len([r for r in data.results if not r.success])}}",
            "",
        ]

        for result in data.results:
            if not result.success or self._config.include_passed:
                lines.append(
                    f"- {{result.column or 'table'}}: "
                    f"{{result.issue_type}} ({{result.severity}})"
                )

        return "\\n".join(lines)


class {config.class_name}Plugin(ReporterPlugin):
    """Plugin that provides custom reporters."""

    def _get_plugin_name(self) -> str:
        return "{config.name}"

    def _get_plugin_version(self) -> str:
        return "{config.version}"

    def _get_description(self) -> str:
        return "{config.description or 'Custom reporter for Truthound'}"

    def _get_author(self) -> str:
        return "{config.author or ''}"

    def get_reporters(self) -> dict[str, type[ValidationReporter]]:
        """Return reporter classes to register.

        Returns:
            Dictionary of reporter name to class.
        """
        return {{"{config.name}": {config.class_name}Reporter}}
'''
        result.add_file(f"{pkg_name}/plugin.py", content)

    def _generate_hook_plugin(
        self, config: ScaffoldConfig, result: ScaffoldResult, pkg_name: str
    ) -> None:
        """Generate hook plugin template."""
        content = f'''{self._get_header(config)}

from __future__ import annotations

import logging
from typing import Any, Callable

from truthound.plugins import HookPlugin, PluginInfo, PluginType, HookType

logger = logging.getLogger(__name__)


def on_validation_start(
    datasource: Any,
    validators: list[Any],
    **kwargs: Any,
) -> None:
    """Called before validation starts.

    Args:
        datasource: The data source being validated.
        validators: List of validators to run.
        **kwargs: Additional context.
    """
    logger.info(f"[{config.name}] Starting validation on {{datasource}}")
    logger.debug(f"[{config.name}] Validators: {{[v.name for v in validators]}}")


def on_validation_complete(
    datasource: Any,
    result: Any,
    issues: list[Any],
    **kwargs: Any,
) -> None:
    """Called after validation completes.

    Args:
        datasource: The data source that was validated.
        result: The validation result.
        issues: List of issues found.
        **kwargs: Additional context.
    """
    logger.info(
        f"[{config.name}] Validation complete: "
        f"{{len(issues)}} issues found"
    )


def on_validator_start(
    validator: Any,
    datasource: Any,
    **kwargs: Any,
) -> None:
    """Called before each validator runs.

    Args:
        validator: The validator about to run.
        datasource: The data source.
        **kwargs: Additional context.
    """
    logger.debug(f"[{config.name}] Starting validator: {{validator.name}}")


def on_validator_complete(
    validator: Any,
    issues: list[Any],
    duration: float,
    **kwargs: Any,
) -> None:
    """Called after each validator completes.

    Args:
        validator: The validator that completed.
        issues: Issues found by this validator.
        duration: Execution time in seconds.
        **kwargs: Additional context.
    """
    logger.debug(
        f"[{config.name}] Validator {{validator.name}} completed in "
        f"{{duration:.3f}}s with {{len(issues)}} issues"
    )


class {config.class_name}Plugin(HookPlugin):
    """Plugin that provides event hooks.

    Available hooks:
        - BEFORE_VALIDATION: Called before validation starts
        - AFTER_VALIDATION: Called after validation completes
        - BEFORE_VALIDATOR: Called before each validator
        - AFTER_VALIDATOR: Called after each validator
    """

    def _get_plugin_name(self) -> str:
        return "{config.name}"

    def _get_plugin_version(self) -> str:
        return "{config.version}"

    def _get_description(self) -> str:
        return "{config.description or 'Event hooks for Truthound'}"

    def _get_author(self) -> str:
        return "{config.author or ''}"

    def get_hooks(self) -> dict[str, Callable[..., Any]]:
        """Return hooks to register.

        Returns:
            Dictionary of hook type to callback function.
        """
        return {{
            HookType.BEFORE_VALIDATION.value: on_validation_start,
            HookType.AFTER_VALIDATION.value: on_validation_complete,
            HookType.BEFORE_VALIDATOR.value: on_validator_start,
            HookType.AFTER_VALIDATOR.value: on_validator_complete,
        }}
'''
        result.add_file(f"{pkg_name}/plugin.py", content)

    def _generate_datasource_plugin(
        self, config: ScaffoldConfig, result: ScaffoldResult, pkg_name: str
    ) -> None:
        """Generate datasource plugin template."""
        content = f'''{self._get_header(config)}

from __future__ import annotations

from typing import Any

import polars as pl

from truthound.plugins import DataSourcePlugin, PluginInfo, PluginType
from truthound.datasources.base import DataSource, DataSourceConfig


class {config.class_name}DataSource(DataSource):
    """Custom data source provided by {config.name} plugin.

    TODO: Implement connection and data reading logic.

    Example:
        >>> ds = {config.class_name}DataSource(connection_string="...")
        >>> lf = ds.read()
    """

    name = "{config.name}"

    def __init__(
        self,
        connection_string: str | None = None,
        config: DataSourceConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the data source.

        Args:
            connection_string: Connection string for the data source.
            config: Data source configuration.
            **kwargs: Additional arguments.
        """
        super().__init__(config, **kwargs)
        self._connection_string = connection_string
        self._connection = None

    def connect(self) -> None:
        """Establish connection to the data source."""
        # TODO: Implement connection logic
        pass

    def disconnect(self) -> None:
        """Close connection to the data source."""
        # TODO: Implement disconnection logic
        self._connection = None

    def read(
        self,
        query: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> pl.LazyFrame:
        """Read data from the source.

        Args:
            query: Optional query or table name.
            limit: Maximum rows to read.
            **kwargs: Additional read options.

        Returns:
            Polars LazyFrame with the data.
        """
        # TODO: Implement data reading logic
        # Example:
        # if query:
        #     result = self._execute_query(query)
        # else:
        #     result = self._read_all()
        #
        # if limit:
        #     result = result.head(limit)
        #
        # return result.lazy()

        raise NotImplementedError("Implement read() method")

    def get_schema(self) -> dict[str, Any]:
        """Get schema information from the data source.

        Returns:
            Dictionary with schema information.
        """
        # TODO: Implement schema retrieval
        return {{}}


class {config.class_name}Plugin(DataSourcePlugin):
    """Plugin that provides custom data source connectors."""

    def _get_plugin_name(self) -> str:
        return "{config.name}"

    def _get_plugin_version(self) -> str:
        return "{config.version}"

    def _get_description(self) -> str:
        return "{config.description or 'Custom data source for Truthound'}"

    def _get_author(self) -> str:
        return "{config.author or ''}"

    def get_datasources(self) -> dict[str, type[DataSource]]:
        """Return data source classes to register.

        Returns:
            Dictionary of data source name to class.
        """
        return {{"{config.name}": {config.class_name}DataSource}}
'''
        result.add_file(f"{pkg_name}/plugin.py", content)

    def _generate_action_plugin(
        self, config: ScaffoldConfig, result: ScaffoldResult, pkg_name: str
    ) -> None:
        """Generate checkpoint action plugin template."""
        content = f'''{self._get_header(config)}

from __future__ import annotations

import logging
from typing import Any

from truthound.plugins import ActionPlugin, PluginInfo, PluginType
from truthound.checkpoint.actions.base import CheckpointAction, ActionConfig
from truthound.checkpoint.core import CheckpointResult

logger = logging.getLogger(__name__)


class {config.class_name}ActionConfig(ActionConfig):
    """Configuration for {config.class_name} action.

    Attributes:
        enabled: Whether the action is enabled.
        trigger_on: When to trigger (success, failure, always).
        custom_option: Your custom configuration option.
    """

    enabled: bool = True
    trigger_on: str = "always"
    custom_option: str = ""


class {config.class_name}Action(CheckpointAction[{config.class_name}ActionConfig]):
    """Custom checkpoint action provided by {config.name} plugin.

    This action is triggered after checkpoint validation completes.

    Example:
        >>> action = {config.class_name}Action(custom_option="value")
        >>> action.execute(checkpoint_result)
    """

    name = "{config.name}"

    def __init__(
        self,
        config: {config.class_name}ActionConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the action.

        Args:
            config: Action configuration.
            **kwargs: Additional arguments passed to config.
        """
        if config is None:
            config = {config.class_name}ActionConfig(**kwargs)
        super().__init__(config)

    def execute(self, result: CheckpointResult) -> bool:
        """Execute the action.

        Args:
            result: The checkpoint result to act on.

        Returns:
            True if action succeeded, False otherwise.
        """
        if not self._should_trigger(result):
            logger.debug(f"[{config.name}] Skipping - trigger condition not met")
            return True

        try:
            # TODO: Implement your action logic
            logger.info(
                f"[{config.name}] Executing action for checkpoint "
                f"'{{result.checkpoint_name}}'"
            )

            # Example: Log results
            logger.info(f"  Status: {{result.status.value}}")
            logger.info(f"  Issues: {{result.issue_count}}")

            if self._config.custom_option:
                logger.info(f"  Custom option: {{self._config.custom_option}}")

            return True

        except Exception as e:
            logger.error(f"[{config.name}] Action failed: {{e}}")
            return False

    def _should_trigger(self, result: CheckpointResult) -> bool:
        """Check if action should trigger based on configuration.

        Args:
            result: The checkpoint result.

        Returns:
            True if should trigger, False otherwise.
        """
        if not self._config.enabled:
            return False

        trigger = self._config.trigger_on.lower()
        if trigger == "always":
            return True
        elif trigger == "success":
            return result.status.value == "success"
        elif trigger == "failure":
            return result.status.value in ("failure", "error")

        return True


class {config.class_name}Plugin(ActionPlugin):
    """Plugin that provides custom checkpoint actions."""

    def _get_plugin_name(self) -> str:
        return "{config.name}"

    def _get_plugin_version(self) -> str:
        return "{config.version}"

    def _get_description(self) -> str:
        return "{config.description or 'Custom checkpoint action for Truthound'}"

    def _get_author(self) -> str:
        return "{config.author or ''}"

    def get_actions(self) -> dict[str, type[CheckpointAction]]:
        """Return action classes to register.

        Returns:
            Dictionary of action name to class.
        """
        return {{"{config.name}": {config.class_name}Action}}
'''
        result.add_file(f"{pkg_name}/plugin.py", content)

    def _generate_full_plugin(
        self, config: ScaffoldConfig, result: ScaffoldResult, pkg_name: str
    ) -> None:
        """Generate full-featured plugin with multiple components."""
        content = f'''{self._get_header(config)}

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

import polars as pl

from truthound.plugins import (
    Plugin,
    PluginInfo,
    PluginType,
    PluginConfig,
    HookType,
)
from truthound.validators.base import Validator, ValidatorConfig, ValidationIssue
from truthound.reporters.base import ValidationReporter, ReporterConfig
from truthound.types import Severity

if TYPE_CHECKING:
    from truthound.plugins import PluginManager
    from truthound.stores.results import ValidationResult

logger = logging.getLogger(__name__)


# =============================================================================
# Validators
# =============================================================================


class {config.class_name}Validator(Validator):
    """Custom validator provided by {config.name} plugin."""

    name = "{config.name}_validator"
    category = "custom"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        # TODO: Implement validation logic
        return issues


# =============================================================================
# Reporters
# =============================================================================


@dataclass
class {config.class_name}ReporterConfig(ReporterConfig):
    """Configuration for {config.class_name} reporter."""
    include_passed: bool = False


class {config.class_name}Reporter(ValidationReporter[{config.class_name}ReporterConfig]):
    """Custom reporter provided by {config.name} plugin."""

    name = "{config.name}_reporter"
    file_extension = ".txt"
    content_type = "text/plain"

    @classmethod
    def _default_config(cls) -> {config.class_name}ReporterConfig:
        return {config.class_name}ReporterConfig()

    def render(self, data: "ValidationResult") -> str:
        lines = [f"Report: {{data.data_asset}}"]
        for r in data.results:
            if not r.success:
                lines.append(f"- {{r.column}}: {{r.issue_type}}")
        return "\\n".join(lines)


# =============================================================================
# Hooks
# =============================================================================


def on_validation_complete(
    datasource: Any,
    result: Any,
    issues: list[Any],
    **kwargs: Any,
) -> None:
    """Hook called after validation completes."""
    logger.info(f"[{config.name}] Validation complete: {{len(issues)}} issues")


# =============================================================================
# Plugin
# =============================================================================


@dataclass
class {config.class_name}PluginConfig(PluginConfig):
    """Configuration for {config.class_name} plugin.

    Attributes:
        enable_validator: Enable the validator component.
        enable_reporter: Enable the reporter component.
        enable_hooks: Enable event hooks.
    """

    enable_validator: bool = True
    enable_reporter: bool = True
    enable_hooks: bool = True


class {config.class_name}Plugin(Plugin[{config.class_name}PluginConfig]):
    """Full-featured Truthound plugin.

    This plugin provides:
        - Custom validators
        - Custom reporters
        - Event hooks

    Components can be enabled/disabled via configuration.
    """

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="{config.name}",
            version="{config.version}",
            plugin_type=PluginType.CUSTOM,
            description="{config.description or 'Full-featured Truthound plugin'}",
            author="{config.author or ''}",
            tags=["validator", "reporter", "hooks"],
        )

    def setup(self) -> None:
        """Initialize the plugin."""
        logger.info(f"[{config.name}] Plugin setup complete")

    def teardown(self) -> None:
        """Cleanup plugin resources."""
        logger.info(f"[{config.name}] Plugin teardown complete")

    def register(self, manager: "PluginManager") -> None:
        """Register plugin components with the manager.

        Args:
            manager: The plugin manager.
        """
        config = self._config or {config.class_name}PluginConfig()

        # Register validators
        if config.enable_validator:
            manager.register_validator({config.class_name}Validator)
            logger.debug(f"[{config.name}] Registered validator")

        # Register reporters
        if config.enable_reporter:
            manager.register_reporter("{config.name}", {config.class_name}Reporter)
            logger.debug(f"[{config.name}] Registered reporter")

        # Register hooks
        if config.enable_hooks:
            manager.register_hook(
                HookType.AFTER_VALIDATION.value,
                on_validation_complete,
            )
            logger.debug(f"[{config.name}] Registered hooks")
'''
        result.add_file(f"{pkg_name}/plugin.py", content)

    def _generate_tests(
        self, config: ScaffoldConfig, result: ScaffoldResult, pkg_name: str
    ) -> None:
        """Generate test files."""
        # Test __init__.py
        result.add_file("tests/__init__.py", "")

        # Main test file
        content = f'''"""Tests for {config.name} plugin."""

import pytest

from {pkg_name} import {config.class_name}Plugin


class Test{config.class_name}Plugin:
    """Test cases for {config.class_name}Plugin."""

    def test_plugin_info(self):
        """Test plugin info properties."""
        plugin = {config.class_name}Plugin()

        assert plugin.name == "{config.name}"
        assert plugin.version == "{config.version}"

    def test_plugin_setup(self):
        """Test plugin setup."""
        plugin = {config.class_name}Plugin()
        plugin.setup()
        # Should not raise

    def test_plugin_teardown(self):
        """Test plugin teardown."""
        plugin = {config.class_name}Plugin()
        plugin.setup()
        plugin.teardown()
        # Should not raise


class Test{config.class_name}Components:
    """Test plugin components."""

    @pytest.mark.parametrize("variant", ["{config.template_variant}"])
    def test_component_exists(self, variant):
        """Test that expected components exist."""
        plugin = {config.class_name}Plugin()

        # Plugin should have required methods
        assert hasattr(plugin, "setup")
        assert hasattr(plugin, "teardown")
'''
        result.add_file("tests/test_plugin.py", content)
