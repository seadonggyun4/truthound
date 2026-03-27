"""Private shared validation helpers for top-level checkpoint configs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

FILE_SOURCE_SUFFIXES = frozenset(
    {
        ".csv",
        ".parquet",
        ".json",
        ".jsonl",
        ".xlsx",
        ".xls",
        ".tsv",
        ".feather",
        ".arrow",
        ".avro",
        ".orc",
    }
)
SCHEMA_FILE_SUFFIXES = frozenset({".yaml", ".yml", ".json"})
VALID_SEVERITIES = frozenset({"low", "medium", "high", "critical"})
VALID_ASYNC_EXECUTION_STRATEGIES = frozenset(
    {"sequential", "concurrent", "pipeline"}
)


def validate_checkpoint_config(
    config: Any,
    *,
    actions: Sequence[Any],
    triggers: Sequence[Any],
    validate_async: bool = False,
) -> list[str]:
    """Validate shared checkpoint configuration and optional async settings."""
    errors: list[str] = []

    _validate_required_fields(config, errors)
    _validate_data_source(config, errors)
    _validate_validators(config, errors)
    _validate_validator_config(config, errors)
    _validate_regex_requirements(config, errors)
    _validate_min_severity(config, errors)
    _validate_schema(config, errors)
    _validate_components(actions, "Action", errors)
    _validate_components(triggers, "Trigger", errors)

    if validate_async:
        _validate_async_options(config, errors)

    return errors


def _validate_required_fields(config: Any, errors: list[str]) -> None:
    if not getattr(config, "name", None):
        errors.append("Checkpoint name is required")
    if not getattr(config, "data_source", None):
        errors.append("Data source is required")


def _validate_data_source(config: Any, errors: list[str]) -> None:
    source = getattr(config, "data_source", None)
    if not source or not isinstance(source, (str, Path)):
        return

    source_path = Path(source)
    source_str = str(source)
    is_connection_string = "://" in source_str
    has_file_extension = source_path.suffix in FILE_SOURCE_SUFFIXES
    if has_file_extension and not is_connection_string and not source_path.exists():
        errors.append(f"Data source file not found: {source}")


def _validate_validators(config: Any, errors: list[str]) -> None:
    validators = getattr(config, "validators", None) or []
    if not validators:
        return

    from truthound.validators import get_validator

    for validator in validators:
        if not isinstance(validator, str):
            continue
        try:
            get_validator(validator)
        except (KeyError, ValueError):
            errors.append(f"Unknown validator: '{validator}'")


def _validate_validator_config(config: Any, errors: list[str]) -> None:
    validator_config = getattr(config, "validator_config", None) or {}
    if not validator_config:
        return

    configured_validator_names = {
        validator
        for validator in (getattr(config, "validators", None) or [])
        if isinstance(validator, str)
    }

    for validator_name, validator_options in validator_config.items():
        if not isinstance(validator_options, dict):
            errors.append(
                f"validator_config['{validator_name}'] must be a dict, "
                f"got {type(validator_options).__name__}"
            )
        if configured_validator_names and validator_name not in configured_validator_names:
            errors.append(
                f"validator_config references '{validator_name}' "
                f"but it's not in validators list"
            )


def _validate_regex_requirements(config: Any, errors: list[str]) -> None:
    validators = getattr(config, "validators", None) or []
    has_regex = any(
        isinstance(validator, str) and validator.lower() == "regex"
        for validator in validators
    )
    if not has_regex:
        return

    regex_config = (getattr(config, "validator_config", None) or {}).get("regex", {})
    if regex_config.get("pattern"):
        return

    errors.append(
        "RegexValidator requires 'pattern' parameter. "
        "Add to validator_config: regex: {pattern: '^your-pattern$'}"
    )


def _validate_min_severity(config: Any, errors: list[str]) -> None:
    min_severity = getattr(config, "min_severity", None)
    if min_severity and str(min_severity).lower() not in VALID_SEVERITIES:
        errors.append(
            f"Invalid min_severity: '{min_severity}'. "
            f"Must be one of: {', '.join(sorted(VALID_SEVERITIES))}"
        )


def _validate_schema(config: Any, errors: list[str]) -> None:
    schema = getattr(config, "schema", None)
    if not schema or not isinstance(schema, (str, Path)):
        return

    schema_path = Path(schema)
    if schema_path.suffix in SCHEMA_FILE_SUFFIXES and not schema_path.exists():
        errors.append(f"Schema file not found: {schema}")


def _validate_components(
    components: Sequence[Any],
    component_label: str,
    errors: list[str],
) -> None:
    for component in components:
        validate_config = getattr(component, "validate_config", None)
        if validate_config is None:
            continue
        for error in validate_config():
            component_name = getattr(component, "name", type(component).__name__)
            errors.append(f"{component_label} '{component_name}': {error}")


def _validate_async_options(config: Any, errors: list[str]) -> None:
    max_concurrent_actions = getattr(config, "max_concurrent_actions", 1)
    if max_concurrent_actions <= 0:
        errors.append("max_concurrent_actions must be greater than 0")

    action_timeout = getattr(config, "action_timeout", 0.0)
    if action_timeout < 0:
        errors.append("action_timeout must be non-negative")

    executor_workers = getattr(config, "executor_workers", 1)
    if executor_workers <= 0:
        errors.append("executor_workers must be greater than 0")

    execution_strategy = getattr(config, "execution_strategy", None)
    if not isinstance(execution_strategy, str):
        errors.append("execution_strategy must be a string")
        return

    if execution_strategy not in VALID_ASYNC_EXECUTION_STRATEGIES:
        errors.append(
            f"Invalid execution_strategy: '{execution_strategy}'. "
            f"Must be one of: {', '.join(sorted(VALID_ASYNC_EXECUTION_STRATEGIES))}"
        )
