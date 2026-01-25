"""Core Checkpoint class for orchestrating validation pipelines.

This module provides the main Checkpoint class that ties together
data sources, validators, actions, and triggers into a complete
validation pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from truthound.checkpoint.actions.base import BaseAction, ActionResult
    from truthound.checkpoint.routing.base import ActionRouter
    from truthound.checkpoint.triggers.base import BaseTrigger
    from truthound.datasources.base import BaseDataSource
    from truthound.stores.results import ValidationResult
    from truthound.validators.base import Validator


class CheckpointStatus(str, Enum):
    """Status of a checkpoint run."""

    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    WARNING = "warning"
    RUNNING = "running"
    PENDING = "pending"

    def __str__(self) -> str:
        return self.value


@dataclass
class CheckpointConfig:
    """Configuration for a checkpoint.

    Attributes:
        name: Unique name for this checkpoint.
        data_source: Data source path, DataSource instance, or connection string.
        validators: List of validator names or Validator instances.
        validator_config: Configuration options for validators.
            Maps validator name to configuration dict. Example:
            {"regex": {"patterns": {"email": r"^[\\w.+-]+@[\\w-]+\\.[\\w.-]+$"}}}
        min_severity: Minimum severity level to include.
        schema: Schema file path or Schema instance.
        auto_schema: Automatically learn and validate against schema.
        run_name_template: Template for run IDs (supports strftime).
        tags: Tags to apply to all runs.
        metadata: Additional metadata for the checkpoint.
        fail_on_critical: Mark as failure if critical issues found.
        fail_on_high: Mark as failure if high severity issues found.
        timeout_seconds: Maximum time for validation to complete.
        sample_size: Sample size for large datasets (None = no sampling).
    """

    name: str = "default_checkpoint"
    data_source: str | Any = ""
    validators: list[str | "Validator"] | None = None
    validator_config: dict[str, dict[str, Any]] = field(default_factory=dict)
    min_severity: str | None = None
    schema: str | Path | Any = None
    auto_schema: bool = False
    run_name_template: str = "%Y%m%d_%H%M%S"
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    fail_on_critical: bool = True
    fail_on_high: bool = False
    timeout_seconds: int = 3600
    sample_size: int | None = None


@dataclass
class CheckpointResult:
    """Result of a checkpoint run.

    Attributes:
        run_id: Unique identifier for this run.
        checkpoint_name: Name of the checkpoint that was run.
        run_time: When the checkpoint ran.
        status: Overall status of the run.
        validation_result: The validation result from check().
        action_results: Results from all executed actions.
        data_asset: Name/path of the data that was validated.
        duration_ms: Total execution time in milliseconds.
        error: Error message if checkpoint failed.
        metadata: Additional metadata about the run.
    """

    run_id: str
    checkpoint_name: str
    run_time: datetime
    status: CheckpointStatus
    validation_result: "ValidationResult | None" = None
    action_results: list["ActionResult"] = field(default_factory=list)
    data_asset: str = ""
    duration_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if checkpoint was successful."""
        return self.status == CheckpointStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "checkpoint_name": self.checkpoint_name,
            "run_time": self.run_time.isoformat(),
            "status": self.status.value,
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "action_results": [r.to_dict() for r in self.action_results],
            "data_asset": self.data_asset,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointResult":
        """Create from dictionary."""
        from truthound.checkpoint.actions.base import ActionResult
        from truthound.stores.results import ValidationResult

        return cls(
            run_id=data["run_id"],
            checkpoint_name=data["checkpoint_name"],
            run_time=datetime.fromisoformat(data["run_time"]),
            status=CheckpointStatus(data["status"]),
            validation_result=ValidationResult.from_dict(data["validation_result"]) if data.get("validation_result") else None,
            action_results=[ActionResult.from_dict(r) for r in data.get("action_results", [])],
            data_asset=data.get("data_asset", ""),
            duration_ms=data.get("duration_ms", 0.0),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )

    def summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            f"Checkpoint: {self.checkpoint_name}",
            f"Status: {self.status.value.upper()}",
            f"Run ID: {self.run_id}",
            f"Run Time: {self.run_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {self.duration_ms:.2f} ms",
        ]

        if self.validation_result:
            stats = self.validation_result.statistics
            lines.extend([
                f"Total Issues: {stats.total_issues}",
                f"  Critical: {stats.critical_issues}",
                f"  High: {stats.high_issues}",
                f"  Medium: {stats.medium_issues}",
                f"  Low: {stats.low_issues}",
            ])

        if self.action_results:
            lines.append(f"Actions: {len(self.action_results)}")
            for action in self.action_results:
                lines.append(f"  - {action.action_name}: {action.status.value}")

        if self.error:
            lines.append(f"Error: {self.error}")

        return "\n".join(lines)


class Checkpoint:
    """Orchestrates data quality validation pipelines.

    A Checkpoint combines a data source, validators, and actions
    into a reusable validation pipeline that can be run manually
    or automatically via triggers.

    Example:
        >>> from truthound.checkpoint import Checkpoint
        >>> from truthound.checkpoint.actions import (
        ...     StoreValidationResult,
        ...     SlackNotification,
        ... )
        >>>
        >>> checkpoint = Checkpoint(
        ...     name="daily_user_validation",
        ...     data_source="users.csv",
        ...     validators=["null", "duplicate", "range"],
        ...     actions=[
        ...         StoreValidationResult(store_path="./results"),
        ...         SlackNotification(
        ...             webhook_url="https://hooks.slack.com/...",
        ...             notify_on="failure",
        ...         ),
        ...     ],
        ... )
        >>>
        >>> result = checkpoint.run()
        >>> print(result.summary())
    """

    def __init__(
        self,
        name: str | None = None,
        config: CheckpointConfig | None = None,
        data_source: str | "BaseDataSource" | None = None,
        validators: list[str | "Validator"] | None = None,
        actions: list["BaseAction[Any]"] | None = None,
        triggers: list["BaseTrigger[Any]"] | None = None,
        router: "ActionRouter | None" = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a checkpoint.

        Args:
            name: Checkpoint name.
            config: Full configuration object.
            data_source: Data source path or DataSource instance.
            validators: List of validators to run.
            actions: Actions to execute after validation (bypasses router).
            triggers: Triggers for automated execution.
            router: Optional ActionRouter for rule-based action routing.
                When provided, actions are selected based on routing rules
                instead of the static actions list (unless use_router=False
                in run()).
            **kwargs: Additional config options.
        """
        # Build config
        if config:
            self._config = config
        else:
            self._config = CheckpointConfig()

        # Apply name if provided
        if name:
            self._config.name = name

        # Apply data source
        if data_source is not None:
            self._config.data_source = data_source

        # Apply validators
        if validators:
            self._config.validators = validators

        # Apply kwargs
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        self._actions = actions or []
        self._triggers = triggers or []
        self._router = router

        # Attach triggers to this checkpoint
        for trigger in self._triggers:
            trigger.attach(self)

    @property
    def name(self) -> str:
        """Get checkpoint name."""
        return self._config.name

    @property
    def config(self) -> CheckpointConfig:
        """Get checkpoint configuration."""
        return self._config

    @property
    def actions(self) -> list["BaseAction[Any]"]:
        """Get configured actions."""
        return self._actions

    @property
    def triggers(self) -> list["BaseTrigger[Any]"]:
        """Get configured triggers."""
        return self._triggers

    @property
    def router(self) -> "ActionRouter | None":
        """Get the action router."""
        return self._router

    @router.setter
    def router(self, value: "ActionRouter | None") -> None:
        """Set the action router."""
        self._router = value

    def set_router(self, router: "ActionRouter") -> "Checkpoint":
        """Set the action router.

        Args:
            router: ActionRouter instance for rule-based routing.

        Returns:
            Self for chaining.
        """
        self._router = router
        return self

    def add_action(self, action: "BaseAction[Any]") -> "Checkpoint":
        """Add an action to the checkpoint.

        Args:
            action: Action to add.

        Returns:
            Self for chaining.
        """
        self._actions.append(action)
        return self

    def add_trigger(self, trigger: "BaseTrigger[Any]") -> "Checkpoint":
        """Add a trigger to the checkpoint.

        Args:
            trigger: Trigger to add.

        Returns:
            Self for chaining.
        """
        trigger.attach(self)
        self._triggers.append(trigger)
        return self

    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now().strftime(self._config.run_name_template)
        unique_id = uuid4().hex[:8]
        return f"{self._config.name}_{timestamp}_{unique_id}"

    def _resolve_data_source(self) -> tuple[Any, str]:
        """Resolve the data source to a usable format.

        Returns:
            Tuple of (data_source, data_asset_name)
        """
        from truthound.datasources.base import BaseDataSource

        source = self._config.data_source

        if isinstance(source, BaseDataSource):
            return source, source.name or str(source)

        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists():
                return str(path), str(path)
            # Might be a connection string
            return source, str(source)

        # Assume it's a DataFrame or similar
        return source, type(source).__name__

    def run(
        self,
        run_id: str | None = None,
        context: dict[str, Any] | None = None,
        use_router: bool = True,
    ) -> CheckpointResult:
        """Run the checkpoint validation pipeline.

        Args:
            run_id: Optional custom run ID.
            context: Additional context for the run.
            use_router: If True and a router is configured, use rule-based
                routing to select actions. If False, use the static actions
                list. Default is True.

        Returns:
            CheckpointResult with validation and action results.
        """
        from truthound.api import check
        from truthound.stores.results import ValidationResult

        run_id = run_id or self._generate_run_id()
        run_time = datetime.now()
        start_time = time.time()

        # Resolve data source
        try:
            data_source, data_asset = self._resolve_data_source()
        except Exception as e:
            return CheckpointResult(
                run_id=run_id,
                checkpoint_name=self.name,
                run_time=run_time,
                status=CheckpointStatus.ERROR,
                data_asset="",
                duration_ms=(time.time() - start_time) * 1000,
                error=f"Failed to resolve data source: {e}",
            )

        # Run validation
        try:
            # Determine if source is a DataSource or raw data
            from truthound.datasources.base import BaseDataSource

            if isinstance(data_source, BaseDataSource):
                # Apply sampling if configured
                if self._config.sample_size and data_source.needs_sampling():
                    data_source = data_source.sample(n=self._config.sample_size)

                report = check(
                    source=data_source,
                    validators=self._config.validators,
                    validator_config=self._config.validator_config,
                    min_severity=self._config.min_severity,
                    schema=self._config.schema,
                    auto_schema=self._config.auto_schema,
                )
            else:
                report = check(
                    data=data_source,
                    validators=self._config.validators,
                    validator_config=self._config.validator_config,
                    min_severity=self._config.min_severity,
                    schema=self._config.schema,
                    auto_schema=self._config.auto_schema,
                )

            # Convert to ValidationResult
            validation_result = ValidationResult.from_report(
                report=report,
                data_asset=data_asset,
                run_id=run_id,
                tags=self._config.tags,
                metadata=self._config.metadata,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

            # Determine status
            if report.has_critical and self._config.fail_on_critical:
                status = CheckpointStatus.FAILURE
            elif report.has_high and self._config.fail_on_high:
                status = CheckpointStatus.FAILURE
            elif report.has_issues:
                status = CheckpointStatus.WARNING
            else:
                status = CheckpointStatus.SUCCESS

        except Exception as e:
            # Validation failed
            validation_result = None
            status = CheckpointStatus.ERROR
            error_msg = str(e)

            checkpoint_result = CheckpointResult(
                run_id=run_id,
                checkpoint_name=self.name,
                run_time=run_time,
                status=status,
                validation_result=validation_result,
                data_asset=data_asset,
                duration_ms=(time.time() - start_time) * 1000,
                error=error_msg,
                metadata=context or {},
            )

            # Still run actions (they might want to notify about errors)
            self._execute_actions(checkpoint_result, use_router=use_router)

            return checkpoint_result

        # Create checkpoint result
        checkpoint_result = CheckpointResult(
            run_id=run_id,
            checkpoint_name=self.name,
            run_time=run_time,
            status=status,
            validation_result=validation_result,
            data_asset=data_asset,
            duration_ms=(time.time() - start_time) * 1000,
            metadata=context or {},
        )

        # Execute actions
        self._execute_actions(checkpoint_result, use_router=use_router)

        # Update duration after actions
        checkpoint_result.duration_ms = (time.time() - start_time) * 1000

        return checkpoint_result

    def _execute_actions(
        self,
        checkpoint_result: CheckpointResult,
        use_router: bool = True,
    ) -> None:
        """Execute configured actions, optionally using rule-based routing.

        Args:
            checkpoint_result: The result to pass to actions.
            use_router: If True and a router is configured, use rule-based
                routing. Otherwise, execute all static actions.
        """
        # Use router if available and enabled
        if use_router and self._router is not None:
            routing_result = self._router.route(
                checkpoint_result, execute_actions=True
            )
            # Copy action results from routing
            for action_result in routing_result.action_results:
                checkpoint_result.action_results.append(action_result)
                # Check for action failures
                if not action_result.success:
                    for action in routing_result.executed_actions:
                        if (
                            action.name == action_result.action_name
                            and action.config.fail_checkpoint_on_error
                        ):
                            checkpoint_result.status = CheckpointStatus.ERROR
                            if not checkpoint_result.error:
                                checkpoint_result.error = (
                                    f"Action failed: {action.name}"
                                )
                            break
            return

        # Fallback to static actions list
        for action in self._actions:
            try:
                result = action.execute(checkpoint_result)
                checkpoint_result.action_results.append(result)

                # Check if action failure should fail checkpoint
                if not result.success and action.config.fail_checkpoint_on_error:
                    checkpoint_result.status = CheckpointStatus.ERROR
                    if not checkpoint_result.error:
                        checkpoint_result.error = f"Action failed: {action.name}"

            except Exception as e:
                from truthound.checkpoint.actions.base import ActionResult, ActionStatus

                error_result = ActionResult(
                    action_name=action.name,
                    action_type=action.action_type,
                    status=ActionStatus.ERROR,
                    message="Action execution failed",
                    error=str(e),
                )
                checkpoint_result.action_results.append(error_result)

    def validate(self) -> list[str]:
        """Validate the checkpoint configuration.

        Performs comprehensive validation including:
        - Required fields (name, data_source)
        - Validator names existence
        - validator_config structure
        - min_severity values
        - Data source file existence (for file paths)
        - Schema file existence (for file paths)
        - Action configurations
        - Trigger configurations

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        # Required fields
        if not self._config.name:
            errors.append("Checkpoint name is required")

        if not self._config.data_source:
            errors.append("Data source is required")

        # Validate data source exists (for file paths)
        if self._config.data_source:
            source = self._config.data_source
            if isinstance(source, (str, Path)):
                source_path = Path(source)
                # Only validate if it looks like a file path (has extension like .csv, .parquet)
                # Skip connection strings (postgresql://, mysql://, etc.)
                source_str = str(source)
                is_connection_string = "://" in source_str
                has_file_extension = source_path.suffix in (
                    ".csv", ".parquet", ".json", ".jsonl", ".xlsx", ".xls",
                    ".tsv", ".feather", ".arrow", ".avro", ".orc",
                )
                if has_file_extension and not is_connection_string:
                    if not source_path.exists():
                        errors.append(f"Data source file not found: {source}")

        # Validate validators exist
        if self._config.validators:
            from truthound.validators import get_validator

            for v in self._config.validators:
                if isinstance(v, str):
                    try:
                        get_validator(v)
                    except (KeyError, ValueError) as e:
                        errors.append(f"Unknown validator: '{v}'")

        # Validate validator_config structure
        if self._config.validator_config:
            for validator_name, config in self._config.validator_config.items():
                if not isinstance(config, dict):
                    errors.append(
                        f"validator_config['{validator_name}'] must be a dict, "
                        f"got {type(config).__name__}"
                    )
                # Check if the validator exists
                if self._config.validators:
                    if validator_name not in self._config.validators:
                        errors.append(
                            f"validator_config references '{validator_name}' "
                            f"but it's not in validators list"
                        )

        # Validate regex validator requires pattern parameter
        if self._config.validators:
            for v in self._config.validators:
                if isinstance(v, str) and v.lower() == "regex":
                    regex_config = (self._config.validator_config or {}).get("regex", {})
                    if not regex_config.get("pattern"):
                        errors.append(
                            "RegexValidator requires 'pattern' parameter. "
                            "Add to validator_config: regex: {pattern: '^your-pattern$'}"
                        )

        # Validate min_severity
        valid_severities = {"low", "medium", "high", "critical"}
        if self._config.min_severity:
            if self._config.min_severity.lower() not in valid_severities:
                errors.append(
                    f"Invalid min_severity: '{self._config.min_severity}'. "
                    f"Must be one of: {', '.join(sorted(valid_severities))}"
                )

        # Validate schema file exists (for file paths)
        if self._config.schema:
            schema = self._config.schema
            if isinstance(schema, (str, Path)):
                schema_path = Path(schema)
                if schema_path.suffix in (".yaml", ".yml", ".json") and not schema_path.exists():
                    errors.append(f"Schema file not found: {schema}")

        # Validate actions
        for action in self._actions:
            action_errors = action.validate_config()
            for err in action_errors:
                errors.append(f"Action '{action.name}': {err}")

        # Validate triggers
        for trigger in self._triggers:
            trigger_errors = trigger.validate_config()
            for err in trigger_errors:
                errors.append(f"Trigger '{trigger.name}': {err}")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize checkpoint configuration to dictionary."""
        result = {
            "name": self._config.name,
            "data_source": str(self._config.data_source),
            "validators": self._config.validators,
            "min_severity": self._config.min_severity,
            "schema": str(self._config.schema) if self._config.schema else None,
            "auto_schema": self._config.auto_schema,
            "run_name_template": self._config.run_name_template,
            "tags": self._config.tags,
            "metadata": self._config.metadata,
            "actions": [a.action_type for a in self._actions],
            "triggers": [t.trigger_type for t in self._triggers],
        }

        # Add router info if configured
        if self._router is not None:
            result["router"] = {
                "mode": self._router.mode.value,
                "routes": [r.name for r in self._router.routes],
            }

        return result

    def __repr__(self) -> str:
        router_info = f", router={len(self._router)}" if self._router else ""
        return f"Checkpoint(name={self.name!r}, actions={len(self._actions)}, triggers={len(self._triggers)}{router_info})"
