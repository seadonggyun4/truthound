"""Custom action for user-defined logic.

This action allows users to define custom logic using Python callables
or shell commands.
"""

from __future__ import annotations

import subprocess
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from truthound.checkpoint.actions.base import (
    ActionConfig,
    ActionResult,
    ActionStatus,
    BaseAction,
    NotifyCondition,
)

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import CheckpointResult


@dataclass
class CustomActionConfig(ActionConfig):
    """Configuration for custom action.

    Attributes:
        callback: Python callable to execute.
        shell_command: Shell command to execute (alternative to callback).
        environment: Additional environment variables for shell commands.
        working_directory: Working directory for shell commands.
        pass_result_as_json: Pass result as JSON to shell command stdin.
        capture_output: Capture command output in action result.
    """

    callback: Callable[["CheckpointResult"], Any] | None = None
    shell_command: str | None = None
    environment: dict[str, str] = field(default_factory=dict)
    working_directory: str | None = None
    pass_result_as_json: bool = True
    capture_output: bool = True
    notify_on: NotifyCondition | str = NotifyCondition.ALWAYS


class CustomAction(BaseAction[CustomActionConfig]):
    """Action to execute custom user-defined logic.

    Allows users to integrate custom logic via Python callables
    or shell commands for maximum flexibility.

    Example:
        >>> # Using a callback
        >>> def my_callback(result):
        ...     print(f"Got result: {result.status}")
        ...     return {"custom_data": "value"}
        >>>
        >>> action = CustomAction(callback=my_callback)
        >>>
        >>> # Using a shell command
        >>> action = CustomAction(
        ...     shell_command="./scripts/notify.sh",
        ...     environment={"API_KEY": "secret"},
        ... )
    """

    action_type = "custom"

    @classmethod
    def _default_config(cls) -> CustomActionConfig:
        return CustomActionConfig()

    def _execute(self, checkpoint_result: "CheckpointResult") -> ActionResult:
        """Execute custom action."""
        config = self._config

        if config.callback:
            return self._execute_callback(checkpoint_result)
        elif config.shell_command:
            return self._execute_shell(checkpoint_result)
        else:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="No callback or shell_command configured",
                error="Either callback or shell_command is required",
            )

    def _execute_callback(self, checkpoint_result: "CheckpointResult") -> ActionResult:
        """Execute Python callback."""
        config = self._config
        callback = config.callback

        if callback is None:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="No callback configured",
            )

        try:
            result = callback(checkpoint_result)

            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.SUCCESS,
                message="Callback executed successfully",
                details={
                    "callback_name": callback.__name__ if hasattr(callback, "__name__") else str(callback),
                    "result": result if isinstance(result, (dict, list, str, int, float, bool)) else str(result),
                },
            )

        except Exception as e:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="Callback execution failed",
                error=str(e),
            )

    def _execute_shell(self, checkpoint_result: "CheckpointResult") -> ActionResult:
        """Execute shell command."""
        import json

        config = self._config

        if not config.shell_command:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="No shell command configured",
            )

        # Build environment
        env = os.environ.copy()
        env.update(config.environment)

        # Add result as environment variables
        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None

        env["TRUTHOUND_STATUS"] = checkpoint_result.status.value
        env["TRUTHOUND_RUN_ID"] = checkpoint_result.run_id
        env["TRUTHOUND_CHECKPOINT"] = checkpoint_result.checkpoint_name
        env["TRUTHOUND_DATA_ASSET"] = checkpoint_result.data_asset or ""
        env["TRUTHOUND_TOTAL_ISSUES"] = str(stats.total_issues if stats else 0)
        env["TRUTHOUND_CRITICAL_ISSUES"] = str(stats.critical_issues if stats else 0)
        env["TRUTHOUND_HIGH_ISSUES"] = str(stats.high_issues if stats else 0)
        env["TRUTHOUND_PASS_RATE"] = str(stats.pass_rate if stats else 1.0)

        # Prepare stdin
        stdin_data = None
        if config.pass_result_as_json:
            stdin_data = json.dumps(checkpoint_result.to_dict(), default=str).encode()

        try:
            result = subprocess.run(
                config.shell_command,
                shell=True,
                env=env,
                cwd=config.working_directory,
                input=stdin_data,
                capture_output=config.capture_output,
                timeout=config.timeout_seconds,
            )

            stdout = result.stdout.decode() if result.stdout else ""
            stderr = result.stderr.decode() if result.stderr else ""

            if result.returncode == 0:
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.SUCCESS,
                    message="Shell command executed successfully",
                    details={
                        "command": config.shell_command,
                        "return_code": result.returncode,
                        "stdout": stdout[:1000] if stdout else None,
                        "stderr": stderr[:1000] if stderr else None,
                    },
                )
            else:
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.FAILURE,
                    message=f"Shell command failed with exit code {result.returncode}",
                    details={
                        "command": config.shell_command,
                        "return_code": result.returncode,
                        "stdout": stdout[:1000] if stdout else None,
                        "stderr": stderr[:1000] if stderr else None,
                    },
                    error=stderr or f"Exit code: {result.returncode}",
                )

        except subprocess.TimeoutExpired:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message=f"Shell command timed out after {config.timeout_seconds}s",
                error="Timeout",
            )
        except Exception as e:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="Shell command execution failed",
                error=str(e),
            )

    def validate_config(self) -> list[str]:
        """Validate configuration."""
        errors = []

        config = self._config
        if not config.callback and not config.shell_command:
            errors.append("Either callback or shell_command is required")

        if config.callback and config.shell_command:
            errors.append("Cannot specify both callback and shell_command")

        return errors
