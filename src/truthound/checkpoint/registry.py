"""Checkpoint registry for global checkpoint management.

This module provides a global registry for checkpoints, allowing
them to be registered, retrieved, and managed by name.
"""

from __future__ import annotations

import json
import yaml
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import Checkpoint


class CheckpointRegistry:
    """Global registry for checkpoints.

    The registry provides a centralized place to manage checkpoints,
    allowing them to be retrieved by name from anywhere in the application.

    Example:
        >>> from truthound.checkpoint import Checkpoint, register_checkpoint, get_checkpoint
        >>>
        >>> checkpoint = Checkpoint(
        ...     name="daily_check",
        ...     data_source="data.csv",
        ... )
        >>>
        >>> register_checkpoint(checkpoint)
        >>>
        >>> # Later, retrieve by name
        >>> cp = get_checkpoint("daily_check")
        >>> result = cp.run()
    """

    _instance: "CheckpointRegistry | None" = None
    _initialized: bool = False

    def __new__(cls) -> "CheckpointRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the registry."""
        if self._initialized:
            return

        self._checkpoints: dict[str, "Checkpoint"] = {}
        self._initialized = True

    def register(self, checkpoint: "Checkpoint", replace: bool = False) -> None:
        """Register a checkpoint.

        Args:
            checkpoint: Checkpoint to register.
            replace: If True, replace existing checkpoint with same name.

        Raises:
            ValueError: If checkpoint with same name already exists and replace=False.
        """
        if checkpoint.name in self._checkpoints and not replace:
            raise ValueError(f"Checkpoint '{checkpoint.name}' already registered")

        self._checkpoints[checkpoint.name] = checkpoint

    def unregister(self, name: str) -> bool:
        """Unregister a checkpoint.

        Args:
            name: Name of checkpoint to unregister.

        Returns:
            True if unregistered, False if not found.
        """
        if name in self._checkpoints:
            del self._checkpoints[name]
            return True
        return False

    def get(self, name: str) -> "Checkpoint":
        """Get a checkpoint by name.

        Args:
            name: Checkpoint name.

        Returns:
            The checkpoint.

        Raises:
            KeyError: If checkpoint not found.
        """
        if name not in self._checkpoints:
            raise KeyError(f"Checkpoint not found: {name}")
        return self._checkpoints[name]

    def exists(self, name: str) -> bool:
        """Check if a checkpoint exists.

        Args:
            name: Checkpoint name.

        Returns:
            True if exists.
        """
        return name in self._checkpoints

    def list_names(self) -> list[str]:
        """List all checkpoint names.

        Returns:
            List of checkpoint names.
        """
        return list(self._checkpoints.keys())

    def list_all(self) -> list["Checkpoint"]:
        """List all checkpoints.

        Returns:
            List of checkpoints.
        """
        return list(self._checkpoints.values())

    def clear(self) -> None:
        """Clear all registered checkpoints."""
        self._checkpoints.clear()

    def __iter__(self) -> Iterator["Checkpoint"]:
        """Iterate over checkpoints."""
        return iter(self._checkpoints.values())

    def __len__(self) -> int:
        """Get number of registered checkpoints."""
        return len(self._checkpoints)

    def __contains__(self, name: str) -> bool:
        """Check if checkpoint exists."""
        return name in self._checkpoints

    def load_from_yaml(self, path: str | Path) -> list["Checkpoint"]:
        """Load checkpoints from a YAML configuration file.

        Args:
            path: Path to YAML file.

        Returns:
            List of loaded checkpoints.
        """
        from truthound.checkpoint.checkpoint import Checkpoint, CheckpointConfig
        from truthound.checkpoint.actions import (
            StoreValidationResult,
            UpdateDataDocs,
            SlackNotification,
            EmailNotification,
            WebhookAction,
            PagerDutyAction,
            GitHubAction,
            CustomAction,
        )
        from truthound.checkpoint.triggers import (
            ScheduleTrigger,
            CronTrigger,
            EventTrigger,
            FileWatchTrigger,
        )

        path = Path(path)
        with open(path) as f:
            config = yaml.safe_load(f)

        loaded = []

        checkpoints_config = config.get("checkpoints", [])
        if isinstance(checkpoints_config, dict):
            checkpoints_config = [
                {"name": name, **cp_config}
                for name, cp_config in checkpoints_config.items()
            ]

        action_classes = {
            "store_result": StoreValidationResult,
            "update_docs": UpdateDataDocs,
            "slack": SlackNotification,
            "email": EmailNotification,
            "webhook": WebhookAction,
            "pagerduty": PagerDutyAction,
            "github": GitHubAction,
            "custom": CustomAction,
        }

        trigger_classes = {
            "schedule": ScheduleTrigger,
            "cron": CronTrigger,
            "event": EventTrigger,
            "file_watch": FileWatchTrigger,
        }

        for cp_config in checkpoints_config:
            # Parse actions
            actions = []
            for action_config in cp_config.get("actions", []):
                action_type = action_config.pop("type", "webhook")
                if action_type in action_classes:
                    actions.append(action_classes[action_type](**action_config))

            # Parse triggers
            triggers = []
            for trigger_config in cp_config.get("triggers", []):
                trigger_type = trigger_config.pop("type", "schedule")
                if trigger_type in trigger_classes:
                    triggers.append(trigger_classes[trigger_type](**trigger_config))

            # Remove actions/triggers from config before passing to Checkpoint
            cp_config.pop("actions", None)
            cp_config.pop("triggers", None)

            # Get valid field names from CheckpointConfig dataclass
            valid_fields = set(CheckpointConfig.__dataclass_fields__.keys())
            checkpoint = Checkpoint(
                config=CheckpointConfig(**{
                    k: v for k, v in cp_config.items()
                    if k in valid_fields
                }),
                actions=actions,
                triggers=triggers,
            )

            self.register(checkpoint, replace=True)
            loaded.append(checkpoint)

        return loaded

    def load_from_json(self, path: str | Path) -> list["Checkpoint"]:
        """Load checkpoints from a JSON configuration file.

        Args:
            path: Path to JSON file.

        Returns:
            List of loaded checkpoints.
        """
        path = Path(path)
        with open(path) as f:
            config = json.load(f)

        # Convert to YAML-style config and use load_from_yaml logic
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tf:
            yaml.dump(config, tf)
            temp_path = tf.name

        try:
            return self.load_from_yaml(temp_path)
        finally:
            Path(temp_path).unlink()

    def save_to_yaml(self, path: str | Path) -> None:
        """Save all checkpoints to a YAML file.

        Args:
            path: Output path.
        """
        path = Path(path)
        config = {
            "checkpoints": [cp.to_dict() for cp in self._checkpoints.values()]
        }
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def save_to_json(self, path: str | Path) -> None:
        """Save all checkpoints to a JSON file.

        Args:
            path: Output path.
        """
        path = Path(path)
        config = {
            "checkpoints": [cp.to_dict() for cp in self._checkpoints.values()]
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)


# Global registry instance
_registry = CheckpointRegistry()


def register_checkpoint(checkpoint: "Checkpoint") -> None:
    """Register a checkpoint in the global registry.

    Args:
        checkpoint: Checkpoint to register.
    """
    _registry.register(checkpoint)


def get_checkpoint(name: str) -> "Checkpoint":
    """Get a checkpoint from the global registry.

    Args:
        name: Checkpoint name.

    Returns:
        The checkpoint.
    """
    return _registry.get(name)


def list_checkpoints() -> list[str]:
    """List all registered checkpoint names.

    Returns:
        List of names.
    """
    return _registry.list_names()


def load_checkpoints(path: str | Path) -> list["Checkpoint"]:
    """Load checkpoints from a configuration file.

    Supports YAML and JSON formats.

    Args:
        path: Path to configuration file.

    Returns:
        List of loaded checkpoints.
    """
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        return _registry.load_from_yaml(path)
    elif path.suffix == ".json":
        return _registry.load_from_json(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
