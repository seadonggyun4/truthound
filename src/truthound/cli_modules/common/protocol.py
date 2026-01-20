"""Command protocol and base classes for CLI commands.

This module provides the core abstractions for CLI command implementations:
    - CommandProtocol: Interface for command implementations
    - CommandResult: Result of command execution
    - BaseCommand: Base class with common functionality
    - CommandRegistry: Central registry for commands
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Protocol,
    TypeVar,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class ExitCode(Enum):
    """Standard CLI exit codes."""

    SUCCESS = 0
    GENERAL_ERROR = 1
    USAGE_ERROR = 2
    DATA_ERROR = 3
    IO_ERROR = 4
    VALIDATION_FAILED = 10
    CONFIGURATION_ERROR = 11
    DEPENDENCY_ERROR = 12


class OutputFormat(Enum):
    """Supported output formats."""

    CONSOLE = "console"
    JSON = "json"
    HTML = "html"
    TABLE = "table"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CommandContext:
    """Context passed to commands during execution.

    Attributes:
        verbose: Enable verbose output
        debug: Enable debug mode
        quiet: Suppress non-essential output
        no_color: Disable colored output
        output_format: Default output format
        working_dir: Current working directory
        config: Additional configuration
    """

    verbose: bool = False
    debug: bool = False
    quiet: bool = False
    no_color: bool = False
    output_format: OutputFormat = OutputFormat.CONSOLE
    working_dir: Path = field(default_factory=Path.cwd)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class CommandResult:
    """Result of command execution.

    Attributes:
        success: Whether the command succeeded
        exit_code: Exit code for the process
        message: Human-readable result message
        data: Structured result data
        errors: List of error messages
        warnings: List of warning messages
    """

    success: bool
    exit_code: ExitCode = ExitCode.SUCCESS
    message: str = ""
    data: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def ok(
        cls,
        message: str = "",
        data: dict[str, Any] | None = None,
    ) -> "CommandResult":
        """Create a successful result.

        Args:
            message: Success message
            data: Result data

        Returns:
            Successful CommandResult
        """
        return cls(success=True, message=message, data=data)

    @classmethod
    def error(
        cls,
        message: str,
        exit_code: ExitCode = ExitCode.GENERAL_ERROR,
        errors: list[str] | None = None,
    ) -> "CommandResult":
        """Create an error result.

        Args:
            message: Error message
            exit_code: Exit code
            errors: Additional error details

        Returns:
            Error CommandResult
        """
        return cls(
            success=False,
            exit_code=exit_code,
            message=message,
            errors=errors or [message],
        )

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.success = False

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "exit_code": self.exit_code.value,
            "message": self.message,
            "data": self.data,
            "errors": self.errors,
            "warnings": self.warnings,
        }


# =============================================================================
# Protocol
# =============================================================================


@runtime_checkable
class CommandProtocol(Protocol):
    """Protocol for CLI command implementations.

    Implement this protocol to create a new CLI command.
    """

    name: ClassVar[str]
    help: ClassVar[str]
    aliases: ClassVar[tuple[str, ...]]

    def execute(self, context: CommandContext, **kwargs: Any) -> CommandResult:
        """Execute the command.

        Args:
            context: Command context
            **kwargs: Command-specific arguments

        Returns:
            Command result
        """
        ...

    def validate_args(self, **kwargs: Any) -> list[str]:
        """Validate command arguments.

        Args:
            **kwargs: Command arguments

        Returns:
            List of validation error messages (empty if valid)
        """
        ...


# =============================================================================
# Base Command Class
# =============================================================================


class BaseCommand:
    """Base class for CLI command implementations.

    Provides common functionality for all commands.
    """

    name: ClassVar[str] = "base"
    help: ClassVar[str] = "Base command"
    aliases: ClassVar[tuple[str, ...]] = ()
    category: ClassVar[str] = "general"

    def __init__(self) -> None:
        """Initialize the command."""
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    def execute(self, context: CommandContext, **kwargs: Any) -> CommandResult:
        """Execute the command.

        Args:
            context: Command context
            **kwargs: Command-specific arguments

        Returns:
            Command result
        """
        # Validate arguments
        validation_errors = self.validate_args(**kwargs)
        if validation_errors:
            return CommandResult.error(
                message="Invalid arguments",
                exit_code=ExitCode.USAGE_ERROR,
                errors=validation_errors,
            )

        # Execute implementation
        try:
            return self._execute(context, **kwargs)
        except Exception as e:
            self.logger.exception("Command execution failed")
            return CommandResult.error(
                message=str(e),
                exit_code=ExitCode.GENERAL_ERROR,
            )

    def validate_args(self, **kwargs: Any) -> list[str]:
        """Validate command arguments.

        Override to add custom validation.

        Args:
            **kwargs: Command arguments

        Returns:
            List of validation error messages
        """
        return []

    @abstractmethod
    def _execute(self, context: CommandContext, **kwargs: Any) -> CommandResult:
        """Execute the command implementation.

        Args:
            context: Command context
            **kwargs: Command-specific arguments

        Returns:
            Command result
        """
        pass

    def _read_file(self, path: Path) -> Any:
        """Read a data file.

        Args:
            path: Path to the file

        Returns:
            Polars LazyFrame

        Raises:
            ValueError: If file type is not supported
        """
        import polars as pl

        suffix = path.suffix.lower()
        readers = {
            ".parquet": pl.scan_parquet,
            ".csv": pl.scan_csv,
            ".json": pl.scan_ndjson,
            ".ndjson": pl.scan_ndjson,
            ".jsonl": pl.scan_ndjson,
        }

        if suffix not in readers:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {list(readers.keys())}"
            )

        return readers[suffix](path)


# =============================================================================
# Command Registry
# =============================================================================


class CommandRegistry:
    """Central registry for CLI commands.

    This class manages all available command implementations and
    provides lookup and execution capabilities.

    Example:
        registry = CommandRegistry()
        registry.register(MyCommand())

        result = registry.execute("my_command", context, **kwargs)
    """

    _instance: ClassVar["CommandRegistry | None"] = None

    def __init__(self) -> None:
        """Initialize the registry."""
        self._commands: dict[str, BaseCommand] = {}
        self._aliases: dict[str, str] = {}
        self._categories: dict[str, list[str]] = {}
        self.logger = logging.getLogger(__name__)

    @classmethod
    def get_instance(cls) -> "CommandRegistry":
        """Get the singleton registry instance.

        Returns:
            The global registry instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, cmd: BaseCommand) -> None:
        """Register a command implementation.

        Args:
            cmd: Command to register

        Raises:
            ValueError: If command name is already registered
        """
        name = cmd.name.lower()

        if name in self._commands:
            raise ValueError(f"Command '{name}' is already registered")

        self._commands[name] = cmd

        # Register aliases
        for alias in cmd.aliases:
            alias_lower = alias.lower()
            if alias_lower in self._aliases:
                self.logger.warning(
                    f"Alias '{alias_lower}' already registered for "
                    f"'{self._aliases[alias_lower]}'"
                )
            self._aliases[alias_lower] = name

        # Track categories
        category = getattr(cmd, "category", "general")
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)

        self.logger.debug(f"Registered command: {name}")

    def unregister(self, name: str) -> bool:
        """Unregister a command.

        Args:
            name: Command name to unregister

        Returns:
            True if unregistered, False if not found
        """
        name_lower = name.lower()
        name_lower = self._aliases.get(name_lower, name_lower)

        if name_lower not in self._commands:
            return False

        cmd = self._commands.pop(name_lower)

        # Remove aliases
        for alias in cmd.aliases:
            self._aliases.pop(alias.lower(), None)

        # Remove from category
        category = getattr(cmd, "category", "general")
        if category in self._categories:
            self._categories[category] = [
                n for n in self._categories[category] if n != name_lower
            ]

        return True

    def get(self, name: str) -> BaseCommand | None:
        """Get a command by name or alias.

        Args:
            name: Command name or alias

        Returns:
            Command instance or None if not found
        """
        name_lower = name.lower()
        name_lower = self._aliases.get(name_lower, name_lower)
        return self._commands.get(name_lower)

    def list_commands(self) -> list[tuple[str, str]]:
        """List all registered commands.

        Returns:
            List of (name, help) tuples
        """
        return [(c.name, c.help) for c in self._commands.values()]

    def list_names(self) -> list[str]:
        """List all command names.

        Returns:
            List of command names
        """
        return list(self._commands.keys())

    def list_by_category(self) -> dict[str, list[tuple[str, str]]]:
        """List commands grouped by category.

        Returns:
            Dictionary of category -> [(name, help), ...]
        """
        result: dict[str, list[tuple[str, str]]] = {}
        for category, names in self._categories.items():
            result[category] = [
                (name, self._commands[name].help)
                for name in names
                if name in self._commands
            ]
        return result

    def execute(
        self,
        command_name: str,
        context: CommandContext,
        **kwargs: Any,
    ) -> CommandResult:
        """Execute a command.

        Args:
            command_name: Name of command to execute
            context: Command context
            **kwargs: Command arguments

        Returns:
            Command result
        """
        cmd = self.get(command_name)

        if cmd is None:
            return CommandResult.error(
                message=f"Unknown command '{command_name}'. "
                f"Available: {', '.join(self.list_names())}",
                exit_code=ExitCode.USAGE_ERROR,
            )

        return cmd.execute(context, **kwargs)

    def __contains__(self, name: str) -> bool:
        """Check if a command is registered."""
        return self.get(name) is not None

    def __len__(self) -> int:
        """Get number of registered commands."""
        return len(self._commands)


def get_command_registry() -> CommandRegistry:
    """Get the global command registry.

    Returns:
        The singleton registry instance
    """
    return CommandRegistry.get_instance()


# =============================================================================
# Decorator
# =============================================================================


CommandT = TypeVar("CommandT", bound=BaseCommand)


def command(
    name: str | None = None,
    help: str | None = None,
    aliases: tuple[str, ...] = (),
    category: str = "general",
) -> Callable[[type[CommandT]], type[CommandT]]:
    """Decorator to register a command class.

    Args:
        name: Optional override for command name
        help: Optional override for help text
        aliases: Additional aliases for the command
        category: Command category

    Returns:
        Decorator function

    Example:
        @command("my_command", "My custom command", category="custom")
        class MyCommand(BaseCommand):
            ...
    """

    def decorator(cls: type[CommandT]) -> type[CommandT]:
        # Override class attributes if provided
        if name:
            cls.name = name
        if help:
            cls.help = help
        if aliases:
            cls.aliases = aliases
        cls.category = category

        # Create instance and register
        instance = cls()
        get_command_registry().register(instance)

        return cls

    return decorator
