"""CLI error handling utilities.

This module provides standardized error handling for CLI commands.
"""

from __future__ import annotations

import functools
import logging
import sys
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeVar

import typer

logger = logging.getLogger(__name__)


# =============================================================================
# Error Codes
# =============================================================================


class ErrorCode(Enum):
    """Standard CLI error codes."""

    # General errors (1-9)
    GENERAL_ERROR = 1
    USAGE_ERROR = 2

    # File errors (10-19)
    FILE_NOT_FOUND = 10
    FILE_NOT_READABLE = 11
    FILE_NOT_WRITABLE = 12
    INVALID_FILE_FORMAT = 13

    # Validation errors (20-29)
    VALIDATION_FAILED = 20
    SCHEMA_ERROR = 21
    CONSTRAINT_ERROR = 22

    # Configuration errors (30-39)
    CONFIG_NOT_FOUND = 30
    CONFIG_INVALID = 31
    CONFIG_PARSE_ERROR = 32

    # Dependency errors (40-49)
    DEPENDENCY_MISSING = 40
    DEPENDENCY_VERSION = 41

    # Data errors (50-59)
    DATA_ERROR = 50
    DATA_EMPTY = 51
    DATA_CORRUPT = 52


# =============================================================================
# Exception Classes
# =============================================================================


class CLIError(Exception):
    """Base exception for CLI errors.

    Attributes:
        message: Error message
        code: Error code
        details: Additional error details
        hint: Helpful hint for resolution
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.GENERAL_ERROR,
        details: dict[str, Any] | None = None,
        hint: str | None = None,
    ) -> None:
        """Initialize CLI error.

        Args:
            message: Error message
            code: Error code
            details: Additional details
            hint: Resolution hint
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.hint = hint

    def __str__(self) -> str:
        """Get string representation."""
        parts = [self.message]
        if self.hint:
            parts.append(f"Hint: {self.hint}")
        return "\n".join(parts)


class FileNotFoundError(CLIError):
    """Error when a file is not found."""

    def __init__(self, path: Path | str, hint: str | None = None) -> None:
        """Initialize file not found error.

        Args:
            path: Path to the missing file
            hint: Resolution hint
        """
        super().__init__(
            message=f"File not found: {path}",
            code=ErrorCode.FILE_NOT_FOUND,
            details={"path": str(path)},
            hint=hint or "Check that the file exists and the path is correct.",
        )
        self.path = path


class ValidationError(CLIError):
    """Error when validation fails."""

    def __init__(
        self,
        message: str,
        errors: list[str] | None = None,
        hint: str | None = None,
    ) -> None:
        """Initialize validation error.

        Args:
            message: Error message
            errors: List of validation errors
            hint: Resolution hint
        """
        super().__init__(
            message=message,
            code=ErrorCode.VALIDATION_FAILED,
            details={"errors": errors or []},
            hint=hint,
        )
        self.errors = errors or []


class ConfigurationError(CLIError):
    """Error with configuration."""

    def __init__(
        self,
        message: str,
        config_path: Path | str | None = None,
        hint: str | None = None,
    ) -> None:
        """Initialize configuration error.

        Args:
            message: Error message
            config_path: Path to the configuration file
            hint: Resolution hint
        """
        super().__init__(
            message=message,
            code=ErrorCode.CONFIG_INVALID,
            details={"config_path": str(config_path) if config_path else None},
            hint=hint or "Check the configuration file format and values.",
        )
        self.config_path = config_path


class DependencyError(CLIError):
    """Error with missing dependency."""

    def __init__(
        self,
        package: str,
        install_command: str | None = None,
        hint: str | None = None,
    ) -> None:
        """Initialize dependency error.

        Args:
            package: Missing package name
            install_command: Command to install the package
            hint: Resolution hint
        """
        default_hint = f"Install with: {install_command}" if install_command else None
        super().__init__(
            message=f"Missing required dependency: {package}",
            code=ErrorCode.DEPENDENCY_MISSING,
            details={"package": package, "install_command": install_command},
            hint=hint or default_hint,
        )
        self.package = package
        self.install_command = install_command


class DataError(CLIError):
    """Error with data."""

    def __init__(
        self,
        message: str,
        data_path: Path | str | None = None,
        hint: str | None = None,
    ) -> None:
        """Initialize data error.

        Args:
            message: Error message
            data_path: Path to the data file
            hint: Resolution hint
        """
        super().__init__(
            message=message,
            code=ErrorCode.DATA_ERROR,
            details={"data_path": str(data_path) if data_path else None},
            hint=hint,
        )
        self.data_path = data_path


# =============================================================================
# Error Handler
# =============================================================================


@dataclass
class ErrorContext:
    """Context for error handling.

    Attributes:
        verbose: Show verbose error output
        debug: Show debug information (stack traces)
        exit_on_error: Exit the process on error
    """

    verbose: bool = False
    debug: bool = False
    exit_on_error: bool = True


def handle_cli_error(
    error: Exception,
    context: ErrorContext | None = None,
) -> int:
    """Handle a CLI error.

    Args:
        error: The exception to handle
        context: Error handling context

    Returns:
        Exit code
    """
    context = context or ErrorContext()

    # Determine error type and format message
    if isinstance(error, CLIError):
        typer.echo(typer.style(f"Error: {error.message}", fg="red"), err=True)

        if error.hint:
            typer.echo(typer.style(f"Hint: {error.hint}", fg="yellow"), err=True)

        if context.verbose and error.details:
            typer.echo("\nDetails:", err=True)
            for key, value in error.details.items():
                typer.echo(f"  {key}: {value}", err=True)

        exit_code = error.code.value

    elif isinstance(error, typer.Exit):
        # Re-raise typer exits
        raise error

    else:
        # Generic error
        typer.echo(typer.style(f"Error: {error}", fg="red"), err=True)
        exit_code = ErrorCode.GENERAL_ERROR.value

    # Show stack trace in debug mode
    if context.debug:
        typer.echo("\nStack trace:", err=True)
        typer.echo(traceback.format_exc(), err=True)

    # Exit if configured
    if context.exit_on_error:
        raise typer.Exit(exit_code)

    return exit_code


# =============================================================================
# Decorator
# =============================================================================


F = TypeVar("F", bound=Callable[..., Any])


def handle_errors(
    verbose: bool = False,
    debug: bool = False,
) -> Callable[[F], F]:
    """Decorator to handle errors in CLI commands.

    Args:
        verbose: Show verbose error output
        debug: Show debug information

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            context = ErrorContext(verbose=verbose, debug=debug)
            try:
                return func(*args, **kwargs)
            except typer.Exit:
                raise
            except Exception as e:
                handle_cli_error(e, context)
                return None

        return wrapper  # type: ignore

    return decorator


def error_boundary(func: F) -> F:
    """Simple error boundary decorator.

    Catches all exceptions and converts them to CLI errors.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except typer.Exit:
            raise
        except CLIError as e:
            typer.echo(typer.style(f"Error: {e.message}", fg="red"), err=True)
            if e.hint:
                typer.echo(typer.style(f"Hint: {e.hint}", fg="yellow"), err=True)
            raise typer.Exit(e.code.value)
        except Exception as e:
            logger.exception("Unexpected error")
            typer.echo(typer.style(f"Error: {e}", fg="red"), err=True)
            raise typer.Exit(ErrorCode.GENERAL_ERROR.value)

    return wrapper  # type: ignore


# =============================================================================
# Validation Helpers
# =============================================================================


def require_file(path: Path, description: str = "File") -> Path:
    """Require that a file exists.

    Args:
        path: Path to check
        description: Description for error message

    Returns:
        The validated path

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def require_dependency(
    package: str,
    import_name: str | None = None,
    install_extra: str | None = None,
) -> Any:
    """Require that a dependency is installed.

    Args:
        package: Package name for error message
        import_name: Module to import (defaults to package)
        install_extra: Optional extra to install (e.g., "truthound[reports]")

    Returns:
        The imported module

    Raises:
        DependencyError: If dependency is not installed
    """
    import_name = import_name or package

    try:
        import importlib

        return importlib.import_module(import_name)
    except ImportError:
        install_cmd = f"pip install {install_extra or package}"
        raise DependencyError(package, install_command=install_cmd)
