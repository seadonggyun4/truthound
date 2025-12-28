"""Common CLI infrastructure.

This package provides shared components for CLI commands:
    - protocol: Command protocol and base classes
    - options: Reusable CLI options and arguments
    - output: Output formatting utilities
    - errors: CLI error handling
    - decorators: Command decorators for common patterns
"""

from truthound.cli_modules.common.protocol import (
    CommandProtocol,
    CommandResult,
    BaseCommand,
    command,
    CommandRegistry,
    get_command_registry,
)
from truthound.cli_modules.common.options import (
    FileArg,
    OutputOpt,
    FormatOpt,
    StrictOpt,
    VerboseOpt,
    ColumnsOpt,
    file_exists_callback,
    parse_list_option,
)
from truthound.cli_modules.common.output import (
    OutputFormatter,
    ConsoleOutput,
    JsonOutput,
    TableOutput,
)
from truthound.cli_modules.common.errors import (
    CLIError,
    FileNotFoundError,
    ValidationError,
    ConfigurationError,
    handle_cli_error,
)

__all__ = [
    # Protocol
    "CommandProtocol",
    "CommandResult",
    "BaseCommand",
    "command",
    "CommandRegistry",
    "get_command_registry",
    # Options
    "FileArg",
    "OutputOpt",
    "FormatOpt",
    "StrictOpt",
    "VerboseOpt",
    "ColumnsOpt",
    "file_exists_callback",
    "parse_list_option",
    # Output
    "OutputFormatter",
    "ConsoleOutput",
    "JsonOutput",
    "TableOutput",
    # Errors
    "CLIError",
    "FileNotFoundError",
    "ValidationError",
    "ConfigurationError",
    "handle_cli_error",
]
