"""E2E Test Utilities and Helpers.

This module provides utilities and helper classes for E2E testing:
- Custom assertions for common test patterns
- CLI test helpers
- File operation helpers
- Profile validation helpers

Example:
    from tests.e2e.utils import CLITestHelper, assert_cli_success

    helper = CLITestHelper()
    result = helper.run("generate-suite", "profile.json", "-o", "rules.yaml")
    assert_cli_success(result)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from typer.testing import CliRunner, Result

if TYPE_CHECKING:
    from truthound.profiler import TableProfile, ValidationSuite


# =============================================================================
# CLI Test Helper
# =============================================================================


class CLITestHelper:
    """Helper class for CLI testing.

    Provides convenient methods for invoking CLI commands and
    validating their outputs.

    Example:
        helper = CLITestHelper()
        result = helper.run("check", "data.csv")
        assert result.exit_code == 0
    """

    def __init__(self, app: Any | None = None):
        """Initialize CLI test helper.

        Args:
            app: Typer app instance (defaults to truthound.cli.app)
        """
        if app is None:
            from truthound.cli import app
        self.app = app
        self.runner = CliRunner()

    def run(
        self,
        *args: str,
        catch_exceptions: bool = True,
        env: dict[str, str] | None = None,
    ) -> Result:
        """Run a CLI command.

        Args:
            *args: Command and arguments
            catch_exceptions: Whether to catch exceptions
            env: Environment variables

        Returns:
            CLI result
        """
        return self.runner.invoke(
            self.app,
            list(args),
            catch_exceptions=catch_exceptions,
            env=env,
        )

    def run_with_file(
        self,
        tmp_path: Path,
        *args: str,
        **kwargs: Any,
    ) -> Result:
        """Run CLI command with isolated filesystem.

        Args:
            tmp_path: Temporary directory
            *args: Command and arguments
            **kwargs: Additional arguments

        Returns:
            CLI result
        """
        with self.runner.isolated_filesystem(temp_dir=tmp_path):
            return self.run(*args, **kwargs)

    def run_generate_suite(
        self,
        profile_path: Path,
        output_path: Path | None = None,
        format: str = "yaml",
        strictness: str = "medium",
        **options: Any,
    ) -> Result:
        """Run generate-suite command.

        Args:
            profile_path: Path to profile file
            output_path: Output file path
            format: Output format
            strictness: Strictness level
            **options: Additional options

        Returns:
            CLI result
        """
        args = ["generate-suite", str(profile_path)]

        if output_path:
            args.extend(["-o", str(output_path)])

        args.extend(["-f", format])
        args.extend(["-s", strictness])

        for key, value in options.items():
            key = key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
            elif isinstance(value, list):
                for v in value:
                    args.extend([f"--{key}", str(v)])
            else:
                args.extend([f"--{key}", str(value)])

        return self.run(*args)

    def run_quick_suite(
        self,
        data_path: Path,
        output_path: Path | None = None,
        format: str = "yaml",
        strictness: str = "medium",
        **options: Any,
    ) -> Result:
        """Run quick-suite command.

        Args:
            data_path: Path to data file
            output_path: Output file path
            format: Output format
            strictness: Strictness level
            **options: Additional options

        Returns:
            CLI result
        """
        args = ["quick-suite", str(data_path)]

        if output_path:
            args.extend(["-o", str(output_path)])

        args.extend(["-f", format])
        args.extend(["-s", strictness])

        for key, value in options.items():
            key = key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
            else:
                args.extend([f"--{key}", str(value)])

        return self.run(*args)

    def run_auto_profile(
        self,
        data_path: Path,
        output_path: Path | None = None,
        **options: Any,
    ) -> Result:
        """Run auto-profile command.

        Args:
            data_path: Path to data file
            output_path: Output file path
            **options: Additional options

        Returns:
            CLI result
        """
        args = ["auto-profile", str(data_path)]

        if output_path:
            args.extend(["-o", str(output_path)])

        for key, value in options.items():
            key = key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
            else:
                args.extend([f"--{key}", str(value)])

        return self.run(*args)

    def run_check(
        self,
        data_path: Path,
        schema_path: Path | None = None,
        **options: Any,
    ) -> Result:
        """Run check command.

        Args:
            data_path: Path to data file
            schema_path: Path to schema file
            **options: Additional options

        Returns:
            CLI result
        """
        args = ["check", str(data_path)]

        if schema_path:
            args.extend(["--schema", str(schema_path)])

        for key, value in options.items():
            key = key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
            else:
                args.extend([f"--{key}", str(value)])

        return self.run(*args)

    def run_scan(
        self,
        data_path: Path,
        **options: Any,
    ) -> Result:
        """Run scan command for PII detection.

        Args:
            data_path: Path to data file
            **options: Additional options

        Returns:
            CLI result
        """
        args = ["scan", str(data_path)]

        for key, value in options.items():
            key = key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
            else:
                args.extend([f"--{key}", str(value)])

        return self.run(*args)


# =============================================================================
# File Test Helper
# =============================================================================


class FileTestHelper:
    """Helper class for file operations in tests.

    Provides methods for creating, reading, and validating files.
    """

    @staticmethod
    def read_yaml(path: Path) -> dict[str, Any]:
        """Read and parse a YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Parsed YAML content
        """
        content = path.read_text(encoding="utf-8")
        # Simple YAML parsing for test purposes
        return FileTestHelper._parse_simple_yaml(content)

    @staticmethod
    def read_json(path: Path) -> dict[str, Any]:
        """Read and parse a JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Parsed JSON content
        """
        content = path.read_text(encoding="utf-8")
        return json.loads(content)

    @staticmethod
    def read_text(path: Path) -> str:
        """Read text file content.

        Args:
            path: Path to text file

        Returns:
            File content
        """
        return path.read_text(encoding="utf-8")

    @staticmethod
    def write_yaml(path: Path, data: dict[str, Any]) -> None:
        """Write data to a YAML file.

        Args:
            path: Path to output file
            data: Data to write
        """
        import yaml
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    @staticmethod
    def write_json(path: Path, data: dict[str, Any]) -> None:
        """Write data to a JSON file.

        Args:
            path: Path to output file
            data: Data to write
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _parse_simple_yaml(content: str) -> dict[str, Any]:
        """Simple YAML parser for basic structures."""
        try:
            import yaml
            return yaml.safe_load(content) or {}
        except ImportError:
            # Fallback to simple parsing
            result: dict[str, Any] = {}
            for line in content.split("\n"):
                if ":" in line and not line.strip().startswith("#"):
                    key, _, value = line.partition(":")
                    key = key.strip()
                    value = value.strip()
                    if value:
                        result[key] = value
            return result


# =============================================================================
# Profile Test Helper
# =============================================================================


class ProfileTestHelper:
    """Helper class for profile-related testing.

    Provides methods for creating and validating profiles.
    """

    @staticmethod
    def create_profile(data_path: Path, **kwargs: Any) -> "TableProfile":
        """Create a profile from a data file.

        Args:
            data_path: Path to data file
            **kwargs: Arguments for profiler

        Returns:
            Generated profile
        """
        from truthound.profiler import profile_file
        return profile_file(str(data_path), **kwargs)

    @staticmethod
    def save_profile(profile: "TableProfile", output_path: Path) -> Path:
        """Save a profile to file.

        Args:
            profile: Profile to save
            output_path: Output path

        Returns:
            Path to saved file
        """
        from truthound.profiler import save_profile
        save_profile(profile, output_path)
        return output_path

    @staticmethod
    def load_profile(path: Path) -> "TableProfile":
        """Load a profile from file.

        Args:
            path: Path to profile file

        Returns:
            Loaded profile
        """
        from truthound.profiler import load_profile
        return load_profile(path)

    @staticmethod
    def validate_profile_structure(profile: "TableProfile") -> list[str]:
        """Validate profile structure and return issues.

        Args:
            profile: Profile to validate

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        if not profile.name:
            issues.append("Profile name is empty")

        if profile.row_count < 0:
            issues.append("Row count is negative")

        if profile.column_count < 0:
            issues.append("Column count is negative")

        if profile.column_count != len(profile.columns):
            issues.append(
                f"Column count mismatch: {profile.column_count} != {len(profile.columns)}"
            )

        for col in profile.columns:
            if not col.name:
                issues.append("Column with empty name found")
            if col.null_ratio < 0 or col.null_ratio > 1:
                issues.append(f"Invalid null_ratio for {col.name}: {col.null_ratio}")

        return issues


# =============================================================================
# Custom Assertions
# =============================================================================


class E2EAssertions:
    """Custom assertions for E2E tests.

    Provides rich assertion methods with detailed error messages.
    """

    @staticmethod
    def assert_cli_success(
        result: Result,
        expected_output: str | None = None,
    ) -> None:
        """Assert CLI command succeeded.

        Args:
            result: CLI result
            expected_output: Optional expected output string
        """
        if result.exit_code != 0:
            raise AssertionError(
                f"CLI command failed with exit code {result.exit_code}.\n"
                f"Output:\n{result.output}\n"
                f"Exception:\n{result.exception}"
            )

        if expected_output and expected_output not in result.output:
            raise AssertionError(
                f"Expected output not found.\n"
                f"Expected: {expected_output}\n"
                f"Actual:\n{result.output}"
            )

    @staticmethod
    def assert_cli_error(
        result: Result,
        expected_code: int | None = None,
        expected_message: str | None = None,
    ) -> None:
        """Assert CLI command failed.

        Args:
            result: CLI result
            expected_code: Expected exit code
            expected_message: Expected error message
        """
        if result.exit_code == 0:
            raise AssertionError(
                f"CLI command should have failed but succeeded.\n"
                f"Output:\n{result.output}"
            )

        if expected_code is not None and result.exit_code != expected_code:
            raise AssertionError(
                f"Wrong exit code. Expected: {expected_code}, Got: {result.exit_code}"
            )

        if expected_message and expected_message not in result.output:
            raise AssertionError(
                f"Expected error message not found.\n"
                f"Expected: {expected_message}\n"
                f"Actual:\n{result.output}"
            )

    @staticmethod
    def assert_file_exists(path: Path, message: str = "") -> None:
        """Assert file exists.

        Args:
            path: Path to file
            message: Optional message
        """
        if not path.exists():
            raise AssertionError(
                f"File not found: {path}. {message}"
            )

    @staticmethod
    def assert_file_contains(
        path: Path,
        expected: str | list[str],
    ) -> None:
        """Assert file contains expected content.

        Args:
            path: Path to file
            expected: Expected content (string or list of strings)
        """
        content = path.read_text(encoding="utf-8")

        if isinstance(expected, str):
            expected = [expected]

        for exp in expected:
            if exp not in content:
                raise AssertionError(
                    f"Expected content not found in {path}.\n"
                    f"Expected: {exp}\n"
                    f"Content:\n{content[:500]}..."
                )

    @staticmethod
    def assert_valid_yaml(path: Path) -> dict[str, Any]:
        """Assert file contains valid YAML.

        Args:
            path: Path to YAML file

        Returns:
            Parsed YAML content
        """
        try:
            return FileTestHelper.read_yaml(path)
        except Exception as e:
            raise AssertionError(
                f"Invalid YAML in {path}: {e}"
            )

    @staticmethod
    def assert_valid_json(path: Path) -> dict[str, Any]:
        """Assert file contains valid JSON.

        Args:
            path: Path to JSON file

        Returns:
            Parsed JSON content
        """
        try:
            return FileTestHelper.read_json(path)
        except Exception as e:
            raise AssertionError(
                f"Invalid JSON in {path}: {e}"
            )

    @staticmethod
    def assert_profile_valid(profile: "TableProfile") -> None:
        """Assert profile is valid.

        Args:
            profile: Profile to validate
        """
        issues = ProfileTestHelper.validate_profile_structure(profile)
        if issues:
            raise AssertionError(
                f"Invalid profile:\n" + "\n".join(f"  - {i}" for i in issues)
            )

    @staticmethod
    def assert_suite_has_rules(
        suite: "ValidationSuite",
        min_rules: int = 1,
        categories: list[str] | None = None,
    ) -> None:
        """Assert suite has expected rules.

        Args:
            suite: Validation suite
            min_rules: Minimum number of rules
            categories: Expected categories
        """
        if len(suite) < min_rules:
            raise AssertionError(
                f"Suite has {len(suite)} rules, expected at least {min_rules}"
            )

        if categories:
            suite_cats = set(r.category.value for r in suite.rules)
            for cat in categories:
                if cat not in suite_cats:
                    raise AssertionError(
                        f"Category '{cat}' not found in suite. "
                        f"Found: {suite_cats}"
                    )

    @staticmethod
    def assert_output_format(
        content: str,
        format: str,
    ) -> None:
        """Assert output is in expected format.

        Args:
            content: Output content
            format: Expected format (yaml, json, python)
        """
        if format == "yaml":
            if not ("rules:" in content or "name:" in content):
                raise AssertionError("Output doesn't look like YAML")
        elif format == "json":
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                raise AssertionError(f"Output is not valid JSON: {e}")
        elif format == "python":
            if "def " not in content and "class " not in content:
                raise AssertionError("Output doesn't look like Python code")


# =============================================================================
# Convenience Functions
# =============================================================================


def assert_cli_success(
    result: Result,
    expected_output: str | None = None,
) -> None:
    """Assert CLI command succeeded."""
    E2EAssertions.assert_cli_success(result, expected_output)


def assert_cli_error(
    result: Result,
    expected_code: int | None = None,
    expected_message: str | None = None,
) -> None:
    """Assert CLI command failed."""
    E2EAssertions.assert_cli_error(result, expected_code, expected_message)


def assert_file_contains(path: Path, expected: str | list[str]) -> None:
    """Assert file contains expected content."""
    E2EAssertions.assert_file_contains(path, expected)


def assert_valid_yaml(path: Path) -> dict[str, Any]:
    """Assert file contains valid YAML."""
    return E2EAssertions.assert_valid_yaml(path)


def assert_valid_json(path: Path) -> dict[str, Any]:
    """Assert file contains valid JSON."""
    return E2EAssertions.assert_valid_json(path)


# =============================================================================
# Test Context Manager
# =============================================================================


@dataclass
class E2ETestContext:
    """Context manager for E2E tests.

    Provides a clean context with all helpers initialized.

    Example:
        with E2ETestContext(tmp_path) as ctx:
            data_file = ctx.create_data("pii")
            result = ctx.cli.run_quick_suite(data_file)
            ctx.assert_success(result)
    """

    tmp_path: Path
    cli: CLITestHelper | None = None
    file_helper: FileTestHelper | None = None
    profile_helper: ProfileTestHelper | None = None

    def __enter__(self) -> "E2ETestContext":
        self.cli = CLITestHelper()
        self.file_helper = FileTestHelper()
        self.profile_helper = ProfileTestHelper()
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def create_data(
        self,
        scenario: str = "clean",
        format: str = "csv",
        row_count: int = 100,
    ) -> Path:
        """Create test data file."""
        from tests.e2e.fixtures import create_test_data
        return create_test_data(
            self.tmp_path,
            format=format,
            scenario=scenario,
            row_count=row_count,
        )

    def assert_success(self, result: Result) -> None:
        """Assert CLI success."""
        assert_cli_success(result)

    def assert_error(self, result: Result) -> None:
        """Assert CLI error."""
        assert_cli_error(result)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Helpers
    "CLITestHelper",
    "FileTestHelper",
    "ProfileTestHelper",
    # Assertions
    "E2EAssertions",
    "assert_cli_success",
    "assert_cli_error",
    "assert_file_contains",
    "assert_valid_yaml",
    "assert_valid_json",
    # Context
    "E2ETestContext",
]
