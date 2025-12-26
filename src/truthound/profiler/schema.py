"""Profile schema versioning for forward/backward compatibility.

This module provides schema versioning and migration support for
profile serialization, ensuring that profiles created with older
versions can be read by newer code and vice versa.

Key features:
- Semantic versioning for profile schemas
- Automatic schema migration
- Forward compatibility support
- Validation of profile data
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeVar

from truthound.profiler.base import (
    ColumnProfile,
    DataType,
    DistributionStats,
    PatternMatch,
    TableProfile,
    ValueFrequency,
)


# =============================================================================
# Schema Version
# =============================================================================


@dataclass(frozen=True)
class SchemaVersion:
    """Semantic version for profile schemas.

    Format: major.minor.patch
    - major: Breaking changes (incompatible)
    - minor: New features (backward compatible)
    - patch: Bug fixes (fully compatible)
    """

    major: int
    minor: int
    patch: int = 0

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: "SchemaVersion") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __le__(self, other: "SchemaVersion") -> bool:
        return (self.major, self.minor, self.patch) <= (other.major, other.minor, other.patch)

    def __gt__(self, other: "SchemaVersion") -> bool:
        return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)

    def __ge__(self, other: "SchemaVersion") -> bool:
        return (self.major, self.minor, self.patch) >= (other.major, other.minor, other.patch)

    @classmethod
    def from_string(cls, version: str) -> "SchemaVersion":
        """Parse version from string."""
        parts = version.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid version string: {version}")

        major = int(parts[0])
        minor = int(parts[1])
        patch = int(parts[2]) if len(parts) > 2 else 0

        return cls(major=major, minor=minor, patch=patch)

    def is_compatible_with(self, other: "SchemaVersion") -> bool:
        """Check if this version can read data from other version.

        Same major version = compatible (with possible migrations)
        """
        return self.major == other.major

    def is_breaking_change(self, other: "SchemaVersion") -> bool:
        """Check if there's a breaking change between versions."""
        return self.major != other.major


# Current schema version
CURRENT_SCHEMA_VERSION = SchemaVersion(1, 0, 0)


# =============================================================================
# Schema Migration
# =============================================================================


class MigrationStep(ABC):
    """Abstract base for schema migration steps."""

    from_version: SchemaVersion
    to_version: SchemaVersion

    @abstractmethod
    def migrate(self, data: dict[str, Any]) -> dict[str, Any]:
        """Migrate data from source to target version.

        Args:
            data: Profile data in source version format

        Returns:
            Profile data in target version format
        """
        pass

    @abstractmethod
    def rollback(self, data: dict[str, Any]) -> dict[str, Any]:
        """Rollback migration (for downgrade).

        Args:
            data: Profile data in target version format

        Returns:
            Profile data in source version format
        """
        pass


class SchemaMigrator:
    """Handles schema migrations between versions.

    The migrator maintains a chain of migration steps and can
    automatically apply the necessary migrations to upgrade or
    downgrade profile data.

    Example:
        migrator = SchemaMigrator()
        migrator.register(MyMigrationStep())

        # Migrate data
        old_data = {"schema_version": "0.9.0", ...}
        new_data = migrator.migrate(old_data, to_version=CURRENT_SCHEMA_VERSION)
    """

    def __init__(self) -> None:
        self._migrations: dict[tuple[SchemaVersion, SchemaVersion], MigrationStep] = {}

    def register(self, step: MigrationStep) -> None:
        """Register a migration step."""
        key = (step.from_version, step.to_version)
        self._migrations[key] = step

    def migrate(
        self,
        data: dict[str, Any],
        *,
        to_version: SchemaVersion | None = None,
    ) -> dict[str, Any]:
        """Migrate profile data to target version.

        Args:
            data: Profile data (must include schema_version)
            to_version: Target version (defaults to current)

        Returns:
            Migrated profile data
        """
        to_version = to_version or CURRENT_SCHEMA_VERSION

        # Parse source version
        source_version_str = data.get("schema_version", "1.0.0")
        source_version = SchemaVersion.from_string(source_version_str)

        if source_version == to_version:
            return data  # No migration needed

        if not source_version.is_compatible_with(to_version):
            raise ValueError(
                f"Cannot migrate between incompatible versions: "
                f"{source_version} -> {to_version}"
            )

        # Find migration path
        result = dict(data)

        # Upgrade path
        if source_version < to_version:
            current = source_version
            while current < to_version:
                step = self._find_upgrade_step(current)
                if step is None:
                    break
                result = step.migrate(result)
                current = step.to_version
        # Downgrade path
        else:
            current = source_version
            while current > to_version:
                step = self._find_downgrade_step(current)
                if step is None:
                    break
                result = step.rollback(result)
                current = step.from_version

        result["schema_version"] = str(to_version)
        return result

    def _find_upgrade_step(self, from_version: SchemaVersion) -> MigrationStep | None:
        """Find the next upgrade step from a version."""
        for (src, dst), step in self._migrations.items():
            if src == from_version:
                return step
        return None

    def _find_downgrade_step(self, to_version: SchemaVersion) -> MigrationStep | None:
        """Find the step that can be rolled back to reach a version."""
        for (src, dst), step in self._migrations.items():
            if dst == to_version:
                return step
        return None


# Global migrator instance
schema_migrator = SchemaMigrator()


# =============================================================================
# Profile Serialization with Versioning
# =============================================================================


@dataclass
class ProfileSerializer:
    """Serializes and deserializes profiles with schema versioning.

    This class handles the conversion between TableProfile objects
    and serializable dictionaries, including schema version handling.

    Example:
        serializer = ProfileSerializer()

        # Serialize
        data = serializer.serialize(profile)

        # Deserialize with automatic migration
        profile = serializer.deserialize(old_data)
    """

    version: SchemaVersion = field(default_factory=lambda: CURRENT_SCHEMA_VERSION)
    migrator: SchemaMigrator = field(default_factory=lambda: schema_migrator)

    def serialize(self, profile: TableProfile) -> dict[str, Any]:
        """Serialize a TableProfile to a dictionary.

        Args:
            profile: Profile to serialize

        Returns:
            Dictionary with schema version and profile data
        """
        data = profile.to_dict()
        data["schema_version"] = str(self.version)
        data["serialized_at"] = datetime.now().isoformat()
        return data

    def deserialize(self, data: dict[str, Any]) -> TableProfile:
        """Deserialize a dictionary to a TableProfile.

        Automatically migrates data if schema version differs.

        Args:
            data: Dictionary with profile data

        Returns:
            Reconstructed TableProfile
        """
        # Check and migrate schema if needed
        source_version_str = data.get("schema_version", "1.0.0")
        source_version = SchemaVersion.from_string(source_version_str)

        if source_version != self.version:
            data = self.migrator.migrate(data, to_version=self.version)

        # Reconstruct column profiles
        columns = []
        for col_data in data.get("columns", []):
            columns.append(self._deserialize_column(col_data))

        # Reconstruct correlations
        correlations = tuple(
            (c["column1"], c["column2"], c["correlation"])
            for c in data.get("correlations", [])
        )

        return TableProfile(
            name=data.get("name", ""),
            row_count=data.get("row_count", 0),
            column_count=data.get("column_count", 0),
            estimated_memory_bytes=data.get("estimated_memory_bytes", 0),
            columns=tuple(columns),
            duplicate_row_count=data.get("duplicate_row_count", 0),
            duplicate_row_ratio=data.get("duplicate_row_ratio", 0.0),
            correlations=correlations,
            source=data.get("source", ""),
            profiled_at=self._parse_datetime(data.get("profiled_at")),
            profile_duration_ms=data.get("profile_duration_ms", 0.0),
        )

    def _deserialize_column(self, data: dict[str, Any]) -> ColumnProfile:
        """Deserialize a column profile."""
        # Reconstruct nested objects
        distribution = None
        if "distribution" in data and data["distribution"]:
            distribution = DistributionStats(**data["distribution"])

        top_values = ()
        if "top_values" in data:
            top_values = tuple(ValueFrequency(**v) for v in data["top_values"])

        bottom_values = ()
        if "bottom_values" in data:
            bottom_values = tuple(ValueFrequency(**v) for v in data["bottom_values"])

        detected_patterns = ()
        if "detected_patterns" in data:
            detected_patterns = tuple(
                PatternMatch(
                    pattern=p["pattern"],
                    regex=p["regex"],
                    match_ratio=p["match_ratio"],
                    sample_matches=tuple(p.get("sample_matches", [])),
                )
                for p in data["detected_patterns"]
            )

        suggested_validators = ()
        if "suggested_validators" in data:
            suggested_validators = tuple(data["suggested_validators"])

        # Handle inferred_type
        inferred_type = DataType.UNKNOWN
        if "inferred_type" in data:
            try:
                inferred_type = DataType(data["inferred_type"])
            except ValueError:
                inferred_type = DataType.UNKNOWN

        return ColumnProfile(
            name=data.get("name", ""),
            physical_type=data.get("physical_type", ""),
            inferred_type=inferred_type,
            row_count=data.get("row_count", 0),
            null_count=data.get("null_count", 0),
            null_ratio=data.get("null_ratio", 0.0),
            empty_string_count=data.get("empty_string_count", 0),
            distinct_count=data.get("distinct_count", 0),
            unique_ratio=data.get("unique_ratio", 0.0),
            is_unique=data.get("is_unique", False),
            is_constant=data.get("is_constant", False),
            distribution=distribution,
            top_values=top_values,
            bottom_values=bottom_values,
            min_length=data.get("min_length"),
            max_length=data.get("max_length"),
            avg_length=data.get("avg_length"),
            detected_patterns=detected_patterns,
            min_date=self._parse_datetime(data.get("min_date")),
            max_date=self._parse_datetime(data.get("max_date")),
            date_gaps=data.get("date_gaps", 0),
            suggested_validators=suggested_validators,
            profiled_at=self._parse_datetime(data.get("profiled_at")) or datetime.now(),
            profile_duration_ms=data.get("profile_duration_ms", 0.0),
        )

    def _parse_datetime(self, value: str | None) -> datetime | None:
        """Parse datetime from ISO format string."""
        if value is None:
            return None
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return None


# =============================================================================
# Schema Validation
# =============================================================================


class SchemaValidationStatus(Enum):
    """Result of schema validation."""

    VALID = "valid"
    RECOVERABLE = "recoverable"  # Has issues but can be fixed
    INVALID = "invalid"  # Cannot be processed


# Alias for backward compatibility
ValidationResult = SchemaValidationStatus


@dataclass
class SchemaValidationResult:
    """Result of validating profile data against schema."""

    result: ValidationResult
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fixed_data: dict[str, Any] | None = None


class SchemaValidator:
    """Validates profile data against the current schema.

    This validator checks that profile data conforms to the expected
    schema and can optionally fix minor issues.

    Example:
        validator = SchemaValidator()
        result = validator.validate(data)

        if result.result == ValidationResult.INVALID:
            print("Errors:", result.errors)
        elif result.result == ValidationResult.RECOVERABLE:
            print("Fixed issues:", result.warnings)
            data = result.fixed_data
    """

    REQUIRED_FIELDS = {"name", "row_count", "column_count", "columns"}
    COLUMN_REQUIRED_FIELDS = {"name", "physical_type"}

    def validate(
        self,
        data: dict[str, Any],
        *,
        fix_issues: bool = True,
    ) -> SchemaValidationResult:
        """Validate profile data.

        Args:
            data: Profile data to validate
            fix_issues: Whether to attempt fixing minor issues

        Returns:
            Validation result with any errors/warnings
        """
        errors: list[str] = []
        warnings: list[str] = []
        fixed_data = dict(data) if fix_issues else None

        # Check required fields
        for field_name in self.REQUIRED_FIELDS:
            if field_name not in data:
                errors.append(f"Missing required field: {field_name}")

        if errors:
            return SchemaValidationResult(
                result=ValidationResult.INVALID,
                errors=errors,
            )

        # Validate columns
        for i, col in enumerate(data.get("columns", [])):
            col_errors, col_warnings, col_fixed = self._validate_column(
                col, i, fix_issues
            )
            errors.extend(col_errors)
            warnings.extend(col_warnings)
            if fixed_data and col_fixed:
                fixed_data["columns"][i] = col_fixed

        # Check schema version
        if "schema_version" not in data:
            warnings.append("Missing schema_version, defaulting to 1.0.0")
            if fixed_data:
                fixed_data["schema_version"] = "1.0.0"

        # Determine result
        if errors:
            return SchemaValidationResult(
                result=ValidationResult.INVALID,
                errors=errors,
            )
        elif warnings:
            return SchemaValidationResult(
                result=ValidationResult.RECOVERABLE,
                warnings=warnings,
                fixed_data=fixed_data,
            )
        else:
            return SchemaValidationResult(
                result=ValidationResult.VALID,
                fixed_data=data if fix_issues else None,
            )

    def _validate_column(
        self,
        col: dict[str, Any],
        index: int,
        fix_issues: bool,
    ) -> tuple[list[str], list[str], dict[str, Any] | None]:
        """Validate a column profile."""
        errors: list[str] = []
        warnings: list[str] = []
        fixed = dict(col) if fix_issues else None

        for field_name in self.COLUMN_REQUIRED_FIELDS:
            if field_name not in col:
                errors.append(f"Column {index}: missing required field {field_name}")

        # Validate inferred_type
        if "inferred_type" in col:
            try:
                DataType(col["inferred_type"])
            except ValueError:
                warnings.append(
                    f"Column {index}: unknown inferred_type '{col['inferred_type']}', "
                    f"defaulting to 'unknown'"
                )
                if fixed:
                    fixed["inferred_type"] = "unknown"

        # Validate numeric ranges
        if "null_ratio" in col:
            ratio = col["null_ratio"]
            if not 0.0 <= ratio <= 1.0:
                warnings.append(f"Column {index}: null_ratio out of range: {ratio}")
                if fixed:
                    fixed["null_ratio"] = max(0.0, min(1.0, ratio))

        return errors, warnings, fixed


# =============================================================================
# Convenience Functions
# =============================================================================


def save_profile(
    profile: TableProfile,
    path: str | Path,
    *,
    indent: int = 2,
) -> None:
    """Save a profile to a JSON file with schema versioning.

    Args:
        profile: Profile to save
        path: Output file path
        indent: JSON indentation level
    """
    serializer = ProfileSerializer()
    data = serializer.serialize(profile)

    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=_json_serializer)


def load_profile(path: str | Path) -> TableProfile:
    """Load a profile from a JSON file with automatic migration.

    Args:
        path: Path to the profile JSON file

    Returns:
        Reconstructed TableProfile
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    serializer = ProfileSerializer()
    return serializer.deserialize(data)


def validate_profile(
    data: dict[str, Any],
    *,
    fix_issues: bool = True,
) -> SchemaValidationResult:
    """Validate profile data against current schema.

    Args:
        data: Profile data dictionary
        fix_issues: Whether to attempt fixing minor issues

    Returns:
        Validation result
    """
    validator = SchemaValidator()
    return validator.validate(data, fix_issues=fix_issues)


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
