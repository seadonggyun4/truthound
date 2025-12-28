"""Version resolver for compatibility checking.

This module provides version compatibility checking and resolution
for plugins and their dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from truthound.plugins.versioning.constraints import VersionConstraint

logger = logging.getLogger(__name__)


class CompatibilityLevel(Enum):
    """Level of compatibility between versions."""

    COMPATIBLE = auto()      # Fully compatible
    COMPATIBLE_WARNING = auto()  # Compatible with warnings
    INCOMPATIBLE = auto()    # Not compatible
    UNKNOWN = auto()         # Cannot determine


@dataclass(frozen=True)
class CompatibilityReport:
    """Report on version compatibility.

    Provides detailed information about compatibility between
    a plugin version and host version.

    Attributes:
        is_compatible: Whether versions are compatible
        level: Compatibility level
        host_version: Host application version
        plugin_version: Plugin version
        required_constraint: Version constraint from plugin
        errors: List of compatibility errors
        warnings: List of compatibility warnings
        migration_hints: Suggested migration steps
    """

    is_compatible: bool
    level: CompatibilityLevel
    host_version: str
    plugin_version: str
    required_constraint: VersionConstraint
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    migration_hints: tuple[str, ...] = ()

    @classmethod
    def compatible(
        cls,
        host_version: str,
        plugin_version: str,
        constraint: VersionConstraint,
        warnings: tuple[str, ...] = (),
    ) -> "CompatibilityReport":
        """Create a compatible report."""
        level = CompatibilityLevel.COMPATIBLE_WARNING if warnings else CompatibilityLevel.COMPATIBLE
        return cls(
            is_compatible=True,
            level=level,
            host_version=host_version,
            plugin_version=plugin_version,
            required_constraint=constraint,
            warnings=warnings,
        )

    @classmethod
    def incompatible(
        cls,
        host_version: str,
        plugin_version: str,
        constraint: VersionConstraint,
        errors: tuple[str, ...],
        migration_hints: tuple[str, ...] = (),
    ) -> "CompatibilityReport":
        """Create an incompatible report."""
        return cls(
            is_compatible=False,
            level=CompatibilityLevel.INCOMPATIBLE,
            host_version=host_version,
            plugin_version=plugin_version,
            required_constraint=constraint,
            errors=errors,
            migration_hints=migration_hints,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_compatible": self.is_compatible,
            "level": self.level.name,
            "host_version": self.host_version,
            "plugin_version": self.plugin_version,
            "constraint": str(self.required_constraint),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "migration_hints": list(self.migration_hints),
        }


class VersionResolver:
    """Resolves version compatibility for plugins.

    Checks if plugin versions are compatible with the host version
    and provides detailed compatibility reports.

    Example:
        >>> resolver = VersionResolver()
        >>> constraint = VersionConstraint(min_version="0.1.0")
        >>> report = resolver.check_compatibility("0.5.0", constraint, "0.2.0")
        >>> print(report.is_compatible)
    """

    def __init__(self) -> None:
        """Initialize version resolver."""
        self._breaking_changes: dict[str, list[str]] = {}

    def register_breaking_change(
        self,
        version: str,
        description: str,
    ) -> None:
        """Register a breaking change at a version.

        Args:
            version: Version where breaking change occurred
            description: Description of the breaking change
        """
        if version not in self._breaking_changes:
            self._breaking_changes[version] = []
        self._breaking_changes[version].append(description)

    def check_compatibility(
        self,
        plugin_version: str,
        constraint: VersionConstraint,
        host_version: str,
    ) -> CompatibilityReport:
        """Check if plugin is compatible with host version.

        Args:
            plugin_version: Version of the plugin
            constraint: Version constraint from plugin manifest
            host_version: Current host application version

        Returns:
            CompatibilityReport with detailed information
        """
        errors: list[str] = []
        warnings: list[str] = []
        hints: list[str] = []

        # Check if host version satisfies plugin's constraint
        if not constraint.is_satisfied_by(host_version):
            if constraint.min_version:
                errors.append(
                    f"Plugin requires host version >= {constraint.min_version}, "
                    f"but current version is {host_version}"
                )
                hints.append(
                    f"Upgrade host to at least version {constraint.min_version}"
                )

            if constraint.max_version:
                max_ver = constraint.max_version.rstrip("*")
                errors.append(
                    f"Plugin requires host version < {max_ver}, "
                    f"but current version is {host_version}"
                )
                hints.append(
                    f"Use a plugin version compatible with host {host_version}"
                )

            if host_version in constraint.excluded_versions:
                errors.append(
                    f"Plugin explicitly excludes host version {host_version}"
                )

        # Check for breaking changes
        breaking = self._check_breaking_changes(
            plugin_version,
            host_version,
        )
        if breaking:
            for change in breaking:
                warnings.append(f"Breaking change: {change}")
            hints.extend([
                "Review the changelog for breaking changes",
                "Update plugin code for compatibility",
            ])

        if errors:
            return CompatibilityReport.incompatible(
                host_version=host_version,
                plugin_version=plugin_version,
                constraint=constraint,
                errors=tuple(errors),
                migration_hints=tuple(hints),
            )

        return CompatibilityReport.compatible(
            host_version=host_version,
            plugin_version=plugin_version,
            constraint=constraint,
            warnings=tuple(warnings),
        )

    def _check_breaking_changes(
        self,
        plugin_version: str,
        host_version: str,
    ) -> list[str]:
        """Check for breaking changes between versions."""
        changes = []

        # Get breaking changes in host versions since plugin was written
        for version, descriptions in self._breaking_changes.items():
            # Check if this version is between plugin and host
            if self._is_version_between(version, plugin_version, host_version):
                changes.extend(descriptions)

        return changes

    def _is_version_between(
        self,
        version: str,
        low: str,
        high: str,
    ) -> bool:
        """Check if version is between low and high."""
        try:
            from packaging.version import Version
            v = Version(version)
            l = Version(low)
            h = Version(high)
            return l <= v <= h
        except ImportError:
            # Simple comparison
            return low <= version <= high

    def find_compatible_versions(
        self,
        constraint: VersionConstraint,
        available_versions: list[str],
    ) -> list[str]:
        """Find all versions that satisfy a constraint.

        Args:
            constraint: Version constraint to satisfy
            available_versions: List of available versions

        Returns:
            List of compatible versions (sorted newest first)
        """
        compatible = [
            v for v in available_versions
            if constraint.is_satisfied_by(v)
        ]

        # Sort by version (newest first)
        try:
            from packaging.version import Version
            compatible.sort(key=Version, reverse=True)
        except ImportError:
            compatible.sort(reverse=True)

        return compatible

    def find_best_version(
        self,
        constraint: VersionConstraint,
        available_versions: list[str],
        prefer_stable: bool = True,
    ) -> str | None:
        """Find the best version that satisfies a constraint.

        Args:
            constraint: Version constraint to satisfy
            available_versions: List of available versions
            prefer_stable: Prefer stable over pre-release

        Returns:
            Best matching version or None
        """
        compatible = self.find_compatible_versions(constraint, available_versions)

        if not compatible:
            return None

        if prefer_stable:
            # Filter out pre-releases
            try:
                from packaging.version import Version
                stable = [v for v in compatible if not Version(v).is_prerelease]
                if stable:
                    return stable[0]
            except ImportError:
                pass

        return compatible[0]

    def check_upgrade_path(
        self,
        from_version: str,
        to_version: str,
    ) -> list[str]:
        """Get list of breaking changes in upgrade path.

        Args:
            from_version: Starting version
            to_version: Target version

        Returns:
            List of breaking change descriptions
        """
        changes = []

        for version, descriptions in self._breaking_changes.items():
            if self._is_version_between(version, from_version, to_version):
                changes.extend(descriptions)

        return changes
