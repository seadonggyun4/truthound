"""Version constraint definitions.

This module provides version constraint handling for plugins,
supporting semantic versioning and various constraint formats.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Try to use packaging, fall back to simple comparison
try:
    from packaging.version import Version, InvalidVersion
    HAS_PACKAGING = True
except ImportError:
    HAS_PACKAGING = False


@dataclass(frozen=True)
class VersionConstraint:
    """Version constraint specification.

    Defines the acceptable version range for a dependency.
    Supports semantic versioning and various constraint formats.

    Attributes:
        min_version: Minimum required version (inclusive)
        max_version: Maximum allowed version (exclusive if ends with *)
        excluded_versions: Versions explicitly excluded
        prerelease_ok: Whether pre-release versions are acceptable
    """

    min_version: str | None = None
    max_version: str | None = None
    excluded_versions: tuple[str, ...] = ()
    prerelease_ok: bool = False

    def __post_init__(self) -> None:
        """Validate version strings."""
        if HAS_PACKAGING:
            if self.min_version:
                try:
                    Version(self.min_version.rstrip("*"))
                except InvalidVersion as e:
                    raise ValueError(f"Invalid min_version: {e}")
            if self.max_version:
                try:
                    Version(self.max_version.rstrip("*"))
                except InvalidVersion as e:
                    raise ValueError(f"Invalid max_version: {e}")

    def is_satisfied_by(self, version: str) -> bool:
        """Check if a version satisfies this constraint.

        Args:
            version: Version string to check

        Returns:
            True if version satisfies constraint
        """
        if version in self.excluded_versions:
            return False

        if HAS_PACKAGING:
            return self._check_with_packaging(version)
        else:
            return self._check_simple(version)

    def _check_with_packaging(self, version: str) -> bool:
        """Check version using packaging library."""
        try:
            ver = Version(version)

            # Reject pre-releases unless allowed
            if not self.prerelease_ok and (ver.is_prerelease or ver.is_devrelease):
                return False

            if self.min_version:
                min_ver = Version(self.min_version.rstrip("*"))
                if ver < min_ver:
                    return False

            if self.max_version:
                max_str = self.max_version
                exclusive = max_str.endswith("*")
                max_ver = Version(max_str.rstrip("*"))
                if exclusive:
                    if ver >= max_ver:
                        return False
                else:
                    if ver > max_ver:
                        return False

            return True
        except InvalidVersion:
            # If version can't be parsed, be permissive
            return True

    def _check_simple(self, version: str) -> bool:
        """Simple version comparison without packaging library."""
        def parse_version(v: str) -> tuple[int, ...]:
            parts = v.rstrip("*").split(".")
            return tuple(int(p) for p in parts if p.isdigit())

        try:
            ver = parse_version(version)

            if self.min_version:
                min_ver = parse_version(self.min_version)
                if ver < min_ver:
                    return False

            if self.max_version:
                max_ver = parse_version(self.max_version)
                exclusive = self.max_version.endswith("*")
                if exclusive:
                    if ver >= max_ver:
                        return False
                else:
                    if ver > max_ver:
                        return False

            return True
        except (ValueError, TypeError):
            return True

    def __str__(self) -> str:
        """String representation of constraint."""
        parts = []
        if self.min_version:
            parts.append(f">={self.min_version}")
        if self.max_version:
            parts.append(f"<{self.max_version}")
        if self.excluded_versions:
            parts.append(f"!={','.join(self.excluded_versions)}")
        return " && ".join(parts) if parts else "*"

    @classmethod
    def any_version(cls) -> "VersionConstraint":
        """Create constraint that accepts any version."""
        return cls(prerelease_ok=True)

    @classmethod
    def exact(cls, version: str) -> "VersionConstraint":
        """Create constraint for exact version match."""
        return cls(min_version=version, max_version=version)

    @classmethod
    def at_least(cls, version: str) -> "VersionConstraint":
        """Create constraint for minimum version."""
        return cls(min_version=version)

    @classmethod
    def compatible_with(cls, version: str) -> "VersionConstraint":
        """Create constraint compatible with version (same major).

        Following semantic versioning, ^1.2.3 means >=1.2.3 and <2.0.0
        """
        parts = version.split(".")
        if len(parts) >= 1:
            major = int(parts[0])
            max_version = f"{major + 1}.0.0*"
            return cls(min_version=version, max_version=max_version)
        return cls.at_least(version)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_version": self.min_version,
            "max_version": self.max_version,
            "excluded_versions": list(self.excluded_versions),
            "prerelease_ok": self.prerelease_ok,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VersionConstraint":
        """Create from dictionary."""
        return cls(
            min_version=data.get("min_version"),
            max_version=data.get("max_version"),
            excluded_versions=tuple(data.get("excluded_versions", [])),
            prerelease_ok=data.get("prerelease_ok", False),
        )


# Constraint parsing patterns
CONSTRAINT_PATTERNS = [
    (r"^\^(\d+\.\d+\.\d+)$", "caret"),      # ^1.2.3 - compatible
    (r"^~(\d+\.\d+\.\d+)$", "tilde"),       # ~1.2.3 - patch updates
    (r"^>=(\d+\.\d+\.\d+)$", "gte"),        # >=1.2.3
    (r"^>(\d+\.\d+\.\d+)$", "gt"),          # >1.2.3
    (r"^<=(\d+\.\d+\.\d+)$", "lte"),        # <=1.2.3
    (r"^<(\d+\.\d+\.\d+)$", "lt"),          # <1.2.3
    (r"^(\d+\.\d+\.\d+)$", "exact"),        # 1.2.3 - exact
    (r"^\*$", "any"),                        # * - any
]


def parse_constraint(spec: str) -> VersionConstraint:
    """Parse a version constraint specification.

    Supports formats:
        - * : any version
        - 1.2.3 : exact version
        - >=1.2.3 : at least
        - >1.2.3 : greater than
        - <=1.2.3 : at most
        - <1.2.3 : less than
        - ^1.2.3 : compatible (same major)
        - ~1.2.3 : patch updates (same minor)
        - >=1.0.0,<2.0.0 : range (comma-separated)

    Args:
        spec: Constraint specification string

    Returns:
        VersionConstraint

    Raises:
        ValueError: If specification cannot be parsed
    """
    spec = spec.strip()

    # Handle any
    if spec == "*" or not spec:
        return VersionConstraint.any_version()

    # Handle comma-separated constraints
    if "," in spec:
        min_ver = None
        max_ver = None
        excluded = []

        for part in spec.split(","):
            part = part.strip()
            if part.startswith(">="):
                min_ver = part[2:]
            elif part.startswith(">"):
                # Convert > to >= next patch
                min_ver = _increment_patch(part[1:])
            elif part.startswith("<="):
                max_ver = part[2:]
            elif part.startswith("<"):
                max_ver = part[1:] + "*"  # Exclusive
            elif part.startswith("!="):
                excluded.append(part[2:])

        return VersionConstraint(
            min_version=min_ver,
            max_version=max_ver,
            excluded_versions=tuple(excluded),
        )

    # Try each pattern
    for pattern, constraint_type in CONSTRAINT_PATTERNS:
        match = re.match(pattern, spec)
        if match:
            version = match.group(1) if match.lastindex else None

            if constraint_type == "any":
                return VersionConstraint.any_version()
            elif constraint_type == "exact":
                return VersionConstraint.exact(version)
            elif constraint_type == "gte":
                return VersionConstraint.at_least(version)
            elif constraint_type == "gt":
                return VersionConstraint(min_version=_increment_patch(version))
            elif constraint_type == "lte":
                return VersionConstraint(max_version=version)
            elif constraint_type == "lt":
                return VersionConstraint(max_version=version + "*")
            elif constraint_type == "caret":
                return VersionConstraint.compatible_with(version)
            elif constraint_type == "tilde":
                return _parse_tilde(version)

    raise ValueError(f"Cannot parse version constraint: {spec}")


def _increment_patch(version: str) -> str:
    """Increment patch version."""
    parts = version.split(".")
    if len(parts) >= 3:
        parts[2] = str(int(parts[2]) + 1)
    return ".".join(parts)


def _parse_tilde(version: str) -> VersionConstraint:
    """Parse tilde constraint (~1.2.3 means >=1.2.3 and <1.3.0)."""
    parts = version.split(".")
    if len(parts) >= 2:
        major, minor = parts[0], parts[1]
        max_version = f"{major}.{int(minor) + 1}.0*"
        return VersionConstraint(min_version=version, max_version=max_version)
    return VersionConstraint.at_least(version)
