"""Version compatibility checking for validators.

This module provides semantic versioning support:
- Semantic version parsing and comparison
- Version constraint matching
- Compatibility matrix management
- Dependency resolution

Example:
    from truthound.validators.sdk.enterprise.versioning import (
        SemanticVersion,
        VersionChecker,
        VersionConstraint,
    )

    # Parse version
    version = SemanticVersion.parse("2.1.0")

    # Check constraint
    constraint = VersionConstraint.parse(">=1.0.0,<3.0.0")
    is_compatible = constraint.matches(version)

    # Check validator compatibility
    checker = VersionChecker()
    checker.check_compatibility(validator, truthound_version="0.2.0")
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import total_ordering
from typing import Any

logger = logging.getLogger(__name__)


class VersionConflictError(Exception):
    """Raised when version conflict is detected."""

    def __init__(
        self,
        message: str,
        validator_name: str = "",
        required: str = "",
        actual: str = "",
    ):
        self.validator_name = validator_name
        self.required = required
        self.actual = actual
        super().__init__(message)


class VersionCompatibility(Enum):
    """Compatibility levels."""

    COMPATIBLE = auto()      # Fully compatible
    COMPATIBLE_MINOR = auto() # Minor version difference, likely compatible
    COMPATIBLE_PATCH = auto() # Patch version difference, should be compatible
    INCOMPATIBLE = auto()    # Not compatible
    UNKNOWN = auto()         # Cannot determine


@total_ordering
@dataclass(frozen=True)
class SemanticVersion:
    """Semantic version representation.

    Follows semver 2.0.0 specification:
    - MAJOR.MINOR.PATCH-PRERELEASE+BUILD
    - MAJOR: Breaking changes
    - MINOR: Backward-compatible features
    - PATCH: Backward-compatible fixes
    """

    major: int
    minor: int
    patch: int
    prerelease: str = ""
    build: str = ""

    def __str__(self) -> str:
        """Convert to string representation."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __eq__(self, other: object) -> bool:
        """Check equality (build metadata is ignored per semver)."""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.prerelease == other.prerelease
        )

    def __lt__(self, other: object) -> bool:
        """Compare versions (build metadata is ignored per semver)."""
        if not isinstance(other, SemanticVersion):
            return NotImplemented

        # Compare major.minor.patch
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

        # Prerelease versions have lower precedence
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False
        if self.prerelease and other.prerelease:
            return self._compare_prerelease(self.prerelease, other.prerelease) < 0

        return False

    @staticmethod
    def _compare_prerelease(a: str, b: str) -> int:
        """Compare prerelease strings per semver spec."""
        a_parts = a.split(".")
        b_parts = b.split(".")

        for a_part, b_part in zip(a_parts, b_parts):
            # Numeric identifiers have lower precedence than alphanumeric
            a_is_num = a_part.isdigit()
            b_is_num = b_part.isdigit()

            if a_is_num and b_is_num:
                if int(a_part) != int(b_part):
                    return int(a_part) - int(b_part)
            elif a_is_num:
                return -1
            elif b_is_num:
                return 1
            else:
                if a_part != b_part:
                    return -1 if a_part < b_part else 1

        return len(a_parts) - len(b_parts)

    def __hash__(self) -> int:
        """Hash (build metadata is excluded per semver)."""
        return hash((self.major, self.minor, self.patch, self.prerelease))

    @classmethod
    def parse(cls, version_str: str) -> "SemanticVersion":
        """Parse version string.

        Args:
            version_str: Version string (e.g., "1.2.3-alpha+build")

        Returns:
            SemanticVersion instance

        Raises:
            ValueError: If version string is invalid
        """
        # Remove leading 'v' if present
        version_str = version_str.lstrip("v")

        # Pattern for semver 2.0.0
        pattern = r"""
            ^
            (?P<major>0|[1-9]\d*)
            \.
            (?P<minor>0|[1-9]\d*)
            \.
            (?P<patch>0|[1-9]\d*)
            (?:-(?P<prerelease>
                (?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)
                (?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*
            ))?
            (?:\+(?P<build>
                [0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*
            ))?
            $
        """

        match = re.match(pattern, version_str, re.VERBOSE)
        if not match:
            # Try simple format (major.minor.patch)
            simple_match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version_str)
            if simple_match:
                return cls(
                    major=int(simple_match.group(1)),
                    minor=int(simple_match.group(2)),
                    patch=int(simple_match.group(3)),
                )
            # Try two-part format (major.minor) - common for Python versions
            two_part_match = re.match(r"^(\d+)\.(\d+)$", version_str)
            if two_part_match:
                return cls(
                    major=int(two_part_match.group(1)),
                    minor=int(two_part_match.group(2)),
                    patch=0,
                )
            raise ValueError(f"Invalid version string: {version_str}")

        return cls(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
            prerelease=match.group("prerelease") or "",
            build=match.group("build") or "",
        )

    def is_prerelease(self) -> bool:
        """Check if this is a prerelease version."""
        return bool(self.prerelease)

    def is_compatible_with(self, other: "SemanticVersion") -> VersionCompatibility:
        """Check compatibility with another version.

        Args:
            other: Version to check against

        Returns:
            VersionCompatibility level
        """
        # Same version
        if self == other:
            return VersionCompatibility.COMPATIBLE

        # Different major version
        if self.major != other.major:
            # 0.x versions are all potentially breaking
            if self.major == 0 or other.major == 0:
                return VersionCompatibility.INCOMPATIBLE
            return VersionCompatibility.INCOMPATIBLE

        # Same major, different minor
        if self.minor != other.minor:
            return VersionCompatibility.COMPATIBLE_MINOR

        # Same major.minor, different patch
        if self.patch != other.patch:
            return VersionCompatibility.COMPATIBLE_PATCH

        return VersionCompatibility.COMPATIBLE

    def bump_major(self) -> "SemanticVersion":
        """Create version with bumped major."""
        return SemanticVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> "SemanticVersion":
        """Create version with bumped minor."""
        return SemanticVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> "SemanticVersion":
        """Create version with bumped patch."""
        return SemanticVersion(self.major, self.minor, self.patch + 1)


class ConstraintOperator(Enum):
    """Version constraint operators."""

    EQ = "="      # Exactly equal
    NE = "!="     # Not equal
    GT = ">"      # Greater than
    GE = ">="     # Greater than or equal
    LT = "<"      # Less than
    LE = "<="     # Less than or equal
    TILDE = "~"   # Compatible with (same major.minor)
    CARET = "^"   # Compatible with (same major)


@dataclass(frozen=True)
class VersionConstraint:
    """Single version constraint.

    Examples:
        - ">=1.0.0" - At least version 1.0.0
        - "<2.0.0" - Less than version 2.0.0
        - "~1.2.0" - Compatible with 1.2.x (>=1.2.0, <1.3.0)
        - "^1.2.0" - Compatible with 1.x.x (>=1.2.0, <2.0.0)
    """

    operator: ConstraintOperator
    version: SemanticVersion

    def __str__(self) -> str:
        """Convert to string."""
        return f"{self.operator.value}{self.version}"

    def matches(self, version: SemanticVersion) -> bool:
        """Check if version matches this constraint.

        Args:
            version: Version to check

        Returns:
            True if version matches
        """
        if self.operator == ConstraintOperator.EQ:
            return version == self.version
        elif self.operator == ConstraintOperator.NE:
            return version != self.version
        elif self.operator == ConstraintOperator.GT:
            return version > self.version
        elif self.operator == ConstraintOperator.GE:
            return version >= self.version
        elif self.operator == ConstraintOperator.LT:
            return version < self.version
        elif self.operator == ConstraintOperator.LE:
            return version <= self.version
        elif self.operator == ConstraintOperator.TILDE:
            # ~1.2.3 means >=1.2.3, <1.3.0
            return (
                version >= self.version
                and version.major == self.version.major
                and version.minor == self.version.minor
            )
        elif self.operator == ConstraintOperator.CARET:
            # ^1.2.3 means >=1.2.3, <2.0.0
            # ^0.2.3 means >=0.2.3, <0.3.0 (special case for 0.x)
            if self.version.major == 0:
                return (
                    version >= self.version
                    and version.major == 0
                    and version.minor == self.version.minor
                )
            return (
                version >= self.version
                and version.major == self.version.major
            )
        else:
            return False

    @classmethod
    def parse(cls, constraint_str: str) -> "VersionConstraint":
        """Parse constraint string.

        Args:
            constraint_str: Constraint string (e.g., ">=1.0.0")

        Returns:
            VersionConstraint instance
        """
        constraint_str = constraint_str.strip()

        # Try each operator (longest first)
        for op in [ConstraintOperator.GE, ConstraintOperator.LE,
                   ConstraintOperator.NE, ConstraintOperator.GT,
                   ConstraintOperator.LT, ConstraintOperator.EQ,
                   ConstraintOperator.TILDE, ConstraintOperator.CARET]:
            if constraint_str.startswith(op.value):
                version_str = constraint_str[len(op.value):].strip()
                return cls(
                    operator=op,
                    version=SemanticVersion.parse(version_str),
                )

        # Default to exact match
        return cls(
            operator=ConstraintOperator.EQ,
            version=SemanticVersion.parse(constraint_str),
        )


@dataclass
class VersionSpec:
    """Version specification with multiple constraints.

    Supports AND (comma) and OR (||) combinations:
    - ">=1.0.0,<2.0.0" - Both must match (AND)
    - ">=1.0.0 || >=2.0.0,<3.0.0" - Either must match (OR)
    """

    constraints: list[list[VersionConstraint]] = field(default_factory=list)

    def __str__(self) -> str:
        """Convert to string."""
        or_parts = []
        for and_group in self.constraints:
            or_parts.append(",".join(str(c) for c in and_group))
        return " || ".join(or_parts)

    def matches(self, version: SemanticVersion) -> bool:
        """Check if version matches specification.

        Args:
            version: Version to check

        Returns:
            True if version matches any OR group
        """
        if not self.constraints:
            return True

        for and_group in self.constraints:
            if all(c.matches(version) for c in and_group):
                return True
        return False

    @classmethod
    def parse(cls, spec_str: str) -> "VersionSpec":
        """Parse version specification string.

        Args:
            spec_str: Specification string

        Returns:
            VersionSpec instance
        """
        spec_str = spec_str.strip()
        if not spec_str or spec_str == "*":
            return cls(constraints=[])

        # Split by OR
        or_parts = spec_str.split("||")
        constraints: list[list[VersionConstraint]] = []

        for or_part in or_parts:
            # Split by AND (comma)
            and_parts = [p.strip() for p in or_part.split(",")]
            and_group = [
                VersionConstraint.parse(p)
                for p in and_parts
                if p
            ]
            if and_group:
                constraints.append(and_group)

        return cls(constraints=constraints)


@dataclass
class ValidatorVersionInfo:
    """Version information for a validator.

    Attributes:
        name: Validator name
        version: Current version
        min_truthound_version: Minimum Truthound version required
        max_truthound_version: Maximum Truthound version supported
        python_version: Required Python version spec
        dependencies: Dependencies with version specs
    """

    name: str
    version: SemanticVersion
    min_truthound_version: SemanticVersion | None = None
    max_truthound_version: SemanticVersion | None = None
    python_version: VersionSpec = field(default_factory=lambda: VersionSpec.parse(">=3.11"))
    dependencies: dict[str, VersionSpec] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": str(self.version),
            "min_truthound_version": str(self.min_truthound_version)
                if self.min_truthound_version else None,
            "max_truthound_version": str(self.max_truthound_version)
                if self.max_truthound_version else None,
            "python_version": str(self.python_version),
            "dependencies": {k: str(v) for k, v in self.dependencies.items()},
        }


class VersionChecker:
    """Checks version compatibility for validators.

    Validates that:
    - Validator is compatible with current Truthound version
    - Required Python version is met
    - Dependencies are available and compatible
    """

    def __init__(
        self,
        truthound_version: str = "0.2.0",
        python_version: str | None = None,
    ):
        """Initialize version checker.

        Args:
            truthound_version: Current Truthound version
            python_version: Current Python version (auto-detected if None)
        """
        self.truthound_version = SemanticVersion.parse(truthound_version)

        if python_version is None:
            import sys
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.python_version = SemanticVersion.parse(python_version)

        self._compatibility_cache: dict[str, VersionCompatibility] = {}

    def get_validator_version_info(
        self,
        validator_class: type,
    ) -> ValidatorVersionInfo:
        """Extract version information from validator.

        Args:
            validator_class: Validator class

        Returns:
            ValidatorVersionInfo
        """
        name = getattr(validator_class, "name", validator_class.__name__)
        version_str = getattr(validator_class, "version", "1.0.0")

        # Look for version requirements
        min_th = getattr(validator_class, "min_truthound_version", None)
        max_th = getattr(validator_class, "max_truthound_version", None)
        python_req = getattr(validator_class, "python_version", ">=3.11")
        deps = getattr(validator_class, "dependencies", {})

        # Ensure deps is a dict (not a set or other iterable)
        if not isinstance(deps, dict):
            deps = {}

        return ValidatorVersionInfo(
            name=name,
            version=SemanticVersion.parse(version_str),
            min_truthound_version=SemanticVersion.parse(min_th) if min_th else None,
            max_truthound_version=SemanticVersion.parse(max_th) if max_th else None,
            python_version=VersionSpec.parse(python_req),
            dependencies={k: VersionSpec.parse(v) for k, v in deps.items()},
        )

    def check_compatibility(
        self,
        validator_class: type,
        raise_on_incompatible: bool = True,
    ) -> VersionCompatibility:
        """Check if validator is compatible with current environment.

        Args:
            validator_class: Validator class to check
            raise_on_incompatible: Whether to raise on incompatibility

        Returns:
            VersionCompatibility level

        Raises:
            VersionConflictError: If incompatible and raise_on_incompatible is True
        """
        info = self.get_validator_version_info(validator_class)

        # Check Truthound version
        if info.min_truthound_version:
            if self.truthound_version < info.min_truthound_version:
                if raise_on_incompatible:
                    raise VersionConflictError(
                        f"Validator '{info.name}' requires Truthound >="
                        f"{info.min_truthound_version}, but {self.truthound_version} is installed",
                        validator_name=info.name,
                        required=f">={info.min_truthound_version}",
                        actual=str(self.truthound_version),
                    )
                return VersionCompatibility.INCOMPATIBLE

        if info.max_truthound_version:
            if self.truthound_version > info.max_truthound_version:
                if raise_on_incompatible:
                    raise VersionConflictError(
                        f"Validator '{info.name}' requires Truthound <="
                        f"{info.max_truthound_version}, but {self.truthound_version} is installed",
                        validator_name=info.name,
                        required=f"<={info.max_truthound_version}",
                        actual=str(self.truthound_version),
                    )
                return VersionCompatibility.INCOMPATIBLE

        # Check Python version
        if not info.python_version.matches(self.python_version):
            if raise_on_incompatible:
                raise VersionConflictError(
                    f"Validator '{info.name}' requires Python {info.python_version}, "
                    f"but {self.python_version} is installed",
                    validator_name=info.name,
                    required=str(info.python_version),
                    actual=str(self.python_version),
                )
            return VersionCompatibility.INCOMPATIBLE

        # Check dependencies
        for dep_name, dep_spec in info.dependencies.items():
            try:
                import importlib.metadata
                dep_version = importlib.metadata.version(dep_name)
                dep_semver = SemanticVersion.parse(dep_version)

                if not dep_spec.matches(dep_semver):
                    if raise_on_incompatible:
                        raise VersionConflictError(
                            f"Validator '{info.name}' requires {dep_name} {dep_spec}, "
                            f"but {dep_version} is installed",
                            validator_name=info.name,
                            required=f"{dep_name} {dep_spec}",
                            actual=dep_version,
                        )
                    return VersionCompatibility.INCOMPATIBLE
            except importlib.metadata.PackageNotFoundError:
                if raise_on_incompatible:
                    raise VersionConflictError(
                        f"Validator '{info.name}' requires {dep_name}, which is not installed",
                        validator_name=info.name,
                        required=f"{dep_name} {dep_spec}",
                        actual="not installed",
                    )
                return VersionCompatibility.INCOMPATIBLE

        return VersionCompatibility.COMPATIBLE

    def check_all(
        self,
        validators: list[type],
        raise_on_first: bool = False,
    ) -> dict[str, VersionCompatibility]:
        """Check compatibility of multiple validators.

        Args:
            validators: List of validator classes
            raise_on_first: Raise on first incompatibility

        Returns:
            Dictionary mapping validator names to compatibility
        """
        results = {}
        for validator_class in validators:
            name = getattr(validator_class, "name", validator_class.__name__)
            try:
                results[name] = self.check_compatibility(
                    validator_class,
                    raise_on_incompatible=raise_on_first,
                )
            except VersionConflictError as e:
                if raise_on_first:
                    raise
                results[name] = VersionCompatibility.INCOMPATIBLE
        return results


def check_compatibility(
    validator_class: type,
    truthound_version: str = "0.2.0",
) -> VersionCompatibility:
    """Check validator compatibility.

    Args:
        validator_class: Validator to check
        truthound_version: Truthound version to check against

    Returns:
        VersionCompatibility level
    """
    checker = VersionChecker(truthound_version=truthound_version)
    return checker.check_compatibility(validator_class, raise_on_incompatible=False)
