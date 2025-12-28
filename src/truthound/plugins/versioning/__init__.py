"""Plugin versioning module.

This module provides version constraint handling and compatibility
checking for plugins.

Components:
    - VersionConstraint: Defines version requirements
    - VersionResolver: Checks compatibility
    - CompatibilityReport: Detailed compatibility analysis

Example:
    >>> from truthound.plugins.versioning import (
    ...     VersionConstraint,
    ...     VersionResolver,
    ... )
    >>>
    >>> constraint = VersionConstraint(min_version="0.1.0", max_version="1.0.0")
    >>> resolver = VersionResolver()
    >>> report = resolver.check_compatibility("0.5.0", constraint, "0.2.0")
    >>> print(report.is_compatible)
"""

from __future__ import annotations

from truthound.plugins.versioning.constraints import (
    VersionConstraint,
    parse_constraint,
)
from truthound.plugins.versioning.resolver import (
    VersionResolver,
    CompatibilityReport,
    CompatibilityLevel,
)

__all__ = [
    "VersionConstraint",
    "parse_constraint",
    "VersionResolver",
    "CompatibilityReport",
    "CompatibilityLevel",
]
