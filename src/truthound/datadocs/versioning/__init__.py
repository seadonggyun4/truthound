"""Report versioning system for Data Docs.

This module provides version control for generated reports,
enabling tracking, comparison, and rollback of report versions.

Features:
- Multiple versioning strategies (Incremental, Semantic, Timestamp, GitLike)
- Version history tracking
- Report diff and comparison
- Integration with existing storage backends
"""

from truthound.datadocs.versioning.version import (
    ReportVersion,
    VersionInfo,
    VersioningStrategy,
    IncrementalStrategy,
    SemanticStrategy,
    TimestampStrategy,
    GitLikeStrategy,
)
from truthound.datadocs.versioning.storage import (
    VersionStorage,
    InMemoryVersionStorage,
    FileVersionStorage,
)
from truthound.datadocs.versioning.diff import (
    Change,
    ChangeType,
    DiffResult,
    DiffStrategy,
    TextDiffStrategy,
    StructuralDiffStrategy,
    SemanticDiffStrategy,
    ReportDiffer,
    diff_versions,
)

__all__ = [
    # Version
    "ReportVersion",
    "VersionInfo",
    "VersioningStrategy",
    "IncrementalStrategy",
    "SemanticStrategy",
    "TimestampStrategy",
    "GitLikeStrategy",
    # Storage
    "VersionStorage",
    "InMemoryVersionStorage",
    "FileVersionStorage",
    # Diff
    "Change",
    "ChangeType",
    "DiffResult",
    "DiffStrategy",
    "TextDiffStrategy",
    "StructuralDiffStrategy",
    "SemanticDiffStrategy",
    "ReportDiffer",
    "diff_versions",
]
