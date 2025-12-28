"""Report diff utilities for version comparison.

This module provides utilities for comparing report versions
and generating human-readable diffs.
"""

from __future__ import annotations

import difflib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from truthound.datadocs.versioning.version import ReportVersion


class ChangeType(Enum):
    """Types of changes between versions."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class Change:
    """Represents a single change between versions.

    Attributes:
        change_type: Type of change.
        path: Path to the changed element (e.g., "sections.overview.row_count").
        old_value: Previous value (None for additions).
        new_value: New value (None for deletions).
        line_number: Line number for text diffs.
    """

    change_type: ChangeType
    path: str
    old_value: Any = None
    new_value: Any = None
    line_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "change_type": self.change_type.value,
            "path": self.path,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "line_number": self.line_number,
        }


@dataclass
class DiffResult:
    """Result of comparing two report versions.

    Attributes:
        old_version: Source version number.
        new_version: Target version number.
        report_id: Report identifier.
        changes: List of detected changes.
        summary: Summary statistics.
        unified_diff: Unified diff text for content.
    """

    old_version: int
    new_version: int
    report_id: str
    changes: list[Change] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)
    unified_diff: str | None = None

    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return len(self.changes) > 0

    @property
    def added_count(self) -> int:
        """Count of additions."""
        return sum(1 for c in self.changes if c.change_type == ChangeType.ADDED)

    @property
    def removed_count(self) -> int:
        """Count of removals."""
        return sum(1 for c in self.changes if c.change_type == ChangeType.REMOVED)

    @property
    def modified_count(self) -> int:
        """Count of modifications."""
        return sum(1 for c in self.changes if c.change_type == ChangeType.MODIFIED)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "old_version": self.old_version,
            "new_version": self.new_version,
            "report_id": self.report_id,
            "changes": [c.to_dict() for c in self.changes],
            "summary": {
                "added": self.added_count,
                "removed": self.removed_count,
                "modified": self.modified_count,
                "total": len(self.changes),
            },
            "unified_diff": self.unified_diff,
        }


class DiffStrategy(ABC):
    """Abstract base class for diff strategies."""

    @abstractmethod
    def diff(
        self,
        old_version: ReportVersion,
        new_version: ReportVersion,
    ) -> DiffResult:
        """Compare two report versions.

        Args:
            old_version: Source version.
            new_version: Target version.

        Returns:
            DiffResult with detected changes.
        """
        pass


class TextDiffStrategy(DiffStrategy):
    """Text-based diff strategy using unified diff.

    Compares report content as text, suitable for HTML/Markdown.
    """

    def __init__(
        self,
        context_lines: int = 3,
        ignore_whitespace: bool = False,
    ) -> None:
        """Initialize text diff strategy.

        Args:
            context_lines: Lines of context around changes.
            ignore_whitespace: Ignore whitespace differences.
        """
        self._context_lines = context_lines
        self._ignore_whitespace = ignore_whitespace

    def diff(
        self,
        old_version: ReportVersion,
        new_version: ReportVersion,
    ) -> DiffResult:
        """Compare report content as text.

        Args:
            old_version: Source version.
            new_version: Target version.

        Returns:
            DiffResult with text diff.
        """
        old_content = self._get_content_str(old_version)
        new_content = self._get_content_str(new_version)

        if self._ignore_whitespace:
            old_content = self._normalize_whitespace(old_content)
            new_content = self._normalize_whitespace(new_content)

        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        # Generate unified diff
        unified = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"v{old_version.version}",
                tofile=f"v{new_version.version}",
                n=self._context_lines,
            )
        )

        # Parse changes from diff
        changes = self._parse_diff_changes(unified)

        return DiffResult(
            old_version=old_version.version,
            new_version=new_version.version,
            report_id=old_version.info.report_id,
            changes=changes,
            unified_diff="".join(unified) if unified else None,
        )

    def _get_content_str(self, version: ReportVersion) -> str:
        """Get content as string."""
        content = version.content
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="replace")
        return content

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace for comparison."""
        lines = []
        for line in text.splitlines():
            lines.append(line.strip())
        return "\n".join(lines)

    def _parse_diff_changes(self, diff_lines: list[str]) -> list[Change]:
        """Parse unified diff into Change objects."""
        changes = []
        line_num = 0

        for line in diff_lines:
            if line.startswith("@@"):
                # Parse line number from hunk header
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        line_num = int(parts[1].split(",")[0].lstrip("-"))
                    except (ValueError, IndexError):
                        pass
            elif line.startswith("-") and not line.startswith("---"):
                changes.append(
                    Change(
                        change_type=ChangeType.REMOVED,
                        path="content",
                        old_value=line[1:].rstrip("\n"),
                        line_number=line_num,
                    )
                )
                line_num += 1
            elif line.startswith("+") and not line.startswith("+++"):
                changes.append(
                    Change(
                        change_type=ChangeType.ADDED,
                        path="content",
                        new_value=line[1:].rstrip("\n"),
                        line_number=line_num,
                    )
                )
            elif not line.startswith("\\"):
                line_num += 1

        return changes


class StructuralDiffStrategy(DiffStrategy):
    """Structural diff strategy for JSON/structured data.

    Compares report metadata and structure rather than raw content.
    """

    def __init__(
        self,
        include_content: bool = False,
        max_depth: int = 10,
    ) -> None:
        """Initialize structural diff strategy.

        Args:
            include_content: Include content comparison.
            max_depth: Maximum recursion depth.
        """
        self._include_content = include_content
        self._max_depth = max_depth

    def diff(
        self,
        old_version: ReportVersion,
        new_version: ReportVersion,
    ) -> DiffResult:
        """Compare report structure.

        Args:
            old_version: Source version.
            new_version: Target version.

        Returns:
            DiffResult with structural changes.
        """
        changes = []

        # Compare version info
        old_info = old_version.info.to_dict()
        new_info = new_version.info.to_dict()
        changes.extend(self._diff_dicts(old_info, new_info, "info"))

        # Compare metadata
        old_meta = old_version.info.metadata
        new_meta = new_version.info.metadata
        changes.extend(self._diff_dicts(old_meta, new_meta, "metadata"))

        # Optionally compare content
        if self._include_content:
            old_content = self._parse_content(old_version)
            new_content = self._parse_content(new_version)
            if old_content != new_content:
                if isinstance(old_content, dict) and isinstance(new_content, dict):
                    changes.extend(
                        self._diff_dicts(old_content, new_content, "content")
                    )
                else:
                    changes.append(
                        Change(
                            change_type=ChangeType.MODIFIED,
                            path="content",
                            old_value=str(old_content)[:100],
                            new_value=str(new_content)[:100],
                        )
                    )

        return DiffResult(
            old_version=old_version.version,
            new_version=new_version.version,
            report_id=old_version.info.report_id,
            changes=changes,
        )

    def _parse_content(self, version: ReportVersion) -> Any:
        """Parse content for comparison."""
        content = version.content
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")

        if version.format == "json":
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content

        return content

    def _diff_dicts(
        self,
        old: dict,
        new: dict,
        prefix: str,
        depth: int = 0,
    ) -> list[Change]:
        """Recursively diff two dictionaries."""
        if depth >= self._max_depth:
            return []

        changes = []

        all_keys = set(old.keys()) | set(new.keys())

        for key in all_keys:
            path = f"{prefix}.{key}"

            if key not in old:
                changes.append(
                    Change(
                        change_type=ChangeType.ADDED,
                        path=path,
                        new_value=new[key],
                    )
                )
            elif key not in new:
                changes.append(
                    Change(
                        change_type=ChangeType.REMOVED,
                        path=path,
                        old_value=old[key],
                    )
                )
            elif old[key] != new[key]:
                old_val = old[key]
                new_val = new[key]

                if isinstance(old_val, dict) and isinstance(new_val, dict):
                    changes.extend(
                        self._diff_dicts(old_val, new_val, path, depth + 1)
                    )
                elif isinstance(old_val, list) and isinstance(new_val, list):
                    changes.extend(
                        self._diff_lists(old_val, new_val, path, depth + 1)
                    )
                else:
                    changes.append(
                        Change(
                            change_type=ChangeType.MODIFIED,
                            path=path,
                            old_value=old_val,
                            new_value=new_val,
                        )
                    )

        return changes

    def _diff_lists(
        self,
        old: list,
        new: list,
        prefix: str,
        depth: int = 0,
    ) -> list[Change]:
        """Diff two lists."""
        if depth >= self._max_depth:
            return []

        changes = []

        # Simple length-based diff
        max_len = max(len(old), len(new))
        for i in range(max_len):
            path = f"{prefix}[{i}]"

            if i >= len(old):
                changes.append(
                    Change(
                        change_type=ChangeType.ADDED,
                        path=path,
                        new_value=new[i],
                    )
                )
            elif i >= len(new):
                changes.append(
                    Change(
                        change_type=ChangeType.REMOVED,
                        path=path,
                        old_value=old[i],
                    )
                )
            elif old[i] != new[i]:
                old_val = old[i]
                new_val = new[i]

                if isinstance(old_val, dict) and isinstance(new_val, dict):
                    changes.extend(
                        self._diff_dicts(old_val, new_val, path, depth + 1)
                    )
                else:
                    changes.append(
                        Change(
                            change_type=ChangeType.MODIFIED,
                            path=path,
                            old_value=old_val,
                            new_value=new_val,
                        )
                    )

        return changes


class SemanticDiffStrategy(DiffStrategy):
    """Semantic diff strategy for reports.

    Understands report structure and provides meaningful change descriptions.
    """

    def __init__(self) -> None:
        """Initialize semantic diff strategy."""
        self._structural = StructuralDiffStrategy(include_content=True)

    def diff(
        self,
        old_version: ReportVersion,
        new_version: ReportVersion,
    ) -> DiffResult:
        """Compare reports semantically.

        Args:
            old_version: Source version.
            new_version: Target version.

        Returns:
            DiffResult with semantic changes.
        """
        # Start with structural diff
        result = self._structural.diff(old_version, new_version)

        # Enhance with semantic understanding
        enhanced_changes = self._enhance_changes(result.changes)

        return DiffResult(
            old_version=result.old_version,
            new_version=result.new_version,
            report_id=result.report_id,
            changes=enhanced_changes,
        )

    def _enhance_changes(self, changes: list[Change]) -> list[Change]:
        """Enhance changes with semantic meaning."""
        enhanced = []

        for change in changes:
            # Skip internal metadata
            if any(
                skip in change.path
                for skip in ["created_at", "checksum", "size_bytes"]
            ):
                continue

            enhanced.append(change)

        return enhanced


class ReportDiffer:
    """High-level API for comparing report versions.

    Example:
        differ = ReportDiffer()
        result = differ.compare(old_version, new_version)
        print(result.unified_diff)
    """

    def __init__(
        self,
        strategy: DiffStrategy | None = None,
    ) -> None:
        """Initialize report differ.

        Args:
            strategy: Diff strategy to use.
        """
        self._strategy = strategy or TextDiffStrategy()

    def compare(
        self,
        old_version: ReportVersion,
        new_version: ReportVersion,
    ) -> DiffResult:
        """Compare two report versions.

        Args:
            old_version: Source version.
            new_version: Target version.

        Returns:
            DiffResult with changes.
        """
        return self._strategy.diff(old_version, new_version)

    def compare_with_strategy(
        self,
        old_version: ReportVersion,
        new_version: ReportVersion,
        strategy: str = "text",
    ) -> DiffResult:
        """Compare using a named strategy.

        Args:
            old_version: Source version.
            new_version: Target version.
            strategy: Strategy name ("text", "structural", "semantic").

        Returns:
            DiffResult with changes.
        """
        strategies = {
            "text": TextDiffStrategy(),
            "structural": StructuralDiffStrategy(),
            "semantic": SemanticDiffStrategy(),
        }

        diff_strategy = strategies.get(strategy, TextDiffStrategy())
        return diff_strategy.diff(old_version, new_version)

    def format_diff(
        self,
        result: DiffResult,
        format: str = "unified",
    ) -> str:
        """Format diff result for display.

        Args:
            result: Diff result.
            format: Output format ("unified", "summary", "json").

        Returns:
            Formatted diff string.
        """
        if format == "unified":
            return self._format_unified(result)
        elif format == "summary":
            return self._format_summary(result)
        elif format == "json":
            return json.dumps(result.to_dict(), indent=2)
        else:
            return self._format_unified(result)

    def _format_unified(self, result: DiffResult) -> str:
        """Format as unified diff."""
        if result.unified_diff:
            return result.unified_diff

        lines = [
            f"Comparing {result.report_id}: v{result.old_version} → v{result.new_version}",
            f"Changes: +{result.added_count} -{result.removed_count} ~{result.modified_count}",
            "",
        ]

        for change in result.changes:
            if change.change_type == ChangeType.ADDED:
                lines.append(f"+ {change.path}: {change.new_value}")
            elif change.change_type == ChangeType.REMOVED:
                lines.append(f"- {change.path}: {change.old_value}")
            elif change.change_type == ChangeType.MODIFIED:
                lines.append(f"~ {change.path}: {change.old_value} → {change.new_value}")

        return "\n".join(lines)

    def _format_summary(self, result: DiffResult) -> str:
        """Format as summary."""
        return (
            f"Report: {result.report_id}\n"
            f"Versions: v{result.old_version} → v{result.new_version}\n"
            f"Added: {result.added_count}\n"
            f"Removed: {result.removed_count}\n"
            f"Modified: {result.modified_count}\n"
            f"Total changes: {len(result.changes)}"
        )


def diff_versions(
    old_version: ReportVersion,
    new_version: ReportVersion,
    strategy: str = "text",
) -> DiffResult:
    """Convenience function to diff two versions.

    Args:
        old_version: Source version.
        new_version: Target version.
        strategy: Diff strategy name.

    Returns:
        DiffResult with changes.
    """
    differ = ReportDiffer()
    return differ.compare_with_strategy(old_version, new_version, strategy)
