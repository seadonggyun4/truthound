"""Schema evolution CLI commands.

This module provides CLI commands for schema evolution detection,
history management, and continuous monitoring.

Commands:
    - schema-check: Compare schemas and detect changes
    - schema-history: Manage schema version history
    - schema-watch: Monitor schemas for changes
    - schema-diff: Show diff between schema versions
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from truthound.cli_modules.common import CLIError, ConsoleOutput


# =============================================================================
# Schema Check Command
# =============================================================================


def schema_check_cmd(
    current: str = typer.Argument(..., help="Current schema file (JSON)"),
    baseline: str = typer.Option(
        None, "--baseline", "-b", help="Baseline schema file (JSON)"
    ),
    format: str = typer.Option(
        "console", "--format", "-f", help="Output format (console, json, markdown)"
    ),
    fail_on_breaking: bool = typer.Option(
        False, "--fail-on-breaking", help="Exit with error code if breaking changes found"
    ),
    detect_renames: bool = typer.Option(
        True, "--detect-renames/--no-detect-renames", help="Detect column renames"
    ),
    similarity_threshold: float = typer.Option(
        0.8, "--similarity", help="Similarity threshold for rename detection"
    ),
) -> None:
    """Check schema for changes against a baseline.

    Compare a current schema file against a baseline and detect:
    - Added columns
    - Removed columns
    - Type changes
    - Column renames
    - Nullable changes

    Examples:
        # Compare current schema to baseline
        th schema-check current.json --baseline baseline.json

        # JSON output
        th schema-check current.json -b baseline.json --format json

        # Fail CI/CD if breaking changes
        th schema-check current.json -b baseline.json --fail-on-breaking
    """
    from truthound.profiler.evolution import (
        SchemaEvolutionDetector,
        SchemaChangeSummary,
    )

    console = ConsoleOutput()

    # Load schemas
    try:
        current_path = Path(current)
        if not current_path.exists():
            raise CLIError(f"Current schema file not found: {current}")

        with open(current_path, "r") as f:
            current_schema = json.load(f)

        baseline_schema = None
        if baseline:
            baseline_path = Path(baseline)
            if not baseline_path.exists():
                raise CLIError(f"Baseline schema file not found: {baseline}")

            with open(baseline_path, "r") as f:
                baseline_schema = json.load(f)
    except json.JSONDecodeError as e:
        raise CLIError(f"Invalid JSON: {e}")

    # Detect changes
    detector = SchemaEvolutionDetector(
        detect_renames=detect_renames,
        rename_similarity_threshold=similarity_threshold,
    )

    changes = detector.detect_changes(current_schema, baseline_schema)
    summary = detector.get_change_summary(changes)

    # Output results
    if format == "json":
        output = {
            "summary": summary.to_dict(),
            "changes": [c.to_dict() for c in changes],
        }
        console.print(json.dumps(output, indent=2))

    elif format == "markdown":
        lines = [
            "# Schema Change Report",
            "",
            "## Summary",
            f"- **Total changes:** {summary.total_changes}",
            f"- **Breaking changes:** {summary.breaking_changes}",
            f"- **Compatibility:** {summary.compatibility_level.value}",
            "",
        ]

        if changes:
            lines.append("## Changes")
            lines.append("")
            for change in changes:
                prefix = "🔴 **BREAKING**" if change.breaking else "🟡"
                lines.append(f"- {prefix} {change.description}")
                if change.migration_hint:
                    lines.append(f"  - *Hint:* {change.migration_hint}")

        console.print("\n".join(lines))

    else:  # console
        console.print(f"\n{'='*60}")
        console.print("SCHEMA CHANGE REPORT")
        console.print(f"{'='*60}")
        console.print(f"Total changes:    {summary.total_changes}")
        console.print(f"Breaking changes: {summary.breaking_changes}")
        console.print(f"Compatibility:    {summary.compatibility_level.value}")
        console.print("")

        if changes:
            console.print("Changes:")
            for change in changes:
                prefix = "[BREAKING] " if change.breaking else "  "
                console.print(f"  {prefix}{change.description}")
                if change.migration_hint:
                    console.print(f"    Hint: {change.migration_hint}")
        else:
            console.print("No changes detected.")

    # Exit with error if breaking changes and flag is set
    if fail_on_breaking and summary.is_breaking():
        raise typer.Exit(1)


# =============================================================================
# Schema History Command
# =============================================================================


history_app = typer.Typer(help="Manage schema version history")


@history_app.command("init")
def history_init_cmd(
    path: str = typer.Argument("./schema_history", help="Path for history storage"),
    version_strategy: str = typer.Option(
        "semantic",
        "--strategy",
        "-s",
        help="Version strategy (semantic, incremental, timestamp, git)",
    ),
) -> None:
    """Initialize schema history storage.

    Creates a new schema history storage directory with the specified
    versioning strategy.

    Examples:
        th schema-history init ./my_history
        th schema-history init ./my_history --strategy timestamp
    """
    from truthound.profiler.evolution import SchemaHistory

    console = ConsoleOutput()

    history = SchemaHistory.create(
        storage_type="file",
        path=path,
        version_strategy=version_strategy,
    )

    console.print(f"✓ Schema history initialized at: {path}")
    console.print(f"  Version strategy: {version_strategy}")


@history_app.command("save")
def history_save_cmd(
    schema_file: str = typer.Argument(..., help="Schema file to save (JSON)"),
    history_path: str = typer.Option(
        "./schema_history", "--history", "-H", help="Path to history storage"
    ),
    version: Optional[str] = typer.Option(
        None, "--version", "-v", help="Version string (auto-generated if not provided)"
    ),
    message: Optional[str] = typer.Option(
        None, "--message", "-m", help="Version description/message"
    ),
) -> None:
    """Save a schema version to history.

    Examples:
        th schema-history save schema.json
        th schema-history save schema.json --version 2.0.0
        th schema-history save schema.json -m "Added email column"
    """
    from truthound.profiler.evolution import SchemaHistory

    console = ConsoleOutput()

    # Load schema
    try:
        with open(schema_file, "r") as f:
            schema = json.load(f)
    except FileNotFoundError:
        raise CLIError(f"Schema file not found: {schema_file}")
    except json.JSONDecodeError as e:
        raise CLIError(f"Invalid JSON: {e}")

    # Save to history
    history = SchemaHistory.create(storage_type="file", path=history_path)

    metadata = {}
    if message:
        metadata["message"] = message
    metadata["source_file"] = schema_file

    saved = history.save(schema, version=version, metadata=metadata)

    console.print(f"✓ Schema version saved")
    console.print(f"  Version ID: {saved.version_id}")
    console.print(f"  Version:    {saved.version}")
    console.print(f"  Columns:    {saved.column_count()}")

    if saved.changes_from_parent:
        console.print(f"  Changes:    {len(saved.changes_from_parent)} from previous")
        for change in saved.changes_from_parent[:3]:
            prefix = "[BREAKING] " if change.breaking else ""
            console.print(f"    - {prefix}{change.description}")
        if len(saved.changes_from_parent) > 3:
            console.print(f"    ... and {len(saved.changes_from_parent) - 3} more")


@history_app.command("list")
def history_list_cmd(
    history_path: str = typer.Option(
        "./schema_history", "--history", "-H", help="Path to history storage"
    ),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum versions to show"),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format (table, json)"
    ),
) -> None:
    """List schema versions in history.

    Examples:
        th schema-history list
        th schema-history list --limit 20
        th schema-history list --format json
    """
    from truthound.profiler.evolution import SchemaHistory

    console = ConsoleOutput()

    history = SchemaHistory.create(storage_type="file", path=history_path)
    versions = history.list(limit=limit)

    if not versions:
        console.print("No schema versions found.")
        return

    if format == "json":
        output = [v.to_dict() for v in versions]
        console.print(json.dumps(output, indent=2))
    else:
        console.print(f"\n{'Version':<12} {'ID':<10} {'Created':<20} {'Cols':<6} {'Breaking'}")
        console.print("-" * 70)
        for v in versions:
            breaking = "Yes" if v.has_breaking_changes() else "No"
            created = v.created_at.strftime("%Y-%m-%d %H:%M")
            console.print(f"{v.version:<12} {v.version_id[:8]:<10} {created:<20} {v.column_count():<6} {breaking}")


@history_app.command("show")
def history_show_cmd(
    version: str = typer.Argument(..., help="Version ID or version string"),
    history_path: str = typer.Option(
        "./schema_history", "--history", "-H", help="Path to history storage"
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format (table, json)"
    ),
) -> None:
    """Show details of a specific schema version.

    Examples:
        th schema-history show 1.0.0
        th schema-history show abc12345 --format json
    """
    from truthound.profiler.evolution import SchemaHistory

    console = ConsoleOutput()

    history = SchemaHistory.create(storage_type="file", path=history_path)

    # Try by version string first, then by ID
    schema_version = history.get_by_version(version)
    if not schema_version:
        schema_version = history.get(version)

    if not schema_version:
        raise CLIError(f"Version not found: {version}")

    if format == "json":
        console.print(json.dumps(schema_version.to_dict(), indent=2))
    else:
        console.print(f"\nVersion: {schema_version.version}")
        console.print(f"ID:      {schema_version.version_id}")
        console.print(f"Created: {schema_version.created_at}")
        console.print(f"Columns: {schema_version.column_count()}")
        console.print("")
        console.print("Schema:")
        for col, dtype in schema_version.schema.items():
            console.print(f"  {col}: {dtype}")

        if schema_version.changes_from_parent:
            console.print("\nChanges from parent:")
            for change in schema_version.changes_from_parent:
                prefix = "[BREAKING] " if change.breaking else ""
                console.print(f"  {prefix}{change.description}")


@history_app.command("rollback")
def history_rollback_cmd(
    version: str = typer.Argument(..., help="Version to rollback to"),
    history_path: str = typer.Option(
        "./schema_history", "--history", "-H", help="Path to history storage"
    ),
    reason: str = typer.Option(
        "manual_rollback", "--reason", "-r", help="Reason for rollback"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Rollback to a previous schema version.

    Creates a new version that copies the schema from the target version.

    Examples:
        th schema-history rollback 1.0.0
        th schema-history rollback 1.0.0 --reason "Incompatible change"
    """
    from truthound.profiler.evolution import SchemaHistory

    console = ConsoleOutput()

    history = SchemaHistory.create(storage_type="file", path=history_path)

    # Find target version
    target = history.get_by_version(version)
    if not target:
        target = history.get(version)

    if not target:
        raise CLIError(f"Version not found: {version}")

    # Confirm
    if not yes:
        console.print(f"Rollback to version: {target.version}")
        console.print(f"  ID: {target.version_id}")
        console.print(f"  Columns: {target.column_count()}")
        confirm = typer.confirm("Proceed with rollback?")
        if not confirm:
            raise typer.Abort()

    # Perform rollback
    new_version = history.rollback(target, reason=reason)

    if new_version:
        console.print(f"✓ Rollback complete")
        console.print(f"  New version: {new_version.version}")
        console.print(f"  Version ID:  {new_version.version_id}")
    else:
        raise CLIError("Rollback failed")


# =============================================================================
# Schema Diff Command
# =============================================================================


def schema_diff_cmd(
    source: str = typer.Argument(..., help="Source version (ID or version string)"),
    target: str = typer.Argument(None, help="Target version (default: latest)"),
    history_path: str = typer.Option(
        "./schema_history", "--history", "-H", help="Path to history storage"
    ),
    format: str = typer.Option(
        "text", "--format", "-f", help="Output format (text, json, markdown)"
    ),
) -> None:
    """Show diff between two schema versions.

    Examples:
        th schema-diff 1.0.0 2.0.0
        th schema-diff 1.0.0  # Compare to latest
        th schema-diff 1.0.0 2.0.0 --format json
    """
    from truthound.profiler.evolution import SchemaHistory

    console = ConsoleOutput()

    history = SchemaHistory.create(storage_type="file", path=history_path)

    diff = history.diff(source, target)

    if not diff:
        raise CLIError("Could not compute diff. Check version IDs.")

    if format == "json":
        console.print(json.dumps(diff.to_dict(), indent=2))
    elif format == "markdown":
        lines = [
            f"# Schema Diff: {diff.source_version.version} → {diff.target_version.version}",
            "",
            "## Summary",
            f"- Total changes: {diff.summary.total_changes}",
            f"- Breaking changes: {diff.summary.breaking_changes}",
            f"- Compatibility: {diff.summary.compatibility_level.value}",
            "",
        ]

        if diff.changes:
            lines.append("## Changes")
            for change in diff.changes:
                prefix = "🔴 **BREAKING**" if change.breaking else "🟡"
                lines.append(f"- {prefix} {change.description}")

        console.print("\n".join(lines))
    else:
        console.print(diff.format_text())


# =============================================================================
# Schema Watch Command
# =============================================================================


def schema_watch_cmd(
    sources: list[str] = typer.Argument(..., help="Schema files to watch (JSON)"),
    interval: float = typer.Option(60, "--interval", "-i", help="Poll interval (seconds)"),
    history_path: Optional[str] = typer.Option(
        None, "--history", "-H", help="Path to save history"
    ),
    alert_file: Optional[str] = typer.Option(
        None, "--alert-file", help="File to write alerts to"
    ),
    only_breaking: bool = typer.Option(
        False, "--only-breaking", help="Only alert on breaking changes"
    ),
    once: bool = typer.Option(
        False, "--once", help="Check once and exit (don't watch)"
    ),
) -> None:
    """Watch schema files for changes.

    Continuously monitors schema files and alerts when changes are detected.

    Examples:
        # Watch a single file
        th schema-watch schema.json

        # Watch multiple files
        th schema-watch schema1.json schema2.json

        # Watch with history tracking
        th schema-watch schema.json --history ./schema_history

        # Check once (for CI/CD)
        th schema-watch schema.json --once --only-breaking
    """
    from truthound.profiler.evolution import (
        SchemaWatcher,
        FileSchemaSource,
        LoggingEventHandler,
        HistoryEventHandler,
        SchemaHistory,
        WatchEvent,
    )

    console = ConsoleOutput()

    # Verify source files exist
    for source in sources:
        if not Path(source).exists():
            raise CLIError(f"Schema file not found: {source}")

    # Create watcher
    watcher = SchemaWatcher()

    # Add sources
    for source in sources:
        watcher.add_source(FileSchemaSource(source))
        console.print(f"Watching: {source}")

    # Add logging handler
    watcher.add_handler(LoggingEventHandler())

    # Add history handler if path provided
    if history_path:
        history = SchemaHistory.create(storage_type="file", path=history_path)
        watcher.add_handler(HistoryEventHandler(history))
        console.print(f"History: {history_path}")

    # Add console output handler
    def on_change(event: WatchEvent) -> None:
        if only_breaking and not event.has_breaking_changes():
            return

        console.print(f"\n{'='*60}")
        console.print(f"Schema change detected in: {event.source}")
        console.print(f"Time: {event.timestamp}")
        console.print(f"Changes: {event.summary.total_changes}")
        console.print(f"Breaking: {event.summary.breaking_changes}")
        console.print("")

        for change in event.changes:
            prefix = "[BREAKING] " if change.breaking else ""
            console.print(f"  {prefix}{change.description}")

        # Write to alert file if specified
        if alert_file:
            with open(alert_file, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")

    from truthound.profiler.evolution import CallbackEventHandler
    watcher.add_handler(CallbackEventHandler(on_change))

    if once:
        # Single check
        events = watcher.check_now()
        if events:
            console.print(f"\nDetected {len(events)} schema change event(s)")
            if only_breaking:
                breaking_events = [e for e in events if e.has_breaking_changes()]
                if breaking_events:
                    raise typer.Exit(1)
        else:
            console.print("No schema changes detected.")
    else:
        # Continuous watching
        console.print(f"\nStarting watch (interval: {interval}s)")
        console.print("Press Ctrl+C to stop\n")

        try:
            watcher.start(poll_interval=interval, daemon=False)
        except KeyboardInterrupt:
            console.print("\nStopping watcher...")
            watcher.stop()


# =============================================================================
# Module Registration
# =============================================================================


def register_evolution_commands(parent_app: typer.Typer) -> None:
    """Register schema evolution commands with the parent app.

    Args:
        parent_app: Parent Typer app to register commands to.
    """
    parent_app.command(name="schema-check")(schema_check_cmd)
    parent_app.command(name="schema-diff")(schema_diff_cmd)
    parent_app.command(name="schema-watch")(schema_watch_cmd)
    parent_app.add_typer(history_app, name="schema-history")


__all__ = [
    "register_evolution_commands",
    "schema_check_cmd",
    "schema_diff_cmd",
    "schema_watch_cmd",
    "history_app",
]
