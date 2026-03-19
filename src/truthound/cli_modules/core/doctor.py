"""Doctor command for migration diagnostics and project health checks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from truthound.cli_modules.common.errors import error_boundary


SCAN_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".toml"}


@dataclass(frozen=True)
class MigrationFinding:
    """A single 2.x to 3.0 migration issue."""

    rule_id: str
    message: str
    replacement: str
    path: Path
    line_no: int
    line_text: str

    def to_dict(self) -> dict[str, str | int]:
        return {
            "rule_id": self.rule_id,
            "message": self.message,
            "replacement": self.replacement,
            "path": str(self.path),
            "line_no": self.line_no,
            "line_text": self.line_text,
        }


MIGRATION_RULES: tuple[tuple[str, re.Pattern[str], str, str], ...] = (
    (
        "root-compare-import",
        re.compile(r"from\s+truthound\s+import\s+.*\bcompare\b"),
        "Root-level compare import was removed in Truthound 3.0.",
        "Import compare from truthound.drift instead.",
    ),
    (
        "root-compare-access",
        re.compile(r"\btruthound\.compare\b"),
        "truthound.compare is no longer exported from the root package.",
        "Use truthound.drift.compare.",
    ),
    (
        "legacy-report-import",
        re.compile(r"from\s+truthound\.report\s+import\s+.*\bReport\b"),
        "Legacy Report is no longer the canonical validation contract.",
        "Use ValidationRunResult from truthound.core or truthound.",
    ),
    (
        "legacy-report-annotation",
        re.compile(r":\s*Report\b|\b->\s*Report\b|\bisinstance\([^)]*,\s*Report\)"),
        "Code still assumes th.check() returns Report.",
        "Treat th.check() as returning ValidationRunResult directly.",
    ),
    (
        "legacy-checkpoint-field",
        re.compile(r"\.validation_result\b"),
        "CheckpointResult.validation_result was removed in Truthound 3.0.",
        "Use CheckpointResult.validation_run or CheckpointResult.validation_view.",
    ),
    (
        "legacy-validation-run-assumption",
        re.compile(r"\breport\.validation_run\b"),
        "Code still assumes th.check() returns a legacy report facade.",
        "Use the ValidationRunResult returned by th.check() directly.",
    ),
    (
        "legacy-validator-subclass",
        re.compile(r"^class\s+\w+\((?:[^)]*\bValidator\b[^)]*)\):"),
        "Validator subclass authoring is no longer the supported public extension model.",
        "Register declarative CheckSpecFactory implementations instead.",
    ),
)


def _iter_scan_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]

    files: list[Path] = []
    for path in root.rglob("*"):
        if (
            path.is_file()
            and path.suffix in SCAN_SUFFIXES
            and ".git" not in path.parts
            and ".venv" not in path.parts
            and "site" not in path.parts
        ):
            files.append(path)
    return sorted(files)


def _scan_for_2_to_3_findings(root: Path) -> list[MigrationFinding]:
    findings: list[MigrationFinding] = []

    for path in _iter_scan_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue

        for line_no, line in enumerate(text.splitlines(), start=1):
            for rule_id, pattern, message, replacement in MIGRATION_RULES:
                if pattern.search(line):
                    findings.append(
                        MigrationFinding(
                            rule_id=rule_id,
                            message=message,
                            replacement=replacement,
                            path=path,
                            line_no=line_no,
                            line_text=line.strip(),
                        )
                    )

    return findings


def _render_text(findings: list[MigrationFinding], root: Path) -> str:
    if not findings:
        return (
            f"Truthound 3.0 migration doctor found no blocking 2.x patterns under {root}.\n"
            "The scanned project already matches the 3.0 public contract."
        )

    lines = [
        f"Truthound 3.0 migration doctor found {len(findings)} issue(s) under {root}:",
    ]
    for finding in findings:
        lines.extend(
            [
                "",
                f"[{finding.rule_id}] {finding.path}:{finding.line_no}",
                f"  {finding.message}",
                f"  Replace with: {finding.replacement}",
                f"  Code: {finding.line_text}",
            ]
        )
    return "\n".join(lines)


@error_boundary
def doctor_cmd(
    path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Project file or directory to inspect.",
        ),
    ] = Path("."),
    migrate_2to3: Annotated[
        bool,
        typer.Option(
            "--migrate-2to3",
            help="Scan for APIs removed or changed by Truthound 3.0.",
        ),
    ] = False,
    format: Annotated[
        str,
        typer.Option("--format", help="Output format: text or json."),
    ] = "text",
) -> None:
    """Run migration diagnostics for Truthound 3.0."""
    if not migrate_2to3:
        typer.echo(
            "Specify --migrate-2to3 to scan for Truthound 3.0 breaking changes.",
            err=True,
        )
        raise typer.Exit(2)

    root = path.resolve()
    findings = _scan_for_2_to_3_findings(root)

    if format == "json":
        typer.echo(
            json.dumps(
                {
                    "root": str(root),
                    "issues": [finding.to_dict() for finding in findings],
                },
                indent=2,
            )
        )
    else:
        typer.echo(_render_text(findings, root))

    if findings:
        raise typer.Exit(1)

