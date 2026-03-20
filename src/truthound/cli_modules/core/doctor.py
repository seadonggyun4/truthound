"""Doctor command for migration diagnostics and project health checks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from truthound.cli_modules.common.errors import error_boundary
from truthound.context import _detect_project_root
from truthound.schema import Schema

SCAN_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".toml"}
WORKSPACE_REQUIRED_DIRS = ("catalog", "baselines", "runs", "docs", "plugins")
RUN_ARTIFACT_REQUIRED_KEYS = {
    "run_id",
    "run_time",
    "suite_name",
    "source",
    "row_count",
    "column_count",
    "checks",
    "issues",
    "execution_issues",
}


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


@dataclass(frozen=True)
class WorkspaceFinding:
    """A single zero-config workspace issue."""

    rule_id: str
    message: str
    replacement: str
    path: Path
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "rule_id": self.rule_id,
            "message": self.message,
            "replacement": self.replacement,
            "path": str(self.path),
        }
        if self.details:
            payload["details"] = self.details
        return payload


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


def _workspace_finding(
    rule_id: str,
    message: str,
    replacement: str,
    path: Path,
    **details: Any,
) -> WorkspaceFinding:
    return WorkspaceFinding(
        rule_id=rule_id,
        message=message,
        replacement=replacement,
        path=path,
        details={key: value for key, value in details.items() if value is not None},
    )


def _resolve_workspace_root(path: Path) -> Path:
    resolved = path.resolve()
    start = resolved if resolved.is_dir() else resolved.parent
    return _detect_project_root(start)


def _load_json_object(
    path: Path,
    *,
    missing_rule: str,
    invalid_rule: str,
    missing_message: str,
    invalid_message: str,
    replacement: str,
) -> tuple[dict[str, Any] | None, list[WorkspaceFinding]]:
    if not path.exists():
        return None, [
            _workspace_finding(
                missing_rule,
                missing_message,
                replacement,
                path,
            )
        ]

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, [
            _workspace_finding(
                invalid_rule,
                invalid_message,
                replacement,
                path,
                error=str(exc),
            )
        ]

    if not isinstance(payload, dict):
        return None, [
            _workspace_finding(
                invalid_rule,
                invalid_message,
                replacement,
                path,
                payload_type=type(payload).__name__,
            )
        ]

    return payload, []


def _scan_workspace_findings(root: Path) -> list[WorkspaceFinding]:
    workspace_dir = root / ".truthound"
    findings: list[WorkspaceFinding] = []

    if not workspace_dir.exists():
        return [
            _workspace_finding(
                "workspace-missing",
                "The Truthound 3.0 zero-config workspace is missing.",
                "Run th.check(...) or truthound check once to create .truthound/.",
                workspace_dir,
            )
        ]
    if not workspace_dir.is_dir():
        return [
            _workspace_finding(
                "workspace-not-directory",
                "The .truthound path exists but is not a directory.",
                "Replace it with the standard .truthound/ workspace layout.",
                workspace_dir,
            )
        ]

    for directory_name in WORKSPACE_REQUIRED_DIRS:
        directory_path = workspace_dir / directory_name
        if not directory_path.exists():
            findings.append(
                _workspace_finding(
                    "workspace-dir-missing",
                    f"Required workspace directory '{directory_name}' is missing.",
                    "Recreate the workspace or rerun a Truthound command that manages artifacts.",
                    directory_path,
                    directory=directory_name,
                )
            )
        elif not directory_path.is_dir():
            findings.append(
                _workspace_finding(
                    "workspace-dir-invalid",
                    f"Workspace path '{directory_name}' must be a directory.",
                    "Replace the path with the standard directory layout.",
                    directory_path,
                    directory=directory_name,
                )
            )

    config_path = workspace_dir / "config.yaml"
    if not config_path.exists():
        findings.append(
            _workspace_finding(
                "workspace-config-missing",
                "Workspace config.yaml is missing.",
                "Recreate the workspace or copy back .truthound/config.yaml.",
                config_path,
            )
        )
    else:
        try:
            config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            findings.append(
                _workspace_finding(
                    "workspace-config-invalid",
                    "Workspace config.yaml is not readable YAML.",
                    "Restore a valid Truthound 3.0 workspace config.",
                    config_path,
                    error=str(exc),
                )
            )
        else:
            if not isinstance(config_data, dict):
                findings.append(
                    _workspace_finding(
                        "workspace-config-invalid",
                        "Workspace config.yaml must decode to a mapping.",
                        "Restore a valid Truthound 3.0 workspace config.",
                        config_path,
                        payload_type=type(config_data).__name__,
                    )
                )
            else:
                if config_data.get("version") != 3:
                    findings.append(
                        _workspace_finding(
                            "workspace-config-version-mismatch",
                            "Workspace config version does not match the Truthound 3.0 layout.",
                            "Regenerate the workspace so .truthound/config.yaml declares version: 3.",
                            config_path,
                            version=config_data.get("version"),
                        )
                    )
                context_data = config_data.get("context")
                if context_data is not None and not isinstance(context_data, dict):
                    findings.append(
                        _workspace_finding(
                            "workspace-config-context-invalid",
                            "Workspace context settings must be stored as a mapping.",
                            "Rewrite .truthound/config.yaml with a valid context block.",
                            config_path,
                            payload_type=type(context_data).__name__,
                        )
                    )

    catalog_dir = workspace_dir / "catalog"
    baselines_dir = workspace_dir / "baselines"
    runs_dir = workspace_dir / "runs"

    _, catalog_findings = _load_json_object(
        catalog_dir / "assets.json",
        missing_rule="catalog-index-missing",
        invalid_rule="catalog-index-invalid",
        missing_message="Catalog index is missing.",
        invalid_message="Catalog index is not valid JSON.",
        replacement="Re-run a Truthound command that touches tracked assets to rebuild catalog/assets.json.",
    )
    findings.extend(catalog_findings)

    baseline_index, baseline_findings = _load_json_object(
        baselines_dir / "index.json",
        missing_rule="baseline-index-missing",
        invalid_rule="baseline-index-invalid",
        missing_message="Baseline index is missing.",
        invalid_message="Baseline index is not valid JSON.",
        replacement="Rebuild the workspace baseline index under .truthound/baselines/index.json.",
    )
    findings.extend(baseline_findings)

    metric_history_path = baselines_dir / "metric-history.json"
    if metric_history_path.exists():
        _, metric_findings = _load_json_object(
            metric_history_path,
            missing_rule="metric-history-missing",
            invalid_rule="metric-history-invalid",
            missing_message="Metric history is missing.",
            invalid_message="Metric history is not valid JSON.",
            replacement="Delete or repair .truthound/baselines/metric-history.json so Truthound can rewrite it.",
        )
        findings.extend(metric_findings)

    baselines_root = baselines_dir.resolve()
    if baseline_index is not None:
        for source_key, entry in baseline_index.items():
            if not isinstance(entry, dict):
                findings.append(
                    _workspace_finding(
                        "baseline-entry-invalid",
                        "A baseline index entry must be a JSON object.",
                        "Rewrite the baseline index entry with the standard schema metadata shape.",
                        baselines_dir / "index.json",
                        source_key=source_key,
                        payload_type=type(entry).__name__,
                    )
                )
                continue

            schema_file = entry.get("schema_file")
            if not schema_file:
                findings.append(
                    _workspace_finding(
                        "baseline-entry-missing-schema",
                        "A baseline index entry is missing its schema_file reference.",
                        "Rebuild the baseline entry or regenerate the baseline schema.",
                        baselines_dir / "index.json",
                        source_key=source_key,
                    )
                )
                continue

            schema_path = baselines_dir / str(schema_file)
            resolved_schema_path = schema_path.resolve()
            if not resolved_schema_path.is_relative_to(baselines_root):
                findings.append(
                    _workspace_finding(
                        "baseline-entry-path-escape",
                        "A baseline entry points outside the baselines directory.",
                        "Update schema_file so it stays within .truthound/baselines/.",
                        schema_path,
                        source_key=source_key,
                    )
                )
                continue

            if not schema_path.exists():
                findings.append(
                    _workspace_finding(
                        "baseline-entry-missing-schema",
                        "A baseline index entry points to a missing schema file.",
                        "Restore the schema file or regenerate the baseline.",
                        schema_path,
                        source_key=source_key,
                    )
                )
                continue

            try:
                Schema.load(schema_path)
            except Exception as exc:
                findings.append(
                    _workspace_finding(
                        "baseline-schema-invalid",
                        "A baseline schema file is unreadable or malformed.",
                        "Delete or repair the schema file, then regenerate the baseline.",
                        schema_path,
                        source_key=source_key,
                        error=str(exc),
                    )
                )

    for run_path in sorted(runs_dir.glob("*.json")):
        payload, run_findings = _load_json_object(
            run_path,
            missing_rule="run-artifact-missing",
            invalid_rule="run-artifact-invalid",
            missing_message="A persisted run artifact is missing.",
            invalid_message="A persisted run artifact is not valid JSON.",
            replacement="Delete or repair the run artifact so Truthound can regenerate it on the next run.",
        )
        findings.extend(run_findings)
        if payload is None:
            continue

        missing_keys = sorted(RUN_ARTIFACT_REQUIRED_KEYS - payload.keys())
        if missing_keys:
            findings.append(
                _workspace_finding(
                    "run-artifact-shape-invalid",
                    "A persisted run artifact does not match the canonical ValidationRunResult shape.",
                    "Regenerate the run artifact from a fresh Truthound 3.0 validation run.",
                    run_path,
                    missing_keys=", ".join(missing_keys),
                )
            )

    return findings


def _render_workspace_text(findings: list[WorkspaceFinding], root: Path) -> str:
    if not findings:
        return (
            f"Truthound 3.0 workspace doctor found no structural issues under {root}.\n"
            "The zero-config workspace layout is present and readable."
        )

    lines = [
        f"Truthound 3.0 workspace doctor found {len(findings)} issue(s) under {root}:",
    ]
    for finding in findings:
        lines.extend(
            [
                "",
                f"[{finding.rule_id}] {finding.path}",
                f"  {finding.message}",
                f"  Replace with: {finding.replacement}",
            ]
        )
        if finding.details:
            details = ", ".join(
                f"{key}={value}" for key, value in sorted(finding.details.items())
            )
            lines.append(f"  Details: {details}")
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
    workspace: Annotated[
        bool,
        typer.Option(
            "--workspace",
            help="Inspect the .truthound zero-config workspace layout and artifacts.",
        ),
    ] = False,
    format: Annotated[
        str,
        typer.Option("--format", help="Output format: text or json."),
    ] = "text",
) -> None:
    """Run migration diagnostics and workspace health checks for Truthound 3.0."""
    if not migrate_2to3 and not workspace:
        typer.echo(
            "Specify --migrate-2to3 and/or --workspace to run Truthound 3.0 diagnostics.",
            err=True,
        )
        raise typer.Exit(2)

    payloads: dict[str, Any] = {}
    text_sections: list[str] = []
    has_issues = False

    if migrate_2to3:
        migration_root = path.resolve()
        migration_findings = _scan_for_2_to_3_findings(migration_root)
        payloads["migration_2to3"] = {
            "root": str(migration_root),
            "issues": [finding.to_dict() for finding in migration_findings],
        }
        text_sections.append(_render_text(migration_findings, migration_root))
        has_issues = has_issues or bool(migration_findings)

    if workspace:
        workspace_root = _resolve_workspace_root(path)
        workspace_findings = _scan_workspace_findings(workspace_root)
        payloads["workspace"] = {
            "root": str(workspace_root),
            "workspace_dir": str(workspace_root / ".truthound"),
            "issues": [finding.to_dict() for finding in workspace_findings],
        }
        text_sections.append(_render_workspace_text(workspace_findings, workspace_root))
        has_issues = has_issues or bool(workspace_findings)

    if format == "json":
        if len(payloads) == 1:
            typer.echo(json.dumps(next(iter(payloads.values())), indent=2))
        else:
            typer.echo(json.dumps({"checks": payloads}, indent=2))
    else:
        typer.echo("\n\n".join(text_sections))

    if has_issues:
        raise typer.Exit(1)
