"""Doctor command for migration diagnostics and project health checks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from truthound._applied_suite import (
    APPLIED_SUITE_CHECK_REQUIRED_KEYS,
    APPLIED_SUITE_INDEX_REQUIRED_KEYS,
    APPLIED_SUITE_REQUIRED_KEYS,
    APPLIED_SUITES_DIRNAME,
    APPLIED_SUITES_INDEX_FILENAME,
    applied_suite_filename,
)
from truthound._ai_contract import (
    AI_ANALYSIS_REQUIRED_KEYS,
    AI_ANALYSIS_HISTORY_WINDOW_REQUIRED_KEYS,
    AI_APPROVALS_DIRNAME,
    AI_APPROVAL_LOG_FILENAME,
    AI_APPROVAL_LOG_REQUIRED_KEYS,
    AI_PROPOSAL_REQUIRED_KEYS,
    AI_REQUIRED_DIRS,
    AI_ROOT_DIRNAME,
    TRUTHOUND_AI_ANALYSIS_COMPILER_VERSION,
    TRUTHOUND_AI_PROPOSAL_COMPILER_VERSION,
    TRUTHOUND_AI_SCHEMA_VERSION,
    analysis_artifact_id_for_run,
    is_known_compiler_version,
    is_valid_approval_event_id,
    is_valid_analysis_artifact_id,
    is_valid_proposal_artifact_id,
    known_compiler_versions_for_artifact_type,
)
from truthound._ai_redaction import RedactionViolationError, SummaryOnlyRedactor
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


def _load_jsonl_objects(
    path: Path,
    *,
    invalid_rule: str,
    invalid_message: str,
    replacement: str,
) -> tuple[list[dict[str, Any]] | None, list[WorkspaceFinding]]:
    if not path.exists():
        return [], []

    findings: list[WorkspaceFinding] = []
    payloads: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except Exception as exc:
            findings.append(
                _workspace_finding(
                    invalid_rule,
                    invalid_message,
                    replacement,
                    path,
                    line_no=line_no,
                    error=str(exc),
                )
            )
            continue
        if not isinstance(payload, dict):
            findings.append(
                _workspace_finding(
                    invalid_rule,
                    invalid_message,
                    replacement,
                    path,
                    line_no=line_no,
                    payload_type=type(payload).__name__,
                )
            )
            continue
        payloads.append(payload)
    return payloads, findings


def _validate_ai_redaction_policy(
    payload: dict[str, Any],
    *,
    path: Path,
    label: str,
) -> list[WorkspaceFinding]:
    findings: list[WorkspaceFinding] = []
    policy = payload.get("redaction_policy")
    if not isinstance(policy, dict):
        return [
            _workspace_finding(
                "ai-redaction-policy-invalid",
                "AI artifact redaction_policy must be a JSON object.",
                "Rewrite the artifact with a summary_only redaction policy object.",
                path,
                artifact=label,
                payload_type=type(policy).__name__,
            )
        ]

    if policy.get("mode") != "summary_only":
        findings.append(
            _workspace_finding(
                "ai-redaction-policy-invalid",
                "AI artifact redaction_policy.mode must be summary_only during Phase 1.",
                "Rewrite the artifact so it uses summary_only redaction.",
                path,
                artifact=label,
                mode=policy.get("mode"),
            )
        )
    if policy.get("raw_samples_allowed") not in (False, None):
        findings.append(
            _workspace_finding(
                "ai-redaction-policy-invalid",
                "AI artifacts may not allow raw samples during Phase 1.",
                "Set raw_samples_allowed to false.",
                path,
                artifact=label,
            )
        )
    if policy.get("pii_literals_allowed") not in (False, None):
        findings.append(
            _workspace_finding(
                "ai-redaction-policy-invalid",
                "AI artifacts may not allow PII literals during Phase 1.",
                "Set pii_literals_allowed to false.",
                path,
                artifact=label,
            )
        )

    try:
        SummaryOnlyRedactor().assert_safe(payload, label=label)
    except RedactionViolationError as exc:
        findings.append(
            _workspace_finding(
                "ai-redaction-violation",
                "AI artifact contains row-level or PII-like outbound content.",
                "Rewrite the artifact so it only contains aggregate, redacted content.",
                path,
                artifact=label,
                error=str(exc),
            )
        )
    return findings


def _validate_ai_proposal_diff_preview(
    payload: dict[str, Any],
    *,
    path: Path,
) -> list[WorkspaceFinding]:
    diff_preview = payload.get("diff_preview")
    if not isinstance(diff_preview, dict):
        return [
            _workspace_finding(
                "ai-proposal-diff-invalid",
                "Proposal diff_preview must be a JSON object.",
                "Rewrite the proposal artifact with a valid diff_preview object.",
                path,
            )
        ]

    compiler_version = str(payload.get("compiler_version", ""))
    if compiler_version != TRUTHOUND_AI_PROPOSAL_COMPILER_VERSION:
        return []

    findings: list[WorkspaceFinding] = []
    required_keys = {
        "current_suite",
        "proposed_suite",
        "added",
        "already_present",
        "conflicts",
        "rejected",
        "counts",
    }
    missing_keys = sorted(required_keys - diff_preview.keys())
    if missing_keys:
        findings.append(
            _workspace_finding(
                "ai-proposal-diff-invalid",
                "Formal proposal diff_preview is missing required typed diff fields.",
                "Rewrite the proposal artifact with a full typed ValidationSuite diff preview.",
                path,
                missing_keys=", ".join(missing_keys),
            )
        )
        return findings

    def _validate_suite_snapshot(label: str) -> None:
        snapshot = diff_preview.get(label)
        if not isinstance(snapshot, dict):
            findings.append(
                _workspace_finding(
                    "ai-proposal-diff-invalid",
                    f"Formal proposal diff_preview.{label} must be a JSON object.",
                    "Rewrite the proposal artifact with a typed ValidationSuite snapshot.",
                    path,
                    snapshot=label,
                )
            )
            return
        checks = snapshot.get("checks", [])
        if not isinstance(checks, list):
            findings.append(
                _workspace_finding(
                    "ai-proposal-diff-invalid",
                    f"Formal proposal diff_preview.{label}.checks must be a JSON array.",
                    "Rewrite the proposal artifact with a typed ValidationSuite snapshot.",
                    path,
                    snapshot=label,
                )
            )
            return
        check_count = snapshot.get("check_count")
        if check_count != len(checks):
            findings.append(
                _workspace_finding(
                    "ai-proposal-diff-invalid",
                    f"Formal proposal diff_preview.{label}.check_count must match the nested checks length.",
                    "Rewrite the proposal artifact so typed ValidationSuite counts match nested checks.",
                    path,
                    snapshot=label,
                    check_count=check_count,
                    actual_count=len(checks),
                )
            )

    _validate_suite_snapshot("current_suite")
    _validate_suite_snapshot("proposed_suite")

    counts = diff_preview.get("counts")
    if not isinstance(counts, dict):
        findings.append(
            _workspace_finding(
                "ai-proposal-diff-invalid",
                "Formal proposal diff_preview.counts must be a JSON object.",
                "Rewrite the proposal artifact with typed diff counts.",
                path,
            )
        )
        return findings

    expected_counts = {
        "added": len(diff_preview.get("added", [])) if isinstance(diff_preview.get("added"), list) else None,
        "already_present": len(diff_preview.get("already_present", [])) if isinstance(diff_preview.get("already_present"), list) else None,
        "conflicts": len(diff_preview.get("conflicts", [])) if isinstance(diff_preview.get("conflicts"), list) else None,
        "rejected": len(diff_preview.get("rejected", [])) if isinstance(diff_preview.get("rejected"), list) else None,
    }
    for key, actual in expected_counts.items():
        if actual is None:
            findings.append(
                _workspace_finding(
                    "ai-proposal-diff-invalid",
                    f"Formal proposal diff_preview.{key} must be a JSON array.",
                    "Rewrite the proposal artifact with typed diff arrays.",
                    path,
                    diff_field=key,
                )
            )
            continue
        if counts.get(key) != actual:
            findings.append(
                _workspace_finding(
                    "ai-proposal-diff-invalid",
                    f"Formal proposal diff_preview.counts.{key} must match the diff array length.",
                    "Rewrite the proposal artifact so typed diff counts match array lengths.",
                    path,
                    diff_field=key,
                    count=counts.get(key),
                    actual_count=actual,
                )
            )

    return findings


def _validate_ai_analysis_artifact(
    payload: dict[str, Any],
    *,
    path: Path,
) -> list[WorkspaceFinding]:
    compiler_version = str(payload.get("compiler_version", ""))
    if compiler_version != TRUTHOUND_AI_ANALYSIS_COMPILER_VERSION:
        return []

    findings: list[WorkspaceFinding] = []
    evidence_refs = payload.get("evidence_refs")
    input_refs = payload.get("input_refs")
    if not isinstance(evidence_refs, list) or not evidence_refs:
        findings.append(
            _workspace_finding(
                "ai-analysis-evidence-invalid",
                "Run analysis must include at least one evidence ref.",
                "Rewrite the analysis artifact with at least one valid evidence ref.",
                path,
            )
        )
    elif not isinstance(input_refs, list):
        findings.append(
            _workspace_finding(
                "ai-analysis-evidence-invalid",
                "Run analysis input_refs must be a JSON array.",
                "Rewrite the analysis artifact with a valid input_refs array.",
                path,
            )
        )
    else:
        available_refs = {
            item.get("ref")
            for item in input_refs
            if isinstance(item, dict) and isinstance(item.get("ref"), str)
        }
        missing_refs = [
            ref for ref in evidence_refs
            if not isinstance(ref, str) or ref not in available_refs
        ]
        if missing_refs:
            findings.append(
                _workspace_finding(
                    "ai-analysis-evidence-invalid",
                    "Run analysis evidence_refs must reference existing input_refs.",
                    "Rewrite evidence_refs so they only cite available evidence refs.",
                    path,
                    missing_refs=", ".join(str(item) for item in missing_refs),
                )
            )

    history_window = payload.get("history_window")
    if not isinstance(history_window, dict):
        findings.append(
            _workspace_finding(
                "ai-analysis-history-invalid",
                "Run analysis history_window must be a JSON object.",
                "Rewrite the analysis artifact with a fixed history_window summary object.",
                path,
            )
        )
        return findings

    missing_keys = sorted(AI_ANALYSIS_HISTORY_WINDOW_REQUIRED_KEYS - history_window.keys())
    if missing_keys:
        findings.append(
            _workspace_finding(
                "ai-analysis-history-invalid",
                "Run analysis history_window is missing required summary fields.",
                "Rewrite history_window so it includes the fixed Phase 2 summary keys.",
                path,
                missing_keys=", ".join(missing_keys),
            )
        )

    for key in ("failed_checks", "top_columns", "recommended_next_actions"):
        if not isinstance(payload.get(key), list):
            findings.append(
                _workspace_finding(
                    "ai-analysis-shape-invalid",
                    f"Run analysis {key} must be a JSON array.",
                    f"Rewrite {key} as a JSON array of strings.",
                    path,
                    field=key,
                )
            )

    if not isinstance(history_window.get("recent_statuses"), list):
        findings.append(
            _workspace_finding(
                "ai-analysis-history-invalid",
                "Run analysis history_window.recent_statuses must be a JSON array.",
                "Rewrite history_window.recent_statuses as a JSON array of status strings.",
                path,
            )
        )

    return findings


def _validate_ai_artifact_file(
    artifact_path: Path,
    *,
    expected_root: Path,
    expected_type: str,
    required_keys: set[str],
) -> list[WorkspaceFinding]:
    findings: list[WorkspaceFinding] = []
    resolved = artifact_path.resolve()
    if not resolved.is_relative_to(expected_root.resolve()):
        return [
            _workspace_finding(
                "ai-artifact-path-escape",
                "AI artifact path escapes the canonical .truthound/ai subtree.",
                "Move the artifact back under the canonical AI workspace directory.",
                artifact_path,
                artifact_type=expected_type,
            )
        ]

    payload, file_findings = _load_json_object(
        artifact_path,
        missing_rule="ai-artifact-missing",
        invalid_rule="ai-artifact-invalid",
        missing_message="AI artifact file is missing.",
        invalid_message="AI artifact file is not valid JSON.",
        replacement="Delete or repair the AI artifact JSON file.",
    )
    findings.extend(file_findings)
    if payload is None:
        return findings

    missing_keys = sorted(required_keys - payload.keys())
    if missing_keys:
        findings.append(
            _workspace_finding(
                "ai-artifact-shape-invalid",
                "AI artifact does not match the canonical field contract.",
                "Rewrite the artifact so it includes the required AI artifact fields.",
                artifact_path,
                artifact_type=expected_type,
                missing_keys=", ".join(missing_keys),
            )
        )

    file_stem = artifact_path.stem
    artifact_id = payload.get("artifact_id")
    if artifact_id != file_stem:
        findings.append(
            _workspace_finding(
                "ai-artifact-id-mismatch",
                "AI artifact_id must match the JSON filename stem.",
                "Rename the file or rewrite artifact_id so they match exactly.",
                artifact_path,
                artifact_type=expected_type,
                artifact_id=artifact_id,
                file_stem=file_stem,
            )
        )

    if payload.get("schema_version") != TRUTHOUND_AI_SCHEMA_VERSION:
        findings.append(
            _workspace_finding(
                "ai-schema-version-invalid",
                "AI artifact schema_version does not match the canonical AI contract.",
                "Rewrite the artifact with the canonical schema_version.",
                artifact_path,
                artifact_type=expected_type,
                schema_version=payload.get("schema_version"),
            )
        )

    compiler_version = str(payload.get("compiler_version", ""))
    if not is_known_compiler_version(expected_type, compiler_version):
        findings.append(
            _workspace_finding(
                "ai-compiler-version-invalid",
                "AI artifact compiler_version is not recognized for this artifact type.",
                "Rewrite the artifact with a known compiler_version for the artifact type.",
                artifact_path,
                artifact_type=expected_type,
                compiler_version=compiler_version,
                known_versions=", ".join(known_compiler_versions_for_artifact_type(expected_type)),
            )
        )

    if expected_type == "suite_proposal":
        if not is_valid_proposal_artifact_id(str(artifact_id)):
            findings.append(
                _workspace_finding(
                    "ai-artifact-id-invalid",
                    "Suite proposal artifact_id does not follow the canonical naming rule.",
                    "Rewrite the artifact_id as suite-proposal-YYYYMMDDHHMMSS-xxxxxx.",
                    artifact_path,
                    artifact_id=artifact_id,
                )
            )
        if payload.get("artifact_type") != "suite_proposal":
            findings.append(
                _workspace_finding(
                    "ai-artifact-type-invalid",
                    "Proposal artifact_type must be suite_proposal.",
                    "Rewrite artifact_type to suite_proposal.",
                    artifact_path,
                    artifact_type=payload.get("artifact_type"),
                )
            )
        if payload.get("target_type") != "validation_suite":
            findings.append(
                _workspace_finding(
                    "ai-proposal-target-invalid",
                    "Proposal target_type must be validation_suite.",
                    "Rewrite target_type to validation_suite.",
                    artifact_path,
                    target_type=payload.get("target_type"),
                )
            )
        findings.extend(
            _validate_ai_proposal_diff_preview(
                payload,
                path=artifact_path,
            )
        )
    else:
        if not is_valid_analysis_artifact_id(str(artifact_id)):
            findings.append(
                _workspace_finding(
                    "ai-artifact-id-invalid",
                    "Run analysis artifact_id does not follow the canonical naming rule.",
                    "Rewrite the artifact_id as run-analysis-<run_id>.",
                    artifact_path,
                    artifact_id=artifact_id,
                )
            )
        if payload.get("artifact_type") != "run_analysis":
            findings.append(
                _workspace_finding(
                    "ai-artifact-type-invalid",
                    "Analysis artifact_type must be run_analysis.",
                    "Rewrite artifact_type to run_analysis.",
                    artifact_path,
                    artifact_type=payload.get("artifact_type"),
                )
            )
        expected_id = analysis_artifact_id_for_run(str(payload.get("run_id", "")))
        if artifact_id != expected_id:
            findings.append(
                _workspace_finding(
                    "ai-analysis-run-binding-invalid",
                    "Run analysis artifact_id must be derived from run_id.",
                    "Rewrite artifact_id so it matches run-analysis-<run_id>.",
                    artifact_path,
                    artifact_id=artifact_id,
                    run_id=payload.get("run_id"),
                )
            )
        findings.extend(
            _validate_ai_analysis_artifact(
                payload,
                path=artifact_path,
            )
        )

    findings.extend(
        _validate_ai_redaction_policy(
            payload,
            path=artifact_path,
            label=f"{expected_type} artifact",
        )
    )
    return findings


def _scan_ai_workspace_findings(workspace_dir: Path) -> list[WorkspaceFinding]:
    ai_root = workspace_dir / AI_ROOT_DIRNAME
    if not ai_root.exists():
        return []

    findings: list[WorkspaceFinding] = []
    if not ai_root.is_dir():
        return [
            _workspace_finding(
                "ai-workspace-invalid",
                "The .truthound/ai path exists but is not a directory.",
                "Replace it with the canonical AI workspace directory layout.",
                ai_root,
            )
        ]

    for directory_name in AI_REQUIRED_DIRS:
        directory_path = ai_root / directory_name
        if not directory_path.exists():
            findings.append(
                _workspace_finding(
                    "ai-workspace-dir-missing",
                    f"Required AI workspace directory '{directory_name}' is missing.",
                    "Recreate the AI workspace layout or rewrite artifacts through the AI store.",
                    directory_path,
                    directory=directory_name,
                )
            )
        elif not directory_path.is_dir():
            findings.append(
                _workspace_finding(
                    "ai-workspace-dir-invalid",
                    f"AI workspace path '{directory_name}' must be a directory.",
                    "Replace the path with the canonical AI workspace directory layout.",
                    directory_path,
                    directory=directory_name,
                )
            )

    proposals_dir = ai_root / "proposals"
    analyses_dir = ai_root / "analyses"
    approvals_dir = ai_root / AI_APPROVALS_DIRNAME

    if proposals_dir.is_dir():
        for artifact_path in sorted(proposals_dir.glob("*.json")):
            findings.extend(
                _validate_ai_artifact_file(
                    artifact_path,
                    expected_root=ai_root,
                    expected_type="suite_proposal",
                    required_keys=AI_PROPOSAL_REQUIRED_KEYS,
                )
            )

    if analyses_dir.is_dir():
        for artifact_path in sorted(analyses_dir.glob("*.json")):
            findings.extend(
                _validate_ai_artifact_file(
                    artifact_path,
                    expected_root=ai_root,
                    expected_type="run_analysis",
                    required_keys=AI_ANALYSIS_REQUIRED_KEYS,
                )
            )

    approval_log_path = approvals_dir / AI_APPROVAL_LOG_FILENAME
    approval_events, approval_findings = _load_jsonl_objects(
        approval_log_path,
        invalid_rule="ai-approval-log-invalid",
        invalid_message="AI approval log is not valid JSONL.",
        replacement="Delete or repair the approval log so each line is a JSON object.",
    )
    findings.extend(approval_findings)
    if approval_events:
        for index, event in enumerate(approval_events, start=1):
            missing_keys = sorted(AI_APPROVAL_LOG_REQUIRED_KEYS - event.keys())
            if missing_keys:
                findings.append(
                    _workspace_finding(
                        "ai-approval-log-shape-invalid",
                        "Approval log event does not match the canonical approval field set.",
                        "Rewrite the approval log event with the required approval fields.",
                        approval_log_path,
                        line_no=index,
                        missing_keys=", ".join(missing_keys),
                    )
                )
                continue
            if not is_valid_approval_event_id(str(event.get("event_id"))):
                findings.append(
                    _workspace_finding(
                        "ai-approval-event-id-invalid",
                        "Approval log event_id does not follow the canonical naming rule.",
                        "Rewrite event_id as approval-event-YYYYMMDDHHMMSS-xxxxxx.",
                        approval_log_path,
                        line_no=index,
                        event_id=event.get("event_id"),
                    )
                )
            if not is_valid_proposal_artifact_id(str(event.get("proposal_id"))):
                findings.append(
                    _workspace_finding(
                        "ai-approval-proposal-id-invalid",
                        "Approval log proposal_id must reference a canonical suite proposal artifact.",
                        "Rewrite proposal_id so it matches a canonical suite proposal artifact_id.",
                        approval_log_path,
                        line_no=index,
                        proposal_id=event.get("proposal_id"),
                    )
                )
            findings.extend(
                _validate_ai_redaction_policy(
                    {
                        "redaction_policy": {
                            "mode": "summary_only",
                            "raw_samples_allowed": False,
                            "pii_literals_allowed": False,
                        },
                        **event,
                    },
                    path=approval_log_path,
                    label="approval event",
                )
            )

    return findings


def _validate_applied_suite_snapshot(
    snapshot: dict[str, Any] | Any,
    *,
    path: Path,
) -> list[WorkspaceFinding]:
    if not isinstance(snapshot, dict):
        return [
            _workspace_finding(
                "applied-suite-snapshot-invalid",
                "Applied suite effective snapshot must be a JSON object.",
                "Rewrite the applied suite record with a typed effective_suite_snapshot object.",
                path,
            )
        ]

    checks = snapshot.get("checks")
    if not isinstance(checks, list):
        return [
            _workspace_finding(
                "applied-suite-snapshot-invalid",
                "Applied suite effective snapshot checks must be a JSON array.",
                "Rewrite effective_suite_snapshot.checks as a JSON array.",
                path,
            )
        ]

    findings: list[WorkspaceFinding] = []
    if snapshot.get("check_count") != len(checks):
        findings.append(
            _workspace_finding(
                "applied-suite-snapshot-invalid",
                "Applied suite effective snapshot check_count must match nested checks length.",
                "Rewrite effective_suite_snapshot so check_count matches nested checks.",
                path,
                check_count=snapshot.get("check_count"),
                actual_count=len(checks),
            )
        )
    return findings


def _validate_applied_suite_record(
    suite_path: Path,
    *,
    suites_dir: Path,
    proposals_dir: Path,
    source_key: str,
    index_entry: dict[str, Any],
) -> list[WorkspaceFinding]:
    findings: list[WorkspaceFinding] = []
    resolved = suite_path.resolve()
    if not resolved.is_relative_to(suites_dir.resolve()):
        return [
            _workspace_finding(
                "applied-suite-path-escape",
                "Applied suite path escapes the canonical .truthound/suites subtree.",
                "Move the suite record back under .truthound/suites/.",
                suite_path,
            )
        ]

    payload, file_findings = _load_json_object(
        suite_path,
        missing_rule="applied-suite-missing",
        invalid_rule="applied-suite-invalid",
        missing_message="Applied suite record is missing.",
        invalid_message="Applied suite record is not valid JSON.",
        replacement="Rewrite the applied suite record through the proposal apply lifecycle.",
    )
    findings.extend(file_findings)
    if payload is None:
        return findings

    missing_keys = sorted(APPLIED_SUITE_REQUIRED_KEYS - payload.keys())
    if missing_keys:
        findings.append(
            _workspace_finding(
                "applied-suite-shape-invalid",
                "Applied suite record does not match the canonical field contract.",
                "Rewrite the applied suite record so it includes the required keys.",
                suite_path,
                missing_keys=", ".join(missing_keys),
            )
        )
        return findings

    if payload.get("source_key") != source_key:
        findings.append(
            _workspace_finding(
                "applied-suite-source-key-mismatch",
                "Applied suite record source_key must match the index entry key.",
                "Rewrite the applied suite record so it uses the canonical source_key.",
                suite_path,
                source_key=payload.get("source_key"),
                expected_source_key=source_key,
            )
        )
    if payload.get("proposal_id") != index_entry.get("proposal_id"):
        findings.append(
            _workspace_finding(
                "applied-suite-proposal-id-mismatch",
                "Applied suite record proposal_id must match the suites index entry.",
                "Rewrite the applied suite record or suites index so they reference the same proposal_id.",
                suite_path,
                proposal_id=payload.get("proposal_id"),
                expected_proposal_id=index_entry.get("proposal_id"),
            )
        )
    if payload.get("diff_hash") != index_entry.get("diff_hash"):
        findings.append(
            _workspace_finding(
                "applied-suite-diff-hash-mismatch",
                "Applied suite record diff_hash must match the suites index entry.",
                "Rewrite the applied suite record or suites index so they reference the same diff_hash.",
                suite_path,
                diff_hash=payload.get("diff_hash"),
                expected_diff_hash=index_entry.get("diff_hash"),
            )
        )

    checks = payload.get("checks")
    if not isinstance(checks, list):
        findings.append(
            _workspace_finding(
                "applied-suite-checks-invalid",
                "Applied suite checks must be a JSON array.",
                "Rewrite the applied suite record with a JSON array of applied checks.",
                suite_path,
            )
        )
    else:
        for index, check in enumerate(checks, start=1):
            if not isinstance(check, dict):
                findings.append(
                    _workspace_finding(
                        "applied-suite-checks-invalid",
                        "Applied suite checks must be JSON objects.",
                        "Rewrite applied checks as JSON objects with canonical fields.",
                        suite_path,
                        check_index=index,
                    )
                )
                continue
            missing_check_keys = sorted(APPLIED_SUITE_CHECK_REQUIRED_KEYS - check.keys())
            if missing_check_keys:
                findings.append(
                    _workspace_finding(
                        "applied-suite-checks-invalid",
                        "Applied suite check payload does not match the canonical field set.",
                        "Rewrite the applied suite check payload with canonical keys.",
                        suite_path,
                        check_index=index,
                        missing_keys=", ".join(missing_check_keys),
                    )
                )

    findings.extend(
        _validate_applied_suite_snapshot(
            payload.get("effective_suite_snapshot"),
            path=suite_path,
        )
    )

    proposal_id = str(payload.get("proposal_id", ""))
    proposal_path = proposals_dir / f"{proposal_id}.json"
    if not proposal_path.exists():
        findings.append(
            _workspace_finding(
                "applied-suite-proposal-missing",
                "Applied suite record must reference an existing AI proposal artifact.",
                "Restore the referenced proposal artifact or rewrite the applied suite record.",
                suite_path,
                proposal_id=proposal_id,
            )
        )

    try:
        SummaryOnlyRedactor().assert_safe(payload, label="applied suite record")
    except RedactionViolationError as exc:
        findings.append(
            _workspace_finding(
                "applied-suite-redaction-invalid",
                "Applied suite record contains row-level or PII-like outbound content.",
                "Rewrite the applied suite record so it contains only summary-safe data.",
                suite_path,
                error=str(exc),
            )
        )

    return findings


def _scan_applied_suites_findings(workspace_dir: Path) -> list[WorkspaceFinding]:
    suites_dir = workspace_dir / APPLIED_SUITES_DIRNAME
    if not suites_dir.exists():
        return []
    if not suites_dir.is_dir():
        return [
            _workspace_finding(
                "applied-suites-dir-invalid",
                "The .truthound/suites path exists but is not a directory.",
                "Replace it with the canonical applied suites directory layout.",
                suites_dir,
            )
        ]

    findings: list[WorkspaceFinding] = []
    index_path = suites_dir / APPLIED_SUITES_INDEX_FILENAME
    index_payload, index_findings = _load_json_object(
        index_path,
        missing_rule="applied-suites-index-missing",
        invalid_rule="applied-suites-index-invalid",
        missing_message="Applied suites index.json is missing.",
        invalid_message="Applied suites index.json is not valid JSON.",
        replacement="Rewrite the applied suites index through the proposal apply lifecycle.",
    )
    findings.extend(index_findings)
    if index_payload is None:
        return findings

    proposals_dir = workspace_dir / AI_ROOT_DIRNAME / "proposals"
    for source_key, entry in sorted(index_payload.items()):
        if not isinstance(entry, dict):
            findings.append(
                _workspace_finding(
                    "applied-suites-index-shape-invalid",
                    "Applied suites index entries must be JSON objects.",
                    "Rewrite the applied suites index so each source key maps to an object.",
                    index_path,
                    source_key=source_key,
                )
            )
            continue

        missing_keys = sorted(APPLIED_SUITE_INDEX_REQUIRED_KEYS - entry.keys())
        if missing_keys:
            findings.append(
                _workspace_finding(
                    "applied-suites-index-shape-invalid",
                    "Applied suites index entry does not match the canonical field set.",
                    "Rewrite the applied suites index entry with canonical keys.",
                    index_path,
                    source_key=source_key,
                    missing_keys=", ".join(missing_keys),
                )
            )
            continue

        if entry.get("source_key") != source_key:
            findings.append(
                _workspace_finding(
                    "applied-suites-index-source-key-mismatch",
                    "Applied suites index entry source_key must match the enclosing index key.",
                    "Rewrite the applied suites index so source_key matches the key.",
                    index_path,
                    source_key=source_key,
                    entry_source_key=entry.get("source_key"),
                )
            )

        expected_file = applied_suite_filename(source_key)
        suite_file = entry.get("suite_file")
        if suite_file != expected_file:
            findings.append(
                _workspace_finding(
                    "applied-suites-index-file-invalid",
                    "Applied suites index suite_file must follow the canonical source-key hash naming rule.",
                    "Rewrite suite_file as <source_key_hash>.json.",
                    index_path,
                    source_key=source_key,
                    suite_file=suite_file,
                    expected_suite_file=expected_file,
                )
            )
        suite_path = suites_dir / str(suite_file)
        findings.extend(
            _validate_applied_suite_record(
                suite_path,
                suites_dir=suites_dir,
                proposals_dir=proposals_dir,
                source_key=source_key,
                index_entry=entry,
            )
        )

    return findings


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

    findings.extend(_scan_ai_workspace_findings(workspace_dir))
    findings.extend(_scan_applied_suites_findings(workspace_dir))

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
