"""Pydantic models for Truthound AI artifacts, compiler contracts, and providers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from truthound._ai_contract import (
    AI_ANALYSIS_HISTORY_WINDOW_REQUIRED_KEYS,
    TRUTHOUND_AI_ANALYSIS_COMPILER_VERSION,
    TRUTHOUND_AI_COMPILER_VERSION,
    TRUTHOUND_AI_PROPOSAL_COMPILER_VERSION,
    TRUTHOUND_AI_PROPOSAL_COMPILER_VERSION_V1,
    TRUTHOUND_AI_SCHEMA_VERSION,
    analysis_artifact_id_for_run,
    default_compiler_version_for_artifact_type,
    generate_approval_event_id,
    generate_suite_proposal_id,
    is_known_compiler_version,
    is_valid_approval_event_id,
    is_valid_analysis_artifact_id,
    is_valid_proposal_artifact_id,
)
from truthound._ai_redaction import SummaryOnlyRedactor


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ArtifactType(str, Enum):
    SUITE_PROPOSAL = "suite_proposal"
    RUN_ANALYSIS = "run_analysis"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    NOT_REQUIRED = "not_required"


class RedactionMode(str, Enum):
    SUMMARY_ONLY = "summary_only"


class CompileStatus(str, Enum):
    READY = "ready"
    PARTIAL = "partial"
    REJECTED = "rejected"


class BaseStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)


class InputRef(BaseStrictModel):
    kind: str
    ref: str
    hash: str | None = None
    redacted: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActorRef(BaseStrictModel):
    actor_id: str
    actor_name: str


class RedactionPolicy(BaseStrictModel):
    mode: RedactionMode = RedactionMode.SUMMARY_ONLY
    raw_samples_allowed: bool = False
    pii_literals_allowed: bool = False

    @model_validator(mode="after")
    def _validate_summary_only(self) -> "RedactionPolicy":
        if self.mode != RedactionMode.SUMMARY_ONLY:
            raise ValueError("Phase 1 only supports summary_only redaction")
        if self.raw_samples_allowed:
            raise ValueError("Phase 1 forbids raw samples in AI artifacts and payloads")
        if self.pii_literals_allowed:
            raise ValueError("Phase 1 forbids PII literals in AI artifacts and payloads")
        return self


class ProposedCheckIntent(BaseStrictModel):
    intent: str
    columns: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "params" not in payload and "parameters" in payload:
            payload["params"] = payload.pop("parameters")
        if "columns" not in payload and "column" in payload:
            payload["columns"] = [payload.pop("column")]
        elif isinstance(payload.get("columns"), str):
            payload["columns"] = [payload["columns"]]
        return payload

    @model_validator(mode="after")
    def _validate_redaction_contract(self) -> "ProposedCheckIntent":
        SummaryOnlyRedactor().assert_safe(
            {
                "intent": self.intent,
                "columns": self.columns,
                "params": self.params,
                "rationale": self.rationale,
            },
            label="proposed check intent",
        )
        return self


class RejectedProposalItem(BaseStrictModel):
    source: Literal["compiler", "model"] = "compiler"
    intent: str
    columns: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str
    rationale: str | None = None

    @model_validator(mode="after")
    def _validate_redaction_contract(self) -> "RejectedProposalItem":
        SummaryOnlyRedactor().assert_safe(
            self.model_dump(mode="json"),
            label="rejected proposal item",
        )
        return self


class CompiledProposalCheck(BaseStrictModel):
    check_key: str
    validator_name: str
    category: str
    columns: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "check_key" not in payload and "id" in payload:
            payload["check_key"] = str(payload.pop("id"))
        if "validator_name" not in payload and "name" in payload:
            payload["validator_name"] = str(payload.pop("name"))
        if "validator_name" not in payload and "validator" in payload:
            payload["validator_name"] = str(payload.pop("validator"))
        if "columns" not in payload and "column" in payload:
            payload["columns"] = [payload.pop("column")]
        elif isinstance(payload.get("columns"), str):
            payload["columns"] = [payload["columns"]]
        payload.setdefault("category", "general")
        payload.setdefault("params", {})
        payload.setdefault("rationale", "")
        return payload

    @model_validator(mode="after")
    def _validate_redaction_contract(self) -> "CompiledProposalCheck":
        SummaryOnlyRedactor().assert_safe(
            self.model_dump(mode="json"),
            label="compiled proposal check",
        )
        return self

    def to_check_spec(self) -> Any:
        from truthound.core.suite import CheckSpec
        from truthound.validators import get_validator

        validator_name = self.validator_name
        validator_params = dict(self.params)
        if self.columns:
            validator_params.setdefault("columns", list(self.columns))

        def factory():
            validator_cls = get_validator(validator_name)
            return validator_cls(**validator_params) if validator_params else validator_cls()

        return CheckSpec(
            id=self.check_key,
            name=validator_name,
            category=self.category,
            factory=factory,
            tags=(self.category,),
            metadata={
                "proposal_check_key": self.check_key,
                "config": validator_params,
                "rationale": self.rationale,
            },
        )


class SuiteCheckSnapshot(BaseStrictModel):
    check_id: str
    check_key: str
    validator_name: str
    category: str
    columns: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    rationale: str = ""
    origin: str

    @model_validator(mode="after")
    def _validate_redaction_contract(self) -> "SuiteCheckSnapshot":
        SummaryOnlyRedactor().assert_safe(
            self.model_dump(mode="json"),
            label="suite check snapshot",
        )
        return self


class ValidationSuiteSnapshot(BaseStrictModel):
    suite_name: str
    check_count: int = 0
    schema_check_present: bool = False
    evidence_mode: str = "summary"
    min_severity: str | None = None
    checks: list[SuiteCheckSnapshot] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_shape(self) -> "ValidationSuiteSnapshot":
        if self.check_count != len(self.checks):
            object.__setattr__(self, "check_count", len(self.checks))
        return self


class ValidationSuiteConflict(BaseStrictModel):
    proposed: SuiteCheckSnapshot
    existing: SuiteCheckSnapshot


class ValidationSuiteDiffCounts(BaseStrictModel):
    added: int = 0
    already_present: int = 0
    conflicts: int = 0
    rejected: int = 0


class ValidationSuiteDiffPreview(BaseStrictModel):
    current_suite: ValidationSuiteSnapshot
    proposed_suite: ValidationSuiteSnapshot
    added: list[SuiteCheckSnapshot] = Field(default_factory=list)
    already_present: list[SuiteCheckSnapshot] = Field(default_factory=list)
    conflicts: list[ValidationSuiteConflict] = Field(default_factory=list)
    rejected: list[RejectedProposalItem] = Field(default_factory=list)
    counts: ValidationSuiteDiffCounts = Field(default_factory=ValidationSuiteDiffCounts)

    @model_validator(mode="after")
    def _validate_counts(self) -> "ValidationSuiteDiffPreview":
        expected = ValidationSuiteDiffCounts(
            added=len(self.added),
            already_present=len(self.already_present),
            conflicts=len(self.conflicts),
            rejected=len(self.rejected),
        )
        if self.counts != expected:
            object.__setattr__(self, "counts", expected)
        if self.current_suite.check_count != len(self.current_suite.checks):
            object.__setattr__(
                self,
                "current_suite",
                self.current_suite.model_copy(
                    update={"check_count": len(self.current_suite.checks)}
                ),
            )
        if self.proposed_suite.check_count != len(self.proposed_suite.checks):
            object.__setattr__(
                self,
                "proposed_suite",
                self.proposed_suite.model_copy(
                    update={"check_count": len(self.proposed_suite.checks)}
                ),
            )
        return self


class SuiteProposalLLMResponse(BaseStrictModel):
    summary: str
    rationale: str
    proposed_checks: list[ProposedCheckIntent] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    rejected_requests: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_redaction_contract(self) -> "SuiteProposalLLMResponse":
        SummaryOnlyRedactor().assert_safe(
            self.model_dump(mode="json"),
            label="suite proposal llm response",
        )
        return self


class RunAnalysisLLMResponse(BaseStrictModel):
    summary: str
    recommended_next_actions: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_redaction_contract(self) -> "RunAnalysisLLMResponse":
        if not self.evidence_refs:
            raise ValueError("run analysis response must include at least one evidence ref")
        SummaryOnlyRedactor().assert_safe(
            self.model_dump(mode="json"),
            label="run analysis llm response",
        )
        return self


def _canonical_check_key(
    *,
    validator_name: str,
    columns: list[str] | tuple[str, ...],
    params: dict[str, Any],
) -> str:
    normalized_params = json.dumps(
        _normalize_json_value(params),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    normalized_columns = ",".join(sorted(str(column) for column in columns if column))
    return f"{validator_name}|{normalized_columns}|{normalized_params}"


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


HISTORY_WINDOW_REQUIRED_KEYS = set(AI_ANALYSIS_HISTORY_WINDOW_REQUIRED_KEYS)


def _history_key_from_input_refs(input_refs: list[InputRef] | list[dict[str, Any]], default: str) -> str:
    for item in input_refs:
        if isinstance(item, InputRef):
            ref = item.ref
        elif isinstance(item, dict):
            ref = item.get("ref")
        else:
            ref = None
        if isinstance(ref, str) and ref.startswith("history-window:"):
            return ref.split(":", 1)[1] or default
    return default


def _normalize_history_window_payload(
    value: Any,
    *,
    history_key: str,
    included: bool,
) -> dict[str, Any]:
    payload = dict(value) if isinstance(value, dict) else {}
    statuses = payload.get("recent_statuses")
    if not isinstance(statuses, list):
        statuses = []

    run_count = int(payload.get("run_count", 0) or 0)
    failure_count = int(payload.get("failure_count", 0) or 0)
    success_count = int(payload.get("success_count", max(run_count - failure_count, 0)) or 0)

    window_size = payload.get("window_size", 10)
    try:
        window_size = max(1, int(window_size))
    except (TypeError, ValueError):
        window_size = 10

    return {
        "included": bool(payload.get("included", included)),
        "history_key": str(payload.get("history_key") or history_key),
        "window_size": window_size,
        "run_count": run_count,
        "failure_count": failure_count,
        "success_count": success_count,
        "latest_run_id": (
            str(payload["latest_run_id"])
            if payload.get("latest_run_id") is not None
            else None
        ),
        "recent_statuses": [str(item) for item in statuses],
    }


def _augment_legacy_analysis_input_refs(
    input_refs: list[Any],
    *,
    evidence_refs: list[Any],
) -> list[Any]:
    refs = list(input_refs)
    known_refs: set[str] = set()
    for item in refs:
        if isinstance(item, dict) and isinstance(item.get("ref"), str):
            known_refs.add(item["ref"])
    for evidence_ref in evidence_refs:
        if not isinstance(evidence_ref, str) or evidence_ref in known_refs:
            continue
        refs.append(
            {
                "kind": "legacy_evidence_ref",
                "ref": evidence_ref,
                "redacted": True,
                "metadata": {},
            }
        )
        known_refs.add(evidence_ref)
    return refs


def _empty_suite_snapshot() -> ValidationSuiteSnapshot:
    return ValidationSuiteSnapshot(
        suite_name="truthound-auto-suite",
        check_count=0,
        schema_check_present=False,
        evidence_mode="summary",
        min_severity=None,
        checks=[],
    )


def _snapshot_from_existing_suite_summary(summary: dict[str, Any] | None) -> ValidationSuiteSnapshot:
    if not isinstance(summary, dict):
        return _empty_suite_snapshot()

    raw_checks = summary.get("checks", [])
    checks: list[SuiteCheckSnapshot] = []
    for item in raw_checks if isinstance(raw_checks, list) else []:
        if not isinstance(item, dict):
            continue
        columns = [str(column) for column in item.get("columns", []) if column]
        params = _normalize_json_value(item.get("params", {}))
        validator_name = str(item.get("validator_name", "unknown"))
        checks.append(
            SuiteCheckSnapshot(
                check_id=str(item.get("check_id") or item.get("check_key") or validator_name),
                check_key=str(
                    item.get("check_key")
                    or _canonical_check_key(
                        validator_name=validator_name,
                        columns=columns,
                        params=params,
                    )
                ),
                validator_name=validator_name,
                category=str(item.get("category", "general")),
                columns=columns,
                params=params,
                tags=[str(tag) for tag in item.get("tags", []) if tag],
                rationale=str(item.get("rationale", "")),
                origin=str(item.get("origin", "current")),
            )
        )
    return ValidationSuiteSnapshot(
        suite_name=str(summary.get("suite_name", "truthound-auto-suite")),
        check_count=len(checks),
        schema_check_present=any(check.validator_name == "schema" for check in checks),
        evidence_mode=str(summary.get("evidence_mode", "summary")),
        min_severity=str(summary["min_severity"]) if summary.get("min_severity") is not None else None,
        checks=checks,
    )


def _condense_existing_suite_summary(snapshot: ValidationSuiteSnapshot) -> dict[str, Any]:
    return {
        "suite_name": snapshot.suite_name,
        "check_count": snapshot.check_count,
        "checks": [
            {
                "check_key": check.check_key,
                "validator_name": check.validator_name,
                "category": check.category,
                "columns": list(check.columns),
                "params": dict(check.params),
            }
            for check in snapshot.checks
        ],
    }


def _base_key(validator_name: str, columns: list[str] | tuple[str, ...]) -> str:
    return f"{validator_name}|{','.join(sorted(str(column) for column in columns if column))}"


def _snapshot_from_diff_item(item: Any) -> SuiteCheckSnapshot | None:
    if isinstance(item, SuiteCheckSnapshot):
        return item
    if isinstance(item, CompiledProposalCheck):
        return SuiteCheckSnapshot(
            check_id=item.check_key,
            check_key=item.check_key,
            validator_name=item.validator_name,
            category=item.category,
            columns=list(item.columns),
            params=_normalize_json_value(item.params),
            tags=[item.category],
            rationale=item.rationale,
            origin="proposal",
        )
    if isinstance(item, str):
        parts = item.split("|", 2)
        validator_name = parts[0] if parts else "unknown"
        columns = parts[1].split(",") if len(parts) > 1 and parts[1] else []
        return SuiteCheckSnapshot(
            check_id=item,
            check_key=item,
            validator_name=validator_name,
            category="general",
            columns=columns,
            params={},
            tags=[],
            rationale="",
            origin="proposal",
        )
    if isinstance(item, dict):
        columns = [str(column) for column in item.get("columns", []) if column]
        params = _normalize_json_value(item.get("params", {}))
        validator_name = str(item.get("validator_name", "unknown"))
        return SuiteCheckSnapshot(
            check_id=str(item.get("check_id") or item.get("check_key") or validator_name),
            check_key=str(
                item.get("check_key")
                or _canonical_check_key(
                    validator_name=validator_name,
                    columns=columns,
                    params=params,
                )
            ),
            validator_name=validator_name,
            category=str(item.get("category", "general")),
            columns=columns,
            params=params,
            tags=[str(tag) for tag in item.get("tags", []) if tag],
            rationale=str(item.get("rationale", "")),
            origin=str(item.get("origin", "proposal")),
        )
    return None


def _upgrade_suite_diff_preview_payload(payload: dict[str, Any]) -> ValidationSuiteDiffPreview:
    legacy = payload.get("diff_preview")
    if isinstance(legacy, ValidationSuiteDiffPreview):
        return legacy
    if not isinstance(legacy, dict):
        return ValidationSuiteDiffPreview(
            current_suite=_snapshot_from_existing_suite_summary(payload.get("existing_suite_summary")),
            proposed_suite=_snapshot_from_existing_suite_summary(payload.get("existing_suite_summary")),
        )
    if {"current_suite", "proposed_suite", "added", "already_present", "conflicts", "rejected", "counts"} <= legacy.keys():
        return ValidationSuiteDiffPreview.model_validate(legacy)

    current_suite = _snapshot_from_existing_suite_summary(payload.get("existing_suite_summary"))
    current_by_key = {check.check_key: check for check in current_suite.checks}
    current_by_base = {
        _base_key(check.validator_name, check.columns): check
        for check in current_suite.checks
    }

    added: list[SuiteCheckSnapshot] = []
    already_present: list[SuiteCheckSnapshot] = []
    conflicts: list[ValidationSuiteConflict] = []

    raw_added = legacy.get("added", [])
    raw_already_present = legacy.get("already_present", [])
    raw_conflicts = legacy.get("conflicts", [])

    if isinstance(raw_conflicts, list):
        for item in raw_conflicts:
            if not isinstance(item, dict):
                continue
            proposed = _snapshot_from_diff_item(item.get("proposed"))
            existing = _snapshot_from_diff_item(item.get("existing"))
            if proposed is None or existing is None:
                continue
            conflicts.append(
                ValidationSuiteConflict(
                    proposed=proposed,
                    existing=existing,
                )
            )

    if isinstance(raw_added, list):
        for item in raw_added:
            snapshot = _snapshot_from_diff_item(item)
            if snapshot is None:
                continue
            added.append(snapshot)

    if isinstance(raw_already_present, list):
        for item in raw_already_present:
            snapshot = _snapshot_from_diff_item(item)
            if snapshot is None:
                continue
            already_present.append(snapshot)

    if not added and not already_present and not conflicts:
        for item in payload.get("checks", []):
            snapshot = _snapshot_from_diff_item(item)
            if snapshot is None:
                continue
            if snapshot.check_key in current_by_key:
                already_present.append(snapshot)
                continue
            base_key = _base_key(snapshot.validator_name, snapshot.columns)
            existing = current_by_base.get(base_key)
            if existing is not None:
                conflicts.append(
                    ValidationSuiteConflict(
                        proposed=snapshot,
                        existing=existing,
                    )
                )
                continue
            added.append(snapshot)

    proposed_suite = ValidationSuiteSnapshot(
        suite_name=current_suite.suite_name,
        check_count=current_suite.check_count + len(added),
        schema_check_present=current_suite.schema_check_present,
        evidence_mode=current_suite.evidence_mode,
        min_severity=current_suite.min_severity,
        checks=[*current_suite.checks, *added],
    )
    rejected = [
        item if isinstance(item, RejectedProposalItem) else RejectedProposalItem.model_validate(item)
        for item in payload.get("rejected_items", [])
    ]
    return ValidationSuiteDiffPreview(
        current_suite=current_suite,
        proposed_suite=proposed_suite,
        added=added,
        already_present=already_present,
        conflicts=conflicts,
        rejected=rejected,
        counts=ValidationSuiteDiffCounts(
            added=len(added),
            already_present=len(already_present),
            conflicts=len(conflicts),
            rejected=len(rejected),
        ),
    )


class ArtifactBase(BaseStrictModel):
    schema_version: str = TRUTHOUND_AI_SCHEMA_VERSION
    artifact_id: str
    artifact_type: ArtifactType
    source_key: str
    input_refs: list[InputRef]
    model_provider: str
    model_name: str
    prompt_hash: str
    compiler_version: str = TRUTHOUND_AI_COMPILER_VERSION
    approval_status: ApprovalStatus
    approved_by: ActorRef | None = None
    approved_at: datetime | None = None
    redaction_policy: RedactionPolicy = Field(default_factory=RedactionPolicy)
    created_at: datetime = Field(default_factory=_utc_now)
    created_by: str
    workspace_root: str

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        if value != TRUTHOUND_AI_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {TRUTHOUND_AI_SCHEMA_VERSION!r} during Phase 1"
            )
        return value

    @model_validator(mode="after")
    def _validate_compiler_contract(self) -> "ArtifactBase":
        artifact_type = (
            self.artifact_type.value
            if isinstance(self.artifact_type, ArtifactType)
            else str(self.artifact_type)
        )
        if not is_known_compiler_version(artifact_type, self.compiler_version):
            expected = default_compiler_version_for_artifact_type(artifact_type)
            raise ValueError(
                f"compiler_version must be a known value for {artifact_type!r}; "
                f"default is {expected!r}"
            )
        SummaryOnlyRedactor().assert_safe(
            self.model_dump(mode="json"),
            label=f"{self.artifact_type} artifact",
        )
        return self


class SuiteProposalArtifact(ArtifactBase):
    artifact_id: str = Field(default_factory=generate_suite_proposal_id)
    artifact_type: Literal[ArtifactType.SUITE_PROPOSAL] = ArtifactType.SUITE_PROPOSAL
    compiler_version: str = TRUTHOUND_AI_PROPOSAL_COMPILER_VERSION
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    target_type: Literal["validation_suite"] = "validation_suite"
    summary: str
    rationale: str
    checks: list[CompiledProposalCheck] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    compile_status: CompileStatus = CompileStatus.REJECTED
    diff_preview: ValidationSuiteDiffPreview = Field(
        default_factory=lambda: ValidationSuiteDiffPreview(
            current_suite=_empty_suite_snapshot(),
            proposed_suite=_empty_suite_snapshot(),
        )
    )
    rejected_items: list[RejectedProposalItem] = Field(default_factory=list)
    existing_suite_summary: dict[str, Any] | None = None
    compiled_check_count: int = 0
    rejected_check_count: int = 0
    compiler_errors: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        diff_preview = _upgrade_suite_diff_preview_payload(payload)
        payload["diff_preview"] = diff_preview
        payload["existing_suite_summary"] = _condense_existing_suite_summary(
            diff_preview.current_suite
        )
        return payload

    @field_validator("artifact_id")
    @classmethod
    def _validate_artifact_id(cls, value: str) -> str:
        if not is_valid_proposal_artifact_id(value):
            raise ValueError(
                "suite proposal artifact_id must match suite-proposal-YYYYMMDDHHMMSS-xxxxxx"
            )
        return value

    @model_validator(mode="after")
    def _validate_counts(self) -> "SuiteProposalArtifact":
        compiled_count = len(self.checks)
        rejected_count = len(self.rejected_items)
        if self.compiled_check_count != compiled_count:
            object.__setattr__(self, "compiled_check_count", compiled_count)
        if self.rejected_check_count != rejected_count:
            object.__setattr__(self, "rejected_check_count", rejected_count)
        current_summary = _condense_existing_suite_summary(self.diff_preview.current_suite)
        if self.existing_suite_summary != current_summary:
            object.__setattr__(self, "existing_suite_summary", current_summary)

        status = str(self.compile_status)
        if status == CompileStatus.READY.value and (rejected_count or self.compiler_errors):
            raise ValueError("ready proposals cannot include rejected items or compiler errors")
        if status == CompileStatus.REJECTED.value and compiled_count > 0:
            raise ValueError("rejected proposals cannot include compiled checks")
        if status == CompileStatus.PARTIAL.value and compiled_count == 0:
            raise ValueError("partial proposals must include at least one compiled check")
        return self


class RunAnalysisArtifact(ArtifactBase):
    artifact_id: str
    artifact_type: Literal[ArtifactType.RUN_ANALYSIS] = ArtifactType.RUN_ANALYSIS
    compiler_version: str = TRUTHOUND_AI_ANALYSIS_COMPILER_VERSION
    approval_status: ApprovalStatus = ApprovalStatus.NOT_REQUIRED
    run_id: str
    summary: str
    evidence_refs: list[str]
    failed_checks: list[str]
    top_columns: list[str]
    recommended_next_actions: list[str]
    history_window: dict[str, Any]

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        compiler_version = str(
            payload.get("compiler_version", TRUTHOUND_AI_ANALYSIS_COMPILER_VERSION)
        )
        input_refs = list(payload.get("input_refs") or [])
        if compiler_version == TRUTHOUND_AI_COMPILER_VERSION:
            input_refs = _augment_legacy_analysis_input_refs(
                input_refs,
                evidence_refs=list(payload.get("evidence_refs") or []),
            )
            payload["input_refs"] = input_refs

        history_key = _history_key_from_input_refs(
            input_refs,
            str(payload.get("source_key", "unknown")),
        )
        included = True
        if isinstance(payload.get("history_window"), dict) and "included" in payload["history_window"]:
            included = bool(payload["history_window"]["included"])
        payload["history_window"] = _normalize_history_window_payload(
            payload.get("history_window"),
            history_key=history_key,
            included=included,
        )
        return payload

    @field_validator("artifact_id")
    @classmethod
    def _validate_artifact_id(cls, value: str) -> str:
        if not is_valid_analysis_artifact_id(value):
            raise ValueError("run analysis artifact_id must match run-analysis-<run_id>")
        return value

    @model_validator(mode="after")
    def _validate_run_binding(self) -> "RunAnalysisArtifact":
        expected = analysis_artifact_id_for_run(self.run_id)
        if self.artifact_id != expected:
            raise ValueError(f"artifact_id must match run_id ({expected})")
        if not self.evidence_refs:
            raise ValueError("run analysis artifacts require at least one evidence_ref")
        available_refs = {item.ref for item in self.input_refs}
        missing_refs = [ref for ref in self.evidence_refs if ref not in available_refs]
        if missing_refs:
            raise ValueError(
                "run analysis evidence_refs must reference existing input_refs"
            )

        history_window = _normalize_history_window_payload(
            self.history_window,
            history_key=_history_key_from_input_refs(self.input_refs, self.source_key),
            included=bool(self.history_window.get("included", True)),
        )
        if self.history_window != history_window:
            object.__setattr__(self, "history_window", history_window)
        if self.compiler_version == TRUTHOUND_AI_ANALYSIS_COMPILER_VERSION:
            missing_keys = HISTORY_WINDOW_REQUIRED_KEYS - history_window.keys()
            if missing_keys:
                raise ValueError(
                    "run analysis history_window must include the fixed summary keys"
                )
        return self


class ApprovalLogEvent(BaseStrictModel):
    event_id: str = Field(default_factory=generate_approval_event_id)
    proposal_id: str
    action: str
    actor_id: str
    actor_name: str
    acted_at: datetime = Field(default_factory=_utc_now)
    comment: str
    diff_hash: str

    @field_validator("event_id")
    @classmethod
    def _validate_event_id(cls, value: str) -> str:
        if not is_valid_approval_event_id(value):
            raise ValueError(
                "approval event_id must match approval-event-YYYYMMDDHHMMSS-xxxxxx"
            )
        return value

    @field_validator("proposal_id")
    @classmethod
    def _validate_proposal_id(cls, value: str) -> str:
        if not is_valid_proposal_artifact_id(value):
            raise ValueError(
                "proposal_id must reference a canonical suite proposal artifact_id"
            )
        return value

    @model_validator(mode="after")
    def _validate_redaction_contract(self) -> "ApprovalLogEvent":
        SummaryOnlyRedactor().assert_safe(
            self.model_dump(mode="json"),
            label="approval event",
        )
        return self


class ProposalDecisionResult(BaseStrictModel):
    proposal: SuiteProposalArtifact
    changed: bool
    event: ApprovalLogEvent | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "ProposalDecisionResult":
        if self.changed and self.event is None:
            raise ValueError("changed proposal decisions must include an approval event")
        if not self.changed and self.event is not None:
            raise ValueError("unchanged proposal decisions may not include an approval event")
        return self


class ProposalApplyResult(BaseStrictModel):
    proposal: SuiteProposalArtifact
    changed: bool
    event: ApprovalLogEvent | None = None
    target: Literal["validation_suite"] = "validation_suite"
    applied_check_count: int = 0
    effective_suite_snapshot: ValidationSuiteSnapshot

    @model_validator(mode="after")
    def _validate_shape(self) -> "ProposalApplyResult":
        if self.changed and self.event is None:
            raise ValueError("changed proposal apply results must include an approval event")
        if not self.changed and self.event is not None:
            raise ValueError("unchanged proposal apply results may not include an approval event")
        return self


class ProviderConfig(BaseStrictModel):
    provider_name: str
    model_name: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 60.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class StructuredProviderRequest(BaseStrictModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    provider_name: str
    model_name: str
    system_prompt: str
    user_prompt: str
    response_format_name: str | None = None
    response_model: type[BaseModel] | None = None
    input_refs: list[InputRef] = Field(default_factory=list)
    redaction_policy: RedactionPolicy = Field(default_factory=RedactionPolicy)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_redaction_contract(self) -> "StructuredProviderRequest":
        SummaryOnlyRedactor().assert_safe(
            {
                "system_prompt": self.system_prompt,
                "user_prompt": self.user_prompt,
                "input_refs": [item.model_dump(mode="json") for item in self.input_refs],
                "metadata": self.metadata,
            },
            label="provider request",
        )
        return self


class StructuredProviderResponse(BaseStrictModel):
    provider_name: str
    model_name: str
    output_text: str
    parsed_output: dict[str, Any] | list[Any] | str | None = None
    usage: dict[str, int] | None = None
    finish_reason: str | None = None
    raw_response: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_redaction_contract(self) -> "StructuredProviderResponse":
        SummaryOnlyRedactor().assert_safe(
            {
                "output_text": self.output_text,
                "parsed_output": self.parsed_output,
            },
            label="provider response",
        )
        return self


class AIProvider(Protocol):
    provider_name: str
    api_key_env: str | None
    supports_structured_outputs: bool

    def generate_structured(
        self,
        request: StructuredProviderRequest,
    ) -> StructuredProviderResponse:
        """Generate a structured response synchronously."""


class OpenAIProviderSpec(BaseStrictModel):
    provider_name: Literal["openai"] = "openai"
    api_key_env: Literal["OPENAI_API_KEY"] = "OPENAI_API_KEY"
    supports_structured_outputs: Literal[True] = True


class OpenAISmokeResult(BaseStrictModel):
    provider_name: Literal["openai"] = "openai"
    model_name: str | None = None
    success: bool
    artifact_id: str | None = None
    compile_status: str | None = None
    compiled_check_count: int = 0
    rejected_check_count: int = 0
    proposal_path: str | None = None
    workspace_dir: str | None = None
    workspace_retained: bool = False
    failure_stage: Literal["config", "provider", "parse", "compile", "persist", "verify"] | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "OpenAISmokeResult":
        if self.success:
            if self.failure_stage is not None:
                raise ValueError("successful smoke results cannot include failure_stage")
            if self.error_message is not None:
                raise ValueError("successful smoke results cannot include error_message")
        else:
            if self.failure_stage is None:
                raise ValueError("failed smoke results must include failure_stage")
            if not self.error_message:
                raise ValueError("failed smoke results must include error_message")
        if self.workspace_retained and not self.workspace_dir:
            raise ValueError("workspace_retained requires workspace_dir")
        return self


class OpenAIExplainRunSmokeResult(BaseStrictModel):
    provider_name: Literal["openai"] = "openai"
    model_name: str | None = None
    success: bool
    run_id: str | None = None
    artifact_id: str | None = None
    analysis_path: str | None = None
    failed_check_count: int = 0
    top_column_count: int = 0
    evidence_ref_count: int = 0
    workspace_dir: str | None = None
    workspace_retained: bool = False
    failure_stage: Literal["config", "prepare", "provider", "parse", "persist", "verify"] | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "OpenAIExplainRunSmokeResult":
        if self.success:
            if self.failure_stage is not None:
                raise ValueError("successful explain-run smoke results cannot include failure_stage")
            if self.error_message is not None:
                raise ValueError("successful explain-run smoke results cannot include error_message")
        else:
            if self.failure_stage is None:
                raise ValueError("failed explain-run smoke results must include failure_stage")
            if not self.error_message:
                raise ValueError("failed explain-run smoke results must include error_message")
        if self.workspace_retained and not self.workspace_dir:
            raise ValueError("workspace_retained requires workspace_dir")
        return self


__all__ = [
    "AIProvider",
    "ActorRef",
    "ApprovalLogEvent",
    "ApprovalStatus",
    "ArtifactType",
    "CompileStatus",
    "CompiledProposalCheck",
    "InputRef",
    "OpenAIExplainRunSmokeResult",
    "OpenAIProviderSpec",
    "OpenAISmokeResult",
    "ProposalApplyResult",
    "ProposalDecisionResult",
    "ProposedCheckIntent",
    "ProviderConfig",
    "RedactionMode",
    "RedactionPolicy",
    "RejectedProposalItem",
    "RunAnalysisLLMResponse",
    "RunAnalysisArtifact",
    "SuiteCheckSnapshot",
    "StructuredProviderRequest",
    "StructuredProviderResponse",
    "SuiteProposalArtifact",
    "SuiteProposalLLMResponse",
    "ValidationSuiteConflict",
    "ValidationSuiteDiffCounts",
    "ValidationSuiteDiffPreview",
    "ValidationSuiteSnapshot",
]
