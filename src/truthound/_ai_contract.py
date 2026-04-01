"""Shared Truthound AI contract constants.

This module is intentionally stdlib-only so core workspace checks can validate
AI artifacts without importing the optional ``truthound.ai`` namespace.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from uuid import uuid4

TRUTHOUND_AI_SCHEMA_VERSION = "1"
TRUTHOUND_AI_COMPILER_VERSION = "phase0-schema-v1"
TRUTHOUND_AI_ANALYSIS_COMPILER_VERSION = "phase2-explain-run-v1"
TRUTHOUND_AI_PROPOSAL_COMPILER_VERSION_V1 = "phase1-suggest-suite-v1"
TRUTHOUND_AI_PROPOSAL_COMPILER_VERSION = "phase1-suggest-suite-v2"

AI_ROOT_DIRNAME = "ai"
AI_PROPOSALS_DIRNAME = "proposals"
AI_ANALYSES_DIRNAME = "analyses"
AI_APPROVALS_DIRNAME = "approvals"
AI_APPROVAL_LOG_FILENAME = "approval-log.jsonl"
AI_REQUIRED_DIRS = (
    AI_PROPOSALS_DIRNAME,
    AI_ANALYSES_DIRNAME,
    AI_APPROVALS_DIRNAME,
)

SUITE_PROPOSAL_ID_RE = re.compile(r"^suite-proposal-\d{14}-[0-9a-f]{6}$")
RUN_ANALYSIS_ID_RE = re.compile(r"^run-analysis-[A-Za-z0-9._-]+$")
APPROVAL_EVENT_ID_RE = re.compile(r"^approval-event-\d{14}-[0-9a-f]{6}$")

AI_PROPOSAL_REQUIRED_KEYS = {
    "schema_version",
    "artifact_id",
    "artifact_type",
    "source_key",
    "input_refs",
    "model_provider",
    "model_name",
    "prompt_hash",
    "compiler_version",
    "approval_status",
    "approved_by",
    "approved_at",
    "redaction_policy",
    "created_at",
    "created_by",
    "workspace_root",
    "target_type",
    "summary",
    "rationale",
    "checks",
    "risks",
    "compile_status",
    "diff_preview",
}

AI_ANALYSIS_REQUIRED_KEYS = {
    "schema_version",
    "artifact_id",
    "artifact_type",
    "source_key",
    "input_refs",
    "model_provider",
    "model_name",
    "prompt_hash",
    "compiler_version",
    "approval_status",
    "approved_by",
    "approved_at",
    "redaction_policy",
    "created_at",
    "created_by",
    "workspace_root",
    "run_id",
    "summary",
    "evidence_refs",
    "failed_checks",
    "top_columns",
    "recommended_next_actions",
    "history_window",
}

AI_ANALYSIS_HISTORY_WINDOW_REQUIRED_KEYS = {
    "included",
    "history_key",
    "window_size",
    "run_count",
    "failure_count",
    "success_count",
    "latest_run_id",
    "recent_statuses",
}

AI_APPROVAL_LOG_REQUIRED_KEYS = {
    "event_id",
    "proposal_id",
    "action",
    "actor_id",
    "actor_name",
    "acted_at",
    "comment",
    "diff_hash",
}

AI_KNOWN_COMPILER_VERSIONS_BY_ARTIFACT_TYPE: dict[str, tuple[str, ...]] = {
    "suite_proposal": (
        TRUTHOUND_AI_PROPOSAL_COMPILER_VERSION,
        TRUTHOUND_AI_PROPOSAL_COMPILER_VERSION_V1,
        TRUTHOUND_AI_COMPILER_VERSION,
    ),
    "run_analysis": (
        TRUTHOUND_AI_ANALYSIS_COMPILER_VERSION,
        TRUTHOUND_AI_COMPILER_VERSION,
    ),
}


def utc_timestamp_compact() -> str:
    """Return a compact UTC timestamp suitable for artifact identifiers."""

    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def generate_suite_proposal_id() -> str:
    """Return a canonical suite proposal artifact identifier."""

    return f"suite-proposal-{utc_timestamp_compact()}-{uuid4().hex[:6]}"


def generate_approval_event_id() -> str:
    """Return a canonical approval event identifier."""

    return f"approval-event-{utc_timestamp_compact()}-{uuid4().hex[:6]}"


def analysis_artifact_id_for_run(run_id: str) -> str:
    """Return the canonical analysis artifact identifier for a run."""

    return f"run-analysis-{run_id}"


def is_valid_proposal_artifact_id(artifact_id: str) -> bool:
    return bool(SUITE_PROPOSAL_ID_RE.fullmatch(artifact_id))


def is_valid_analysis_artifact_id(artifact_id: str) -> bool:
    return bool(RUN_ANALYSIS_ID_RE.fullmatch(artifact_id))


def is_valid_approval_event_id(event_id: str) -> bool:
    return bool(APPROVAL_EVENT_ID_RE.fullmatch(event_id))


def known_compiler_versions_for_artifact_type(artifact_type: str) -> tuple[str, ...]:
    return AI_KNOWN_COMPILER_VERSIONS_BY_ARTIFACT_TYPE.get(artifact_type, ())


def default_compiler_version_for_artifact_type(artifact_type: str) -> str:
    versions = known_compiler_versions_for_artifact_type(artifact_type)
    if not versions:
        return TRUTHOUND_AI_COMPILER_VERSION
    return versions[0]


def is_known_compiler_version(artifact_type: str, compiler_version: str) -> bool:
    return compiler_version in known_compiler_versions_for_artifact_type(artifact_type)
