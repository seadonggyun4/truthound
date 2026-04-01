"""Shared contract helpers for applied validation suites."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

APPLIED_SUITES_DIRNAME = "suites"
APPLIED_SUITES_INDEX_FILENAME = "index.json"

APPLIED_SUITE_INDEX_REQUIRED_KEYS = {
    "source_key",
    "suite_file",
    "proposal_id",
    "diff_hash",
    "updated_at",
}

APPLIED_SUITE_REQUIRED_KEYS = {
    "source_key",
    "proposal_id",
    "diff_hash",
    "applied_by",
    "applied_at",
    "checks",
    "effective_suite_snapshot",
}

APPLIED_SUITE_CHECK_REQUIRED_KEYS = {
    "check_key",
    "validator_name",
    "category",
    "columns",
    "params",
    "rationale",
}


@dataclass(frozen=True)
class AppliedSuiteActor:
    actor_id: str
    actor_name: str

    def to_dict(self) -> dict[str, str]:
        return {
            "actor_id": self.actor_id,
            "actor_name": self.actor_name,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AppliedSuiteActor":
        return cls(
            actor_id=str(payload.get("actor_id", "")),
            actor_name=str(payload.get("actor_name", "")),
        )


@dataclass(frozen=True)
class AppliedSuiteRecord:
    source_key: str
    proposal_id: str
    diff_hash: str
    applied_by: AppliedSuiteActor
    applied_at: str
    checks: tuple[dict[str, Any], ...]
    effective_suite_snapshot: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_key": self.source_key,
            "proposal_id": self.proposal_id,
            "diff_hash": self.diff_hash,
            "applied_by": self.applied_by.to_dict(),
            "applied_at": self.applied_at,
            "checks": [dict(item) for item in self.checks],
            "effective_suite_snapshot": dict(self.effective_suite_snapshot),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AppliedSuiteRecord":
        missing = sorted(APPLIED_SUITE_REQUIRED_KEYS - payload.keys())
        if missing:
            raise ValueError(
                f"applied suite record missing required keys: {', '.join(missing)}"
            )
        actor_payload = payload.get("applied_by")
        if not isinstance(actor_payload, dict):
            raise ValueError("applied suite record applied_by must be a JSON object")
        checks = payload.get("checks")
        if not isinstance(checks, list):
            raise ValueError("applied suite record checks must be a JSON array")
        snapshot = payload.get("effective_suite_snapshot")
        if not isinstance(snapshot, dict):
            raise ValueError(
                "applied suite record effective_suite_snapshot must be a JSON object"
            )
        return cls(
            source_key=str(payload.get("source_key", "")),
            proposal_id=str(payload.get("proposal_id", "")),
            diff_hash=str(payload.get("diff_hash", "")),
            applied_by=AppliedSuiteActor.from_dict(actor_payload),
            applied_at=str(payload.get("applied_at", "")),
            checks=tuple(
                dict(item)
                for item in checks
                if isinstance(item, dict)
            ),
            effective_suite_snapshot=dict(snapshot),
        )


def applied_suite_filename(source_key: str) -> str:
    return f"{applied_suite_source_hash(source_key)}.json"


def applied_suite_source_hash(source_key: str) -> str:
    return hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16]


def canonical_check_key(
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


def canonical_validator_signature(
    *,
    validator_name: str,
    columns: list[str] | tuple[str, ...],
) -> str:
    normalized_columns = ",".join(sorted(str(column) for column in columns if column))
    return f"{validator_name}|{normalized_columns}"


def normalize_json_value(value: Any) -> Any:
    return _normalize_json_value(value)


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


__all__ = [
    "APPLIED_SUITE_CHECK_REQUIRED_KEYS",
    "APPLIED_SUITE_INDEX_REQUIRED_KEYS",
    "APPLIED_SUITE_REQUIRED_KEYS",
    "APPLIED_SUITES_DIRNAME",
    "APPLIED_SUITES_INDEX_FILENAME",
    "AppliedSuiteActor",
    "AppliedSuiteRecord",
    "applied_suite_filename",
    "applied_suite_source_hash",
    "canonical_check_key",
    "canonical_validator_signature",
    "normalize_json_value",
]
