"""Result value contracts; evaluation rules and release gates belong to later phases."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ._contract import Contract, opaque, require, unique_ids
from .models import Reason


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True, kw_only=True)
class Finding(Contract):
    id: str
    rule_id: str
    rule_version: str
    object_id: str
    verdict: Verdict
    reason: Reason

    def _validate(self) -> None:
        for key in ("id", "rule_id", "rule_version", "object_id"):
            opaque(getattr(self, key), "/" + key)
        if self.verdict is Verdict.UNKNOWN:
            require(
                self.reason not in (Reason.NONE, Reason.OUT_OF_SCOPE), "unknown_reason_required"
            )
        elif self.verdict is Verdict.NOT_APPLICABLE:
            require(self.reason is Reason.OUT_OF_SCOPE, "scope_reason_required")
        else:
            require(self.reason is Reason.NONE, "unexpected_reason")


@dataclass(frozen=True, kw_only=True)
class FindingSet(Contract):
    findings: tuple[Finding, ...]

    def _validate(self) -> None:
        unique_ids(self.findings, "/findings")

    @property
    def ordered(self) -> tuple[Finding, ...]:
        return tuple(sorted(self.findings, key=lambda f: (f.rule_id, f.object_id, f.id)))
