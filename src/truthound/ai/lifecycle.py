"""Lifecycle services for AI suite proposal review and apply flows."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from truthound._applied_suite import AppliedSuiteActor, AppliedSuiteRecord
from truthound.ai.models import (
    ActorRef,
    ApprovalLogEvent,
    ApprovalStatus,
    ProposalApplyResult,
    ProposalDecisionResult,
    SuiteProposalArtifact,
)
from truthound.ai.store import AIArtifactStore
from truthound.context import TruthoundContext, get_context


class ProposalLifecycleError(RuntimeError):
    """Base error raised by proposal lifecycle operations."""


class ProposalNotFoundError(ProposalLifecycleError):
    """Raised when a proposal artifact cannot be resolved."""


class ProposalStateError(ProposalLifecycleError):
    """Raised when a lifecycle transition is not allowed."""


class ProposalLifecycleService:
    """High-level façade for proposal review state and apply mutation."""

    def __init__(self, context: TruthoundContext) -> None:
        self._context = context
        self._store = AIArtifactStore(context)

    def approve(self, proposal_id: str, *, actor: ActorRef, comment: str) -> ProposalDecisionResult:
        proposal = self._read_proposal(proposal_id)
        status = _approval_status_value(proposal.approval_status)
        if status == ApprovalStatus.APPLIED.value:
            raise ProposalStateError("applied proposals are terminal and cannot be re-approved")
        if status == ApprovalStatus.APPROVED.value:
            return ProposalDecisionResult(proposal=proposal, changed=False, event=None)

        updated = proposal.model_copy(
            update={
                "approval_status": ApprovalStatus.APPROVED,
                "approved_by": actor,
                "approved_at": _utc_now(),
            }
        )
        return self._write_decision(
            proposal=updated,
            action="approve",
            actor=actor,
            comment=comment,
        )

    def reject(self, proposal_id: str, *, actor: ActorRef, comment: str) -> ProposalDecisionResult:
        proposal = self._read_proposal(proposal_id)
        status = _approval_status_value(proposal.approval_status)
        if status == ApprovalStatus.APPLIED.value:
            raise ProposalStateError("applied proposals are terminal and cannot be rejected")
        if status == ApprovalStatus.REJECTED.value:
            return ProposalDecisionResult(proposal=proposal, changed=False, event=None)

        updated = proposal.model_copy(
            update={
                "approval_status": ApprovalStatus.REJECTED,
                "approved_by": None,
                "approved_at": None,
            }
        )
        return self._write_decision(
            proposal=updated,
            action="reject",
            actor=actor,
            comment=comment,
        )

    def apply(
        self,
        proposal_id: str,
        *,
        actor: ActorRef,
        comment: str | None = None,
        target: str = "validation_suite",
    ) -> ProposalApplyResult:
        if target != "validation_suite":
            raise ProposalLifecycleError("Phase 2.2 apply supports only target='validation_suite'")

        proposal = self._read_proposal(proposal_id)
        status = _approval_status_value(proposal.approval_status)
        if status == ApprovalStatus.APPLIED.value:
            return ProposalApplyResult(
                proposal=proposal,
                changed=False,
                event=None,
                target="validation_suite",
                applied_check_count=len(_select_applied_checks(proposal)),
                effective_suite_snapshot=proposal.diff_preview.proposed_suite,
            )
        if status != ApprovalStatus.APPROVED.value:
            raise ProposalStateError("proposal must be approved before it can be applied")

        diff_hash = _proposal_diff_hash(proposal)
        applied_checks = _select_applied_checks(proposal)
        record = AppliedSuiteRecord(
            source_key=proposal.source_key,
            proposal_id=proposal.artifact_id,
            diff_hash=diff_hash,
            applied_by=AppliedSuiteActor(
                actor_id=actor.actor_id,
                actor_name=actor.actor_name,
            ),
            applied_at=_utc_now().isoformat(),
            checks=tuple(
                check.model_dump(mode="json")
                for check in applied_checks
            ),
            effective_suite_snapshot=proposal.diff_preview.proposed_suite.model_dump(mode="json"),
        )
        self._context.write_applied_suite(record)

        updated = proposal.model_copy(
            update={
                "approval_status": ApprovalStatus.APPLIED,
                "approved_by": actor,
                "approved_at": _utc_now(),
            }
        )
        event = self._build_event(
            proposal=updated,
            action="apply",
            actor=actor,
            comment=comment or "",
        )
        self._store.write_proposal(updated)
        self._store.append_approval(event)
        return ProposalApplyResult(
            proposal=updated,
            changed=True,
            event=event,
            target="validation_suite",
            applied_check_count=len(applied_checks),
            effective_suite_snapshot=proposal.diff_preview.proposed_suite,
        )

    def history(self, proposal_id: str) -> list[ApprovalLogEvent]:
        self._read_proposal(proposal_id)
        events = self._store.list_approval_events(proposal_id=proposal_id)
        return sorted(events, key=lambda item: item.acted_at, reverse=True)

    def _write_decision(
        self,
        *,
        proposal: SuiteProposalArtifact,
        action: str,
        actor: ActorRef,
        comment: str,
    ) -> ProposalDecisionResult:
        event = self._build_event(
            proposal=proposal,
            action=action,
            actor=actor,
            comment=comment,
        )
        self._store.write_proposal(proposal)
        self._store.append_approval(event)
        return ProposalDecisionResult(
            proposal=proposal,
            changed=True,
            event=event,
        )

    def _read_proposal(self, proposal_id: str) -> SuiteProposalArtifact:
        try:
            return self._store.read_proposal(proposal_id)
        except FileNotFoundError as exc:
            raise ProposalNotFoundError(f"unknown proposal: {proposal_id}") from exc

    def _build_event(
        self,
        *,
        proposal: SuiteProposalArtifact,
        action: str,
        actor: ActorRef,
        comment: str,
    ) -> ApprovalLogEvent:
        return ApprovalLogEvent(
            proposal_id=proposal.artifact_id,
            action=action,
            actor_id=actor.actor_id,
            actor_name=actor.actor_name,
            comment=comment,
            diff_hash=_proposal_diff_hash(proposal),
        )


def approve_proposal(
    proposal_id: str,
    *,
    actor: ActorRef,
    comment: str,
    context: TruthoundContext | None = None,
) -> ProposalDecisionResult:
    active_context = context or get_context()
    return ProposalLifecycleService(active_context).approve(
        proposal_id,
        actor=actor,
        comment=comment,
    )


def reject_proposal(
    proposal_id: str,
    *,
    actor: ActorRef,
    comment: str,
    context: TruthoundContext | None = None,
) -> ProposalDecisionResult:
    active_context = context or get_context()
    return ProposalLifecycleService(active_context).reject(
        proposal_id,
        actor=actor,
        comment=comment,
    )


def apply_proposal(
    proposal_id: str,
    *,
    actor: ActorRef,
    comment: str | None = None,
    context: TruthoundContext | None = None,
    target: str = "validation_suite",
) -> ProposalApplyResult:
    active_context = context or get_context()
    return ProposalLifecycleService(active_context).apply(
        proposal_id,
        actor=actor,
        comment=comment,
        target=target,
    )


def list_proposal_approval_events(
    proposal_id: str,
    *,
    context: TruthoundContext | None = None,
) -> list[ApprovalLogEvent]:
    active_context = context or get_context()
    return ProposalLifecycleService(active_context).history(proposal_id)


def _proposal_diff_hash(proposal: SuiteProposalArtifact) -> str:
    payload = proposal.diff_preview.model_dump(mode="json")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _approval_status_value(status: ApprovalStatus | str) -> str:
    if isinstance(status, ApprovalStatus):
        return status.value
    return str(status)


def _select_applied_checks(proposal: SuiteProposalArtifact):
    added_keys = {item.check_key for item in proposal.diff_preview.added}
    return [check for check in proposal.checks if check.check_key in added_keys]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "ProposalLifecycleError",
    "ProposalNotFoundError",
    "ProposalStateError",
    "ProposalLifecycleService",
    "approve_proposal",
    "apply_proposal",
    "list_proposal_approval_events",
    "reject_proposal",
]
