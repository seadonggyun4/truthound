"""Proposal review and lifecycle helpers for ``truthound.ai``."""

from __future__ import annotations

from truthound.ai.lifecycle import (
    approve_proposal,
    apply_proposal,
    list_proposal_approval_events,
    reject_proposal,
)
from truthound.ai.models import (
    ActorRef,
    CompileStatus,
    ProposalApplyResult,
    ProposalDecisionResult,
    SuiteProposalArtifact,
)
from truthound.ai.store import AIArtifactStore
from truthound.context import TruthoundContext, get_context


def list_proposals(
    *,
    context: TruthoundContext | None = None,
    source_key: str | None = None,
    compile_status: str | CompileStatus | None = None,
    limit: int | None = None,
) -> list[SuiteProposalArtifact]:
    active_context = context or get_context()
    proposals = AIArtifactStore(active_context).list_proposals()
    if source_key is not None:
        proposals = [item for item in proposals if item.source_key == source_key]
    if compile_status is not None:
        expected = str(compile_status)
        proposals = [item for item in proposals if str(item.compile_status) == expected]
    if limit is not None:
        proposals = proposals[: max(0, int(limit))]
    return proposals


def show_proposal(
    proposal_id: str,
    *,
    context: TruthoundContext | None = None,
) -> SuiteProposalArtifact:
    active_context = context or get_context()
    return AIArtifactStore(active_context).read_proposal(proposal_id)


__all__ = [
    "ActorRef",
    "ProposalApplyResult",
    "ProposalDecisionResult",
    "approve_proposal",
    "apply_proposal",
    "list_proposals",
    "list_proposal_approval_events",
    "reject_proposal",
    "show_proposal",
]
