"""Optional Truthound AI namespace.

This namespace intentionally stays additive so importing ``truthound`` keeps the
core hot path free from AI dependencies and workspace side effects.
"""

from __future__ import annotations

from truthound.ai._compat import ensure_ai_dependencies

ensure_ai_dependencies()

from truthound._ai_contract import TRUTHOUND_AI_COMPILER_VERSION, TRUTHOUND_AI_SCHEMA_VERSION
from truthound.ai.analyses import list_analyses, show_analysis
from truthound.ai.analysis import explain_run
from truthound.ai.lifecycle import (
    ProposalLifecycleError,
    ProposalNotFoundError,
    ProposalStateError,
)
from truthound.ai.models import (
    AIProvider,
    ActorRef,
    ApprovalLogEvent,
    ApprovalStatus,
    ArtifactType,
    CompileStatus,
    CompiledProposalCheck,
    InputRef,
    OpenAIExplainRunSmokeResult,
    OpenAIProviderSpec,
    OpenAISmokeResult,
    ProposalApplyResult,
    ProposalDecisionResult,
    ProposedCheckIntent,
    ProviderConfig,
    RedactionMode,
    RedactionPolicy,
    RejectedProposalItem,
    RunAnalysisLLMResponse,
    RunAnalysisArtifact,
    SuiteCheckSnapshot,
    StructuredProviderRequest,
    StructuredProviderResponse,
    SuiteProposalArtifact,
    SuiteProposalLLMResponse,
    ValidationSuiteConflict,
    ValidationSuiteDiffCounts,
    ValidationSuiteDiffPreview,
    ValidationSuiteSnapshot,
)
from truthound.ai.proposals import (
    approve_proposal,
    apply_proposal,
    list_proposal_approval_events,
    list_proposals,
    reject_proposal,
    show_proposal,
)
from truthound.ai.providers import (
    AIProviderError,
    OpenAIStructuredProvider,
    ProviderConfigurationError,
    ProviderResponseError,
    ProviderTransportError,
)
from truthound.ai.redaction import (
    RedactionViolation,
    RedactionViolationError,
    SummaryOnlyRedactor,
)
from truthound.ai.smoke import run_openai_explain_run_smoke, run_openai_smoke
from truthound.ai.suggest import suggest_suite
from truthound.ai.store import AIArtifactStore

__all__ = [
    "AIArtifactStore",
    "AIProvider",
    "AIProviderError",
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
    "OpenAIStructuredProvider",
    "ProposalApplyResult",
    "ProposalDecisionResult",
    "ProposalLifecycleError",
    "ProposalNotFoundError",
    "ProposalStateError",
    "ProposedCheckIntent",
    "ProviderConfig",
    "ProviderConfigurationError",
    "ProviderResponseError",
    "ProviderTransportError",
    "RedactionMode",
    "RedactionPolicy",
    "RedactionViolation",
    "RedactionViolationError",
    "RejectedProposalItem",
    "RunAnalysisLLMResponse",
    "RunAnalysisArtifact",
    "SuiteCheckSnapshot",
    "StructuredProviderRequest",
    "StructuredProviderResponse",
    "SuiteProposalArtifact",
    "SuiteProposalLLMResponse",
    "SummaryOnlyRedactor",
    "TRUTHOUND_AI_COMPILER_VERSION",
    "TRUTHOUND_AI_SCHEMA_VERSION",
    "ValidationSuiteConflict",
    "ValidationSuiteDiffCounts",
    "ValidationSuiteDiffPreview",
    "ValidationSuiteSnapshot",
    "approve_proposal",
    "explain_run",
    "apply_proposal",
    "list_analyses",
    "list_proposal_approval_events",
    "list_proposals",
    "run_openai_explain_run_smoke",
    "run_openai_smoke",
    "reject_proposal",
    "show_analysis",
    "show_proposal",
    "suggest_suite",
]
