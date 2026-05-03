"""Optional Truthound AI namespace.

This namespace intentionally stays additive so importing ``truthound`` keeps the
core hot path free from AI dependencies and workspace side effects.
"""

from __future__ import annotations

# Imports below intentionally happen after the optional-dependency gate.
# ruff: noqa: E402,I001

from truthound.ai._compat import ensure_ai_dependencies

ensure_ai_dependencies()

from truthound._ai_contract import TRUTHOUND_AI_COMPILER_VERSION, TRUTHOUND_AI_SCHEMA_VERSION
from truthound.ai.analyses import list_analyses, show_analysis
from truthound.ai.analysis import explain_run
from truthound.ai.dsl_coverage import (
    IntentCoverage,
    ValidationDSLCoverageMatrix,
    get_default_validation_dsl_coverage,
    load_validation_dsl_coverage_from_text,
)
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
    OpenAIPromptAcceptanceCanaryResult,
    OpenAIPromptCanaryCaseResult,
    OpenAISmokeMatrixEntry,
    OpenAISmokeMatrixItemResult,
    OpenAISmokeMatrixResult,
    OpenAIProviderSpec,
    OpenAISmokeResult,
    ProposalApplyResult,
    ProposalDecisionResult,
    ProposedCheckIntent,
    ProviderConfig,
    ProviderEvent,
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
from truthound.ai.normalization import (
    ClarificationRequest,
    ColumnResolver,
    IntentCanonicalizer,
    NormalizedIntentCandidate,
    NormalizedPrompt,
    PROMPT_NORMALIZATION_ENV,
    PromptNormalizationEvent,
    PromptNormalizationMode,
    PromptTextNormalizationResult,
    PromptNormalizer,
    UnresolvedPromptTerm,
    get_prompt_normalization_mode,
    normalize_prompt_text,
    normalize_prompt_text_with_audit,
    normalize_suite_prompt,
)
from truthound.ai.prompt_lexicon import (
    PromptLexicon,
    SUPPORTED_FORMAT_KINDS,
    get_default_prompt_lexicon,
    load_prompt_lexicon_from_text,
)
from truthound.ai.prompt_metrics import (
    get_ai_prompt_metrics_snapshot,
    reset_ai_prompt_metrics,
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
    get_provider_metrics_snapshot,
    reset_provider_metrics,
)
from truthound.ai.resolution import resolve_source_key
from truthound.ai.redaction import (
    RedactionViolation,
    RedactionViolationError,
    SummaryOnlyRedactor,
)
from truthound.ai.smoke import (
    parse_openai_smoke_model_matrix,
    run_openai_prompt_acceptance_canary,
    run_openai_explain_run_smoke,
    run_openai_smoke,
    run_openai_smoke_matrix,
)
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
    "ClarificationRequest",
    "ColumnResolver",
    "InputRef",
    "IntentCanonicalizer",
    "IntentCoverage",
    "NormalizedIntentCandidate",
    "NormalizedPrompt",
    "OpenAIExplainRunSmokeResult",
    "OpenAIPromptAcceptanceCanaryResult",
    "OpenAIPromptCanaryCaseResult",
    "OpenAISmokeMatrixEntry",
    "OpenAISmokeMatrixItemResult",
    "OpenAISmokeMatrixResult",
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
    "ProviderEvent",
    "ProviderResponseError",
    "ProviderTransportError",
    "PROMPT_NORMALIZATION_ENV",
    "PromptLexicon",
    "PromptNormalizationEvent",
    "PromptNormalizationMode",
    "PromptTextNormalizationResult",
    "PromptNormalizer",
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
    "SUPPORTED_FORMAT_KINDS",
    "UnresolvedPromptTerm",
    "ValidationSuiteConflict",
    "ValidationDSLCoverageMatrix",
    "ValidationSuiteDiffCounts",
    "ValidationSuiteDiffPreview",
    "ValidationSuiteSnapshot",
    "approve_proposal",
    "explain_run",
    "get_prompt_normalization_mode",
    "get_default_prompt_lexicon",
    "get_default_validation_dsl_coverage",
    "get_ai_prompt_metrics_snapshot",
    "get_provider_metrics_snapshot",
    "apply_proposal",
    "list_analyses",
    "list_proposal_approval_events",
    "list_proposals",
    "load_prompt_lexicon_from_text",
    "load_validation_dsl_coverage_from_text",
    "run_openai_explain_run_smoke",
    "run_openai_prompt_acceptance_canary",
    "run_openai_smoke",
    "reject_proposal",
    "reset_provider_metrics",
    "reset_ai_prompt_metrics",
    "resolve_source_key",
    "normalize_prompt_text",
    "normalize_prompt_text_with_audit",
    "normalize_suite_prompt",
    "parse_openai_smoke_model_matrix",
    "show_analysis",
    "show_proposal",
    "suggest_suite",
    "run_openai_smoke_matrix",
]
