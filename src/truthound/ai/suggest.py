"""Phase 1 prompt-to-suite suggestion pipeline."""

from __future__ import annotations

import hashlib
from typing import Any

from truthound.ai.compiler import ProposalCompiler
from truthound.ai.context import ContextBundleBuilder
from truthound.ai.models import (
    AIProvider,
    ProviderConfig,
    StructuredProviderRequest,
    SuiteProposalArtifact,
    SuiteProposalLLMResponse,
)
from truthound.ai.providers import resolve_model_name, resolve_provider
from truthound.ai.store import AIArtifactStore
from truthound.context import TruthoundContext, get_context


def suggest_suite(
    prompt: str,
    data: Any = None,
    source: Any = None,
    context: TruthoundContext | None = None,
    provider: AIProvider | ProviderConfig | None = None,
    model: str | None = None,
    sample_size: int = 1000,
    redact: str = "summary_only",
) -> SuiteProposalArtifact:
    """Compile a natural-language request into a persisted suite proposal artifact."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("suggest_suite requires a non-empty prompt")
    if redact != "summary_only":
        raise ValueError("Phase 1 supports only redact='summary_only'")

    active_context = context or get_context()
    bundle = ContextBundleBuilder(summary_budget=sample_size).build(
        data=data,
        source=source,
        context=active_context,
    )
    resolved_provider = resolve_provider(provider)
    resolved_model_name = resolve_model_name(
        model=model,
        provider=provider,
        resolved_provider=resolved_provider,
    )

    system_prompt = bundle.build_system_prompt()
    user_prompt = bundle.build_user_prompt(prompt.strip())
    prompt_hash = _hash_prompt(system_prompt, user_prompt)
    provider_request = StructuredProviderRequest(
        provider_name=resolved_provider.provider_name,
        model_name=resolved_model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format_name="suite_proposal",
        response_model=SuiteProposalLLMResponse,
        input_refs=list(bundle.input_refs),
        metadata={
            "source_key": bundle.source_key,
            "summary_budget": bundle.summary_budget,
            "artifact_type": "suite_proposal",
        },
    )

    response = resolved_provider.generate_structured(provider_request)
    compiler = ProposalCompiler()

    try:
        if isinstance(response.parsed_output, dict):
            llm_response = SuiteProposalLLMResponse.model_validate(response.parsed_output)
        else:
            llm_response = SuiteProposalLLMResponse.model_validate_json(response.output_text)
        artifact = compiler.compile_artifact(
            response=llm_response,
            context_bundle=bundle,
            model_provider=response.provider_name,
            model_name=response.model_name,
            prompt_hash=prompt_hash,
        )
    except Exception:
        artifact = compiler.build_rejected_artifact(
            context_bundle=bundle,
            model_provider=response.provider_name,
            model_name=response.model_name,
            prompt_hash=prompt_hash,
            error_code="provider_output_validation_failed",
        )

    AIArtifactStore(active_context).write_proposal(artifact)
    return artifact


def _hash_prompt(system_prompt: str, user_prompt: str) -> str:
    digest = hashlib.sha256()
    digest.update(system_prompt.encode("utf-8"))
    digest.update(b"\n")
    digest.update(user_prompt.encode("utf-8"))
    return digest.hexdigest()[:16]


__all__ = [
    "suggest_suite",
]
