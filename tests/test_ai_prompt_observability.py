from __future__ import annotations

import json
import sys
import types
from typing import Any

import polars as pl
import pytest

from truthound.context import TruthoundContext

pytest.importorskip("pydantic")

pytestmark = pytest.mark.contract


class _Provider:
    provider_name = "fake-openai"
    api_key_env = None
    supports_structured_outputs = True
    default_model_name = "gpt-fake"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.requests: list[Any] = []

    def generate_structured(self, request: Any):
        from truthound.ai import StructuredProviderResponse

        self.requests.append(request)
        return StructuredProviderResponse(
            provider_name=self.provider_name,
            model_name=request.model_name,
            output_text=json.dumps(self.payload, ensure_ascii=False),
            parsed_output=self.payload,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            finish_reason="stop",
        )


def _proposal_payload() -> dict[str, Any]:
    return {
        "summary": "Email completeness proposal.",
        "rationale": "Email should be present.",
        "proposed_checks": [
            {
                "intent": "not_null",
                "columns": ["email"],
                "params": {},
                "rationale": "Email is required.",
            }
        ],
        "risks": [],
        "rejected_requests": [],
    }


def test_prompt_metrics_record_normalization_and_compilation_without_raw_prompt(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from truthound.ai import (
        get_ai_prompt_metrics_snapshot,
        reset_ai_prompt_metrics,
        suggest_suite,
    )

    monkeypatch.setenv("TRUTHOUND_AI_PROMPT_NORMALIZATION", "enforce")
    reset_ai_prompt_metrics()

    artifact = suggest_suite(
        prompt="이메일은 비어 있으면 안 됩니다",
        data=pl.DataFrame({"email": ["a@example.com", None]}),
        context=TruthoundContext(tmp_path),
        provider=_Provider(_proposal_payload()),
    )

    snapshot = get_ai_prompt_metrics_snapshot()
    serialized = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)

    assert str(artifact.compile_status) == "ready"
    assert snapshot["ai_prompt_normalization_mode"] == "enforce"
    assert snapshot["ai_prompt_normalization_requests_total"] == 1
    assert snapshot["ai_prompt_normalization_actionable_total"] == 1
    assert snapshot["ai_prompt_normalization_candidate_total"] >= 1
    assert snapshot["ai_prompt_normalization_modes"]["enforce"] == 1
    assert snapshot["ai_prompt_normalization_languages"]["ko"] == 1
    assert snapshot["ai_prompt_candidate_intents"]["not_null"] >= 1
    assert snapshot["ai_proposal_compile_statuses"]["ready"] == 1
    assert "이메일은 비어 있으면 안 됩니다" not in serialized
    assert "a@example.com" not in serialized


def test_prompt_metrics_record_clarification_and_rejection_reason(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from truthound.ai import (
        get_ai_prompt_metrics_snapshot,
        reset_ai_prompt_metrics,
        suggest_suite,
    )

    monkeypatch.setenv("TRUTHOUND_AI_PROMPT_NORMALIZATION", "enforce")
    reset_ai_prompt_metrics()
    provider = _Provider(_proposal_payload())

    artifact = suggest_suite(
        prompt="데이터를 알아서 잘 검증해줘",
        data=pl.DataFrame({"email": ["a@example.com", None]}),
        context=TruthoundContext(tmp_path),
        provider=provider,
    )

    snapshot = get_ai_prompt_metrics_snapshot()

    assert provider.requests == []
    assert str(artifact.compile_status) == "rejected"
    assert snapshot["ai_prompt_normalization_clarification_total"] == 1
    assert snapshot["ai_prompt_clarification_reasons"]["prompt_too_ambiguous"] == 1
    assert snapshot["ai_proposal_compile_statuses"]["rejected"] == 1
    assert snapshot["ai_proposal_rejection_sources"]["normalizer"] == 1
    assert snapshot["ai_proposal_rejection_reasons"]["prompt_too_ambiguous"] == 1


def test_invalid_prompt_normalization_mode_falls_back_and_is_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from truthound.ai import (
        PromptNormalizationMode,
        get_ai_prompt_metrics_snapshot,
        get_prompt_normalization_mode,
        reset_ai_prompt_metrics,
    )

    reset_ai_prompt_metrics()
    monkeypatch.setenv("TRUTHOUND_AI_PROMPT_NORMALIZATION", "broken")

    assert get_prompt_normalization_mode() == PromptNormalizationMode.ENFORCE
    snapshot = get_ai_prompt_metrics_snapshot()
    assert snapshot["ai_prompt_normalization_mode"] == "enforce"
    assert snapshot["ai_prompt_normalization_invalid_mode_total"] == 1
    assert snapshot["ai_prompt_normalization_invalid_modes"]["broken"] == 1


def test_provider_metrics_include_reason_code_breakdown(monkeypatch: pytest.MonkeyPatch) -> None:
    from truthound.ai import (
        ProviderConfig,
        StructuredProviderRequest,
        SuiteProposalLLMResponse,
    )
    from truthound.ai.providers import (
        OpenAIStructuredProvider,
        get_provider_metrics_snapshot,
        reset_provider_metrics,
    )

    def fake_create(**kwargs):
        response_format = kwargs["response_format"]
        if response_format["type"] == "json_schema":
            raise ValueError("unsupported response_format json_schema")
        message = types.SimpleNamespace(
            content='{"summary":"ok","rationale":"ok","proposed_checks":[],"risks":[],"rejected_requests":[]}'
        )
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        return types.SimpleNamespace(choices=[choice], usage=None)

    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=fake_create))

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAIClient))
    reset_provider_metrics()

    provider = OpenAIStructuredProvider(ProviderConfig(provider_name="openai", model_name="gpt-test"))
    provider.generate_structured(
        StructuredProviderRequest(
            provider_name="openai",
            model_name="gpt-test",
            system_prompt="Return JSON.",
            user_prompt="Return JSON.",
            response_format_name="suite_proposal",
            response_model=SuiteProposalLLMResponse,
        )
    )

    snapshot = get_provider_metrics_snapshot()
    assert snapshot["ai_provider_fallback_total"] == 1
    assert snapshot["ai_provider_reason_codes"]["schema_unsupported"] == 1
    assert snapshot["ai_provider_reason_codes"]["json_mode_fallback_succeeded"] == 1
    assert snapshot["ai_provider_reason_codes"]["json_parse_succeeded"] == 1
