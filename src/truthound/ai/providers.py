"""Provider implementations and resolution helpers for ``truthound.ai``."""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Literal

from truthound.ai.models import (
    AIProvider,
    OpenAIProviderSpec,
    ProviderConfig,
    ProviderEvent,
    StructuredProviderRequest,
    StructuredProviderResponse,
)

DEFAULT_PROVIDER_NAME = "openai"
DEFAULT_MODEL_ENV = "TRUTHOUND_AI_MODEL"
ProviderResponseFormat = Literal["json_schema", "json_object", "unknown"]


_PROVIDER_METRIC_KEYS = (
    "ai_provider_requests_total",
    "ai_provider_json_schema_total",
    "ai_provider_json_mode_total",
    "ai_provider_fallback_total",
    "ai_provider_repair_total",
    "ai_provider_refusal_total",
)
_provider_metrics_lock = threading.Lock()
_provider_metrics: dict[str, int] = dict.fromkeys(_PROVIDER_METRIC_KEYS, 0)
_provider_reason_codes: dict[str, int] = {}


class AIProviderError(RuntimeError):
    """Base error raised by AI provider operations."""

    def __init__(
        self,
        message: str = "",
        *,
        reason_code: str | None = None,
        provider_events: list[ProviderEvent] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.provider_events = list(provider_events or [])


class ProviderConfigurationError(AIProviderError):
    """Raised when provider configuration is incomplete or unsupported."""


class ProviderTransportError(AIProviderError):
    """Raised when the provider request fails before a usable response is produced."""


class ProviderResponseError(AIProviderError):
    """Raised when the provider response is malformed or incomplete."""


def get_provider_metrics_snapshot() -> dict[str, Any]:
    with _provider_metrics_lock:
        return {
            **dict(_provider_metrics),
            "ai_provider_reason_codes": dict(sorted(_provider_reason_codes.items())),
        }


def reset_provider_metrics() -> None:
    with _provider_metrics_lock:
        for key in _PROVIDER_METRIC_KEYS:
            _provider_metrics[key] = 0
        _provider_reason_codes.clear()


def _record_provider_metrics(
    *,
    response_format_type: ProviderResponseFormat,
    used_json_mode_fallback: bool,
    repair_attempted: bool,
    provider_events: list[ProviderEvent],
) -> None:
    with _provider_metrics_lock:
        _provider_metrics["ai_provider_requests_total"] += 1
        if response_format_type == "json_schema":
            _provider_metrics["ai_provider_json_schema_total"] += 1
        elif response_format_type == "json_object":
            _provider_metrics["ai_provider_json_mode_total"] += 1
        if used_json_mode_fallback:
            _provider_metrics["ai_provider_fallback_total"] += 1
        if repair_attempted:
            _provider_metrics["ai_provider_repair_total"] += 1
        if any(event.reason_code == "provider_refusal" for event in provider_events):
            _provider_metrics["ai_provider_refusal_total"] += 1
        for event in provider_events:
            if event.reason_code:
                _provider_reason_codes[event.reason_code] = _provider_reason_codes.get(event.reason_code, 0) + 1


def _event(
    *,
    phase: str,
    response_format: ProviderResponseFormat,
    outcome: str,
    reason_code: str | None = None,
) -> ProviderEvent:
    return ProviderEvent(
        phase=phase,
        response_format=response_format,
        outcome=outcome,
        reason_code=reason_code,
    )


class OpenAIStructuredProvider:
    """OpenAI-backed structured provider using JSON-object responses."""

    provider_name = "openai"
    api_key_env = "OPENAI_API_KEY"
    supports_structured_outputs = True

    def __init__(self, config: ProviderConfig | None = None) -> None:
        self._config = config or ProviderConfig(
            provider_name="openai",
            api_key_env=self.api_key_env,
        )
        self.default_model_name = self._config.model_name
        self.api_key_env = self._config.api_key_env or self.api_key_env

    def generate_structured(
        self,
        request: StructuredProviderRequest,
    ) -> StructuredProviderResponse:
        if request.provider_name != self.provider_name:
            raise ProviderConfigurationError(
                f"OpenAIStructuredProvider cannot serve provider {request.provider_name!r}"
            )

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised by optional dependency tests
            raise ImportError(
                "OpenAI provider support requires the optional AI dependency set. "
                "Install it with: pip install truthound[ai]"
            ) from exc

        client_kwargs: dict[str, Any] = {
            "timeout": self._config.timeout_seconds,
        }
        if self.api_key_env:
            api_key = os.getenv(self.api_key_env)
            if api_key:
                client_kwargs["api_key"] = api_key
        if self._config.base_url:
            client_kwargs["base_url"] = self._config.base_url

        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]
        response_format = self._build_response_format(request)
        initial_response_format_type = self._response_format_type(response_format)
        active_response_format_type = initial_response_format_type
        used_json_mode_fallback = False
        repair_attempted = False
        repair_succeeded = False
        provider_events: list[ProviderEvent] = []
        try:
            client = OpenAI(**client_kwargs)
        except Exception as exc:
            provider_events.append(
                _event(
                    phase="client_init",
                    response_format=initial_response_format_type,
                    outcome="failed",
                    reason_code="client_initialization_failed",
                )
            )
            _record_provider_metrics(
                response_format_type=initial_response_format_type,
                used_json_mode_fallback=False,
                repair_attempted=False,
                provider_events=provider_events,
            )
            raise ProviderTransportError(
                f"OpenAI client initialization failed: {exc}",
                reason_code="client_initialization_failed",
                provider_events=provider_events,
            ) from exc

        active_response_format = response_format
        try:
            response = self._create_chat_completion(
                client=client,
                model=request.model_name,
                messages=messages,
                response_format=response_format,
            )
            provider_events.append(
                _event(
                    phase="request",
                    response_format=initial_response_format_type,
                    outcome="success",
                    reason_code="provider_request_succeeded",
                )
            )
        except Exception as exc:
            fallback_reason = self._json_mode_fallback_reason(exc, response_format)
            if fallback_reason:
                provider_events.append(
                    _event(
                        phase="request",
                        response_format=initial_response_format_type,
                        outcome="fallback",
                        reason_code=fallback_reason,
                    )
                )
                try:
                    used_json_mode_fallback = True
                    active_response_format = {"type": "json_object"}
                    active_response_format_type = "json_object"
                    response = self._create_chat_completion(
                        client=client,
                        model=request.model_name,
                        messages=messages,
                        response_format=active_response_format,
                    )
                    provider_events.append(
                        _event(
                            phase="request",
                            response_format="json_object",
                            outcome="success",
                            reason_code="json_mode_fallback_succeeded",
                        )
                    )
                except Exception as fallback_exc:
                    provider_events.append(
                        _event(
                            phase="request",
                            response_format="json_object",
                            outcome="failed",
                            reason_code="json_mode_fallback_failed",
                        )
                    )
                    _record_provider_metrics(
                        response_format_type=active_response_format_type,
                        used_json_mode_fallback=used_json_mode_fallback,
                        repair_attempted=repair_attempted,
                        provider_events=provider_events,
                    )
                    raise ProviderTransportError(
                        f"OpenAI request failed: {fallback_exc}",
                        reason_code="json_mode_fallback_failed",
                        provider_events=provider_events,
                    ) from fallback_exc
            else:
                provider_events.append(
                    _event(
                        phase="request",
                        response_format=initial_response_format_type,
                        outcome="failed",
                        reason_code="provider_transport_error",
                    )
                )
                _record_provider_metrics(
                    response_format_type=initial_response_format_type,
                    used_json_mode_fallback=False,
                    repair_attempted=False,
                    provider_events=provider_events,
                )
                raise ProviderTransportError(
                    f"OpenAI request failed: {exc}",
                    reason_code="provider_transport_error",
                    provider_events=provider_events,
                ) from exc

        output_text, response, choice = self._extract_output_text(
            response,
            response_format_type=active_response_format_type,
            provider_events=provider_events,
        )
        parsed_output: dict[str, Any] | list[Any] | str | None
        try:
            parsed_output = json.loads(output_text)
            provider_events.append(
                _event(
                    phase="parse",
                    response_format=active_response_format_type,
                    outcome="success",
                    reason_code="json_parse_succeeded",
                )
            )
        except Exception as exc:
            provider_events.append(
                _event(
                    phase="parse",
                    response_format=active_response_format_type,
                    outcome="failed",
                    reason_code="invalid_json",
                )
            )
            if active_response_format.get("type") == "json_object":
                repair_attempted = True
                try:
                    output_text, response, choice = self._repair_json_response(
                        client=client,
                        model=request.model_name,
                        messages=messages,
                        invalid_output=output_text,
                        provider_events=provider_events,
                    )
                except ProviderTransportError as repair_exc:
                    _record_provider_metrics(
                        response_format_type=active_response_format_type,
                        used_json_mode_fallback=used_json_mode_fallback,
                        repair_attempted=repair_attempted,
                        provider_events=provider_events,
                    )
                    raise repair_exc
                try:
                    parsed_output = json.loads(output_text)
                    repair_succeeded = True
                    provider_events.append(
                        _event(
                            phase="repair",
                            response_format="json_object",
                            outcome="success",
                            reason_code="repair_json_parse_succeeded",
                        )
                    )
                except Exception:
                    parsed_output = None
                    provider_events.append(
                        _event(
                            phase="repair",
                            response_format="json_object",
                            outcome="failed",
                            reason_code="repair_invalid_json",
                        )
                    )
            else:
                _record_provider_metrics(
                    response_format_type=active_response_format_type,
                    used_json_mode_fallback=used_json_mode_fallback,
                    repair_attempted=repair_attempted,
                    provider_events=provider_events,
                )
                raise ProviderResponseError(
                    f"OpenAI response was not valid JSON: {exc}",
                    reason_code="invalid_json_schema_output",
                    provider_events=provider_events,
                ) from exc

        usage = None
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            usage = {
                "prompt_tokens": int(getattr(raw_usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(raw_usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(raw_usage, "total_tokens", 0) or 0),
            }

        finish_reason = getattr(choice, "finish_reason", None)
        provider_response = StructuredProviderResponse(
            provider_name=self.provider_name,
            model_name=request.model_name,
            output_text=output_text,
            parsed_output=parsed_output,
            usage=usage,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            response_format_type=active_response_format_type,
            used_json_mode_fallback=used_json_mode_fallback,
            repair_attempted=repair_attempted,
            repair_succeeded=repair_succeeded,
            provider_events=provider_events,
        )
        _record_provider_metrics(
            response_format_type=provider_response.response_format_type,
            used_json_mode_fallback=provider_response.used_json_mode_fallback,
            repair_attempted=provider_response.repair_attempted,
            provider_events=provider_response.provider_events,
        )
        return provider_response

    def _create_chat_completion(
        self,
        *,
        client: Any,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any],
    ) -> Any:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format,
        )

    def _build_response_format(self, request: StructuredProviderRequest) -> dict[str, Any]:
        if request.response_model is None or not self.supports_structured_outputs:
            return {"type": "json_object"}
        schema = request.response_model.model_json_schema()
        return {
            "type": "json_schema",
            "json_schema": {
                "name": request.response_format_name or request.response_model.__name__,
                "strict": True,
                "schema": schema,
            },
        }

    def _response_format_type(self, response_format: dict[str, Any]) -> ProviderResponseFormat:
        response_format_type = response_format.get("type")
        if response_format_type == "json_schema":
            return "json_schema"
        if response_format_type == "json_object":
            return "json_object"
        return "unknown"

    def _json_mode_fallback_reason(
        self,
        exc: Exception,
        response_format: dict[str, Any],
    ) -> str | None:
        if response_format.get("type") != "json_schema":
            return None
        message = str(exc).lower()
        if any(marker in message for marker in ("auth", "api key", "quota", "rate limit", "timeout", "network")):
            return None
        if "unsupported" in message:
            return "schema_unsupported"
        if "response_format" in message or "json_schema" in message or "strict" in message:
            return "schema_unsupported"
        if "schema" in message or "invalid" in message:
            return "schema_invalid"
        return None

    def _extract_output_text(
        self,
        response: Any,
        *,
        response_format_type: ProviderResponseFormat,
        provider_events: list[ProviderEvent],
    ) -> tuple[str, Any, Any]:
        try:
            choice = response.choices[0]
            message = choice.message
            refusal = self._extract_refusal(message)
            if refusal:
                provider_events.append(
                    _event(
                        phase="parse",
                        response_format=response_format_type,
                        outcome="failed",
                        reason_code="provider_refusal",
                    )
                )
                _record_provider_metrics(
                    response_format_type=response_format_type,
                    used_json_mode_fallback=any(
                        event.reason_code == "json_mode_fallback_succeeded"
                        for event in provider_events
                    ),
                    repair_attempted=any(event.phase == "repair" for event in provider_events),
                    provider_events=provider_events,
                )
                raise ProviderResponseError(
                    "OpenAI response was refused by the provider",
                    reason_code="provider_refusal",
                    provider_events=provider_events,
                )
            output_text = self._coerce_content_text(message.content)
        except Exception as exc:
            if isinstance(exc, ProviderResponseError):
                raise
            raise ProviderResponseError(f"OpenAI response did not include message content: {exc}") from exc
        return output_text, response, choice

    def _extract_refusal(self, message: Any) -> str | None:
        refusal = getattr(message, "refusal", None)
        if isinstance(refusal, str) and refusal.strip():
            return refusal
        if isinstance(message, dict):
            refusal = message.get("refusal")
            if isinstance(refusal, str) and refusal.strip():
                return refusal
        return None

    def _repair_json_response(
        self,
        *,
        client: Any,
        model: str,
        messages: list[dict[str, str]],
        invalid_output: str,
        provider_events: list[ProviderEvent],
    ) -> tuple[str, Any, Any]:
        repair_messages = [
            *messages,
            {
                "role": "user",
                "content": (
                    "The previous response was not valid JSON. Return one valid JSON object "
                    "for the same schema, with no markdown fences or extra text. "
                    f"Invalid response preview: {invalid_output[:400]}"
                ),
            },
        ]
        try:
            response = self._create_chat_completion(
                client=client,
                model=model,
                messages=repair_messages,
                response_format={"type": "json_object"},
            )
            provider_events.append(
                _event(
                    phase="repair",
                    response_format="json_object",
                    outcome="response_received",
                    reason_code="repair_request_succeeded",
                )
            )
        except Exception as exc:
            provider_events.append(
                _event(
                    phase="repair",
                    response_format="json_object",
                    outcome="failed",
                    reason_code="repair_request_failed",
                )
            )
            raise ProviderTransportError(
                f"OpenAI JSON repair request failed: {exc}",
                reason_code="repair_request_failed",
                provider_events=provider_events,
            ) from exc
        return self._extract_output_text(
            response,
            response_format_type="json_object",
            provider_events=provider_events,
        )

    def _coerce_content_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                text_value = getattr(item, "text", None)
                if isinstance(text_value, str):
                    parts.append(text_value)
                    continue
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if parts:
                return "".join(parts)
        raise ProviderResponseError("OpenAI message content was empty or unsupported")


def resolve_provider(provider: AIProvider | ProviderConfig | None) -> AIProvider:
    if provider is None:
        return OpenAIStructuredProvider(
            ProviderConfig(
                provider_name=DEFAULT_PROVIDER_NAME,
                api_key_env=OpenAIProviderSpec().api_key_env,
            )
        )

    if isinstance(provider, ProviderConfig):
        if provider.provider_name != DEFAULT_PROVIDER_NAME:
            raise ProviderConfigurationError(
                f"Unsupported AI provider {provider.provider_name!r}. Phase 1 supports only 'openai'."
            )
        return OpenAIStructuredProvider(provider)

    if hasattr(provider, "generate_structured") and hasattr(provider, "provider_name"):
        return provider

    raise ProviderConfigurationError(
        "provider must be an AIProvider instance, ProviderConfig, or None"
    )


def resolve_model_name(
    *,
    model: str | None,
    provider: AIProvider | ProviderConfig | None,
    resolved_provider: AIProvider,
) -> str:
    if model:
        return model

    if isinstance(provider, ProviderConfig) and provider.model_name:
        return provider.model_name

    provider_default = getattr(resolved_provider, "default_model_name", None) or getattr(
        resolved_provider,
        "model_name",
        None,
    )
    if provider_default:
        return str(provider_default)

    env_model = os.getenv(DEFAULT_MODEL_ENV)
    if env_model:
        return env_model

    raise ProviderConfigurationError(
        "No AI model configured. Pass model=..., set ProviderConfig.model_name, "
        f"or export {DEFAULT_MODEL_ENV}."
    )


__all__ = [
    "AIProvider",
    "AIProviderError",
    "DEFAULT_MODEL_ENV",
    "DEFAULT_PROVIDER_NAME",
    "OpenAIProviderSpec",
    "OpenAIStructuredProvider",
    "ProviderConfig",
    "ProviderConfigurationError",
    "ProviderResponseError",
    "ProviderTransportError",
    "StructuredProviderRequest",
    "StructuredProviderResponse",
    "get_provider_metrics_snapshot",
    "reset_provider_metrics",
    "resolve_model_name",
    "resolve_provider",
]
