"""Provider implementations and resolution helpers for ``truthound.ai``."""

from __future__ import annotations

import json
import os
from typing import Any

from truthound.ai.models import (
    AIProvider,
    OpenAIProviderSpec,
    ProviderConfig,
    StructuredProviderRequest,
    StructuredProviderResponse,
)

DEFAULT_PROVIDER_NAME = "openai"
DEFAULT_MODEL_ENV = "TRUTHOUND_AI_MODEL"


class AIProviderError(RuntimeError):
    """Base error raised by AI provider operations."""


class ProviderConfigurationError(AIProviderError):
    """Raised when provider configuration is incomplete or unsupported."""


class ProviderTransportError(AIProviderError):
    """Raised when the provider request fails before a usable response is produced."""


class ProviderResponseError(AIProviderError):
    """Raised when the provider response is malformed or incomplete."""


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

        try:
            client = OpenAI(**client_kwargs)
            response = client.chat.completions.create(
                model=request.model_name,
                messages=[
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise ProviderTransportError(f"OpenAI request failed: {exc}") from exc

        try:
            choice = response.choices[0]
            message = choice.message
            output_text = self._coerce_content_text(message.content)
        except Exception as exc:
            raise ProviderResponseError(f"OpenAI response did not include message content: {exc}") from exc

        parsed_output: dict[str, Any] | list[Any] | str | None
        try:
            parsed_output = json.loads(output_text)
        except Exception:
            parsed_output = None

        usage = None
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            usage = {
                "prompt_tokens": int(getattr(raw_usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(raw_usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(raw_usage, "total_tokens", 0) or 0),
            }

        finish_reason = getattr(choice, "finish_reason", None)
        return StructuredProviderResponse(
            provider_name=self.provider_name,
            model_name=request.model_name,
            output_text=output_text,
            parsed_output=parsed_output,
            usage=usage,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
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
    "resolve_model_name",
    "resolve_provider",
]
