# Provider Contract

Truthound AI standardizes providers behind a small structured-output contract.

## Core Types

- `ProviderConfig`
- `StructuredProviderRequest`
- `StructuredProviderResponse`
- `AIProvider`

The current default provider family is OpenAI, but the dashboard and other
consumers are intentionally coupled to the public Truthound AI contract rather
than to the OpenAI SDK directly.

## Resolution Rules

Provider resolution follows a narrow contract:

1. explicit provider instance
2. explicit provider config
3. default OpenAI-backed provider

Model resolution follows:

1. explicit model argument
2. provider config model name
3. environment fallback

If no model can be resolved, the request fails early as a configuration error.

## Error Taxonomy

Provider failures stay typed so downstream callers can distinguish them:

- `ProviderConfigurationError`
- `ProviderTransportError`
- `ProviderResponseError`

This matters for dashboards and CLIs because UI behavior should depend on
contracted failure classes, not provider-specific exception strings.
