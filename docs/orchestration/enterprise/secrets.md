---
title: Secrets
---

# Secrets

The enterprise secrets module provides pluggable secret providers, wrappers, caching,
encryption helpers, rotation support, and tenant-aware secret isolation.

## Main Components

- `SecretProviderRegistry`
- provider backends for Vault, AWS, GCP, Azure, env, file, and memory
- `SecretCache` and `TieredSecretCache`
- encryption helpers
- `SecretRotationManager`

## When To Use It

Use the secrets module when:

- orchestration workers need centrally managed credentials
- you want providers to be swappable across environments
- the same deployment needs tenant-aware secret boundaries

## Production Guidance

- prefer external secret managers over hardcoded config files
- use environment or provider-based auth for local and CI execution
- add caching only when latency or rate limits justify it
- rotate secrets deliberately and audit read/write events

## Rollout Pattern

- start with one provider per environment
- wrap providers for validation, caching, or encryption only when needed
- document the fallback and failure policy clearly so operators know whether missing
  secrets should fail fast or degrade gracefully

## Related Pages

- [Notifications](notifications.md)
- [CI/CD and Production Rollout](ci-cd-production.md)
