---
title: Observability And Resilience
---

# Observability And Resilience

The shared runtime also owns the operational helpers that keep long-lived workflow integrations predictable: logging, retries, circuit breakers, health checks, metrics, rate limiting, caching, and structured observability events.

## Why These Helpers Are Shared

All host platforms hit the same production problems:

- flaky upstream data access
- expensive validation on large datasets
- intermittent secret or connection failures
- repeated checks that should not overwhelm shared systems
- workflows that need auditable execution metadata

Keeping those helpers shared prevents each host from growing its own incompatible operational policy layer.

## Core Areas

| Area | Shared Responsibility |
|------|------------------------|
| Logging | structured, masked logging with consistent context keys |
| Retry | bounded retry policies and backoff strategies |
| Circuit Breaker | protecting repeated failures from cascading into the host |
| Health | surface-level health checks for runtime readiness |
| Metrics | counters, gauges, histograms, and platform-neutral metric emission |
| Rate Limiting | keeping data access or validation throughput bounded |
| Caching | avoiding repeated expensive setup or lookup operations |
| Observability Events | lifecycle events for execution and lineage |

## OpenLineage And Shared Observability

The runtime exposes structured observability config instead of making each host implement its own lineage emitter from scratch. That gives you:

- a consistent backend model
- shared producer metadata
- execution context attached to the same runtime event types
- fewer platform-specific observability gaps

## Operational Defaults

The defaults remain conservative:

- zero-config should stay easy to debug
- retries should not hide systemic misconfiguration
- circuit breakers should fail clearly when a dependency is unhealthy
- logging should help operators without leaking sensitive data

## When To Make Behavior Explicit

Move from defaults to explicit policies when:

- workflows are shared across teams
- external rate limits matter
- downstream alerting depends on stable thresholds
- you need deterministic escalation or SLA enforcement
- the host UI is not enough and you need external observability backends

## Detailed Helper Pages

- [Logging](logging.md)
- [Retry](retry.md)
- [Circuit Breaker](circuit-breaker.md)
- [Health Check](health.md)
- [Metrics](metrics.md)
- [Rate Limiting](rate-limiter.md)
- [Caching](cache.md)
