---
title: Common Module
---

# Shared Runtime

The shared runtime is the contract that keeps every first-party host integration aligned. Airflow, Dagster, Prefect, Mage, Kestra, and dbt all wrap different host primitives, but they rely on the same runtime for engine resolution, source normalization, compatibility checks, and result serialization.

## What Lives Here

The shared runtime provides:

- engine creation and engine registry behavior
- host/runtime context normalization
- source resolution for DataFrames, paths, URIs, SQL, streams, and callables
- preflight and compatibility reports
- shared wire serialization
- operational helpers such as logging, retries, circuit breakers, health checks, metrics, rate limiting, and caching

## Start Here

- [Engine Resolution and Selection](engine-resolution-selection.md) explains how engine names become real runtimes.
- [Source Resolution](source-resolution.md) explains how host inputs are normalized.
- [Source Resolution Cookbook](source-resolution-cookbook.md) shows host-by-host patterns and concrete source shapes.
- [Preflight and Compatibility](preflight-compatibility.md) explains capability checks and failure reporting.
- [Failure Catalog](failure-catalog.md) maps common operational failures to the shared runtime layer that owns them.
- [Result Serialization](result-serialization.md) explains the shared result contract.
- [Observability and Resilience](observability-resilience.md) explains how retries, health checks, metrics, rate limiting, and cache fit together.

## Runtime Primitives

These types show up repeatedly across the platform packages:

| Primitive | Why It Matters |
|-----------|----------------|
| `EngineCreationRequest` | declares which engine to build and which runtime context applies |
| `PlatformRuntimeContext` | captures platform identity, host metadata, and zero-config policy |
| `ResolvedDataSource` | describes the normalized source and whether it requires a connection |
| `CompatibilityReport` | summarizes host, engine, and capability checks |
| `PreflightReport` | adds source and serializer readiness to the compatibility story |

## Runtime Responsibilities

The shared runtime is responsible for answering questions that should not vary by host:

- what engine should this name resolve to?
- does the selected engine support this operation?
- is the input a file path, SQL string, remote URI, DataFrame, or stream?
- can the source be executed without a host connection or profile?
- what is the canonical serialized result shape?
- which operational helpers are safe to reuse across hosts?

## Core Result Types

The core result family remains shared even when the host wraps it with metadata:

| Status | Description |
|--------|-------------|
| `PASSED` | Validation passed |
| `FAILED` | Validation failed |
| `WARNING` | Warning (threshold exceeded) |
| `SKIPPED` | Validation skipped |
| `ERROR` | Error occurred |

Shared operations commonly emit:

- `CheckResult`
- `ProfileResult`
- `LearnResult`
- drift and anomaly result types for advanced engine paths

The important rule is that hosts can wrap these results, but should not redefine their meaning.

## Operational Helpers

These modules are shared because the same production concerns show up in every host:

- [Logging](logging.md)
- [Retry](retry.md)
- [Circuit Breaker](circuit-breaker.md)
- [Health Check](health.md)
- [Metrics](metrics.md)
- [Rate Limiting](rate-limiter.md)
- [Caching](cache.md)

## How To Use This Section

Read the overview pages first, then the specific helper or primitive you need:

1. engine resolution and selection
2. source resolution and cookbook patterns
3. preflight and compatibility, including the failure catalog
4. serialization
5. observability and resilience
6. helper-level detail pages
