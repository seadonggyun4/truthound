# ADR 002: Plugin Platform

## Status

Accepted

## Context

Truthound previously maintained more than one plugin lifecycle path, which caused version drift and duplicated orchestration concerns.

## Decision

Use `PluginManager` as the single lifecycle runtime and model enterprise-oriented behavior as optional capabilities. `EnterprisePluginManager` becomes an async facade over that shared runtime.

## Consequences

- one source of truth for lifecycle behavior
- version compatibility logic uses package metadata consistently
- plugins can target stable manager ports instead of internal manager variants
