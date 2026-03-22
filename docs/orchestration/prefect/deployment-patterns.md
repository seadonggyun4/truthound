---
title: Prefect Deployment Patterns
---

# Prefect Deployment Patterns

## Ephemeral Local Flow

Best for:

- local development
- smoke checks
- onboarding

Pattern:

- use task helpers directly
- omit saved blocks
- keep source access simple

## Shared Deployment With Saved Block

Best for:

- repeated runs in a work pool
- shared environment-specific configuration
- teams that want named reusable config objects

Pattern:

- save a `DataQualityBlock`
- load it inside the flow or deployment
- keep schedule and environment config in Prefect deployment settings

## Flow Factory Pattern

Best for:

- several datasets following the same quality workflow shape
- teams standardizing deployment construction

Pattern:

- use flow decorators or factory helpers
- keep only dataset-specific parameters at the call site

## Operational Advice

- choose one pattern per environment when possible
- avoid mixing ephemeral and persisted configuration invisibly
- document block ownership and naming
