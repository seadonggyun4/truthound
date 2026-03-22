---
title: Multi-Tenant
---

# Multi-Tenant

Truthound's multi-tenant module is for shared deployments where one orchestration
service, worker fleet, or internal platform must safely serve multiple tenants.

## Main Components

- `TenantRegistry`
- `TenantContextManager`
- isolation strategies
- middleware and context propagation helpers

## When To Use It

Use multi-tenant support when:

- one deployment serves multiple customers or business units
- quotas, isolation, or audit boundaries matter
- the same orchestration stack needs tenant-aware secrets or notifications

## Recommended Operating Model

- maintain tenant definitions centrally through the registry
- enter tenant context explicitly around work that should be isolated
- keep isolation strategy aligned with the real risk model of the platform

## Isolation Levels

The module supports shared, logical, and stronger isolation strategies. The right
choice depends on whether you need convenience, logical separation, or tighter
environmental boundaries.

## Practical Guidance

- treat tenant context as required execution metadata
- avoid hidden global state outside the tenant context helpers
- test cross-tenant access rules as part of platform verification, not only at runtime

## Related Pages

- [Secrets](secrets.md)
- [CI/CD and Production Rollout](ci-cd-production.md)
