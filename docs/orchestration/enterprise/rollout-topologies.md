---
title: Rollout Topologies
---

# Rollout Topologies

Truthound orchestration deployments can stay simple or become highly segmented
depending on whether one team, many teams, or many tenants share the same host.

## Who This Is For

- platform operators planning multi-environment rollout
- teams deciding how to separate development, staging, and production
- organizations introducing tenant isolation or governed secrets

## When To Use It

Use this page before the first organization-wide rollout or whenever a local
single-team setup grows into a shared platform surface.

## Common Topologies

| Topology | Best Fit | Notes |
|----------|----------|-------|
| single team / single host | early rollout | lowest overhead, fastest feedback |
| shared host / multiple teams | central platform teams | requires documented ownership of configs and alerts |
| multi-tenant shared runtime | regulated or heavily shared environments | pair with [Multi-Tenant](multi-tenant.md) and [Secrets](secrets.md) |
| CI-gated package promotion | larger organizations | promotes adapter and docs changes through explicit release gates |

## Production Pattern

- keep environment promotion explicit
- keep host-native secrets and notification paths separate per environment
- treat docs snapshot sync as part of the release flow, not an afterthought

## Related Pages

- [CI/CD and Production Rollout](ci-cd-production.md)
- [Multi-Tenant](multi-tenant.md)
- [Rollback Playbooks](rollback-playbooks.md)
