---
title: Platform Adoption Playbook
---

# Platform Adoption Playbook

This playbook is the operator-facing bridge between "the adapter works" and "the adapter is now a supported part of the platform". It is intentionally cross-platform and sits above the host-specific pages.

## Who This Is For

- platform teams adopting Truthound across more than one orchestration host
- operators standardizing rollout order and ownership boundaries
- engineering leads choosing how to phase adapter adoption by workload type

## When To Use It

Use this page when:

- one team has succeeded locally and now others want the same pattern
- multiple hosts such as Airflow, Prefect, and dbt are in scope at once
- you need a rollout plan that stays aligned with support matrix and CI coverage

## Prerequisites

- a supported Truthound 3.x release
- at least one validated host/platform pairing from the compatibility matrix
- a documented owner for rules, secrets, notifications, and incident response

## Minimal Quickstart

Recommended adoption order:

1. validate one representative dataset on the chosen host
2. document the source shape, rule set, and expected result contract
3. add CI coverage for the host-specific path
4. wire alerting and operational ownership
5. roll the pattern out to adjacent datasets

## Production Pattern

Use this decision table across the adapter line:

| Question | Recommended Decision |
|---------|----------------------|
| where should quality run? | pick the host that already owns orchestration for that workload |
| when is zero-config enough? | only for first runs and clearly local sources |
| when should secrets become explicit? | before production SQL or remote-source execution |
| what is the release gate? | supported tuple + preflight pass + documented result consumer |

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| each team adopts a different pattern | no shared playbook exists | publish one blessed host-specific path per workload type |
| quality checks exist but no one owns incidents | rollout focused on code, not operations | define owner and escalation before general adoption |
| support matrix is ignored during rollout | adoption happened from examples only | make compatibility review part of promotion criteria |

## Related Pages

- [Choose a Platform](choose-a-platform.md)
- [Compatibility](compatibility.md)
- [Production Readiness](production-readiness.md)
- [Enterprise and Operations](enterprise/index.md)
