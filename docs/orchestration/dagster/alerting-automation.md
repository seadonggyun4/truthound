---
title: Dagster Alerting and Automation
---

# Dagster Alerting and Automation

Dagster can surface quality information through hooks, schedules, automation policies, and external operators. Truthound's Dagster package focuses on keeping the payload and SLA semantics stable while letting Dagster own the automation layer.

## Who This Is For

- platform teams routing quality failures into operational alerting
- operators deciding how much automation should happen in Dagster itself
- teams moving from manual reruns to automation-driven response

## When To Use It

Use this page when:

- you need SLA-aware quality operations in Dagster
- failures should trigger notifications or follow-up jobs
- you want to standardize response to warning vs failure severity

## Prerequisites

- `SLAResource` or another quality result evaluation path
- a Dagster deployment with jobs, schedules, or automation rules
- a defined notification or escalation policy

## Minimal Quickstart

Use the SLA surface as the policy boundary:

```python
from truthound_dagster import SLAConfig, SLAResource

sla = SLAResource(
    default_config=SLAConfig(max_failure_rate=0.05).to_dict()
)
```

For richer environments, pair the SLA resource with metrics or logging hooks:

```python
from truthound_dagster import CompositeSLAHook, LoggingSLAHook, MetricsSLAHook

hooks = CompositeSLAHook(hooks=[LoggingSLAHook(), MetricsSLAHook()])
```

## Production Pattern

Recommended automation split:

| Concern | Owner |
|--------|-------|
| validation execution | Truthound resource/op/asset helper |
| run orchestration | Dagster job or asset automation |
| alert routing | SLA hooks plus operator tooling |

Checklist:

- define whether warnings should notify or only log
- route high-noise checks to metrics before paging
- keep remediation jobs separate from primary validation jobs

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| too many alerts | low-value checks are treated as incidents | downgrade to warn or metrics-only |
| no operator context in alerts | automation reads only check pass/fail | include dataset and partition metadata |
| remediation loops | automated rerun policy restarts unstable quality jobs | cap retries and preserve manual review gates |

## Related Pages

- [Dagster Overview](index.md)
- [SLA and Hooks](sla-hooks.md)
- [Dagster Jobs and Schedules](jobs-schedules.md)
- [Enterprise Notifications](../enterprise/notifications.md)
