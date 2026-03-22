---
title: Kestra Task Runners and Retries
---

# Kestra Task Runners and Retries

Truthound does not override Kestra retry semantics. Instead, generated flows and script helpers expose the places where retries belong so teams can keep execution policy aligned with the rest of their Kestra estate.

## Who This Is For

- Kestra operators defining retry policy for validation flows
- platform teams using generated flow templates
- engineers separating transient infrastructure failures from data failures

## When To Use It

Use this page when:

- generated flows need retry configuration
- a script task should be rerun on transient issues
- the team is deciding what the task runner, not the validation engine, should own

## Prerequisites

- a Kestra deployment with task runners
- familiarity with generated flow config and `RetryConfig`
- a clear policy for transient failure vs deterministic quality failure

## Minimal Quickstart

Generated flows accept retry configuration:

```python
from truthound_kestra import RetryConfig, generate_check_flow

yaml_content = generate_check_flow(
    flow_id="users_quality",
    namespace="production",
    retry=RetryConfig(max_attempts=3),
)
```

## Production Pattern

Recommended policy:

| Failure Type | Retry? | Owner |
|-------------|--------|-------|
| transient runner/storage/network issue | yes, cautiously | Kestra retry policy |
| deterministic bad data | no | quality result and operator response |
| unsupported source or config error | no | fix configuration first |

Checklist:

- retry infrastructure failures, not broken data
- keep retry counts low for expensive quality tasks
- pair retries with clear logging so the final failure is diagnosable

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| retries never help | the failure is deterministic data quality | remove retries and surface the result directly |
| validation saturates the runner | retry counts are too high for expensive tasks | lower attempts and improve gating |
| root cause is hard to inspect | retries overwrite context without clear logs | emit metrics and structured outputs on each attempt |

## Related Pages

- [Kestra Overview](index.md)
- [Scripts and Flow Templates](scripts-templates.md)
- [Outputs and Metrics](outputs-metrics.md)
