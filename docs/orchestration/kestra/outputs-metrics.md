---
title: Kestra Outputs and Metrics
---

# Kestra Outputs and Metrics

Kestra is output-driven: flows hand off data through script outputs, files, and metrics. Truthound's Kestra package embraces that by exposing `KestraOutputHandler`, `send_check_result`, and related helpers rather than inventing a non-Kestra reporting surface.

## Who This Is For

- Kestra operators surfacing quality results in flow outputs
- platform teams wiring metrics and summaries into downstream systems
- engineers debugging what a script task actually emitted

## When To Use It

Use this page when:

- a Python script task should expose a structured quality result
- a downstream task or dashboard needs the output
- metrics and summaries must be emitted in a Kestra-friendly form

## Prerequisites

- `truthound-orchestration[kestra]` installed
- a Kestra flow using Truthound script helpers
- familiarity with the team's output and metrics conventions

## Minimal Quickstart

Use the script entry point for the check:

```python
from truthound_kestra import check_quality_script

result = check_quality_script(
    data_uri="s3://warehouse/curated/users.parquet",
    rules=[{"column": "id", "check": "not_null"}],
)
```

Then emit the result through the Kestra output helper:

```python
from truthound_kestra import send_check_result

send_check_result(result)
```

## Production Pattern

Use a layered output model:

| Layer | Purpose |
|------|---------|
| shared Truthound result | canonical status, counts, and details |
| Kestra output handler | task output transport and formatting |
| metrics and notifications | operational signals for the wider platform |

Recommended practices:

- emit one stable output contract per task
- keep alerting and dashboard logic downstream from the shared result
- distinguish raw detail from summarized metrics

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| downstream task cannot parse the output | result was formatted ad hoc | standardize on `send_check_result` or `KestraOutputHandler` |
| flow metrics exist but lack quality detail | only task success/failure is recorded | publish summarized counts from the result payload |
| different flows emit different result shapes | multiple custom handlers were introduced | converge on one shared output pattern |

## Related Pages

- [Kestra Overview](index.md)
- [Scripts and Flow Templates](scripts-templates.md)
- [Input and Output Files](input-output-files.md)
