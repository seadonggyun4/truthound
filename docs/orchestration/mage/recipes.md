---
title: Mage Recipes
---

# Mage Recipes

This page collects a few high-signal patterns that map well to the current package
surface.

## Validate A Dataset Before A Load

```python
from truthound_mage import CheckBlockConfig, CheckTransformer


def transform(df, *args, **kwargs):
    result = CheckTransformer(
        config=CheckBlockConfig(
            rules=[
                {"column": "id", "check": "not_null"},
                {"column": "email", "check": "email_format"},
            ]
        )
    ).execute(df)
    return result.result_dict
```

Use this pattern when a pipeline should fail before data is written downstream.

## Generate A Profile For Operator Review

Use `ProfileTransformer` for baselining, onboarding, and change review.

This is especially useful for pipelines that ingest third-party feeds whose quality
shape changes over time.

## Gate Downstream Work With A Sensor

Use `QualityGateSensor` or `DataQualitySensor` when downstream blocks should run only
after a check result meets pass-rate or failure-rate thresholds.

This keeps quality gate semantics explicit instead of scattering them across block code.

## Route Hard Failures And Soft Warnings Differently

Combine:

- strict `CheckTransformer` configs for blocking datasets
- condition or sensor blocks for soft rollout paths
- SLA hooks for operator visibility

## Keep Shared Config In One Place

Create one helper module that returns prebuilt `CheckBlockConfig`,
`ProfileBlockConfig`, and `SensorBlockConfig` objects. This prevents configuration
drift across pipelines.

## Related Pages

- [Project Layout](project-layout.md)
- [`io_config.yaml`](io-config.md)
- [Troubleshooting](troubleshooting.md)
