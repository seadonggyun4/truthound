---
title: Engine Selection Guide
---

# Engine Selection Guide

Choosing an engine is less about brand preference and more about matching the
validation model to the host, data shape, and operational goal.

## Who This Is For

- teams standardizing one engine per platform
- operators reviewing an engine change request
- contributors designing adapter defaults

## When To Use It

Use this guide when you need a default engine recommendation for a host or when
you are deciding whether to keep Truthound as the primary path.

## Minimal Decision Table

| If your main need is... | Start With... |
|-------------------------|---------------|
| the full first-party orchestration feature line | Truthound |
| dataframe schema contracts in Python workflows | Pandera |
| expectations-style validation carried over from existing GE usage | Great Expectations |
| stream, drift, anomaly, or learn workflows | Truthound |

## Host-Specific Defaults

| Host | Recommended Default | Why |
|------|---------------------|-----|
| Airflow | Truthound | best match for operator, sensor, and SLA patterns |
| Dagster | Truthound | best match for assets, asset checks, and metadata-rich outputs |
| Prefect | Truthound | best match for blocks, tasks, and flow factories |
| dbt | Truthound | the package itself is the first-party Truthound SQL surface |
| Mage | Truthound | best match for block-level checks and runtime helpers |
| Kestra | Truthound | best match for scripts, generated flows, outputs, and stream helpers |

## Production Pattern

- Default to one engine per team unless there is a strong operational reason to
  mix them.
- If you introduce a second engine, keep the selection logic in shared runtime
  configuration rather than scattering it across host code.
- Validate the chosen engine in CI using the same host and source shape used in
  production.

## Related Pages

- [Capability Matrix](capability-matrix.md)
- [Custom Engines](custom-engines.md)
- [Engine Resolution and Selection](../common/engine-resolution-selection.md)
