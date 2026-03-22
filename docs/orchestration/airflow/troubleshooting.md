---
title: Airflow Troubleshooting
---

# Airflow Troubleshooting

## Local Paths Work But SQL Fails

Cause:

- local file zero-config is working as intended, but SQL still needs an Airflow connection

Fix:

- configure or reference the Airflow connection explicitly
- verify the hook path before debugging the engine

## DAG Imports Fail On Install

Cause:

- unsupported Airflow or Python tuple
- dependency resolution mismatch

Fix:

- compare the runtime against [Compatibility](../compatibility.md)
- reinstall against the documented Airflow line

## Sensor Waits Forever

Cause:

- threshold is too strict for the current source
- source is stale
- the sensor is waiting on a connection-backed path that never resolves correctly

Fix:

- lower the threshold temporarily for diagnostics
- inspect the same source with `DataQualityCheckOperator`
- prefer `reschedule` or deferrable behavior for long waits

## XCom Consumers Break

Cause:

- downstream code assumes a custom result shape instead of the shared wire contract

Fix:

- consume documented result fields only
- keep any extra metadata additive
