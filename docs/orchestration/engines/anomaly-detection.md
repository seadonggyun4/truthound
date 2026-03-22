---
title: Anomaly Detection
---

# Anomaly Detection

Anomaly detection is the advanced layer for spotting unusual patterns that might not be
expressed as simple deterministic validation rules.

## When To Use It

- the system needs behavior-aware monitoring, not only rule evaluation
- you want an early warning signal for unusual values or trends
- baseline drift and operational outliers both matter

## How To Use It Safely

- treat anomalies as operator signals first, not always as immediate hard failures
- combine them with host-native notifications and review workflows
- document which anomalies are informational, warning, or blocking

## Related Pages

- [Drift Detection](drift-detection.md)
- [Observability and Resilience](../common/observability-resilience.md)
