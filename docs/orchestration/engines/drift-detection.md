---
title: Drift Detection
---

# Drift Detection

Drift detection is an advanced engine capability for comparing current data behavior to
an earlier baseline. It is useful when a dataset can still pass simple checks while
quietly changing in ways operators care about.

## When To Use It

- baseline distributions matter, not just row-level validity
- upstream systems change gradually and silently
- model quality needs trend-aware monitoring

## Operational Guidance

- pair drift detection with profiling and baseline management
- use it as an additional signal, not a replacement for core checks
- keep severity and notification policy explicit so expected seasonal changes do not
  page the team unnecessarily

## Related Pages

- [Anomaly Detection](anomaly-detection.md)
- [Truthound Engine](truthound.md)
