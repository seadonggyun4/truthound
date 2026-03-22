---
title: Dagster Troubleshooting
---

# Dagster Troubleshooting

## Resource Not Initialized

Cause:

- `DataQualityResource` is being used outside Dagster's resource lifecycle

Fix:

- use it through `Definitions` or the proper Dagster execution context

## Asset Check Behavior Differs Across Versions

Cause:

- older Dagster lanes expose slightly different asset-check API surfaces

Fix:

- compare against the documented support matrix
- keep examples aligned with the primary supported line when writing new definitions

## Result Metadata Looks Right But Downstream Logic Fails

Cause:

- Dagster metadata is fine, but downstream code is assuming a non-shared result contract

Fix:

- keep downstream logic on documented result fields
