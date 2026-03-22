---
title: Dagster Recipes
---

# Dagster Recipes

## Add A Resource To Existing Definitions

Start by adding `DataQualityResource()` to your existing `Definitions` object and validating one asset with a simple rule.

## Profile Before Turning On Hard Fails

Use `profiled_asset` or a profile op in staging before enabling strict asset checks in production.

## Use Asset Checks For Catalog Visibility

Prefer asset checks when you want failures to show up as first-class Dagster quality signals rather than buried inside asset logic.

## Separate Quality Execution From Alerting

Run validation through resources or asset helpers, then attach SLA hooks or separate downstream ops for alerting and incident routing.
