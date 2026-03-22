---
title: Kestra Recipes
---

# Kestra Recipes

## Run A Quality Check In A Python Script Task

Use `check_quality_script` when the flow already has a resolved data URI and only
needs a focused quality gate before the next task.

This is the simplest production pattern for Kestra-backed validation.

## Stream Validation For Large Inputs

Use `stream_quality_script` when the data source is large enough that you want chunked
or streaming-style validation rather than one eager in-memory pass.

## Generate A Standard Team Template

Use `generate_check_flow` or `generate_quality_pipeline` when multiple teams need the
same basic quality-flow skeleton with only namespace, dataset URI, or rule differences.

## Emit Structured Outputs For Downstream Tasks

Use the output helpers when downstream Kestra tasks need a normalized payload rather
than scraping console logs.

## Add SLA Hooks For Production Pipelines

Kestra is often the place where operator-facing alerts matter most. Pair script
execution with SLA monitoring and notification hooks so production incidents are routed
consistently.

## Related Pages

- [Scripts and Flow Templates](scripts-templates.md)
- [Troubleshooting](troubleshooting.md)
