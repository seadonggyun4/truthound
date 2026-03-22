---
title: Kestra Overview
---

# Truthound for Kestra

Truthound's Kestra integration is YAML-first. It gives Kestra users Python helpers for
script tasks, generated flow templates, structured outputs, and SLA-aware execution
without asking them to abandon native Kestra flow design.

## Who This Is For

- teams orchestrating data quality with Kestra flow YAML
- operators who prefer script-task execution over heavy embedded framework code
- platform engineers standardizing reusable flow templates

## What The Package Provides

- script entry points such as `check_quality_script`
- reusable executor classes for check, stream, profile, and learn workflows
- flow generation helpers such as `generate_check_flow`
- output handlers for Kestra-friendly result emission
- SLA monitoring hooks and presets

## Minimal Quickstart

```python
from truthound_kestra import check_quality_script

result = check_quality_script(
    data_uri="s3://warehouse/curated/users.parquet",
    rules=[
        {"column": "id", "check": "not_null"},
        {"column": "email", "check": "email_format"},
    ],
)
```

## When To Use Kestra

Choose Kestra when:

- you want flow YAML to remain the main control plane
- Python script tasks are already an accepted pattern in your organization
- you need reusable generated flow templates for common quality workflows

## Recommended Reading Order

- [Scripts and Flow Templates](scripts-templates.md)
- [Recipes](recipes.md)
- [Troubleshooting](troubleshooting.md)
