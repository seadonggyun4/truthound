---
title: Kestra Input and Output Files
---

# Kestra Input and Output Files

Kestra flows commonly move data through files and URIs. Truthound's Kestra integration fits that model: script helpers accept URIs and the shared runtime resolves them into supported sources before execution begins.

## Who This Is For

- teams using Kestra script tasks against files and object storage
- operators debugging how inputs are resolved
- platform engineers defining standard file-based validation templates

## When To Use It

Use this page when:

- a flow passes a file or URI between tasks
- a Python script task should validate the prior task's output
- you want to know how source resolution interacts with Kestra file semantics

## Prerequisites

- a Kestra flow with script tasks
- access to the file or URI used as the validation source
- familiarity with shared source resolution

## Minimal Quickstart

Use the output of a previous step as a validation input:

```python
from truthound_kestra import check_quality_script

check_quality_script(
    data_uri="{{ outputs.extract.uri }}",
    rules=[{"column": "email", "check": "email_format"}],
)
```

For generated templates, keep the URI boundary explicit:

```python
from truthound_kestra import generate_check_flow

yaml_content = generate_check_flow(
    flow_id="users_quality",
    namespace="production",
)
```

## Production Pattern

Recommended file-oriented pattern:

1. produce a file or URI in an upstream task
2. pass that location into a Truthound script task
3. let the shared runtime resolve the source
4. emit results through outputs or metrics

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| validation cannot read the file | the URI is not resolvable in the task runtime | verify the working directory and storage accessibility |
| a local path works in dev but not in production | Kestra task runners do not share the same filesystem layout | prefer explicit URIs or staged files |
| output handoff is inconsistent | each flow chooses a different file contract | standardize one input/output file pattern |

## Related Pages

- [Kestra Overview](index.md)
- [Outputs and Metrics](outputs-metrics.md)
- [Shared Source Resolution](../common/source-resolution.md)
