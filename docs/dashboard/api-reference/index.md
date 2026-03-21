# API Reference

The dashboard REST API is a thin operational layer over Truthound 3.0.

## Core domains

- `/sources`
- `/validations`
- `/schemas`
- `/profile`
- `/history`
- `/drift`
- `/scans`
- `/mask`
- `/schedules`
- `/notifications`
- `/alerts`
- `/triggers`
- `/reports`
- `/quality-reporter`
- `/plugins`
- `/lineage`
- `/anomaly`
- `/observability`

## Contract rules

- validation responses normalize `ValidationRunResult`
- checkpoint-oriented views normalize `CheckpointResult.validation_run` and
  `validation_view`
- unsupported dashboard-only product domains are intentionally absent
