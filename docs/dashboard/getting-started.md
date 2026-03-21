# Getting Started

## Install

```bash
pip install truthound-dashboard
```

For local development:

```bash
pip install -e ".[dev,docs]"
cd frontend
npm install
```

## Run the dashboard

```bash
truthound serve
```

The default address is `http://localhost:8765`.

## Core workflows

1. Register a source.
2. Learn a schema or run a profile.
3. Run validations and review `ValidationRunResult`-backed issues.
4. Compare baseline vs current data with drift tools.
5. Configure checkpoint-style schedules, actions, alerts, and reports.
