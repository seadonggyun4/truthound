# Python API Reference

The Python API is the canonical programmatic surface for **Truthound Core**.
Use it when you want direct access to the 3.0 validation kernel,
`ValidationRunResult`, schema learning, checkpoint orchestration inside the core
repo, or core-adjoining namespaces such as drift, reporters, profiler, and
Data Docs.

This page is intentionally scoped to the `truthound` repository. Host-native
adapter APIs live in [Truthound Orchestration](../orchestration/index.md), and
control-plane APIs live in [Truthound Dashboard](../dashboard/index.md).

## When To Use The Python API

Choose Python when you need to:

- validate dataframes, lazyframes, dictionaries, SQL-backed datasources, or cloud warehouse connectors
- compose Truthound Core into notebooks, scripts, services, jobs, or application code
- inspect `ValidationRunResult` directly for automation, routing, or custom output
- work with namespace modules such as `truthound.drift`, `truthound.checkpoint`, `truthound.reporters`, and `truthound.profiler`

## Installation

```bash
pip install truthound
```

## Quick Start

```python
import truthound as th
from truthound.drift import compare

# Validate data through the 3.0 kernel
run = th.check("data.csv")

# Learn a reusable baseline schema
schema = th.learn("baseline.csv")

# Scan for PII
pii_report = th.scan("customers.csv")

# Mask sensitive data
masked_df = th.mask(run.source, strategy="hash")

# Profile data
profile = th.profile("data.csv")

# Compare datasets for drift
drift = compare("baseline.csv", "current.csv")
```

## Core 3.0 Mental Model

Truthound 3.0 keeps the root package intentionally small:

- `truthound` exposes the thin validation facade and core result/context types
- `th.check()` returns `ValidationRunResult`
- advanced core-adjoining capabilities live in namespaces such as `truthound.drift`, `truthound.checkpoint`, and `truthound.reporters`
- reporters, checkpoints, and Data Docs are outer services built on the same canonical core result model
- result helpers such as `render()`, `write()`, and `build_docs()` are convenience facades that lazy-import those outer services

Start with [Core Functions](core-functions.md) if you want the most important
runtime contract first.

## Import Patterns

Use the root package for the thin public facade:

```python
from truthound import (
    check,
    scan,
    mask,
    profile,
    read,
    learn,
    Schema,
    TruthoundContext,
    ValidationRunResult,
)
```

Use namespace imports for core-adjoining features:

```python
from truthound.drift import compare
from truthound.reporters import get_reporter
from truthound.checkpoint import Checkpoint, CheckpointConfig
from truthound.profiler import profile_data
```

If you need the module object, import the namespace directly:

```python
import truthound as th
import truthound.checkpoint as checkpoint
import truthound.drift as drift
```

## Recommended Reading Path

1. [Core Functions](core-functions.md)
2. [Schema](schema.md)
3. [Validators](validators.md)
4. [Datasources](datasources.md)
5. [Reporters](reporters.md)
6. [Advanced Features](advanced.md)

If you are new to Truthound, read [Quick Start](../getting-started/quickstart.md)
first and then return here for lookup-oriented detail.

## API Overview

### Root Facade

| Symbol | Description |
|--------|-------------|
| [`th.check()`](core-functions.md#thcheck) | Validate data and return `ValidationRunResult` |
| [`th.learn()`](core-functions.md#thlearn) | Learn schema from baseline data |
| [`th.scan()`](core-functions.md#thscan) | Scan data for PII |
| [`th.mask()`](core-functions.md#thmask) | Mask sensitive values |
| [`th.profile()`](core-functions.md#thprofile) | Generate a data profile |
| [`th.read()`](core-functions.md#thread) | Load supported sources into Polars |
| `ValidationRunResult` | Canonical validation runtime output |
| `TruthoundContext` | Zero-config workspace and artifact boundary |

### Core-Adjacent Namespaces

| Namespace | Description |
|----------|-------------|
| `truthound.drift` | Drift comparison via `compare()` and `DriftReport` |
| `truthound.checkpoint` | Checkpoint orchestration, actions, and CI integration |
| `truthound.reporters` | Rendering and serialization of validation runs |
| `truthound.profiler` | Profiling, quality scoring, and comparison workflows |
| `truthound.datadocs` | HTML docs generation from validation runs |
| `truthound.lineage` | Lineage and dependency analysis |
| `truthound.realtime` | Streaming and incremental validation |
| `truthound.ml` | ML-assisted anomaly and drift tooling |

### Core Types

| Type | Description |
|------|-------------|
| `ValidationRunResult` | Immutable validation run result |
| `CheckResult` | Per-check execution outcome inside a run |
| `ValidationIssue` | Individual issue emitted by a validator |
| `Schema` | Schema container for learned or authored contracts |

## Supported Input Types

The Python API accepts the same broad input surface across the root validation
functions:

```python
import pandas as pd
import polars as pl
import truthound as th

# File paths
run = th.check("data.csv")
run = th.check("data.parquet")
run = th.check("data.json")

# Polars DataFrame / LazyFrame
df = pl.read_csv("data.csv")
run = th.check(df)
run = th.check(df.lazy())

# Pandas DataFrame
pdf = pd.read_csv("data.csv")
run = th.check(pdf)

# Dictionary input
run = th.check({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

# SQL-backed DataSource
from truthound.datasources.sql import PostgreSQLDataSource

source = PostgreSQLDataSource(
    table="users",
    host="localhost",
    database="mydb",
    user="postgres",
)
run = th.check(source=source)
```

## Error Handling

```python
import truthound as th
from truthound.datasources.base import DataSourceError
from truthound.validators.base import (
    ValidationTimeoutError,
    ColumnNotFoundError,
    RegexValidationError,
)

try:
    run = th.check("data.csv", catch_exceptions=False)
    if run.issues:
        print(f"Found {len(run.issues)} issues")
except DataSourceError as exc:
    print(f"Data source error: {exc}")
except ValidationTimeoutError as exc:
    print(f"Validation timed out: {exc}")
except ColumnNotFoundError as exc:
    print(f"Column not found: {exc}")
except RegexValidationError as exc:
    print(f"Invalid regex: {exc}")
```

## Type Hints

Truthound is fully typed. For static analysis, import the canonical result and
namespace types directly:

```python
from truthound import ValidationRunResult, check, learn, mask, profile, read, scan
from truthound.core.results import CheckResult
from truthound.datasources.base import BaseDataSource
from truthound.drift import ColumnDrift, DriftReport, compare
from truthound.schema import Schema, ColumnSchema
from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
```

## See Also

- [Reference Overview](../reference/index.md) - Python API and CLI lookup hub
- [CLI Reference](../cli/index.md) - Command-line interface
- [Guides](../guides/index.md) - Task-oriented usage guides
- [Tutorials](../tutorials/index.md) - Step-by-step learning paths
- [Migration to 3.0](../guides/migration-3.0.md) - Removed root imports and legacy result surfaces
