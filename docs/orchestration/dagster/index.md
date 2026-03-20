---
title: Dagster
---

# Dagster

Dagster integration is centered on `ConfigurableResource`. The resource is the boundary; shared runtime logic stays in `common`.

## Install

```bash
pip install truthound-orchestration[dagster] "truthound>=3.0,<4.0"
```

## Zero-Config Path

```python
from dagster import Definitions, asset
from truthound_dagster.resources import DataQualityResource

@asset
def validated_users(data_quality: DataQualityResource):
    return data_quality.check(load_users())

defs = Definitions(resources={"data_quality": DataQualityResource()})
```

`DataQualityResource()` with no arguments is the canonical default.

## Runtime Shape

- resource setup uses the shared resolver
- preflight runs before engine creation proceeds
- result metadata stays Dagster-native, but the underlying payload semantics stay shared
