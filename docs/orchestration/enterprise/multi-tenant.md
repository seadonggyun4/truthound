---
title: Multi-Tenant
---

# Multi-Tenant

`truthound.multitenancy` provides native tenant administration, explicit context,
quota tracking and data-isolation strategies for shared deployments.

## Main Components

- `TenantManager` with a `TenantStore`, resolver and quota tracker
- `TenantContext` and `TenantManager.context(tenant_id)`
- `IsolationManager`, `SharedIsolation` and `RowLevelIsolation`
- schema/database namespace helpers and middleware integration

## When To Use It

Use tenant support when one deployment serves several customers or business units
and needs explicit ownership, quotas and data-source boundaries. Tenant metadata
or an entered context alone does not authorize reading an unrelated data source.

## Recommended Operating Model

Keep tenant definitions in a selected store, enter context around the intended
work, and pass the active tenant to native row filtering. Configure callback
policies in trusted application code; policy names must be unique.

```python
import polars as pl
from truthound.multitenancy import (
    IsolationLevel, IsolationManager, RowLevelIsolation, RowLevelPolicy, TenantManager,
)

rows = RowLevelIsolation()
rows.add_policy(RowLevelPolicy("positive", lambda row, tenant: row["value"] > 0))
isolation = IsolationManager()
isolation.register(IsolationLevel.ROW_LEVEL, rows)
manager = TenantManager(isolation_manager=isolation)
tenant = manager.create("Example", tenant_id="example")
frame = pl.DataFrame({"tenant_id": ["example", "other", "example"], "value": [-1, 9, 2]})
filtered = manager.apply_isolation(frame, tenant).select("value").collect()
assert filtered["value"].to_list() == [2]
```

## Isolation Levels

Shared isolation filters the tenant ID column. Row-level isolation first applies
that native tenant predicate, then every enabled policy in descending priority.
Policies return actual booleans; truthy strings and integers do not grant access.
With default `fail_closed=True`, missing tenant columns yield no rows and policy
errors deny the affected row. With `fail_closed=False`, a missing tenant column
passes through and policy errors raise a redacted `TenantIsolationError`.

Row policy evaluation remains lazy and preserves the schema. Projection and
slice pushdown cannot remove policy columns or limit rows before policy checks.
The tenant is checked again when batches are collected. Policy configuration is
snapshotted for each filter plan; `policies` returns configuration copies.

Schema and database strategies assume that the application already selected the
correct tenant source. Their namespace/context checks do not prove that an
arbitrary mixed-tenant dataframe came from that source. Instance isolation is
outside this module's runtime scope.

## Practical Guidance

The opt-in MCP tenant service accepts separately approved pinned asset handles
for shared/row-level filtering and factory-owned policy aliases for
catalog/add/remove/enabled/priority edits. It does not accept Python expressions,
physical tenant IDs or schema/database connections from model requests.
Schema/database isolation of arbitrary caller-supplied assets is unsupported.
Row-policy edits are owner-local configuration, not persisted tenant-store data.
Releasing the input asset revokes its resident service and derived results.

Local tests cover actual Polars row filtering and SQLite tenant metadata. Remote
storage, resolvers and distributed quota trackers need separate verification.

## Related Pages

- [MCP tenant operations](../../guides/mcp.md#bound-tenant-administration)
- [Secrets](secrets.md)
- [CI/CD and Production Rollout](ci-cd-production.md)
