# Cross-Region Replication

Replicate validation results across multiple storage backends for high availability and disaster recovery.

## Overview

The replication module provides:

- **Multi-region replication** - Sync data across geographic regions
- **Multiple modes** - Sync, async, and semi-sync replication
- **Read preferences** - Route reads to optimal replicas
- **Conflict resolution** - Handle concurrent writes
- **Health monitoring** - Track replica health and lag
- **Automatic failover** - Switch to healthy replicas on failure

## Replication Modes

```python
from truthound.stores.replication.base import ReplicationMode

ReplicationMode.SYNC       # Wait for all replicas
ReplicationMode.ASYNC      # Fire and forget
ReplicationMode.SEMI_SYNC  # Wait for at least N replicas
```

| Mode | Consistency | Latency | Use Case |
|------|-------------|---------|----------|
| `SYNC` | Strong | High | Critical data |
| `ASYNC` | Eventual | Low | High throughput |
| `SEMI_SYNC` | Bounded | Medium | Balanced |

## Quick Start

```python
from truthound.stores import get_store
from truthound.stores.replication.base import (
    ReplicationConfig,
    ReplicationMode,
    ReplicaTarget,
)

# Create primary and replica stores
primary = get_store("s3", bucket="primary-bucket", region="us-east-1")
replica1 = get_store("s3", bucket="replica-bucket", region="us-west-2")
replica2 = get_store("s3", bucket="replica-bucket", region="eu-west-1")

# Configure replication
config = ReplicationConfig(
    mode=ReplicationMode.ASYNC,
    targets=[
        ReplicaTarget(
            name="us-west-2",
            store=replica1,
            region="us-west-2",
            priority=1,
        ),
        ReplicaTarget(
            name="eu-west-1",
            store=replica2,
            region="eu-west-1",
            priority=2,
        ),
    ],
)
```

## ReplicaTarget

Define each replication target:

```python
from truthound.stores.replication.base import ReplicaTarget, ReplicaHealth

target = ReplicaTarget(
    name="us-west-2",                  # Unique name
    store=replica_store,               # Store backend
    region="us-west-2",                # Geographic region
    priority=1,                        # Failover priority (lower = higher)
    is_read_replica=False,             # Read-only replica
    sync_timeout_seconds=30.0,         # Sync operation timeout
    max_retry_attempts=3,              # Max retries on failure
    retry_delay_seconds=1.0,           # Base delay between retries
    health=ReplicaHealth.UNKNOWN,      # Initial health status
)
```

### ReplicaTarget Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Unique replica name |
| `store` | `ValidationStore` | required | Store backend |
| `region` | `str` | required | Geographic region |
| `priority` | `int` | `1` | Failover priority |
| `is_read_replica` | `bool` | `False` | Read-only replica |
| `sync_timeout_seconds` | `float` | `30.0` | Sync timeout |
| `max_retry_attempts` | `int` | `3` | Max retry attempts |
| `retry_delay_seconds` | `float` | `1.0` | Retry delay |

### Replica Health Management

```python
# Mark health status
target.mark_healthy()
target.mark_degraded(lag_ms=5000)
target.mark_unhealthy()
target.mark_syncing()

# Pause/resume replication
target.pause()
target.resume()
```

## Configuration

### ReplicationConfig

```python
from truthound.stores.replication.base import (
    ReplicationConfig,
    ReplicationMode,
    ReadPreference,
    ConflictResolution,
)

config = ReplicationConfig(
    mode=ReplicationMode.SEMI_SYNC,
    targets=[...],
    read_preference=ReadPreference.NEAREST,
    conflict_resolution=ConflictResolution.LAST_WRITE_WINS,
    min_sync_replicas=1,               # For semi-sync mode
    enable_health_checks=True,
    health_check_interval_seconds=30.0,
    max_replication_lag_ms=5000.0,
    enable_failover=True,
    enable_metrics=True,
)

# Validate configuration
config.validate()
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `mode` | `ReplicationMode` | `ASYNC` | Replication mode |
| `targets` | `list[ReplicaTarget]` | `[]` | Replica targets |
| `read_preference` | `ReadPreference` | `PRIMARY` | Read routing |
| `conflict_resolution` | `ConflictResolution` | `LAST_WRITE_WINS` | Conflict strategy |
| `min_sync_replicas` | `int` | `1` | Min replicas for semi-sync |
| `enable_health_checks` | `bool` | `True` | Enable health monitoring |
| `health_check_interval_seconds` | `float` | `30.0` | Health check interval |
| `max_replication_lag_ms` | `float` | `5000.0` | Max acceptable lag |
| `enable_failover` | `bool` | `True` | Enable auto failover |
| `enable_metrics` | `bool` | `True` | Collect metrics |

## Read Preferences

Control where reads are routed:

```python
from truthound.stores.replication.base import ReadPreference

ReadPreference.PRIMARY     # Always read from primary
ReadPreference.SECONDARY   # Prefer secondary replicas
ReadPreference.NEAREST     # Read from nearest replica
ReadPreference.ANY         # Read from any available
```

| Preference | Consistency | Latency | Use Case |
|------------|-------------|---------|----------|
| `PRIMARY` | Strong | Higher | Critical reads |
| `SECONDARY` | Eventual | Lower | Read scaling |
| `NEAREST` | Eventual | Lowest | Geo-distributed |
| `ANY` | Eventual | Variable | High availability |

## Conflict Resolution

Handle concurrent writes to multiple regions:

```python
from truthound.stores.replication.base import ConflictResolution

ConflictResolution.LAST_WRITE_WINS   # Most recent write wins
ConflictResolution.FIRST_WRITE_WINS  # First write wins
ConflictResolution.PRIMARY_WINS      # Primary always wins
ConflictResolution.MERGE             # Custom merge logic
ConflictResolution.MANUAL            # Require manual resolution
```

## Health Status

```python
from truthound.stores.replication.base import ReplicaHealth, ReplicaState

# Health status
ReplicaHealth.HEALTHY     # Up and synchronized
ReplicaHealth.DEGRADED    # Up but lagging
ReplicaHealth.UNHEALTHY   # Down or unreachable
ReplicaHealth.UNKNOWN     # Status unknown

# Replica state
ReplicaState.ACTIVE       # Actively receiving writes
ReplicaState.PAUSED       # Temporarily paused
ReplicaState.SYNCING      # Catching up after failure
ReplicaState.FAILED       # Replication failed
ReplicaState.REMOVED      # Removed from replication
```

## Replication Metrics

Track replication performance:

```python
from truthound.stores.replication.base import ReplicationMetrics

metrics = ReplicationMetrics()

# Record operations
metrics.record_primary_write()
metrics.record_replication_success("us-west-2", time_ms=50.0)
metrics.record_replication_failure("eu-west-1")
metrics.record_conflict(resolved=True)
metrics.record_failover()
metrics.update_lag("us-west-2", lag_ms=100.0)

# Get statistics
print(f"Primary writes: {metrics.writes_to_primary}")
print(f"Replicated: {metrics.writes_replicated}")
print(f"Failed: {metrics.writes_failed}")
print(f"Conflicts: {metrics.conflicts_detected}")
print(f"Failovers: {metrics.failovers}")

# Success rate per replica
rate = metrics.get_replica_success_rate("us-west-2")
print(f"US-West-2 success rate: {rate:.1f}%")

# Average replication time
avg_time = metrics.get_average_replication_time()
print(f"Avg replication time: {avg_time:.2f}ms")

# Full metrics dict
metrics_dict = metrics.to_dict()
```

### ReplicationMetrics Fields

| Field | Description |
|-------|-------------|
| `writes_to_primary` | Total writes to primary |
| `writes_replicated` | Successful replications |
| `writes_failed` | Failed replications |
| `conflicts_detected` | Conflict count |
| `conflicts_resolved` | Resolved conflicts |
| `failovers` | Failover count |
| `current_lag_ms` | Current lag per replica |

## Real-World Examples

### Multi-Region HA Setup

```python
config = ReplicationConfig(
    mode=ReplicationMode.SEMI_SYNC,
    min_sync_replicas=1,  # At least 1 replica must ack
    targets=[
        ReplicaTarget(
            name="us-west-2",
            store=get_store("s3", bucket="bucket-west", region="us-west-2"),
            region="us-west-2",
            priority=1,
        ),
        ReplicaTarget(
            name="eu-west-1",
            store=get_store("s3", bucket="bucket-eu", region="eu-west-1"),
            region="eu-west-1",
            priority=2,
        ),
    ],
    read_preference=ReadPreference.NEAREST,
    enable_failover=True,
    max_replication_lag_ms=5000.0,
)
```

### Read Scaling

```python
config = ReplicationConfig(
    mode=ReplicationMode.ASYNC,
    targets=[
        ReplicaTarget(
            name="read-replica-1",
            store=read_store_1,
            region="us-east-1",
            is_read_replica=True,
        ),
        ReplicaTarget(
            name="read-replica-2",
            store=read_store_2,
            region="us-east-1",
            is_read_replica=True,
        ),
    ],
    read_preference=ReadPreference.SECONDARY,
)
```

### Disaster Recovery

```python
config = ReplicationConfig(
    mode=ReplicationMode.SYNC,  # Strong consistency
    targets=[
        ReplicaTarget(
            name="dr-site",
            store=dr_store,
            region="us-west-2",
            priority=1,
            sync_timeout_seconds=60.0,
            max_retry_attempts=5,
        ),
    ],
    conflict_resolution=ConflictResolution.PRIMARY_WINS,
    enable_health_checks=True,
    health_check_interval_seconds=10.0,
)
```

## Best Practices

### Sync Mode

- Use for critical data requiring strong consistency
- Accept higher latency for guaranteed durability
- Monitor replica health closely

### Async Mode

- Use for high-throughput scenarios
- Accept eventual consistency
- Monitor replication lag

### Semi-Sync Mode

- Balance between consistency and latency
- Set `min_sync_replicas` based on durability requirements
- Use with at least 2 replicas

### Health Monitoring

```python
# Check health before operations
for target in config.targets:
    if target.health == ReplicaHealth.UNHEALTHY:
        # Alert or take action
        pass

# Monitor lag
for target in config.targets:
    if target.replication_lag_ms > config.max_replication_lag_ms:
        target.mark_degraded(target.replication_lag_ms)
```

## Next Steps

- [Observability](observability.md) - Audit, metrics, tracing
- [Caching](caching.md) - In-memory caching layer
- [Tiering](tiering.md) - Hot/Warm/Cold/Archive storage
