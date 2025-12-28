"""Cross-region replication for storage backends.

This module provides replication capabilities to distribute validation
results across multiple regions for disaster recovery and read scalability.

Features:
    - Sync and async replication modes
    - Multiple replication strategies (full, incremental, selective)
    - Conflict resolution policies
    - Health monitoring and failover
    - Replication lag tracking

Example:
    >>> from truthound.stores.replication import (
    ...     ReplicatedStore,
    ...     ReplicationConfig,
    ...     ReplicaTarget,
    ... )
    >>>
    >>> primary = S3Store(region="us-east-1")
    >>> replica_eu = S3Store(region="eu-west-1")
    >>> replica_ap = S3Store(region="ap-northeast-1")
    >>>
    >>> config = ReplicationConfig(
    ...     mode=ReplicationMode.ASYNC,
    ...     targets=[
    ...         ReplicaTarget(name="eu", store=replica_eu, region="eu-west-1"),
    ...         ReplicaTarget(name="ap", store=replica_ap, region="ap-northeast-1"),
    ...     ],
    ... )
    >>> store = ReplicatedStore(primary, config)
"""

from truthound.stores.replication.base import (
    ConflictResolution,
    ReadPreference,
    ReplicaHealth,
    ReplicaState,
    ReplicaTarget,
    ReplicationConfig,
    ReplicationMetrics,
    ReplicationMode,
)
from truthound.stores.replication.store import (
    ReplicatedStore,
)
from truthound.stores.replication.syncer import (
    ReplicationSyncer,
    SyncResult,
)
from truthound.stores.replication.monitor import (
    ReplicationMonitor,
    ReplicationEvent,
    ReplicationEventType,
)

__all__ = [
    # Base
    "ConflictResolution",
    "ReadPreference",
    "ReplicaHealth",
    "ReplicaState",
    "ReplicaTarget",
    "ReplicationConfig",
    "ReplicationMetrics",
    "ReplicationMode",
    # Store
    "ReplicatedStore",
    # Syncer
    "ReplicationSyncer",
    "SyncResult",
    # Monitor
    "ReplicationMonitor",
    "ReplicationEvent",
    "ReplicationEventType",
]
