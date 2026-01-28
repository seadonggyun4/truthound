# Storage Tiering

Manage data across Hot, Warm, Cold, and Archive storage tiers.

## Overview

Storage tiering automatically migrates data between different storage backends based on access patterns, age, or size. This optimizes costs by keeping frequently accessed data on fast storage and moving older data to cheaper storage.

## Tier Types

```python
from truthound.stores.tiering.base import TierType

TierType.HOT      # Frequently accessed, fast, expensive
TierType.WARM     # Occasionally accessed, moderate speed/cost
TierType.COLD     # Rarely accessed, slow, cheap
TierType.ARCHIVE  # Very rarely accessed, very slow, cheapest
```

## Quick Start

```python
from truthound.stores import get_store
from truthound.stores.tiering.base import StorageTier, TierType, TieringConfig
from truthound.stores.tiering.policies import AgeBasedTierPolicy

# Create tier backends
hot_store = get_store("filesystem", base_path=".truthound/hot")
warm_store = get_store("s3", bucket="my-bucket", prefix="warm/")
cold_store = get_store("s3", bucket="archive-bucket", prefix="cold/")

# Define tiers
tiers = [
    StorageTier(
        name="hot",
        store=hot_store,
        tier_type=TierType.HOT,
        priority=1,
    ),
    StorageTier(
        name="warm",
        store=warm_store,
        tier_type=TierType.WARM,
        priority=2,
    ),
    StorageTier(
        name="cold",
        store=cold_store,
        tier_type=TierType.COLD,
        priority=3,
    ),
]

# Define migration policies
config = TieringConfig(
    policies=[
        AgeBasedTierPolicy("hot", "warm", after_days=7),
        AgeBasedTierPolicy("warm", "cold", after_days=30),
    ],
    default_tier="hot",
)
```

## Tier Policies

### AgeBasedTierPolicy

Migrate items based on age.

```python
from truthound.stores.tiering.policies import AgeBasedTierPolicy
from truthound.stores.tiering.base import MigrationDirection

# Move to warm after 7 days
policy = AgeBasedTierPolicy(
    from_tier="hot",
    to_tier="warm",
    after_days=7,
)

# Move to cold after 30 days
policy = AgeBasedTierPolicy(
    from_tier="warm",
    to_tier="cold",
    after_days=30,
)

# Combined: 1 day 12 hours
policy = AgeBasedTierPolicy(
    from_tier="hot",
    to_tier="warm",
    after_days=1,
    after_hours=12,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `from_tier` | `str` | required | Source tier name |
| `to_tier` | `str` | required | Destination tier name |
| `after_days` | `int` | `0` | Days before migration |
| `after_hours` | `int` | `0` | Additional hours |
| `direction` | `MigrationDirection` | `DEMOTE` | Migration direction |

### AccessBasedTierPolicy

Migrate based on access patterns.

```python
from truthound.stores.tiering.policies import AccessBasedTierPolicy
from truthound.stores.tiering.base import MigrationDirection

# Demote items not accessed in 30 days
policy = AccessBasedTierPolicy(
    from_tier="hot",
    to_tier="warm",
    inactive_days=30,
)

# Promote frequently accessed items
policy = AccessBasedTierPolicy(
    from_tier="warm",
    to_tier="hot",
    min_access_count=100,
    access_window_days=7,
    direction=MigrationDirection.PROMOTE,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `from_tier` | `str` | required | Source tier name |
| `to_tier` | `str` | required | Destination tier name |
| `inactive_days` | `int \| None` | `None` | Days without access for demotion |
| `min_access_count` | `int \| None` | `None` | Accesses needed for promotion |
| `access_window_days` | `int` | `7` | Window for counting accesses |
| `direction` | `MigrationDirection` | `DEMOTE` | Migration direction |

### SizeBasedTierPolicy

Migrate based on item size or tier capacity.

```python
from truthound.stores.tiering.policies import SizeBasedTierPolicy

# Move large items (>100MB) to cold storage
policy = SizeBasedTierPolicy(
    from_tier="hot",
    to_tier="cold",
    min_size_mb=100,
)

# Limit hot tier to 10GB total
policy = SizeBasedTierPolicy(
    from_tier="hot",
    to_tier="warm",
    tier_max_size_gb=10,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `from_tier` | `str` | required | Source tier name |
| `to_tier` | `str` | required | Destination tier name |
| `min_size_bytes` | `int` | `0` | Minimum item size in bytes |
| `min_size_kb` | `int` | `0` | Minimum item size in KB |
| `min_size_mb` | `int` | `0` | Minimum item size in MB |
| `min_size_gb` | `int` | `0` | Minimum item size in GB |
| `tier_max_size_bytes` | `int` | `0` | Maximum total tier size in bytes |
| `tier_max_size_gb` | `int` | `0` | Maximum total tier size in GB |
| `direction` | `MigrationDirection` | `DEMOTE` | Migration direction |

### ScheduledTierPolicy

Migrate on a schedule (specific days/times).

```python
from truthound.stores.tiering.policies import ScheduledTierPolicy

# Migrate to cold storage on weekends at 2 AM
policy = ScheduledTierPolicy(
    from_tier="warm",
    to_tier="cold",
    on_days=[5, 6],  # Saturday, Sunday
    at_hour=2,
)

# Migrate items older than 7 days on Monday mornings
policy = ScheduledTierPolicy(
    from_tier="hot",
    to_tier="warm",
    on_days=[0],  # Monday
    at_hour=6,
    min_age_days=7,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `from_tier` | `str` | required | Source tier name |
| `to_tier` | `str` | required | Destination tier name |
| `on_days` | `list[int] \| None` | `None` | Days to run (0=Mon, 6=Sun) |
| `at_hour` | `int \| None` | `None` | Hour to run (0-23) |
| `min_age_days` | `int` | `0` | Minimum item age |
| `direction` | `MigrationDirection` | `DEMOTE` | Migration direction |

### CompositeTierPolicy

Combine multiple policies with AND/OR logic for complex migration rules.

```python
from truthound.stores.tiering.policies import (
    AgeBasedTierPolicy,
    SizeBasedTierPolicy,
    AccessBasedTierPolicy,
    CompositeTierPolicy,
)

# AND logic: Migrate if old AND large (both conditions must be true)
policy = CompositeTierPolicy(
    from_tier="hot",
    to_tier="cold",
    policies=[
        AgeBasedTierPolicy("hot", "cold", after_days=30),
        SizeBasedTierPolicy("hot", "cold", min_size_mb=100),
    ],
    require_all=True,
)

# OR logic: Migrate if old OR large (either condition triggers migration)
policy = CompositeTierPolicy(
    from_tier="hot",
    to_tier="cold",
    policies=[
        AgeBasedTierPolicy("hot", "cold", after_days=90),
        SizeBasedTierPolicy("hot", "cold", min_size_mb=500),
    ],
    require_all=False,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `from_tier` | `str` | required | Source tier name |
| `to_tier` | `str` | required | Destination tier name |
| `policies` | `list[TierPolicy]` | required | Child policies to combine |
| `require_all` | `bool` | `True` | `True` = AND logic (all must match), `False` = OR logic (any match) |
| `direction` | `MigrationDirection` | `DEMOTE` | Migration direction |

#### Advanced Examples

**Combining Three or More Policies:**

```python
# Migrate if: old AND large AND inactive
policy = CompositeTierPolicy(
    from_tier="hot",
    to_tier="archive",
    policies=[
        AgeBasedTierPolicy("hot", "archive", after_days=180),
        SizeBasedTierPolicy("hot", "archive", min_size_mb=500),
        AccessBasedTierPolicy("hot", "archive", inactive_days=90),
    ],
    require_all=True,
)
```

**Nested Composite Policies:**

```python
# Complex rule: (old AND large) OR (very old)
age_and_size = CompositeTierPolicy(
    from_tier="hot",
    to_tier="cold",
    policies=[
        AgeBasedTierPolicy("hot", "cold", after_days=30),
        SizeBasedTierPolicy("hot", "cold", min_size_mb=100),
    ],
    require_all=True,
)

very_old = AgeBasedTierPolicy("hot", "cold", after_days=365)

combined = CompositeTierPolicy(
    from_tier="hot",
    to_tier="cold",
    policies=[age_and_size, very_old],
    require_all=False,  # Either nested condition triggers
)
```

**Batch Processing:**

```python
# CompositeTierPolicy delegates prepare_batch() to all child policies
# This enables efficient batch evaluation
policy.prepare_batch(tier_items)  # Prepares all child policies

# Description shows combined logic
print(policy.description)
# Output:
# Migrate from hot to cold when all of:
#   - Migrate from hot to cold if older than 30 days
#   - Migrate from hot to cold if size >= 100 MB
```

**Serialization:**

```python
# Serialize to dictionary (useful for configuration persistence)
config = policy.to_dict()
# {
#     "type": "CompositeTierPolicy",
#     "from_tier": "hot",
#     "to_tier": "cold",
#     "direction": "DEMOTE",
#     "policies": [
#         {"type": "AgeBasedTierPolicy", "from_tier": "hot", ...},
#         {"type": "SizeBasedTierPolicy", "from_tier": "hot", ...},
#     ],
#     "require_all": true
# }
```

### CustomTierPolicy

Define custom migration logic.

```python
from truthound.stores.tiering.policies import CustomTierPolicy
from truthound.stores.tiering.base import TierInfo

def large_and_idle(info: TierInfo) -> bool:
    """Migrate large items that are rarely accessed."""
    return info.size_bytes > 1024 * 1024 and info.access_count < 5

policy = CustomTierPolicy(
    from_tier="hot",
    to_tier="warm",
    predicate=large_and_idle,
    description="Large but rarely accessed items",
)
```

## Configuration

### StorageTier

Define each storage tier:

```python
from truthound.stores.tiering.base import StorageTier, TierType

tier = StorageTier(
    name="hot",                  # Unique identifier
    store=hot_store,             # Store backend
    tier_type=TierType.HOT,      # Tier classification
    priority=1,                  # Read order (lower = higher priority)
    cost_per_gb=0.023,           # For cost analysis
    retrieval_time_ms=10,        # Expected latency
    metadata={"region": "us-east-1"},
)
```

### TieringConfig

Configure tiering behavior:

```python
from truthound.stores.tiering.base import TieringConfig

config = TieringConfig(
    policies=[...],                    # Migration policies
    default_tier="hot",                # Default for new items
    enable_promotion=True,             # Promote on frequent access
    promotion_threshold=10,            # Accesses to trigger promotion
    check_interval_hours=24,           # Hours between auto-checks
    batch_size=100,                    # Items per migration batch
    enable_parallel_migration=False,   # Parallel migration
    max_parallel_migrations=4,         # Max concurrent migrations
)
```

## TierInfo

Metadata about an item's tier placement:

```python
from truthound.stores.tiering.base import TierInfo
from datetime import datetime

info = TierInfo(
    item_id="run-123",
    tier_name="hot",
    created_at=datetime.now(),
    migrated_at=None,              # When last migrated
    access_count=5,                # Number of accesses
    last_accessed=datetime.now(),  # Last access time
    size_bytes=1024,               # Item size
    next_migration=None,           # Scheduled migration time
)
```

### TierInfo Fields

| Field | Type | Description |
|-------|------|-------------|
| `item_id` | `str` | Item identifier |
| `tier_name` | `str` | Current tier name |
| `created_at` | `datetime` | Creation timestamp |
| `migrated_at` | `datetime \| None` | Last migration time |
| `access_count` | `int` | Access count |
| `last_accessed` | `datetime \| None` | Last access time |
| `size_bytes` | `int` | Item size |
| `next_migration` | `datetime \| None` | Scheduled migration |

## TieringResult

Result of a tiering operation:

```python
from truthound.stores.tiering.base import TieringResult

result = TieringResult(
    start_time=datetime.now(),
    end_time=datetime.now(),
    items_scanned=1000,
    items_migrated=50,
    bytes_migrated=1024 * 1024 * 100,  # 100 MB
    migrations=[
        {"item_id": "run-1", "from": "hot", "to": "warm"},
        ...
    ],
    errors=[],
    dry_run=False,
)

print(f"Migrated: {result.items_migrated}")
print(f"Duration: {result.duration_seconds}s")
print(f"Bytes moved: {result.bytes_migrated}")
```

## Migration Direction

```python
from truthound.stores.tiering.base import MigrationDirection

MigrationDirection.DEMOTE   # Move to cheaper/slower tier
MigrationDirection.PROMOTE  # Move to faster/more expensive tier
```

## Tier Metadata Store

Track tier placement with the metadata store:

```python
from truthound.stores.tiering.base import InMemoryTierMetadataStore

store = InMemoryTierMetadataStore()

# Save tier info
store.save_info(tier_info)

# Get tier info
info = store.get_info("run-123")

# List items in a tier
hot_items = store.list_by_tier("hot")

# Update access stats (called on read)
store.update_access("run-123")

# Delete info
store.delete_info("run-123")
```

## Error Handling

```python
from truthound.stores.tiering.base import (
    TieringError,
    TierNotFoundError,
    TierMigrationError,
    TierAccessError,
)

try:
    # Access non-existent tier
    ...
except TierNotFoundError as e:
    print(f"Tier not found: {e.tier_name}")

try:
    # Migration failure
    ...
except TierMigrationError as e:
    print(f"Migration failed: {e.item_id} from {e.from_tier} to {e.to_tier}")

try:
    # Tier access error
    ...
except TierAccessError as e:
    print(f"Access error on tier {e.tier_name}")
```

## Real-World Examples

### Cost-Optimized Tiering

```python
# Define tiers with cost info
tiers = [
    StorageTier(
        name="hot",
        store=ssd_store,
        tier_type=TierType.HOT,
        cost_per_gb=0.10,
        retrieval_time_ms=10,
    ),
    StorageTier(
        name="warm",
        store=s3_standard_store,
        tier_type=TierType.WARM,
        cost_per_gb=0.023,
        retrieval_time_ms=100,
    ),
    StorageTier(
        name="cold",
        store=s3_glacier_store,
        tier_type=TierType.COLD,
        cost_per_gb=0.004,
        retrieval_time_ms=180000,  # 3 hours
    ),
]

config = TieringConfig(
    policies=[
        AgeBasedTierPolicy("hot", "warm", after_days=7),
        AgeBasedTierPolicy("warm", "cold", after_days=90),
        AccessBasedTierPolicy("cold", "warm", min_access_count=3,
                              direction=MigrationDirection.PROMOTE),
    ],
)
```

### Access-Pattern Tiering

```python
config = TieringConfig(
    policies=[
        # Demote inactive items
        AccessBasedTierPolicy("hot", "warm", inactive_days=14),
        AccessBasedTierPolicy("warm", "cold", inactive_days=60),
        # Promote frequently accessed items
        AccessBasedTierPolicy(
            "warm", "hot",
            min_access_count=50,
            access_window_days=7,
            direction=MigrationDirection.PROMOTE,
        ),
    ],
    enable_promotion=True,
    promotion_threshold=50,
)
```

### Size-Constrained Tiering

```python
config = TieringConfig(
    policies=[
        # Keep hot tier under 100GB
        SizeBasedTierPolicy("hot", "warm", tier_max_size_gb=100),
        # Move very large items directly to cold
        SizeBasedTierPolicy("hot", "cold", min_size_gb=1),
    ],
)
```

## Next Steps

- [Retention](retention.md) - TTL and retention policies
- [Caching](caching.md) - In-memory caching layer
- [Observability](observability.md) - Audit, metrics, tracing
