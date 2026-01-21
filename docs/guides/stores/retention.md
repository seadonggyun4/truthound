# Retention Policies

Manage storage lifecycle with automatic cleanup and retention rules.

## Overview

Retention policies define rules for keeping or deleting validation results based on:

- Age (time-based)
- Count (number of items)
- Size (storage quota)
- Status (validation result status)
- Tags (metadata-based)

## Quick Start

```python
from truthound.stores.retention.policies import (
    TimeBasedPolicy,
    CountBasedPolicy,
    CompositePolicy,
)
from truthound.stores.retention.base import PolicyMode

# Keep results for 30 days
time_policy = TimeBasedPolicy(max_age_days=30)

# Keep max 1000 results
count_policy = CountBasedPolicy(max_count=1000)

# Combine policies (must satisfy both)
policy = CompositePolicy(
    policies=[time_policy, count_policy],
    mode=PolicyMode.ALL,
)
```

## Policy Types

### TimeBasedPolicy

Delete items older than a specified age.

```python
from truthound.stores.retention.policies import TimeBasedPolicy
from truthound.stores.retention.base import RetentionAction

# Keep items for 30 days
policy = TimeBasedPolicy(max_age_days=30)

# Keep items for 72 hours
policy = TimeBasedPolicy(max_age_hours=72)

# Combined: 7 days, 12 hours
policy = TimeBasedPolicy(max_age_days=7, max_age_hours=12)

# Archive instead of delete
policy = TimeBasedPolicy(
    max_age_days=90,
    action=RetentionAction.ARCHIVE,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_age_days` | `int` | `0` | Maximum age in days |
| `max_age_hours` | `int` | `0` | Additional hours |
| `max_age_minutes` | `int` | `0` | Additional minutes |
| `action` | `RetentionAction` | `DELETE` | Action on expired items |

### CountBasedPolicy

Keep only a maximum number of items.

```python
from truthound.stores.retention.policies import CountBasedPolicy

# Keep max 1000 items globally
policy = CountBasedPolicy(max_count=1000)

# Keep max 100 items per data asset
policy = CountBasedPolicy(max_count=100, per_asset=True)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_count` | `int` | required | Maximum items to keep |
| `per_asset` | `bool` | `False` | Apply limit per data asset |
| `action` | `RetentionAction` | `DELETE` | Action on excess items |

### SizeBasedPolicy

Keep items within a storage quota.

```python
from truthound.stores.retention.policies import SizeBasedPolicy

# Max 500 MB globally
policy = SizeBasedPolicy(max_size_mb=500)

# Max 10 GB per data asset
policy = SizeBasedPolicy(max_size_gb=10, per_asset=True)

# Max 1 GB + 500 MB = 1.5 GB
policy = SizeBasedPolicy(max_size_gb=1, max_size_mb=500)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_size_bytes` | `int` | `0` | Maximum size in bytes |
| `max_size_kb` | `int` | `0` | Maximum size in KB |
| `max_size_mb` | `int` | `0` | Maximum size in MB |
| `max_size_gb` | `int` | `0` | Maximum size in GB |
| `per_asset` | `bool` | `False` | Apply limit per data asset |
| `action` | `RetentionAction` | `DELETE` | Action on excess items |

### StatusBasedPolicy

Apply different retention rules based on validation status.

```python
from truthound.stores.retention.policies import StatusBasedPolicy

# Delete failed results after 7 days
policy = StatusBasedPolicy(
    status="failure",
    max_age_days=7,
)

# Always delete error results
policy = StatusBasedPolicy(
    status="error",
    retain=False,
)

# Keep successful results
policy = StatusBasedPolicy(
    status="success",
    retain=True,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `status` | `str` | required | Status to match |
| `max_age_days` | `int \| None` | `None` | Max age for matching items |
| `retain` | `bool` | `True` | Whether to keep matching items |
| `action` | `RetentionAction` | `DELETE` | Action on non-retained items |

### TagBasedPolicy

Apply retention based on item tags.

```python
from truthound.stores.retention.policies import TagBasedPolicy

# Keep items tagged as production
policy = TagBasedPolicy(required_tags={"env": "production"})

# Delete items tagged as temporary
policy = TagBasedPolicy(delete_tags={"type": "temp"})

# Keep if ANY required tag matches
policy = TagBasedPolicy(
    required_tags={"env": "production", "critical": "true"},
    any_match=True,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `required_tags` | `dict[str, str] \| None` | `None` | Tags that must match to retain |
| `delete_tags` | `dict[str, str] \| None` | `None` | Tags that trigger deletion |
| `any_match` | `bool` | `False` | Match any tag vs all tags |
| `action` | `RetentionAction` | `DELETE` | Action on non-retained items |

### CompositePolicy

Combine multiple policies with AND/OR logic.

```python
from truthound.stores.retention.policies import (
    TimeBasedPolicy,
    CountBasedPolicy,
    StatusBasedPolicy,
    CompositePolicy,
)
from truthound.stores.retention.base import PolicyMode

# AND: Keep if age < 30 days AND count < 1000
policy = CompositePolicy(
    policies=[
        TimeBasedPolicy(max_age_days=30),
        CountBasedPolicy(max_count=1000),
    ],
    mode=PolicyMode.ALL,  # All must agree
)

# OR: Keep if production OR age < 7 days
policy = CompositePolicy(
    policies=[
        TagBasedPolicy(required_tags={"env": "production"}),
        TimeBasedPolicy(max_age_days=7),
    ],
    mode=PolicyMode.ANY,  # Any can keep
)
```

### CustomPolicy

Define custom retention logic with a callable.

```python
from truthound.stores.retention.policies import CustomPolicy
from truthound.stores.retention.base import ItemMetadata

def keep_weekday_only(item: ItemMetadata) -> bool:
    """Keep only items created on weekdays."""
    return item.created_at.weekday() < 5  # Mon-Fri

policy = CustomPolicy(
    predicate=keep_weekday_only,
    description="Keep items created on weekdays",
)

# With lambda
policy = CustomPolicy(
    predicate=lambda item: item.access_count > 10,
    description="Keep frequently accessed items",
)
```

## Retention Actions

Define what happens when an item doesn't pass retention:

```python
from truthound.stores.retention.base import RetentionAction

# Available actions
RetentionAction.DELETE      # Permanently delete
RetentionAction.ARCHIVE     # Move to archive storage
RetentionAction.COMPRESS    # Compress data
RetentionAction.TIER_DOWN   # Move to cheaper storage tier
```

## Configuration

### RetentionConfig

```python
from truthound.stores.retention.base import (
    RetentionConfig,
    RetentionSchedule,
    PolicyMode,
    RetentionAction,
)

config = RetentionConfig(
    policies=[
        TimeBasedPolicy(max_age_days=30),
        CountBasedPolicy(max_count=1000),
    ],
    mode=PolicyMode.ALL,              # Combine mode
    default_action=RetentionAction.DELETE,
    schedule=RetentionSchedule(
        enabled=True,
        interval_hours=24,            # Run daily
        run_at_hour=2,                # At 2 AM
        run_on_days=[0, 1, 2, 3, 4],  # Mon-Fri
        max_duration_seconds=3600,    # Max 1 hour
        batch_size=100,               # Process 100 at a time
    ),
    dry_run=False,
    archive_store_name="archive",     # For ARCHIVE action
    preserve_latest=True,             # Always keep 1 per asset
    excluded_tags={"protected": "true"},
    excluded_assets=["critical_data"],
)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `policies` | `list[RetentionPolicy]` | `[]` | Policies to apply |
| `mode` | `PolicyMode` | `ALL` | How to combine policies |
| `default_action` | `RetentionAction` | `DELETE` | Default cleanup action |
| `schedule` | `RetentionSchedule` | (defaults) | Automatic cleanup schedule |
| `dry_run` | `bool` | `False` | Only report, don't delete |
| `archive_store_name` | `str \| None` | `None` | Archive store name |
| `preserve_latest` | `bool` | `True` | Keep at least 1 per asset |
| `excluded_tags` | `dict[str, str]` | `{}` | Tags that exempt items |
| `excluded_assets` | `list[str]` | `[]` | Assets exempt from cleanup |

## ItemMetadata

Policies evaluate items using this metadata structure:

```python
from truthound.stores.retention.base import ItemMetadata
from datetime import datetime

metadata = ItemMetadata(
    item_id="run-123",
    data_asset="customers.csv",
    created_at=datetime(2024, 1, 1, 12, 0, 0),
    size_bytes=1024,
    status="success",
    tags={"env": "production"},
    access_count=10,
    last_accessed=datetime(2024, 1, 15, 8, 0, 0),
)
```

### ItemMetadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `item_id` | `str` | Unique identifier |
| `data_asset` | `str` | Associated data asset |
| `created_at` | `datetime` | Creation timestamp |
| `size_bytes` | `int` | Size in bytes |
| `status` | `str` | Validation status |
| `tags` | `dict[str, str]` | Item tags |
| `access_count` | `int` | Access count |
| `last_accessed` | `datetime \| None` | Last access time |

## RetentionResult

Cleanup operations return a result object:

```python
from truthound.stores.retention.base import RetentionResult

result = RetentionResult(
    start_time=datetime.now(),
    end_time=datetime.now(),
    items_scanned=1000,
    items_deleted=50,
    items_archived=10,
    items_compressed=0,
    items_tiered=0,
    items_preserved=940,
    items_excluded=20,
    bytes_freed=1024 * 1024,  # 1 MB
    errors=[],
    dry_run=False,
)

print(f"Scanned: {result.items_scanned}")
print(f"Deleted: {result.items_deleted}")
print(f"Duration: {result.duration_seconds}s")
print(f"Freed: {result.bytes_freed} bytes")
```

## Policy Evaluation

### PolicyEvaluator

Evaluate multiple policies against an item:

```python
from truthound.stores.retention.base import PolicyEvaluator, PolicyMode

evaluator = PolicyEvaluator(
    policies=[
        TimeBasedPolicy(max_age_days=30),
        CountBasedPolicy(max_count=1000),
    ],
    mode=PolicyMode.ALL,
)

# Evaluate an item
should_retain, delete_policies = evaluator.evaluate(item_metadata)

if not should_retain:
    print(f"Delete due to: {[p.name for p in delete_policies]}")

# Get deletion priority (higher = delete first)
priority = evaluator.get_deletion_priority(item_metadata)

# Get earliest expiry time
expiry = evaluator.get_earliest_expiry(item_metadata)
```

## Batch Processing

Some policies need batch context for accurate evaluation:

```python
# CountBasedPolicy needs to see all items
policy = CountBasedPolicy(max_count=100, per_asset=True)

# Prepare batch context
items = [ItemMetadata(...) for ...]
policy.prepare_batch(items)

# Now evaluate individual items
for item in items:
    if policy.should_retain(item):
        print(f"Keep: {item.item_id}")
    else:
        print(f"Delete: {item.item_id}")
```

## Real-World Examples

### Development Environment

```python
# Short retention for dev
config = RetentionConfig(
    policies=[
        TimeBasedPolicy(max_age_days=7),
        CountBasedPolicy(max_count=100, per_asset=True),
    ],
    mode=PolicyMode.ALL,
)
```

### Production Environment

```python
# Longer retention, tiered by status
config = RetentionConfig(
    policies=[
        # Keep failures for 30 days
        StatusBasedPolicy(status="failure", max_age_days=30),
        # Keep successes for 90 days
        TimeBasedPolicy(max_age_days=90),
        # But limit total storage
        SizeBasedPolicy(max_size_gb=100),
    ],
    mode=PolicyMode.ALL,
    preserve_latest=True,
    excluded_tags={"audit": "true"},
)
```

### Compliance-Driven Retention

```python
# Keep for regulatory compliance
config = RetentionConfig(
    policies=[
        # 7 years for compliance
        TimeBasedPolicy(max_age_days=2555),  # ~7 years
        # But archive after 1 year
        TimeBasedPolicy(
            max_age_days=365,
            action=RetentionAction.ARCHIVE,
        ),
    ],
    mode=PolicyMode.ANY,
    excluded_assets=["compliance_critical"],
)
```

## Next Steps

- [Tiering](tiering.md) - Hot/Warm/Cold/Archive storage
- [Versioning](versioning.md) - Version history management
- [Observability](observability.md) - Audit, metrics, tracing
