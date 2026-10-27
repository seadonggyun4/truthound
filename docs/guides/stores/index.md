# Storage Backends Guide

This guide covers persisting validation results with Truthound's Python API. It includes practical workflows for configuring storage backends, combining enterprise features, and managing result lifecycle.

---

## Quick Start

```python
import truthound as th
from truthound.stores import get_store
from truthound.stores.results import ValidationResult

# Create store
store = get_store("filesystem", base_path=".truthound/store")

# Run validation and persist the storage DTO
run = th.check("customers.csv")
stored_result = ValidationResult.from_report(run, "customers.csv")
run_id = store.save(stored_result)

# Retrieve result
loaded = store.get(run_id)
```

---

## Common Workflows

### Workflow 1: Production Storage with S3

```python
from truthound.stores import get_store
from truthound.stores.results import ValidationResult
import truthound as th

# Configure S3 store
store = get_store(
    "s3",
    bucket="company-validation-results",
    prefix="prod/daily/",
    region="us-east-1",
)

# Run validation
run = th.check("data.csv")

# Save with metadata
stored_result = ValidationResult.from_report(
    run,
    data_asset="data.csv",
    tags={"environment": "production", "team": "data-eng"},
)
run_id = store.save(stored_result)
print(f"Saved as: {run_id}")
```

### Workflow 2: Combining Versioning + Caching + Observability

```python
from truthound.stores import get_store
from truthound.stores.versioning import VersionedStore, VersioningConfig
from truthound.stores.caching import CachedStore, CacheMode
from truthound.stores.caching.backends import LRUCache
from truthound.stores.observability import ObservableStore
from truthound.stores.observability.config import (
    ObservabilityConfig,
    AuditConfig,
    MetricsConfig,
)
from truthound.stores.results import ValidationResult
import truthound as th

# Layer 1: Base storage
base = get_store("s3", bucket="validation-results", prefix="prod/")

# Layer 2: Add version history (keep last 10 versions)
versioned = VersionedStore(base, VersioningConfig(max_versions=10))

# Layer 3: Add caching (LRU cache with 1-hour TTL)
cache = LRUCache(max_size=1000, ttl_seconds=3600)
cached = CachedStore(versioned, cache, mode=CacheMode.READ_WRITE)

# Layer 4: Add observability (audit logs + metrics)
store = ObservableStore(
    cached,
    ObservabilityConfig(
        audit=AuditConfig(enabled=True),
        metrics=MetricsConfig(enabled=True),
    ),
)

# Run validation and persist the storage DTO
run = th.check("data.csv")
stored_result = ValidationResult.from_report(run, data_asset="data.csv")
run_id = store.save(stored_result)

# Version operations
history = versioned.get_history(run_id, limit=5)
versioned.rollback(run_id, version=3)
```

### Workflow 3: Query Historical Results

```python
from truthound.stores import get_store
from truthound.stores.base import StoreQuery
from datetime import datetime, timedelta

store = get_store("database", connection_url="postgresql://localhost/truthound")

# Query last 7 days of failures
query = StoreQuery(
    data_asset="customers.csv",
    start_time=datetime.now() - timedelta(days=7),
    status="failure",
    limit=100,
    order_by="run_time",
    ascending=False,
)

results = store.query(query)
for result in results:
    print(f"{result.run_time}: {result.issue_count} issues")
```

### Workflow 4: Retention Policy with Auto-Cleanup

```python
from truthound.stores import get_store
from truthound.stores.retention import (
    RetentionStore,
    TimeBasedPolicy,
    CountBasedPolicy,
    CompositePolicy,
)
from datetime import timedelta

# Base store
base = get_store("filesystem", base_path=".truthound/store")

# Define retention policies
policy = CompositePolicy([
    TimeBasedPolicy(max_age=timedelta(days=90)),  # Delete after 90 days
    CountBasedPolicy(max_count=1000),              # Keep max 1000 results
])

# Wrap with retention
store = RetentionStore(base, policy, cleanup_interval_hours=24)

# Results older than 90 days are automatically cleaned up
```

---

## Full Documentation

Truthound provides flexible storage backends for persisting validation results, schemas, and profiles. All stores implement a common interface with support for CRUD operations, querying, and optional features like versioning, caching, and replication.

## Backend Comparison

| Backend | Use Case | Persistence | Scalability | Dependencies |
|---------|----------|-------------|-------------|--------------|
| [FileSystem](filesystem.md) | Local development, single-node | File-based | Single machine | None |
| [Memory](filesystem.md#memory-store) | Testing, ephemeral | None | Single process | None |
| [S3](cloud-storage.md#s3-store) | AWS production | Cloud | Unlimited | `boto3` |
| [GCS](cloud-storage.md#gcs-store) | GCP production | Cloud | Unlimited | `google-cloud-storage` |
| [Azure Blob](cloud-storage.md#azure-blob-store) | Azure production | Cloud | Unlimited | `azure-storage-blob` |
| [Database](cloud-storage.md#database-store) | SQL-based storage | Database | Depends on DB | `sqlalchemy` |

## Installation

```bash
# Core (includes FileSystem and Memory stores)
pip install truthound

# Cloud storage backends
pip install truthound[s3]      # AWS S3
pip install truthound[gcs]     # Google Cloud Storage
pip install truthound[azure]   # Azure Blob Storage
pip install truthound[db]      # Database (SQLAlchemy)

# All storage backends
pip install truthound[all-stores]
```

## Quick Start

### Using the Factory

```python
from truthound.stores import get_store

# FileSystem (default)
store = get_store("filesystem", base_path=".truthound/store")

# S3
store = get_store(
    "s3",
    bucket="my-bucket",
    prefix="validations/",
    region="us-east-1",
)

# Database
store = get_store(
    "database",
    connection_url="postgresql://user:pass@localhost/db",
)
```

### Direct Instantiation

```python
from truthound.stores.backends.filesystem import FileSystemStore, FileSystemConfig
from truthound.stores.backends.s3 import S3Store

# FileSystem with config
config = FileSystemConfig(
    base_path=".truthound/store",
    use_compression=True,
    pretty_print=False,
)
store = FileSystemStore(config=config)

# S3 with direct parameters
store = S3Store(
    bucket="my-validation-results",
    prefix="prod/",
    region="us-west-2",
    compression=True,
)
```

## Common Operations

All stores implement the `ValidationStore` protocol:

```python
from truthound.stores.results import ValidationResult
from truthound.stores.base import StoreQuery
import truthound as th

# Initialize store (lazy initialization)
store.initialize()

# Save a result
run = th.check("customers.csv")
stored_result = ValidationResult.from_report(run, "customers.csv")
run_id = store.save(stored_result)

# Retrieve a result
result = store.get(run_id)

# Check existence
exists = store.exists(run_id)

# Delete a result
deleted = store.delete(run_id)

# List all IDs
ids = store.list_ids()

# Query with filters
query = StoreQuery(
    data_asset="customers.csv",
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2024, 12, 31),
    status="failure",
    limit=100,
)
results = store.query(query)

# Close connection
store.close()
```

## StoreQuery Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_asset` | `str \| None` | Filter by data asset name |
| `start_time` | `datetime \| None` | Filter results after this time |
| `end_time` | `datetime \| None` | Filter results before this time |
| `status` | `str \| None` | Filter by status (`success`, `failure`, `error`) |
| `tags` | `dict[str, str] \| None` | Filter by tags |
| `limit` | `int \| None` | Maximum results to return |
| `offset` | `int` | Skip first N results (default: 0) |
| `order_by` | `str` | Sort field (default: `run_time`) |
| `ascending` | `bool` | Sort order (default: `False`) |

## Enterprise Features

Truthound stores support advanced features through wrapper stores:

| Feature | Module | Description |
|---------|--------|-------------|
| [Versioning](versioning.md) | `stores/versioning/` | Version history, diff, rollback |
| [Retention](retention.md) | `stores/retention/` | TTL policies, automatic cleanup |
| [Tiering](tiering.md) | `stores/tiering/` | Hot/Warm/Cold/Archive storage |
| [Caching](caching.md) | `stores/caching/` | LRU, LFU, TTL caches |
| [Replication](replication.md) | `stores/replication/` | Cross-region sync |
| [Observability](observability.md) | `stores/observability/` | Audit, metrics, tracing |

### Composing Features

```python
from truthound.stores import get_store
from truthound.stores.versioning import VersionedStore, VersioningConfig
from truthound.stores.caching import CachedStore, CacheConfig, CacheMode
from truthound.stores.observability import ObservableStore
from truthound.stores.observability.config import (
    ObservabilityConfig,
    AuditConfig,
    MetricsConfig,
    TracingConfig,
)

# Base store
base = get_store("s3", bucket="my-bucket", prefix="results/")

# Add versioning
versioned = VersionedStore(base, VersioningConfig(max_versions=10))

# Add caching
from truthound.stores.caching.backends import LRUCache
cache = LRUCache(max_size=1000, ttl_seconds=3600)
cached = CachedStore(versioned, cache, mode=CacheMode.READ_WRITE)

# Add observability
store = ObservableStore(
    cached,
    ObservabilityConfig(
        audit=AuditConfig(enabled=True),
        metrics=MetricsConfig(enabled=True),
        tracing=TracingConfig(enabled=True),
    ),
)
```

## Error Handling

All stores raise consistent exceptions:

```python
from truthound.stores.base import (
    StoreError,           # Base exception
    StoreNotFoundError,   # Item not found
    StoreConnectionError, # Connection failed
    StoreWriteError,      # Write operation failed
    StoreReadError,       # Read operation failed
)

try:
    result = store.get("nonexistent-id")
except StoreNotFoundError as e:
    print(f"Result not found: {e.identifier}")
except StoreConnectionError as e:
    print(f"Connection failed to {e.backend}: {e}")
```

## Configuration Reference

### Base Configuration

All stores extend `StoreConfig`:

```python
from truthound.stores.base import StoreConfig

config = StoreConfig(
    namespace="production",    # Logical grouping
    prefix="validations/",     # Path/key prefix
)
```

### Compression

Most stores support gzip compression:

```python
# FileSystem
FileSystemConfig(use_compression=True)

# S3
S3Config(use_compression=True)

# Azure
AzureBlobConfig(use_compression=True)
```

## Architecture

```
stores/
├── backends/               # Storage backend implementations
│   ├── filesystem.py       # Local filesystem
│   ├── memory.py           # In-memory (testing)
│   ├── s3.py               # AWS S3
│   ├── gcs.py              # Google Cloud Storage
│   ├── azure_blob.py       # Azure Blob Storage
│   └── database.py         # SQL databases
│
├── versioning/             # Result versioning
├── retention/              # TTL and retention policies
├── tiering/                # Hot/Warm/Cold/Archive
├── caching/                # Result caching layer
├── replication/            # Cross-region replication
├── observability/          # Audit, metrics, tracing
├── batching/               # Batch write optimization
└── backpressure/           # Flow control
```

## Next Steps

- [FileSystem Store](filesystem.md) - Local file storage
- [Cloud Storage](cloud-storage.md) - S3, GCS, Azure Blob
- [Versioning](versioning.md) - Version control for results
- [Caching](caching.md) - In-memory caching layer


## Bounded iteration and MCP

`BaseStore.iter_query(query, batch_size)` requires a positive batch size. It honors the total `query.limit` across pages, including zero. `VersionedStore.config` and `RetentionStore.config` exposes its wrapped backend configuration. The opt-in [Core MCP profile](../mcp.md#native-execution-engines-and-general-stores) provides approved native store operations with factory-owned backend settings.

### Native schema migration recovery

`MigratableStore` now persists its version in `ValidationResult.metadata`, including after lazy or batch migration and reopening. Required backup and validation failures prevent rewriting the original. Lazy write failures propagate, and query no longer silently omits failed migrations. Generated backup originals remain readable without migration and are skipped by subsequent migration passes; backup ID conflicts fail closed. Status reports sampled failures explicitly. Configure positive `batch_size` and `max_retries`, and a current version at least as new as the minimum. The optional trusted `set_io_guard` runs before migration callbacks and backend effects. Core MCP exposes native status, per-record path validation and preview/actual migration through approved store bindings; callbacks and paths stay operator-owned. Multi-record migration can partially commit and provides no whole-store rollback.

### Observable store lifecycle

`ObservableStore.initialize()` uses its wrapped backend configuration and initializes once through `BaseStore`. Save and query observers count each operation once; `StoreMetrics.record_save` and `record_query` retain their standalone behavior, with `record_operation=False` available when an enclosing observer already counts it. Backend close failures still run observability shutdown. Audit, tracer and owned metrics-server shutdown are all attempted, and a stopped metrics HTTP server releases its socket. Core MCP supports private metric aliases, projected native audit queries and flush through approved store bindings; physical IDs, labels, actor data and error text stay private. Flush completion alone does not establish remote delivery.

### Concurrent commit failures

Concurrent filesystem operations hold data locks through index changes. Index commit refreshes disk state under the process lock and rejects stale conflicting additions. Failed index writes are not acknowledged and the uncertain owner cannot continue; corrupt index/WAL evidence is retained. WAL recovery preserves prior committed entries. Lock cycle checks distinguish actual incompatible waits from normal contention.

`with store.batch() as batch` requires an explicit `commit`; exit otherwise rolls back the buffer. Pre-index failure restores touched files and stops later actions. Index commit uncertainty or incomplete restoration retains private backup hashes and a manifest and quarantines both the instance and reopening. Use `list_batch_recovery(limit=100,offset=0)` and `reconcile_batch(recovery_id,resolution="rollback"|"keep_current",expected_fingerprint=...)` for explicit native recovery. Original hashes and current file fingerprints are verified under the batch-owner and affected-file locks. Rollback restores original bytes; keep-current only indexes actual partial files. Completed recovery/commit state moves to `.batch-resolved-*` before deletion, so a cleanup crash leaves usable data. `cleanup_batch_recovery(recovery_id,expected_fingerprint=...)` handles retained completed evidence. Recovery requires real native process locking and fails on stale/corrupt/traversing evidence. Preparing receipts precede all file effects; their explicit reconciliation leaves current files intact. This is not automatic multi-file crash atomicity or power-loss proof. Core MCP exposes bounded preview/commit, aggregate lock statistics and native index rebuild through approved store aliases.

Portable store locking is available with `pip install 'truthound[stores-concurrency]'` (`filelock` and `portalocker`); the MCP extra also includes filelock for its default cross-platform recovery provider. Native nonblocking/zero timeouts do not become infinite waits. Portalocker uses its supported lock signature and independent descriptors for simultaneous shared readers. Failed opens release local ownership, and fcntl lock files retain stable inodes across handoff; do not unlink them while other processes may be waiting. These provider and crash contracts run on actual local processes; full Windows verification remains a final CI gate.


### Native streaming storage

`StreamingFileSystemStore`, `StreamingDatabaseStore` and `StreamingS3Store` are streaming validation stores, separate from `BaseStore`. Stores own directly created writers and readers and attempt every owned cleanup even after a failure. Failed sessions stay failed. Failed writes preserve buffered records and do not acknowledge a phantom chunk. Backend retries require an explicit idempotent-write implementation; an uncertain append is not automatically replayed. Checkpointed sessions retain base stream settings and datetime metrics; backend endpoints and credentials remain constructor-owned.

Filesystem manifests use a flushed private sibling followed by replacement, and chunks publish without replacing existing chunk evidence. Run identifiers and filesystem reads must remain in the selected directory. Resuming rejects unlisted, missing, corrupt or inconsistent chunks before appending. Missing/corrupt manifests are retained for explicit investigation rather than inferred from estimated record counts. This does not establish whole-stream atomicity or power-loss durability. SQLite checkpoints persist the session after chunk commit; an interruption between those transactions is detected on resume and requires reconciliation. New and reopened databases enforce one session per run. CSV chunks include an additional `__truthound_record_v1__` payload column (with a collision-safe suffix when needed) to restore JSON record types and nested values; legacy plain CSV without the reserved payload marker remains string-valued.

`finalize_result` drains the store-owned writer and refuses failed or aborted sessions. It produces the persistence `ValidationResult`, including the sum of issue counts. Finalization materializes its result list in memory; use readers for chunked consumption. Filesystem and actual local SQLite behavior have focused tests; S3 protocol tests use a local synthetic client and do not verify a real cloud service.

Install optional codecs and portable file locking with `pip install 'truthound[stores-streaming]'`. Parquet uses Core Polars to produce independent files with a `__truthound_record_v1__` JSON payload column, preserving empty and heterogeneous nested records. Snappy reuses the existing Core compression provider. `BufferedStreamWriter` maintains one document on a seekable binary output instead of concatenating CSV/Parquet/compression frames; it applies the configured memory bound to the full document. Caller-supplied outputs remain caller-owned; `get_output` requires memory output. `writer.abort()` stops without flushing and cannot undo I/O already started. Filesystem and file-based SQLite writers use real native process locks; this does not establish power-loss durability or remote-service fencing.

Filesystem `recovery_status(run_id)` hashes the current manifest/chunk bytes while no writer owns the run. `reconcile_stream(run_id, resolution="adopt"|"discard", expected_fingerprint=...)` rechecks that fingerprint under the same native lock and explicitly adopts or discards only unconfirmed chunks, preserving the confirmed prefix. Adoption requires contiguous indexes, the persisted codec and actual validator records. The resulting session is paused and needs a separate `resume_session`. Changed fingerprints, corrupt/missing original manifests or confirmed chunks prevent mutation. Multi-file discard can partially fail; reconciliation is not a whole-operation atomic transaction. Uncertain chunks are never reconciled automatically. S3 uses the explicit optimistic protocol described below.

File SQLite exposes the same `recovery_status`/`reconcile_stream` API, holding its run lease and one `BEGIN IMMEDIATE` transaction for row deletion plus session updates. Inspection exceeding the session `max_memory_mb` bound fails without mutation. New DB chunks persist a `db-json-sha256:` checksum of canonical stored-record JSON, detecting same-count row changes before reading or resuming. Legacy DB chunks retain count-only checks; no automatic checksum backfill claims historical integrity. Remote DB/S3 fencing and recovery parity remain unverified.

S3 streaming requires a client/provider supporting conditional requests. New chunks use `IfNoneMatch="*"`; manifests use the retained opaque ETag with `IfMatch`, and `HeadObject` checks the receipt before further writes. Missing/stale receipts or inconsistent chunk inventories fail without falling back to unconditional overwrite. Multipart cleanup aborts only known owned uploads; missing part/completion receipts never acknowledge records. Execution plus abort failures are both retained, and cleanup attempts every owned upload. These are local protocol tests, not live S3 verification, multi-object atomicity or an exclusive remote lease. Request contracts follow AWS [PutObject](https://docs.aws.amazon.com/AmazonS3/latest/API/API_PutObject.html), [HeadObject](https://docs.aws.amazon.com/AmazonS3/latest/API/API_HeadObject.html) and [CompleteMultipartUpload](https://docs.aws.amazon.com/AmazonS3/latest/API/API_CompleteMultipartUpload.html).

Memory SQLite shares its owned engine connection across async executor threads and serializes connection work for store-created writers/readers, preventing one transaction from committing or rolling back another writer. Filesystem and SQLite chunk deletion also checks actual writer ownership. Serialized/encoded chunks exceeding `max_memory_mb` are rejected before storage. This is not an OS sandbox guaranteeing a decoder’s maximum heap use.

Typed `iter_results`, `read_result`, `get_chunk`, finalization and chunk adoption require actual boolean verdicts and nonnegative integer counts. A legacy string `"False"` is never counted as a passing check, and string counts are not silently coerced. Original bytes remain intact; ordinary `reader.read`/`read_batch` can inspect raw records with invalid types.

Release requires the exact opaque `LockHandle` instance returned by acquisition. A released handle or reconstructed copy cannot unlock a new owner at the same path or a reused descriptor. Filelock uses a shared thread context for async executor cleanup; SDK unlock failure retains ownership so cleanup can retry the same resource. Actual local filelock/portalocker tests do not replace final Windows CI.

Streaming configuration validates actual booleans, integral positive sizes and finite nonnegative retry/timer values. A string such as `"False"` cannot enable checkpoints. MCP session/writer/reader/stats projections expose stored error counts without historical native error bodies; the original diagnostic state remains in operator-owned storage.

S3 also provides `recovery_status` and `reconcile_stream(adopt|discard)`. They recheck the selected run’s manifest/object SHA and ETag fingerprint, confirmed prefix, offsets, codec and counts. Adoption returns a paused session requiring separate resume. Discard first conditionally journals its intent in the manifest, then conditionally deletes only inspected object ETags. Partial failure or a missing response receipt blocks normal readers, writers and resume until fresh inspection and explicit discard. Whole-run `delete_chunks` removes the manifest last; partial deletion retains `pending_resolution=delete` and requires another explicit approved call. Its count includes current objects and the manifest confirmed by that call; it does not permanently purge historical versions/delete markers. Conditional header semantics follow AWS [DeleteObject](https://docs.aws.amazon.com/AmazonS3/latest/API/API_DeleteObject.html). This is not multi-object atomicity or an exclusive remote lease. Readers fail if their manifest receipt changes.

`truthound[s3,stores-streaming]` requires boto3 1.42.26+ with conditional API models. Actual offline botocore tests check ordinary/multipart write headers; a local synthetic object protocol checks concurrent uploads, partial discard, changed objects and recovery after worker replacement. Neither verifies a live S3 service. Multipart limits in-flight requests to `max_concurrency` and drains started requests before aborting an owned upload. SDK network calls remain subject to provider timeout configuration.

In this unreleased checkout, general `DatabaseStore` queries and updates are namespace-scoped. Fresh tables use unique `(namespace, run_id)` keys. Legacy tables with a global unique run ID remain readable; a cross-namespace collision fails without moving the existing record or silently rewriting the table schema. Such legacy schemas need operator-managed migration before duplicate run IDs can coexist. SQL errors from exists/list/query/count and corrupt result JSON are failures, not false/empty/zero success. Tags are filtered before offset/limit; ordering is explicit, and limit zero returns no rows. Closing a database store resets its initialization state so a file-backed store can reopen its own connection pool.

Pool metrics include the initial connection. Disposal attempts every owned Session and engine; failed cleanup retains ownership, blocks new Sessions and remains retryable instead of claiming disposal succeeded. Initial connection failure cleans up the owned engine. Unexpected health-check failures update failure metrics without logging native error bodies. These changes have actual local SQLite tests; remote SQL runtimes remain separately unverified.

`DatabaseStore(..., table_prefix="tenant_")` selects `tenant_validation_results`
and independently prefixed indexes/constraints. Empty prefix keeps the default
table. Prefixes are empty or simple ASCII identifiers beginning with a letter or
underscore, up to 24 characters. One store's initialization creates only its
selected table; namespaces remain independent inside each table. This does not
rename existing tables or migrate their records. Both pooled and legacy SQLite
stores preserve prefixed records across close/reopen. MCP factories own this
configuration; caller commands cannot supply table names. SQL initialization
failures use a fixed error message and close partially owned connections rather
than exposing SQL/provider details or leaving an initialized success state.


## Native codec and flow-control utilities

`CompressionPipeline` composes native transforms, compressors and checksums.
`EncryptionPipeline` composes compression, encryption and checksums with a
serialized `THEP` header. Algorithm, key and custom components belong to the
caller or approved factory. For example:

```python
from truthound.stores.encryption.base import EncryptionAlgorithm, generate_key
from truthound.stores.encryption.pipeline import EncryptionPipeline

key = generate_key(EncryptionAlgorithm.AES_256_GCM)
pipeline = EncryptionPipeline().add_compression("gzip").add_encryption(key=key)
encoded = pipeline.process_to_bytes(b"synthetic-data", max_output_bytes=1024 * 1024)
assert pipeline.reverse_from_bytes(encoded, max_output_bytes=1024 * 1024) == b"synthetic-data"
```

Encryption restore validates the complete selected stage count, order, types,
algorithms and header version before executing a stage. Truncated or malformed
headers raise `DecryptionError`; checksums and declared decompressed sizes are
verified. The header CRC detects corruption; it is not an authentication claim
for arbitrary header metadata. Use the selected native pipeline and key rather
than creating code from an input header. Compression and encryption pipelines
accept a positive `max_output_bytes` limit checked on input and after each native
stage. This bounds accepted stage results, not decoder allocation before return.
Execution guards run around each stage; revoked authorization stops later stages.
Provider error bodies do not become compression pipeline error messages.

Native backpressure strategies report metrics and control admission. A composite
raises its collected child failures instead of filtering them into successful
admission. Nonboolean admission results are invalid. Cancellation clears an
in-progress pause flag. Token-bucket reset restores its configured permits;
explicit bucket capacity must be finite and at least one token. These operations
are local flow control, not a distributed lock or external-service transaction.

The unreleased MCP Core profile exposes `store_utilities_open/execute` through
private `store_utilities` bindings. An approved installed factory yields
`StoreUtilityRuntime(compression={}, encryption={}, backpressure={}, payloads={},
slots={})`. Mappings use public aliases; codec values are native pipelines,
backpressure values are native strategies, initial payloads are bytes and slots
are fixed absolute operator paths. Factory code, providers, keys and physical
paths are never caller-selected wire values. Factories own component cleanup.

Commands are `inspect`, `payload_capture`, `payload_import`, `payload_export`,
`payload_release`, `payload_status`, `codec_process`, `codec_reverse`, `slot_save`,
`slot_load`, `slot_delete`, `pressure_status`, `pressure_update`,
`pressure_acquire`, `pressure_release`, `pressure_reset` and `shutdown`.
Capture reads the exact bytes of one pinned asset or artifact; asset bytes are
captured IPC, not an invented JSON representation. Codec commands select
`codec_kind="compression"|"encryption"`, `codec_id` and an owner-local
`payload_id`. Payload results provide opaque references, byte counts and SHA256;
export returns tagged binary data in a private artifact. Import accepts only that
export format and pins its source artifact. Use a separate owner when importing
an existing owner's descendant artifact; resident dependency cycles are denied.

Each owner retains at most 32 payloads and 16 MiB in total, also constrained by
the active resource budget. Reads check file size before materialization. Codec
stage outputs and retained bytes are bounded; this is not an OS memory sandbox.
Encryption output includes native header bytes and ciphertext, never the key.
Metric replies exclude provider configuration, custom stage names and errors.
Every operation needs exact approval and sensitive disclosure consent, including
plaintext export. Captured input revocation stops the owner and revokes derived
artifacts. Fixed slots persist independently; loading one needs new approval.

Slot save writes a private, flushed sibling and replaces only the fixed target.
It does not establish whole-operation power-loss durability or secure erasure.
Slot deletion removes the selected current file. Native flow-control errors
return failed `native_pressure_incomplete` artifacts and latch that strategy;
subsequent acquisitions do not rerun it. Explicit `pressure_reset` calls native
reset and clears the latch only after success, acknowledging that prior permit
consumption cannot be undone. Denial (`acquired=false`) is a valid decision.
Approved shutdown acknowledges owner and factory cleanup before replying.
Resource release alone does not prove a slow finalizer completed. Default
payloads and pressure state are in memory; actual external providers and remote
key stores remain separately unverified. See [MCP](../mcp.md).


### Key persistence and approved key use

Native `KeyManager.create_key` creates a new identity and rejects an existing one;
use explicit rotation to replace a current key. Rotation creates a protected
archive before publishing the new version. `FileKeyStore` publishes complete
private files through a flushed sibling; create-only publication does not replace
an existing identity. Failed replacement preserves the old record. Corrupt,
unreadable or mismatched records raise a fixed `EncryptionError`, rather than
looking absent to `get_or_create_key`. Explicit zero TTL expires immediately.
Expired cleanup counts confirmed deletions and fails on an unconfirmed receipt.
Failed unlink leaves the previous file intact. Unlink and Python object clearing
are not secure-erasure guarantees; archive/current rotation is not a transaction
across two files. A partial failure requires operator investigation, not blind
rotation retry. Generic key-store extensions define their own concurrency scope.
`EncryptionKey` repr excludes key material. Its explicit native export API remains
available to operator code and is not exposed over MCP.

`FileKeyStore(master_key=...)` requires a 32-byte master key; a master password
uses the persisted salt and native KDF. Existing native construction without a
master still stores plaintext key records in private files; select an encrypted
backend explicitly. Keys, passwords, salt files, metadata and file locations are
factory-owned and are not model inputs or returned key metadata.

`StoreUtilityRuntime.keys` maps aliases to `StoreKeyBinding(manager, native_id)`.
The manager must be a native `KeyManager`. Commands `key_status`, `key_history`,
`key_create`, `key_get_or_create`, `key_rotate`, `key_delete`, `key_apply` and
`key_cleanup_expired` require separate exact approval. Output contains only public
alias, algorithm/type, version, timestamps and expiry/uncertainty; it excludes key
bytes, physical IDs and arbitrary key metadata. `key_history(limit=...)` includes
only that fixed identity and its native version archives. `key_version` selects
an archive/current version for status, apply or deletion. Rotation defaults to
`archive_old=true`; explicitly disabling it changes retention of the old key.
Expired cleanup covers the entire factory-selected key manager, including owned
keys without public aliases, and that effect appears in approval.

`key_apply(key_id=..., codec_id=...)` selects only a bound encryption pipeline and
requires matching cipher algorithms. It never returns key material. Subsequent
codec use checks that the applied key still exists, is unexpired and matches the
captured version/material; rotation requires explicit reapplication, and an old
ciphertext can use its separately selected archive. Key audit callbacks and native
storage operations execute under the factory's authorization guard.

An unconfirmed key operation returns failed `native_key_incomplete` and latches
all aliases of the same manager in that owner. Reads can report current state,
but mutations and linked codec use do not automatically repeat. The latch is
owner-local, not a durable notification journal. Investigate the native backend
before opening a newly approved owner and choosing any further action. Local
installed tests use synthetic keys and an actual encrypted file backend, including
rotation, archive restore, fresh-owner reload and deletion. Remote key services
remain unverified.

### Authenticated streaming and chunked encryption

`StreamingEncryptor` now writes THSE v2. Each native AEAD record authenticates
the header, record kind and sequence, plus the caller's associated data. A final
empty authenticated record binds the observed chunk count and plaintext size.
`StreamingDecryptor.authenticated_complete` becomes true only after this record
and EOF are verified. Consuming an individual chunk authenticates that chunk;
consume the complete iterator before committing an all-stream result. Missing
completion, altered headers, truncated or trailing records, and wrong keys for
empty streams raise typed errors. Short reads/writes and nonseekable streaming
inputs are supported. Plaintext chunk size must be 1–16,777,216 bytes. Invalid
configuration is rejected before opening an output file.

Failed writes latch the writer; exceptional context exits abort without adding
a completion record. Aborting does not erase already written bytes or provide an
atomic replacement of an existing file. Caller-owned streams remain open; native
file handles close on normal exit, abort or initialization failure. Owner guards
run before native encryption/decryption and before accepting subsequent effects.

`ChunkedEncryptor` returns a v2 `ChunkIndex` with an AEAD authentication tag over
its canonical metadata. `ChunkedDecryptor` captures and validates that index
before exposing any plaintext, including for an empty payload. It verifies
sequential offsets, nonce derivation, sizes, bounds and chunk authentication.
`decrypt_all` rejects trailing payload bytes; a selected range authenticates the
index and selected chunks and does not verify unread chunks. Zero-length valid
ranges return empty bytes; negative indices and invalid ranges fail.

New readers reject v1 streaming data and unsigned legacy chunk indices by
default. Pass `allow_legacy=True` explicitly for controlled migration using the
original key/AAD; `authenticated_complete`/`authenticated_index` remain false.
Legacy data cannot prove completeness or metadata authenticity and should not
be silently upgraded to that claim. Reading and writing separate ciphertext and
index files is not a multi-file transaction. These native formats have not yet
been published as part of the ongoing MCP development change set.

`StoreUtilityRuntime` additionally binds native `adaptive`,
`chunked_compression` and `chunked_encryption` codec aliases. Select those values
in `codec_kind` for `codec_process`/`codec_reverse`. Native chunked encryption
bundles contain a v2 authenticated index and ciphertext; native chunked
compression bundles contain a checked header/index and checksums, which detect
corruption and do not provide cryptographic authenticity. `codec_range` accepts
`start`/`end` byte offsets for either chunked kind. All restored payloads share
the existing retained-byte budget; native declared sizes are checked before full
restore, and returned output is checked again. This does not bound a provider's
peak decoder heap allocation.

For `adaptive`, `codec_analyze` returns native numerical characteristics and
algorithm sample ratios/errors; `codec_recommend` returns algorithm/score/ratio/
speed fields. Raw samples, provider diagnostics and free-form recommendation
reasons are excluded. Factory-owned `enable_sampling`, `sample_size`, allowed
algorithms, fallback, level and analysis budget apply to native selection.
`max_analysis_time_ms` stops additional probes at checkpoints and records
`analysis_budget_exhausted`; it cannot interrupt a native probe already running.
Corrupt sizes, flags, algorithm indices, checksums and missing/extra iterator
chunks raise errors. Selected codec and bundle metadata must match. Incremental
codec sessions are described below.

### Incremental codec sessions

Factories can bind `compression_streams` using
`StoreCompressionStreamBinding(compressor_factory, decompressor_factory)` and
`encryption_streams` using
`StoreEncryptionStreamBinding(encryptor_factory, decryptor_factory)`.
Compression factories return fresh native `StreamingCompressor` and
`StreamingDecompressor` objects without arguments. Encryption factories receive
an owned binary buffer and return a native `StreamingEncryptor` or
`StreamingDecryptor` configured with operator-owned key/AAD/algorithm settings.
The binding chooses all code and secrets; callers supply only aliases.

`stream_open` selects `codec_id`, `codec_kind` (`compression` or `encryption`) and
`direction` (`encode` or `decode`). Encryption decoding additionally selects an
existing encrypted `payload_id`. `stream_write` consumes a retained payload;
`stream_flush`, `stream_read`, `stream_finalize`, `stream_abort`, `stream_status`
and `stream_release` select the returned opaque `stream_id`. All operations keep
the existing approvals, disclosure consent and parent grant lifetime. At most
eight native streams are retained; pending input/output buffers count against
the owner's byte budget. Encryption output buffers reject writes exceeding that
budget. Provider peak allocations are still outside this bound.

Compression `write`/`flush` outputs are independent complete native frames,
returned as a payload or null when still buffered. Feed each complete frame to
a decoding stream's `write`; arbitrary transport fragments are not accepted by
this native frame API. `stream_finalize` flushes the remaining frame. Encryption
encoding retains its output until finalization, then returns the completed THSE
v2 payload. Decoding `stream_read` returns one authenticated chunk; it does not
claim total completion. `stream_finalize` returns remaining plaintext and checks
the final authentication record/EOF. A zero-byte remaining payload is valid.

Results include state, finite metrics, buffered bytes and the current output
receipt. `authenticated_complete` applies to encrypted decoding after EOF.
Native failures return `native_codec_incomplete` and fence further processing;
explicit abort/release is required. Finalized handles cannot be written again.
Every selected native context exit runs once; unknown cleanup is reported by
`cleanup_uncertain` and is never automatically retried. Shutdown attempts all
stream cleanup and only acknowledges confirmed teardown. Revocation invalidates
the owner and derived artifacts. Closing an unfinished session aborts it without
automatically finalizing pending bytes. These sessions are owner-local and do
not claim process-loss continuation.

Native backpressure strategy construction rejects nonfinite timing/rate values,
boolean numeric settings, invalid windows and nonpositive sampling intervals.
`update_metrics` validates the whole observation before mutation: memory is a
finite 0–100 percentage, latency is finite/nonnegative, and queue depth is a
nonnegative integer. A rejected mixed observation leaves prior metrics intact.


### Durable S3 multipart ownership

Streaming S3 writes publish a conditional manifest intent before creating an
upload, then publish its exact upload receipt before submitting parts. A worker
loss during creation therefore retains an unresolved intent; a loss during part
upload retains the owned receipt. Normal open/resume/write, run deletion and chunk recovery
remain blocked while multipart intents exist. A failed multipart attempt cannot
automatically start another upload. Legacy manifests without intents cannot
prove that no historical unfinished uploads exist.

`recovery_status(run_id)` reports `multipart_uploads` and
`unresolved_upload_intents` alongside chunk evidence. Explicit
`reconcile_multipart(run_id, expected_fingerprint=..., receipts=...)` aborts exact
owned uploads and conditionally clears each journal entry after acknowledgement.
Missing abort receipts preserve the journal; a fresh inspection and explicit
recovery can acknowledge `NoSuchUpload` without claiming another abort occurred.
Results distinguish `resolved_uploads` from successful HTTP-204
`aborted_uploads`. New operator-supplied receipts are journaled before cleanup.
A missing creation receipt requires a private operator mapping from the exact
journaled object key to the independently verified upload ID. The mapping asserts
ownership; Core never guesses an ID or aborts unrelated uploads by listing the
bucket. A wrong assertion cannot establish what happened to the original upload.

After multipart cleanup, inspect/reconcile any unconfirmed completed chunks
before resuming. This remains an optimistic object protocol with native abort
acknowledgements, not an exclusive remote lease or a claim that remote in-flight
requests or provider billing have ceased. Quiesce the former owner before
recovery. Actual process-loss checks cover both creation and part stages using
persistent synthetic objects/uploads; botocore model checks remain distinct from
live cloud verification.
