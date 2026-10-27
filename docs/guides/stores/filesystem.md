# FileSystem & Memory Stores

`FileSystemStore` persists validation results in a selected local directory.
`MemoryStore` keeps canonical record mappings for one Python instance. Use
`ConcurrentFileSystemStore` or `ConcurrentFileSystemExpectationStore` from
`truthound.stores.backends.concurrent_filesystem` when threads or processes share
filesystem writes. Their initialization lock does not replace the native
inter-process lock provider; retain the default `auto` strategy for that use.

## FileSystem Store

### Basic Usage

```python
from datetime import datetime
from tempfile import TemporaryDirectory

from truthound.stores.backends.filesystem import FileSystemStore
from truthound.stores.results import ResultStatus, ValidationResult

with TemporaryDirectory() as directory:
    store = FileSystemStore(base_path=directory, compression=True)
    item = ValidationResult(
        run_id="run-123", run_time=datetime(2026, 1, 1),
        data_asset="customers", status=ResultStatus.SUCCESS,
    )
    store.save(item)
    assert store.get("run-123").data_asset == "customers"
```

Results live under `{base_path}/{namespace}/{prefix}`. The constructor defaults
are `.truthound/store`, `default` and `validations`. Namespace/prefix are relative
paths; traversal and drive paths fail. Run IDs must be single path components
and cannot be `_index`. Reads check the identity inside the file. Record and
index paths cannot be symbolic links.

### Using the Factory

```python
from truthound.stores import get_store

store = get_store("filesystem", base_path=".truthound/results")
```

### Configuration

Select individual constructor settings or a captured `FileSystemConfig`, without
mixing both forms. Unknown settings fail instead of being ignored. Config
metadata supplies defaults; each record's metadata takes precedence. Later edits
to the original config object do not change the store's captured settings.

```python
from truthound.stores.backends.filesystem import FileSystemConfig, FileSystemStore

config = FileSystemConfig(
    base_path=".truthound/results", namespace="production", prefix="validations",
    serialization_format="yaml", file_extension=".yaml",
    use_compression=True, metadata={"environment": "production"},
)
store = FileSystemStore(config=config)
```

| Option | Default | Behavior |
|--------|---------|----------|
| `serialization_format` | `json` | JSON or safe YAML; pickle is rejected |
| `file_extension` | `.json` | Filename suffix; constructors infer YAML from `.yaml`/`.yml` when format is omitted |
| `create_dirs` | `True` | Create the selected directory; false requires it to exist |
| `pretty_print` | `True` | Indent JSON output |
| `compression` | `None` | Selected native codec name, such as `gzip` or `zstd` |
| `use_compression` | `False` | Legacy flag; true selects gzip when no codec name is supplied |
| `metadata` | `{}` | Per-record metadata defaults; suite stores use `meta` |

The config dataclass inherits an empty `prefix`; set it explicitly when using a
config object. Constructor defaults remain `validations` and `expectations` for
the respective store types. YAML uses safe loading/dumping; unsupported metadata
values cause a write error. `file_extension` is a suffix, while an explicitly
selected `serialization_format` controls the content.

### Compression

`FileSystemStore(compression=True)` selects gzip and a `.gz` suffix. A codec name
such as `compression="zstd"` selects that actual native provider and `.zst`.
Other native codec names use their name as an additional suffix. The selected
provider must be installed. JSON/YAML parsing and decompression are automatic on
read; reopening requires the same format/codec configuration. Changing settings
in place is not a migration between formats.

### Directory Structure

```text
{base_path}/{namespace}/{prefix}/
    _index.json
    run-123.json
    run-456.yaml.gz
```

The example illustrates different configured stores; each store uses its own
selected suffix. Keep different configurations in separate directories.

### Index Management

Record files and `_index.json` are individually published by flushing a private
sibling, syncing it and atomically replacing the destination. They are separate
files, not a multi-file transaction. New index entries contain metadata and the
record's SHA-256; legacy entries without a hash remain readable when their
metadata agrees. Initialization validates record/index consistency, which reads
the stored records and costs more than loading the index alone.

Malformed records/indexes, missing records, identity mismatches and changed
content are errors. Query does not silently omit damaged records. An unconfirmed
record/index write fences later mutation in that owner. A fresh owner checks
actual consistency. Recovery is explicit:

```python
from truthound.stores.backends.filesystem import FileSystemStore

store = FileSystemStore(base_path=".truthound/results")
# Do not initialize first when the index itself is damaged.
indexed = store.rebuild_index()
```

`rebuild_index` validates every selected record before replacing the index. A
corrupt record stops recovery and preserves the old index. Repair or restore that
record through an operator-controlled process before trying again. Rebuilding
reconciles current files; it does not roll back a previously replaced record or
prove exactly-once side effects. These ordinary stores do not provide shared
writer coordination; use the concurrent backend for that workflow.

### Expectation Store

`FileSystemExpectationStore` supports the same config object, safe format,
compression and atomic publication. Suite names are encoded as filename
components, so `a/b` and `a_b` are distinct. Simple legacy names keep their paths.
Legacy slash-normalized files are read only when their stored suite identity
matches; a conflicting legacy filename requires explicit migration.

```python
from truthound.stores.backends.filesystem import FileSystemExpectationStore
from truthound.stores.expectations import ExpectationSuite

store = FileSystemExpectationStore(
    base_path=".truthound/suites", use_compression=True,
)
store.save(ExpectationSuite.create("customer_suite", "customers"))
assert store.get("customer_suite").data_asset == "customers"
```

## Memory Store

### Basic Usage

```python
from datetime import datetime
from truthound.stores.backends.memory import MemoryStore
from truthound.stores.results import ResultStatus, ValidationResult

store = MemoryStore(max_items=1)
for identity in ("first", "second"):
    store.save(ValidationResult(identity, datetime(2026, 1, 1), "customers", ResultStatus.SUCCESS))
assert store.list_ids() == ["second"]
```

### Configuration

`MemoryConfig(max_items=0, deep_copy=True, metadata={})` can be passed with
`MemoryStore(config=config)` or `MemoryExpectationStore(config=config)`. Zero
capacity means unlimited; negative/boolean capacities and nonboolean copy flags
fail. Unknown settings fail. Memory stores retain canonical mappings and reject
byte compression or non-JSON serialization settings; they do not pickle data.
Each instance owns its data; closing a context does not persist it elsewhere.

### Memory Management

At capacity, insertion of a new ID evicts the existing record with the earliest
run time (suite stores use update time, with ID as a tie breaker). Updating an
existing ID never evicts an unrelated record. Operations use an instance lock;
this does not share state between processes. `clear_all()` explicitly removes
all stored entries.

### Deep Copy Behavior

With `deep_copy=True`, record bodies and filter metadata use the captured copy.
Changing the original or a retrieved record's nested tags cannot alter later
queries. With `deep_copy=False`, reconstructed results can share nested values
with the stored mapping; editing those values can change subsequent reads.
The returned result object itself is reconstructed, so this is not a promise
that assigning any top-level attribute mutates the stored mapping.

### Expectation Store

`MemoryExpectationStore` provides the same `max_items`, config, copy, metadata,
filter and pagination behavior for named `ExpectationSuite` values. Its data is
also local to the instance lifetime.

## Common Operations

Both validation stores provide save/get/exists/delete/list/query and inherited
history/latest helpers. `StoreQuery` applies data-asset/status/tags/time filters,
selected `order_by` (`run_time`, `data_asset`, `status`), direction, offset and
limit. `limit=0` returns no records; invalid pagination or ordering fails. Suite
queries apply their available metadata, selected ordering (`run_time` uses `updated_at`, `data_asset`, or `name`), direction, and pagination. With a query, the default is newest-updated first; without a query, suite names remain alphabetical. JSON records and indexes reject duplicate keys and nonfinite numbers, including during explicit index rebuild.

## Error Handling

`StoreNotFoundError` identifies an absent record. `StoreReadError` reports invalid
filesystem data/index state, and `StoreWriteError` reports failed publication or
required reconciliation. Preserve failed files and investigate unconfirmed
writes; error returns do not prove that a preceding file replacement was absent.
In the unreleased MCP Core profile, approved `StoreRuntime` factories own these
settings. `rebuild_index` selects the native recovery operation; a new memory
owner has no previous owner's records. Sensitive values remain subject to the
binding's disclosure policy.

## Choosing Between Stores

Use memory for temporary work/tests, filesystem for persistent single-writer
records, and concurrent filesystem stores for coordinated shared writes.

## Next Steps

- [Cloud Storage](cloud-storage.md)
- [Versioning](versioning.md)
- [Caching](caching.md)

JSON and safe YAML both reject duplicate mapping keys and nonfinite numbers. Publication rejects cyclic container metadata and nesting deeper than 64 levels before replacing record bytes. Read failures retain the original files for investigation; explicit rebuild does not adopt ambiguous records. YAML merge keys that produce duplicate mappings are rejected as well.

## Concurrent filesystem settings

Both concurrent stores capture nested concurrency settings and metadata. Unknown
constructor options and invalid mode flags, paths or lock timeouts fail instead
of being ignored. The `config` property returns a copy. `create_dirs=False`
requires the complete selected directory to exist; initialization does not create
it indirectly through the index. The expectation store now honors the same
`auto`, `fcntl`, `filelock`, `portalocker` or `none` strategy as validation records.
`none` intentionally provides no cross-process exclusion.

Both stores share the ordinary filesystem JSON/safe-YAML and compression codec.
For YAML/gzip, select `serialization_format="yaml"`, `file_extension=".yaml"`
and `use_compression=True` for expectations or `compression=True` for validation
records. Registered codec names can be selected with `compression`. Metadata
provides record defaults (`metadata` for validation results, `meta` for suites);
item values take precedence. Nonfinite/cyclic values fail before record replacement.
Duplicate JSON/YAML keys and malformed records remain errors on reads and queries.
Index rebuild aborts on an unreadable record and preserves the prior index;
index and WAL JSON reject duplicate keys and nonfinite numbers.

Names are percent-encoded into individual filenames, so `a/b`, `a_b` and `a%b`
do not collide or escape the selected directory. Internal index, batch-owner and
recovery name prefixes are reserved; record symlinks are rejected. Legacy nested
validation filenames and underscore-substituted suite filenames require explicit
migration to the encoded filename after verifying the embedded record identity;
they are not guessed or silently adopted. Standard unchanged names keep their
filenames. Existing files remain untouched until an explicit operation changes them.

Queries apply sorting and pagination once, and `limit=0` returns no records.
Validation ordering supports `run_time`, `data_asset`, `status`; suite ordering
supports `run_time` (updated time), `data_asset`, `name`. Suite indexes written
before updated-time metadata use their recorded creation time until resaved.
Concurrent record/index publication remains separate: these settings do not turn
multi-file batches into crash-atomic transactions. Use the explicit batch recovery
workflow for uncertain commits.


All index operations are validated before the first WAL write. Invalid later metadata cannot leave an earlier operation as a recoverable prefix. Physical write failures still require explicit uncertain-outcome handling.

Concurrent index and WAL paths must be regular files within the selected directory; existing and dangling symlinks are rejected. On platforms providing `O_NOFOLLOW`, WAL append uses it as well. Locking coordinates cooperating writers; it does not grant authority to rewrite an unrelated path.
