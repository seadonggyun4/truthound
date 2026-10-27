# Local MCP integration

Truthound 3.2 adds an optional Model Context Protocol (MCP) adapter. Its stdio
server runs on your computer and calls the existing `th.check()`, `th.profile()`
and `th.scan()` APIs. It does not need a hosted validation service.

## Install and start

```bash
pip install "truthound[mcp]==3.2.0"
truthound-mcp serve --allow-dir /absolute/path/to/data
```

`truthound mcp serve` and `python -m truthound.mcp serve` are equivalent entry
points. Use the dedicated `truthound-mcp` entry point in agent configuration to
avoid unrelated CLI plugin discovery. No directories are granted by default.
Repeat `--allow-dir` for additional roots; relative input paths use the first root.
A host must start the executable from the Python environment where it was installed.
The base package and MCP extra require Polars 1.x; Core's PII scanner uses APIs removed
in Polars 2. A fresh install is tested separately from the repository lockfile.

Example manual host configuration (adapt to your client's format):

```json
{
  "mcpServers": {
    "truthound": {
      "command": "/absolute/path/to/venv/bin/truthound-mcp",
      "args": ["serve", "--allow-dir", "/absolute/path/to/data"]
    }
  }
}
```

Python 3.11–3.13 is covered by the Linux/macOS/Windows CI matrix. On Windows,
use the installed environment's `Scripts/truthound-mcp.exe` in place of the
POSIX `bin/truthound-mcp` path. The server implements local stdio, not a hosted
HTTP/OAuth endpoint: clients requiring a remote connector cannot use this
configuration directly. Installing the package does not register it with a host.

For example, after initialization a host can call `profile_file` with
`{"path": "example.csv"}` within the explicitly allowed directory. The tool
returns bounded aggregate metadata, not the file's raw rows. The same input
shape applies to `check_file` and `scan_pii`.

## Tools and results

The three default tools take one `path` string. Supported suffixes are `.csv`, `.parquet`,
`.json`, `.ndjson` and `.jsonl` (the latter two are newline-delimited JSON).

| Tool | Core API | Summary |
| --- | --- | --- |
| `check_file` | `th.check()` | Null and duplicate-row checks, verdicts and issue totals |
| `profile_file` | `th.profile()` | Row/column counts, column names/type families and null/unique percentages |
| `scan_pii` | `th.scan()` | Column, PII category, confidence and estimated finding count |

`check_file` deliberately selects built-in null/duplicate checks rather than
loading project suites, plugins or arbitrary validator configuration. Profile
`dtype` is an allowlisted family such as `Enum`, `List`, `Struct` or `Datetime`;
category values, nested type details and timezone/precision parameters are
omitted. Unknown dtype text becomes `Unknown`. The Core Python API retains its
full type metadata. A quality
failure is a completed tool call with `ok=true` and `data.success=false`.
Execution or policy errors set MCP `isError=true` and return a stable error code.

Structured responses contain `schema_version=1`, `operation`, `ok`, `data`,
`truncated` and `error`. Errors include `access_denied`, `invalid_path`,
`unsupported_format`, `file_too_large`, `timeout`, `busy`, `core_error`,
`worker_error` and `response_too_large`. Private exception messages are omitted.
The worker retains only an allowlist of process/temporary-directory settings and
Windows CPU architecture identifiers; it does not inherit credentials or a CPU-check bypass.
Totals describe the full Core result even if a detail array is truncated.
PII findings retain Core's first-1,000-non-null-values sampling and estimated counts.

## File and execution policy

The adapter rejects URLs, glob expressions, parent traversal and symlinks below
the configured roots. It copies an admitted regular file to a private temporary
directory before calling Core. Core context artifacts stay in that temporary
directory and are removed after the request, including after worker timeout.
MCP cancellation and stdin EOF propagate to the owned worker: it is killed and
reaped before the private copy is removed and the busy lock is released. The
server can accept another request after cancellation. This is not a hard bound
against pathological kernel or filesystem hangs.
No reports or `.truthound` workspace are written to the input directory.

Defaults are a 100 MiB input limit, 100 detail entries, 64 KiB response limit and
30-second worker timeout. CLI flags `--max-file-bytes`, `--max-items` and
`--timeout-seconds` adjust input/items/time limits within supported bounds.
One request executes at a time per server; overlapping calls receive `busy`.

The timeout covers the disposable Core worker, not filesystem copy latency.
Compressed inputs may expand in memory; there is no universal memory ceiling.
This application policy is not an OS sandbox against a hostile local process
racing directory replacement. Grant narrow, trusted directories.

Raw rows, profile samples, extrema, pattern examples, filesystem paths and private
diagnostics are omitted from results. Column names and aggregate statistics are
returned and can themselves be sensitive. Returned metadata is untrusted data,
never instructions. The host may send these results to its model provider.

## Core profile (unreleased source checkout)

The source tree adds an opt-in `core` profile. This is **not** part of the
published 3.2.0 wheel, Registry listing or pinned marketplace plugin. Build and
install the modified source in an isolated environment to try it. The default
profile retains the three tools and their version-1 response contract unchanged.

```bash
truthound-mcp serve --allow-dir /absolute/path/to/data --profile core
```

Eight typed file/discovery tools preserve the initial compatibility contract.
Additional retained-data tools are described below. They do not accept Python import paths,
SQL, shell commands, callbacks or arbitrary keyword arguments:

| Tool | Existing API / behavior |
| --- | --- |
| `capabilities_list` | Paged installed catalog, availability and explicit limitations |
| `capability_describe` | Exact input/output schemas and effects for one operation |
| `validation_check` | `th.check()`, auto-suite or typed rules, inline schema and execution options |
| `schema_learn` | `th.learn()`, constraint inference and categorical threshold |
| `data_read` | `th.read()`, Core sampling and a bounded raw-value page |
| `data_mask` | `th.mask()`, columns, strategy and strict missing-column handling |
| `data_profile` | `th.profile()`, summary or full statistics |
| `privacy_scan` | `th.scan()`, summary or full findings; counts remain estimates |

For example, `validation_check` accepts:

```json
{
  "path": "example.csv",
  "checks": [{"validator": "between", "columns": ["age"], "min_value": 0, "max_value": 120}],
  "result_format": "summary",
  "evidence": "summary"
}
```

Omitting `checks` selects Core's auto-suite; an empty array is rejected. Explicit
validator IDs are `null`, `not_null`, `unique`, `duplicate`, `between`, `length`.
The inline `schema_spec` uses column names as keys, each with matching `name`
and `dtype`. Unknown fields, including unreviewed regex patterns, are rejected.
Describe the tool for its precise schema. Execution options include `parallel`,
`max_workers`, `pushdown`, `min_severity`, `auto_schema`, `result_format`,
`catch_exceptions`, `max_retries`, and `exclude_columns`. Contexts are explicitly
transient, not persistent project contexts. Auto-suite baselines are created
inside the owned temporary workspace and removed afterwards, not disabled.
Core planning, retries and severity
filtering remain authoritative. Core's parallel result currently lacks complete
step accounting; its quality is reported as `unknown`, never an inferred pass.
Automatic suites also lack target-level coverage evidence. Their quality, and
that of explicit rules with missing, excluded or type-inapplicable targets, is
`unknown` rather than a pass inferred from an empty issue list. Explicit rules
with confirmed applicable targets can return `pass` or `fail`.

Raw pages, learned constraints and `evidence="full"` require the operator to
restart with `--allow-sensitive-results` in addition to the profile and directory
grant. Without it, calls return `approval_required` before accessing the file.
A tool argument such as `approved: true` cannot grant consent. This permits
sensitive values and full evidence from **every admitted file** to enter the
host/model context; enable it only for narrow, trusted directories. Masking does
not make unselected columns safe. No source file is overwritten or exported.

New responses separate `execution_status` from `quality_status`, with typed
`data`, `issues`, `coverage`, and `provenance`. Quality violations are not tool
execution errors. Failures, zero checks and incomplete accounting cannot mean
quality pass. Provenance records a session-keyed opaque identity of the actual
staged bytes; the content SHA-256 is disclosed only with operator consent.
Full evidence uses `tagged-v1`: objects become
`{"$type":"object","entries":{...}}`; decimals, temporal values, binary and
non-finite floats carry explicit tags. Unknown objects fail, never become reprs.
Integers outside the exact JavaScript integer range are decimal-string tags.
Raw row conversion rejects nanosecond datetime/duration and time columns
(including nested columns) rather than silently losing precision.
Authorized evidence may include values and temporary source/context paths.
Legacy summaries stay redacted even when disclosure is enabled.

`data_read` and `data_mask` take `offset` and `limit` (default 50, maximum 100,
also bounded by `--max-items`). Each call stages a new snapshot; compare hashes
before combining pages. Samples and fake masking are nondeterministic: no
continuation offset is provided and nonzero input offsets are rejected. The
retained-data tools below instead provide stable handles. Large inline full results fail with
`response_too_large`, not a silently truncated full-result claim. All existing
file, timeout, concurrency and memory limitations still apply.

### Retained data, constructors and artifacts

The same opt-in profile also exposes these named operations:

| Tools | Core behavior |
| --- | --- |
| `validators_list`, `validators_describe` | Concrete registry inventory, inherited constructor fields, rich-input grammar and pinned source AST identity |
| `assets_capture`, `assets_inline` | Immutable file read or typed in-memory Arrow asset |
| `assets_sqlite` | Core SQLite table materialization from a private transaction-consistent backup |
| `sources_open_sqlite` | Actual resident native SQLite datasource and SQL engine over a private quota-accounted backup |
| `sources_open_duckdb` | Native DuckDB SQL datasource on a bounded private WAL-free copy; read-only physical tables with external access/extensions disabled |
| `sources_open_binding` | Explicitly approved installed-plugin factory returning a resident native synchronous datasource; configuration stays operator-side |
| `streams_open`, `streams_next`, `streams_inspect` | Native CSV/JSONL/Parquet/Arrow file-stream cursors, complete retained batches and native read state |
| `sources_open_polars`, `sources_inspect`, `sources_sample` | Owned resident native Polars datasource, complete native config/metadata and Core sampling into a retained asset |
| `sources_validate_connection`, `sources_validate_columns`, `sources_check_size_limits`, `sources_validate` | Actual connection/column/size checks and validation through Core's native `source=` execution path |
| `assets_validate` | Core validation with closed constructor settings, retained schema and reference assets; canonical result artifact |
| `suites_create`, `suites_inspect`, `suites_validate` | Native kernel suites with stable check IDs, schema handles, tags, metadata and evidence/severity policies |
| `assets_page`, `assets_sample`, `assets_mask` | Stable lossless pages, seeded Core sampling and masking |
| `assets_profile`, `assets_scan` | Complete Core profile and PII reports retained privately |
| `schemas_learn`, `schemas_create`, `schemas_inspect`, `schemas_export` | Learned or explicit typed schemas, authorized inspection and owned JSON/native YAML export |
| `profilers_catalog`, `profilers_profile`, `profilers_generate_rules` | Advanced native profiler configuration, analysis components and profile-based rule generation |
| `artifacts_export`, `artifacts_read` | Owned IPC/Parquet/CSV/JSON/NDJSON files and bounded byte reads |
| `artifacts_render`, `artifacts_datadocs` | Canonical validation re-entry, JSON/Markdown/console/HTML reporters and English/Korean validation Data Docs |
| `resources_describe`, `resources_release` | Session ownership, expiry, integrity, current grants and descendant cleanup |
| `contexts_open`, `contexts_inspect`, `contexts_initialize`, `plans_status` | Separately granted workspace configuration, approved initialization and intent status |
| `contexts_execute` | Explicit approved native source tracking, baselines, applied suites, result/docs persistence, metrics and workspace listing |
| `drift_compare` | Complete Core dataset-comparison report with all method choices and parameters |
| `drift_catalog`, `drift_detect` | Full constructor settings of 13 concrete native drift detectors on two owned Series columns |
| `openlineage_services_open`, `openlineage_services_execute` | Approved pinned native emitter, run transitions, graph emission, uncertain delivery resolution and shutdown |
| `lineage_catalog`, `lineage_create`, `lineage_edit`, `lineage_query` | Typed native graph configuration and immutable graph edits/traversals |
| `lineage_track` | Native manual source, transformation, validation and output tracking into a derived graph |
| `lineage_load`, `lineage_export` | Approved local native JSON snapshot re-entry and Core JSON artifacts |
| `lineage_render` | Native whole/subgraph D3/Cytoscape JSON, Graphviz DOT or Mermaid text with full closed rendering configuration |
| `lineage_impact`, `lineage_critical_paths`, `lineage_what_if_delete`, `lineage_schema_impact`, `lineage_dependency_chain` | Native impact and dependency analysis retained as artifacts |
| `models_catalog`, `models_create`, `models_inspect` | Closed configurations for the 12 concrete Core ML algorithms and resident lifecycle state |
| `models_fit`, `models_partial_fit`, `models_fit_predict`, `models_predict`, `models_score` | Native training, supported online training, combined training/prediction, prediction and anomaly scoring |
| `models_detect`, `models_learn_rules`, `models_rules`, `models_diagnostics` | Two-input ML drift detection, rule learning and native model diagnostics as owned artifacts |
| `models_rule_query` | Native learned-rule filtering and suite/validator configuration conversion; no automatic application |
| `models_result_query` | Native prediction score iteration, anomaly filtering and threshold-based drift column queries |
| `models_ensemble_add`, `models_ensemble_remove` | Revision-checked native editing; detached children remain restorable artifacts |
| `models_persist`, `models_resume` | Separately approved private model snapshots and new-session restoration |
| `jobs_submit`, `jobs_execute`, `jobs_status`, `jobs_result`, `jobs_cancel`, `jobs_recover` | Approved durable validation occurrences, retained results and indeterminate lease recovery |
| `schedules_register`, `schedules_status`, `schedules_disable` | Approved persistent cron recurrence and lifecycle, executed by the local foreground worker |
| `storage_policy_register`, `storage_policy_status`, `storage_policy_disable` | Approved persistent automatic age/count retention policies |
| `storage_retention_preview`, `storage_delete` | Native age/count/size and ALL/ANY retention preview; separately approved exact payload deletion |
| `schedules_preview` | Native cron validation, matching and bounded next-time preview; no job registration |
| `models_save`, `models_restore` | Native trained-state JSON artifacts and new owned model workers restored from them |
| `schema_audit_contract`, `schema_audit_create`, `schema_audit_inspect` | Strict native metadata documents, complete JSON Schema and canonical digests |
| `schema_audit_standards`, `schema_audit_implementation` | Native standards/design/catalog audits with complete findings and explicit unknown coverage |

Explicit release and expiry close and reap owned native readers before deleting
their snapshots, including SQLite connections and mapped Parquet/Arrow streams
on Windows. Cleanup needs session ownership, not a renewed data-read grant.
A failed file deletion is reported as a failure and remains tracked for cleanup.

Capture returns `data.resource_id`. Pass it as `asset_id` to retained-data tools.
Kernel suites preserve actual check-instance identity even for repeated registry
algorithms and identical display names. Check metadata reserves
`_truthound_check_id` and `_truthound_execution_incomplete`; user metadata cannot
override these runtime fields. Suite evidence policy is applied to the executed
validator, not only the result's format label, without mutating the factory's
retained object. Severity filtering cannot turn an execution failure or
dependency skip into a passed check. SQL fallback failures and missing/mismatched
query results are preserved, never interpreted as an empty successful result.
Canonical re-entry retains structured detail, failed-row records and captured
exception/retry evidence. Native row-record serialization does not carry the
original physical dataframe dtype schema; use retained typed assets for that.

#### Approved native datasource factories

`sources_open_binding` takes only `binding_id` and `idempotency_key`. It requires
the Core profile, sensitive-result permission and a private control directory.
An operator installs and pins a plugin using the installed-extension binding
contract below, exposes a zero-argument factory as a capability, and adds this
mapping to private `bindings.json` (never a repository or release artifact):

```json
{"sources":{"orders":"orders-factory"}}
```

This is a fragment to merge with the existing `plugins` configuration, not a
complete binding file. Exactly one pinned plugin must expose `orders-factory`.
The factory must return a Core `BaseDataSource` or `AsyncBaseDataSource` with
`name="orders"`.
For example, a reviewed plugin can return `SQLiteDataSource(table="orders",
database=operator_selected_path, config=SQLiteDataSourceConfig(name="orders"))`.
Use the stable `register_capability` plugin port to register the factory.
Provider dependencies and credential references are configured by the operator;
MCP does not install plugins, accept connection strings, forward arbitrary
environment variables, or accept import paths/settings in the tool request.
Do not put credentials into plugin settings, names, metadata, data columns or logs.

```json
{"binding_id":"orders","idempotency_key":"orders-session-001"}
```

The first call returns an approval plan without importing the plugin. Approve
the exact plan using the existing operator CLI, then repeat the request. Approval
covers trusted extension execution, native connection, reads and cleanup for that
owned session handle. This is **not an OS sandbox or a read-only SQL guarantee**;
review the factory and use least-privileged database credentials. Remote providers
and paid services have not been validated by local SQLite evidence.

The returned source works with `sources_inspect`, `sources_validate_connection`,
`sources_validate_columns`, `sources_check_size_limits`, `sources_sample`,
`sources_sample_source` and
`sources_validate`. Validation passes the actual resident source to Core's
`source=` path; it does not replace the provider with a captured DataFrame.
Inspection deliberately returns `config=null` for bound sources to avoid exposing
connection configuration. Native data freshness/caching and engine behavior remain
the provider's semantics; this is not an immutable snapshot or a reconnect service.
Validation leaves `input_sha256` and `input_identity` unknown for a live binding;
the source handle's resource hash describes its binding label, not database rows.
Async sources use Core's `AsyncToSyncAdapter` on an owned event loop in the
resident worker. It preserves native size limits and schema helpers, connects
sampled async sources on that loop, and closes each connection before the loop.
Failed entry releases partial connections; no automatic reconnection/retry is
added. Local async-source lifecycle and canonical-result comparison are tested;
external MongoDB/Elasticsearch and other remote providers remain unverified.
Arbitrary protocol objects are not accepted by the binding factory contract.

The same root or sampled source also accepts `sources_read`, `sources_profile`,
`sources_scan`, `sources_mask` and `sources_learn`. These call Core's public
`read(source, sample_size=...)`, `profile(source=...)`, `scan(source=...)`,
`mask(source=...)` and `learn(source=...)` directly, preserving the provider's
materialization limits and sampling behavior. Read uses Core's fixed sampling
seed of 42; `sample_size` is optional and bounded to 1–10,000,000. Profile and scan
accept only the source handle. Mask accepts optional `columns`, native `redact`,
`hash` or `fake` strategy, and `strict` (default false). Learn accepts
`infer_constraints` (default true) and `categorical_threshold` (default 20,
0–10,000).

Read and mask retain immutable Arrow **assets** for paging and downstream work;
fake values are generated once per call. They leave the resident source unchanged.
Profile and PII reports retain their native JSON representation and public source
name. Learned **schema** handles can enter `schemas_inspect`, `schemas_export`
and `sources_validate(schema_id=...)`. Reports and schemas can contain data
values; use the existing sensitive-results policy. Every result remains dependent
on its source's grants, expiry and lifetime. Output quota or native operation
failure does not publish partial output or discard a healthy source connection.
PII scanning retains Core's sampled detection behavior; it does not prove that
all sensitive values were found.

`sources_sample` keeps its Arrow-snapshot contract. `sources_sample_source`
accepts the same `source_id`, `n` and optional `seed`, and returns a **source**
handle for the actual native sampled datasource in the existing worker. It does
not re-create the provider from the factory or replace it with an Arrow frame.
The sampled source supports inspect, connection/column/size checks, validation,
Arrow capture and further native sampling. At most 16 child handles are retained
per worker. Native sampling and caching semantics still apply; a native provider
may return the same object rather than a new sample.

Children borrow the owner's connection/event-loop lifetime. Child release closes
its native context when its final alias is released, and leaves the root and
siblings alive. Parent release revokes descendants and closes their native contexts
before the root/provider and plugin teardown. As with root close, bounded cleanup
failure kills the owning worker; provider-side cleanup is not guaranteed. Derived
live samples have unknown input content hashes, and every access reauthorizes
parent grants. No extra SQL/Python, provider path, credential or callback can be
supplied to sample a live source. Local Polars and actual async-source tests cover
nested samples, validation, Arrow capture, shared-loop cleanup and capacity reuse.


Repeated successful opens with the same key return the same live handle, never a
new connection. Factory failures/cancellation after execution begins are recorded
as `outcome_unknown` and cannot be blindly retried with that key. Released or expired
handles cannot be resurrected by replay. Release and session shutdown close the
source before plugin teardown; timeout/failed bounded cleanup may require killing
the owned worker and cannot guarantee provider-level cleanup. Binding changes,
package identity changes and policy changes revoke the source and derived artifacts
when next checked; TTL bounds idle workers. No monitoring process is started.

`sources_open_polars` takes an owned `asset_id` and optional complete native
`PolarsDataSourceConfig` under `config`; it returns a `source` resource ID.
Use that ID as `source_id`. `sources_validate` accepts the same validation
settings as `assets_validate`, except that callers cannot substitute `asset_id`.
It passes the actual resident datasource to Core, not a DataFrame stand-in.
Inspection/check results are retained artifacts, read only with disclosure
consent. A negative connection result remains `connected=false`; native size
gate exceptions fail the operation. Sampling returns an owned Arrow asset, not a
new resident connection. Releasing the input asset/source revokes descendants and
closes the owned worker. Normal shutdown permits native context-manager cleanup;
on timeout/EOF/protocol failure only that owned process is invalidated/reaped.
The source handle hash identifies configuration, not a separate dataset snapshot.
`sources_open_sqlite` takes `path`, a literal `table` name and optional `config`.
It opens an actual `SQLiteDataSource` over a transaction-consistent private
backup, not a materialized DataFrame. SQL/base config fields are forwarded;
the native `database` field is bound to that backup, never a caller-selected
URI. The backup counts against file/session quotas and its hash identifies the
database bytes. Only rollback-journal databases and real tables are admitted:
WAL, views and model-supplied SQL are rejected. Releasing a SQLite source also
releases its owned backup. `assets_sqlite` remains the separate materialization
operation. This does not establish other SQL/cloud/NoSQL/async provider parity.
Native config is not an OS sandbox.

`sources_open_duckdb` takes `path`, a literal physical `table`, and optional
`config` with the native base/SQL/DuckDB fields except `database` and `read_only`:
the private copy and read-only mode are fixed. From this unreleased checkout,
install `pip install '.[mcp,duckdb]' 'duckdb>=1.4,<2'`. Stable DuckDB 1.4 or
newer, below 2, is required. Older/prerelease versions return
`unsupported_dependency`. Native SQL-engine dispatch and sampling are retained,
not replaced by an in-memory dataframe. WAL-bearing files, views and arbitrary
SQL are not admitted; a live-copy stamp/WAL change rejects publication.
Connections disable external reads, extension auto-install/load, community and
unsigned extensions, then lock configuration following the
[DuckDB security guidance](https://duckdb.org/docs/current/operations_manual/securing_duckdb/overview).
Threads default to 1 (explicit maximum 16); memory defaults to `max_memory_mb`
in MB unless `memory_limit` is supplied. These are connection controls, not an
OS sandbox or a complete process-memory bound. `unsupported_table`, malformed
database, quota and cancellation failures publish no usable source. Explicit
release closes the worker and deletes its private copy, never the original.

`streams_open` takes a path and a closed format-specific `source`:

```json
{"path":"/approved/data.csv","source":{"format":"csv","config":{"chunk_size":1000,"columns":["age"],"skip_rows":0,"max_rows":10000,"separator":",","has_header":true}}}
```

Formats are `csv`, `jsonl`, `parquet` and `arrow`. All take `chunk_size`,
`columns`, `skip_rows` and `max_rows`; Parquet/Arrow also take `use_mmap`, CSV
also takes `separator` and `has_header`. CSV `skip_rows` skips physical lines
before the header, exactly as the native source does. Arrow requires PyArrow.
The returned resource ID is `stream_id` for `streams_inspect` and
`streams_next`; each advance requires an `idempotency_key`. A batch returns
`{artifact: <Arrow asset metadata>, exhausted: false}`; EOF returns
`{artifact: null, exhausted: true}`. Feed the batch resource ID to
`assets_validate` or other retained-data tools. This is **per-batch** validation,
not global uniqueness, cross-batch windows, Kafka offsets or continuous-stream
validation. Native chunk boundaries are retained rather than silently reshaped.
Retries of the same key replay the same retained batch, never another advance.
Receipts are bounded by `max_items` per session. These cursors/receipts are not
durable across server restart. An uncertain advance invalidates the worker and
must not be retried against a guessed position. Releasing the stream releases
its owned input snapshot and revokes batches. This local file-stream support
does not establish Kafka/Kinesis/PubSub account or realtime parity.
For example, after capture, call `assets_validate` with:

```json
{
  "asset_id": "asset_<returned-session-token>",
  "checks": [{"validator": "between", "settings": {"columns": ["age"], "min_value": 0, "max_value": 120}}]
}
```

Use actual returned identifiers, not the placeholder. `validators_describe`
documents the closed fields for each of the 260 concrete registered entries,
including inherited `ValidatorConfig`, cache and algorithm options. Unknown
settings are rejected instead of being silently dropped by the Core constructor.
Numeric constructor fields retain finite Python integers without coercion to
float, including integers outside floating-point range. Native constructor or
execution errors remain errors; NaN, infinity and booleans are not accepted as
numeric settings. This does not increase resource or response budgets.
The inventory includes six native streaming validators (`streaming_null`,
`streaming_completeness`, `streaming_nan`, `streaming_range`,
`streaming_positive`, `streaming_outlier`), with their native `chunk_size` option.
That inventory is **not** evidence that all 260 algorithms have passed parity.
Constructor rejection, native execution failure and an exception deliberately
caught by an algorithm are separate coverage categories. An invalid constructor
returns `validator_configuration_error` without raw settings; after approved
trusted code has begun, failures instead preserve `outcome_unknown` and prohibit
automatic replay. Missing optional libraries remain failures, not successful
quality verdicts. Polygon operations require `truthound[geospatial,mcp]`.
`column_consistency.rules` accepts only `when`, `then` and optional `description`;
the two conditions use the closed Polars expression grammar below, not Python.

Development-source Core corrections also change observable algorithm behavior:
numeric PSI/CSI and PSI-based feature drift include observations outside the
reference histogram range rather than discarding them. Cached and uncached PSI
paths use the same tail-accounting rule; thresholds are unchanged.
Constant reference distributions retain their point mass rather than becoming
empty histograms. Default reference caches are instance-scoped; cross-instance
reuse requires an explicit versioned `cache_key` for the same baseline.
`validators.memory.SGDOneClassSVM`, used by `online_svm`, now delegates to the
native unsupervised scikit-learn SGD one-class estimator, not a binary classifier
trained with synthetic negative labels. Its scaler and kernel projection are
fitted on the first batch and remain fixed while subsequent batches update the
estimator. Predictions from the earlier binary surrogate are not equivalent.
This composition follows the [scikit-learn one-class SGD guide](https://scikit-learn.org/stable/modules/outlier_detection.html#scaling-up-the-one-class-svm).
These are unreleased source changes, not changes to already published 3.2.0 files.

The Core JSON-schema validator uses Draft 7 when its optional library is present.
Untrusted schema settings never fetch external references. The transport follows
local pointers/anchors, including references into otherwise literal JSON data,
and checks the reached schemas too. Reference analysis has a separate bounded
depth/node budget; exceeding it fails rather than silently truncating validation.
General property names such as `id` remain ordinary data. Executable regex or
external-reference schemas require a reviewed operator binding.
Untrusted SQL, callbacks, caller filesystem paths and unrestricted regex return
`operator_configuration_required`. Exact operator-owned validator templates and
installed native extension bindings are available through the separate approval
boundary below; they do not make model-supplied executable inputs trusted.
Inline JSON Schema may use local `#` references, not filesystem/network `$ref`.
This adapter does not silently replace these unsupported settings.

A DataFrame/LazyFrame constructor parameter takes `{"asset_id":"<returned-id>"}`;
maps of reference tables contain such references as values. Every reference is
separately authorized and retained in the output's parent chain. Non-string-key
maps use `{"entries":[{"key":1,"value":0.5}]}` to preserve key types.
Datetime/date settings use `{"value":"2026-01-01T00:00:00"}` or an ISO date;
timedelta settings use `{"value":1000000}` (integer microseconds). Type settings
use the same closed `DType` schema as `assets_inline`.
Polars expression settings use a bounded AST, e.g.
`{"op":"gt","left":{"op":"column","name":"age"},"right":{"op":"literal","value":120}}`.
Supported nodes are column/literal, arithmetic, comparisons, Boolean operators,
null tests, absolute value and sum/mean/min/max/count/unique-count aggregates.
Python expressions, SQL and serialized expressions are not accepted.

File capture exposes all `FileDataSourceConfig` fields through `options`;
`options.sample_size` is source configuration, while the top-level `sample_size`
is the independent `th.read()` sample. `.pq` is also accepted. SQLite requires an
admitted `.db`, `.sqlite` or `.sqlite3` and a real table name, not SQL or a view.
It bounds backup size and Core materialization, closes connections and never
executes caller SQL. WAL databases are rejected: SQLite read-only WAL access may
still modify shared-memory bookkeeping. Other SQL/cloud providers and their
credentials are not implied by this local binding.

Handles are private to one server session. Defaults are 128 handles, 1 GiB total
artifact bytes and 900 seconds TTL; configure `--max-handles`,
`--max-session-bytes` and `--handle-ttl-seconds`. Derivatives cannot outlive their
parents. Release invalidates descendants. Each access rechecks grants and bytes;
EOF closes the session and deletes owned files. These are not durable cursors.
`assets_page` preserves nested nanosecond temporal values via physical-value tags.
Untrusted inline datatypes are limited to 16 nesting levels and 524,288 fixed
scalar slots per value. Before native allocation, scalar slots across all rows
and actual nested list/struct elements × 16 bytes must fit the remaining session
byte budget. This conservative admission check
prevents tiny nested-array/null declarations from requesting enormous buffers;
it is not an OS memory sandbox or a change to Core's data algorithms.
`artifacts_read` returns base64 byte pages (maximum 32 KiB), SHA-256 and a next
offset. Raw pages, schema inspection and byte reads require
`--allow-sensitive-results`. Artifact creation alone does not disclose its data.
Renderers never accept output/template paths, launch a dashboard or open a browser.
Rendered HTML remains untrusted content; host/browser handling is separate.

`schemas_create` accepts `schema_spec` with the same closed column constraints
and statistics as inline validation. `schemas_export` takes `resource_id` and
`format=json|yaml` (default YAML), producing an owned artifact, not writing an
agent-selected path. YAML uses Core `Schema.save()` and can be re-entered with
Core `Schema.load()`. Arbitrary regex is not an inline schema input; use a
reviewed native validator binding for executable schema patterns.

`profilers_catalog` describes `ProfilerConfig` and the closed built-in component
constructors. `profilers_profile` takes `asset_id`, `config`, `options`, optional
`column_analyzers`, `table_analyzers`, `type_inferrers`, and table `name`/`source`
labels. Component lists contain `{"component":"CorrelationAnalyzer","settings":{"threshold":0.8}}`
(in this case a table analyzer). It exposes six column analyzers, three table
analyzers and three type inferrers. `config` and inherited profiler `options`
are explicit Core inputs. `config.sample_size` controls the leading-row sample;
a supplied `options.sample_size` overrides it without mutating the provided config.
The effective config also controls default correlation, pattern and distribution
analysis. This path does not use `random_seed` to select a random sample.
Unknown fields, arbitrary component imports and custom regex
pattern definitions are rejected. Custom extension sessions and suite execution use the separately approved
`profiler_services_*` lifecycle below. Streaming/incremental/distributed
profilers require their own contracts.

Advanced native profiling retains component failures in `execution_issues`
(stage, component, exception type and optional column), excluding exception bodies.
A column/table profile's `complete` property reports whether the selected analysis
finished; it does not mean every input row was analyzed when sampling is configured.
Incomplete profiles serialize `complete: false` and their diagnostics, and both
native save/load and dictionary adaptation preserve that verdict. MCP returns
`execution_status: failed` with `native_profile_incomplete` and an owned partial
profile artifact. Read it under sensitive-result consent; source authorization
still applies. Rule generation stops before invoking any generator when profiling
is incomplete. The native suite generator also raises on a selected generator's
failure instead of returning a successful partial suite.

`profilers_generate_rules` accepts the same profile inputs plus a required
`generator`, for example `{"component":"SchemaRuleGenerator","settings":{}}`,
and `strictness` (`loose`, `medium`, or `strict`, default `medium`). The closed
catalog also includes `StatsRuleGenerator`, `PatternRuleGenerator` and
`MLRuleGenerator`. One selected native generator receives the fresh native
profile; its complete `GeneratedRule.to_dict()` results are retained as an
owned artifact. Generation does not execute validators or apply a suite.
Set `representation` to `validator_configs` to retain a list of
`{"validator_class": "...", "config": {...}}` records instead. Each config comes
directly from native `GeneratedRule.to_validator_config()`, including columns
and `mostly`; it is not a configuration inferred from a summary. The default
`rules` representation remains unchanged. Export does not instantiate, approve
or execute a generated validator; later use must pass the normal configuration
and authorization checks.
Omit `column` (or pass `null`) for table-wide generation. To use the native
column-only algorithm, pass a literal column name such as `"column":"amount"`.
This calls `generate_for_column()` on that column's fresh profile, not a filter
over table-wide rules; table-level correlation rules are therefore not added.
An absent column fails with `core_error` and does not produce a partial artifact.
Exceptions fail the operation rather than silently returning partial rules.
Artifact reads require sensitive-result permission and retain the source asset's
authorization; releasing that asset also prevents reading its rule artifact.

### Approved custom profiler sessions

`profiler_services_open` and `profiler_services_execute` retain custom native
profiler compositions in one approved worker. An installed capability factory
is an async context manager yielding `truthound.mcp.profiler_services.ProfilerRuntime`.
The runtime selects `profilers` (native `Profiler` instances), `generators`
(`ValidationSuiteGenerator` instances with custom rule generators), `executors`
(`SuiteExecutor` instances with the operator's registry/listeners), and optional
`profile_slots`/`suite_slots` mapping fixed aliases to absolute `Path` objects.
The factory owns all configuration, custom patterns/analyzers/inferrers, physical
paths and cleanup. Add the capability to private `plugins` bindings and map
`"profiler_services":{"local":"profiler-factory"}` in `bindings.json`.
No import string, callback, regex definition, validator class or path is accepted
from the model. This is trusted installed code, subject to exact operation approval.

After opening, pass the returned `resource_id` as `service_id` with `binding_id`
and a unique `idempotency_key`. Commands are:

| Commands | Native effect |
|---|---|
| `inspect` | Bounded composition, slot and retained-result aliases |
| `profile` (`profiler_id`, `asset_id`) | Run the selected native composition; retain its exact `TableProfile` |
| `profile_get`, `profile_export` (`profile_id`) | Read the native profile or export a re-enterable advanced-profile artifact |
| `profile_import` (`artifact_id`) | Decode an owned advanced-profile artifact and restore through native `ProfileSerializer` |
| `profile_save`, `profile_restore` (`slot_id`) | Save a retained `profile_id` or restore one from an operator-selected slot |
| `generate_suite` (`profile_id`, `generator_id`) | Run the selected native suite generator with `strictness`, optional category/confidence filters and `name` |
| `suite_get`, `suite_export` (`suite_id`) | Read/export the exact native generated suite |
| `suite_save`, `suite_restore` (`slot_id`) | Save a retained `suite_id` or load native JSON from a fixed slot |
| `suite_execute` (`suite_id`, `executor_id`, `asset_id`) | Execute with optional `parallel`, `fail_fast` and `dry_run` |
| `profile_release`, `suite_release` | Release a worker-local result alias |
| `slot_delete` (`slot_kind`, `slot_id`) | Explicitly delete the selected persisted profile/suite |

Each owner retains at most 16 profiles and 16 suites within the session byte
budget. JSON save uses Core serialization followed by replacement of the fixed
slot; it is not a transactional database or a power-loss durability guarantee.
Slots contain results, while composition configuration stays in the pinned
factory. Restoring a fixed slot in a new owner requires a fresh approval and
explicitly authorizes that stored data independently of a prior session's input.
Artifact import keeps the source artifact's authorization dependencies. Import
into a separate owner: self/descendant dependency cycles are rejected before approval. No
arbitrary suite artifact import is provided: suite loading uses operator-owned
slots because rule-to-validator conversion can execute trusted code.

Incomplete profiles retain diagnostics, return `native_profile_incomplete` and
cannot generate suites. Failed profile responses are idempotent and do not
re-execute analyzers on replay. Missing validator conversions, runtime errors or
skipped rules produce a failed `native_suite_incomplete` artifact rather than
successful empty execution. A completed validation may still have `success: false`
when data violates rules. Dry-run success proves construction only, not validation.
Native reports retain error types rather than exception bodies. Sync/async
executors use the supplied native registry and the same failure accounting.
Native components, generators and validators check authorization before their
next callback; the owner also watches policy changes. Release, expiry, timeout,
revocation and session close reclaim the owner and factory resources. Incremental and chunked native profilers use the same retained profile/suite
lifecycle. The factory may provide `incrementals` (`IncrementalProfiler`),
`streaming` (`StreamingProfiler`), `chunk_factories` (async context managers
returning native `ChunkIterator` objects) and `stream_progress` callback aliases.
`incremental_profile` takes `incremental_id`, `asset_id`, optional
`previous_profile_id` and `force_refresh`; it returns native profiled/skipped
columns and change reasons beside the retained profile. `stream_profile` takes
`streaming_id`, `stream_id` and optional `progress_id`. It reads the operator's
chunk source directly, closes its iterator and source context on success/failure,
and never asks the model for a source path or callback.

Incremental profiles persist a per-column `content_fingerprint` with the native
modulo sampling configuration/hash. Equal row/null counts alone do not permit
reuse; missing fingerprints and incomplete columns are re-profiled. The sample
can miss changes outside selected rows and is not a full-data integrity check.
Chunk errors remain incomplete with current-run diagnostics, while an inconsistent
chunk schema raises and closes the source. Chunked analysis uses native streaming
statistics, not every advanced DataProfiler analyzer. Distinct-value tracking can
grow with cardinality; chunking alone is not a constant-memory guarantee.
Distributed profiler lifecycle remains a separate contract.

Native profile caches can also be selected by the factory: `caches` maps aliases
to `ProfileCache` instances, `cache_keys` maps aliases to native cache-key objects,
and `cache_patterns` maps aliases to fixed logical glob patterns. Commands
`cache_get`, `cache_set` and `cache_get_or_compute` require `cache_id` and
`cache_key_id`; set additionally takes `profile_id`, while compute takes
`profiler_id` and `asset_id`. Optional `ttl_seconds` includes an immediate-expiry
value of `0`; omission uses the native cache default. `cache_invalidate` removes
one selected key, `cache_invalidate_pattern` uses `pattern_id`, `cache_clear`
clears the selected cache and `cache_stats` returns selected counters. The
operator must namespace keys by dataset, configuration and implementation version;
this binding does not invent a new cache-key algorithm.

Cache hits retain the actual native profile and incomplete hits remain failed
artifacts. File caches serialize to a private temporary file before replacement;
corruption or access-stat write failure preserves committed bytes and raises
`ProfileCacheReadError`/`ProfileCacheWriteError`. The MCP intent becomes
`outcome_unknown` after a worker error and is not automatically replayed. Repair
uses a fresh approved owner and explicit invalidation. File pattern invalidation
validates all logical key/file mappings before deleting anything; older entries
without stored keys must first be read by their known key or explicitly cleared.
Unsupported custom backends raise instead of pretending to invalidate zero keys.
Redis parsing/connection errors retain corrupt values and report safe error types;
failed construction and close attempt all owned client/pool cleanup. Redis
namespace scans escape the physical prefix and recheck each logical match.
`cache_stats` excludes physical keys, paths, endpoints, raw metadata and exception
bodies. `connected` is native observed state, not a fresh health check; unavailable
native key counts return `native_cache_incomplete`. Actual local file persistence
and synthetic Redis protocol failures are covered; a real Redis service remains
unverified. These changes do not promise a transactional cache or power-loss
recovery.

Generated-suite execution also accepts `timeout_seconds` (positive, finite,
at most 3600) and `max_workers` (1–32). Omission preserves the selected executor's
configured deadline/worker defaults; the whole MCP operation still uses the server
policy timeout. A per-validator deadline uses native `ProcessExecutionStrategy`,
with importable/picklable validator state and a serializable binding guard checked
before validation. Serialization failures are explicit failures, never a fallback
to an uninterruptible thread. The native deadline includes child startup and result
waiting; parent-side serialization and final cleanup can add overhead. The child
owns a separate inherited process group, private result pipe and parent-lifetime
lease. Timeout, normal completion and parent loss close that execution scope.
Errors cross the native pipe by type without original text/tracebacks, and result
serialization failure is an error rather than a fabricated timeout.

The generic native thread strategy can time out while its daemon still runs:
`was_terminated` stays false, `outcome_unknown` is true and automatic timeout
retries stop. It is not used for generated-suite validator deadlines. Parallel
fail-fast cancels queued work and drains work already running; skipped work is
retained in the failure accounting. Native rule start/completion listeners are
serialized, and their failures remain visible. These lifecycle controls are not
an OS sandbox for installed extension code.

### Approved continuous schema monitoring

`schema_monitor_open` runs native `SchemaWatcher` or `AsyncSchemaWatcher` in a
session-owned process. It accepts `binding_id`, `mode` (`sync` default or `async`),
`poll_interval` (0.1–3,600 seconds, default 60), `max_events` (1–1,000, default 100,
also bounded by operator `max_items`) and `idempotency_key`. Core determines
changes, versions and breaking alerts; this adapter does not implement a second
schema comparison algorithm.

The operator privately configures `bindings.json`, alongside installed plugin
distribution/version/entrypoint/SHA-256 pins:

```json
{"schema_monitors":{"orders-watch":{"source":"orders-schema-factory","notifier":"orders-alert-factory"}}}
```

Both names must be exposed by the pinned plugins' `capabilities` maps. Each is a
zero-argument factory: the source returns Core's `SchemaSource` protocol (for
example `FileSchemaSource`); the optional notifier returns `AlertNotifier` with
`send(alert) -> bool`. A native `CallbackNotifier` can adapt an operator-owned
HTTP/email implementation. There is no built-in hosted endpoint, model-selected
URL, import string or credential input. Destination, credentials and connection
policy remain operator-side. Installed plugins are trusted code, not sandboxed.
Alerts contain schema details and source identifiers, which may include local
paths; approving sending authorizes disclosure to that fixed destination.

```json
{"binding_id":"orders-watch","mode":"async","poll_interval":60,"max_events":100,"idempotency_key":"orders-watch-01"}
```

Opening first returns an approval plan for installed extension execution,
continuous schema reads and, when configured, external schema alert sending.
Use the existing operator approval flow, then repeat the exact request. No plugin
imports, reads or sends occur before approval. The returned `resource_id` is the
`monitor_id` for `schema_monitor_status` and `schema_monitor_stop`. Both return
an artifact resource; read it with `artifacts_read` for state, error, poll/event
counts, successful send count, native versions and alerts. Initial registration
is a baseline; only breaking changes send alerts. Sync polling begins after the
interval; the native async poller checks immediately after starting.

Monitoring continues without further tool calls, until stopped, released, expired
or the session closes. Stop joins the native poller and shuts down the plugin;
release/expiry/timeout force-reap only this owned worker. Session termination is
not a graceful delivery guarantee. Binding and package identities are rechecked
before each poll and send; a parent-side supervisor also reaps the worker on
operator policy changes without waiting for another tool call (0.1-second checks).
A failed source, rejected/unknown send, revoked binding or exhausted event/byte
budget stops further effects and exposes `state="failed"`, never healthy status.
The status read itself can succeed while the reported monitor is failed. No
automatic notification retry is attempted: a network error may follow a remote
effect. Bytes are bounded by the smaller operator file/session limit and 16 MiB.
Replaying an approved open returns its existing handle, not a replacement worker.

This monitoring history is private in-memory state, separate from
`schema_history_*` checkpoints. There is no autonomous cross-session restart,
durable delivery queue, exactly-once delivery, monitor pause/resume or transfer of
source execution authority through these session-owned operations. Local real HTTP transport and sync/async native parity
are tested; actual cloud, email and broker services remain unverified.

### Durable schema recovery and notification outbox

For separately approved restartable monitoring, call `schema_watch_register`
with the same binding/mode/interval/event settings plus `lifetime_seconds`
(1–2,592,000 seconds, default 86,400) and a new `idempotency_key`. Review and
approve the returned plan using the operator CLI, then repeat the exact request.
This consent explicitly covers persistent plugin execution, automatic restart
reads and, when a notifier is configured, persistent external notification delivery.
Registration does not load plugins or poll. The returned `watch_id` is durable,
not a session handle. There is no model-selected URL or credential.

Run the existing `python -m truthound.mcp worker` with the **same** Core profile,
sensitive-result consent, allowed directories, private control directory and
limits as registration. `--once` performs one bounded sweep; otherwise the
foreground worker continues until stopped. Restarting that worker recovers
registered watches without another MCP session or approval, until the approved
expiry, disablement, policy/binding change or failure. Nothing installs an OS
service or starts itself at boot. Each sweep performs at most one queued send
or due native poll per watch; queued delivery takes priority, so the effective
poll interval also depends on the worker interval and backlog.

Native sync/async `check_now`, version serialization and breaking-alert creation
are reused. A signed SQLite transaction commits the last schema/checksum,
native history/counters and newly queued alerts together. Internal schema state
preserves closed rich values losslessly and native Polars dtype metadata through
empty Arrow IPC; unsupported or lossy values fail instead of being stringified.
Native history/alert presentation still uses Core serialization. Earlier
unversioned development checkpoints fail closed and retain their evidence; they
require a new approved registration rather than an implicit migration.
Changes while the worker is stopped are
compared against the last committed baseline when it resumes; intermediate
changes between polls cannot be reconstructed. This is separate from the
`schema_history_*` checkpoint API.

The outbox uses `pending → sending → sent | unknown`. A send is marked durably
before calling the pinned native notifier. Only an explicit `True` response
counts as sent. Exceptions, false responses or a crashed sender leave `unknown`;
these entries are never automatically retried. A dead poll owner can be recovered,
but concurrent workers cannot claim a live owner's work. The isolated child exits
when its parent's pipe closes, including while a plugin is blocked. This is **at most one
automatic attempt**, not guaranteed delivery or exactly-once remote effects.
Operator-side notifiers must not hide their own retries. Unknown outcomes need
destination-side reconciliation; no automated reconciliation is provided.

`schema_watch_status` and `schema_watch_disable` accept `watch_id` and optional
`include_snapshot: true`. Status reports state, expiry, poll and delivery counts;
the optional artifact exposes sensitive native history and outbox payloads via
`artifacts_read`. Disable prevents future claims but cannot undo an in-flight
send. Evidence is retained. Storage is bounded to 32 definitions and at most
1 MiB per definition (also bounded by operator file/session limits); event limits
remain enforced. Exhaustion fails closed, not by silently dropping notifications.
Real loopback HTTP, separate-process restart and crash-after-send are tested;
cloud/email/broker accounts remain unverified. These are unreleased source features.

#### Approved lifecycle, reconciliation and retention

`schema_watch_update` requires `watch_id`, the current status `revision` as
`expected_revision`, a new `idempotency_key`, and `action`. Review its operator
plan, approve through the existing CLI, then repeat the exact request. Changes
since review make the plan stale; obtain fresh status and use a new key. Retrying
a committed request returns its saved result, not another transition.

- `pause` stops new polls and sends while keeping the baseline and queue.
  In-flight work may finish. `resume` requires a paused, unexpired watch and
  does not extend consent. Changes during the pause are compared to the last
  baseline, not replayed as intermediate events. Disabled/failed/expired watches
  cannot be resumed. These are durable scheduling controls over Core polling,
  not a claim that the native async watcher exposes a pause method.
- `renew` requires `lifetime_seconds` (1–2,592,000), setting expiry relative
  to approval execution time. Active, paused, expired and failed watches qualify;
  disabled watches do not. An expired active watch becomes paused. Failed watches
  remain failed until explicit recovery. Renewal preserves all history and outbox.
- `restart` requires a failed, unexpired watch and moves it to paused. It preserves
  its baseline and queue, so fix the underlying provider failure before resuming.
  An incompatible checkpoint requires `reset_checkpoint: true` together with
  `accept_history_loss: true`: this deletes the checkpoint, starts a new baseline
  on the next poll and resets its event budget. The outbox and monotonic alert
  counter survive. Reset can miss changes during the discarded baseline interval.
- `delete` requires a non-active watch, no in-flight lease and
  `accept_history_loss: true`. Pending/sending/unknown notifications additionally
  require `discard_unresolved: true`. It atomically removes the definition and
  returns a `deleted` receipt, freeing its slot. Status then returns `invalid_handle`;
  exact committed deletion replay returns the saved receipt. Approval receipts,
  exported artifacts, backups and SQLite free pages are not securely erased.
- `confirm_delivery`, `discard`, or `retry` require an **unknown** `alert_id`
  and `evidence_reference` from operator-side destination reconciliation.
  Use a non-secret case identifier, not credentials or raw data. Confirmation
  records an operator assertion (`confirmed`), not a successful observed send;
  discard records `discarded`, not delivery. Retry additionally requires
  `accept_duplicate_risk: true` and explicitly approved possible duplicate
  external delivery. It queues the same native alert for one new attempt, never
  sends during the approval call, and remains paused if the watch is paused.
  Every decision records its plan, reference and time; attempts are counted.
- `retention` accepts `retention: {"max_age_minutes": 1440, "max_count": 100,
  "mode": "all"}`; at least one age/count limit is required. Native Core
  `TimeBasedPolicy`, `CountBasedPolicy` and `CompositePolicy` choose retained
  entries (`all`: retain only if all rules retain; `any`: if any rule retains).
  `retention: null` disables future cleanup. Approval authorizes immediate and
  subsequent worker-sweep removal of timestamped terminal queue entries only
  (`sent`, `confirmed`, `discarded`). Pending/sending/unknown and legacy entries
  without resolution timestamps are protected. No active lease is reordered.
  Status exposes the policy and cumulative `pruned_notifications` count.

- `history_retention` uses the same `retention` policy shape, independently for
  checkpoint schema versions and alerts. It preserves the latest version, current
  polling baseline, event/poll counters and every alert referenced by the outbox.
  Protected entries may exceed the requested count. Old parent references may
  refer to removed versions. `retention: null` stops future history cleanup.
  Status reports `history_retention` and cumulative `pruned_history`; alert IDs
  remain monotonic across pruning, process restart and checkpoint reset.

Retention runs on worker sweeps, including paused or disabled watches, under
unchanged authority. Queue retention alone does not erase native checkpoint history; use the separate
history policy for that. Neither retention policy removes approvals, external
copies or definition slots, and neither provides secure physical erasure.
History/event/byte budgets still apply. No service-side delivery lookup or
exactly-once guarantee is implied. Real external services remain unverified.

### Session-owned schema history

`schema_history_create` constructs Core `InMemorySchemaHistory` (`backend="memory"`,
default) or `FileSchemaHistory` (`backend="file"`) in an owned private worker.
Choose `strategy`: `semantic` (default), `incremental`,
`timestamp` or `git_like`; `max_versions` defaults to 100 and must be between 1
and the operator's `max_items` (at most 1,000). The result contains `history`
(resource information) and `revision=0`.

`schema_history_execute` takes `history_id`, `expected_revision` and one closed
`command`. Its result contains the history information, the current revision
and an `artifact`; read the complete native result with `artifacts_read`.

| Command action | Fields | Native result |
| --- | --- | --- |
| `save` | `asset_id`, optional `version`, `metadata` | New version using the owned asset's schema, not its rows |
| `get` | `version_id` | Version or null |
| `latest`, `baseline` | None | Newest or oldest retained version, or null |
| `count` | None | Native retained version count |
| `breaking_since` | `version_id` (ID or version string) | Native breaking-change comparison against latest |
| `by_version` | `version` | Version with that version string, or null |
| `list` | Optional `limit`, `since`, `until` | Newest-first versions; ISO datetime filters are exclusive |
| `diff` | `source_id`, optional `target_id` (ID or version string) | Native changes and summary; target defaults to latest; null if absent |
| `rollback` | `version_id` (ID or version string), optional `reason` | New copied version with native rollback metadata, or null |
| `delete` | `version_id` | Whether a version was removed |
| `watch_add`, `watch_check` | `name`, `asset_id` | Register an initial schema, or update its snapshot and run native `SchemaWatcher.check_now`; status or change events |
| `watch_remove`, `watch_status` | `name` for removal only | Removal result or native source status |
| `alerts`, `alert_stats` | None | Native local alert history or statistics |
| `alert_get`, `alert_acknowledge`, `alert_resolve` | `alert_id` | Native alert or acknowledgement/resolution result |
| `snapshot` | None | File backend only: closed native JSON file checkpoint; an artifact, not durable storage approval |

Example commands, using the returned identifiers:

```json
{"history_id":"<history.resource_id>","expected_revision":0,"command":{"action":"save","asset_id":"<asset.resource_id>","metadata":{"author":"operator"}}}
```

```json
{"history_id":"<history.resource_id>","expected_revision":1,"command":{"action":"list","limit":10}}
```

`save`, `delete` and `rollback` advance the revision even when a missing version
produces false/null. Queries do not advance it. Stale revisions fail before
mutation. Retention evicts the oldest native versions; rollback creates a new
version rather than overwriting history. Native `limit=0` means no limit.
Use timezone-naive filters for the Core's timezone-naive creation timestamps;
an aware/naive comparison fails as it does in Core. UUIDs, creation/diff times
and timestamp/hash version strategies retain native nondeterminism.

Watch add/check/remove and alert acknowledgement/resolution also advance revision.
The watcher compares explicitly supplied immutable asset schemas; it does not poll
external files or start a background thread. Initial registration is a baseline,
not a saved version. Changes pass through native `HistoryEventHandler` and
`AlertingEventHandler`: changed schemas become versions and breaking changes become
local alerts. Read events/alerts through artifacts; no email, webhook or remote
notification is sent. Source and alert counts are bounded by `max_versions`;
reaching the alert cap rejects further checks rather than silently losing alerts.
Native logged storage/watcher failures are failures at the MCP boundary.

For cross-session recovery, create with `backend="file"`, then call
`schema_history_persist` with `history_id`, `expected_revision`, and an
`idempotency_key`. It requires the existing operator approval flow and control
directory. The immutable signed checkpoint contains the native version files,
index and local alerts, including acknowledgement/resolution state. It stores no
rows. Save returns `snapshot_id`, `sha256`, and `bytes`. Repeating the same approved
intent at the same revision returns the original result.

```json
{"history_id":"<history.resource_id>","expected_revision":1,"idempotency_key":"history-checkpoint-01"}
```

Call `schema_history_resume` in a new session with exactly one of `snapshot_id` or
the original `idempotency_key`. It revalidates signatures, file hashes, operator
policy and original source/workspace grants, and creates a new private history
handle at revision zero. Native version IDs, dates, schemas and alerts survive;
old session handles and watcher registrations do not. Each resume is an independent
working copy, not a shared mutable directory; subsequent edits require a new
approved checkpoint to survive session termination. There is no automatic crash
recovery of uncheckpointed edits. Operator-installed executable source bindings
cannot be transferred through this snapshot path.

Checkpoints are limited to 16 MiB (or the smaller operator file limit), 128 records
and 32 MiB total. `storage_retention_preview` and approved automatic retention
accept `kind="histories"`; `storage_delete` accepts the returned snapshot IDs.
Deletion also revokes active restored handles. Private working files are removed
on release/expiry; approved checkpoints remain until explicit/approved retention
deletion. Arbitrary directories, archives and compressed-history imports are not
accepted. The history resource hash describes the creation configuration,
not mutable history contents. Save observes schema only, but all observed asset
grants and handles remain dependencies, even after version deletion/eviction.
Release, expiry, revoked input grants or worker failure invalidate the history
and derived results. A worker/output failure after a command starts invalidates
the handle instead of claiming rollback or permitting an ambiguous retry.
Reaping occurs on the next session operation, and an owned lifetime timer bounds
idle workers. Use `resources_release` to close explicitly. Output bytes and
retained resources remain subject to session quotas. Custom strategies, caller
paths, arbitrary code and remote/paid services are not accepted.

### Drift and resident ML

`profilers_schema_evolution` compares the native schemas of two owned assets
using `baseline_id` and `current_id`. Optional `detect_renames` (default `true`)
and `rename_similarity_threshold` (default `0.8`) are passed to Core's
`SchemaEvolutionDetector`. The owned artifact contains all native change
records and the native compatibility summary, including its detection timestamp.
Change ordering follows Core's unordered column-set iteration. This binding
detects additions, removals, inferred renames and type changes; it does not infer
constraint/nullability changes or persist a baseline, start a watcher, or apply
migrations. Artifact reads require sensitive-result permission and both source
assets to remain authorized and live.

`profilers_column_renames` analyzes only columns added/removed between two assets
through Core's `ColumnRenameDetector`. It returns all confirmed and possible
candidates, similarity/confidence, type compatibility, reasons, metadata and
unmatched names in an owned artifact. It does not rename columns or migrate data.
Inputs are `baseline_id`, `current_id`, `similarity_threshold=0.8`,
`require_type_match=true`, `allow_compatible_types=true`,
`high_confidence_threshold=0.9` and `low_confidence_threshold=0.7`.
An omitted `calculator` preserves Core's default composite. Explicit calculator
kinds are `levenshtein`, `token`, `jaro_winkler` (`prefix_scale=0.1`), `ngram`
(`n=2`), and `composite` (up to 16 weighted, non-nested calculators).
Composite `calculators=null` keeps native defaults; `[]` deliberately preserves
the native empty-weight behavior. Finite weights/thresholds and native clamping
are preserved; no arbitrary calculator code is accepted. For example:

```json
{"baseline_id":"<baseline-handle>","current_id":"<current-handle>","calculator":{"kind":"composite","calculators":[{"calculator":{"kind":"ngram","n":2},"weight":0.6},{"calculator":{"kind":"token"},"weight":0.4}]}}
```

As with schema evolution, reading results requires sensitive-result permission
and both live, authorized parent assets. Column ordering and matching follow
Core; this is rename inference, not proof that two columns have the same meaning.

`profilers_schema_compatibility` calls Core's `SchemaCompatibilityAnalyzer`
on two owned asset schemas. Input fields are `baseline_id`, `current_id`,
`rules` (ordered names `types`, `nullability`, `column_presence`) and `strict`.
Omitting `rules`, setting it to `null`, or supplying `[]` preserves Core's
default three rules. The complete artifact includes `is_compatible`, `level`,
`issues`, and `rules_checked`; reading it requires sensitive-result permission
and both live parent assets. Custom Python rules are not accepted.
In this unreleased checkout, `strict=true` rejects any reported issue and
returns compatibility level `none`. Failed or wrongly typed native rule verdicts
raise `CompatibilityAnalysisError`; they cannot produce a successful compatibility
artifact. Optional `baseline_nullability` and `current_nullability` map column names
to declared booleans, with both maps required and at most 1,000 entries each.
They require the nullability rule; declarations for a common column must be paired.
Core treats nullable-to-non-nullable tightening as breaking. Unknown columns and
untyped declarations fail. Polars dtypes alone still carry no declared nullability;
omitting these maps does not validate nullability or infer constraints from a
sample's observed values. Ordered/repeated built-in rule selection is preserved.

```json
{"baseline_id":"<baseline-handle>","current_id":"<current-handle>","rules":["types","column_presence"],"strict":false}
```

Native panics are distinct from ordinary validation failures: when a native
extension aborts a worker without returning a canonical result, MCP reports
`worker_error` without a successful or partial result artifact. For example,
Polars timestamps outside Python's datetime range can trigger this during
privacy-validator value conversion. This is failure isolation, not a repaired
Core result or a successful privacy assessment.

For concrete detector configuration, use `drift_catalog` and `drift_detect` with
`detector`, `settings`, `baseline_id`, `current_id`, `baseline_column` and an
optional different `current_column`. This calls the selected native detector's
`detect()` rather than substituting the comparison facade defaults.

`drift_compare` takes `baseline_id`, `current_id`, optional `columns`, `threshold`,
`sample_size` and `method`. Methods are `auto`, `ks`, `psi`, `chi2`, `js`, `kl`,
`wasserstein`, `cvm`, `anderson`, `hellinger`, `bhattacharyya`, `tv` (alias
`total_variation`), `energy` and `mmd`. Computation and sampling remain Core's;
the full report is an artifact, not a summary presented as complete evidence.
Both snapshots are retained and reauthorized. A comparison that Core applies to
no columns is not an MCP quality pass.

Use `models_catalog` (paged, maximum five entries) for actual algorithm names and
all closed configuration fields. `models_create` takes `algorithm`, `config`,
and optional nested `detectors` for `EnsembleAnomalyDetector`. The catalog covers
ZScore/IQR/MAD/IsolationForest/Ensemble anomaly detectors, Distribution/Feature/
Concept/Multivariate drift detectors, DataProfileRuleLearner, ConstraintMiner and
PatternRuleLearner. A name is a fixed catalog key, never an import path. Unknown
options fail instead of becoming ignored ML kwargs.

Creation returns `model_id` and `revision=0`. Fit with `asset_id` and
`expected_revision=0`; success returns revision 1. Predict/score/inspect require
the current revision. Learning, fit-predict and explicit two-input detection
also advance the revision. `models_detect` adds `reference_id` and optional
`columns` to the ordinary model/data input. Inapplicable methods and unsupported
online learning are failures, never skips or implicit fallback retraining.
`models_rules` and `models_diagnostics` require `model_id` and `expected_revision`.
`models_result_query` takes `model_id`, `expected_revision`, `asset_id`, and
`action`: `all_scores`, `anomalies`, or `drifted_columns`. It predicts with the
native model and calls the Core result iterator/filter, retaining the output in
an owned artifact without incrementing revision. `all_scores` includes normal
rows and their score metadata, not just the anomalies in the usual summary.
Only `drifted_columns` accepts a finite `threshold` (default `0.5`); Core's
inclusive comparison is preserved. A mismatched model type, untrained model or
stale revision fails explicitly. Artifacts depend on both model and input, and
reading them requires disclosure consent. Each call makes a fresh prediction;
it does not query an earlier prediction artifact.

`models_rule_query` additionally requires `asset_id` and `action`: `suite`,
`validator_specs`, `by_type` (with `rule_type`), or `by_column` (with `column`).
It calls the native rule learner's prediction and result conversion/filtering
methods, retaining the full output as an owned artifact without changing the
model revision. Selectors for other actions are rejected. Unknown types/columns
produce the native empty list, not a validation pass. The generated suite is
configuration data; it is never implicitly applied or executed. Model and input
revocation invalidate derived artifacts. Non-rule models fail explicitly.
Diagnostics retain the native configuration/info plus applicable statistics,
feature importance/profiles, reference correlations, column profiles, learned
patterns and ensemble names/weights. Read their artifact under disclosure consent.

Models live in owned isolated processes, preserving their actual learned state
between calls. Up to four resident model workers exist per session, also bounded
by the handle quota. Training/reference assets remain authorization parents.
Revision mismatches do not execute Core. Timeout/cancellation/protocol failure
kills and reaps the owned worker; a failed mutation invalidates its model rather
than reusing uncertain state. TTL expiry, release, revoked grants and session EOF
also invalidate model workers and dependent handles. These processes still have
the host user's OS privileges; the application policy is not an OS sandbox.
On Windows virtual environments, workers launch the underlying interpreter
directly while preserving the virtual environment. Shutdown waits for that
interpreter, not an intermediate launcher, before removing its working directory.
Resident native workers resolve `HOME` and `USERPROFILE` to their owned temporary
workspace, including on Windows. Default native caches are therefore temporary;
durable state requires explicit operator-pinned slots or factory configuration.
The operator's actual home directory and credentials are not inherited.
Private worker output sinks use UTF-8 explicitly, including Windows SDK progress
output; these streams remain separate from MCP protocol responses.

ML result artifacts contain `encoding`, `model_algorithm`, `model_revision` and
the tagged native `result`. This records the model state used without lossy
Python object serialization. `models_save` takes `model_id` and `expected_revision`
and retains native JSON; `models_restore` takes that artifact's `resource_id`
and returns a new model at revision 1. The twelve catalog algorithms preserve
trained trees, ensemble children, drift reference data and learned rules. The
versioned data-only format rejects executable types, duplicate JSON keys and
mismatched model identities. Incomplete legacy states fail instead of becoming
ready. Restored models inherit the saved artifact's authorization parents;
releasing training data also revokes restored models. These are session-owned
artifacts unless explicitly transferred to durable storage as described below.
Arbitrary plugin models and complete method-by-method lifecycle equivalence remain outstanding.

`models_ensemble_add` takes `model_id`, `expected_revision`, a fresh `detector`
specification and finite `weight` (default 1.0). It invokes native `add_detector`;
it does not implicitly train the child. `models_ensemble_remove` takes an `index`
(-8 through 7) and invokes native removal. Its model artifact retains the removed
child's configuration and learned state. Restore produces revision 1 for a trained
child or revision 0 for an untrained child. A successful edit advances the ensemble
revision once; failed mutations invalidate the resident handle. Up to eight
children and four nested levels are admitted. Instead of `detector`, supply
`source_model_id` and its exact `source_revision` to copy an existing trained
detector through native data-only save/load. This preserves learned state without
retraining; it does not share mutable Python objects. Subsequent source edits do
not change the copy. Self-inclusion and dependency cycles are rejected. The copy
retains source authorization and lifetime: releasing the source invalidates its
dependent ensemble. Stale source revisions fail before editing the ensemble.

`models_persist` takes a model artifact `resource_id` and an `idempotency_key`
(8–128 characters). It returns `approval_required` until the operator approves
the exact plan using `approve-plan`. This explicitly transfers the snapshot's
lifetime beyond the session; merely saving a model does not do so. `models_resume`
takes the resulting `snapshot_id` in a new session with the same private control
directory and policy. It restores native JSON without retraining, rechecks source
directory grants and creates new handles. Policy changes deny both restoration
and access to derived handles. Provider-bound model snapshots are not admitted.
If the successful save reply was lost, `models_resume` accepts the original
`idempotency_key` instead of `snapshot_id`; it resolves the committed receipt
without repeating persistence. Supply exactly one identity selector.
Storage is private local SQLite, not a cloud service or workspace export: at most
128 snapshots, 16 MiB per snapshot and 32 MiB total, additionally bounded by the
file budget. Snapshot bytes and signed metadata are checked on restoration; the
snapshot and successful intent receipt commit in one transaction. Use the approved
storage deletion operation below; do not edit the control database manually.

### Approved native Checkpoint execution

Development Core profiles expose `checkpoints_run`. Unlike validation jobs,
this calls `truthound.checkpoint.Checkpoint.run` itself, including configured
actions, notification conditions, action retries and optional routing. It is
not part of the published 3.2.0 package.

An operator installs a trusted plugin and pins its distribution version, hash
and entry point in private `bindings.json`. Add
`"checkpoints": {"daily": "checkpoint-factory"}` and expose that capability
through the existing `plugins` binding. The zero-argument capability must return
a synchronous context manager yielding a native `Checkpoint`. That owner must
close its datasources and action resources on exit. Never place credentials in
model arguments or public plugin metadata. This executes trusted installed code
with the host user's privileges, not sandboxed code.

```json
{"binding_id":"daily","use_router":true,"idempotency_key":"daily-reviewed-run-01"}
```

The first call returns an approval plan before importing the plugin. Review it
through the existing operator approval CLI; approval includes the pinned
factory's actions, external notifications and configured native retries. Repeat
the exact request after approval. Changing the binding, package or policy
invalidates authority. Arbitrary imports, data connections, actions and context
objects cannot be supplied over this wire contract.

The response is a sensitive artifact handle containing the rich-value encoded
native `CheckpointResult.to_dict()`, including `validation_run`, `action_results`,
status and error. Read it through `artifacts_read`. Transport success only means
the result was retained; inspect the native status and action results to judge
validation/delivery success. Operator permission for sensitive results is
required; native diagnostics may contain private information.

Replay returns the retained result without rerunning actions. Timeout, shutdown,
revocation during execution or an unconfirmed failure is `outcome_unknown` and
is never automatically retried. Native action retries are different: they are
explicitly included in the approved configuration and can repeat external
effects. Terminating a worker cannot undo an effect or guarantee graceful
third-party cleanup. This operation does not start triggers, a Checkpoint
scheduler or an escalation service. Remote services remain unverified.

`checkpoints_run_async` separately approves the native `AsyncCheckpoint.run_async`
path. It accepts `binding_id` and `idempotency_key`, without `use_router`. The
bound zero-argument factory must return an **asynchronous context manager**
yielding `AsyncCheckpoint`, and close its executor, sources and actions on exit.
The same result-artifact, revocation and unknown-outcome rules apply; this is
not a conversion of the synchronous Checkpoint into a thread wrapper.

### Native Checkpoint services

Development Core profiles also expose `checkpoint_services_open` and
`checkpoint_services_execute`. Use private `checkpoints` bindings for these
factories as well. For `kind="runner"`, the factory returns a synchronous
context manager yielding a stopped native `CheckpointRunner`, with its
Checkpoints, triggers, callbacks and queue limits configured by the operator.
For `kind="escalation"`, it returns an asynchronous context manager yielding a
stopped native `EscalationEngine`, with operator-selected policies, store,
scheduler and notification handler. Owners must stop services and close all
resources on exit. No factory, credentials, notification destination, arbitrary
context or Python object is accepted from the model.

```json
{"binding_id":"daily-runner","kind":"runner","idempotency_key":"runner-open-01"}
```

Approve the returned plan with the operator approval CLI, then repeat exactly
the same request. The returned `monitor_...` handle belongs to this MCP session.
Opening does not start native scheduling. Each command, including status and
queue consumption, requires its own handle-bound approval:

```json
{"binding_id":"daily-runner","service_id":"monitor_<returned-id>","command":"start","idempotency_key":"runner-start-01"}
```

- Both kinds support `start`, `stop` and `status`.
- A runner additionally supports `results`, with `max_results` (1–1000,
  default 100, also bounded by policy). This **consumes** the native result
  queue. Replaying the approved request returns the retained artifact without
  draining again. Native queue capacity/drop behavior is unchanged.
- An escalation engine supports `trigger` with `incident_id` and optional
  `policy_name`; `record`, `acknowledge`, `resolve`, `cancel` and `escalate`
  require `record_id`. Acknowledgment/resolution/cancellation record
  `mcp-approved-operator`, not a model-provided person's identity. Escalation
  does not force otherwise-disallowed transitions. Native failure results are
  retained, not rewritten as successful delivery. Stats encode integer-keyed
  `escalations_by_level` as `[{"level":1,"count":2}]` to preserve key types.

Read command results through `artifacts_read`; they remain sensitive and tied
to the service grant. `stop` stops native execution but retains the owner and
its native in-memory state for a later approved `start`. Releasing the handle,
closing the session, expiry, revocation or a failed exchange terminates the
worker. Unknown command effects are never automatically retried. Termination
cannot retract notifications or guarantee third-party cleanup. These are
session-owned services. Policy configuration can be persisted and restored as
described below. Durable native execution is separately opt-in through the pinned
operator factory, as described under durable checkpoint recovery.
Real external notification and store services remain unverified. These tools
are development-source features, not part of published 3.2.0.

Additional approved management commands reuse the native service methods:

- Runner `schedule_update` takes `checkpoint_name`, `trigger_name` and a nonempty
  `schedule_update` patch for one built-in interval or cron trigger. It reuses
  `CheckpointRunner.edit_schedule`; the runner and its execution thread must be
  stopped. Editing retains the trigger's run count, last-run clock and attachment,
  rather than creating a new first run. Interval seconds/minutes/hours, enabled,
  max runs, run-immediately, catch-up, max-concurrent, JSON metadata, start/end
  times, weekdays and timezone can be edited. Omitted settings remain unchanged;
  explicit null clears only start/end time, weekdays or timezone. Boundaries use
  naive local datetimes; empty weekdays and nonpositive total intervals are
  rejected. Native scheduling semantics apply: timezone/catch-up/concurrency
  settings do not add guarantees absent from Core. Cron triggers accept `expression`
  and shared trigger settings; interval-only settings on cron are rejected. Renaming
  is not supported. For provider-defined triggers, only shared fields (`enabled`,
  `max_runs`, `run_immediately`, `catch_up`, `max_concurrent`, JSON `metadata`)
  can be edited; provider-specific paths, connections and callbacks stay in the factory. It does
  not itself run or restart a checkpoint. Each edit needs exact-request approval.
- Runner `checkpoint_update` takes `checkpoint_name` and a nonempty
  `checkpoint_update` patch. It reuses `CheckpointRunner.edit_checkpoint` while
  quiescent. `checks` reuses the closed registry constructor/settings contract
  (1–1000 checks, without callback bindings); an explicit checks list replaces
  validation configuration. Other fields are `min_severity`, `auto_schema`,
  `run_name_template`, `tags`, JSON `metadata`, `fail_on_critical`, `fail_on_high`,
  `timeout_seconds` (1–86400) and `sample_size` (positive). Only `min_severity`
  and `sample_size` accept explicit null. Omitted fields retain their current values.
  Sources, schema objects/paths, actions, routers and callbacks remain factory-owned.
- Runner `run_once` takes `checkpoint_name`; `run_all` executes every registered
  checkpoint. Native actions and result callbacks run; results are returned in
  artifacts, not inserted into the scheduled-result queue. `remove_checkpoint`
  takes `checkpoint_name` and returns the native removal boolean. Removal does
  not cancel an already executing checkpoint.
- Escalation `policy` and `policy_remove` take `policy_name`. Missing policies
  return null/false. Removing a policy does not resolve existing incidents.
  `active_records` optionally filters by `policy_name`; `incident_records`
  requires `incident_id`. Results are complete native lists subject to the
  artifact byte budget, not silently truncated.
- `policy_update` requires `policy_name` and a nonempty `policy_update` object.
  It updates an existing policy's `enabled`, `severity_filter`,
  `max_escalations`, `cooldown_minutes`, `business_hours_only`,
  `business_hours_start`, `business_hours_end`, `business_days` or `timezone`.
  Omitted settings are preserved; explicit nulls and unknown fields are rejected.
  It also accepts `description`, `triggers`, `levels` and JSON `metadata`.
  A supplied levels list replaces the whole list (unique positive level numbers).
  Each level supports delay, targets, repeat count/interval, acknowledgment,
  auto-resolve, JSON conditions and metadata. Each target supports native type,
  identifier, name, priority and JSON metadata. Omitted fields are preserved.
  All eight native trigger and target types are accepted; custom callbacks and
  delivery transports still come from the pinned operator factory. Approval of
  the exact request includes destinations, not just the policy name. Do not put
  secrets in these settings: requests and policy artifacts are retained locally.
  This replaces the registered native policy, not existing records or scheduled
  jobs. Subsequent native execution decides when it consults that policy.
  Core's explicit `policy_name` lookup bypasses enabled/severity selection;
  disabling a policy affects automatic selection, not explicit triggering.
  Business-hours enforcement is also Core-defined (currently advisory).
  Timezone strings are passed through: Core treats timezone lookup failure as
  within business hours. This setting is not an authorization boundary.

```json
{"binding_id":"incident-engine","service_id":"monitor_<returned-id>","command":"policy_update","policy_name":"approved-policy","policy_update":{"enabled":false,"cooldown_minutes":30},"idempotency_key":"policy-disable-01"}
```

Every command, including reads, needs its own exact-request approval. Cached
replay never re-executes a removed checkpoint or restores a removed policy.
Invalid command shapes are rejected before execution. A native exception or
lost receipt after execution starts remains `outcome_unknown`, not safe retry.
`policy_create` takes the same `policy_name` / `policy_update` shape, applying
native defaults for omitted fields. It refuses to overwrite an existing policy.
`policy_update` refuses to create a missing policy. Empty levels/triggers are
allowed and retain native semantics, rather than inventing a default recipient.
The `trigger` command also accepts `trigger_type` (native default
`unacknowledged`) and JSON `context`. Both are forwarded to Core, including
native automatic policy selection; they are rejected on unrelated commands.

`persist_configuration` takes explicit unique `policy_names` (1–100), requires a
stopped escalation service, and returns a signed immutable `snapshot_id`, hash
and byte count in its artifact. Missing policies are stored as removal markers.
Storage is private approval storage, bounded to 1 MiB per snapshot, 128 snapshots
and 16 MiB total; no caller-controlled file paths. Native non-JSON metadata is
not silently converted. Use `checkpoint_services_open` with `kind="escalation"`,
the same `binding_id`, this `snapshot_id` and a fresh idempotency key to restore
after session shutdown. A separate exact-request approval is required; altered
payloads, policy or binding changes are rejected before factory execution.
The pinned owner factory is reopened, selected policies/removals are applied,
and the service stays stopped. Unselected factory policies remain unchanged.
This restores **policy configuration only**, not incident records, pending jobs,
notification delivery receipts or runner schedules. Those records remain owned
by the operator's native store. They can use the separate durable native recovery
contract below. Opening a session never installs an automatically restarting service.

For a runner, `persist_configuration` instead takes 1–100 unique
`checkpoint_names`. The service and execution thread must be quiescent. Its signed
snapshot retains the listed checkpoints' editable scalar settings, explicitly edited
closed check specifications and interval/cron/shared trigger configurations. Missing
checkpoints are removal markers. Trigger names must be explicit and unique. Restore
with `checkpoint_services_open(kind="runner", snapshot_id=..., ...)` under the same
binding and fresh approval. The factory supplies all sources, actions, provider fields
and unchanged validators; these opaque objects are never serialized or imported from
a snapshot. Rich non-JSON metadata is rejected. Snapshot restore edits matching native
objects and preserves durable trigger history; it neither starts scheduling nor
replays previous actions. Runner and escalation snapshots cannot be interchanged.

### Durable checkpoint recovery

The operator factory can opt into `RunnerConfig(journal_path=...)` or
`SchedulerConfig(journal_path=...)`. MCP requests never accept journal paths.
The APScheduler adapter uses its supported 3.x API; install the optional
`.[mcp,checkpoint-scheduler]` extra from the development checkout. Asyncio and actual APScheduler restart/
notification-drain tests exercise the journal path. An escalation engine requires a file-backed SQLite record store and journal in
**the same file**; memory stores and separate files are rejected. This lets the
notification receipt, incident counters and following reservation commit together.
A runner requires explicitly named, unique triggers. Trigger configurations must
be JSON compatible; providers with additional mutable state must implement
`export_runtime_state` and `restore_runtime_state` alongside the base clocks.
Custom provider state and external broker delivery remain provider responsibilities.

Opening a durable service takes exclusive process-bound ownership of that journal.
A live owner is never displaced by a timeout; `stop` retains ownership, and `close`
or process exit releases it. Use the same approved binding/factory in a new session,
with fresh approval, then approve `start` to resume unclaimed reservations.
Policy, binding, control-directory or plugin identity changes reject recovery.
Scheduled effects recheck the current binding and installed plugin identity.
Escalation jobs also pin the native policy definition; changed or disabled policies
quarantine the job instead of notifying a new destination.

Intent commits before native actions or notification delivery. Interrupted claimed
work becomes `unknown`, never automatically replayable. Unknown notifications also
block later/manual escalation for that incident. A handler returning false,
raising, being cancelled or losing its completion commit cannot register a following
notification. This does not promise exactly-once delivery to an external service.
Runner run counts and last-run clocks survive restart. In-memory result queues do
not; retain results separately if required. A runner's unknown action/callback
outcome blocks that trigger, including a new manual run after an unknown manual run.

Both service kinds support these separately approved commands:

- `recovery_status`: returns `enabled`, bounded `attempts` containing only
  `identity`/`state`, and `next_offset`. `recovery_limit` is 1–1000 (default 100,
  also bounded by policy); `recovery_offset` defaults to 0. Payloads, raw source
  values, exception messages and filesystem paths are not returned. A live attempt
  is `attempting`; only a new exclusive owner classifies abandoned claims as unknown.
- `recovery_resolve`: takes `attempt_id` and `resolution` (`completed` or
  `cancelled`), while stopped/quiescent. The operator must independently determine
  the outcome. This never sends a notification or executes a checkpoint.
  Confirming an unknown notification updates its count and reserves its following
  level under the unchanged policy. Discarding it records cancellation without
  creating a following notification. Confirming/discarding an unknown scheduled
  runner attempt consumes that trigger slot and updates its last-run clock.
- `recovery_forget`: takes 1–100 unique `attempt_ids`, while stopped/quiescent.
  Only completed/cancelled receipts can be deleted, atomically; pending, attempting
  and unknown evidence is protected. This is logical deletion, not secure erasure.

```json
{"binding_id":"incident-engine","service_id":"monitor_<returned-id>","command":"recovery_status","recovery_limit":100,"idempotency_key":"recovery-status-01"}
```

Native users should call `await engine.close()` or `runner.close()` when finished;
MCP worker cleanup does this automatically. Reopening never implicitly approves
execution or retry. Local subprocess restart tests cover native actions, notification
interruption and installed factory/session reopening; real external notification,
cloud, broker and paid AI services remain unverified.

### Native schedule preview

`schedules_preview` takes a cron `expression` (up to 256 characters), an explicit
ISO datetime `after`, and `count` (1–100, default 5). It reuses Core's parser,
`matches` and `next_n` in an owned, deadline-limited worker. Five/six/seven-field
expressions, presets and special fields have the same native semantics. Naive
times remain naive; explicit UTC offsets are retained, not converted to the
server timezone. This is not IANA timezone/DST scheduling support.

The result distinguishes invalid expressions (`valid=false`, native `errors`)
from a valid expression whose native four-year search horizon yields fewer
times (`search_exhausted=true`). Neither is a data-quality PASS. Preview does
not register a schedule, start a checkpoint or execute any action. Persistent
scheduling is available separately for bounded validation occurrences below.

### Durable validation occurrences

`jobs_submit` takes `validation` (an `assets_validate` request), an
`idempotency_key`, and optionally `schedule` with `expression`, offset-aware
`after`, and `count` (1–100). Due times use native cron semantics. All requested
times must be found; partial cron search results are rejected. Submission
requires separate operator plan approval to persist the captured input and
authorize deferred computation. Context/schema handles, reference-frame handles,
provider bindings and external actions are not admitted; inline `schema_spec`
and closed validator settings remain available.

Submission atomically persists all occurrences or none, returning the first
`job_id` and the complete `occurrence_ids`. Each occurrence owns its own state,
lease and result; cancelling one does not cancel the others. All input copies
count against the unchanged storage quotas. These are a finite series, not an
unbounded recurrence rule. Use each ID with `jobs_status`, `jobs_execute`, `jobs_result`,
`jobs_cancel`, or `jobs_recover`. A client must explicitly execute a due job;
`jobs_submit` itself does not install a daemon or authorize recurring execution.
There are no paid provider calls or automatic retries.
Queued input survives session closure. Completed native results survive restart
and can become newly authorized session artifacts. A completed execution can
still contain data-quality violations; native execution failures remain `failed`.
Cancelled queued work is not executed. Cancelling a running job returns `busy`
rather than claiming another process was stopped. After the execution deadline
plus a 30-second cleanup allowance, explicit recovery marks an unfinished lease
`outcome_unknown`, never PASS or an implicit retry. This allowance is not an
extension of the native worker timeout. Successful receipts require computation,
owned-resource cleanup and a transactional result commit.

The private store admits at most 128 jobs, 8 MiB per input/result and 16 MiB total
job payloads. To recover a lost submission reply, the five job operations after
submission also accept the original `idempotency_key` instead of `job_id`; they
resolve and reauthorize the existing record rather than submit another job.
For a series, the original key resolves the first occurrence and its complete
ID list. Supply exactly one selector. These limits remain subject to existing
file/session budgets. Provider-backed jobs remain unsupported; this is not full scheduling parity.

### Persistent recurrence and foreground worker

`schedules_register` takes the same validation request as `jobs_submit`, but requires
a schedule with `count: 1`. Its separately approved plan authorizes captured-input
storage and indefinite recurring execution. The definition and first occurrence
commit atomically. `schedules_status` and `schedules_disable` accept either its
`control_id` or original `idempotency_key`. Disabling cancels a queued head; work
already running may finish. Failed, cancelled or indeterminate heads pause the
definition without retrying. Successful heads advance by one native cron time per
sweep, including missed times in order. A native cron search with no next time
stops with an error, never a fabricated next occurrence.

The operator must start the foreground process explicitly with the same grants
and limits as the registering server (replace paths with existing local directories):

```bash
truthound-mcp worker --allow-dir /data --control-dir /private/control \
  --profile core --allow-sensitive-results --poll-interval 5
```

`--once` performs one sweep; `--poll-interval` is an integer from 1 to 3600 seconds.
SIGINT/SIGTERM stops after the current bounded operation. No system service is
installed. Separate workers share transactional cursors and execution leases.
Storage quotas do not increase: exhausted storage stops the worker. At most 64
definitions, including disabled ones, are retained. Persistent retention policies
can prune old completed payloads; current active/paused schedule heads are protected.
Policy changes, corrupt records and uncertain outcomes are not silently retried.

`storage_policy_register` accepts the preview policy plus an `idempotency_key` and
requires separate approval for future automatic deletion. The worker evaluates
approved policies before schedules on each sweep. `storage_policy_status` and
`storage_policy_disable` accept a `control_id` or original `idempotency_key`.
Disabling prevents later sweeps, not an already committed deletion. Policies and
schedule cursors survive session/process restart. This does not install an
always-on OS service or add external provider execution.

### Durable storage retention and deletion

`storage_retention_preview` takes `kind` (`models`, `jobs` or `histories`) and at least one of
`max_age_minutes` (1–5,256,000), `max_count` (1–128), or `max_size_bytes`
(1–33,554,432). It reuses Core `TimeBasedPolicy`, `CountBasedPolicy`,
`SizeBasedPolicy` and `CompositePolicy`. With `mode="all"` (default), every selected
policy must retain an item; `mode="any"` lets any selected policy preserve it.
Size is the actual stored input plus result bytes, or model/history checkpoint payload bytes, not
SQLite file size or a new storage quota. Persistent automatic policies accept
the same fields and retain their approved composition after restart.
It returns `candidates`, `preserved` and `excluded`
identity lists without deleting anything. Queued/running/indeterminate jobs and
legacy records without a creation timestamp are excluded. Equal timestamps use
identity ordering for deterministic native count-policy evaluation.

`storage_delete` takes exact `identities` (1–128 unique snapshot/job IDs) and an
`idempotency_key`. It requires separate operator approval and rechecks ownership,
signatures, current grants and inactive state inside the deletion transaction.
Active and indeterminate jobs cannot be deleted. All requested payloads and the
successful receipt commit atomically; replay returns the receipt without deleting
again. Existing resumed models/result handles are denied after their durable
source is deleted. Approval receipts remain for audit and replay prevention.
Deletion is logical SQLite payload removal, not guaranteed disk sanitization.
Explicit previews do not delete data; automatic cleanup requires the separate
persistent policy approval and operator-started worker above. Archival, compression
and full retention subsystem parity are not claimed. Legacy undated records may still be explicitly selected for approved
deletion. All operations require the same private control store and operator
policy plus sensitive-result consent.

### Native loopback dashboard lifecycle

`dashboards_open` uses a private `dashboards` binding whose pinned async factory
yields a stopped `truthound.profiler.dashboard.server.DashboardServer` or
`truthound.datadocs.dashboard.ReflexDashboardServer`. Install
`.[mcp,dashboard-streamlit]` for Streamlit or `.[mcp,dashboard]` for Reflex from
the development checkout. For Reflex, the operator explicitly prepares a private
project with Node/npm before opening the binding; MCP start never installs or
builds dependencies. See the [native preparation guide](https://github.com/seadonggyun4/Truthound/blob/main/docs/guides/datadocs/dashboard.md).
The factory owns host, port, page settings and data-access controls. No source path,
Python script, port, host or configuration is accepted from the model. MCP permits
only `localhost`, `127.0.0.1` or `::1`; it does not create a remote ChatGPT server.

`dashboards_execute` repeats the binding, live `service_id`, exact idempotency key
and `command` (`start`, `stop` or `status`). Each needs its own approval.
`start` accepts `timeout_seconds` in (0, 30] (default 10), reports native health
readiness separately from process liveness, and refuses an already occupied port.
Status returns `running`, `ready`, `url` and `exit_code`. Stop terminates only its
owned server/process group. Handle release, expiry and revocation reap the owned
worker's descendants; the native child also watches parent disconnect, including
hard parent termination. This is a local process lifecycle, not OS-service installation.

The native launcher now sends all dataclass settings as private JSON data to a
static application entry point; titles are never interpolated into executable
Python. Local tests exercise actual Streamlit and prepared Reflex startup,
HTTP readiness, stop/restart, handle release and parent hard-kill, plus
Streamlit occupied-port failure and native title rendering. Release requests
bounded native/factory cleanup before reaping the captured process tree; blocking
provider cleanup can still require hard termination.
Native UI configuration still does not create an authentication boundary:
`require_auth` is not enforced by the existing UI, and its file-loading controls
retain native semantics. The Reflex viewer's display/export flags control its UI,
not authorization or removal of profile values from the private compiled project.
Prepared configuration changes are refused at start and require a fresh build.
Remote deployment remains unverified.

### Native broker adapters and processing

`brokers_open` takes a `binding_id` from private `brokers` bindings and an
`idempotency_key`. Its pinned installed capability returns an async context manager
yielding `truthound.mcp.brokers.BrokerRuntime`: a disconnected native
`BaseStreamAdapter`, optional native `WindowProcessor` and `ExactlyOnceProcessor`,
and an optional operator-selected `output_topic`. Opening does not connect. The
model cannot supply broker endpoints, credentials, import paths, topics or callbacks.
Raw messages require sensitive-result consent. The returned `service_id` is a
session-owned `monitor_...` handle; release, expiry and revocation terminate its owner.

`brokers_execute` repeats `binding_id`/`service_id`, an exact `idempotency_key` and:

- `connect`, `disconnect` or `status`. Status exposes native state/metrics, retained
  batch count and configured processor statistics/watermark. Disconnect drops local
  batches. It cannot retract external effects or guarantee third-party cleanup.
- `consume`, with `limit` (1–1000, default 100, policy bounded) and `timeout_ms`
  (1–30000, default 1000), returns a `batch_id`, rich-value encoded native messages
  and `size`. Consumption advances the native cursor and never implicitly acks.
  At most 16 batches are retained within the current artifact-byte budget.
- `ack`, with a retained `batch_id`, calls native `commit_batch` and releases that
  batch. Native adapters may commit only the highest offset per partition.
  Replaying the same approved request returns its receipt without committing again.
- `window_add` adds a retained batch once to the operator-selected native window;
  `window_flush` calls `force_emit`. Neither command acks the source.
- `process` passes a retained batch once through native
  `ExactlyOnceProcessor.process_message`. Native processing may commit offsets or
  produce through an operator-selected sink. Failures and deduplication remain
  explicit; exception text is replaced with a closed failure code.
- `recover` calls native processor transaction rollback recovery and returns
  `rolled_back`. It does not establish an external exactly-once delivery guarantee.
- `seek` takes nonnegative `partition`/`offset` while no batches are retained.
  `produce` takes a JSON `message` with `value`, explicit ISO `timestamp`, optional
  `key` and nonnegative `partition`; its destination is factory-owned. `flush` calls
  the adapter's native flush. These may have external effects and native retries.

Every command, including reads, requires separate operator approval. Lost effects
are `outcome_unknown`, never automatically retried. Batch/window state stays in the
resident owner; cross-session state, source offsets and recovery durability depend
on the operator's native backend. Local tests use the real Core test adapter;
Kafka/Kinesis/cloud services remain unverified.

### Native AI proposal review and application

`ai_services_open` uses private `ai_services` bindings. The pinned async owner yields
`truthound.mcp.ai_services.AIRuntime` with a native `TruthoundContext`, a required
explicit provider, operator-selected data/source/model and optional native run.
Its workspace must be within an operator-allowed workspace root. The model cannot
supply provider credentials, provider/model selection, source paths or actor identity.
Opening and every command need exact-request approval; provider calls can disclose
native redacted summaries and incur provider charges. Local tests use a deterministic
provider and make no paid API calls.

`ai_services_execute` repeats the binding, session `service_id`, `idempotency_key`
and one closed command:

- `suggest` requires `prompt` (1–16000 characters); optional `sample_size` is
  1–10000 (default 1000). It calls native `suggest_suite` and persists the proposal.
- `explain` uses an optional `run_id` or the operator-provided native run, with
  `include_history=true` by default. It calls native `explain_run` and persists analysis.
- `list_proposals` and `list_analyses` take `limit` (1–1000, default 100, policy
  bounded); the latter optionally filters by `run_id`. `proposal` and `history`
  require `proposal_id`; `analysis` requires a native `analysis_id`.
- `approve`, `reject` and `apply` require `proposal_id` and `expected_sha256` from
  the previous result's `artifact_sha256`, plus optional `comment` (up to 4096
  characters). The digest pins the native proposal revision in the approved request;
  stale revisions are rejected before mutation. Each decision returns a new digest.
  Approval and application are separate exact operator decisions. Actor identity is
  fixed to `mcp-approved-operator`. Native applied proposals remain terminal.

Redaction is fixed to native `summary_only`; raw samples and executable code are not
accepted. Responses preserve native fields with rich-value transport, excluding
`workspace_root`. Lifecycle state lives in the native workspace; reopening requires
fresh approval and the same bound factory. MCP receipts do not make native multi-file
writes transactional against unrelated workspace writers. Unknown provider/application
effects are not retried. Paid providers and real external deployments remain unverified.

### Approved OpenLineage delivery

`openlineage_services_open` resolves private `openlineage_services` bindings to
an installed async factory yielding
`truthound.mcp.openlineage_services.OpenLineageRuntime(emitter, datasets={alias: dataset})`.
The emitter is a native `OpenLineageEmitter`; endpoint, authorization and dataset
metadata are configured by the operator. Opening sends no lineage event. This is
an unreleased source-checkout feature, not a published-service claim.

Every `openlineage_services_execute` call repeats `binding_id`, `service_id` and
an exact `idempotency_key` and requires approval. Closed commands are `status`,
`start`, `running`, `complete`, `fail`, `abort`, `graph`, `resolve`, `shutdown`.
`start` takes `job_name`, optional owned active `parent_run_id`, and optional
`datasets` aliases for inputs; `complete` takes aliases for outputs. Transition
commands take the owned `run_id`. `fail` sends a fixed operator-reported failure
message. The wire cannot supply exceptions, credentials, endpoints or imports.
`graph` takes an owned `graph_id` and `job_name`, sends only dataset names, and
retains that graph's permission and expiry dependencies on the service.

Delivery failures return an `outcome_unknown` envelope with a sensitive artifact
containing `run_id`, `event_type` and `outcome_unknown: true`. The owner remains
available for `status` and `resolve`; repeated transitions cannot resend that
uncertain event. `resolve` requires `run_id` and an explicit boolean `delivered`
after external receiver inspection. It records the operator's determination
without a request. Approval replay returns the retained result without sending.
State artifacts omit transport configuration and dataset contents. New runs are
bounded to 1000 retained active/uncertain records. Graph network failures may be
partially delivered; there is no whole-graph transaction or automatic retry.

`shutdown` joins native work and factory teardown before acknowledgement. Releasing
an input graph revokes its dependent service; releasing the service, expiry or
parent loss closes the resident owner. Forced termination cannot determine remote
delivery. Local emitter state is not durable across process loss. Actual owned
loopback receiver tests do not establish live external service compatibility.

### Native audit and telemetry administration

`audit_services_open` uses private `audit_services` bindings whose async owner yields
`truthound.mcp.administration.AuditRuntime` containing a native `AuditStorage`.
`audit_services_execute` supports separately approved `query`, `read`, `count`,
`record`, `flush` and `delete_before`. Query accepts UTC-offset-bearing
`start_time`/`end_time`, native `event_types`, `actor_id`, `resource_id`, `outcome`,
`limit` (1–1000, default 100, policy bounded) and nonnegative `offset`. Count accepts
the time/event-type filters only. Read requires `event_id`. Record requires `action`
and optionally `outcome`, `message` and JSON `data`; it creates a native **custom**
event with the fixed operator actor and native checksum. It does not impersonate
an authentication event. `delete_before` takes one explicit offset-bearing `before`
and deletes matching native records irreversibly after its own approval. No SQL,
storage paths, actor overrides or credentials are accepted from the model.

`telemetry_services_open` uses private `telemetry_services` bindings whose owner
yields `TelemetryRuntime`: native `MetricsRegistry`, `TracerProvider`, and explicit
maps of named metric/exporter instances. `telemetry_services_execute` supports:

- `collect` for native registry values; `export` requires an operator-owned
  `exporter_id` and may send externally or write through the native exporter.
- `metric` takes an existing `metric_id`, finite `value`, optional string `labels`
  and `update`: `inc` for Counter/Gauge, `set` for Gauge, `observe` for
  Histogram/Summary. Native label/type/range validation applies.
- `span` requires `span_name` and optional JSON `attributes`; it creates/ends one
  native span in this owner and returns its trace/span identity.
- `flush` takes `timeout_ms` (1–30000, default 1000); `shutdown` ends the tracer
  provider. Native false flush/shutdown results remain false. Shutdown ends further
  telemetry commands for that handle.

All administrative commands require exact approval and sensitive-result consent.
Factories own storage, credentials, exporter endpoints, filtering and provider
configuration. These tools control their resident instances, not other processes'
global telemetry or a remote security control plane. Real external audit stores and
telemetry exporters remain unverified. Tests use temporary SQLite and native local
memory exporters, including a deliberately failing flush processor.

### Bound Core security and plugin administration

`security_services_open` uses private `security_services` bindings. The pinned
async factory yields `truthound.mcp.platform_services.RBACRuntime`: a native
`RBACManager`, optional fixed `tenant_id` and explicit writable/readable
`attribute_keys` and optional additional `condition_fields`. The model cannot choose tenant, stores, credentials, authentication
callbacks or policy evaluators. A fixed tenant rejects targets belonging to another
tenant before CRUD or assignment. This manages the selected Core manager; its
permission decisions never change MCP consent, directory grants or approval requirements.

`security_services_execute` repeats the binding, live service handle, exact
idempotency key and one command; every read/mutation needs approval:

- `roles`/`principals` list native objects in the fixed tenant with `limit`
  (1–1000, default 100, policy bounded) and nonnegative `offset`.
  `role`/`principal` require the corresponding `role_id`/`principal_id`.
- `role_create`/`role_update` require `role_id` and nonempty `role_patch`.
  Fields are `name`, `description`, `enabled`, `permissions`, `parent_roles` and `conditions`.
  Creation requires a name. Permissions use the strings/objects described below;
  list fields replace their whole native set. Parent roles must exist
  within the bound scope. `permission_add`/`permission_remove` take `role_id`
  and one `permission`. `role_delete` returns the native deletion boolean.
- `principal_create`/`principal_update` take `principal_id` and a nonempty
  `principal_patch`: `name`, `email`, `enabled`, `direct_permissions`, `attributes`.
  Creation requires a name and uses the native default principal type; attributes
  are restricted to factory-declared keys. `principal_delete` returns native deletion.
- `assign_role`/`revoke_role` require both IDs. `permissions` returns native assigned/
  inherited permission inventory; it is not an authorization decision.
  `check` requires `principal_id`, `resource`, `action` and optional JSON
  `resource_attributes`; it returns native allowed/effect/reason/matches.
  `cache_clear` uses native cache invalidation.

Permission inputs accept `resource:action[:scope]` ALLOW strings or complete JSON
objects with `resource`, `action`, optional `scope` (default empty), and `effect`
(`allow` or `deny`, default `allow`). Components cannot contain `:`. This applies
to role permissions, principal direct permissions and permission add/remove.
Removing an ALLOW string does not remove a distinct DENY object.

Role conditions use `field`, `operator`, required JSON `value` and optional
`description`. Operators are the native `eq`, `neq`, `contains`, `not_contains`,
`starts_with`, `ends_with`, `matches`, `in`, `not_in`, `gt`, `lt`, `gte`, `lte`,
`exists` and `not_exists`. Edits permit caller-supplied `resource.*`,
`environment.timestamp`, projected principal id/tenant/name/email/type/enabled/roles,
factory-allowed `principal.attributes.<key>` and exact factory `condition_fields`.
Dynamic `${...}` value references obey the same allowlist. Other principal
attributes cannot be queried through a condition. Role views flag
`conditions_redacted=true` if any factory-owned conditions are outside this scope.
Do not store credentials in condition literals or descriptions.

Outputs expose defined role/principal/permission fields, including permission effects,
and omit arbitrary metadata and non-allowlisted principal attributes. Credentials
are never returned. Native workspace/store persistence and hooks remain factory-owned;
there is no implicit cross-process RBAC-store cache-coherence guarantee. The Core
manager now serializes decisions with mutations, invalidates cached grants after
changes, denies disabled principals/roles, gives explicit DENY precedence over
direct ALLOW, and treats a plain resource name as its actual resource type.
Shared ancestors in role diamonds are valid; actual cycles still fail. Native
checks apply inherited role conditions. Permission inventories/helpers are not
context-dependent authorization; use `check` for conditional access decisions.
Decision caching is limited to native unconditional role evaluators. Attribute,
ownership, time, custom or conditional-role evaluations run fresh; evaluator
registry edits invalidate cached decisions. An evaluator failure denies access
instead of authorizing from another evaluator. Cache identity components remain
separate when IDs/resources contain colons.

Native file and SQLite stores preserve DENY effects and role conditions across
reopen. Ordinary ALLOW records retain legacy strings; DENY records use structured
permission objects. Older Core readers fail on these objects and need upgrading.
Already-lost DENY/condition information in older records cannot be inferred by this
change and needs operator review. File stores reject corrupt records without
overwriting them, serialize before atomic replacement, and invalidate mutated
in-memory caches after failed persistence. Local tests include fresh native stores
and actual approved SQLite worker close/reopen; remote stores remain unverified.

`plugin_services_open` uses private `plugin_services` bindings. Its factory yields
`PluginRuntime` containing one native `PluginManager` and explicit alias maps for
plugin `classes`, native `configs`, custom `hooks`, `providers`, `reporters`,
`capabilities` and `check_factories`. The model cannot submit import paths,
distribution names, Python, arbitrary constructors or callback objects.

`plugin_services_execute` supports:

- `catalog` lists factory-selected class aliases. `load` takes `alias` and calls
  native `load_from_class` without activation. `activate`, `deactivate`, `enable`,
  `disable`, `unload`, `status` and `health` operate on that loaded alias. Native
  states and boolean health are retained; an unloaded alias must be loaded anew.
- `reload` takes a loaded `alias` and `reload_strategy="graceful"|"immediate"`.
  The factory must supply `PluginRuntime.reload_factories[alias]`, a callable
  returning a fresh native class; no code, path or loader is accepted on the wire.
  It performs native teardown/setup/activation with binding guards before each
  admission phase. Results include `reloaded`, `loaded`, `rolled_back`,
  `native_failed` and redacted `error_type`. Failed setup restores a fresh old-class
  instance only after cleanup is confirmed; external effects are not rolled back.
  Signed bindings still require explicit signature renewal. This approved command
  does not start automatic file watching.
- `ports` lists native capability/provider/check-factory names and reporter contract
  versions. `provider_register`, `reporter_register`, `capability_register` and
  `check_factory_register` register only the matching factory-provided alias object.
  Matching `provider_unregister`, `reporter_unregister`, `capability_unregister`
  and `check_factory_unregister` remove manager-local port lookups. Removing a
  provider/capability does not close a shared resource; its factory still owns
  cleanup. Reporter removal only removes a global constructor if this manager
  registered it and it has not been replaced by another owner.
- `hook_register` takes a factory-provided callback `alias`, optional `priority`
  (0–10000, default 100) and `hook_type` (default `custom`). Built-in native lifecycle
  events are supported; optional `PluginRuntime.hook_types` maps additional public
  event aliases to operator-selected native event names. `hook_unregister` removes
  the selected handler/event. `hooks` reports metadata, without callback reprs.
  `hook_fire` keeps the native legacy result list for JSON `payload`; a None value
  cannot distinguish success from a failed hook. `hook_fire_detailed` reports
  per-hook success/result/error type and `native_failed`, never exception bodies.
  This is not external-delivery proof or automatic retry authority.
- `validate` uses the same manager's registered `check_aliases` and exactly one
  retained `asset_id` or registered `provider_id`. Factories implement native
  `CheckSpecFactory.build_check_specs()`; legacy native validator classes/instances
  use Core suite compilation. Providers return native `DataAsset` objects. Suite
  name, result format, parallel/max-workers and pushdown are explicit closed fields.
  Core `ScanPlanner`/`ValidationRuntime` do the actual work, with native before/after/
  issue hook outcomes retained separately from data-quality verdicts.
- `render(alias,run_id)` consumes an actual retained canonical run through the
  registered v3 reporter. It also accepts a canonical validation `result_id` instead
  of `run_id`. Reporter class/constructor or configured object stays factory-owned;
  no template/output path, constructor settings or renderer executable is accepted.
  Rendered text/bytes and report-hook outcomes are retained in a bounded JSON artifact.
- `run_get`, `run_export` and `run_release` inspect, export or free a retained native
  result. Use the returned opaque `plugin_run_...` handle, distinct from its native
  canonical run ID. At most 16 runs are retained within the remaining byte budget.
  `run_export` creates a standard canonical validation artifact that works with
  existing reporters and `store_services_execute(command="save",result_id=...)`.
  Releasing a native run does not delete already-exported immutable artifacts.
  `provider_inspect(alias)` reads selected provider asset dimensions/backend.

The new native `PluginManager.validate`/`validate_async` and `render`/`render_async`
provide these same manager-local workflows directly in Core. Async variants await
actual hook results (including decorated async functions) while preserving the
calling thread for native provider/kernel operations. Sync variants reject a
running event loop before effects. Kernel-port validation creates no implicit
context/workspace persistence; configured providers/reporters own their resources.
Validation hooks receive immutable `CheckSpec` intent in `checks`/`validators`,
and issue hooks receive the corresponding check intent even when a native validator
name differs from its alias. Hook failures are visible, non-vetoing outcomes; they
are not converted into successful None results by the detailed API. Hook return
values crossing MCP must use the existing lossless JSON value grammar.

Retained asset/result inputs bind the owner's authorization lifetime: releasing,
expiring or revoking any consumed input closes the owner and its dependent artifacts.
All owned artifact reads synchronously reauthorize the native policy and binding,
including while a watchdog is unavailable; a background tick is not the access gate.
Native `set_execution_guard` rechecks the binding before provider/factory/report
operations and each hook; revoked authorization prevents later handlers. Plugin
unload still tears down/removes owned resources when an unload hook is denied.
Configured reporter instances remain manager-local and cannot corrupt the global
constructor registry. Normal `unload` retains `{loaded:false}`; native cleanup
  failures add `native_failed=true` and a cleanup receipt with deactivation,
  teardown completion, removal, hook failure and exception **types**, without
  exception bodies. The underlying Core `unload_plugin(name,detailed=True)` returns
  `PluginCleanupResult`; legacy calls keep their None return. A callback completing
  is not proof that a remote provider released every resource. Trusted extension code is not an OS sandbox, and a started
external effect still has no exactly-once delivery guarantee.

Each operation requires exact approval and sensitive-result consent. Registrations
apply to this resident manager and do not silently rewrite durable MCP binding files
or other workers' registries. Native plugin callbacks own unregister/teardown behavior;
release reaps the worker and cannot promise third-party cleanup. This contract does
not expose arbitrary discovery, hot reload, credential retrieval or remote security
control. Those broader native extension/admin surfaces still require separate work.

### Native execution engines and general stores

Direct Python execution engines copy their configuration and cached result values.
Row limits and offsets reject negative, boolean and non-integer values. A native
`limit=0` returns an empty value/frequency result; `limit=None` with an offset has
no hidden one-million-row cap. MCP retains its explicit 1–1,000 result limit.
SQL value pagination uses the provider limit/TOP/ROWNUM syntax; offsets without a
limit work on SQLite and use provider-specific forms elsewhere. SQL row order is
not stable without an explicit source ordering; SQL Server offset pages order by
the requested column. Remote-provider pagination remains unverified on live
services. Polars and Pandas filtering preserve native parse/source exceptions;
MCP still returns only its closed error codes. Frequency aggregation does not
reserve a user column named `count` or `__truthound_count`. Configuration row
budgets are positive integers, and explicit `to_numpy_safe(max_rows=0)` is rejected.

`execution_services_open` resolves private `execution_services` bindings. Its
approved async factory yields `truthound.mcp.execution_services.ExecutionRuntime`
with a native `BaseExecutionEngine`, optional `predicates` and `queries` alias maps
of complete operator-owned SQL strings, and optional native Polars `expressions`.
Factory configuration owns connections, backend/cluster setup and final provider
shutdown. No caller SQL, Python, import string, connection URL or expression source
is accepted. All commands use `binding_id`, `service_id`, `command` and an exact
approved `idempotency_key`. The engine starts at `engine_id="root"`.

`execution_services_execute` delegates row/null/distinct counts, completeness,
statistics, quantiles, value counts, column/sample/distinct values, aggregations,
regex/range/set/duplicate operations and cache clearing to the native engine.
`count_matching`, `count_not_matching` and `filter_by_condition` use `predicate_id`;
`execute_query` uses `query_id` with no interpolation. Query statements can have
operator-defined effects: approval must cover the complete selected statement.
Native unsupported operations remain failures rather than fallback algorithms.
`inspect` projects base execution limits and public aliases, excluding provider
configuration. Value counts are `{value,count}` rows, preserving non-string keys.

`sample`, `filter_by_condition`, and native Polars `select(columns)`,
`filter(expression_ids)` and `with_columns(expression_ids)` retain actual child
engines, at most 16. Reuse the returned `engine_id`, and `release` it when finished;
root release uses `resources_release(service_id)`. Native SQL filtering now builds
a lazy subquery view, with nested filters and samples borrowing the parent connection.
Parent/factory teardown follows child cleanup. A limit of 1–1,000, further bounded
by operator `max_items`, controls value/sample/table/NumPy results. `to_numpy`
rejects a row count above the requested limit; `to_numpy_safe` delegates native
sampling. Native engine limits still apply. `to_polars` returns bounded rows from
the native lazy frame. NaN, timestamps and other supported values use the existing
lossless tagged-value encoding. Local Polars/SQLite and actual local Dask, Ray and
Spark SDK tests do not verify remote SQL or remote cluster deployments.

Install `truthound[execution-dask,mcp]`, `truthound[execution-ray,mcp]` or
`truthound[execution-spark,mcp]` for the selected runtime from this unreleased
checkout. Spark also requires an operator-provisioned Java 17+ runtime and a
correct Python worker interpreter; those are not installed by pip. Distributed engine
commands include native partition counts, `repartition`/`coalesce` with 1–1,000
partitions, metrics/clear, and bounded Arrow conversion. Native Dask supports
`persist`, selected-column engines, alias-selected `native_filter`, head/tail and
`compute`; no model scheduler address, path or query expression is accepted.
Ray ports expose native materialize, selected columns, take and factory-owned
`row_filters` selected by `filter_rows(row_filter_id)`. Spark ports expose native
persist/unpersist, factory-configured checkpoint and alias-selected SQL returning
a child engine. Unsupported backend commands remain explicit failures. Factory
cleanup owns shared client/cluster shutdown; releasing a child does not shut down
a shared cluster. Persist/checkpoint can write to operator-configured storage and
are included in exact approval effects. Full compute/Arrow conversions preflight
the native row count against `limit`; select/sample first for larger inputs.
Arrow output preserves the native Arrow schema, including backend index columns.
Metrics expose counts and failure flags without native error/warning text.

Use the approved root `shutdown` command when native teardown completion must
be acknowledged. It closes retained children, the root engine and its factory
before returning; a teardown error or deadline expiry cannot become success.
The same shutdown approval key replays its retained result. Other engine calls
then fail with `invalid_handle`; release the service resource afterward.
`resources_release` remains bounded forced cleanup and alone does not prove that
every native factory finalizer completed. Process supervision records owned
descendants while the driver is alive, preserving cleanup ownership after driver
failure. No global Ray/Spark stop command is used.

Ray and Spark samples return `min(n, row_count)` rows using randomized native
ordering; this can require a shuffle. Ray `get_stats.count` excludes null values.
Spark distinct aggregation uses the same non-null semantics as its
`count_distinct` method; Ray's native distinct list includes null. Spark quantiles
retain native approximate rank semantics with relative error 0.01, accepting
integer probability endpoints. Ray exact median reuses Core's Polars bridge and
its configured conversion row bound; it is not an unbounded distributed median.
Spark uses native median. PySpark's legacy `MEMORY_ONLY_SER` name maps to its
serialized `MEMORY_ONLY` level; unknown native storage-level names are rejected.

Standard NumPy integer, boolean, string, bytes and float scalars up to 64-bit are
normalized to exact Python values before existing tagged-value encoding. Large
unsigned counts and NaN/Inf remain lossless. Arrays, complex/temporal scalars and
extended-precision floats are not silently coerced. Distributed FIRST/LAST now
select the corresponding operation rather than a row-count fallback. Their order
is the backend's native execution order; this creates no global ordering contract
for unordered distributed input. Actual Dask tests cover partitions, aggregation,
null/empty first/last, selected filters, persist, Arrow and child cleanup.

`store_services_open` resolves private `store_services` bindings to an approved
async factory yielding `truthound.mcp.store_services.StoreRuntime`. It selects a
native `BaseStore`, `CachedStore`, `BatchedStore` or `ReplicatedStore`, optional `records` alias map
of native persistence `ValidationResult`/`ExpectationSuite` objects, `data_assets`
aliases and a fixed version actor. The factory owns backend, namespace, serializer,
compression, credentials, native policy/wrapper configuration and store close.
`VersionedStore` is a native `BaseStore` subclass; its `config` reflects the wrapped
backend. No connection or storage path can be supplied through the tool.

`store_services_execute` supports native initialization, save/get/exists/delete,
query/list/count, bounded clear, latest/history/failure reads, suite operations,
and paged iteration. Filters select `data_asset_id`, ISO datetimes, status, tags
and native ordering. Pagination uses `limit` and `offset`; iteration also accepts
`batch_size`, and Core iteration honors the exact overall limit including zero.
Record IDs are owner-local opaque references; after reopening a persistent store,
list/query it again for fresh references. Durability is the selected backend's:
this contract does not make memory stores or native in-memory version metadata
persistent.

`save` accepts exactly one `record_id`, retained canonical validation `result_id`,
or a declarative `suite`. A retained result is converted using native
`ValidationResult.from_validation_run_result`, with its full native persistence
payload preserved. The service retains and reauthorizes the result's input grants;
releasing/expiring/revoking that input closes the owner and dependent artifacts.
A `suite` selects `data_asset_id` and supplies a safe suite name and native
expectations/settings/tags. This is the persistence expectation DTO, distinct
from the immutable kernel `ValidationSuite`. Record views expose native verdicts,
statistics, timestamps, tags and expectation settings, while omitting arbitrary
record metadata, runtime environment, issue messages/details and physical asset
identifiers. Version views omit operator metadata/authors/messages; native diffs
return structural paths/types without value payloads or private metadata paths.

Versioned stores support `version_history`, `version_info`, `latest_version`,
`count_versions`, `diff`, `rollback`, versioned get/exists/delete and save with
`expected_version`/`message`. Cached stores expose native invalidate/all, refresh,
warm, get-many and statistics. Batched stores expose add/start/stop/flush/statistics;
`save` remains the native immediate write. Background batch writes recheck the
pinned binding before effects. Pending native batches are not a durable journal,
and crash/failed/uncertain flush is not proof of persistence. Wrapper-specific
operations reject incompatible stores rather than bypassing wrapper semantics.
Filesystem stores support native `rebuild_index`.

Native `RetentionStore` adds `retention_cleanup(dry_run=true)`, info, statistics,
history and estimate. Setting `dry_run=false` needs exact approval for the native
policy's delete/archive/compress/tier action. Result projections preserve counts,
`dry_run`, `error_count` and `native_failed`, without native error text. Core cleanup
history/statistics include preview runs; their counters are not independent proof
of actual deletion. The native retention schedule runs on store operations, not an
MCP-installed daemon. `RetentionStore.config` reflects its wrapped backend.

Native `TieredStore` adds list/info/statistics, `tier_move(record_id,tier_id)` and
`tier_run(dry_run=true)`. `StoreRuntime.tiers` maps public aliases to every configured
native tier name. Preview and actual migrations return opaque record references,
public tier aliases and native counts/failure counts. Physical tier metadata and
error messages are omitted. Placement durability follows the configured native
metadata store, whose default is memory.

`ConcurrentFileSystemStore` adds `concurrent_apply(actions,dry_run=true)` and
`concurrent_stats`, and supports its native `rebuild_index`. Each of 1–1000 actions
has `kind="save"` or `"delete"` and an opaque `record_id`; save selects an actual
factory record or an already stored native validation DTO. Preview fills and
rolls back the native buffer, without validating serialization or performing the
pending file changes. `dry_run=false` commits within this single exactly approved
call; locks are not retained between MCP calls. Receipts report native buffered
or applied operation counts, not an independent proof of full crash atomicity.
Statistics contain aggregate lock values, without backend paths.

Native file locks now span the associated index mutation. Index commits reload
committed state under the interprocess lock, preserve disjoint owners' changes
and reject conflicting additions. Corrupt indexes/WALs retain evidence and fail;
valid WAL recovery starts from existing committed entries. Disk commit failure
blocks reuse of the uncertain index owner. Lock detection follows incompatible
thread wait cycles; ordinary contention is allowed to wait.

A native batch context rolls back uncommitted buffers on exit. A failure before
index commit stops later operations and restores touched files while their locks
remain held. An uncertain index commit or incomplete restoration retains private
`.batch-recovery-*` backups with original hashes and a file mapping manifest;
the selected store and a reopened store are quarantined pending explicit operator
reconciliation. MCP records `outcome_unknown`, closes that owner and does not
replay it. `concurrent_recovery_list(limit,offset)` delegates native recovery
inspection and returns session-owned opaque `recovery_id`, phase and file count.
The worker keeps physical directories, original hashes and the inspected state
fingerprint private. `concurrent_reconcile(recovery_id,resolution)` requires a new
exact approval: `rollback` restores original bytes and removes originally absent
files; `keep_current` retains actual files, including partial effects, and
reconciles only their index entries. It does not resend missing saves. Native
`list_batch_recovery` and `reconcile_batch(...,expected_fingerprint=...)` recheck
manifest/backup integrity, safe paths and the current file fingerprint under the
batch-owner and affected-file locks. A changed state requires fresh inspection
and approval; failure retains quarantine and evidence.

The MCP extra includes `filelock` so default cross-platform store recovery has a
real process-locking provider. Install `truthound[stores-concurrency]` in the
operator environment to select both `filelock` and `portalocker` explicitly.
Native filelock zero/nonblocking timeouts remain zero; portalocker uses its actual
lock API with bounded polling, unique shared-reader descriptors and complete open
failure cleanup. Fcntl retains its lock-file inode after release so queued and new
owners cannot split ownership across different files. Actual local provider
contention and three-process owner handoff are tested; Windows runtime proof still
belongs to the final platform CI.

One stable native batch-owner lock fences publication, execution and recovery.
Live owners block reconciliation; interrupted owners release their OS locks.
`preparing` receipts precede backup copying and all file changes, so an explicitly
resolved preparing batch leaves current data intact. Completed data/index state
moves atomically to a private `.batch-resolved-*` namespace before any backup
deletion. A crash during deletion leaves usable data and a `cleanup_pending`
entry; `concurrent_recovery_cleanup(recovery_id)` separately approves the remaining
cleanup through native `cleanup_batch_recovery`. Cleanup fingerprints are also
pinned privately. Native no-op locking cannot provide process fencing and is
rejected for recovery. Manifest reads are capped at 16 MiB and 100,000 entries.
Pending entries count affected data files; cleanup entries count remaining evidence
files. These are explicit reconciliation and cleanup controls, not automatic
multi-file crash atomicity or an external-delivery guarantee. Actual local process
crashes before index commit, after index commit and during cleanup are tested;
power-loss durability and remote backends are not established.

`ObservableStore` uses the same factory-owned context, audit backend, metrics and
tracer for ordinary store operations. Native initialization now delegates through
`BaseStore` correctly, save/query operation counters increment once, and shutdown
attempts audit, tracing and owned metrics-server cleanup even if another close
fails. Underlying backend close failure also runs observability cleanup.

`audit_query` accepts optional `record_id`, `audit.start_time`, `audit.end_time`,
`audit.event_type`, `limit` and `offset`. Native ISO timestamp comparison rules
apply. The bounded scan requires `offset + limit <= 10,000`. Results include native
timestamp, event type, status, duration and opaque record reference; actor/tenant,
physical IDs, trace IDs, summaries, arbitrary metadata and error text are omitted.
`metric_read(metric_id)` selects `StoreRuntime.metrics` aliases to trusted
`StoreMetric(name,labels,kind="counter")` queries. The native in-memory or
Prometheus backend returns its actual counter/gauge scalar; names and labels stay
private. These queries do not expose arbitrary metrics or raw Prometheus exports.
`observability_flush` calls native audit/tracer flushing; it does not prove remote
export delivery. Factory-owned exporter configuration and external services remain
unverified.

`MigratableStore` supports `migration_status`, `migration_validate(record_id)` and
`migration_run(dry_run=true,batch_size=100)`. Batch size is 1–1000; callbacks,
registry, target/minimum versions, retry policy and required backups are fixed
by the approved factory. Status retains native version distributions, estimated
migration need, sampled failures and in-memory migration counters. Validation
checks the native migration path; preview does not run custom migration callbacks
or prove their output is valid. An actual run processes the native store and can
partially commit before a failure or timeout. Receipts retain counts, dry-run and
success verdicts, with `native_failed` and error counts; raw record IDs, errors and
callback data are omitted. No automatic replay is added.

Core persists migration versions in `ValidationResult.metadata`, so reopening a
store does not apply a completed migration again. Required backup failure or a
false migration validator prevents rewriting the original; lazy write failures
and incomplete queries propagate rather than appearing successful. Original
backups remain readable and are excluded from later migration passes. Existing
conflicting backup identities fail instead of being overwritten. The native
`set_io_guard` callback rechecks authorization before callbacks and writes.
Lazy reads may themselves create backups and rewrite data under factory policy;
those effects are included in exact operator approval. Backup storage is native
backend state; migration counters remain process-local, and this is not a
transactional all-store rollback or a durable migration scheduler.

Native `ReplicatedStore` adds start/stop/flush/statistics and full synchronization,
optionally selecting `replica_id`. `StoreRuntime.replicas` must alias every configured
target. Save uses native `save_async`; sync/semi-sync replica failure raises after
the primary may already have been written. This remains `outcome_unknown` in MCP,
closes the owner and cannot safely replay. Native batch/full-sync failures stay
failed, and failed full sync leaves the target unhealthy for read routing. Native
`stop`/`flush` drain instance-owned async work and queued writes. `set_io_guard`
checks authorization inside native primary/replica I/O, including background
replication and health checks; retries cannot bypass revoked authorization.
Native retry and best-effort delete policies still apply. Async queues are memory,
not durable receipts; successful primary save is not a replica delivery guarantee.
Statistics use public replica aliases, including lag and success-rate map keys.
Local temporary-store tests exercise these effects and partial-write quarantine.
They do not establish actual cloud replication, regions, provider credentials or
cross-process metadata recovery.

### Bound tenant administration

`tenant_services_open` uses private `tenant_services` bindings whose async factory
yields `truthound.mcp.tenancy.TenantRuntime`: one native `TenantManager`, a public
`tenants` alias map to fixed native IDs, and optional `settings` aliases to native
`TenantSettings` objects. The factory owns store, resolver, quota tracker, hooks,
physical database/schema names and notification configuration. This administers
the selected Core manager and does not grant MCP data access or change its consent.

`tenant_services_execute` repeats binding/service/key and needs exact approval:

- `catalog` lists tenant/settings aliases with `limit` (1–1000, default 100,
  policy bounded) and nonnegative `offset` per collection. Other calls select a
  `tenant_id` alias; physical IDs and tenants outside this map are refused.
- `get`, `delete`, `suspend`, `activate` use the native lifecycle; suspend accepts
  optional `reason` up to 4096 characters. Delete also invokes native usage reset
  and the factory's configured lifecycle hooks.
- `create`/`update` take nonempty `patch`: `name`, `owner_id`, `owner_email`,
  `isolation_level`, quota settings and operator `settings_id`. Creation requires
  a name and generates its slug; `slug` is editable through update. Optional create
  `tier` uses native defaults. `set_tier` requires `tier` and optional `update_quota`
  (default true), preserving the native choice to reset quotas to tier defaults.
- `patch.quota` accepts partial native quota fields: validations per day/hour,
  rows per validation, storage bytes, API calls per minute/day, checkpoints per day,
  maximum retention days/concurrent jobs/users and `custom_limits`. Integer values
  are bounded to -1..2^63-1; native tracker semantics determine enforcement.
  Partial create quotas extend native tier defaults; updates preserve other fields.
- `quota_check` and `usage_track` require a native `resource` name and positive
  `amount` (default 1, maximum 10^9). Check uses the native decision; track records
  usage and does not implicitly perform a quota-checked workload. `usage_summary`
  selects `resource` and `window` (`hour`, `day`, `month`, default day), returning
  native counters/limits/flags with the public tenant alias.
- `context` enters/inspects/exits one native tenant context inside this worker.
  It does not persist a tenant context into other owners. Resolver integration and
  dataset row isolation require their separate explicit operations below.

Views expose defined tenant status/tier/owner/quota/time fields, omitting arbitrary
metadata, physical storage names, settings endpoints and credentials. No model
callback/import/path/store configuration is accepted. Native persistence is
provider-selected: local tests cover actual SQLite owner close/reopen, status,
quota and scoped context, including suspension metadata restoration. Usage tracker
state is independent; the default memory tracker ends with its process. Native
zero/unlimited tenant quotas do not report an exceeded summary. File/remote tenant
stores and distributed trackers are not established by this SQLite evidence.

### Native rate budgets and redacted secret administration

`rate_limit_services_open` uses private `rate_limit_services` bindings whose pinned
async factory yields `truthound.mcp.governance.RateLimitRuntime`. It owns a native
`RateLimiter`, public-to-native `subjects` aliases, optional opaque `contexts`,
native `policies`/`quotas` aliases, and the same `PolicyRegistry`/`QuotaManager`
instances supplied to the limiter. Storage, extractors and configurations remain
operator-side. These budgets do not change MCP rate limits or consent.

Every `rate_limit_services_execute` request repeats the binding, live service ID,
exact idempotency key and command, with its own approval:

- `catalog` lists aliases; `config` projects native scalar settings without callbacks.
- `acquire`/`peek`/`reset` require `subject_id`; callers cannot supply native keys.
  Acquire takes `tokens` (1–1000000, default 1). `wait=true` requires
  `timeout_seconds` in (0, 10]; nonwaiting requests omit it. Optional `context_id`
  selects an operator object for native policy evaluation. Native `peek` semantics
  apply; it is not a simulation of consuming tokens through every policy.
- `quota_register` takes an operator `quota_id`. Registered quotas then support
  `quota_check`, `quota_consume`, `quota_usage` and `quota_reset`, with `subject_id`
  and `quota_id`; check/consume use `tokens` as the amount. Missing registration
  is refused. Consumption returns native usage; exceeded consumption can raise a
  native error and quarantine the handle rather than report a successful mutation.
- `policy_register`/`policy_unregister` select `policy_id`. `policy_resolve` selects
  `context_id` and returns native match/configuration with public policy aliases.

Decisions expose native allowed/remaining/limit/reset/retry/utilization and native
HTTP headers. Native bucket keys and arbitrary policy metadata are omitted. Exact
successful replay returns its retained artifact and does not consume twice.
State persistence is the selected Core storage/manager's semantics: memory budgets
end with their owner; durable/distributed recovery is not supplied by MCP receipts.
Local tests use native memory storage and quotas; remote Redis/Memcached are unverified.

`secret_services_open` uses private `secret_services` bindings yielding
`truthound.mcp.governance.SecretRuntime`: a native `SecretManager`, fixed public
reference aliases to native `SecretReference` objects, and provider aliases to
`SecretProviderSlot(provider, name, priority)`. Factories own physical keys,
provider endpoints, versions, default values and credential resolution. Model
requests cannot supply these fields or secret values.

`secret_services_execute` supports `catalog`, `resolve(reference_id)`,
`cache_clear` (all or selected `provider_id`) and `provider_add`/`provider_remove`
for an existing `provider_id`. Resolution performs the native lookup but returns
only `resolved`, and when present, `expired`, `created_at`, `expires_at`. It never
returns value, hash, length, physical key/provider name, URI or arbitrary metadata.
`resolved` describes native lookup/cache state, not proof that credentials are
still accepted by an external service. Native consumers resolve/use credentials
inside operator-selected factories; raw `get_value`/resolved-config output is
not exposed through this transport. Close clears the selected manager's caches;
the factory owns external client cleanup. Other owners' caches are independent.

Both catalogs accept `limit` (1–1000, default 100, policy bounded) and nonnegative
`offset` per collection. All calls require sensitive-result permission and exact
approval. Provider failures never return native exception text and do not replay
effects automatically. Core also emits a generic lookup warning without provider
exception material; non-strict absent providers return `None`. Tests use synthetic
local providers, verify actual cache hits/clearing and factory cleanup, and do not
read production environment secrets or invoke cloud secret services.

### Native schema audits

`schema_audit_contract` retains the native document JSON Schema. Pass strict
UTF-8 JSON as `document_json` to `schema_audit_create`; this produces an owned
document artifact. The overall operation input limit remains 64 KiB. Unknown
fields, duplicate keys, invalid references and wrong document kinds fail.
`schema_audit_inspect` takes `resource_id` and returns canonical JSON and raw and
semantic digests. `schema_audit_standards` takes `standard_id` and `design_id`;
`schema_audit_implementation` takes `design_id` and `catalog_id`. All findings,
rule coverage and native verdicts are retained, including UNKNOWN. Empty or
incomplete coverage is not promoted to PASS. No database connection is made.

Optional term mappings are operator-owned entries under `schema_audit` in the
private `bindings.json`, not model-supplied approval claims. Using `binding_id`
requires `idempotency_key` and approval of the exact plan. Input digests and
the binding file identity are checked; changing the operator bindings revokes
access to the resulting artifact. Documents and reports remain subject to
session TTL, quotas, parent release and sensitive-result disclosure policy.

### Native lineage graphs

`lineage_track` takes `graph_id` and an `event` with `action` equal to `source`,
`transformation`, `validation` or `output`. It calls the corresponding native
`LineageTracker.track_*` method and returns a new graph handle, retaining the
original and its ownership chain. All events take `name`, optional `description`,
`owner`, `tags` and JSON `properties` (which cannot override named arguments).
Sources take `source_type`, `location`, and `schema_spec`; transformations take
`sources`, `operation`, `location`, `schema_spec`, and `column_mapping` whose values
are lists of `[source_node, source_column]` pairs. Validations take `sources`,
`validators` and `result_summary`; outputs take `sources`, `output_type` and
`location`. Locations are metadata: nothing is opened, executed or published.
Native node replacement and omission of edges from missing source IDs are
preserved, not silently reinterpreted. Native-generated timestamps are retained.
This is manual tracking, not automatic execution history, remote emission or a
claim that the described validation actually ran.

The additional `dependency` event takes only `source_id`, `target_id`, and
`edge_type` (default `depends_on`), calling `LineageTracker.add_dependency`.
Missing nodes and cycles fail without modifying the original graph.
`lineage_query` also accepts `view: "tracked_lineage"` with `node_id` and
`direction` (`upstream`, `downstream`, or `both`, the default), or
`view: "tracked_path"` with `source_id` and `target_id`. These call the native
tracker query methods without reinterpreting their edge-direction behavior;
an absent path, including a missing endpoint, is JSON `null`, not an empty graph.
All resulting artifacts retain parent ownership and revocation checks.

Large relation schemas can be provided through approved operator validator
bindings; the 64 KB tool-request limit is unchanged. Recursive relation traversal
can raise a native `RecursionError` on deep schemas. MCP preserves that execution
failure and canonical evidence rather than reporting a successful validation.
Parity tests normalize only repeated traceback frame counts that differ with
interpreter call-stack depth; exception types, messages and frames remain checked.

`lineage_load` accepts an approved local `.json` `path` and optional native
`config` (without `persist_path`). It captures an immutable, size-bounded file
snapshot before calling `LineageGraph.load`; imported graphs are limited to
1,000 node records and 2,000 edge records. Invalid/empty native files fail rather
than becoming successful empty graphs. Both loading and exporting require
sensitive-result permission. `lineage_export` takes `graph_id` and returns an
owned `application/json` artifact with the exact UTF-8 bytes produced by
`LineageTracker.export_to_json`. This is native JSON, **not** the tagged-value
format of `lineage_query`. Use paged `artifacts_read`, Base64-decode and concatenate
the bytes; preserve them when writing a file for later native/MCP re-entry.
There is no caller-selected output path or implicit filesystem publication.
Origin permission revocation and parent release invalidate derived access.

`lineage_catalog` describes all native config/node/edge fields and rich record
definitions. `lineage_create` accepts `config`, `nodes` and `edges`; caller
`persist_path` is rejected. `lineage_edit` takes `graph_id` and a discriminated
`edit`: `add_node`, `remove_node`, `add_edge`, `remove_edge` or `clear`. Edits
produce a new graph handle and leave the source immutable.
`lineage_query` takes a discriminated `query` containing `graph_id`, `view` and
view-specific fields. Views are `graph`, `nodes`, `edges`, `roots`, `leaves`,
`topological`, `node`, `has_node`, `node_edges`, `upstream` and `downstream`.
Traversal depth and edge direction are explicit inputs.

Impact tools take `graph_id`, `node_id` and optional native `impact_rules`.
`lineage_impact` adds `max_depth` and `include_validations`;
`lineage_schema_impact` adds `new_schema`. `lineage_dependency_chain` instead
takes `source_id` and `target_id`. All queries/analyses retain full native
artifacts. Hypothetical deletion never deletes the graph or external data.
`lineage_impact` defaults to `view="full"`. `view="summary"` delegates to
native `ImpactResult.summary()`, `critical` to `get_critical_nodes()`,
`by_level` requires `level=none|low|medium|high|critical`, and `by_type` requires
the native `node_type` (for example `validation`). Selectors not applicable to
the chosen view are rejected. Each call analyzes the captured graph anew with
the same depth, validation-inclusion and impact-rule settings; it does not
filter a previously stored result. Text and node lists remain owned sensitive
artifacts, and releasing the source graph revokes them. For example,
`{"graph_id":"graph_<token>","node_id":"a","view":"by_level","level":"high"}`.

`lineage_render` takes `graph_id`, `renderer=d3|cytoscape|graphviz|mermaid`,
closed native `RenderConfig` fields in `settings`, optional `root_node_id` and
`direction=upstream|downstream|both`. For subgraphs, `settings.max_depth` controls
the traversal. It returns an owned JSON (D3/Cytoscape) or text (DOT/Mermaid)
artifact. Partial graphs preserve native graph insertion order across processes.
For example, `{"graph_id":"graph_<token>","renderer":"mermaid","settings":{"orientation":"LR","theme":"dark"}}`.
`output_format="native"` preserves these JSON/text outputs and supports the rooted
subgraph selectors above. Whole-graph `output_format="html"` delegates to native
D3, Cytoscape or Mermaid HTML. Mermaid also supports `output_format="markdown"`
and optional `title`. Graphviz supports `output_format="svg"|"png"`; install
`.[mcp,lineage-visualization]` and a system `dot` executable. Other format/renderer
pairs and rooted HTML/image requests are rejected. Width and height for document
and image requests are bounded to 1–4096; retention uses the existing byte budget.

No caller output path or browser launch is accepted. Graphviz images execute the
local native renderer; the server does not fetch remote resources. HTML includes
interactive client code and public library URLs which a browser may fetch when
an operator opens it. Artifacts require the sensitive-result grant, and releasing
the graph revokes their reads. Labels remain data and distinct node identities
are preserved across diagram formats. Native D3 fixed layouts retain deterministic
positions; native Cytoscape honors selected sizes. Subgraph depth inherits native
`RenderConfig.max_depth` unless explicitly overridden; full-graph documents do
not select a subgraph. Tracker context/history sessions and OpenLineage delivery use their separately
approved owners described below.

### Persistent contexts and operator approval

`suites_create` retains a version-1 intent, not executable Python. Each check has
a unique `check_id`, concrete `validator`, closed `settings`, optional name,
category, tags and metadata. `schema_id` may reference a retained schema;
schema-only suites are supported and reserve check ID `schema`. Releasing that
schema revokes its dependent suites and results. `evidence_policy` and
`severity_policy` use the native kernel policies. Execute with `suites_validate`
and `asset_id`/`suite_id`; these are explicit kernel suites, not auto-suite
baseline inference. Persistent execution additionally requires `context_id`,
an idempotency key and exact operator approval.

`contexts_execute` takes `context_id`, `idempotency_key` and a discriminated
`operation` object. Supported `action` values include `resolve_source_key`,
`resolve_fingerprint`, `track_asset`, `get_or_create_schema_baseline`,
`read_applied_suite`, `write_applied_suite`, `persist_run`, `persist_docs`,
`metric_get`, `metric_put`, `metric_append_history`, `list_runs`, `list_docs`,
`list_baselines` and `list_applied_suites`. Use `capability_describe` for the
exact action fields. Even reads require approval: constructing the native
context can create directories or quarantine corrupt state. Persistence accepts
only owned canonical validation artifacts; applied-suite actor metadata does not
grant approval. Releasing a context handle does not delete its user workspace.

Workspace permission is separate from input permission. Start the server with
`--workspace-dir /absolute/project --control-dir /absolute/private-control` as
well as `--allow-dir`. Create the control directory outside both data and
workspace roots; on POSIX it must be user-owned with mode `0700`.
`contexts_open` reads project configuration or an explicit typed `config`, without
creating `.truthound`. Open a context, then request `contexts_initialize` with
`context_id` and an `idempotency_key` of 8–128 characters. Alternatively supply
both fields to `assets_validate` to execute and persist a real Core project run.

The response initially contains an `awaiting_approval` plan. A human operator,
outside the agent's MCP tools, reviews the exact request and then approves it:

```bash
truthound-mcp review-plan --control-dir /absolute/private-control --plan-id plan_<returned-token>
truthound-mcp approve-plan --control-dir /absolute/private-control --plan-id plan_<returned-token>
```

The dedicated CLI also provides `revoke-plan`. Approval binds operation, request,
session, grants, workspace state, effects and expiry. Retry the identical tool
request only after approval. Changed grants/workspace or forged consent fail
closed. The control directory contains private approval/signing state, not a
public artifact; do not share it or expose it through MCP. It is an application
authority boundary, not protection from a hostile process running as your user.

Persistent validation calls the real Core context, including configured catalog,
baseline, run, metric-history and docs effects. A committed intent replays its
receipt rather than writing again. If its old session artifact expired, the
receipt explicitly reports `artifact_available=false`; it does not recreate the
run. Timeout/cancellation after effects begin is `outcome_unknown`, never a
successful rollback or permission to retry. Automated crash reconciliation is
not yet implemented. Retained validation's aggregate quality remains `unknown`
until complete target/skip accounting is established; canonical Core outcomes
remain available in the artifact.

### Reviewed native validator bindings

An operator may place `bindings.json` in the protected control directory (on
POSIX, user-owned mode `0600`). This is executable configuration, not an agent
tool or a file in an admitted data directory. For example:

```json
{"validators":{"capital-a":{"validator":"regex","settings":{"pattern":"^A+$","columns":["code"]}}}}
```

Use `checks=[{"validator":"regex","binding_id":"capital-a"}]` and an
`idempotency_key` with `assets_validate`. The agent cannot override fixed
settings or approve the resulting plan. Review and approve with the CLI above.
Requests/receipts retain constructor digests rather than raw SQL or settings.
Approval binds the exact template bytes, grants, session and effects. A changed
or revoked binding denies dependent artifact access. Replay preserves the
authorization/configuration identity without granting new disclosure rights.

Non-JSON inputs use installed Core plugins, not pickle or imported model text.
The operator declares a plugin's installed distribution, exact version,
`truthound.plugins` entry-point name, file `sha256`, public capability mapping
and optional plugin settings in `plugins` within the same protected file.
Obtain its digest without importing plugin code:

```bash
truthound-mcp plugin-fingerprint --distribution your-reviewed-distribution
```

Validator templates can supply typed callable fields with
`{"operator_capability":"approved-capability-id"}`. Their separate
`object_parameters` map can bind a constructor field to an approved native
capability object such as a complete Polars expression, dtype mapping or
reference frame. Fixed JSON, object parameters and caller settings cannot
overlap. Ordinary JSON objects are never interpreted as executable references.
No extension imports occur before approval. An owned worker verifies the
distribution/version/files again and uses Core `PluginManager` load, activation
and shutdown. Discovery/automatic installation are disabled. Package changes
block execution; uncertain effects are not replayed as success. Editable
packages and binary-only entry points are unsupported. Approved extensions run
with the host user's privileges; this is **not an OS sandbox** or a transitive
dependency lock. Review extension code and data/network access before approving.

Approval does not make an object type-correct for its constructor field.
Ordinary malformed JSON settings are rejected before execution; an approved
native object can instead be accepted by a Core constructor and fail during
validation. A captured Core execution failure remains `core_execution_failed`,
with its canonical result preserved; retrying its idempotency key does not
execute the extension again. This is distinct from constructor rejection or
uncertain effects, and approved-object error coverage does not establish
ordinary-input algorithm error coverage.

Discovery reports `full_core_parity=false`: the implemented operations and
approved native services below have scoped regression evidence, rather than
exhaustive algorithm, configuration and provider equivalence. Native tracker
contexts/history/rendering, checkpoint services, streams, stores, plugin lifecycle
and durable jobs are exposed through their documented contracts. Read each
service's configuration, persistence, recovery and cleanup limits; a registered
operation does not establish every provider's behavior. Live external services
and recovery after forced owner loss require their own evidence. No new remote
server, credentials or paid calls are introduced.

Parity evidence separates constructor rejection, native execution failure and
exceptions caught by an algorithm. For example, Core query/expression validators
may encode a query exception as a critical quality issue rather than an
`ExecutionIssue`; MCP preserves that native distinction and the complete issue.
Constructor rejection across every registry entry does not prove every
algorithm-specific runtime error branch.

Evidence also distinguishes native exceptions swallowed without a reported
issue. Tests observe the real exception in the exact native method, with a
nonexceptional control and no injected failure, then compare complete Core/MCP
results. Such behavior is not relabelled as an execution failure or used to claim
data quality passed; aggregate quality remains unknown. This is separate from
exceptions returned as critical quality issues.

## Claude Desktop bundle

For the published release, [download the 3.2.0 MCPB](https://github.com/seadonggyun4/truthound/releases/download/v3.2.0/truthound-3.2.0.mcpb), then select **Settings → Extensions → Advanced settings → Install Extension…** and choose the file and allowed directory. See the [official installation guide](https://support.claude.com/en/articles/10949351-getting-started-with-local-mcp-servers-on-claude-desktop). This is a custom extension installation, not a confirmed Claude directory listing.

Build the bundle from the matching installed release:

```bash
truthound-mcp bundle --output truthound-3.2.0.mcpb
```

The `.mcpb` contains a manifest, a small launcher, dependency declaration and
license. Its UV runtime installs exactly `truthound[mcp]==3.2.0` from PyPI and
resolves supporting dependencies. It needs network access on first setup and a
Claude Desktop version supporting MCPB manifest 0.4 and the UV runtime. This is
a deterministic bundle archive, not a fully locked offline environment.
The generated bundles are unsigned; installation still requires host consent.

Before publishing that version to PyPI, embed its built wheel instead:

```bash
truthound-mcp bundle --wheel dist/truthound-3.2.0-py3-none-any.whl --output truthound-3.2.0-local.mcpb
```

Install the file through Claude Desktop's extension installation UI and select
an allowed data directory. Package installation and filesystem permission consent
are separate steps. Directory listing/review is separate from file distribution;
building a bundle does not establish Anthropic approval or tested GUI behavior.
The bundle never contains project data, development environments or internal notes.

## OpenAI local plugin and repository marketplace

The standalone OpenAI plugin is version `0.1.4`, pins the published Core
`truthound[mcp]==3.2.0`, and includes a frozen UV dependency lock. It is local
desktop software, not a hosted service or a ChatGPT web connector. Install UV
and make it available on the desktop application's PATH. First setup needs
network access and a compatible Python 3.11–3.13 runtime.

Add the repository marketplace once:

```bash
codex plugin marketplace add seadonggyun4/truthound --ref main
```

Refresh or restart the desktop app if needed. In **Plugins**, select the
**Truthound Plugins** source, open **Truthound Local**, and choose **Install**.
Then choose **Set up** and approve the exact absolute data directory. In the
CLI, installation is also available through `codex plugin add truthound@truthound`.
Client version and organization policy can affect marketplace availability.
Packages include the portable manifest and a matching Codex compatibility
manifest for older clients. If **Set up** is unavailable, ask the agent to run
the packaged **Truthound setup** skill; do not bypass directory approval.
The MCP command uses a plugin-relative `cwd` instead of depending on host-specific
argument placeholder interpolation. Version 0.1.1 fixes 0.1.0 startup on Codex 0.141.0.
Version 0.1.2 uses the standard `./` prefix, with both portable manifests checked
against vendored official Agent Plugins 1.0.0 schemas before ZIP creation.
Version 0.1.4 fills the square icon and logo canvas with orange, without the
previous gray surround or baked-in rounded tile. The host controls its own
corner clipping and padding. Existing installations may keep
the previous image until the plugin is updated and the client refreshes its cache.
The OS validation workflow checks actual Codex installation and normalized runtime
configuration as well as the frozen UV launch. On Windows, verification invokes
npm's Codex JS entry point through Node rather than shell-evaluating a `.cmd` shim.

Installation grants no data access. The three MCP tools initialize normally but
return `access_denied` until you approve a directory. Setup runs the packaged
`permissions.py` only after your approval. It stores an atomic local allowlist
at `~/.truthound-openai/permissions.json`, outside the plugin and data workspace.
The next tool call reads the updated policy; no server restart is required.
Re-running setup replaces the previous directory list; `permissions.py --revoke`
revokes access. Revocation affects subsequent calls, not an already-running call.
Missing, malformed, unsafe or empty permission files deny access. POSIX ownership
and write modes are checked; Windows uses the user's directory/NTFS permissions,
not a claim that POSIX modes enforce Windows ACLs.

The server reuses Core's MCP transport, service, file policy and worker APIs;
it adds only the local consent loader. No HTTP listener, upload tool, OAuth,
credentials or paid model call is introduced. Raw values are redacted, but file
paths and column names are tool metadata the host may send to its model provider.
The optional `TRUTHOUND_OPENAI_CONFIG` absolute-path override is for an
operator-selected alternative consent file or isolated tests, not implicit access.

### OpenAI official directory approval

Repository distribution is **not** official public directory publication.
The current plugin has no OpenAI approval or assigned public listing ID.
To request a public **Desktop only** listing:

1. Have the submitting organization owner contact an OpenAI representative for
   local MCP partner eligibility and organization allowlisting; wait for confirmation.
2. Complete developer/business identity verification. The submitter needs owner
   access or **Apps Management Write**. Do not send credentials in chat or the ZIP.
3. Prepare real publisher-owned privacy and terms URLs, review metadata, five
   positive and three negative cases, and an accessible video walkthrough. The
   repository package is not a completed legal-policy or video submission.
4. Test a clean desktop install, new/existing-chat onboarding and primary workflows.
5. In the [submission portal](https://platform.openai.com/plugins), use the
   approved local MCP **Skills only / Upload ZIP** route (retain the actual MCP
   configuration), resolve findings and obtain the publisher's policy attestations.
6. Submit for review; only after approval choose **Publish plugin**. Record the
   actual listing URL and independently verify the directory's Install flow.

See the [official packaging guide](https://developers.openai.com/plugins/build/plugins),
[desktop-only eligibility](https://developers.openai.com/plugins/deploy/desktop-only)
and [submission requirements](https://developers.openai.com/plugins/deploy/submission).
No allowlist, legal acceptance, review approval or web support is implied by a
successful ZIP build or repository marketplace installation.

## Registry metadata and installation

Find `io.github.seadonggyun4/truthound` in the [official MCP Registry](https://registry.modelcontextprotocol.io/); [version 3.2.0 is published](https://registry.modelcontextprotocol.io/v0.1/servers/io.github.seadonggyun4%2Ftruthound/versions/3.2.0). If a connected client's marketplace exposes the listing, choose **Install** and configure an explicitly allowed directory. Registry publication does not guarantee visibility or approval in every client marketplace.

The repository's `server.json` describes `io.github.seadonggyun4/truthound`,
the existing PyPI package `truthound` at version `3.2.0`, and local stdio.
It requests the MCP extra with UV's `--with` and starts the existing CLI alias:

```bash
uvx --with "truthound[mcp]==3.2.0" truthound@3.2.0 mcp serve --allow-dir /absolute/path/to/data
```

The Registry's GitHub namespace follows the actual repository owner; the package
README's `mcp-name` must match exactly. The versioned official JSON Schema is
checked locally by `python verification/mcp_registry.py` in the dev environment.
A required directory input has no default: discovery never grants filesystem access.
The isolated UV environment avoids unrelated installed CLI plugins; manual host
configuration can continue to prefer the dedicated `truthound-mcp` executable.

PyPI hosts Python artifacts; pip/pipx/uvx install them. The official MCP Registry
hosts discovery metadata, not the package or a remote validation server. A valid
manifest is not proof of a live listing or downstream marketplace approval.
Publishing requires Registry authentication distinct from a GitHub CLI login.
The user approves any new OAuth grant; release tooling does not configure tokens,
trusted publishers or new accounts automatically. See the official
[PyPI ownership rule](https://modelcontextprotocol.io/registry/package-types)
and [authentication guide](https://modelcontextprotocol.io/registry/authentication).

## Extending the adapter

The MCP CI matrix keeps its 15-minute job limit and partitions the entire required
test collection into four deterministic, disjoint Linux Python 3.11 shards.
Other selected OS/Python pairs run platform boundary tests, with two disjoint
partitions on Windows and one elsewhere. Their evidence is aggregated before
installation and cannot substitute for the full semantic collection.
Packaging starts only after every shard succeeds. The aggregate gate compares
the full collection and source hashes, checks all setup/call/teardown phases and
JUnit results, and rejects missing, overlapping, skipped or failed executions.
Shard completion alone does not establish complete validation.

Keep policy in `truthound.mcp.policy`, summaries in `projection`, Core invocation
in `worker`, bounded request execution in `service`, and SDK registration in
`server`. Introduce new tools with explicit input/result contracts and fault
tests; add providers separately rather than accepting arbitrary URLs, Python or
SQL through a file tool. See the [design rationale](../adr/005-local-mcp.md).


### Approved native streaming stores

In the unreleased Core profile, `streaming_store_services_open` uses an approved installed capability returning `StreamingStoreRuntime(store, runs={alias: StreamingRun(run_id, data_asset, metadata)})`. The store must be a native `StreamingValidationStore`; this group is separate from general `BaseStore` bindings. Add its alias-to-capability mapping under `streaming_store_services` in operator bindings. Backend URLs, credentials, codecs and physical run identities stay in the factory.

`streaming_store_services_execute` explicitly provides `inspect`, `session_create/open/resume/status`, `writer_open/write/write_run/flush/close/abort/status`, `reader_open/read/reset/close/status`, `chunks_list`, `chunk_get`, `chunks_delete`, `finalize`, `stats` and filesystem-only `storage_size` commands (use their underscore spellings, such as `session_create` and `writer_write_run`). Every operation requires approval and returns a bounded private artifact. `run_id` selects a fixed run alias; session/writer/reader/chunk IDs are owner-local references. `writer_write` accepts at most 1,000 typed persistence validator records; `writer_write_run` consumes a pinned canonical validation artifact and rejects runs with execution issues that a verdict-only stream cannot preserve. Writer/reader creation accepts `asynchronous=true` for native async wrappers. Reads and chunk listings are paged with a bounded `limit`; listings also accept `offset`.

Explicit `writer_close` and `finalize` flush pending records; `writer_abort` stops without flushing. Release, expiry or input revocation first aborts open writers, so cleanup does not write new records. It cannot undo an already-started I/O or recover a buffer never written to storage. A fresh approved owner can `session_open` the same fixed run and explicitly `session_resume` an incomplete stream. Native inconsistent recovery state fails closed and is retained; no automatic adoption or replay is offered. Finalization returns the native persistence result projection and can materialize the full result in memory. Installed local factory tests exercise filesystem and actual SQLite round trips; real S3/other database services remain unverified.

Filesystem and file-SQLite `recovery_status(run_id=alias)` returns an owner-local `recovery_id` and confirmed/unconfirmed/missing chunk counts instead of sensitive paths or file hashes. Separately approved `recovery_reconcile(recovery_id=..., resolution="adopt"|"discard")` checks the retained fingerprint under the native lock and applies only that explicit selection. Success invalidates earlier session/chunk references; reopen with `session_open`. A changed or corrupt confirmed prefix is never reported as recovered.

File-SQLite recovery holds the real run lease and `BEGIN IMMEDIATE` while committing row deletion and session updates together. Changed fingerprints or inconsistent confirmed chunks fail closed; a failed update or process loss rolls back the transaction. Recovery inspection applies the session `max_memory_mb` bound. `session_release`, `chunk_release` and `recovery_release` free owner-local references without deleting persisted data.

Native S3 streaming uses conditional manifest ETags, create-only chunk requests, receipt checks before writes and owned multipart aborts. Paths, ETags and upload IDs stay private. Local protocol doubles cover uncertain completion receipts and cleanup failures; live S3 and actual SDK deployment verification remain separate.

`writer_status` observes buffer/closed/metric state without flushing. `reader_status` metrics describe I/O records/bytes and can include prefetch, so they may exceed the returned page count. A serialized or encoded chunk exceeding the session read-memory bound is rejected before backend writes, retaining the buffer.

The same opaque streaming-store `recovery_status`/`recovery_reconcile` contract also delegates to approved S3 factories. Model requests never receive object keys, ETags, SHA fingerprints or multipart IDs. Failed S3 discard retains a blocking manifest journal and needs a fresh recovery reference and explicit discard. A whole-run deletion marked `pending_resolution=delete` needs a new approved `chunks_delete` call. Actual filesystem/SQLite process-loss tests and worker replacement over a synthetic S3 object protocol remain distinct from live S3 verification.

General database-store bindings expose `pool_status`, `pool_recycle` and root `shutdown` through `store_services_execute`. Pool status projects native state and numeric metrics without connection URLs, SQL or provider messages. Recycling uses the selected native pool. Approved `shutdown` stops owned background work and closes the factory before acknowledging; failure or expiry cannot become teardown success. After shutdown, the same approval key can replay its result, other commands fail with `invalid_handle`, and the service resource can be released. Actual SQLite pooled/legacy factories persist data across recycle, shutdown and a fresh approved owner. SQL read failures remain failed/unknown operations rather than successful empty artifacts.


`tenant_services_execute(command="isolation_apply")` requires a fixed `tenant_id`
alias, pinned `asset_id`, optional selected `columns`, and bounded `offset`/`limit`.
It delegates to `TenantManager.apply_isolation` and returns a private row page;
`returned_rows` describes that page and `truncated` indicates another row. Only
shared/row-level strategies apply to arbitrary captured assets; schema/database
provenance cannot be inferred from those assets. Tenant activity is enforced.
These operations require sensitive-result consent and exact approval; tenant
administration does not enlarge the asset's data grant. Input release revokes
the service and derived artifacts.

Factories may supply `TenantRuntime.policies` mapping aliases to native
`RowLevelPolicy` callbacks. `policy_catalog`, `policy_add`, `policy_remove` and
`policy_update(policy_patch={enabled, priority})` select the tenant's native
row-level strategy; policy edits never accept model code. A nonempty update may
change either field, with priority bounded to ±1,000,000. Policy names and physical
tenant IDs stay out of metadata views. Policies are owner-local configuration,
not SQLite tenant metadata persisted by these commands. Default Core policies
deny nonboolean verdicts and policy errors, and run after the tenant predicate
but before projection/page limits. The module guide documents the explicit
native `fail_closed=False` behavior.


For operator-owned custom rules, an approved profiler factory may expose native
`SchemaCompatibilityAnalyzer` instances through `ProfilerRuntime.schema_analyzers`.
`profiler_services_execute(command="schema_compare", analyzer_id=..., baseline_id=...,
current_id=...)` pins both asset inputs and delegates to the selected analyzer.
An analyzer-only factory is supported. Callbacks, strict mode and declared
nullability configuration remain factory-owned; requests cannot import rules.
The result preserves the four native compatibility fields. Failed rules produce
no compatible result, and input release revokes the resident owner and derived
artifacts. Execution guards run before normalization and around each native rule.

### Native ML monitor lifecycle

`ml_services_open` selects an approved `MLRuntime` factory through private
`ml_services` bindings. Fixed model aliases select native configuration and
asset-column mappings; fixed rule/handler aliases select native code and delivery
configuration. `ml_services_execute` provides registration, pinned-asset
prediction/reference/feedback, periodic start/stop, manual collect/evaluate,
metrics/dashboard, rule/handler add/remove, explicit uncertain-buffer discard
and acknowledged shutdown. The [model monitoring guide](ml-monitoring.md)
documents every command, collector window and failure contract.

Every operation requires approval and sensitive disclosure consent. Starting
includes future periodic collection, alert evaluation and retention deletion
until stopped or revoked. Inputs and derived artifacts share the owner's grant
lifetime. Outputs use public model/rule/feature aliases and finite metric fields;
native IDs, arbitrary alert messages and handler endpoints are withheld. A failed
`native_monitor_incomplete` artifact does not prove external effects absent and
is never automatically resent. Default buffers/history remain in memory; process
loss recovery is not claimed. Local installed tests use native in-memory storage
and local handlers; actual external services remain unverified.


### Native storage codec and backpressure utilities

`store_utilities_open` resolves private `store_utilities` bindings to an approved
`StoreUtilityRuntime` factory. Native compression/encryption pipelines and
backpressure strategies remain factory-configured; keys, providers, stage code
and physical slots are not caller inputs. `store_utilities_execute` explicitly
captures pinned asset/artifact bytes, retains/processes/reverses/exports/imports
payloads, saves/loads/deletes fixed private slots, controls native pressure
metrics/admission/release/reset and acknowledges shutdown. The [store utility
guide](stores/index.md#native-codec-and-flow-control-utilities) documents commands,
budgets, binary format, persistence and failure semantics.

All operations require approval and sensitive disclosure consent. Native stage
guards prevent later execution after revocation, and input release invalidates
the owner and derived artifacts. Incomplete pressure acquisition remains a failed
artifact and cannot be retried until explicit native reset; denial is a valid
boolean decision. Encryption restore rejects selected-plan/header mismatches
before stages execute. Payloads and metric projections exclude key material and
provider diagnostics. Local native tests exercise actual codecs and cryptography;
no live external key store or provider verification is claimed.


Ray factory configuration now applies `concurrency` through a native task pool,
`use_actors`/`actor_pool_size` through an actual callable-class actor pool, and
`target_max_block_size` to the selected Dataset's context. Actor mode rejects a
simultaneous task-concurrency setting. Batches, prefetch and pool/block sizes
validate their numeric bounds and actual boolean mode. This does not reconfigure
an already initialized shared Ray cluster's CPU/GPU capacity. Querying native
`num_partitions` on a lazy Ray child materializes that selected dataset to observe
its real blocks; it does not return a planned estimate. Partition observation can
therefore execute the factory-selected source plan. Local SDK tests exercise
actual actors and task overlap; remote cluster behavior remains unverified.


Storage utility key bindings now support native create/get-or-create, redacted
status/history, versioned rotation/deletion, selected-cipher application and
manager-wide expired-key cleanup. Key material and physical key IDs never enter
replies. Applied keys are revalidated before codec use; unknown effects latch the
manager aliases without automatic replay. See the [key persistence contract](stores/index.md#key-persistence-and-approved-key-use).

### Distributed plans and validator execution

`ExecutionRuntime` factories may bind `plans` (aliases for native
`AggregationPlan` objects), `validators` (aliases for Core `Validator` objects)
and a fixed `adapter_config` (`AdapterConfig`). `inspect` lists these aliases.
`aggregate_plan` selects `aggregation_plan_id`; `aggregate_stream` additionally
accepts `batch_size` (1–100,000) and the bounded `limit`. `validate` selects
`validator_ids` (1–32 aliases) and optional `columns` on a distributed engine.
Caller-authored plans, SQL filters, validator constructors and callbacks are not
wire inputs. Existing approval, disclosure consent and owner lifetime apply.

Plans execute on local Polars/Pandas/Arrow, Dask, Ray or Spark data. Core's
registered aggregators merge native worker states; averages and standard
deviations merge accumulator state rather than partition summaries. Filters
are fixed SQL expressions evaluated by Polars inside each worker batch. Global
results map aliases to values. Grouped results retain every typed key:

```json
{"groups": [{"keys": {"region": null}, "aggregations": {"amount_mean": 7.0}}]}
```

`aggregate_stream` returns independent cumulative `snapshots` and `complete`.
`complete: false` means the response limit truncated the stream; its last
snapshot is not the final aggregate. The driver checks owner authority before
requesting another batch and after observing it. Previously submitted native
work may already be running. State memory still grows with group cardinality;
`batch_size` is not a total memory sandbox. Distinct counts store exact values,
including colliding hashes, up to `max_sample`; exceeding that bound raises an
error instead of returning a truncated count. Null values are excluded from
column distinct/count; `count("*")` includes every row. Empty mean/std results
follow the registered Core aggregators (0.0). Duplicate result aliases fail.

`DistributedValidatorAdapter` uses exact built-in `NullValidator` and
`NotNullValidator` native counts and their Core issue builders. Detailed failure
samples/counts/complete rows use the bounded Arrow-to-Polars bridge when needed.
A custom validator can explicitly implement
`validate_distributed(engine, columns=None) -> list[ValidationIssue]`. Names and
categories never authorize substituting validation logic. `AUTO` runs other
validators through full conversion when row count is at most `native_threshold`,
and sampling above it. Forced `NATIVE` rejects a validator without a native port.
`FULL_CONVERSION` and `SAMPLE_AND_LOCAL` run the actual validator. Requested
columns apply to both. Parallel execution honors `max_parallel`, preserves
validator order and drains submitted work before propagating failure; use
`parallel_validators=False` for validators requiring serial access.

Sample issues retain observed counts and structured percentages. When sampling
omits rows, each issue adds `sampling: {total_rows, sample_rows, seed}`; counts are
not extrapolated. A sampled child is released even when validation fails, while
a sample that reuses the root does not take ownership of it.

Dask sampling returns exactly `min(n, row_count)` rows, using distributed random
ranks and a seed tied to partition number. Dask applies the configured scheduler
to compute/persist, honors disabled result caching, and performs Arrow conversion
once. An engine-created distributed client is closed on context exit; a borrowed
client remains caller-owned. Child engines borrow the parent's client and must
finish before that parent closes. Selected `persist_intermediate` persists on
construction. Ray's nullable grouped counts use a copied Dataset with native
hash shuffle, preserving typed null equality and the caller's shuffle context.
Actual local Ray 2.48 and current-SDK checks cover this compatibility path;
external clusters and services remain unverified.

Storage utilities also select native `adaptive`, `chunked_compression` and
`chunked_encryption` codecs, with `codec_analyze`/`codec_recommend` for fixed
adaptive configurations and `codec_range` for selected plaintext byte ranges.
Header/index/size/checksum failures propagate; v2 encrypted chunk indices are
authenticated before plaintext is returned. See [native codec configuration and
format contracts](stores/index.md#authenticated-streaming-and-chunked-encryption).

Incremental `stream_open/write/read/flush/finalize/abort/status/release` commands
use approved `StoreCompressionStreamBinding`/`StoreEncryptionStreamBinding`
factories. Failures and unconfirmed native context cleanup fence further work;
shutdown aborts unfinished streams. See [incremental codec sessions](stores/index.md#incremental-codec-sessions).

Ordinary filesystem store factories honor JSON/safe-YAML and the selected native
codec, capture config objects and publish records/indexes atomically as separate
files. Corruption is an error; `rebuild_index` validates records before explicit
reconciliation. Memory factories honor capacity/copy/default metadata and remain
owner-local across shutdown. See the [local store contracts](stores/filesystem.md)
for single-writer boundaries, identity rules, startup checks and legacy names.


Pandas engines borrow the supplied mutable frame. After changing it, call
`clear_cache()` to invalidate the materialized Polars conversion as well as
computed results. `cache_results=False` converts the current frame on every call.

Ray `get_stats()` computes row/valid counts, mean, sample standard deviation
(`ddof=1`) and extrema with one native multi-aggregation, including all-null and
typed empty columns. NaN behavior follows the native SDK. A missing reduction
result for a non-empty dataset fails instead of being cached as zero statistics.

`clear_cache` also invalidates native Ray/Spark row-count memoization, and
`cache_results=False` bypasses Core result caching. Ray global COUNT/SUM/MEAN/MIN/MAX/STD/VAR/NULL_COUNT reductions
without explicit per-operation parameters reuse exact cached column statistics
and memoize results, including a shared MIN/MAX pair. `clear_cache()`
invalidates both. Explicit parameters such as `ddof=0`, grouped/filtered plans, FIRST/LAST,
distinct counts and the bounded median conversion retain their own execution; `COUNT(*)` remains a
row count and cached null results stay null. Native dataset persistence is
separate from this cache. `AggregationPlan.to_spec()` captures independent
aggregation parameters, group columns and the fixed SQL filter; grouped/filtered
specs use the same Core state aggregation as plans. Dask plans/specs retain the
selected engine scheduler; standalone `DistributedAggregator` also accepts a
`dask_scheduler` keyword for an explicit scheduler/client callable. Local Polars
and Arrow table plans do not require pandas.


Distributed engines apply a positive `num_partitions` at initial construction.
An explicit child `repartition`/`coalesce` keeps its requested partition count;
sampling, filtering and projection do not reapply the root default. Partition
counts reject booleans, zero and negative values. Spark captures the supplied
configuration and applies both `True` and `False` Arrow/adaptive flags to its
selected session; extra Spark settings override these fields.

An approved `AggregationPlan(scope=AggregationScope.PARTITION)` and native
`AggregationSpec` with that scope finalize each native partition separately:
`{"partitions": [{"partition_id": 0, "result": {"x_sum": 3}}]}`. IDs are observed
iteration ordinals, not durable identities across executions or repartitioning.
Filters and optional groups apply within each partition; an empty partition has
the registered aggregators' empty result. Ray partitions are native blocks;
[Ray's batch API](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html)
uses whole blocks when `batch_size=None`. Internal batches still merge states
inside each block. Partition-plan streams return independent cumulative lists
of completed partition results, and the usual `complete` truncation contract
applies. Factories select these plans for MCP through the existing plan aliases.

Approved ordinary S3 factories now apply native format/compression, metadata and
object tags, conditional record/index receipts, error-preserving queries and
explicit `rebuild_index`. Shutdown closes the owned client. See [S3 recovery and
provider contracts](stores/cloud-storage.md#conditional-publication-and-recovery);
local synthetic/botocore checks do not establish live cloud verification.


`ExecutionMode.EAGER` materializes the selected native graph during engine
construction and applies to derived engines too. Dask persists it and, with a
distributed client, waits for its selected futures and propagates task errors
without gathering full partitions into the driver; `timeout_seconds` bounds this
initial wait. This wait does not prove remote task interruption on timeout.
`persist_intermediate` alone retains Dask's usual asynchronous submission behavior.
Ray retains a native `MaterializedDataset`; context exit drops the engine's own
materialized reference, while borrowed materialized datasets remain caller-owned.
Spark persists and counts immediately, then unpersists only cache created by this
engine on exit, with blocking cleanup. An already persisted input stays owned by
the caller. Explicit `unpersist` releases the selected cache. Default `LAZY` keeps
deferred graphs. MCP `inspect.config` includes selected execution mode, initial
partition default, Arrow batch size and metrics flag without cluster addresses.


Factory-owned distributed configuration now applies `partition_strategy` on
initial partitioning and explicit `repartition`. `ROW_HASH` uses the selected
`partition_columns`, or all columns when none are supplied. `COLUMN` requires
explicit keys and hashes equal keys together. `ROW_RANGE` uses ordered native
ranges in Spark/Ray; Dask supports one range column, plans bounded per-partition
key samples using the selected scheduler, and distributes/sorts rows in native
Dask tasks without dropping nulls. Its driver samples grow with input partition
count times requested partition count. Spark may produce fewer range partitions
for small keyspaces; `num_partitions` reports the observed count.
`ROUND_ROBIN` starts at zero within each input partition/block and distributes
rows cyclically; it is not a durable global row ordering. `CUSTOM` requires an
operator-owned `partitioner(data, count)` returning the same native data type and
schema and the requested partition count. Invalid outputs fail. MCP never accepts
caller partitioner code. Ordinary projection/filter children keep their native
partitioning. `coalesce` uses each backend's native reduction independently of
the shuffle strategy. Range planning and native schema/partition inspection may
read source keys or execute native graphs even in `LAZY` mode. Native Ray
partition aggregation uses per-block Arrow references so empty blocks keep their
empty results; raw block values stay in Ray workers.


Streaming S3 `recovery_status` also reports durable multipart intent/receipt
counts. `multipart_reconcile` consumes a retained `recovery_id` and optionally
`receipt_id`, an alias in factory-owned `StreamingStoreRuntime.multipart_receipts`
(private exact-key→upload-ID mappings). Unknown creation receipts require that
operator-verified alias; known receipts can be cleaned without it. Inspect lists
aliases only. The approved command aborts scoped uploads, preserves unconfirmed
cleanup, returns a new recovery reference and counts, and withholds provider keys,
upload IDs and fingerprints. It does not adopt/discard completed chunks. See
[durable multipart ownership](stores/index.md#durable-s3-multipart-ownership).

Approved ordinary GCS factories now use native generation receipts and raw stored-byte
reads through the shared Core object-store state, including explicit index rebuild
and actual client shutdown. See [GCS generation and recovery contracts](stores/cloud-storage.md#generation-receipts-and-recovery).

Approved ordinary Azure factories use the same Core inventory/recovery path with
native conditional ETags and stored-byte decoding. Blob-header `metadata` and
persisted `record_metadata` have separate roles. `get_access_tier` reads the
selected record's observed tier/status; `set_access_tier` requires `access_tier`
(`Hot`, `Cool`, `Cold`, `Archive`) and returns `acknowledged` plus the observed
state. This is not a rehydration-completion receipt. These commands require an
unwrapped selected Azure store; client input cannot supply credentials, endpoints
or executable callbacks. Shutdown closes the owned client and default credential.
See [Azure recovery and ownership](stores/cloud-storage.md#etag-receipts-recovery-and-client-ownership).

### Native engine checkpoints and Arrow cursors

For Dask and Ray, `checkpoint_dir` selects an operator-owned local directory.
`checkpoint()` writes native Parquet into a unique staging directory, flushes the
files and SHA-256 inventory, and publishes a completed snapshot by directory
rename. The returned child exposes its local `checkpoint_path` in Python.
Ray creates its local destination directory and explicitly selects Arrow's local
filesystem for native writes and reads, including Windows drive paths.
The `execution-ray`, `all` and `dev` extras exclude Ray 2.59.0 because that SDK's
local-path resolver probes an unavailable optional Azure filesystem on Windows.
This dependency restriction also applies outside CI; the native computation and
checkpoint assertions are unchanged.
`DaskExecutionEngine.from_checkpoint(path, config, client=...)` and
`RayExecutionEngine.from_checkpoint(path, config)` validate the complete inventory
before opening the native reader. Pending directories, corrupt files and wrong
backend receipts fail. Failed writes retain their pending directory for inspection;
closing an engine does not remove checkpoint data. Treat completed snapshots as
immutable while readers use them. The manifest detects accidental corruption;
it is not an authentication signature. This local protocol does not establish
remote filesystem or object-store transaction guarantees.

`checkpoint_enabled=True` now requires a nonempty `checkpoint_dir` and executes
checkpointing at root construction, after selected repartitioning. Derived
projection/filter engines do not automatically checkpoint again. Dask/Ray reopen
the committed native data; the source graph need not rerun. Spark delegates to
its native reliable DataFrame checkpoint and selected SparkContext checkpoint
directory. Spark's checkpoint is part of its native context lifetime, not the
portable Dask/Ray manifest format. MCP `checkpoint` retains the native child for
all three engines; physical paths stay in the installed factory. A subsequent
factory can explicitly reopen a selected Dask/Ray snapshot after worker restart.
See [Dask native Parquet publication](https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.to_parquet.html).

`iter_arrow_batches(batch_size=None)` yields typed native batches and supports
explicit iterator `close()`. Dask computes one partition at a time, Ray uses
native batch iteration with configured prefetch, and Spark uses its local row
iterator without partition prefetch. Empty Dask streams preserve their schema;
Ray normalizes null-only batches to the dataset schema. `STREAMING` selects this
path for Dask/Spark Arrow conversion; Ray uses it in every mode. APIs returning a
complete Arrow table or Polars frame still retain that complete result. Native
partition allocation can exceed a batch's row count, and this mode does not start
an unbounded broker or Spark Structured Streaming query. See the
[Ray iterator lifetime](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.iter_batches.html)
and [Spark local iterator](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.toLocalIterator.html).

Approved MCP `arrow_stream_open` accepts `batch_size` and returns an opaque
`stream_id`. `arrow_stream_read` requires that ID and the same `engine_id`, applies
`limit` to rows, and returns `rows` and `complete`. A page ending exactly at the
last row may need one more read to observe exhaustion. `arrow_stream_close`
releases the cursor; releasing its child engine or shutting down also attempts
cleanup. At most 16 cursors can remain open. Reads check the binding before and
after native iteration and enforce the private artifact byte budget. Cursors are
owner-local; a failed or lost consumed read is not replayed as a fresh iterator.

Native computation entry points share a cooperative `timeout_seconds` deadline
across nested calls. Late results raise `DistributedOperationTimeout` and clear
result caches. Arrow cursors check the deadline around every batch and close on
failure or early exit. A blocking SDK call can continue until it returns; this
policy does not kill borrowed workers or promise a hard wall-clock limit.
`retry_count` allows additional attempts only when `RetryableExecutionError`
propagates from code explicitly declaring the computation safe to repeat.
Ordinary exceptions, timeouts and uncertain side effects are not automatically
retried. Native frameworks can wrap worker exceptions and have their own task
retry policies. Checkpoint/write/cluster-lifecycle operations retain their native
failure semantics. `memory_fraction` bounds each accepted Arrow batch against that fraction of
available driver memory observed at iterator start. Oversized batches raise
`MemoryError` before being yielded. This cooperative transfer check occurs after
native allocation, does not reserve memory, and does not cap worker heaps, native
partition buffers, prefetch or complete collected results. Configure those through
the selected backend. The execution extras include `psutil` for this observation.

### Selected resolver contexts and verified SQL sources

`TenantRuntime.contexts` binds aliases to private context dictionaries. The worker
copies them on entry and on each resolution. `resolver_catalog` lists aliases;
`resolve(context_id)` invokes the native resolver and returns an allowed tenant
alias or `resolved: false`. A resolution outside the tenant alias map fails.
`resolve_or_create(context_id, tenant_id, default_name)` resolves once and checks
the selected native ID **before** lookup or creation through Core's
`TenantManager.resolve_or_create(..., expected_tenant_id=...)`. Header resolver
`required=True` rejects missing headers; malformed IDs, including trailing
newlines, are rejected. Headers, tokens and resolver code are not wire inputs.

`TenantRuntime.sources` binds aliases to factory-owned
`truthound.multitenancy.sources.TenantSQLSource`. `source_catalog` lists aliases;
`source_read(tenant_id, source_id, columns?, offset?, limit?)` reads a selected
source and returns a bounded page under sensitive-result consent and exact
approval. The native wrapper borrows a `SQLiteDataSource`, binds one native tenant
ID, and verifies the actual reading connection before and after fetching rows
inside a SQLite savepoint. DATABASE isolation compares the actual database file
with the tenant's absolute `database_name`; SCHEMA isolation requires both the
native schema name and the wrapper's explicit absolute `namespace_database`.
This also supports operator-attached SQLite databases. The factory owns connection
cleanup; source paths and namespaces stay private. Suspended tenants, another
tenant ID, query mode, views, virtual tables, in-memory databases and unverified
remote providers fail closed. Ordinary table reads are bounded by the source's
`materialization_row_limit`; exceeding it fails before returning any rows. Page
limits do not replace that bound. This is a materialized read snapshot, not ongoing
revocation of data already returned. Caller-provided Arrow assets still cannot
establish schema/database provenance. Installed-worker tests cover SQLite owner
restart; they do not establish remote database provider support.

Profiler services now expose bound `quality_scorers` and `quality_rules` through
`quality_score`, `quality_compare` and `quality_cache_clear`. Inputs are pinned
assets with an optional aligned Boolean `label_column`; sampling/configuration
and custom rule code remain factory-owned. See [quality scoring](profiler/quality-scoring.md).

Bound profiler services also expose native sampling snapshots, local type-inference
feedback/model persistence, and complete-profile HTML export to budgeted artifacts
or fixed slots. See [sampling](profiler/sampling.md), [ML inference](profiler/ml-inference.md)
and [visualization](profiler/visualization.md) for configuration and limits.

Plugin owners also accept `plugin_services_execute(command="shutdown")` without an alias. The approved command waits for plugin and reload cleanup plus factory exit before acknowledging success. A failed cleanup remains failed/uncertain and is not replayed as a fresh teardown.


Concurrent store factories also honor captured JSON/safe-YAML, registered compression, metadata defaults and the selected expectation-store lock strategy. `query.order_by="name"` is available for expectation stores; tag filters and pagination apply natively. Invalid records fail queries and index rebuild rather than disappearing from successful results. Index operations are validated before WAL publication; this does not promise atomicity across record files and index files.

### Approved benchmark workloads

Private `benchmark_services` bindings resolve to `BenchmarkRuntime` in
`truthound.mcp.benchmark_services`. Its `workloads` map contains
`BenchmarkWorkload(runner, kind, target, parameters)` values; `kind` selects
`single`, `suite`, `all`, `category`, or `parity`. The operator pins native runners,
suites, framework choices, parameters and artifact destinations in the factory.
The wire accepts aliases, never arbitrary imports, paths or runner keyword arguments.
Optional `comparators` and `thresholds` maps hold native comparators and thresholds.

`benchmark_services_open` and `benchmark_services_execute` use approval and
idempotency receipts. Execute repeats `binding_id`, `service_id` and
`idempotency_key`. Commands are `inspect`, `run` (`workload_id`), `result`,
`release`, `report` (`result_id`, optional `format`: json/markdown/html),
`compare` (`result_id`, `baseline_id`, `comparator_id`), `threshold`
(`result_id`, `threshold_id`) and `shutdown`. Comparisons require matching native
result kinds. Parity reports currently support JSON only.

Results retain native timings and verdicts. Failed measurements and empty suites
produce `native_benchmark_incomplete` with a retained result artifact, rather than
a successful execution receipt. Threshold checks reject failed or empty
measurements. Parameters, tags, tracebacks, detailed parity observations and
operator artifact paths are excluded from returned reports; ordinary benchmark
environment metadata remains sensitive and requires disclosure approval. Owners
retain at most 16 results within the serialized native-result budget; `release` removes a
result. Cancellation joins the owned native thread before normal cleanup. Forced
process loss remains an unknown outcome, with no automatic replay.

Native `BenchmarkConfig` rejects invalid iteration counts and nonfinite budgets.
Time and process RSS limits are checked around setup and iterations; cooldown
respects the remaining time budget. These checks do not preempt a blocking
iteration or detect every transient allocation peak. Teardown still runs on
failure. This source implementation has local synthetic workload evidence, not
a fresh release parity benchmark. Complete parity-provider verification remains outstanding.

### Approved API documentation generation

Private `documentation_services` bindings return
`truthound.mcp.documentation_services.DocumentationRuntime(jobs={alias: job})`.
Each `DocumentationJob(generator, source, kind="source")` pins a native
`DocumentationGenerator`, its renderer/configuration and source in the installed
factory. `kind="module"` uses native module extraction/rendering;
`kind="info"` uses an operator-provided `ModuleInfo` or `PackageInfo`. Requests
cannot choose source paths, module imports, renderers or output paths.

`documentation_services_open` and `documentation_services_execute` require
approval and idempotency receipts. Execute repeats `binding_id`, `service_id`,
`idempotency_key` and selects `inspect`, `generate` (`job_id`), `result`
(`result_id`), `read` (`result_id`, `file_id`), `release` (`result_id`) or
`shutdown`. A generation retains a manifest of file IDs, relative names, sizes
and SHA-256 digests. `read` returns the exact generated bytes as an artifact.
The native JSON, MkDocs, Sphinx and HTML formats are selected by the factory.
Module-only output is retained as `docs/module.txt` without changing its content.

Generation uses an owned temporary directory, including MkDocs' sibling
configuration file. Partial native failure retains already generated files with
`native_documentation_incomplete`; original error messages are redacted. Empty
output is incomplete. Symlinks and nonregular files are rejected. Retention is
bounded by the service byte budget, 256 files per generation and 16 generations;
`release` or owner shutdown removes retained results. Cancellation joins owned
native work before cleanup. This is an in-memory artifact lifecycle, without
cross-owner recovery or automatic site publication.

Documentation content may include source text, docstrings and source links.
Operators must select sources intended for disclosure; approval covers that
content, not just the manifest. The byte budget limits retained artifacts after
native generation; it is not a hard disk, CPU or memory sandbox for a renderer.
The factory remains trusted installed code. Generated sites have not been built,
installed or deployed merely because generation succeeded.

Benchmark factories can also supply `BenchmarkRuntime.histories`, mapping aliases
to native `RegressionDetector` instances with operator-selected private history
roots. `history_load` and `history_inspect` require `history_id`;
`history_baseline` restores the baseline as a retained result (or returns
`available: false`). `history_save`, `history_check`, `history_report` and
`history_thresholds` require `history_id` and a suite `result_id`.
`history_threshold` adds an already bound `threshold_id`. `threshold`,
`history_check` and `history_report` accept an optional retained `baseline_id`.
History comparisons with no baseline, missing benchmarks or failed measurements
produce an incomplete receipt. History reports return native text artifacts.
Saving is an approved write to the factory's private root; a new owner can restore
that baseline, without exposing the root path. The two baseline/history file
writes are individually atomic, not a multi-file transaction; uncertain writes
must not be blindly replayed. Corrupt or legacy summary-only evidence is rejected
instead of being treated as absent or successfully restored.

Pinned benchmark workloads also accept `kind="generator"` or `"generator_lazy"`:
`runner` is a native `DataGenerator`, `target` is a captured `GeneratorConfig`, and
`parameters` must be empty. `run` retains the resulting frame and returns schema
and row-count metadata. `data_page` accepts `result_id`, nonnegative `offset` and
`limit` from 0 to 1000; `report(format="ipc")` exports exact native Arrow IPC bytes.
JSON reports describe the frame, while Markdown/HTML apply to benchmark results.
Generated frames share the owner's 16-result and memory-estimate budget, with
`release`/shutdown cleanup. This byte accounting does not cap transient native
generation memory. Seeded output and IPC are verified with local synthetic data;
no real PII or external baseline cohort is used by that verification.

`kind="scaffold"` pins a native `BaseScaffold` and `ScaffoldConfig`; generated
files use an owned output root. All paths are preflighted before writing:
absolute paths, traversal, duplicates and symlinks are rejected. Each file write
is atomic; the entire scaffold is not a multi-file transaction. Manifests also
include the executable flag. Templates require custom logic; unimplemented
datasource/action behavior raises an error.

`kind="validate"` pins a `DocumentationGenerator` and source for native
`validate_docstrings`. `strict=True` is supported only for this kind. Native
issues are retained in `docs/docstrings.json`; validation failure retains the
artifact with `native_documentation_incomplete`. These files can contain selected
source/docstring content and require the same disclosure approval as generation.

Benchmark `compare` and history comparisons return `native_benchmark_incomplete` when suite names are missing or duplicated; they do not silently compare only the intersection.

Scaffold jobs preserve native generation failures for invalid settings. The retained reporter files execute against canonical Core results: passed-check projection is shared with Core, and a zero issue limit suppresses detail rows without changing summaries.


### Approved native tracker sessions and context history

Private `lineage_services` bindings return
`LineageRuntime(tracker, jobs={alias: LineageJob(callback, name, operation_type)},
files={alias: absolute_operator_path})` from `truthound.mcp.lineage_services`.
The factory owns a native `LineageTracker`, its configuration and file aliases.
Callbacks are pinned synchronous `(tracker, context) -> None` functions; requests
cannot supply code, imports or file paths. The tracker must have no active context
when the factory yields it.

`lineage_services_open` and `lineage_services_execute` require operator approval,
a control directory and sensitive-result consent. Execute repeats `binding_id`,
`service_id`, `idempotency_key` and a closed `command`:

- `inspect` returns state counts and job/file aliases without private paths.
- `track` takes the same typed `event` as `lineage_track`, without `graph_id`.
- `run(job_id)` executes the callback inside native `tracker.track(...)` on one
  owned thread. Nested/current-tracker semantics and native auto tracking apply.
- `history(offset, limit)` pages native contexts, including completed flags and
  redacted failures; limit zero returns no rows. `snapshot` returns the graph.
- `lineage(node_id, direction)` and `path(source_id, target_id)` use native queries.
- `save(file_id)`, `load(file_id)` use pinned private graph files; `clear` clears
  graph and context history. Graph serialization does not persist history.
- `export` returns native graph JSON; `render(rendering={...})` reuses the closed
  renderer/output-format settings above. A native rendering failure is a failed
  JSON artifact, not a mislabeled image, and leaves the tracker available.
- `shutdown` acknowledges only after the owner and factory have finished.

Each response is an immutable session artifact and approval-key replay does not
repeat the callback or filesystem effects. Completed history entries are snapshots
of normal structured metadata. Opaque native metadata can be unsupported by the
transport. Callback failures retain native partial graph changes and failed
contexts; they do not imply rollback. Native graph persistence uses version fences
and verifies published bytes. Explicit primary saves cannot overwrite a newer
tracker. Reloading the authoritative primary file confirms its version and can
resolve an unconfirmed publication; loading a separate export does not resolve
primary uncertainty. A fresh owner reloads configured graph state with empty
native context history. Cancellation joins the owned native thread; this is not a
hard CPU sandbox or durable callback/history journal. Forced process loss remains
an unknown outcome with no automatic job replay.

Expression-based validation retains each reported column's own failure samples,
value counts and full rows. Passing earlier columns and checks suppressed by
`mostly` do not shift evidence onto a later failure; standalone and batched Core
execution use the same association.

### Core 3.3.0 installation candidate

The published `truthound` 0.1.4 plugin remains pinned to Core 3.2.0. The separate `truthound-core` 0.2.0 candidate pins Core 3.3.0; normal public-index installation requires publication of that version. Transitive dependencies resolve within package bounds at installation; no frozen public-index lock is shipped for the unpublished candidate. Candidate CI installs a freshly built wheel, not the source checkout.

```bash
codex plugin add truthound-core@truthound --json
truthound-mcp bundle --profile core --output truthound-3.3.0-core.mcpb
truthound-mcp bundle --profile core --wheel dist/truthound-3.3.0-py3-none-any.whl --output truthound-3.3.0-core-local.mcpb
```

Installation grants no data access. Inside the installed UV project, use `python -m truthound.mcp.consent configure --allow-dir <approved-directory> --confirm`. Optional `--workspace-dir`, `--control-dir` and `--allow-sensitive-results` are separate operator grants. Configuration replaces all prior grants in `~/.truthound-core/permissions.json`; `python -m truthound.mcp.consent revoke` removes them. Changes close the session and invalidate retained handles; reconnect afterward. Individual execution plans still require trusted operator approval outside MCP.

The Core MCPB exposes the current catalog and separate data, workspace, control and sensitive-result settings. Blank optional directories grant no access; sensitive disclosure defaults to false. Restart after editing desktop settings. The default bundle still exposes three summary tools. Core coverage remains partial; neither candidate has official directory approval.

Windows Ray 2.58 owned local startup queries its existing SDK node client for the exact ALIVE node and valid manager port, avoiding the temporary `GlobalStateAccessor.GetNode` path that produced a native access violation. The scoped hooks are restored on success or failure; other threads and external clusters retain SDK behavior, and the startup deadline is not extended.

Owned local Windows Spark 4.2 startup uses the installed SDK launcher JAR to generate NUL-delimited JVM arguments and launches that real JVM without shell evaluation. This removes the GUID PowerShell process and the wildcard launcher class search; Spark still creates and authenticates its gateway, reads the original configuration, and performs native computation. The original 30-second reply deadline remains. External/borrowed sessions, other SDK versions and operator startup scripts/configuration keep their SDK path; scoped hooks restore on failure and do not affect other threads. Partial startup also unregisters only its captured SDK exit callback before the outer owner reaps descendants.

Native worker processes keep their JSON protocol on a separate, non-inheritable stream. OS stdout/stderr descriptors and Windows CRT/Win32 standard handles point to the private sink before an operator factory imports an SDK. Native extension, subprocess and SDK shutdown output cannot write into the protocol; deadlines and process ownership remain unchanged.

The protocol descriptor inheritance flag is explicitly cleared, including Windows standard-stream duplicates.
