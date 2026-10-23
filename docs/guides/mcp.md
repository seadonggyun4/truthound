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
The factory must return a synchronous Core `BaseDataSource` with `name="orders"`.
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
`sources_validate_columns`, `sources_check_size_limits`, `sources_sample` and
`sources_validate`. Validation passes the actual resident source to Core's
`source=` path; it does not replace the provider with a captured DataFrame.
Inspection deliberately returns `config=null` for bound sources to avoid exposing
connection configuration. Native data freshness/caching and engine behavior remain
the provider's semantics; this is not an immutable snapshot or a reconnect service.
Validation leaves `input_sha256` and `input_identity` unknown for a live binding;
the source handle's resource hash describes its binding label, not database rows.
Async-only providers and arbitrary protocol objects are not accepted.

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
are distinct Core inputs: for example DataProfiler's `options.sample_size`
controls its head sample, independently of `config.sample_size`. The adapter
does not merge these settings or pretend currently unused Core fields change
an algorithm. Unknown fields, arbitrary component imports and custom regex
pattern definitions are rejected. Full custom extension composition, streaming
profilers and generated-rule application lifecycle are not covered by this binding.

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

#### Approved pause, reconciliation and queue retention

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

Retention runs on worker sweeps, including paused or disabled watches, under
unchanged authority. It does not erase native checkpoint history, approvals,
external copies or free definition slots, and is not secure physical erasure.
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
Core currently stores `strict` without changing analysis, and its nullability
rule returns no issues because Polars schemas contain no nullability metadata.
Do not interpret that rule's result as validation of nullable data. Core also
logs and skips rule exceptions; `rules_checked` records evaluated rules, not
a guarantee that every requested rule completed.

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
session-owned services, **not durable restart recovery**: scheduling definitions,
new policies and notification targets are operator-configured, not wire-editable.
Real external notification and store services remain unverified. These tools
are development-source features, not part of published 3.2.0.

Additional approved management commands reuse the native service methods:

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
  Levels, destinations, conditions, triggers and metadata remain unchanged.
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
New policies, target/level editing and durable service recovery remain unavailable.

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
No output path, renderer executable, HTML execution, browser or image conversion
is invoked. Read it with `artifacts_read` and the sensitive-result grant; releasing
the source graph revokes the rendered artifact. Unknown roots/settings fail.
Tracker/history integration, interactive HTML/image rendering and OpenLineage
publication remain separate incomplete surfaces.

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

This implementation is **not full Core parity**. Complete algorithm-by-algorithm
equivalence, all datasource/provider lifecycles, custom profiler composition,
complete ML persistence, automatic lineage tracking/history/interactive rendering, checkpoints,
streams, stores, full plugin lifecycle tooling, administration and durable jobs
remain incomplete. Discovery returns
`full_core_parity=false`; listing gaps is not implementation/provider evidence.
No new remote server, credentials or paid calls are introduced.

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
test collection into four deterministic, disjoint shards for each OS/Python pair.
Packaging starts only after every shard succeeds. The aggregate gate compares
the full collection and source hashes, checks all setup/call/teardown phases and
JUnit results, and rejects missing, overlapping, skipped or failed executions.
Shard completion alone does not establish complete validation.

Keep policy in `truthound.mcp.policy`, summaries in `projection`, Core invocation
in `worker`, bounded request execution in `service`, and SDK registration in
`server`. Introduce new tools with explicit input/result contracts and fault
tests; add providers separately rather than accepting arbitrary URLs, Python or
SQL through a file tool. See the [design rationale](../adr/005-local-mcp.md).
