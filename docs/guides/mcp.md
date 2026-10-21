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
| `streams_open`, `streams_next`, `streams_inspect` | Native CSV/JSONL/Parquet/Arrow file-stream cursors, complete retained batches and native read state |
| `sources_open_polars`, `sources_inspect`, `sources_sample` | Owned resident native Polars datasource, complete native config/metadata and Core sampling into a retained asset |
| `sources_validate_connection`, `sources_validate_columns`, `sources_check_size_limits`, `sources_validate` | Actual connection/column/size checks and validation through Core's native `source=` execution path |
| `assets_validate` | Core validation with closed constructor settings, retained schema and reference assets; canonical result artifact |
| `suites_create`, `suites_inspect`, `suites_validate` | Native kernel suites with stable check IDs, schema handles, tags, metadata and evidence/severity policies |
| `assets_page`, `assets_sample`, `assets_mask` | Stable lossless pages, seeded Core sampling and masking |
| `assets_profile`, `assets_scan` | Complete Core profile and PII reports retained privately |
| `schemas_learn`, `schemas_create`, `schemas_inspect`, `schemas_export` | Learned or explicit typed schemas, authorized inspection and owned JSON/native YAML export |
| `profilers_catalog`, `profilers_profile` | Advanced native profiler configuration, table/column analyzers and type inferrers |
| `artifacts_export`, `artifacts_read` | Owned IPC/Parquet/CSV/JSON/NDJSON files and bounded byte reads |
| `artifacts_render`, `artifacts_datadocs` | Canonical validation re-entry, JSON/Markdown/console/HTML reporters and English/Korean validation Data Docs |
| `resources_describe`, `resources_release` | Session ownership, expiry, integrity, current grants and descendant cleanup |
| `contexts_open`, `contexts_inspect`, `contexts_initialize`, `plans_status` | Separately granted workspace configuration, approved initialization and intent status |
| `contexts_execute` | Explicit approved native source tracking, baselines, applied suites, result/docs persistence, metrics and workspace listing |
| `drift_compare` | Complete Core dataset-comparison report with all method choices and parameters |
| `drift_catalog`, `drift_detect` | Full constructor settings of 13 concrete native drift detectors on two owned Series columns |
| `lineage_catalog`, `lineage_create`, `lineage_edit`, `lineage_query` | Typed native graph configuration and immutable graph edits/traversals |
| `lineage_render` | Native whole/subgraph D3/Cytoscape JSON, Graphviz DOT or Mermaid text with full closed rendering configuration |
| `lineage_impact`, `lineage_critical_paths`, `lineage_what_if_delete`, `lineage_schema_impact`, `lineage_dependency_chain` | Native impact and dependency analysis retained as artifacts |
| `models_catalog`, `models_create`, `models_inspect` | Closed configurations for the 12 concrete Core ML algorithms and resident lifecycle state |
| `models_fit`, `models_partial_fit`, `models_fit_predict`, `models_predict`, `models_score` | Native training, supported online training, combined training/prediction, prediction and anomaly scoring |
| `models_detect`, `models_learn_rules`, `models_rules`, `models_diagnostics` | Two-input ML drift detection, rule learning and native model diagnostics as owned artifacts |

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
documents the closed fields for each of the 254 concrete registered entries,
including inherited `ValidatorConfig`, cache and algorithm options. Unknown
settings are rejected instead of being silently dropped by the Core constructor.
That inventory is **not** evidence that all 254 algorithms have passed parity.
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
profilers and generated-rule lifecycle are not covered by this binding.

### Drift and resident ML

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

ML result artifacts contain `encoding`, `model_algorithm`, `model_revision` and
the tagged native `result`. This records the model state used without lossy
Python object serialization. Core's JSON model save/load is not uniformly
state-preserving (for example Isolation Forest does not restore trees); durable
model snapshot/import is therefore not exposed as a successful full-state
roundtrip. Post-creation ensemble composition, arbitrary plugin models and
complete method-by-method lifecycle equivalence remain outstanding.

### Native lineage graphs

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

This implementation is **not full Core parity**. Complete algorithm-by-algorithm
equivalence, all datasource/provider lifecycles, custom profiler composition,
complete ML persistence, lineage trackers/history/interactive rendering, checkpoints,
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

The standalone OpenAI plugin is version `0.1.2`, pins the published Core
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

Keep policy in `truthound.mcp.policy`, summaries in `projection`, Core invocation
in `worker`, bounded request execution in `service`, and SDK registration in
`server`. Introduce new tools with explicit input/result contracts and fault
tests; add providers separately rather than accepting arbitrary URLs, Python or
SQL through a file tool. See the [design rationale](../adr/005-local-mcp.md).
