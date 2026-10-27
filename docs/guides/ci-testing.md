# Repository CI and full nightly verification

Development workflows run on PRs, merge queues and main pushes, not every branch
push. Main validation remains enabled while direct pushes are allowed. Superseded
PR executions may be cancelled; cancellation is never evidence of success.
Contract tests retain four shards and fault/e2e; the three lint targets share one
job without removing checks. Documentation retains full/public strict builds,
Korean checks and public-surface exclusions.

MCP semantic coverage runs every mandatory file listed in
`verification/mcp_ci_gate.py` in eight disjoint Linux Python 3.11 shards. No algorithm case is removed. The source/execution aggregate
still rejects missing, duplicate, failed or skipped tests. Platform boundary
contracts cover subprocess cleanup, permissions, UTF-8, resources, persistence
and approval on other selected runtimes. Boundary results cannot substitute for
the full partition. Both retain their original 15-minute job deadlines.

The canonical wheel, sdist and two MCPBs are built once. Every installation job
verifies SHA256SUMS and installs those exact files; the embedded wheel must match
the canonical wheel byte-for-byte. All seven fresh installation/launcher checks
remain. Development uses Linux 3.11/3.13 and Windows/macOS 3.11.
Weekly and release verification use all nine supported OS/Python combinations. Platform-independent semantic cases are not repeated nine times.
Every additional OS/Python boundary runtime uses four disjoint shards. Installation first aggregates the matching
partitions and rejects missing, duplicated, failed or wrong-scope evidence.
This changes neither test coverage nor the 15-minute deadline and does not
turn a failing test into a passing one.
Platform contract jobs run alongside the full Linux shards, separately from
installation jobs, with the same 15-minute deadline per job. This avoids charging
both boundary tests and seven installations against one deadline. No tests or
runtime combinations are removed or repeated. Installation jobs download the
matching boundary proof, and the final gate requires full tests, build, platform
contracts and installation success; failed or missing contracts cannot be bypassed.
Shared source, test, configuration and dependency changes remain conservative MCP
inputs. This is not a claim of complete Core capability parity.

`release-verify.yml` explicitly coordinates full Nightly, full MCP compatibility,
strict documentation, plugin installation, benchmark contracts, lint ratchets and
the unchanged fixed-runner performance gate on one SHA.
It never publishes. `release-pypi.yml` requires its successful run ID on the exact
checkout SHA, downloads its canonical artifacts and rechecks installed behavior
before publication, without rebuilding or repeating the full source test suite.
The post-publication verifier reads the release execution receipts instead of
re-running the entire MCP suite under an obsolete short diagnostic deadline.
Diagnostic and live-account workflows remain manual, not development gates.

The `dev` extra includes the MCP SDK so shared installed-service fixtures collect
and execute in the ordinary quality lanes. The `all` extra also enables MCP.
Fresh base-package installation checks still verify operation without the SDK.

## Exact full partition

`Tests Nightly` runs the full configured scope weekly. An integration-only manual
diagnostic remains available; it cannot satisfy release verification. The release
orchestrator calls it with `run_full_suite=true`. The collector uses
the unchanged expression `contract or fault or integration or soak or stress or
scale_100m or e2e` with all three opt-ins: `--run-integration`, `--run-expensive`
and `--run-soak`. This is that configured marker universe, not a claim that every
test in the repository is selected. All jobs use the same frozen dev/MCP extras.

Each collected node ID has exactly one owner, in this precedence:
scale, soak/stress, integration, fault/e2e, then contract. Stress-directory tests
already gain `soak` in the root conftest; overlapping markers are not separate
executions. Contract owners use four shards. Ordinary files retain the PR
file-grouping algorithm. The independently executable parameter cases in
`test_mcp_algorithm_parity.py` are spread evenly across those same four shards,
as they already are in the mandatory MCP matrix. This prevents one large native
parity module from monopolizing a lane; no case, native computation, fixture row,
assertion or deadline is removed or reduced. The split allowlist is explicit;
other files keep their fixture locality. Collection order does not affect the plan.
The other four lanes isolate integration, soak/stress, fault/e2e and 100M scale.
The plan proves exact union and zero intersection before any lane runs.
The collector also requires each item's actual imported module file to equal its
collected path and records the source file SHA-256. Each lane checks those source
identities again; repeated basenames cannot silently substitute another module.
Full jobs explicitly install LZ4, Argon2 and cryptography through the dev extra
for compression/encryption roundtrips and default password-protected key stores.
Optional dependency diagnostics record absence without erasing the original test
result; the full collector/aggregate still rejects a missing mandatory MCP SDK.

The stress metrics collector uses a reentrant snapshot lock for nested summary
properties. Its bounded regression covers concurrent writes and JSON/Markdown/HTML
reports. Stress result failure rates are derived from actual completed operation counts,
including the early-stop path, rather than an absent result attribute.
The scale uniqueness fixture retains 100 million rows and its existing
300-second threshold. It uses an exact Python set (not a Core HLL validator) and
takes the same first 1,000 prior IDs without repeatedly copying the entire set.
The in-memory HEAD/NONE sampling timing test keeps its 5x overhead ceiling but
uses one equal warmup and seven fixed alternating pairs, comparing medians. It
includes construction, sampling and collection for both paths, checks actual row
counts, and has a deliberately slow-path regression. This measures warm relative
overhead, not cold-start latency; it does not retry until passing or alter the
separate fixed-runner release performance criteria.

A matrix runs at most three lanes concurrently with fail-fast disabled. Jobs
have independent environments, filesystems and fixtures: session fixtures and
in-memory service setup cannot be shared across them. Setup is intentionally
repeated. Balancing by file/test count preserves fixture locality but does not
guarantee balanced duration. No new paid runner or service account is provisioned.

## Deadlines and evidence

| Lane | Job minutes | Hard test seconds | Soft stack seconds |
| --- | ---: | ---: | ---: |
| Each contract shard | 35 | 120 | 60 |
| Fault/e2e | 25 | 180 | 90 |
| Integration | 35 | 180 | 90 |
| Soak/stress | 35 | 300 | 120 |
| 100M scale | 120 | 600 | 120 |

The scale per-test ceiling follows the existing scale suite's 600-second example;
ten tests plus environment setup fit within the job ceiling. Other bounds allow
the existing tens-of-seconds stress generators and fault deadlines headroom;
they are engineering budgets, not measured performance promises. Original test
volume, assertions and stricter throughput/runtime thresholds are unchanged.
The scale lane checks at least 4 logical CPUs, 16 GB total RAM, 12 GB available
RAM and 10 GB free disk before starting. These are decimal bytes, not GiB.
A standard public Ubuntu runner nominally meets the CPU/RAM profile; actual
availability still gates execution. Preflight is not an OOM guarantee.

Every test emits its node ID at start. Partial NDJSON records phase outcomes and
durations as they arrive; JUnit, final execution manifests, environment versions,
resource observations and the collection plan are retained for seven days.
The manifest runner adds the checkout root to its Python import path so spawned
native workers can import the same test callables as the parent process.
Ray runtime packaging also retains each explicitly allowlisted public JSON
resource required by the package build; private JSON remains excluded.
Native checkpoint publication and atomic stores flush their staged files using writable
descriptors, including on Windows. The manually dispatched Windows native
regression workflow runs the real engines and installed operator lifecycles on
Python 3.11–3.13 to diagnose platform failures; it does not replace full release
verification. Failure diagnostics contain exception types, structural stack
frames and closed cause flags, without native messages or full filesystem paths.
Collection installs the native PDF import libraries. Execution lanes retain
Poppler and CJK fonts for actual PDF rendering checks. APT acquisition uses
bounded timeouts and retries; download failures still fail the job.
On Windows, native SDK scratch directories stay under the private driver
workspace until that driver exits. Its parent then removes the workspace after
reaping owned descendants, so native log handles cannot race directory deletion.
Windows Spark checkpoint tests first require the SDK's Hadoop native/winutils
availability checks to pass, then execute the checkpoint. Missing prerequisites
fail the test. Diagnostics never invoke an SDK exception's remote formatter.
The Windows SDK workflow builds Hadoop 3.5.0 from Apache source commit
`dbcc7cd797100e6b32cd84f85b53a5193a5f9af0` using its `native-win` compile phase.
Only the native binaries, license, notice and source/file hash provenance are
retained. Cache identity includes the build script and runner image; restored
files are verified before the native product tests run. Operator factories pin
SDK settings privately, rather than accepting paths through MCP requests.
Pytest faulthandler emits stacks at the soft threshold without killing the test.
The separate pytest-timeout thread deadline terminates a stuck pytest process;
such termination can prevent final JSON/JUnit cleanup. Partial evidence remains,
and missing final evidence fails the gate rather than becoming a pass.

The always-running final gate requires successful collector and every matrix
job, matching checkout SHA/run ID/plan digest/dependencies, exact executed node
IDs, setup/call/teardown results and agreeing JUnit totals. Unexpected skips,
xfail, failures, cancellation, missing/extra phases or artifacts cannot pass.
Only skip/skipif conditions evaluated by pytest during collection are recorded
as expected optional skips. Their reason and execution must agree; counts stay
separate from passed tests. MCP and scale requirements cannot receive this waiver.
Runtime-only skips are not silently allowlisted. Collection-time module skips
are preserved separately and do not count as executed tests.

Live BigQuery, Snowflake, Redshift and Databricks fixture closures require
`TRUTHOUND_CLOUD_DW_RUN_LIVE=1`; the existing paid OpenAI smoke tests require
`TRUTHOUND_AI_RUN_LIVE_SMOKE=1`. Set either only with separate account/cost
authorization and prerequisites. The release does not enable them: their selected
tests remain explicit `live_account` SKIP/unverified, never PASS. Hermetic warehouse
configuration/generator tests remain active. An existing optional sibling-consumer
snapshot check is likewise recorded as `optional_consumer` when unavailable.
Unexpected runtime skips, including missing local libraries and service startup
failures, are not covered by this boundary.
Other pre-existing collection-time optional skips (credential-gated cloud stores,
Kafka/LocalStack Testcontainers and a default-disabled Redis-stream test) remain
separately reported as `other_optional_skipped`; they are not claimed as covered.
This is not a waiver for newly failing required tests or runtime skips.

Local Nightly prerequisites additionally install the existing benchmark/PDF
extras and the `nightly-local` dependency group (OpenTelemetry SDK, Zstandard,
pypdf). Ubuntu jobs install Pango, Poppler and CJK fonts and require PDF smoke and
render checks. Missing local prerequisites are not converted into skip waivers.
The integration VM owns disposable Redis and Elasticsearch service containers,
accessed as real local endpoints, not mocks. Existing mock KMS/TMS fixtures are
explicitly recorded as mock coverage, not live cloud verification. Startup or
readiness failure fails the tests. SQLAlchemy queue pools forward
`pool_timeout` to the engine; real SQLite tests verify the configured timeout and
query execution without a database account.

Elasticsearch log fixtures retain all 100 rows while datetime arithmetic rolls
seconds/minutes into valid timestamps. Cleanup first resolves a literal test-index
prefix and deletes only the explicit matching names; empty/wildcard prefixes and
foreign resolved names are rejected. Elasticsearch's destructive-action name
protection remains enabled. Fixture shutdown still runs if cleanup raises, and
cleanup failures remain failures rather than being suppressed.

The dev environment requires pytest 9. Native subtests are recorded separately
from the parent's setup/call/teardown phases with consecutive parent-local
ordinals, without storing their context/parameter values. Parent testcase counts
and parent-plus-subtest JUnit totals must both agree. Failed, skipped or xfailed
subtests, missing ordinals, duplicate events and foreign parents fail the gate;
subtests are neither ignored nor counted as extra top-level selected tests.
Collection validation runs before execution rather than raising from a finally
hook that could mask pytest's original missing-node error.

The previous cancelled monolithic run selected 9,477 nodes and stopped printing
progress at 19%. This establishes an observability gap, not a specific hanging
test, OOM or product root cause. New lane logs/stacks are needed for that diagnosis.

## Design sources and limits

This is a bounded engineering adaptation, not a systematic literature review
or an experimentally established speedup. References consulted include the
[GitHub workflow dependency/matrix/deadline contract](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax),
[hosted-runner resource table](https://docs.github.com/en/actions/reference/runners/github-hosted-runners),
[pytest fault-handler guidance](https://docs.pytest.org/en/stable/how-to/failures.html),
[pytest-timeout's termination limitations](https://github.com/pytest-dev/pytest-timeout),
and [SRE chapter 22 on bounded workloads/resources](https://sre.google/sre-book/addressing-cascading-failures/).
The repository's existing PR manifest runner and Depot's backend coverage/timeout
patterns informed the implementation; Depot workflows are not copied as-is.

Benchmark CLI jobs install the `benchmarks` runtime extra for both source and
released-wheel measurements. Workloads, assertions, fixed-runner identity and
thresholds remain the same. After artifacts are recorded, the release job removes
only its claimed temporary runtime: run ID, attempt, commit and canonical path must
match, and aliases or changed ownership records fail closed. Other run directories
and shared caches are retained.
