# Repository CI and full nightly verification

PR checks retain four file-balanced contract shards, fault/e2e, lint ratchets and
their aggregate gate. The separate MCP distribution workflow retains nine
OS/Python combinations, mandatory no-skip SDK tests and fresh wheel/sdist/UV
installation checks. Core `src/**`, shared pytest settings and the lockfile
remain MCP inputs. Neither a green PR lane nor an MCP matrix substitutes for
full release verification.

## Exact full partition

`Tests Nightly` defaults to the existing integration-only scheduled scope.
For release verification, dispatch `run_full_suite=true`. The collector uses
the unchanged expression `contract or fault or integration or soak or stress or
scale_100m or e2e` with all three opt-ins: `--run-integration`, `--run-expensive`
and `--run-soak`. This is that configured marker universe, not a claim that every
test in the repository is selected. All jobs use the same frozen dev/MCP extras.

Each collected node ID has exactly one owner, in this precedence:
scale, soak/stress, integration, fault/e2e, then contract. Stress-directory tests
already gain `soak` in the root conftest; overlapping markers are not separate
executions. Contract owners reuse the PR file-grouping algorithm in four shards.
The other four lanes isolate integration, soak/stress, fault/e2e and 100M scale.
The plan proves exact union and zero intersection before any lane runs.
The collector also requires each item's actual imported module file to equal its
collected path and records the source file SHA-256. Each lane checks those source
identities again; repeated basenames cannot silently substitute another module.
Full jobs explicitly install LZ4 through the dev extra for encryption roundtrips.
Optional dependency diagnostics record absence without erasing the original test
result; the full collector/aggregate still rejects a missing mandatory MCP SDK.

The stress metrics collector uses a reentrant snapshot lock for nested summary
properties. Its bounded regression covers concurrent writes and JSON/Markdown/HTML
reports. The scale uniqueness fixture retains 100 million rows and its existing
300-second threshold. It uses an exact Python set (not a Core HLL validator) and
takes the same first 1,000 prior IDs without repeatedly copying the entire set.

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
