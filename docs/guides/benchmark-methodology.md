# Benchmark Methodology

## Purpose

Truthound 3.0 benchmark methodology exists to answer one question honestly:

Can the public zero-config Truthound path beat or at least match Great Expectations on the workloads that matter for release?

## Principles

- Measure the public API, not internal helpers.
- Use repo-tracked workload manifests and fixtures.
- Separate first-run baseline cost from warm-run steady state.
- Run each framework in its own child process.
- Reject performance wins that do not also preserve correctness.

## Measurement Model

Each framework/workload observation records:

- framework name and framework version
- workload id and dataset fingerprint
- backend and exactness class
- cold start seconds
- warm median seconds
- peak RSS bytes
- expected issue count
- observed issue count
- artifact paths for the per-workload run

Truthound records cold and warm runs in the same zero-config workspace so that baseline creation cost is visible in cold start and baseline reuse is visible in warm median.

## Child Process Isolation

Framework timing and memory are collected in child processes rather than the parent runner. This reduces contamination from:

- already-imported modules
- allocator state from previous workloads
- mixed framework caches
- shared process RSS inflation

## Runner Policy

Truthound uses a hybrid runner policy:

- GitHub-hosted nightly runners provide trend visibility and early warning
- self-hosted fixed runners provide the official release verdict

Nightly artifacts are informative. Release artifacts are authoritative.

For the authoritative release verdict, the fixed runner must record:

- CPU model
- logical core count
- RAM
- OS and Python minor
- storage class

The release workflow reads these from the fixed host plus the following environment contract:

- `TRUTHOUND_BENCHMARK_RUNNER_CLASS=self-hosted-fixed`
- `TRUTHOUND_BENCHMARK_RUNNER_LABELS=self-hosted,benchmark-fixed`
- `TRUTHOUND_BENCHMARK_RELEASE_VERDICT=true`
- `TRUTHOUND_BENCHMARK_STORAGE_CLASS`
- `TRUTHOUND_BENCHMARK_CPU_MODEL`
- `TRUTHOUND_BENCHMARK_RAM_BYTES`
- `TRUTHOUND_BENCHMARK_CPU_PHYSICAL_CORES`

For self-hosted macOS runners, the workflow also pins `RUNNER_TOOL_CACHE` and `AGENT_TOOLSDIRECTORY`
to a writable directory under the runner workspace so `actions/setup-python` does not attempt to install
toolcache payloads under `/Users/runner`.

## Thresholds

The current 3.0 gate thresholds are:

- local exact: `Truthound >= 1.5x GX`
- SQL exact: `Truthound >= 1.0x GX`, with `1.2x` as the target
- local memory: `Truthound <= 60% of GX peak RSS`

If correctness parity fails, the performance comparison is treated as failed even when timing looks better.

## Commands

```bash
truthound benchmark parity --suite pr-fast --frameworks truthound --backend local
truthound benchmark parity --suite nightly-core --frameworks both --backend local
truthound benchmark parity --suite nightly-sql --frameworks both --backend sqlite
truthound benchmark parity --suite release-ga --frameworks both --strict
```

`release-ga` is intentionally stricter than the nightly suites:

- it must run `--frameworks both`
- it must not use `--backend`
- it must execute all eight tier-1 local and SQLite workloads

## Artifact Layout

The benchmark artifact root is always `.truthound/benchmarks/` in the active project context:

- `results/`
- `baselines/`
- `artifacts/`
- `release/`

When a parity run writes an external output file, the canonical artifact set beside that output is:

- `release-ga.json`
- `release-ga.md`
- `release-ga.html`
- `env-manifest.json`
- `latest-benchmark-summary.md`

To publish the docs summary page from an approved release artifact set:

```bash
python docs/scripts/publish_benchmark_summary.py \
  --json benchmark-artifacts/release-ga.json \
  --artifact-base-url .. \
  --output docs/releases/latest-benchmark-summary.md
```

## Related Reading

- [Performance and Benchmarks](performance.md)
- [Workload Catalog](benchmark-workloads.md)
- [GX Parity Gate](gx-parity.md)
