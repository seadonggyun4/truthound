# Benchmark Methodology

## Purpose

Truthound benchmark methodology exists to answer one question honestly:

Can the public zero-config Truthound path outperform Great Expectations on comparable release-grade workloads while preserving correctness?

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
- every warm wall-clock duration and process CPU duration
- correctness of the cold iteration and every warm iteration
- peak RSS bytes
- expected issue count
- observed issue count
- artifact paths for the per-workload run
- methodology, configured worker thread budget, and observed Polars thread pool size
- the workload contract fingerprint, separate from the dataset fingerprint

Truthound records cold and warm runs in the same zero-config workspace so that baseline creation cost is visible in cold start and baseline reuse is visible in warm median.

## Versioned Measurement Contract

New runs default to `BenchmarkMethodology.name = "truthound-parity-thread-budget-v2"` with one cold iteration, `warm_iterations = 7`, and `worker_threads = 1`. The Python runner can accept an explicit methodology; changing that methodology creates a different comparison contract. Custom methodologies are advisory, not an authoritative release verdict: `release-ga:measurement-policy` requires equality with the complete default `BenchmarkMethodology()`, including its thresholds. The release-wheel verifier also uses the strict defaults. No new `truthound benchmark parity` CLI flags are required.

Observation metadata retains `warm_durations_seconds`, `warm_process_cpu_seconds`, `iteration_correctness`, `worker_threads`, `polars_thread_pool_size`, and `workload_contract_sha256`. The warm median uses every recorded warm duration, not the fastest sample or a selected subset. Correctness must hold for the cold run and every warm run; a correct final iteration does not erase an earlier incorrect result. A dataset's expected quality failures remain genuine failures: correctness means matching the workload's expected outcome, not changing every validation into a pass.

Missing, non-finite, inconsistent, or incomplete measurement receipts fail verification. Recording host load describes the measurement environment; it does not prove that the host was idle or that timing was stable.

Historical `truthound-3.0-parity-gate` artifacts retain their original meaning, including two warm iterations and an unspecified thread budget when the field was absent. They are not silently upgraded to v2 or overwritten. Baseline comparison requires matching methodology, workload contract fingerprint, dataset fingerprint, backend, and exactness. An incompatible baseline is rejected as non-comparable, not reported as a speedup or regression.

## Child Process Isolation

Framework timing and memory are collected in child processes rather than the parent runner. This reduces contamination from:

- already-imported modules
- allocator state from previous workloads
- mixed framework caches
- shared process RSS inflation

Before either framework's child process imports libraries, the runner applies the same thread budget through `POLARS_MAX_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and `TRUTHOUND_BENCHMARK_WORKER_THREADS`. The recorded Polars pool size must match the configured budget. This bounds the participating library pools; it is not a claim that the process has only one operating-system thread or that concurrent host workloads cannot affect it.

## Runner Policy

Truthound uses a hybrid runner policy:

- GitHub-hosted nightly runners provide trend visibility and early warning
- fixed self-hosted runners provide the authoritative release-grade benchmark verification

Nightly artifacts are informative. Fixed-runner release artifacts are authoritative.

For the authoritative release verdict, the runner must record:

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

For self-hosted macOS runners, the release workflow avoids `actions/setup-python` and instead uses `uv` to install a managed Python 3.11 runtime plus a run-owned virtual environment. Each workflow run and attempt uses its own environment; it does not clear a shared `.release-venv`. This also sidesteps hosted-toolcache assumptions that often point at `/Users/runner` on macOS.

## Thresholds

The current release-grade thresholds are:

- local exact: `Truthound >= 1.5x Great Expectations`
- SQL exact: `Truthound >= 1.0x Great Expectations`, with `1.2x` as the target
- local memory: `Truthound <= 60%` of Great Expectations peak RSS

Comparison against a compatible saved Truthound baseline retains the existing maximum warm-median regression of `10%`.

If correctness parity fails, the performance comparison is treated as failed even when timing looks better.

The v2 measurement correction does not lower these thresholds. Historical published numbers remain historical results; a new performance claim requires a new verified artifact set under the stated methodology.

## Published Wheel Verification

The release workflow separates source-checkout verification from verification of an already published wheel. `scripts/benchmarks/verify_release_package.py` prepares and runs the latter using the corrected benchmark harness without replacing the installed Truthound product with checkout code.

For the published-wheel lane, dispatch the release workflow with `target=released-wheel` and explicit `release_version`, `release_tag`, `release_commit` (40 lowercase hexadecimal characters), and `wheel_sha256` (64 lowercase hexadecimal characters). The dispatched workflow revision identifies the harness commit. The release tag and PyPI wheel remain unchanged; the existing source-checkout lane is separate.

Published-wheel provenance records the release version, release tag and commit, wheel SHA-256, installed package identity, harness commit, methodology, and workload manifest digest separately. The harness is loaded in a separate namespace, while the measured public API comes from the verified installed wheel. Worker imports must not fall back to the checkout or a user site package.

The JSON result stores these details under `metadata.release_package`, including `harness_files`, `harness_sha256`, and the content-addressed `workloads` manifest. It identifies `measurement_target="immutable-installed-wheel"` and `harness_namespace="_truthound_release_harness"`. Each observation binds that provenance with `metadata.release_package_provenance` and records `metadata.release_package_measurement_integrity`. The workflow also records installed dependency names and versions in `installed-dependencies.json`, without retaining raw package-source URLs from a freeze listing.

A corrected harness testing the unchanged `3.1.15` wheel is evidence about that wheel under that harness. It is not evidence that a modified source checkout was published as `3.1.15`, and it must not replace the original release artifacts or relabel earlier failures as passes. Both provenance and the actual correctness, timing, and memory gates are required for a release verdict.

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
- it must use the complete default v2 methodology, including unchanged thresholds

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
- [Great Expectations Comparison](gx-parity.md)
- [Workload Catalog](benchmark-workloads.md)
- [Docs Deployment Verification](docs-deployment-verification.md)
