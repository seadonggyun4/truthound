# Latest Benchmark Summary

## Status

The latest release-grade artifact set cleared the fixed-runner 3.0 GA benchmark gate.

- Suite: `release-ga`
- Passed: `yes`
- Official claim eligible: `yes`

## Artifact Links

- [release-ga.json](https://github.com/seadonggyun4/truthound/releases/download/v3.0.0/release-ga.json)
- [release-ga.md](https://github.com/seadonggyun4/truthound/releases/download/v3.0.0/release-ga.md)
- [release-ga.html](https://github.com/seadonggyun4/truthound/releases/download/v3.0.0/release-ga.html)
- [env-manifest.json](https://github.com/seadonggyun4/truthound/releases/download/v3.0.0/env-manifest.json)

## Comparable Workloads

| Workload | Truthound Warm (s) | GX Warm (s) | Speedup | Memory Ratio | Correctness |
| --- | ---: | ---: | ---: | ---: | --- |
| local-mixed-core-suite | 0.028240 | 0.075232 | 2.66x | 44.29% | pass |
| local-null | 0.016487 | 0.024964 | 1.51x | 43.62% | pass |
| local-range | 0.002470 | 0.013219 | 5.35x | 43.84% | pass |
| local-schema | 0.001479 | 0.017303 | 11.70x | 35.88% | pass |
| local-unique | 0.002023 | 0.013785 | 6.81x | 42.28% | pass |
| sqlite-null | 0.007370 | 0.032909 | 4.47x | 48.16% | pass |
| sqlite-range | 0.006053 | 0.022355 | 3.69x | 43.80% | pass |
| sqlite-unique | 0.002066 | 0.015655 | 7.58x | 42.12% | pass |

## Assertions

- [PASS] `local-mixed-core-suite:truthound:correctness`: Observed 3 issues; expected 3.
- [PASS] `local-mixed-core-suite:gx:correctness`: Observed 3 issues; expected 3.
- [PASS] `local-mixed-core-suite:issue-parity`: Truthound observed 3; GX observed 3.
- [PASS] `local-mixed-core-suite:local-speedup`: Truthound speedup 2.66x vs GX (target 1.50x).
- [PASS] `local-mixed-core-suite:local-memory-ratio`: Truthound RSS ratio 44.29% vs GX (max 60%).
- [PASS] `local-null:truthound:correctness`: Observed 1 issues; expected 1.
- [PASS] `local-null:gx:correctness`: Observed 1 issues; expected 1.
- [PASS] `local-null:issue-parity`: Truthound observed 1; GX observed 1.
- [PASS] `local-null:local-speedup`: Truthound speedup 1.51x vs GX (target 1.50x).
- [PASS] `local-null:local-memory-ratio`: Truthound RSS ratio 43.62% vs GX (max 60%).
- [PASS] `local-range:truthound:correctness`: Observed 1 issues; expected 1.
- [PASS] `local-range:gx:correctness`: Observed 1 issues; expected 1.
- [PASS] `local-range:issue-parity`: Truthound observed 1; GX observed 1.
- [PASS] `local-range:local-speedup`: Truthound speedup 5.35x vs GX (target 1.50x).
- [PASS] `local-range:local-memory-ratio`: Truthound RSS ratio 43.84% vs GX (max 60%).
- [PASS] `local-schema:truthound:correctness`: Observed 1 issues; expected 1.
- [PASS] `local-schema:gx:correctness`: Observed 1 issues; expected 1.
- [PASS] `local-schema:issue-parity`: Truthound observed 1; GX observed 1.
- [PASS] `local-schema:local-speedup`: Truthound speedup 11.70x vs GX (target 1.50x).
- [PASS] `local-schema:local-memory-ratio`: Truthound RSS ratio 35.88% vs GX (max 60%).
- [PASS] `local-unique:truthound:correctness`: Observed 1 issues; expected 1.
- [PASS] `local-unique:gx:correctness`: Observed 1 issues; expected 1.
- [PASS] `local-unique:issue-parity`: Truthound observed 1; GX observed 1.
- [PASS] `local-unique:local-speedup`: Truthound speedup 6.81x vs GX (target 1.50x).
- [PASS] `local-unique:local-memory-ratio`: Truthound RSS ratio 42.28% vs GX (max 60%).
- [PASS] `sqlite-null:truthound:correctness`: Observed 1 issues; expected 1.
- [PASS] `sqlite-null:gx:correctness`: Observed 1 issues; expected 1.
- [PASS] `sqlite-null:issue-parity`: Truthound observed 1; GX observed 1.
- [PASS] `sqlite-null:sql-speedup`: Truthound speedup 4.47x vs GX (floor 1.00x, target 1.20x).
- [PASS] `sqlite-range:truthound:correctness`: Observed 1 issues; expected 1.
- [PASS] `sqlite-range:gx:correctness`: Observed 1 issues; expected 1.
- [PASS] `sqlite-range:issue-parity`: Truthound observed 1; GX observed 1.
- [PASS] `sqlite-range:sql-speedup`: Truthound speedup 3.69x vs GX (floor 1.00x, target 1.20x).
- [PASS] `sqlite-unique:truthound:correctness`: Observed 1 issues; expected 1.
- [PASS] `sqlite-unique:gx:correctness`: Observed 1 issues; expected 1.
- [PASS] `sqlite-unique:issue-parity`: Truthound observed 1; GX observed 1.
- [PASS] `sqlite-unique:sql-speedup`: Truthound speedup 7.58x vs GX (floor 1.00x, target 1.20x).
- [PASS] `release-ga:runner-policy`: Release verdicts require a fixed self-hosted runner with `self-hosted,benchmark-fixed` labels and `TRUTHOUND_BENCHMARK_RELEASE_VERDICT=true`.
- [PASS] `release-ga:runner-metadata`: Release artifacts must document CPU model, logical cores, RAM, Python minor, and storage class.
- [PASS] `release-ga:framework-selector`: The release-ga suite must run both Truthound and GX.
- [PASS] `release-ga:backend-filter`: The release-ga suite must execute the full local + SQLite catalog without a backend filter.
- [PASS] `release-ga:catalog-size`: The release-ga suite must contain exactly 8 repo-tracked tier-1 workloads.

## Related Reading

- [Performance and Benchmarks](../guides/performance.md)
- [Benchmark Methodology](../guides/benchmark-methodology.md)
- [GX Parity Gate](../guides/gx-parity.md)
