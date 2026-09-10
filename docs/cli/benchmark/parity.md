# `truthound benchmark parity`

Run the repo-tracked parity suites used for Truthound performance and correctness verification.

## Usage

```bash
truthound benchmark parity [OPTIONS]
```

## Key Options

| Option | Meaning |
| --- | --- |
| `--suite` | `pr-fast`, `nightly-core`, `nightly-sql`, or `release-ga` |
| `--frameworks` | `truthound`, `gx`, or `both` |
| `--backend` | optional filter: `local`, `sqlite`, or `duckdb-shadow` |
| `--output` | write the JSON artifact to a custom path |
| `--save-baseline` | save the suite result as the canonical parity baseline |
| `--compare-baseline` | compare Truthound against the saved baseline |
| `--strict` | fail on missing parity, threshold regressions, or unavailable requested frameworks |

## Measurement and Baseline Compatibility

New runs use `truthound-parity-thread-budget-v2`: one cold iteration, seven warm iterations, and an equal worker thread budget of one for both frameworks, configured before child-process imports. All warm wall-clock and CPU samples and the correctness of every cold/warm iteration are retained. A later correct iteration cannot hide an earlier mismatch. The existing correctness, speed, and memory thresholds are unchanged.

There are no new CLI options for this default. `--compare-baseline` requires matching methodology, workload contract fingerprint, dataset fingerprint, backend, and exactness; incompatible results fail comparison rather than producing a speedup claim. Older artifacts keep their original methodology and are not rewritten as v2.

## Examples

```bash
truthound benchmark parity --suite pr-fast --frameworks truthound --backend local --strict
truthound benchmark parity --suite nightly-core --frameworks both --backend local --strict
truthound benchmark parity --suite nightly-sql --frameworks both --backend sqlite --strict
truthound benchmark parity --suite release-ga --frameworks both --strict
```

## Release-Grade Verification Rules

`release-ga` is the authoritative fixed-runner verification suite and has additional rules:

- `--frameworks` must be `both`
- `--backend` must not be set
- The methodology must exactly match the default v2 contract, including seven warm iterations, one worker thread per library pool, and unchanged thresholds. Custom Python API methodologies are advisory; they fail the authoritative `release-ga:measurement-policy` assertion.

## Artifacts

Parity runs write artifacts under `.truthound/benchmarks/` and generate:

- JSON result
- Markdown summary
- HTML summary
- `env-manifest.json`

The JSON observations include the configured thread budget, observed Polars pool size, complete warm timing samples, per-iteration correctness, and workload contract fingerprint. These fields allow the verifier to reject incomplete or inconsistent measurements instead of trusting a summary alone.

`release-ga` also generates `latest-benchmark-summary.md` beside the chosen output path.

Verification of an unchanged published wheel with a corrected harness is a separate release-workflow lane, not a new option on this command. It records the wheel version and digest separately from the harness revision; see [Published Wheel Verification](../../guides/benchmark-methodology.md#published-wheel-verification).

## Related Reading

- [Performance and Benchmarks](../../guides/performance.md)
- [Benchmark Methodology](../../guides/benchmark-methodology.md)
- [Latest Verified Benchmark Summary](../../releases/latest-benchmark-summary.md)
