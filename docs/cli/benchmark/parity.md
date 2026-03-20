# `truthound benchmark parity`

Run the repo-tracked parity suites used for Truthound 3.0 performance and correctness verification.

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

## Examples

```bash
truthound benchmark parity --suite pr-fast --frameworks truthound --backend local --strict
truthound benchmark parity --suite nightly-core --frameworks both --backend local --strict
truthound benchmark parity --suite nightly-sql --frameworks both --backend sqlite --strict
truthound benchmark parity --suite release-ga --frameworks both --strict
```

## Release-Grade Verification Rules

`release-ga` is the authoritative fixed-runner verification suite and has two additional rules:

- `--frameworks` must be `both`
- `--backend` must not be set

## Artifacts

Parity runs write artifacts under `.truthound/benchmarks/` and generate:

- JSON result
- Markdown summary
- HTML summary
- `env-manifest.json`

`release-ga` also generates `latest-benchmark-summary.md` beside the chosen output path.

## Related Reading

- [Performance and Benchmarks](../../guides/performance.md)
- [Benchmark Methodology](../../guides/benchmark-methodology.md)
- [Latest Verified Benchmark Summary](../../releases/latest-benchmark-summary.md)
