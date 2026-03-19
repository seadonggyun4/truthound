# Latest Benchmark Summary

## Status

No release-grade benchmark artifact has been published yet for Truthound 3.0 GA.

This page exists so the documentation can point to the canonical place where benchmark evidence will be attached once the self-hosted release gate has produced an approved result set.

When an approved artifact set exists, regenerate this page with:

```bash
python docs/scripts/publish_benchmark_summary.py \
  --json benchmark-artifacts/release-ga.json \
  --artifact-base-url .. \
  --output docs/releases/latest-benchmark-summary.md
```

## Publication Policy

Truthound does not publish illustrative benchmark numbers as official claims.

Official benchmark claims must come from:

- the repo-tracked parity workloads
- the self-hosted fixed release runner
- the JSON, Markdown, and HTML artifacts written under `.truthound/benchmarks/release/`

The external release artifact bundle should also contain:

- `env-manifest.json`
- `latest-benchmark-summary.md`

## What Will Appear Here

When the release gate passes, this page should link to:

- the release JSON artifact
- the release Markdown summary
- the release HTML summary
- the release environment manifest
- the workload manifest revision used for the claim
- the runner class and framework versions

## Related Reading

- [Performance and Benchmarks](../guides/performance.md)
- [Benchmark Methodology](../guides/benchmark-methodology.md)
- [GX Parity Gate](../guides/gx-parity.md)
