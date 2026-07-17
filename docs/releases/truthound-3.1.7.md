# Truthound 3.1.7 Release Notes

## Highlights

Truthound 3.1.7 repairs the structured-file contracts discovered by the
Truthound Depot mixed-asset operational gate. Complete JSON documents now have
the same semantics through the root API and `FileDataSource`, and nested JSON
columns can participate in automatic drift comparison without hashability
errors.

## Complete JSON document contract

`.json` is parsed as one complete document rather than being inferred as
NDJSON from its first character.

- an array of objects produces one row per object;
- a top-level object produces one row containing its fields;
- an array of scalars produces a `value` column;
- a top-level scalar produces one `value` row;
- `.ndjson` and `.jsonl` remain separate line-delimited lazy-scan formats.

This removes the former failure where a legal top-level object started with
`{` and was incorrectly sent to the NDJSON scanner. Array JSON no longer
creates an unmanaged temporary NDJSON file.

## Nested drift comparison

`truthound.drift.compare(..., method="auto")` now canonicalizes `Struct` and
`List` values to deterministic JSON categories before invoking categorical
detectors and statistics. Object key order is normalized, null values remain
null, and `ColumnDrift.dtype` continues to report the original Polars dtype.

Explicit numeric methods such as `ks` and `psi` still require numeric columns.

## Consumer upgrade gate

Consumers such as Truthound Depot must install the published 3.1.7 wheel,
verify `truthound.__version__`, and rerun profile, validation, drift, anomaly,
serialization or re-entry, and mixed-asset lifecycle checks. A source checkout
or unpublished wheel is not consumer certification evidence.
