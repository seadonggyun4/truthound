# Schema Audit Contracts

`truthound.schema_audit` is an in-development, separately imported metadata and audit A/B namespace.
It includes a bounded PostgreSQL14/17 catalog adapter, not an ERD editor/importer,
connection authorization, approval or release workflow. Existing
`th.check()` and table-schema validators are unchanged. No new package extra is needed;
the namespace itself uses only the Python standard library. The root package retains
its usual dependencies. Unreleased source functionality is not a PyPI availability claim.

## Design-to-Catalog Audit B

```python
from truthound.schema_audit import audit_implementation
from truthound.schema_audit.postgresql import collect_postgresql

# connection is a dedicated, caller-authenticated asyncpg.Connection.
# The caller must authorize the destination and configure verified TLS.
snapshot = await collect_postgresql(
    connection, schemas=("application",), binding_ref="approved-binding",
    snapshot_id="snapshot-1", timeout_ms=3000, attempts=2,
)
report = audit_implementation(design, snapshot)
result = report.as_dict()
```

The caller installs `asyncpg` and closes the connection. The adapter does not resolve
credentials or accept SQL/URLs. It rejects an already active transaction. It uses
read-only transactions, a repeatable-read catalog snapshot, bounded ACCESS SHARE
locks acquired before catalog reads, and explicit catalog scope. It never selects
application rows or expands scope to follow foreign keys. Source database schemas
must have USAGE and tables SELECT privileges; SELECT is required for locks, not row
collection. System schemas are rejected. Missing/inaccessible schemas and hidden
tables lower coverage rather than producing a false missing-table failure.

Limits: 16 schemas, 500 table-like objects, 10,000 combined tables/columns/constraints,
4,096 characters per metadata value. `timeout_ms` is 100..10,000 and `attempts` 1..3.
Each attempt has a wall-time budget of four times the statement timeout. Only
catalog-change and timeout failures retry. `CollectionError.code` never contains
driver messages, connection details or SQL values. Cancellation propagates and
transactions are rolled back. Collection is not a guarantee against future DDL.

Ordinary PostgreSQL14/17 tables are supported. Versions 15/16/18 are not implicitly
enabled. The design and snapshot must have the same major version. PG14 uses its
own catalog projection rather than referencing PG15+ constraint columns. Negative
numeric scale and scale greater than precision remain unsupported in the PG14 normalizer.
The implementation rule pack is version 2; old results must not be relabeled.
Partitioned/inherited tables, views,
foreign tables, unvalidated constraints, initially deferred constraints, nondefault
FK match semantics, subset SET NULL/DEFAULT and NULLS NOT DISTINCT unique semantics
lower coverage. Cross-scope FK targets are not fetched. Standalone unique indexes
also lower constraint coverage. Custom types/domains/enums, array dimensionality
and timestamp precision zero remain conservative UNKNOWN where the v1 model cannot
represent them exactly. Character lengths are character counts, not byte counts.

Comparison uses exact dialect-aware namespace/table/column/constraint names, ordered
keys and referenced columns, FK update/delete actions, type facets, nullability and
ordinal positions. Renames are not inferred. Logical names use the entire COMMENT
value with NFC normalization; absent comments or external-mapping claims remain
UNKNOWN. Default/check expressions are never executed: equal trimmed strings match;
different simple literals differ; other expression differences are UNKNOWN, not
asserted SQL semantic equivalence or inequality. Default absence versus presence is
a difference. Unsupported names cannot produce trustworthy missing/extra findings.

Every result pins semantic input digests, rule-pack version/digest, dual provenance
and stable check fingerprints. `ImplementationReport.as_dict()` produces the
`truthound.implementation-audit/1` output contract, not an input accepted by
`load_document()`. Any FAIL yields NONCONFORMANT while UNKNOWN counts remain visible;
incomplete coverage or empty targets cannot yield CONFORMANT. No schema is modified.

Research basis: [PostgreSQL17 catalogs](https://www.postgresql.org/docs/17/catalog-pg-constraint.html)
define ordered keys/actions and validation flags; [visibility rules](https://www.postgresql.org/docs/17/infoschema-columns.html)
explain why filtered metadata is not proof of absence. [Transaction consistency](https://www.postgresql.org/docs/17/applevel-consistency.html)
informs bounded locking and snapshot boundaries. [Rahm and Bernstein](https://www.microsoft.com/en-us/research/publication/on-matching-schemas-automatically/)
motivates keeping inferred correspondences separate from exact matching. The
[DDIA author overview](https://dataintensive.net/) informs transaction and immutable
evidence boundaries; this is not a claim to have read the full paid book.

## Loading and Identity

```python
from truthound.schema_audit import (
    load_document, dump_document, raw_digest, semantic_digest, json_schema,
)

payload = b'{"contract_version":"truthound.schema-audit/1","kind":"DESIGN","id":"design1","revision":1,"dialect":{"name":"postgresql","major":17},"tables":[],"sources":[]}'
model = load_document(payload)
assert load_document(dump_document(model)) == model
original_sha256 = raw_digest(payload)
content_sha256 = semantic_digest(model)
structural_schema = json_schema()
```

An empty model is representable; loading it is **not** an audit PASS.
`load_document(bytes) -> StandardRevision | DesignRevision | CatalogSnapshot`
accepts strict UTF-8 JSON. All defined fields must be present; unknown fields,
duplicate JSON keys, invalid versions, duplicate IDs and dangling graph references
are rejected. Constructor arguments are also checked without type coercion.
Models use frozen dataclasses and nested tuples, not mutable input dictionaries.

`ModelError` provides `code` and `pointer`. Known-field validation uses JSON pointers;
duplicate-key and pre-decoding resource errors use structural numeric/container
locations, never unknown property names or payload values. Do not log the input on
failure. JSON syntax errors have a root pointer, not a raw parser excerpt.

Limits: 20 MiB input/output, nesting 32, 4,096 Unicode code points per string,
50,000 elements per array, 1,000 tables, 50,000 total columns and 1,000,000 decoded
nodes. Numbers are integers within ±(2^53−1); bool is not an integer. NaN, Infinity,
floating point, NUL and lone surrogates are rejected. These are contract safety
limits, not measured production throughput promises.

The packaged `schemas/document-v1.json` is JSON Schema 2020-12 and matches
`json_schema()`. It covers structure and primitive types. Cross-reference integrity,
resource limits and observation/constraint invariants additionally require
`load_document`; a generic JSON Schema validator alone is insufficient.

## Models

| Value | Responsibility |
| --- | --- |
| `StandardRevision` | Words, ordered term word IDs, domains, naming policy, revision identity, source locations |
| `DesignRevision` | Dialect, names, tables, columns, constraints, revision identity, source locations |
| `CatalogSnapshot` | Same object graph plus opaque binding, adapter version, UTC acquisition time, explicit scope and facet coverage |
| `TypeDescriptor` | Original native token, type schema, length/unit, precision/scale, timezone, array rank, domain/enum references, collation |
| `Attribute` | `VALUE`, `ABSENT`, `UNKNOWN`, `NOT_APPLICABLE`, typed scalar and closed reason code |
| `Identifier` | Original decoded identifier text and `UNQUOTED`, `QUOTED` or `CATALOG` origin |
| `LogicalName` | Observed name and explicit DESIGN/COMMENT/EXTERNAL_MAPPING/NONE source |
| `SourceLocation` | Object ID, source SHA-256 and source JSON pointer, not a filesystem location |
| `Finding`, `FindingSet` | Stable result IDs, rule version, verdict/reason and deterministic ordering; no evaluator |

`known(value)`, `absent()` and `unknown(reason=Reason.NOT_COLLECTED)` construct
observations. Null payloads mean no observed value and require a non-VALUE state;
SQL `DEFAULT NULL`, when observed, is an expression string, not JSON null.
Unknown observations require a reason; absence is not lack of permission.
Catalog coverage explicitly lists TABLES, COLUMNS, CONSTRAINTS, LOGICAL_NAMES,
TYPES and DEFAULTS. Incomplete catalogs must omit unresolved relationship edges
and mark coverage partial rather than claiming a closed complete graph. Domain/enum
type references are opaque adapter references, not automatically resolved objects.
Approval evidence and tenant authorization remain the consuming application's job.

## Normalization and Digests

`normalize_identifier(identifier, dialect)` and `normalize_type(type, dialect)`
return an `Attribute`, not an equivalence verdict. Unsupported input is UNKNOWN.
PostgreSQL majors 14 and 17 are built in. An immutable `NormalizationRegistry` supports
explicit local injection; no uploaded code or automatic version fallback is used.
Injected normalizers do not alter the built-in versioned semantic digest policy.

* Decoded quoted/catalog names are preserved byte-for-byte. Unquoted ASCII names
  are folded to lowercase. Non-ASCII unquoted input and names exceeding the default
  PostgreSQL UTF-8 63-byte identifier limit are UNKNOWN, never truncated or guessed.
* Resolved built-in type tokens require `type_schema=known("pg_catalog")` and no
  domain/enum override. Exact aliases such as int4/integer and decimal/numeric are
  normalized. Qualified tokens, SQL declarations such as `numeric(10,2)`, serial
  pseudo-types and custom types are not parsed or guessed.
* All facets remain present. Length units, arrays, numeric negative scale (including
  scale greater than precision), timezone and collation are not erased. Contradictory
  temporal timezone observations are UNKNOWN. No SQL/default expression is executed.
* Logical labels use NFC with Python's fixed Unicode 3.2 database for reproducibility
  across supported Python versions. Later Unicode additions are preserved, not fully
  normalized. Physical names, abbreviations and SQL expressions are never NFC-folded.

`raw_digest(bytes)` hashes exact original bytes. `semantic_bytes(model)` uses the
`truthound.schema-audit.semantic/1` profile and `pg17-ascii-nfc32-logical-v1` policy;
`semantic_digest(model)` is its SHA-256. This is **not RFC 8785/JCS**. Object keys
are sorted, ID-addressed collections are sorted by ID, and coverage/scope order is
canonicalized. Composite-key columns, FK pairs, word IDs and column ordinals retain
their semantics. Root ID/revision, sources, capture time, adapter version and binding
are excluded; object IDs, dialect and coverage are retained. Envelope identity and
authorization must therefore be bound separately by callers. Different object IDs
need not hash equally even if a human considers two diagrams equivalent.

## Standards-to-Design Audit A

```python
from truthound.schema_audit import audit_standards, load_document

# Both arguments are previously validated immutable models, not DB connections.
def audit_bytes(standard_json: bytes, design_json: bytes) -> dict:
    return audit_standards(
        load_document(standard_json), load_document(design_json)
    ).as_dict()
```

`audit_standards(StandardRevision, DesignRevision, *, bindings=())` returns a frozen
`StandardsReport`. The `standards-v1` pack checks ASCII naming syntax/case/byte
length, exact NFC logical dictionary lookup, ordered abbreviation composition,
deprecated terms/words, domain type and all explicit facets, and physical-name
uniqueness. Table labels resolve exact words or terms; columns require terms.
No substring, translation, morphology or fuzzy match can produce a PASS. Prefixes
and suffixes must be explicit dictionary words; there is no arbitrary regex policy.

Domain assignment is inherited from the resolved term's `domain_id`. A descriptor's
`domain_ref` represents a DB user-defined type, not a standard-domain identifier.
Missing logical names produce FAIL; unknown or ambiguous labels produce UNKNOWN.
Unresolved terms leave domain checks UNKNOWN. Dangling domain/FK references are
rejected by the common model before execution, not silently repaired by this pack.
Unsupported dialects or custom types remain UNKNOWN; known facet differences still
produce independent FAIL findings. Character length and byte length are not equated.

Every check records rule/version, target, expected/observed observations, both input
source locations, standard IDs and a deterministic fingerprint. Suggestions show an
expected physical name but never rewrite the model or erase a violation. Trusted
callers may supply `TermBinding` values with approval and both semantic input digests;
these values do not authenticate approval themselves. HTTP consumers must resolve
approval from their own authorized repository. Depot v1 does not accept caller bindings.

The report pins input IDs/revisions/semantic digests, mapping digest and rule-pack
digest. Policy changes require a new rule version. Verdict is NONCONFORMANT if any
check fails, INDETERMINATE if any required check is unknown or no targets exist,
otherwise CONFORMANT. Coverage retains FAIL and UNKNOWN independently by rule;
worker completion is not an audit PASS. The report JSON is a separate output contract
`truthound.standards-audit/1`, not an input to `load_document` or `json_schema()`.

Execution is bounded to 10,000 tables plus columns and 32 source locations per object.
Reports permit at most 400,000 checks without relaxing the 50,000-item input array
limit. These are Core computation budgets, not an end-to-end Depot capacity promise:
Depot applies separate 20 MiB input, 80 MiB raw-result and 16 MiB result-envelope
budgets and rejects oversized results. The result envelope may use bounded lossless compression.
Abbreviation combinations exceeding 4,096 code points are UNKNOWN without building
an unbounded concatenation. Repeated term resolutions are cached within one run.
These are safety limits, not a throughput claim. This engine does not execute SQL,
inspect row values, approve a revision or operate a Release gate.

### Audit A References

The public [ISO 11179-31 abstract](https://www.iso.org/standard/78925.html) informed
separate term/domain concepts. [SHACL validation reports](https://www.w3.org/TR/shacl/#validation-report)
informed target/path/constraint evidence, not RDF/SHACL compatibility or its verdict
semantics. [Rahm and Bernstein, On Matching Schemas Automatically](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2001-17.pdf)
informed the boundary between a match candidate and trusted correspondence.
[Helland, Immutability Changes Everything](https://www.cidrdb.org/cidr2015/Papers/CIDR15_Paper16.pdf)
informed fixed-input replay. [Kleppmann's DDIA author overview](https://dataintensive.net/)
was consulted for architecture trade-offs, not treated as a fully read textbook.
These are design references, not certification. Rule acceptance uses exact and
metamorphic synthetic tests in `tests/schema_audit/test_standards.py`.

## Foundation Research Basis

These references inform design, not certification or proof of correctness. Accessed
2026-09-26; paid books/standards were not read in full.

| Reference / reading scope | Applied decision |
| --- | --- |
| [ISO/IEC 11179-3:2023](https://www.iso.org/standard/78915.html), official abstract | Separate registry identities, labels and mappings |
| [PostgreSQL 17 identifiers](https://www.postgresql.org/docs/17/sql-syntax-lexical.html), identifier section | Preserve quoted/catalog names and explicit origin |
| [PostgreSQL 17 numeric types](https://www.postgresql.org/docs/17/datatype-numeric.html), numeric section | Preserve negative scale and precision independently |
| [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785.html), canonicalization section | Separate raw bytes from domain normalization; do not claim JCS compliance |
| [JSON Schema 2020-12](https://json-schema.org/draft/2020-12/json-schema-core), vocabulary/validation overview | Publish structural schema; retain semantic validation |
| [Unicode normalization](https://www.unicode.org/reports/tr15/), normalization/stability sections | Explicit, version-pinned logical normalization only |
| [Cupid, Madhavan et al.](https://www.microsoft.com/en-us/research/publication/generic-schema-matching-with-cupid/), research abstract | Do not equate name similarity with exact conformance |
| [Barr et al., Oracle Problem](https://discovery.ucl.ac.uk/id/eprint/1471263/), abstract | Fixed independent byte vector and metamorphic tests |
| [Specification by Example, Adzic](https://www.manning.com/books/specification-by-example), publisher overview | Concrete normal/invalid acceptance examples |
| [Database System Concepts, 7th ed.](https://db-book.com/online-chapters-dir/index.html), author chapter index | Keep relational structure distinct from future semantic design auditing; no claim of full book review |

## Verification Scope

The portable suite is `tests/schema_audit/test_contract.py`. It exercises malformed
input, immutable graphs, unknowns, normalization and deterministic digest vectors.
Package resource checks run against wheel/sdist artifacts; existing validation and
public-surface regressions remain separate. Actual eXERD import, live database
comparison and Console end-to-end validation are later-phase gates.
