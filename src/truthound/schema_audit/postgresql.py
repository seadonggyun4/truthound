"""Bounded PostgreSQL14/17 catalog-only adapter for a dedicated asyncpg connection.

The caller owns connection authentication/TLS and destination authorization. No
user SQL, user rows, connection URLs or driver error text enter the snapshot.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime

from ._contract import VERSION, ModelError, opaque, require
from .limits import MAX_AUDIT_OBJECTS, MAX_CATALOG_TABLES
from .models import (
    CatalogSnapshot,
    Column,
    Constraint,
    ConstraintKind,
    Coverage,
    Dialect,
    Identifier,
    LogicalName,
    NameOrigin,
    Reason,
    SourceLocation,
    Table,
    TypeDescriptor,
    absent,
    known,
    unknown,
)

ADAPTER_VERSION = "pg17-catalog-v1"
SUPPORTED_MAJORS = (14, 17)
FACETS = ("TABLES", "COLUMNS", "CONSTRAINTS", "LOGICAL_NAMES", "TYPES", "DEFAULTS")
TABLES = f"""SELECT c.oid::bigint AS oid, n.nspname AS ns, c.relname AS name,
 c.relkind::text AS kind, c.relispartition AS partition,
 has_schema_privilege(n.oid, 'USAGE') AND has_table_privilege(c.oid, 'SELECT') AS allowed,
 left(obj_description(c.oid, 'pg_class'),4097) AS comment
 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
 WHERE n.nspname = ANY($1::text[]) AND c.relkind IN ('r','p','v','m','f')
 ORDER BY n.nspname,c.relname LIMIT {MAX_CATALOG_TABLES + 1}"""
COLUMNS = f"""SELECT a.attrelid::bigint AS rel, a.attnum AS num, a.attname AS name,
 a.attnotnull AS notnull, a.attndims AS dims, a.atttypmod AS mod,
 a.attidentity::text AS identity, a.attgenerated::text AS generated,
 t.typname AS type, tn.nspname AS type_ns, t.typtype::text AS type_kind,
 t.typelem::bigint AS elem, left(col_description(a.attrelid,a.attnum),4097) AS comment,
 left(pg_get_expr(d.adbin,d.adrelid,false),4097) AS default,
 CASE WHEN a.attcollation=0 OR a.attcollation=t.typcollation THEN NULL
 ELSE cn.nspname || '.' || co.collname END AS collation
 FROM pg_catalog.pg_attribute a JOIN pg_catalog.pg_type t ON t.oid=a.atttypid
 JOIN pg_catalog.pg_namespace tn ON tn.oid=t.typnamespace
 LEFT JOIN pg_catalog.pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum
 LEFT JOIN pg_catalog.pg_collation co ON co.oid=a.attcollation
 LEFT JOIN pg_catalog.pg_namespace cn ON cn.oid=co.collnamespace
 WHERE a.attrelid = ANY($1::oid[]) AND a.attnum>0 AND NOT a.attisdropped
 ORDER BY a.attrelid,a.attnum LIMIT {MAX_AUDIT_OBJECTS + 1}"""
KEYS = """SELECT k.oid::bigint AS oid,k.conrelid::bigint AS rel,k.conname AS name,
 k.contype::text AS kind,k.conkey AS cols,k.confrelid::bigint AS target,k.confkey AS target_cols,
 k.confupdtype::text AS update,k.confdeltype::text AS delete,k.condeferrable AS deferrable,
 k.condeferred AS deferred,k.convalidated AS validated,k.confmatchtype::text AS match,
 {setcols} AS setcols,k.coninhcount AS inherited,
 left(pg_get_expr(k.conbin,k.conrelid,false),4097) AS expression,
 {nulls_not_distinct} AS nulls_not_distinct
 FROM pg_catalog.pg_constraint k LEFT JOIN pg_catalog.pg_index i ON i.indexrelid=k.conindid
 WHERE k.conrelid = ANY($1::oid[]) ORDER BY k.conrelid,k.conname,k.oid LIMIT {limit}"""


def constraint_query(major: int) -> str:
    """PG14 predates partial SET NULL targets and NULLS NOT DISTINCT."""
    if major not in SUPPORTED_MAJORS:
        raise CollectionError("unsupported_server_version")
    return KEYS.format(
        limit=MAX_AUDIT_OBJECTS + 1,
        setcols="NULL::smallint[]" if major == 14 else "k.confdelsetcols",
        nulls_not_distinct="false" if major == 14 else "COALESCE(i.indnullsnotdistinct,false)",
    )


class CollectionError(ValueError):
    """Safe code only; intentionally discards driver diagnostics."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _identifier(text: str) -> Identifier:
    return Identifier(text=text, origin=NameOrigin.CATALOG)


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _logical(value) -> LogicalName:
    return LogicalName(name=known(value) if value else unknown(), source="COMMENT")


def _type(row) -> TypeDescriptor:
    name, mod = row["type"], row["mod"]
    facets = {
        k: absent()
        for k in (
            "length",
            "length_unit",
            "precision",
            "scale",
            "timezone",
            "domain_ref",
            "enum_values_ref",
            "collation",
        )
    }
    if name in ("varchar", "bpchar") and mod >= 4:
        facets.update(length=known(mod - 4), length_unit=known("CHARACTERS"))
    elif name == "numeric" and mod >= 4:
        raw = mod - 4
        scale = raw & 2047
        if scale >= 1024:
            scale -= 2048
        facets.update(precision=known((raw >> 16) & 65535), scale=known(scale))
    elif name in ("timestamp", "timestamptz", "time", "timetz"):
        # Model precision requires >=1; zero precision is explicitly unsupported.
        facets.update(
            timezone=known(name in ("timestamptz", "timetz")),
            precision=unknown(Reason.UNSUPPORTED) if mod == 0 else known(6 if mod < 0 else mod),
        )
    if row["type_kind"] == "d":
        facets["domain_ref"] = known(row["type_ns"] + "." + name)
    elif row["type_kind"] == "e":
        facets["enum_values_ref"] = unknown(Reason.UNSUPPORTED)
    if row["collation"]:
        facets["collation"] = known(row["collation"])
    return TypeDescriptor(
        native_type=name,
        type_schema=known(row["type_ns"]),
        array_rank=unknown(Reason.UNSUPPORTED) if row["elem"] else known(0),
        **facets,
    )


def _snapshot(rows, columns, keys, schemas, binding, identifier, denied, *, major=17):
    issues = {f: Reason.PERMISSION_DENIED for f in FACETS} if denied else {}
    allowed = {r["oid"] for r in rows if r["allowed"] and r["kind"] == "r" and not r["partition"]}
    if len(allowed) != len(rows):
        for facet in FACETS:
            issues.setdefault(facet, Reason.UNSUPPORTED)
    column_ids = {(r["rel"], r["num"]): f"c{r['rel']}_{r['num']}" for r in columns}
    converted_columns, converted_keys = {}, {}
    for r in columns:
        default = (
            unknown(Reason.UNSUPPORTED)
            if r["identity"] or r["generated"]
            else known(r["default"])
            if r["default"] is not None
            else absent()
        )
        col = Column(
            id=column_ids[r["rel"], r["num"]],
            physical_name=_identifier(r["name"]),
            logical_name=_logical(r["comment"]),
            type=_type(r),
            nullable=known(not r["notnull"]),
            default=default,
            ordinal=r["num"],
        )
        converted_columns.setdefault(r["rel"], []).append(col)
    kinds = {
        "p": ConstraintKind.PRIMARY_KEY,
        "u": ConstraintKind.UNIQUE,
        "f": ConstraintKind.FOREIGN_KEY,
        "c": ConstraintKind.CHECK,
    }
    actions = {
        "a": "NO_ACTION",
        "r": "RESTRICT",
        "c": "CASCADE",
        "n": "SET_NULL",
        "d": "SET_DEFAULT",
    }
    for r in keys:
        if (
            r["kind"] not in kinds
            or not r["validated"]
            or r["deferred"]
            or r["inherited"]
            or r["nulls_not_distinct"]
            or r["setcols"]
            or (r["kind"] == "f" and r["match"] != "s")
        ):
            issues["CONSTRAINTS"] = Reason.UNSUPPORTED
            continue
        if r["target"] and r["target"] not in allowed:
            issues["CONSTRAINTS"] = Reason.INCOMPLETE
            continue
        try:
            local = tuple(column_ids[r["rel"], i] for i in (r["cols"] or []))
            remote = tuple(column_ids[r["target"], i] for i in (r["target_cols"] or []))
        except KeyError:
            raise CollectionError("catalog_closure_incomplete") from None
        key = Constraint(
            id=f"k{r['oid']}",
            name=_identifier(r["name"]),
            kind=kinds[r["kind"]],
            column_ids=local,
            referenced_table_id=f"t{r['target']}" if r["target"] else None,
            referenced_column_ids=remote,
            on_update=actions[r["update"]] if r["target"] else None,
            on_delete=actions[r["delete"]] if r["target"] else None,
            deferrable=known(r["deferrable"]),
            expression=known(r["expression"])
            if r["kind"] == "c" and r["expression"]
            else unknown()
            if r["kind"] == "c"
            else absent(),
        )
        converted_keys.setdefault(r["rel"], []).append(key)
    tables = tuple(
        Table(
            id=f"t{r['oid']}",
            namespace=_identifier(r["ns"]),
            physical_name=_identifier(r["name"]),
            logical_name=_logical(r["comment"]),
            columns=tuple(converted_columns.get(r["oid"], [])),
            constraints=tuple(converted_keys.get(r["oid"], [])),
        )
        for r in rows
        if r["oid"] in allowed
    )
    require(
        sum(1 + len(t.columns) + len(t.constraints) for t in tables) <= MAX_AUDIT_OBJECTS,
        "audit_object_limit",
    )
    # The digest identifies the private catalog observation, never connection data.
    raw_digest = hashlib.sha256(
        json.dumps([rows, columns, keys], sort_keys=True).encode()
    ).hexdigest()
    sources = tuple(
        SourceLocation(object_id=o.id, input_digest=raw_digest, pointer=f"/tables/{i}" + suffix)
        for i, t in enumerate(tables)
        for o, suffix in [(t, "")]
        + [(c, f"/columns/{j}") for j, c in enumerate(t.columns)]
        + [(k, f"/constraints/{j}") for j, k in enumerate(t.constraints)]
    )
    return CatalogSnapshot(
        contract_version=VERSION,
        kind="CATALOG",
        id=identifier,
        revision=1,
        dialect=Dialect(name="postgresql", major=major),
        binding_ref=binding,
        adapter_version=ADAPTER_VERSION if major == 17 else "pg14-catalog-v1",
        captured_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        scope=tuple(_identifier(s) for s in schemas),
        coverage=tuple(
            Coverage(
                facet=f,
                state="PARTIAL" if f in issues else "COMPLETE",
                reason=issues.get(f, Reason.NONE),
            )
            for f in FACETS
        ),
        tables=tables,
        sources=sources,
    )


async def collect_postgresql(
    connection,
    *,
    schemas: tuple[str, ...],
    binding_ref: str,
    snapshot_id: str,
    timeout_ms: int = 5000,
    attempts: int = 2,
) -> CatalogSnapshot:
    """Use a dedicated connected asyncpg connection; always rollback read transactions.

    Supported objects are ordinary PG14/17 tables. Unsupported objects/constraint
    semantics lower coverage; cross-scope FKs never cause implicit scope expansion.
    Cancellation propagates and driver errors are translated to safe fixed codes.
    """
    opaque(binding_ref, "/binding_ref")
    opaque(snapshot_id, "/snapshot_id")
    require(
        type(schemas) is tuple and 1 <= len(schemas) <= 16 and len(set(schemas)) == len(schemas),
        "invalid_scope",
    )
    for schema in schemas:
        _identifier(schema)
        require(
            len(schema.encode()) <= 63
            and not schema.startswith("pg_")
            and schema != "information_schema",
            "invalid_scope",
        )
    require(
        type(timeout_ms) is int
        and 100 <= timeout_ms <= 10000
        and type(attempts) is int
        and 1 <= attempts <= 3,
        "invalid_collection_budget",
    )
    if connection.is_in_transaction():
        raise CollectionError("dedicated_connection_required")
    for attempt in range(attempts):
        try:
            async with asyncio.timeout(timeout_ms / 1000 * 4):
                async with connection.transaction(readonly=True):
                    await connection.execute(f"SET LOCAL statement_timeout = {timeout_ms}")
                    await connection.execute("SET LOCAL search_path = pg_catalog")
                    major = connection.get_server_version().major
                    if major not in SUPPORTED_MAJORS:
                        raise CollectionError("unsupported_server_version")
                    initial = [dict(r) for r in await connection.fetch(TABLES, list(schemas))]
                if len(initial) > MAX_CATALOG_TABLES:
                    raise CollectionError("catalog_object_limit")
                tx = connection.transaction(isolation="repeatable_read", readonly=True)
                await tx.start()
                try:
                    await connection.execute(f"SET LOCAL statement_timeout = {timeout_ms}")
                    await connection.execute(f"SET LOCAL lock_timeout = {timeout_ms}")
                    await connection.execute("SET LOCAL search_path = pg_catalog")
                    # Locks precede the first catalog SELECT. Never read application rows.
                    for row in initial:
                        if row["allowed"] and row["kind"] == "r" and not row["partition"]:
                            await connection.execute(
                                f"LOCK TABLE ONLY {_quote(row['ns'])}.{_quote(row['name'])} IN ACCESS SHARE MODE"
                            )
                    rows = [dict(r) for r in await connection.fetch(TABLES, list(schemas))]

                    def identity(rs):
                        return [(r["oid"], r["ns"], r["name"], r["kind"], r["allowed"]) for r in rs]

                    if identity(initial) != identity(rows):
                        raise CollectionError("catalog_changed")
                    ns = await connection.fetch(
                        "SELECT nspname, has_schema_privilege(oid,'USAGE') AS allowed FROM pg_catalog.pg_namespace WHERE nspname=ANY($1::text[])",
                        list(schemas),
                    )
                    denied = (
                        len(ns) != len(schemas)
                        or any(not r["allowed"] for r in ns)
                        or any(not r["allowed"] for r in rows)
                    )
                    ids = [
                        r["oid"]
                        for r in rows
                        if r["allowed"] and r["kind"] == "r" and not r["partition"]
                    ]
                    columns = [dict(r) for r in await connection.fetch(COLUMNS, ids)]
                    keys = [dict(r) for r in await connection.fetch(constraint_query(major), ids)]
                    if len(columns) + len(keys) + len(rows) > MAX_AUDIT_OBJECTS:
                        raise CollectionError("catalog_object_limit")
                    # Unmodelled unique/expression indexes or inheritance must not appear conformant.
                    unsupported = await connection.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM pg_catalog.pg_index i WHERE i.indrelid=ANY($1::oid[]) AND i.indisunique AND NOT EXISTS(SELECT 1 FROM pg_catalog.pg_constraint k WHERE k.conindid=i.indexrelid)) OR EXISTS(SELECT 1 FROM pg_catalog.pg_inherits WHERE inhrelid=ANY($1::oid[]) OR inhparent=ANY($1::oid[]))",
                        ids,
                    )
                    result = _snapshot(
                        rows, columns, keys, schemas, binding_ref, snapshot_id, denied, major=major
                    )
                    if unsupported:
                        result = replace(
                            result,
                            coverage=tuple(
                                replace(c, state="PARTIAL", reason=Reason.UNSUPPORTED)
                                if c.facet == "CONSTRAINTS"
                                else c
                                for c in result.coverage
                            ),
                        )
                    return result
                finally:
                    await tx.rollback()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            sqlstate = getattr(exc, "sqlstate", "")
            code = (
                exc.code
                if isinstance(exc, CollectionError)
                else "catalog_invalid_metadata"
                if isinstance(exc, ModelError)
                else "catalog_permission_denied"
                if sqlstate == "42501"
                else "catalog_timeout"
                if isinstance(exc, TimeoutError) or sqlstate in ("57014", "55P03")
                else "catalog_changed"
                if sqlstate in ("40001", "42P01", "42704")
                else "catalog_collection_failed"
            )
            if code not in ("catalog_changed", "catalog_timeout") or attempt + 1 == attempts:
                raise CollectionError(code) from None
            await asyncio.sleep(0.05 * (attempt + 1))
    raise CollectionError("catalog_collection_failed")
