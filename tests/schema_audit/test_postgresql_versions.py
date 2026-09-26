"""Version-specific catalog semantics must never be inferred from PG17 columns."""

import unittest
from dataclasses import replace

from truthound.schema_audit import Dialect, State, audit_implementation, known
from truthound.schema_audit.normalization import normalize_type
from truthound.schema_audit.postgresql import CollectionError, _snapshot, constraint_query

try:
    from .test_implementation import snapshot
    from .test_standards import fixture
except ImportError:
    from test_implementation import snapshot
    from test_standards import fixture


class PostgreSQLVersions(unittest.TestCase):
    def test_queries_are_closed_and_versioned(self):
        legacy = constraint_query(14)
        self.assertNotIn("k.confdelsetcols", legacy)
        self.assertNotIn("i.indnullsnotdistinct", legacy)
        self.assertIn("NULL::smallint[] AS setcols", legacy)
        self.assertIn("k.confdelsetcols", constraint_query(17))
        for major in (13, 15, 16, 18):
            with self.assertRaisesRegex(CollectionError, "unsupported_server_version"):
                constraint_query(major)

    def test_snapshot_keeps_actual_dialect(self):
        for major in (14, 17):
            result = _snapshot([], [], [], ("public",), "test", "snapshot", False, major=major)
            self.assertEqual(result.dialect.major, major)
            self.assertEqual(result.adapter_version, f"pg{major}-catalog-v1")

    def test_exact_same_version_required(self):
        _, design = fixture()
        design = replace(design, dialect=Dialect(name="postgresql", major=14))
        catalog = snapshot(design)
        report = audit_implementation(design, catalog)
        check = next(c for c in report.checks if c.rule_id == "dialect")
        self.assertEqual(check.verdict.value, "PASS")
        report = audit_implementation(
            design, replace(catalog, dialect=Dialect(name="postgresql", major=17))
        )
        check = next(c for c in report.checks if c.rule_id == "dialect")
        self.assertEqual(check.verdict.value, "UNKNOWN")

    def test_pg14_rejects_new_numeric_scale_semantics(self):
        _, design = fixture()
        typ = replace(
            design.tables[0].columns[0].type,
            native_type="numeric",
            precision=known(3),
            scale=known(-1),
        )
        self.assertEqual(
            normalize_type(typ, Dialect(name="postgresql", major=14)).state, State.UNKNOWN
        )
        self.assertEqual(
            normalize_type(typ, Dialect(name="postgresql", major=17)), known("numeric")
        )


if __name__ == "__main__":
    unittest.main()
