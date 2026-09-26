"""Real execution at the bounded limit, not merely model construction."""

import unittest
from dataclasses import replace

from truthound.schema_audit import ModelError, audit_implementation, audit_standards
from truthound.schema_audit._contract import MAX_ITEMS, convert
from truthound.schema_audit.limits import MAX_AUDIT_CHECKS, MAX_AUDIT_OBJECTS

try:
    from .test_implementation import snapshot
    from .test_standards import fixture
except ImportError:
    from test_implementation import snapshot
    from test_standards import fixture


class Capacity(unittest.TestCase):
    def test_report_budget_does_not_relax_input_arrays(self):
        standard, design = fixture()
        for report in (
            audit_standards(standard, design),
            audit_implementation(design, snapshot(design)),
        ):
            with self.assertRaisesRegex(ModelError, "item_limit"):
                replace(report, checks=(report.checks[0],) * (MAX_AUDIT_CHECKS + 1))
        with self.assertRaisesRegex(ModelError, "item_limit"):
            convert((1,) * (MAX_ITEMS + 1), tuple[int, ...], "/input", wire=False)

    def test_ten_thousand_objects_and_fail_closed_overflow(self):
        standard, design = fixture()
        t = design.tables[0]
        tables = tuple(
            replace(
                t,
                id=f"t{i}",
                physical_name=replace(t.physical_name, text=f"cust{i}"),
                columns=tuple(
                    replace(
                        t.columns[0],
                        id=f"c{i}_{j}",
                        ordinal=j + 1,
                        physical_name=replace(t.columns[0].physical_name, text=f"cust_no{j}"),
                    )
                    for j in range(99)
                ),
            )
            for i in range(100)
        )
        design = replace(design, tables=tables)
        self.assertEqual(audit_standards(standard, design).target_count, MAX_AUDIT_OBJECTS)
        self.assertEqual(audit_implementation(design, snapshot(design)).verdict, "CONFORMANT")
        extra = replace(tables[0], id="extra", columns=())
        oversized = replace(design, tables=tables + (extra,))
        with self.assertRaisesRegex(ModelError, "audit_object_limit"):
            audit_standards(standard, oversized)
        with self.assertRaisesRegex(ModelError, "audit_object_limit"):
            audit_implementation(oversized, snapshot(oversized))
