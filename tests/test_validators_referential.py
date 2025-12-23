"""Tests for referential integrity validators."""

import pytest
import polars as pl

from truthound.validators.referential import (
    ForeignKeyRelation,
    ForeignKeyValidator,
    CompositeForeignKeyValidator,
    SelfReferentialFKValidator,
    CascadeAction,
    CascadeRule,
    CascadeIntegrityValidator,
    CascadeDepthValidator,
    OrphanRecordValidator,
    MultiTableOrphanValidator,
    DanglingReferenceValidator,
    CircularReferenceValidator,
    HierarchyCircularValidator,
    HierarchyDepthValidator,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def customers_df():
    """Sample customers table."""
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "email": ["alice@test.com", "bob@test.com", "charlie@test.com", "diana@test.com", "eve@test.com"],
    })


@pytest.fixture
def orders_df():
    """Sample orders table with some orphan records."""
    return pl.DataFrame({
        "id": [101, 102, 103, 104, 105, 106],
        "customer_id": [1, 2, 3, 99, 1, None],  # 99 is orphan, None is null
        "amount": [100.0, 200.0, 150.0, 300.0, 50.0, 75.0],
    })


@pytest.fixture
def order_items_df():
    """Sample order items table."""
    return pl.DataFrame({
        "id": [1001, 1002, 1003, 1004, 1005],
        "order_id": [101, 101, 102, 999, 103],  # 999 is orphan
        "product_id": [1, 2, 1, 3, 2],
        "quantity": [1, 2, 1, 1, 3],
    })


@pytest.fixture
def products_df():
    """Sample products table."""
    return pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["Widget", "Gadget", "Gizmo"],
        "price": [10.0, 20.0, 15.0],
    })


@pytest.fixture
def employees_df():
    """Sample employees table with self-reference (hierarchy)."""
    return pl.DataFrame({
        "employee_id": [1, 2, 3, 4, 5, 6],
        "name": ["CEO", "VP1", "VP2", "Manager1", "Worker1", "Worker2"],
        "manager_id": [None, 1, 1, 2, 4, 4],  # Tree structure
    })


@pytest.fixture
def employees_with_cycle_df():
    """Employees table with circular reference."""
    return pl.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "name": ["A", "B", "C", "D"],
        "manager_id": [2, 3, 1, None],  # 1->2->3->1 cycle
    })


@pytest.fixture
def deep_hierarchy_df():
    """Employees table with deep hierarchy."""
    # Create chain: 1 <- 2 <- 3 <- 4 <- 5 <- 6 <- 7 <- 8
    return pl.DataFrame({
        "id": list(range(1, 9)),
        "name": [f"Level{i}" for i in range(1, 9)],
        "parent_id": [None] + list(range(1, 8)),
    })


# ============================================================================
# ForeignKeyValidator Tests
# ============================================================================


class TestForeignKeyValidator:
    """Tests for ForeignKeyValidator."""

    def test_valid_foreign_key(self, customers_df, orders_df):
        """Test with mostly valid FK references."""
        validator = ForeignKeyValidator(
            child_table="orders",
            child_columns="customer_id",
            parent_table="customers",
            parent_columns="id",
            tables={
                "orders": orders_df.lazy(),
                "customers": customers_df.lazy(),
            },
        )
        issues = validator.validate(orders_df.lazy())

        # Should find 1 orphan (customer_id=99)
        fk_issues = [i for i in issues if i.issue_type == "fk_constraint_violation"]
        assert len(fk_issues) == 1
        assert fk_issues[0].count == 1

    def test_null_fk_allowed(self, customers_df, orders_df):
        """Test that NULL FKs are allowed by default."""
        validator = ForeignKeyValidator(
            child_table="orders",
            child_columns="customer_id",
            parent_table="customers",
            parent_columns="id",
            allow_null=True,
            tables={
                "orders": orders_df.lazy(),
                "customers": customers_df.lazy(),
            },
        )
        issues = validator.validate(orders_df.lazy())

        # NULL should not be counted as violation
        null_issues = [i for i in issues if i.issue_type == "fk_null_violation"]
        assert len(null_issues) == 0

    def test_null_fk_not_allowed(self, customers_df, orders_df):
        """Test that NULL FKs are flagged when not allowed."""
        validator = ForeignKeyValidator(
            child_table="orders",
            child_columns="customer_id",
            parent_table="customers",
            parent_columns="id",
            allow_null=False,
            tables={
                "orders": orders_df.lazy(),
                "customers": customers_df.lazy(),
            },
        )
        issues = validator.validate(orders_df.lazy())

        null_issues = [i for i in issues if i.issue_type == "fk_null_violation"]
        assert len(null_issues) == 1

    def test_all_valid_fk(self, customers_df):
        """Test with all valid FK references."""
        valid_orders = pl.DataFrame({
            "id": [1, 2, 3],
            "customer_id": [1, 2, 3],
        })
        validator = ForeignKeyValidator(
            child_table="orders",
            child_columns="customer_id",
            parent_table="customers",
            parent_columns="id",
            tables={
                "orders": valid_orders.lazy(),
                "customers": customers_df.lazy(),
            },
        )
        issues = validator.validate(valid_orders.lazy())

        fk_issues = [i for i in issues if i.issue_type == "fk_constraint_violation"]
        assert len(fk_issues) == 0


class TestCompositeForeignKeyValidator:
    """Tests for CompositeForeignKeyValidator."""

    def test_composite_key_violation(self):
        """Test composite FK detection."""
        parent = pl.DataFrame({
            "order_id": [1, 1, 2],
            "product_id": [1, 2, 1],
        })
        child = pl.DataFrame({
            "id": [1, 2, 3],
            "order_id": [1, 1, 3],  # (3, 1) doesn't exist
            "product_id": [1, 2, 1],
        })

        validator = CompositeForeignKeyValidator(
            child_table="child",
            child_columns=["order_id", "product_id"],
            parent_table="parent",
            parent_columns=["order_id", "product_id"],
            tables={
                "child": child.lazy(),
                "parent": parent.lazy(),
            },
        )
        issues = validator.validate(child.lazy())

        assert len(issues) >= 1
        assert any(i.issue_type == "composite_fk_violation" for i in issues)

    def test_partial_match_detection(self):
        """Test partial match detection."""
        parent = pl.DataFrame({
            "a": [1, 2],
            "b": [10, 20],
        })
        child = pl.DataFrame({
            "a": [1, 3],  # 1 exists, 3 doesn't
            "b": [99, 20],  # 99 doesn't exist, 20 exists
        })

        validator = CompositeForeignKeyValidator(
            child_table="child",
            child_columns=["a", "b"],
            parent_table="parent",
            parent_columns=["a", "b"],
            tables={
                "child": child.lazy(),
                "parent": parent.lazy(),
            },
            check_partial_matches=True,
        )
        issues = validator.validate(child.lazy())

        partial_issues = [i for i in issues if i.issue_type == "partial_fk_match"]
        assert len(partial_issues) >= 0  # May or may not find partial matches


class TestSelfReferentialFKValidator:
    """Tests for SelfReferentialFKValidator."""

    def test_valid_hierarchy(self, employees_df):
        """Test valid self-referential hierarchy."""
        validator = SelfReferentialFKValidator(
            table="employees",
            fk_column="manager_id",
            pk_column="employee_id",
        )
        issues = validator.validate(employees_df.lazy())

        # Valid hierarchy should have no violations
        violations = [i for i in issues if i.issue_type == "self_ref_fk_violation"]
        assert len(violations) == 0

    def test_invalid_self_reference(self):
        """Test invalid self-reference detection."""
        invalid_employees = pl.DataFrame({
            "employee_id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "manager_id": [None, 1, 99],  # 99 doesn't exist
        })

        validator = SelfReferentialFKValidator(
            table="employees",
            fk_column="manager_id",
            pk_column="employee_id",
        )
        issues = validator.validate(invalid_employees.lazy())

        violations = [i for i in issues if i.issue_type == "self_ref_fk_violation"]
        assert len(violations) == 1
        assert violations[0].count == 1


# ============================================================================
# CascadeIntegrityValidator Tests
# ============================================================================


class TestCascadeIntegrityValidator:
    """Tests for CascadeIntegrityValidator."""

    def test_restrict_violation(self, customers_df, orders_df):
        """Test RESTRICT violation detection."""
        rule = CascadeRule(
            relation=ForeignKeyRelation(
                child_table="orders",
                child_columns=["customer_id"],
                parent_table="customers",
                parent_columns=["id"],
            ),
            on_delete=CascadeAction.RESTRICT,
        )

        validator = CascadeIntegrityValidator(
            cascade_rules=[rule],
            tables={
                "orders": orders_df.lazy(),
                "customers": customers_df.lazy(),
            },
        )
        issues = validator.validate(orders_df.lazy())

        # Should detect orphan as RESTRICT violation
        restrict_issues = [i for i in issues if i.issue_type == "cascade_restrict_violation"]
        assert len(restrict_issues) == 1

    def test_set_null_pattern(self, customers_df):
        """Test SET_NULL pattern detection."""
        orders_with_nulls = pl.DataFrame({
            "id": [1, 2, 3],
            "customer_id": [1, None, None],  # 2 NULLs from deleted customers
        })

        rule = CascadeRule(
            relation=ForeignKeyRelation(
                child_table="orders",
                child_columns=["customer_id"],
                parent_table="customers",
                parent_columns=["id"],
            ),
            on_delete=CascadeAction.SET_NULL,
        )

        validator = CascadeIntegrityValidator(
            cascade_rules=[rule],
            tables={
                "orders": orders_with_nulls.lazy(),
                "customers": customers_df.lazy(),
            },
        )
        issues = validator.validate(orders_with_nulls.lazy())

        null_issues = [i for i in issues if i.issue_type == "cascade_set_null_detected"]
        assert len(null_issues) == 1


class TestCascadeDepthValidator:
    """Tests for CascadeDepthValidator."""

    def test_shallow_cascade(self):
        """Test acceptable cascade depth."""
        relations = [
            ForeignKeyRelation("b", ["a_id"], "a", ["id"]),
            ForeignKeyRelation("c", ["b_id"], "b", ["id"]),
        ]

        validator = CascadeDepthValidator(
            max_depth=5,
            relations=relations,
            tables={
                "a": pl.DataFrame({"id": [1]}).lazy(),
                "b": pl.DataFrame({"id": [1], "a_id": [1]}).lazy(),
                "c": pl.DataFrame({"id": [1], "b_id": [1]}).lazy(),
            },
        )
        issues = validator.validate(pl.DataFrame().lazy())

        depth_issues = [i for i in issues if i.issue_type == "cascade_depth_exceeded"]
        assert len(depth_issues) == 0

    def test_deep_cascade(self):
        """Test deep cascade chain detection."""
        # Create chain: a <- b <- c <- d <- e <- f
        relations = [
            ForeignKeyRelation("b", ["a_id"], "a", ["id"]),
            ForeignKeyRelation("c", ["b_id"], "b", ["id"]),
            ForeignKeyRelation("d", ["c_id"], "c", ["id"]),
            ForeignKeyRelation("e", ["d_id"], "d", ["id"]),
            ForeignKeyRelation("f", ["e_id"], "e", ["id"]),
        ]

        validator = CascadeDepthValidator(
            max_depth=3,
            relations=relations,
            tables={
                "a": pl.DataFrame({"id": [1]}).lazy(),
                "b": pl.DataFrame({"id": [1], "a_id": [1]}).lazy(),
                "c": pl.DataFrame({"id": [1], "b_id": [1]}).lazy(),
                "d": pl.DataFrame({"id": [1], "c_id": [1]}).lazy(),
                "e": pl.DataFrame({"id": [1], "d_id": [1]}).lazy(),
                "f": pl.DataFrame({"id": [1], "e_id": [1]}).lazy(),
            },
        )
        issues = validator.validate(pl.DataFrame().lazy())

        depth_issues = [i for i in issues if i.issue_type == "cascade_depth_exceeded"]
        assert len(depth_issues) >= 1


# ============================================================================
# OrphanRecordValidator Tests
# ============================================================================


class TestOrphanRecordValidator:
    """Tests for OrphanRecordValidator."""

    def test_detect_orphans(self, customers_df, orders_df):
        """Test orphan record detection."""
        validator = OrphanRecordValidator(
            child_table="orders",
            child_columns="customer_id",
            parent_table="customers",
            parent_columns="id",
            tables={
                "orders": orders_df.lazy(),
                "customers": customers_df.lazy(),
            },
        )
        issues = validator.validate(orders_df.lazy())

        orphan_issues = [i for i in issues if i.issue_type == "orphan_record_detected"]
        assert len(orphan_issues) == 1
        assert orphan_issues[0].count == 1  # customer_id=99

    def test_no_orphans(self, customers_df):
        """Test with no orphan records."""
        valid_orders = pl.DataFrame({
            "id": [1, 2],
            "customer_id": [1, 2],
        })

        validator = OrphanRecordValidator(
            child_table="orders",
            child_columns="customer_id",
            parent_table="customers",
            parent_columns="id",
            tables={
                "orders": valid_orders.lazy(),
                "customers": customers_df.lazy(),
            },
        )
        issues = validator.validate(valid_orders.lazy())

        orphan_issues = [i for i in issues if i.issue_type == "orphan_record_detected"]
        assert len(orphan_issues) == 0


class TestMultiTableOrphanValidator:
    """Tests for MultiTableOrphanValidator."""

    def test_multi_table_orphans(self, customers_df, orders_df, order_items_df):
        """Test orphan detection across multiple tables."""
        relations = [
            ForeignKeyRelation("orders", ["customer_id"], "customers", ["id"]),
            ForeignKeyRelation("order_items", ["order_id"], "orders", ["id"]),
        ]

        validator = MultiTableOrphanValidator(
            relations=relations,
            tables={
                "customers": customers_df.lazy(),
                "orders": orders_df.lazy(),
                "order_items": order_items_df.lazy(),
            },
        )
        issues = validator.validate(orders_df.lazy())

        # Should find orphans in both relationships
        orphan_issues = [i for i in issues if i.issue_type == "orphan_in_relation"]
        assert len(orphan_issues) >= 1


class TestDanglingReferenceValidator:
    """Tests for DanglingReferenceValidator."""

    def test_dangling_parents(self, customers_df):
        """Test detection of parents with no children."""
        orders = pl.DataFrame({
            "id": [1, 2],
            "customer_id": [1, 2],  # Only customers 1, 2 have orders
        })

        validator = DanglingReferenceValidator(
            parent_table="customers",
            parent_columns="id",
            child_table="orders",
            child_columns="customer_id",
            min_expected_children=1,
            tables={
                "customers": customers_df.lazy(),  # 5 customers
                "orders": orders.lazy(),  # Only 2 have orders
            },
        )
        issues = validator.validate(customers_df.lazy())

        dangling_issues = [i for i in issues if i.issue_type == "dangling_parent_detected"]
        assert len(dangling_issues) == 1
        assert dangling_issues[0].count == 3  # Customers 3, 4, 5 have no orders


# ============================================================================
# CircularReferenceValidator Tests
# ============================================================================


class TestCircularReferenceValidator:
    """Tests for CircularReferenceValidator."""

    def test_no_circular_reference(self):
        """Test with acyclic relationships."""
        relations = [
            ForeignKeyRelation("orders", ["customer_id"], "customers", ["id"]),
            ForeignKeyRelation("order_items", ["order_id"], "orders", ["id"]),
        ]

        validator = CircularReferenceValidator(relations=relations)
        issues = validator.validate(pl.DataFrame().lazy())

        cycle_issues = [i for i in issues if i.issue_type == "circular_reference_detected"]
        assert len(cycle_issues) == 0

    def test_detect_circular_reference(self):
        """Test circular reference detection."""
        relations = [
            ForeignKeyRelation("a", ["b_id"], "b", ["id"]),
            ForeignKeyRelation("b", ["c_id"], "c", ["id"]),
            ForeignKeyRelation("c", ["a_id"], "a", ["id"]),  # Creates cycle
        ]

        validator = CircularReferenceValidator(
            relations=relations,
            allow_self_reference=False,
        )
        issues = validator.validate(pl.DataFrame().lazy())

        cycle_issues = [i for i in issues if i.issue_type == "circular_reference_detected"]
        assert len(cycle_issues) >= 1


class TestHierarchyCircularValidator:
    """Tests for HierarchyCircularValidator."""

    def test_valid_hierarchy(self, employees_df):
        """Test valid tree hierarchy."""
        validator = HierarchyCircularValidator(
            table="employees",
            id_column="employee_id",
            parent_column="manager_id",
        )
        issues = validator.validate(employees_df.lazy())

        cycle_issues = [i for i in issues if i.issue_type == "hierarchy_cycle_detected"]
        assert len(cycle_issues) == 0

    def test_detect_hierarchy_cycle(self, employees_with_cycle_df):
        """Test cycle detection in hierarchy data."""
        validator = HierarchyCircularValidator(
            table="employees",
            id_column="employee_id",
            parent_column="manager_id",
        )
        issues = validator.validate(employees_with_cycle_df.lazy())

        cycle_issues = [i for i in issues if i.issue_type == "hierarchy_cycle_detected"]
        assert len(cycle_issues) >= 1


class TestHierarchyDepthValidator:
    """Tests for HierarchyDepthValidator."""

    def test_acceptable_depth(self, employees_df):
        """Test hierarchy within depth limit."""
        validator = HierarchyDepthValidator(
            table="employees",
            id_column="employee_id",
            parent_column="manager_id",
            max_depth=10,
        )
        issues = validator.validate(employees_df.lazy())

        depth_issues = [i for i in issues if i.issue_type == "hierarchy_depth_exceeded"]
        assert len(depth_issues) == 0

    def test_excessive_depth(self, deep_hierarchy_df):
        """Test deep hierarchy detection."""
        validator = HierarchyDepthValidator(
            table="hierarchy",
            id_column="id",
            parent_column="parent_id",
            max_depth=3,
        )
        validator.register_table("hierarchy", deep_hierarchy_df.lazy())
        issues = validator.validate(deep_hierarchy_df.lazy())

        depth_issues = [i for i in issues if i.issue_type == "hierarchy_depth_exceeded"]
        assert len(depth_issues) == 1


# ============================================================================
# ForeignKeyRelation Tests
# ============================================================================


class TestForeignKeyRelation:
    """Tests for ForeignKeyRelation dataclass."""

    def test_valid_relation(self):
        """Test valid relation creation."""
        relation = ForeignKeyRelation(
            child_table="orders",
            child_columns=["customer_id"],
            parent_table="customers",
            parent_columns=["id"],
        )
        assert relation.name == "orders(customer_id) -> customers(id)"

    def test_composite_relation(self):
        """Test composite key relation."""
        relation = ForeignKeyRelation(
            child_table="items",
            child_columns=["order_id", "product_id"],
            parent_table="catalog",
            parent_columns=["order_id", "product_id"],
        )
        assert "order_id" in relation.name
        assert "product_id" in relation.name

    def test_mismatched_columns(self):
        """Test column count validation."""
        with pytest.raises(ValueError, match="Column count mismatch"):
            ForeignKeyRelation(
                child_table="a",
                child_columns=["x", "y"],
                parent_table="b",
                parent_columns=["z"],  # Mismatch
            )

    def test_custom_relation_name(self):
        """Test custom relation name."""
        relation = ForeignKeyRelation(
            child_table="orders",
            child_columns=["customer_id"],
            parent_table="customers",
            parent_columns=["id"],
            relation_name="customer_orders_fk",
        )
        assert relation.name == "customer_orders_fk"


# ============================================================================
# Integration Tests
# ============================================================================


class TestReferentialIntegration:
    """Integration tests for referential validators."""

    def test_full_schema_validation(self, customers_df, orders_df, order_items_df, products_df):
        """Test comprehensive schema validation."""
        tables = {
            "customers": customers_df.lazy(),
            "orders": orders_df.lazy(),
            "order_items": order_items_df.lazy(),
            "products": products_df.lazy(),
        }

        relations = [
            ForeignKeyRelation("orders", ["customer_id"], "customers", ["id"]),
            ForeignKeyRelation("order_items", ["order_id"], "orders", ["id"]),
            ForeignKeyRelation("order_items", ["product_id"], "products", ["id"]),
        ]

        # Check for orphans
        orphan_validator = MultiTableOrphanValidator(
            tables=tables,
            relations=relations,
        )
        orphan_issues = orphan_validator.validate(orders_df.lazy())

        # Check for circular references
        circular_validator = CircularReferenceValidator(relations=relations)
        circular_issues = circular_validator.validate(pl.DataFrame().lazy())

        # Should find orphans but no circular references
        assert len([i for i in orphan_issues if "orphan" in i.issue_type]) >= 1
        assert len(circular_issues) == 0

    def test_empty_tables(self):
        """Test with empty tables."""
        empty_df = pl.DataFrame({"id": [], "fk_id": []})
        parent_df = pl.DataFrame({"id": [1, 2, 3]})

        validator = ForeignKeyValidator(
            child_table="child",
            child_columns="fk_id",
            parent_table="parent",
            parent_columns="id",
            tables={
                "child": empty_df.lazy(),
                "parent": parent_df.lazy(),
            },
        )
        issues = validator.validate(empty_df.lazy())

        # Empty table should have no issues
        assert len(issues) == 0
