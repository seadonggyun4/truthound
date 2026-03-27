"""Tests for the Schema system."""

import tempfile
from pathlib import Path

import polars as pl

import truthound as th
from truthound.schema import ColumnSchema, Schema
from truthound.validators.schema_validator import SchemaValidator


class TestLearn:
    """Tests for th.learn()."""

    def test_learn_basic(self):
        """Test basic schema learning."""
        data = {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "score": [85.5, 92.0, 78.5, 88.0, 95.5],
        }
        schema = th.learn(data)

        assert len(schema.columns) == 3
        assert schema.row_count == 5
        assert "id" in schema
        assert "name" in schema
        assert "score" in schema

    def test_learn_infers_constraints(self):
        """Test that learn infers min/max constraints."""
        data = {"value": [10, 20, 30, 40, 50]}
        schema = th.learn(data)

        col = schema["value"]
        assert col.min_value == 10.0
        assert col.max_value == 50.0
        assert col.mean == 30.0

    def test_learn_detects_nullability(self):
        """Test that learn detects nullable columns."""
        data = {
            "nullable": [1, None, 3],
            "non_nullable": [1, 2, 3],
        }
        schema = th.learn(data)

        assert schema["nullable"].nullable is True
        assert schema["non_nullable"].nullable is False

    def test_learn_detects_unique(self):
        """Test that learn detects unique columns."""
        data = {
            "unique_col": [1, 2, 3, 4, 5],
            "non_unique": [1, 1, 2, 2, 3],
        }
        schema = th.learn(data)

        assert schema["unique_col"].unique is True
        assert schema["non_unique"].unique is False

    def test_learn_categorical_values(self):
        """Test that learn captures allowed values for low cardinality columns."""
        data = {"status": ["active", "inactive", "pending", "active", "inactive"]}
        schema = th.learn(data, categorical_threshold=10)

        col = schema["status"]
        assert col.allowed_values is not None
        assert set(col.allowed_values) == {"active", "inactive", "pending"}

    def test_learn_string_lengths(self):
        """Test that learn captures string length constraints."""
        data = {"code": ["ABC", "DEFGH", "IJ"]}
        schema = th.learn(data)

        col = schema["code"]
        assert col.min_length == 2
        assert col.max_length == 5


class TestSchemaSaveLoad:
    """Tests for Schema save/load."""

    def test_save_and_load(self):
        """Test saving and loading schema to YAML."""
        data = {
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "value": [1.5, 2.5, 3.5],
        }
        original = th.learn(data)

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            original.save(f.name)
            loaded = Schema.load(f.name)

            assert len(loaded.columns) == len(original.columns)
            assert loaded.row_count == original.row_count

            for col_name in original.columns:
                orig_col = original[col_name]
                load_col = loaded[col_name]
                assert orig_col.dtype == load_col.dtype
                assert orig_col.nullable == load_col.nullable

            Path(f.name).unlink()


class TestSchemaValidator:
    """Tests for SchemaValidator."""

    def test_validates_against_schema(self):
        """Test validation against a learned schema."""
        # Learn from baseline
        baseline = {"id": [1, 2, 3], "value": [10, 20, 30]}
        schema = th.learn(baseline)

        # New data with issues
        new_data = {"id": [1, 2, 3], "value": [10, 20, 100]}  # 100 is above max

        validator = SchemaValidator(schema)
        lf = pl.DataFrame(new_data).lazy()
        issues = validator.validate(lf)

        assert any(i.issue_type == "above_maximum" for i in issues)

    def test_detects_missing_column(self):
        """Test detection of missing columns."""
        schema = Schema(
            columns={
                "id": ColumnSchema(name="id", dtype="Int64"),
                "name": ColumnSchema(name="name", dtype="String"),
            }
        )

        data = {"id": [1, 2, 3]}  # Missing "name" column
        validator = SchemaValidator(schema)
        issues = validator.validate(pl.DataFrame(data).lazy())

        assert any(i.issue_type == "missing_column" and i.column == "name" for i in issues)

    def test_detects_type_mismatch(self):
        """Test detection of type mismatches."""
        schema = Schema(
            columns={"value": ColumnSchema(name="value", dtype="Int64")}
        )

        data = {"value": ["a", "b", "c"]}  # String instead of Int64
        validator = SchemaValidator(schema)
        issues = validator.validate(pl.DataFrame(data).lazy())

        assert any(i.issue_type == "type_mismatch" for i in issues)

    def test_detects_unexpected_nulls(self):
        """Test detection of nulls in non-nullable column."""
        schema = Schema(
            columns={"value": ColumnSchema(name="value", dtype="Int64", nullable=False)}
        )

        data = {"value": [1, None, 3]}
        validator = SchemaValidator(schema)
        issues = validator.validate(pl.DataFrame(data).lazy())

        assert any(i.issue_type == "unexpected_nulls" for i in issues)

    def test_detects_invalid_values(self):
        """Test detection of values not in allowed set."""
        schema = Schema(
            columns={
                "status": ColumnSchema(
                    name="status",
                    dtype="String",
                    allowed_values=["active", "inactive"],
                )
            }
        )

        data = {"status": ["active", "unknown", "inactive"]}
        validator = SchemaValidator(schema)
        issues = validator.validate(pl.DataFrame(data).lazy())

        assert any(i.issue_type == "invalid_value" for i in issues)

    def test_detects_duplicate_and_string_length_issues(self):
        """Test aggregate-path duplicate and string length validation."""
        schema = Schema(
            columns={
                "id": ColumnSchema(name="id", dtype="Int64", unique=True),
                "code": ColumnSchema(name="code", dtype="String", min_length=2, max_length=4),
            }
        )

        data = {
            "id": [1, 1, 2],
            "code": ["A", "ABCD", "ABCDE"],
        }
        validator = SchemaValidator(schema)
        issues = validator.validate(pl.DataFrame(data).lazy())

        issue_counts = {issue.issue_type: issue.count for issue in issues}
        assert issue_counts["duplicate_values"] == 1
        assert issue_counts["string_too_short"] == 1
        assert issue_counts["string_too_long"] == 1

    def test_detects_pattern_mismatch(self):
        """Test targeted regex validation without materializing the whole frame."""
        schema = Schema(
            columns={
                "email": ColumnSchema(
                    name="email",
                    dtype="String",
                    pattern=r"^[\w.+-]+@[\w-]+\.[\w.-]+$",
                )
            }
        )

        data = {"email": ["a@example.com", "not-an-email", None]}
        validator = SchemaValidator(schema)
        issues = validator.validate(pl.DataFrame(data).lazy())

        pattern_issue = next(issue for issue in issues if issue.issue_type == "pattern_mismatch")
        assert pattern_issue.count == 1

    def test_only_targeted_checks_materialize_non_null_columns(self, monkeypatch):
        """Test that targeted value checks only materialize the columns that need them."""
        schema = Schema(
            columns={
                "id": ColumnSchema(name="id", dtype="Int64", unique=True),
                "status": ColumnSchema(
                    name="status",
                    dtype="String",
                    allowed_values=["active", "inactive"],
                ),
                "email": ColumnSchema(
                    name="email",
                    dtype="String",
                    pattern=r"^[\w.+-]+@[\w-]+\.[\w.-]+$",
                ),
            }
        )
        validator = SchemaValidator(schema)
        calls: list[str] = []
        original = validator._collect_non_null_column

        def tracked_collect(lf: pl.LazyFrame, col_name: str) -> pl.Series:
            calls.append(col_name)
            return original(lf, col_name)

        monkeypatch.setattr(validator, "_collect_non_null_column", tracked_collect)

        validator.validate(
            pl.DataFrame(
                {
                    "id": [1, 1, 2],
                    "status": ["active", "unknown", "inactive"],
                    "email": ["a@example.com", "broken", None],
                }
            ).lazy()
        )

        assert calls == ["status", "email"]


class TestCheckWithSchema:
    """Tests for th.check() with schema parameter."""

    def test_check_with_schema_object(self):
        """Test check with Schema object."""
        baseline = {"id": [1, 2, 3], "value": [10, 20, 30]}
        schema = th.learn(baseline)

        new_data = {"id": [1, 2, 3], "value": [10, 20, 100]}
        report = th.check(new_data, schema=schema, validators=[])

        assert report.has_issues

    def test_check_with_schema_file(self):
        """Test check with schema file path."""
        baseline = {"id": [1, 2, 3], "value": [10, 20, 30]}
        schema = th.learn(baseline)

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            schema.save(f.name)

            new_data = {"id": [1, 2, 3], "value": [10, 20, 100]}
            report = th.check(new_data, schema=f.name, validators=[])

            assert report.has_issues

            Path(f.name).unlink()

    def test_check_combines_schema_and_validators(self):
        """Test that schema and other validators run together."""
        baseline = {"value": [10, 20, 30]}
        schema = th.learn(baseline)

        new_data = {"value": [10, None, 100]}  # null + above max
        report = th.check(new_data, schema=schema, validators=["null"])

        # Should have both schema issues and null issues
        issue_types = {i.issue_type for i in report.issues}
        assert "null" in issue_types
        assert "above_maximum" in issue_types
