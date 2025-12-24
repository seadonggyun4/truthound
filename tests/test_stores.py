"""Tests for stores module."""

from datetime import datetime, timedelta
from pathlib import Path
import tempfile

import pytest

from truthound.stores import (
    get_store,
    ValidationResult,
    ValidatorResult,
    ResultStatistics,
    ResultStatus,
    StoreNotFoundError,
    StoreQuery,
)
from truthound.stores.expectations import Expectation, ExpectationSuite
from truthound.stores.backends.memory import MemoryStore, MemoryExpectationStore
from truthound.stores.backends.filesystem import FileSystemStore, FileSystemExpectationStore


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_result() -> ValidationResult:
    """Create a sample validation result for testing."""
    return ValidationResult(
        run_id="test_run_001",
        run_time=datetime.now(),
        data_asset="test_data.csv",
        status=ResultStatus.FAILURE,
        results=[
            ValidatorResult(
                validator_name="null_check",
                success=False,
                column="email",
                issue_type="null_values",
                count=5,
                severity="high",
                message="Found 5 null values",
            ),
            ValidatorResult(
                validator_name="type_check",
                success=True,
                column="age",
            ),
        ],
        statistics=ResultStatistics(
            total_validators=2,
            passed_validators=1,
            failed_validators=1,
            total_rows=100,
            total_columns=5,
            total_issues=1,
            high_issues=1,
        ),
        tags={"env": "test", "version": "1.0"},
    )


@pytest.fixture
def sample_suite() -> ExpectationSuite:
    """Create a sample expectation suite for testing."""
    suite = ExpectationSuite(
        name="test_suite",
        data_asset="test_data.csv",
        tags={"env": "test"},
    )
    suite.add_expectation(Expectation(
        expectation_type="not_null",
        column="email",
        mostly=0.95,
    ))
    suite.add_expectation(Expectation(
        expectation_type="in_range",
        column="age",
        kwargs={"min_value": 0, "max_value": 150},
    ))
    return suite


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_create_from_dict(self, sample_result: ValidationResult) -> None:
        """Test creating result from dictionary."""
        data = sample_result.to_dict()
        restored = ValidationResult.from_dict(data)

        assert restored.run_id == sample_result.run_id
        assert restored.data_asset == sample_result.data_asset
        assert restored.status == sample_result.status
        assert len(restored.results) == len(sample_result.results)

    def test_success_property(self) -> None:
        """Test success property."""
        result = ValidationResult(
            run_id="test",
            run_time=datetime.now(),
            data_asset="test.csv",
            status=ResultStatus.SUCCESS,
        )
        assert result.success is True

        result.status = ResultStatus.FAILURE
        assert result.success is False

    def test_generate_run_id(self) -> None:
        """Test run ID generation."""
        id1 = ValidationResult._generate_run_id()
        id2 = ValidationResult._generate_run_id()

        assert id1 != id2
        assert id1.startswith("run_")
        assert len(id1) > 20

    def test_get_failed_columns(self, sample_result: ValidationResult) -> None:
        """Test getting failed columns."""
        failed_cols = sample_result.get_failed_columns()
        assert "email" in failed_cols
        assert "age" not in failed_cols


# =============================================================================
# MemoryStore Tests
# =============================================================================


class TestMemoryStore:
    """Tests for in-memory store."""

    def test_save_and_get(self, sample_result: ValidationResult) -> None:
        """Test saving and retrieving a result."""
        store = MemoryStore()

        run_id = store.save(sample_result)
        assert run_id == sample_result.run_id

        retrieved = store.get(run_id)
        assert retrieved.run_id == sample_result.run_id
        assert retrieved.data_asset == sample_result.data_asset

    def test_exists(self, sample_result: ValidationResult) -> None:
        """Test checking if result exists."""
        store = MemoryStore()

        assert store.exists(sample_result.run_id) is False

        store.save(sample_result)
        assert store.exists(sample_result.run_id) is True

    def test_delete(self, sample_result: ValidationResult) -> None:
        """Test deleting a result."""
        store = MemoryStore()
        store.save(sample_result)

        assert store.delete(sample_result.run_id) is True
        assert store.exists(sample_result.run_id) is False
        assert store.delete(sample_result.run_id) is False

    def test_list_ids(self, sample_result: ValidationResult) -> None:
        """Test listing result IDs."""
        store = MemoryStore()

        # Create multiple results
        for i in range(3):
            result = ValidationResult(
                run_id=f"run_{i}",
                run_time=datetime.now(),
                data_asset="test.csv",
                status=ResultStatus.SUCCESS,
            )
            store.save(result)

        ids = store.list_ids()
        assert len(ids) == 3

    def test_query_by_data_asset(self) -> None:
        """Test querying by data asset."""
        store = MemoryStore()

        # Save results for different data assets
        for asset in ["customers.csv", "orders.csv", "customers.csv"]:
            result = ValidationResult(
                run_id=ValidationResult._generate_run_id(),
                run_time=datetime.now(),
                data_asset=asset,
                status=ResultStatus.SUCCESS,
            )
            store.save(result)

        query = StoreQuery(data_asset="customers.csv")
        results = store.query(query)
        assert len(results) == 2

    def test_query_by_status(self) -> None:
        """Test querying by status."""
        store = MemoryStore()

        for status in [ResultStatus.SUCCESS, ResultStatus.FAILURE, ResultStatus.SUCCESS]:
            result = ValidationResult(
                run_id=ValidationResult._generate_run_id(),
                run_time=datetime.now(),
                data_asset="test.csv",
                status=status,
            )
            store.save(result)

        query = StoreQuery(status="success")
        results = store.query(query)
        assert len(results) == 2

    def test_max_items_limit(self) -> None:
        """Test max items limit."""
        store = MemoryStore(max_items=2)

        for i in range(5):
            result = ValidationResult(
                run_id=f"run_{i}",
                run_time=datetime.now() + timedelta(seconds=i),
                data_asset="test.csv",
                status=ResultStatus.SUCCESS,
            )
            store.save(result)

        # Should only have 2 items
        assert len(store.list_ids()) == 2

    def test_clear_all(self, sample_result: ValidationResult) -> None:
        """Test clearing all items."""
        store = MemoryStore()
        store.save(sample_result)

        count = store.clear_all()
        assert count == 1
        assert len(store.list_ids()) == 0


# =============================================================================
# FileSystemStore Tests
# =============================================================================


class TestFileSystemStore:
    """Tests for filesystem store."""

    def test_save_and_get(self, sample_result: ValidationResult) -> None:
        """Test saving and retrieving a result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSystemStore(base_path=tmpdir)

            run_id = store.save(sample_result)
            retrieved = store.get(run_id)

            assert retrieved.run_id == sample_result.run_id
            assert retrieved.data_asset == sample_result.data_asset

    def test_exists(self, sample_result: ValidationResult) -> None:
        """Test checking if result exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSystemStore(base_path=tmpdir)

            assert store.exists(sample_result.run_id) is False

            store.save(sample_result)
            assert store.exists(sample_result.run_id) is True

    def test_delete(self, sample_result: ValidationResult) -> None:
        """Test deleting a result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSystemStore(base_path=tmpdir)
            store.save(sample_result)

            assert store.delete(sample_result.run_id) is True
            assert store.exists(sample_result.run_id) is False

    def test_compression(self, sample_result: ValidationResult) -> None:
        """Test saving with compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSystemStore(base_path=tmpdir, compression=True)

            run_id = store.save(sample_result)
            retrieved = store.get(run_id)

            assert retrieved.run_id == sample_result.run_id

    def test_not_found_error(self) -> None:
        """Test StoreNotFoundError is raised."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSystemStore(base_path=tmpdir)

            with pytest.raises(StoreNotFoundError):
                store.get("nonexistent_id")

    def test_query_with_limit(self) -> None:
        """Test querying with limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSystemStore(base_path=tmpdir)

            # Save 5 results
            for i in range(5):
                result = ValidationResult(
                    run_id=f"run_{i}",
                    run_time=datetime.now() + timedelta(seconds=i),
                    data_asset="test.csv",
                    status=ResultStatus.SUCCESS,
                )
                store.save(result)

            query = StoreQuery(limit=3)
            results = store.query(query)
            assert len(results) == 3

    def test_rebuild_index(self, sample_result: ValidationResult) -> None:
        """Test rebuilding index from files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSystemStore(base_path=tmpdir)
            store.save(sample_result)

            # Clear index and rebuild
            store._index = {}
            count = store.rebuild_index()

            assert count == 1
            assert sample_result.run_id in store._index


# =============================================================================
# ExpectationSuite Tests
# =============================================================================


class TestExpectationSuite:
    """Tests for ExpectationSuite."""

    def test_create_and_serialize(self, sample_suite: ExpectationSuite) -> None:
        """Test creating and serializing a suite."""
        data = sample_suite.to_dict()
        restored = ExpectationSuite.from_dict(data)

        assert restored.name == sample_suite.name
        assert restored.data_asset == sample_suite.data_asset
        assert len(restored.expectations) == len(sample_suite.expectations)

    def test_add_expectation(self) -> None:
        """Test adding expectations."""
        suite = ExpectationSuite(name="test", data_asset="test.csv")

        suite.add_expectation(Expectation(
            expectation_type="not_null",
            column="id",
        ))

        assert suite.expectation_count == 1
        assert suite.enabled_count == 1

    def test_get_expectations_for_column(self, sample_suite: ExpectationSuite) -> None:
        """Test getting expectations for a column."""
        email_expectations = sample_suite.get_expectations_for_column("email")
        assert len(email_expectations) == 1
        assert email_expectations[0].expectation_type == "not_null"

    def test_memory_expectation_store(self, sample_suite: ExpectationSuite) -> None:
        """Test memory expectation store."""
        store = MemoryExpectationStore()

        store.save(sample_suite)
        retrieved = store.get(sample_suite.name)

        assert retrieved.name == sample_suite.name
        assert len(retrieved.expectations) == len(sample_suite.expectations)


# =============================================================================
# Factory Tests
# =============================================================================


class TestStoreFactory:
    """Tests for store factory."""

    def test_get_memory_store(self) -> None:
        """Test getting memory store."""
        store = get_store("memory")
        assert isinstance(store, MemoryStore)

    def test_get_filesystem_store(self) -> None:
        """Test getting filesystem store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = get_store("filesystem", base_path=tmpdir)
            assert isinstance(store, FileSystemStore)

    def test_unknown_backend_error(self) -> None:
        """Test error for unknown backend."""
        from truthound.stores.base import StoreError

        with pytest.raises(StoreError) as exc_info:
            get_store("unknown_backend")

        assert "Unknown store backend" in str(exc_info.value)


# =============================================================================
# Integration Tests
# =============================================================================


class TestStoreIntegration:
    """Integration tests for stores."""

    def test_validation_history(self) -> None:
        """Test getting validation history."""
        store = MemoryStore()

        # Save multiple results for same data asset
        for i in range(5):
            result = ValidationResult(
                run_id=f"run_{i}",
                run_time=datetime.now() + timedelta(hours=i),
                data_asset="customers.csv",
                status=ResultStatus.SUCCESS if i % 2 == 0 else ResultStatus.FAILURE,
            )
            store.save(result)

        # Get history
        history = store.get_history("customers.csv", limit=3)
        assert len(history) == 3

        # Most recent should be first
        assert history[0].run_id == "run_4"

    def test_get_latest(self) -> None:
        """Test getting latest result."""
        store = MemoryStore()

        for i in range(3):
            result = ValidationResult(
                run_id=f"run_{i}",
                run_time=datetime.now() + timedelta(hours=i),
                data_asset="test.csv",
                status=ResultStatus.SUCCESS,
            )
            store.save(result)

        latest = store.get_latest("test.csv")
        assert latest is not None
        assert latest.run_id == "run_2"

    def test_get_failures(self) -> None:
        """Test getting failed results."""
        store = MemoryStore()

        for i, status in enumerate([ResultStatus.SUCCESS, ResultStatus.FAILURE, ResultStatus.FAILURE]):
            result = ValidationResult(
                run_id=f"run_{i}",
                run_time=datetime.now(),
                data_asset="test.csv",
                status=status,
            )
            store.save(result)

        failures = store.get_failures()
        assert len(failures) == 2

    def test_context_manager(self) -> None:
        """Test using store as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with FileSystemStore(base_path=tmpdir) as store:
                result = ValidationResult(
                    run_id="test_run",
                    run_time=datetime.now(),
                    data_asset="test.csv",
                    status=ResultStatus.SUCCESS,
                )
                store.save(result)

                assert store.exists("test_run")
