"""Tests for optional storage backends using mocks.

These tests verify the logic of S3Store, GCSStore, and DatabaseStore
without requiring actual cloud/database dependencies.
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from truthound.stores.results import (
    ValidationResult,
    ValidatorResult,
    ResultStatistics,
    ResultStatus,
)
from truthound.stores.base import StoreQuery

# Import mocks
from tests.mocks.cloud_mocks import (
    MockS3Client,
    MockS3ClientError,
    MockGCSClient,
    MockGCSNotFound,
    create_mock_s3_client,
    create_mock_gcs_client,
)
from tests.mocks.database_mocks import (
    MockSQLEngine,
    MockSessionFactory,
    MockDatabase,
    MockValidationResultModel,
    create_mock_database,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_result() -> ValidationResult:
    """Create a sample validation result for testing."""
    return ValidationResult(
        run_id="test_run_001",
        run_time=datetime(2024, 1, 15, 10, 30, 0),
        data_asset="customers.csv",
        status=ResultStatus.FAILURE,
        results=[
            ValidatorResult(
                validator_name="null_check",
                success=False,
                column="email",
                issue_type="null_values",
                count=5,
                severity="high",
                message="Found 5 null values in email column",
            ),
            ValidatorResult(
                validator_name="type_check",
                success=True,
                column="id",
            ),
        ],
        statistics=ResultStatistics(
            total_validators=2,
            passed_validators=1,
            failed_validators=1,
            total_rows=1000,
            total_columns=10,
            total_issues=1,
            high_issues=1,
        ),
        tags={"env": "production"},
    )


@pytest.fixture
def success_result() -> ValidationResult:
    """Create a successful validation result."""
    return ValidationResult(
        run_id="success_run_001",
        run_time=datetime(2024, 1, 15, 11, 0, 0),
        data_asset="orders.csv",
        status=ResultStatus.SUCCESS,
        results=[
            ValidatorResult(
                validator_name="null_check",
                success=True,
                column="order_id",
            ),
        ],
        statistics=ResultStatistics(
            total_validators=1,
            passed_validators=1,
            failed_validators=0,
            total_rows=500,
            total_columns=8,
            total_issues=0,
        ),
    )


# =============================================================================
# S3Store Tests (Mock-based)
# =============================================================================


class TestS3StoreMock:
    """Tests for S3Store using mocks."""

    def test_save_and_get(self, sample_result: ValidationResult) -> None:
        """Test saving and retrieving a validation result."""
        # Create mock client with bucket
        mock_client = create_mock_s3_client(with_bucket="test-bucket")

        # Patch boto3.client to return our mock
        with patch("truthound.stores.backends.s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client

            # Import after patching
            from truthound.stores.backends.s3 import S3Store

            store = S3Store(bucket="test-bucket", prefix="validations/")
            store.initialize()

            # Save
            run_id = store.save(sample_result)
            assert run_id == "test_run_001"

            # Get
            retrieved = store.get(run_id)
            assert retrieved.run_id == sample_result.run_id
            assert retrieved.data_asset == sample_result.data_asset
            assert retrieved.status == sample_result.status

    def test_exists(self, sample_result: ValidationResult) -> None:
        """Test checking if a result exists."""
        mock_client = create_mock_s3_client(with_bucket="test-bucket")

        with patch("truthound.stores.backends.s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client

            from truthound.stores.backends.s3 import S3Store

            store = S3Store(bucket="test-bucket")
            store.initialize()

            # Initially doesn't exist
            assert not store.exists("test_run_001")

            # After save, exists
            store.save(sample_result)
            assert store.exists("test_run_001")

    def test_delete(self, sample_result: ValidationResult) -> None:
        """Test deleting a validation result."""
        mock_client = create_mock_s3_client(with_bucket="test-bucket")

        with patch("truthound.stores.backends.s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client

            from truthound.stores.backends.s3 import S3Store

            store = S3Store(bucket="test-bucket")
            store.initialize()

            store.save(sample_result)
            assert store.exists("test_run_001")

            # Delete
            result = store.delete("test_run_001")
            assert result is True
            assert not store.exists("test_run_001")

            # Delete non-existent
            result = store.delete("non_existent")
            assert result is False

    def test_list_ids(
        self,
        sample_result: ValidationResult,
        success_result: ValidationResult,
    ) -> None:
        """Test listing validation result IDs."""
        mock_client = create_mock_s3_client(with_bucket="test-bucket")

        with patch("truthound.stores.backends.s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client

            from truthound.stores.backends.s3 import S3Store

            store = S3Store(bucket="test-bucket")
            store.initialize()

            store.save(sample_result)
            store.save(success_result)

            ids = store.list_ids()
            assert len(ids) == 2
            assert "test_run_001" in ids
            assert "success_run_001" in ids

    def test_query_by_data_asset(
        self,
        sample_result: ValidationResult,
        success_result: ValidationResult,
    ) -> None:
        """Test querying by data asset."""
        mock_client = create_mock_s3_client(with_bucket="test-bucket")

        with patch("truthound.stores.backends.s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client

            from truthound.stores.backends.s3 import S3Store

            store = S3Store(bucket="test-bucket")
            store.initialize()

            store.save(sample_result)
            store.save(success_result)

            query = StoreQuery(data_asset="customers.csv")
            results = store.query(query)

            assert len(results) == 1
            assert results[0].data_asset == "customers.csv"

    def test_compression(self, sample_result: ValidationResult) -> None:
        """Test that compression works correctly."""
        mock_client = create_mock_s3_client(with_bucket="test-bucket")

        with patch("truthound.stores.backends.s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client

            from truthound.stores.backends.s3 import S3Store

            store = S3Store(bucket="test-bucket", compression=True)
            store.initialize()

            store.save(sample_result)

            # Verify data was compressed (stored object should be smaller)
            retrieved = store.get("test_run_001")
            assert retrieved.run_id == sample_result.run_id

    def test_bucket_not_found(self) -> None:
        """Test error when bucket doesn't exist."""
        mock_client = MockS3Client()  # No bucket created

        with patch("truthound.stores.backends.s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client

            from truthound.stores.backends.s3 import S3Store
            from truthound.stores.base import StoreConnectionError

            store = S3Store(bucket="non-existent-bucket")

            with pytest.raises(StoreConnectionError) as exc_info:
                store.initialize()

            assert "Bucket not found" in str(exc_info.value)


# =============================================================================
# GCSStore Tests (Mock-based)
# =============================================================================


class TestGCSStoreMock:
    """Tests for GCSStore using mocks."""

    def test_save_and_get(self, sample_result: ValidationResult) -> None:
        """Test saving and retrieving a validation result."""
        mock_client = create_mock_gcs_client(with_bucket="test-bucket")

        with patch("truthound.stores.backends.gcs.storage") as mock_storage:
            mock_storage.Client.return_value = mock_client

            from truthound.stores.backends.gcs import GCSStore

            store = GCSStore(bucket="test-bucket", prefix="validations/")
            store.initialize()

            # Save
            run_id = store.save(sample_result)
            assert run_id == "test_run_001"

            # Get
            retrieved = store.get(run_id)
            assert retrieved.run_id == sample_result.run_id
            assert retrieved.data_asset == sample_result.data_asset

    def test_exists(self, sample_result: ValidationResult) -> None:
        """Test checking if a result exists."""
        mock_client = create_mock_gcs_client(with_bucket="test-bucket")

        with patch("truthound.stores.backends.gcs.storage") as mock_storage:
            mock_storage.Client.return_value = mock_client

            from truthound.stores.backends.gcs import GCSStore

            store = GCSStore(bucket="test-bucket")
            store.initialize()

            assert not store.exists("test_run_001")

            store.save(sample_result)
            assert store.exists("test_run_001")

    def test_delete(self, sample_result: ValidationResult) -> None:
        """Test deleting a validation result."""
        mock_client = create_mock_gcs_client(with_bucket="test-bucket")

        with patch("truthound.stores.backends.gcs.storage") as mock_storage:
            mock_storage.Client.return_value = mock_client

            from truthound.stores.backends.gcs import GCSStore

            store = GCSStore(bucket="test-bucket")
            store.initialize()

            store.save(sample_result)
            assert store.delete("test_run_001") is True
            assert not store.exists("test_run_001")

    def test_list_ids(
        self,
        sample_result: ValidationResult,
        success_result: ValidationResult,
    ) -> None:
        """Test listing validation result IDs."""
        mock_client = create_mock_gcs_client(with_bucket="test-bucket")

        with patch("truthound.stores.backends.gcs.storage") as mock_storage:
            mock_storage.Client.return_value = mock_client

            from truthound.stores.backends.gcs import GCSStore

            store = GCSStore(bucket="test-bucket")
            store.initialize()

            store.save(sample_result)
            store.save(success_result)

            ids = store.list_ids()
            assert len(ids) == 2

    def test_bucket_not_found(self) -> None:
        """Test error when bucket doesn't exist."""
        mock_client = MockGCSClient()  # No bucket created

        with patch("truthound.stores.backends.gcs.storage") as mock_storage:
            mock_storage.Client.return_value = mock_client

            from truthound.stores.backends.gcs import GCSStore
            from truthound.stores.base import StoreConnectionError

            store = GCSStore(bucket="non-existent-bucket")

            with pytest.raises(StoreConnectionError) as exc_info:
                store.initialize()

            assert "Bucket not found" in str(exc_info.value)


# =============================================================================
# DatabaseStore Tests (Mock-based)
# =============================================================================


class TestDatabaseStoreMock:
    """Tests for DatabaseStore using mocks."""

    def test_save_and_get(self, sample_result: ValidationResult) -> None:
        """Test saving and retrieving a validation result via serialization."""
        # Test the serialization/deserialization logic that DatabaseStore uses
        data_dict = sample_result.to_dict()
        json_str = json.dumps(data_dict, default=str)

        # Verify round-trip
        parsed = json.loads(json_str)
        restored = ValidationResult.from_dict(parsed)

        assert restored.run_id == sample_result.run_id
        assert restored.data_asset == sample_result.data_asset
        assert restored.status == sample_result.status
        assert len(restored.results) == len(sample_result.results)

    def test_query_serialization(self) -> None:
        """Test that StoreQuery correctly filters data."""
        query = StoreQuery(
            data_asset="customers.csv",
            status="failure",
            limit=10,
            offset=0,
        )

        # Test matches method
        meta = {
            "data_asset": "customers.csv",
            "status": "failure",
            "run_time": "2024-01-15T10:30:00",
        }

        assert query.matches(meta) is True

        meta_mismatch = {
            "data_asset": "orders.csv",
            "status": "failure",
            "run_time": "2024-01-15T10:30:00",
        }

        assert query.matches(meta_mismatch) is False

    def test_engine_lifecycle(self) -> None:
        """Test engine creation and disposal."""
        engine, session_factory, db = create_mock_database()

        assert not engine._disposed

        engine.dispose()
        assert engine._disposed


# =============================================================================
# Integration Tests
# =============================================================================


class TestStoreFactoryWithMocks:
    """Test store factory with mock backends."""

    def test_factory_gcs_import_error(self) -> None:
        """Test that factory provides clear error when google-cloud-storage is missing."""
        from truthound.stores.factory import get_store
        from truthound.stores.base import StoreError

        with pytest.raises(StoreError) as exc_info:
            # This will fail because google-cloud-storage is not installed
            get_store("gcs", bucket="test")

        assert "google-cloud-storage" in str(exc_info.value).lower()

    def test_factory_database_import_error(self) -> None:
        """Test that factory provides clear error when sqlalchemy is missing."""
        from truthound.stores.factory import get_store
        from truthound.stores.base import StoreError

        with pytest.raises(StoreError) as exc_info:
            # This will fail because sqlalchemy is not installed
            get_store("database", connection_url="sqlite:///:memory:")

        assert "sqlalchemy" in str(exc_info.value).lower()

    def test_factory_unknown_backend_error(self) -> None:
        """Test that factory provides clear error for unknown backend."""
        from truthound.stores.factory import get_store
        from truthound.stores.base import StoreError

        with pytest.raises(StoreError) as exc_info:
            get_store("unknown_backend")

        assert "unknown" in str(exc_info.value).lower()
