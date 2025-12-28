"""Integration tests for S3 store backend.

These tests require actual AWS credentials and an S3 bucket to run.
Set the following environment variables:
- AWS_ACCESS_KEY_ID or AWS_PROFILE
- AWS_SECRET_ACCESS_KEY (if using access key)
- TRUTHOUND_TEST_S3_BUCKET (default: truthound-test-bucket)

Run with: pytest tests/integration/stores/test_s3_store.py -v
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta

import pytest

from tests.integration.stores.conftest import skip_no_aws


@skip_no_aws
class TestS3StoreIntegration:
    """Integration tests for S3Store."""

    @pytest.fixture(autouse=True)
    def setup(
        self,
        aws_test_bucket: str,
        aws_test_prefix: str,
        cleanup_s3: None,
    ) -> None:
        """Set up test fixtures."""
        self.bucket = aws_test_bucket
        self.prefix = aws_test_prefix

    def test_s3_store_save_and_get(
        self,
        sample_validation_result: dict,
    ) -> None:
        """Test saving and retrieving a validation result."""
        from truthound.stores import get_store
        from truthound.stores.results import ValidationResult

        store = get_store(
            "s3",
            bucket=self.bucket,
            prefix=self.prefix,
            compression=True,
        )

        try:
            store.initialize()

            # Create result
            result = ValidationResult.from_dict(sample_validation_result)

            # Save
            run_id = store.save(result)
            assert run_id == result.run_id

            # Get
            retrieved = store.get(run_id)
            assert retrieved.run_id == result.run_id
            assert retrieved.data_asset == result.data_asset
            assert retrieved.status == result.status

        finally:
            store.close()

    def test_s3_store_exists(
        self,
        sample_validation_result: dict,
    ) -> None:
        """Test checking if result exists."""
        from truthound.stores import get_store
        from truthound.stores.results import ValidationResult

        store = get_store(
            "s3",
            bucket=self.bucket,
            prefix=self.prefix,
        )

        try:
            store.initialize()

            result = ValidationResult.from_dict(sample_validation_result)
            run_id = store.save(result)

            assert store.exists(run_id)
            assert not store.exists("nonexistent-id")

        finally:
            store.close()

    def test_s3_store_delete(
        self,
        sample_validation_result: dict,
    ) -> None:
        """Test deleting a result."""
        from truthound.stores import get_store
        from truthound.stores.results import ValidationResult

        store = get_store(
            "s3",
            bucket=self.bucket,
            prefix=self.prefix,
        )

        try:
            store.initialize()

            result = ValidationResult.from_dict(sample_validation_result)
            run_id = store.save(result)

            assert store.exists(run_id)
            assert store.delete(run_id)
            assert not store.exists(run_id)

        finally:
            store.close()

    def test_s3_store_list_and_query(
        self,
        multiple_validation_results: list[dict],
    ) -> None:
        """Test listing and querying results."""
        from truthound.stores import get_store, StoreQuery
        from truthound.stores.results import ValidationResult

        store = get_store(
            "s3",
            bucket=self.bucket,
            prefix=self.prefix,
        )

        try:
            store.initialize()

            # Save multiple results
            for result_data in multiple_validation_results:
                result = ValidationResult.from_dict(result_data)
                store.save(result)

            # List all
            ids = store.list_ids()
            assert len(ids) >= len(multiple_validation_results)

            # Query with filter
            query = StoreQuery(status="success", limit=10)
            results = store.query(query)
            assert all(r.status.value == "success" for r in results)

        finally:
            store.close()

    def test_s3_store_compression(
        self,
        sample_validation_result: dict,
    ) -> None:
        """Test that compression works correctly."""
        from truthound.stores import get_store
        from truthound.stores.results import ValidationResult

        # Test with compression
        store_compressed = get_store(
            "s3",
            bucket=self.bucket,
            prefix=f"{self.prefix}compressed/",
            compression=True,
        )

        # Test without compression
        store_uncompressed = get_store(
            "s3",
            bucket=self.bucket,
            prefix=f"{self.prefix}uncompressed/",
            compression=False,
        )

        try:
            store_compressed.initialize()
            store_uncompressed.initialize()

            result = ValidationResult.from_dict(sample_validation_result)

            # Save to both
            id1 = store_compressed.save(result)
            result._run_id = f"uncompressed-{uuid.uuid4().hex[:8]}"
            id2 = store_uncompressed.save(result)

            # Both should be retrievable
            r1 = store_compressed.get(id1)
            r2 = store_uncompressed.get(id2)

            assert r1.data_asset == result.data_asset
            assert r2.data_asset == result.data_asset

        finally:
            store_compressed.close()
            store_uncompressed.close()

    def test_s3_store_namespaces(
        self,
        sample_validation_result: dict,
    ) -> None:
        """Test namespace isolation."""
        from truthound.stores import get_store
        from truthound.stores.results import ValidationResult

        store1 = get_store(
            "s3",
            bucket=self.bucket,
            prefix=self.prefix,
            namespace="namespace1",
        )

        store2 = get_store(
            "s3",
            bucket=self.bucket,
            prefix=self.prefix,
            namespace="namespace2",
        )

        try:
            store1.initialize()
            store2.initialize()

            result = ValidationResult.from_dict(sample_validation_result)
            run_id = store1.save(result)

            # Should be in store1 but not store2
            assert store1.exists(run_id)
            assert not store2.exists(run_id)

        finally:
            store1.close()
            store2.close()

    def test_s3_store_storage_class(
        self,
        sample_validation_result: dict,
    ) -> None:
        """Test using different storage classes."""
        from truthound.stores import get_store
        from truthound.stores.results import ValidationResult

        store = get_store(
            "s3",
            bucket=self.bucket,
            prefix=self.prefix,
            storage_class="STANDARD_IA",
        )

        try:
            store.initialize()

            result = ValidationResult.from_dict(sample_validation_result)
            run_id = store.save(result)

            # Should be retrievable
            retrieved = store.get(run_id)
            assert retrieved.run_id == run_id

        finally:
            store.close()

    def test_s3_store_concurrent_operations(
        self,
        multiple_validation_results: list[dict],
    ) -> None:
        """Test concurrent save and get operations."""
        import concurrent.futures

        from truthound.stores import get_store
        from truthound.stores.results import ValidationResult

        store = get_store(
            "s3",
            bucket=self.bucket,
            prefix=self.prefix,
        )

        try:
            store.initialize()

            def save_result(result_data: dict) -> str:
                result = ValidationResult.from_dict(result_data)
                return store.save(result)

            # Concurrent saves
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(save_result, data)
                    for data in multiple_validation_results
                ]
                run_ids = [f.result() for f in futures]

            assert len(run_ids) == len(multiple_validation_results)

            # Concurrent reads
            def get_result(run_id: str) -> ValidationResult:
                return store.get(run_id)

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(get_result, rid) for rid in run_ids]
                results = [f.result() for f in futures]

            assert len(results) == len(run_ids)

        finally:
            store.close()

    def test_s3_store_large_result(self) -> None:
        """Test handling large validation results."""
        from truthound.stores import get_store
        from truthound.stores.results import ValidationResult

        store = get_store(
            "s3",
            bucket=self.bucket,
            prefix=self.prefix,
            compression=True,  # Important for large data
        )

        try:
            store.initialize()

            # Create a large result
            large_results = [
                {
                    "expectation_type": f"expect_test_{i}",
                    "column": f"column_{i}",
                    "success": i % 2 == 0,
                    "result": {"observed_value": "x" * 1000},  # 1KB per result
                }
                for i in range(1000)  # ~1MB of results
            ]

            large_data = {
                "run_id": f"large-{uuid.uuid4().hex[:8]}",
                "data_asset": "large_dataset.csv",
                "run_time": datetime.now().isoformat(),
                "status": "success",
                "statistics": {
                    "total_expectations": len(large_results),
                    "successful_expectations": len(large_results) // 2,
                },
                "results": large_results,
                "tags": {},
                "meta": {},
            }

            result = ValidationResult.from_dict(large_data)
            run_id = store.save(result)

            # Should be retrievable
            retrieved = store.get(run_id)
            assert len(retrieved.results) == len(large_results)

        finally:
            store.close()
