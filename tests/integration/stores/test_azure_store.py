"""Integration tests for Azure Blob store backend.

These tests require actual Azure credentials and a container to run.
Set the following environment variables:
- AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_NAME
- TRUTHOUND_TEST_AZURE_CONTAINER (default: truthound-test)

Run with: pytest tests/integration/stores/test_azure_store.py -v
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from tests.integration.stores.conftest import skip_no_azure


@skip_no_azure
class TestAzureBlobStoreIntegration:
    """Integration tests for AzureBlobStore."""

    @pytest.fixture(autouse=True)
    def setup(
        self,
        azure_test_container: str,
        azure_test_prefix: str,
        azure_connection_string: str | None,
        cleanup_azure: None,
    ) -> None:
        """Set up test fixtures."""
        self.container = azure_test_container
        self.prefix = azure_test_prefix
        self.connection_string = azure_connection_string

    def test_azure_store_save_and_get(
        self,
        sample_validation_result: dict,
    ) -> None:
        """Test saving and retrieving a validation result."""
        from truthound.stores import get_store
        from truthound.stores.results import ValidationResult

        store = get_store(
            "azure",
            container=self.container,
            connection_string=self.connection_string,
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

    def test_azure_store_exists(
        self,
        sample_validation_result: dict,
    ) -> None:
        """Test checking if result exists."""
        from truthound.stores import get_store
        from truthound.stores.results import ValidationResult

        store = get_store(
            "azure",
            container=self.container,
            connection_string=self.connection_string,
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

    def test_azure_store_delete(
        self,
        sample_validation_result: dict,
    ) -> None:
        """Test deleting a result."""
        from truthound.stores import get_store
        from truthound.stores.results import ValidationResult

        store = get_store(
            "azure",
            container=self.container,
            connection_string=self.connection_string,
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

    def test_azure_store_list_and_query(
        self,
        multiple_validation_results: list[dict],
    ) -> None:
        """Test listing and querying results."""
        from truthound.stores import get_store, StoreQuery
        from truthound.stores.results import ValidationResult

        store = get_store(
            "azure",
            container=self.container,
            connection_string=self.connection_string,
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

    def test_azure_store_compression(
        self,
        sample_validation_result: dict,
    ) -> None:
        """Test that compression works correctly."""
        from truthound.stores import get_store
        from truthound.stores.results import ValidationResult

        # Test with compression
        store_compressed = get_store(
            "azure",
            container=self.container,
            connection_string=self.connection_string,
            prefix=f"{self.prefix}compressed/",
            compression=True,
        )

        # Test without compression
        store_uncompressed = get_store(
            "azure",
            container=self.container,
            connection_string=self.connection_string,
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

    def test_azure_store_namespaces(
        self,
        sample_validation_result: dict,
    ) -> None:
        """Test namespace isolation."""
        from truthound.stores import get_store
        from truthound.stores.results import ValidationResult

        store1 = get_store(
            "azure",
            container=self.container,
            connection_string=self.connection_string,
            prefix=self.prefix,
            namespace="namespace1",
        )

        store2 = get_store(
            "azure",
            container=self.container,
            connection_string=self.connection_string,
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

    def test_azure_store_access_tier(
        self,
        sample_validation_result: dict,
    ) -> None:
        """Test setting access tier for blobs."""
        from truthound.stores import get_store
        from truthound.stores.results import ValidationResult

        store = get_store(
            "azure",
            container=self.container,
            connection_string=self.connection_string,
            prefix=self.prefix,
            access_tier="Cool",
        )

        try:
            store.initialize()

            result = ValidationResult.from_dict(sample_validation_result)
            run_id = store.save(result)

            # Should be retrievable
            retrieved = store.get(run_id)
            assert retrieved.run_id == run_id

            # Test changing tier
            store.set_access_tier(run_id, "Hot")

            # Still retrievable
            retrieved = store.get(run_id)
            assert retrieved.run_id == run_id

        finally:
            store.close()

    def test_azure_store_concurrent_operations(
        self,
        multiple_validation_results: list[dict],
    ) -> None:
        """Test concurrent save and get operations."""
        import concurrent.futures

        from truthound.stores import get_store
        from truthound.stores.results import ValidationResult

        store = get_store(
            "azure",
            container=self.container,
            connection_string=self.connection_string,
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
