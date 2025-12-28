"""Unit tests for the Azure Blob store backend."""

from __future__ import annotations

import gzip
import json
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from truthound.stores.backends.azure_blob import (
    AzureBlobConfig,
    AzureBlobStore,
    HAS_AZURE,
)
from truthound.stores.base import StoreConnectionError, StoreNotFoundError


class TestAzureBlobConfig:
    """Tests for AzureBlobConfig data class."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = AzureBlobConfig()

        assert config.container == ""
        assert config.prefix == "truthound/"
        assert config.use_compression is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = AzureBlobConfig(
            container="my-container",
            prefix="custom/",
            connection_string="DefaultEndpointsProtocol=https;...",
            use_compression=False,
            access_tier="Cool",
        )

        assert config.container == "my-container"
        assert config.prefix == "custom/"
        assert config.use_compression is False
        assert config.access_tier == "Cool"

    def test_get_full_prefix(self) -> None:
        """Test full prefix generation."""
        config = AzureBlobConfig(
            prefix="validations/",
            namespace="production",
        )

        full_prefix = config.get_full_prefix()

        assert full_prefix == "validations/production/"

    def test_get_full_prefix_no_namespace(self) -> None:
        """Test full prefix without namespace."""
        config = AzureBlobConfig(
            prefix="validations/",
            namespace="",
        )

        full_prefix = config.get_full_prefix()

        assert full_prefix == "validations/"


@pytest.mark.skipif(not HAS_AZURE, reason="azure-storage-blob not installed")
class TestAzureBlobStoreWithAzure:
    """Tests for AzureBlobStore that require azure-storage-blob."""

    def test_store_creation(self) -> None:
        """Test store creation with required parameters."""
        store = AzureBlobStore(
            container="my-container",
            connection_string="DefaultEndpointsProtocol=https;...",
        )

        assert store._config.container == "my-container"

    def test_store_creation_with_account_url(self) -> None:
        """Test store creation with account URL."""
        store = AzureBlobStore(
            container="my-container",
            account_url="https://myaccount.blob.core.windows.net",
            sas_token="sv=2021-06-08&ss=bfqt...",
        )

        assert store._config.account_url == "https://myaccount.blob.core.windows.net"
        assert store._config.sas_token is not None


class TestAzureBlobStoreMocked:
    """Tests for AzureBlobStore with mocked Azure SDK."""

    @pytest.fixture
    def mock_azure(self) -> MagicMock:
        """Create mock Azure SDK components."""
        mock_blob_service = MagicMock()
        mock_container_client = MagicMock()
        mock_blob_client = MagicMock()

        mock_blob_service.get_container_client.return_value = mock_container_client
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_container_client.exists.return_value = True

        return {
            "service": mock_blob_service,
            "container": mock_container_client,
            "blob": mock_blob_client,
        }

    @pytest.fixture
    def sample_result_data(self) -> dict[str, Any]:
        """Create sample validation result data."""
        return {
            "run_id": "test-run-123",
            "data_asset": "test_dataset.csv",
            "run_time": datetime.now().isoformat(),
            "status": "success",
            "metrics": {
                "total_rows": 1000,
                "passed": 950,
                "failed": 50,
            },
            "tags": {"env": "test"},
        }

    def test_serialize_with_compression(self) -> None:
        """Test data serialization with compression."""
        store = AzureBlobStore.__new__(AzureBlobStore)
        store._config = AzureBlobConfig(
            container="test",
            use_compression=True,
        )

        data = {"key": "value", "number": 123}
        serialized = store._serialize(data)

        # Should be gzip compressed
        decompressed = gzip.decompress(serialized)
        deserialized = json.loads(decompressed.decode("utf-8"))

        assert deserialized == data

    def test_serialize_without_compression(self) -> None:
        """Test data serialization without compression."""
        store = AzureBlobStore.__new__(AzureBlobStore)
        store._config = AzureBlobConfig(
            container="test",
            use_compression=False,
        )

        data = {"key": "value", "number": 123}
        serialized = store._serialize(data)

        # Should be plain JSON
        deserialized = json.loads(serialized.decode("utf-8"))

        assert deserialized == data

    def test_deserialize_with_compression(self) -> None:
        """Test data deserialization with compression."""
        store = AzureBlobStore.__new__(AzureBlobStore)
        store._config = AzureBlobConfig(
            container="test",
            use_compression=True,
        )

        data = {"key": "value", "number": 123}
        compressed = gzip.compress(json.dumps(data).encode("utf-8"))

        deserialized = store._deserialize(compressed)

        assert deserialized == data

    def test_deserialize_without_compression(self) -> None:
        """Test data deserialization without compression."""
        store = AzureBlobStore.__new__(AzureBlobStore)
        store._config = AzureBlobConfig(
            container="test",
            use_compression=False,
        )

        data = {"key": "value", "number": 123}
        content = json.dumps(data).encode("utf-8")

        deserialized = store._deserialize(content)

        assert deserialized == data

    def test_get_blob_name_with_compression(self) -> None:
        """Test blob name generation with compression."""
        store = AzureBlobStore.__new__(AzureBlobStore)
        store._config = AzureBlobConfig(
            container="test",
            prefix="validations/",
            namespace="prod",
            use_compression=True,
        )

        blob_name = store._get_blob_name("run-123")

        assert blob_name == "validations/prod/results/run-123.json.gz"

    def test_get_blob_name_without_compression(self) -> None:
        """Test blob name generation without compression."""
        store = AzureBlobStore.__new__(AzureBlobStore)
        store._config = AzureBlobConfig(
            container="test",
            prefix="validations/",
            namespace="prod",
            use_compression=False,
        )

        blob_name = store._get_blob_name("run-123")

        assert blob_name == "validations/prod/results/run-123.json"


class TestAzureBlobStoreAuthentication:
    """Tests for Azure Blob Store authentication options."""

    def test_connection_string_auth(self) -> None:
        """Test connection string authentication config."""
        config = AzureBlobConfig(
            container="test",
            connection_string="DefaultEndpointsProtocol=https;AccountName=test;...",
        )

        assert config.connection_string is not None
        assert config.account_url is None

    def test_account_url_with_sas_auth(self) -> None:
        """Test account URL with SAS token authentication config."""
        config = AzureBlobConfig(
            container="test",
            account_url="https://myaccount.blob.core.windows.net",
            sas_token="sv=2021-06-08&ss=bfqt...",
        )

        assert config.account_url is not None
        assert config.sas_token is not None

    def test_account_name_key_auth(self) -> None:
        """Test account name and key authentication config."""
        config = AzureBlobConfig(
            container="test",
            account_name="myaccount",
            account_key="base64encodedkey==",
        )

        assert config.account_name == "myaccount"
        assert config.account_key is not None


class TestAzureBlobStoreAccessTiers:
    """Tests for Azure Blob Store access tiers."""

    def test_access_tier_config(self) -> None:
        """Test access tier configuration."""
        for tier in ["Hot", "Cool", "Archive"]:
            config = AzureBlobConfig(
                container="test",
                access_tier=tier,
            )

            assert config.access_tier == tier

    def test_no_access_tier(self) -> None:
        """Test configuration without access tier."""
        config = AzureBlobConfig(container="test")

        assert config.access_tier is None


class TestAzureBlobStoreMetadata:
    """Tests for Azure Blob Store metadata handling."""

    def test_custom_metadata(self) -> None:
        """Test custom metadata configuration."""
        config = AzureBlobConfig(
            container="test",
            metadata={
                "environment": "production",
                "team": "data-quality",
            },
        )

        assert config.metadata["environment"] == "production"
        assert config.metadata["team"] == "data-quality"

    def test_default_metadata(self) -> None:
        """Test default empty metadata."""
        config = AzureBlobConfig(container="test")

        assert config.metadata == {}


class TestAzureBlobStoreIndex:
    """Tests for Azure Blob Store index management."""

    def test_index_initialization(self) -> None:
        """Test index initialization."""
        store = AzureBlobStore.__new__(AzureBlobStore)
        store._index = {}

        assert len(store._index) == 0

    def test_index_lookup(self) -> None:
        """Test index-based item lookup."""
        store = AzureBlobStore.__new__(AzureBlobStore)
        store._index = {
            "run-123": {
                "data_asset": "test.csv",
                "run_time": "2025-01-01T12:00:00",
                "status": "success",
            },
            "run-456": {
                "data_asset": "test.csv",
                "run_time": "2025-01-02T12:00:00",
                "status": "failure",
            },
        }

        assert "run-123" in store._index
        assert "run-789" not in store._index
