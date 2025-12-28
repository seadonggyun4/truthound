"""Pytest fixtures for store integration tests.

These fixtures provide test data and skip conditions for cloud-based tests.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any, Generator

import pytest


# =============================================================================
# Skip Markers
# =============================================================================


def has_aws_credentials() -> bool:
    """Check if AWS credentials are available."""
    return bool(
        os.environ.get("AWS_ACCESS_KEY_ID")
        and os.environ.get("AWS_SECRET_ACCESS_KEY")
    ) or bool(os.environ.get("AWS_PROFILE"))


def has_gcs_credentials() -> bool:
    """Check if GCS credentials are available."""
    return bool(
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        or os.environ.get("GCS_CREDENTIALS_JSON")
    )


def has_azure_credentials() -> bool:
    """Check if Azure credentials are available."""
    return bool(
        os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        or os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
    )


# Skip markers
skip_no_aws = pytest.mark.skipif(
    not has_aws_credentials(),
    reason="AWS credentials not configured",
)

skip_no_gcs = pytest.mark.skipif(
    not has_gcs_credentials(),
    reason="GCS credentials not configured",
)

skip_no_azure = pytest.mark.skipif(
    not has_azure_credentials(),
    reason="Azure credentials not configured",
)


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def test_run_id() -> str:
    """Generate a unique test run ID."""
    return f"test-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def sample_validation_result() -> dict[str, Any]:
    """Create sample validation result data."""
    return {
        "run_id": f"run-{uuid.uuid4().hex[:8]}",
        "data_asset": "test_dataset.csv",
        "run_time": datetime.now().isoformat(),
        "status": "success",
        "statistics": {
            "total_expectations": 10,
            "successful_expectations": 9,
            "failed_expectations": 1,
            "success_rate": 0.9,
        },
        "results": [
            {
                "expectation_type": "expect_column_to_exist",
                "column": "id",
                "success": True,
                "result": {"observed_value": True},
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "column": "name",
                "success": True,
                "result": {"observed_value": 0},
            },
            {
                "expectation_type": "expect_column_values_to_be_unique",
                "column": "id",
                "success": False,
                "result": {"observed_value": 5},
            },
        ],
        "tags": {"environment": "test", "pipeline": "integration"},
        "meta": {"version": "1.0.0"},
    }


@pytest.fixture
def multiple_validation_results(
    sample_validation_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create multiple sample validation results."""
    results = []
    for i in range(5):
        result = sample_validation_result.copy()
        result["run_id"] = f"run-{i:03d}-{uuid.uuid4().hex[:8]}"
        result["data_asset"] = f"dataset_{i}.csv"
        result["status"] = "success" if i % 2 == 0 else "failure"
        results.append(result)
    return results


# =============================================================================
# AWS Fixtures
# =============================================================================


@pytest.fixture
def aws_test_bucket() -> str:
    """Get the S3 bucket for testing."""
    return os.environ.get("TRUTHOUND_TEST_S3_BUCKET", "truthound-test-bucket")


@pytest.fixture
def aws_test_prefix(test_run_id: str) -> str:
    """Get a unique prefix for test objects."""
    return f"integration-tests/{test_run_id}/"


# =============================================================================
# GCS Fixtures
# =============================================================================


@pytest.fixture
def gcs_test_bucket() -> str:
    """Get the GCS bucket for testing."""
    return os.environ.get("TRUTHOUND_TEST_GCS_BUCKET", "truthound-test-bucket")


@pytest.fixture
def gcs_test_prefix(test_run_id: str) -> str:
    """Get a unique prefix for test objects."""
    return f"integration-tests/{test_run_id}/"


# =============================================================================
# Azure Fixtures
# =============================================================================


@pytest.fixture
def azure_test_container() -> str:
    """Get the Azure container for testing."""
    return os.environ.get("TRUTHOUND_TEST_AZURE_CONTAINER", "truthound-test")


@pytest.fixture
def azure_test_prefix(test_run_id: str) -> str:
    """Get a unique prefix for test objects."""
    return f"integration-tests/{test_run_id}/"


@pytest.fixture
def azure_connection_string() -> str | None:
    """Get Azure connection string."""
    return os.environ.get("AZURE_STORAGE_CONNECTION_STRING")


# =============================================================================
# Cleanup Fixtures
# =============================================================================


@pytest.fixture
def cleanup_s3(aws_test_bucket: str, aws_test_prefix: str) -> Generator[None, None, None]:
    """Cleanup S3 objects after test."""
    yield

    if not has_aws_credentials():
        return

    try:
        import boto3

        s3 = boto3.client("s3")

        # List and delete test objects
        response = s3.list_objects_v2(
            Bucket=aws_test_bucket,
            Prefix=aws_test_prefix,
        )

        if "Contents" in response:
            for obj in response["Contents"]:
                s3.delete_object(Bucket=aws_test_bucket, Key=obj["Key"])

    except Exception as e:
        print(f"S3 cleanup failed: {e}")


@pytest.fixture
def cleanup_gcs(gcs_test_bucket: str, gcs_test_prefix: str) -> Generator[None, None, None]:
    """Cleanup GCS objects after test."""
    yield

    if not has_gcs_credentials():
        return

    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(gcs_test_bucket)

        # List and delete test objects
        blobs = bucket.list_blobs(prefix=gcs_test_prefix)
        for blob in blobs:
            blob.delete()

    except Exception as e:
        print(f"GCS cleanup failed: {e}")


@pytest.fixture
def cleanup_azure(
    azure_test_container: str,
    azure_test_prefix: str,
    azure_connection_string: str | None,
) -> Generator[None, None, None]:
    """Cleanup Azure blobs after test."""
    yield

    if not has_azure_credentials() or not azure_connection_string:
        return

    try:
        from azure.storage.blob import BlobServiceClient

        client = BlobServiceClient.from_connection_string(azure_connection_string)
        container = client.get_container_client(azure_test_container)

        # List and delete test blobs
        blobs = container.list_blobs(name_starts_with=azure_test_prefix)
        for blob in blobs:
            container.delete_blob(blob.name)

    except Exception as e:
        print(f"Azure cleanup failed: {e}")
