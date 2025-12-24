"""Mock implementations for optional dependency testing.

This module provides realistic mock implementations that match the Protocol
definitions, allowing tests to run without actual cloud/database dependencies.
"""

from tests.mocks.cloud_mocks import (
    MockS3Client,
    MockGCSClient,
    MockGCSBucket,
    MockGCSBlob,
    create_mock_s3_client,
    create_mock_gcs_client,
)
from tests.mocks.database_mocks import (
    MockSQLEngine,
    MockSQLSession,
    MockSessionFactory,
    create_mock_database,
)
from tests.mocks.reporter_mocks import (
    MockJinja2Environment,
    MockJinja2Template,
)

__all__ = [
    # S3
    "MockS3Client",
    "create_mock_s3_client",
    # GCS
    "MockGCSClient",
    "MockGCSBucket",
    "MockGCSBlob",
    "create_mock_gcs_client",
    # Database
    "MockSQLEngine",
    "MockSQLSession",
    "MockSessionFactory",
    "create_mock_database",
    # Reporters
    "MockJinja2Environment",
    "MockJinja2Template",
]
