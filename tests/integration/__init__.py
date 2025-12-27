"""Integration tests for Truthound.

This package contains integration tests that require external services
such as cloud data warehouses, databases, and message queues.

These tests are designed to:
- Run in CI/CD pipelines with proper credentials
- Skip gracefully when services are unavailable
- Provide comprehensive coverage of real-world scenarios
- Support multiple authentication methods

Structure:
    cloud_dw/       - Cloud Data Warehouse integration tests
    streaming/      - Streaming source integration tests (Kafka, Kinesis, etc.)
    storage/        - Storage backend integration tests (S3, GCS, etc.)
"""
