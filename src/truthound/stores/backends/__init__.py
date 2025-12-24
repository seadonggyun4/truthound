"""Store backends for different storage systems.

This package contains implementations for various storage backends:

- filesystem: Local filesystem storage (default, no dependencies)
- memory: In-memory storage (for testing, no dependencies)
- s3: AWS S3 storage (requires boto3)
- gcs: Google Cloud Storage (requires google-cloud-storage)
- database: SQL database storage (requires sqlalchemy)

Use the get_store() factory function to create store instances:

    >>> from truthound.stores import get_store
    >>> store = get_store("filesystem", base_path=".truthound/results")
"""

# Backends are imported lazily by the factory to avoid import errors
# when optional dependencies are not installed.
