"""Store validation result action.

This action saves validation results to a persistent store for
historical tracking and analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from truthound.checkpoint.actions.base import (
    ActionConfig,
    ActionResult,
    ActionStatus,
    BaseAction,
    NotifyCondition,
)

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import CheckpointResult


@dataclass
class StoreResultConfig(ActionConfig):
    """Configuration for store result action.

    Attributes:
        store_path: Path to store results (local directory or URI).
        store_type: Type of store ("file", "s3", "gcs", "database").
        format: Serialization format ("json", "yaml", "parquet").
        partition_by: Partition results by ("date", "checkpoint", "status").
        retention_days: Number of days to retain results (0 = forever).
        include_validation_details: Include full validation issue details.
        compress: Whether to compress stored files.
    """

    store_path: str | Path = "./truthound_results"
    store_type: str = "file"
    format: str = "json"
    partition_by: str = "date"
    retention_days: int = 0
    include_validation_details: bool = True
    compress: bool = False
    notify_on: NotifyCondition | str = NotifyCondition.ALWAYS


class StoreValidationResult(BaseAction[StoreResultConfig]):
    """Action to store validation results for historical tracking.

    This action saves checkpoint results to a persistent store,
    enabling historical analysis, trend tracking, and auditing.

    Example:
        >>> action = StoreValidationResult(
        ...     store_path="./results",
        ...     partition_by="date",
        ...     format="json",
        ... )
        >>> result = action.execute(checkpoint_result)
    """

    action_type = "store_result"

    @classmethod
    def _default_config(cls) -> StoreResultConfig:
        return StoreResultConfig()

    def _execute(self, checkpoint_result: "CheckpointResult") -> ActionResult:
        """Store the validation result."""
        config = self._config
        store_path = Path(config.store_path)

        # Determine output path based on partitioning
        output_path = self._get_output_path(store_path, checkpoint_result)

        # Prepare result data
        result_data = self._prepare_result_data(checkpoint_result)

        # Store based on type
        if config.store_type == "file":
            stored_path = self._store_to_file(output_path, result_data)
        elif config.store_type == "s3":
            stored_path = self._store_to_s3(output_path, result_data)
        elif config.store_type == "gcs":
            stored_path = self._store_to_gcs(output_path, result_data)
        else:
            stored_path = self._store_to_file(output_path, result_data)

        return ActionResult(
            action_name=self.name,
            action_type=self.action_type,
            status=ActionStatus.SUCCESS,
            message=f"Result stored to {stored_path}",
            details={
                "stored_path": str(stored_path),
                "store_type": config.store_type,
                "format": config.format,
                "size_bytes": self._get_file_size(stored_path),
            },
        )

    def _get_output_path(
        self,
        base_path: Path,
        checkpoint_result: "CheckpointResult",
    ) -> Path:
        """Determine output path based on partitioning strategy."""
        config = self._config
        run_time = checkpoint_result.run_time

        if config.partition_by == "date":
            partition = run_time.strftime("%Y/%m/%d")
        elif config.partition_by == "checkpoint":
            partition = checkpoint_result.checkpoint_name
        elif config.partition_by == "status":
            partition = checkpoint_result.status.value
        else:
            partition = ""

        # Create filename with run_id
        filename = f"{checkpoint_result.run_id}.{config.format}"
        if config.compress:
            filename += ".gz"

        return base_path / partition / filename

    def _prepare_result_data(self, checkpoint_result: "CheckpointResult") -> dict[str, Any]:
        """Prepare result data for storage."""
        data = checkpoint_result.to_dict()

        if not self._config.include_validation_details:
            # Remove detailed validation results to reduce size
            if "validation_result" in data:
                validation = data["validation_result"]
                if "results" in validation:
                    # Keep only summary, not individual results
                    data["validation_result"]["results"] = []

        return data

    def _store_to_file(self, output_path: Path, data: dict[str, Any]) -> Path:
        """Store result to local filesystem."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        config = self._config

        if config.format == "json":
            content = json.dumps(data, indent=2, default=str)
            if config.compress:
                import gzip
                with gzip.open(output_path, "wt", encoding="utf-8") as f:
                    f.write(content)
            else:
                output_path.write_text(content)
        elif config.format == "yaml":
            try:
                import yaml
                content = yaml.dump(data, default_flow_style=False)
                output_path.write_text(content)
            except ImportError:
                # Fall back to JSON
                content = json.dumps(data, indent=2, default=str)
                output_path = output_path.with_suffix(".json")
                output_path.write_text(content)

        return output_path

    def _store_to_s3(self, output_path: Path, data: dict[str, Any]) -> str:
        """Store result to AWS S3."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 storage. "
                "Install with: pip install boto3"
            )

        # Parse S3 path
        path_str = str(self._config.store_path)
        if path_str.startswith("s3://"):
            path_str = path_str[5:]
        parts = path_str.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        key = f"{prefix}/{output_path}" if prefix else str(output_path)
        key = key.lstrip("/")

        # Upload
        s3 = boto3.client("s3")
        content = json.dumps(data, indent=2, default=str)
        s3.put_object(Bucket=bucket, Key=key, Body=content.encode())

        return f"s3://{bucket}/{key}"

    def _store_to_gcs(self, output_path: Path, data: dict[str, Any]) -> str:
        """Store result to Google Cloud Storage."""
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required for GCS storage. "
                "Install with: pip install google-cloud-storage"
            )

        # Parse GCS path
        path_str = str(self._config.store_path)
        if path_str.startswith("gs://"):
            path_str = path_str[5:]
        parts = path_str.split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        blob_name = f"{prefix}/{output_path}" if prefix else str(output_path)
        blob_name = blob_name.lstrip("/")

        # Upload
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        content = json.dumps(data, indent=2, default=str)
        blob.upload_from_string(content)

        return f"gs://{bucket_name}/{blob_name}"

    def _get_file_size(self, path: Path | str) -> int:
        """Get file size if local, otherwise return 0."""
        if isinstance(path, Path) and path.exists():
            return path.stat().st_size
        return 0

    def validate_config(self) -> list[str]:
        """Validate configuration."""
        errors = []

        if self._config.store_type not in ("file", "s3", "gcs", "database"):
            errors.append(f"Invalid store_type: {self._config.store_type}")

        if self._config.format not in ("json", "yaml", "parquet"):
            errors.append(f"Invalid format: {self._config.format}")

        if self._config.partition_by not in ("date", "checkpoint", "status", ""):
            errors.append(f"Invalid partition_by: {self._config.partition_by}")

        return errors
