"""Model storage and versioning for ReDoS ML models.

This module provides robust model persistence with versioning,
metadata tracking, and format compatibility checks.

Features:
    - Model serialization with pickle/joblib
    - Metadata tracking (version, training info, metrics)
    - Compatibility validation on load
    - Model registry for multiple model versions
    - Export/import in different formats

Example:
    >>> storage = ModelStorage(base_path="./models")
    >>> storage.save(model, name="redos_rf", version="1.0.0")
    >>> loaded_model = storage.load("redos_rf", version="1.0.0")
    >>> print(storage.list_models())
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from truthound.validators.security.redos.ml.base import (
    BaseReDoSModel,
    ModelConfig,
    ReDoSModelMetrics,
)


logger = logging.getLogger(__name__)


# Storage format version for compatibility checking
STORAGE_FORMAT_VERSION = "1.0"


@dataclass
class ModelMetadata:
    """Metadata for a saved model.

    Attributes:
        name: Model name/identifier
        version: Model version string
        model_type: Type of the model (e.g., "random_forest")
        model_version: Internal model version
        storage_format_version: Version of storage format
        created_at: When the model was saved
        trained_at: When the model was trained
        training_samples: Number of training samples
        metrics: Training metrics
        config: Model configuration
        feature_names: Names of features used
        file_hash: SHA-256 hash of the model file
        description: Optional description
        tags: Optional tags for categorization
        extra: Additional metadata
    """

    name: str
    version: str
    model_type: str
    model_version: str
    storage_format_version: str = STORAGE_FORMAT_VERSION
    created_at: datetime = field(default_factory=datetime.now)
    trained_at: Optional[datetime] = None
    training_samples: int = 0
    metrics: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    feature_names: List[str] = field(default_factory=list)
    file_hash: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "storage_format_version": self.storage_format_version,
            "created_at": self.created_at.isoformat(),
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "training_samples": self.training_samples,
            "metrics": self.metrics,
            "config": self.config,
            "feature_names": self.feature_names,
            "file_hash": self.file_hash,
            "description": self.description,
            "tags": self.tags,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            version=data["version"],
            model_type=data["model_type"],
            model_version=data["model_version"],
            storage_format_version=data.get("storage_format_version", "1.0"),
            created_at=datetime.fromisoformat(data["created_at"]),
            trained_at=(
                datetime.fromisoformat(data["trained_at"])
                if data.get("trained_at")
                else None
            ),
            training_samples=data.get("training_samples", 0),
            metrics=data.get("metrics"),
            config=data.get("config"),
            feature_names=data.get("feature_names", []),
            file_hash=data.get("file_hash", ""),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            extra=data.get("extra", {}),
        )

    @classmethod
    def from_model(
        cls,
        model: BaseReDoSModel,
        name: str,
        version: str,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> "ModelMetadata":
        """Create metadata from a trained model.

        Args:
            model: Trained model
            name: Model name
            version: Version string
            description: Optional description
            tags: Optional tags

        Returns:
            ModelMetadata instance
        """
        metrics = model.metrics.to_dict() if model.metrics else None

        return cls(
            name=name,
            version=version,
            model_type=model.name,
            model_version=model.version,
            trained_at=model.metrics.trained_at if model.metrics else None,
            training_samples=model.metrics.training_samples if model.metrics else 0,
            metrics=metrics,
            config=model.config.to_dict(),
            feature_names=model.feature_names,
            description=description,
            tags=tags or [],
        )


class ModelStorage:
    """Storage manager for ReDoS ML models.

    This class provides a unified interface for saving, loading,
    and managing trained ML models with proper versioning.

    The storage structure:
        base_path/
            model_name/
                v1.0.0/
                    model.pkl
                    metadata.json
                v1.1.0/
                    model.pkl
                    metadata.json
                latest -> v1.1.0  (symlink)

    Example:
        >>> storage = ModelStorage("./models")
        >>> storage.save(model, "redos_detector", "1.0.0")
        >>> model = storage.load("redos_detector")  # loads latest
        >>> models = storage.list_models()
    """

    MODEL_FILENAME = "model.pkl"
    METADATA_FILENAME = "metadata.json"

    def __init__(self, base_path: str | Path = "./models"):
        """Initialize the model storage.

        Args:
            base_path: Base directory for model storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: BaseReDoSModel,
        name: str,
        version: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        set_latest: bool = True,
    ) -> Path:
        """Save a trained model.

        Args:
            model: Trained model to save
            name: Model name/identifier
            version: Version string (e.g., "1.0.0")
            description: Optional description
            tags: Optional tags for categorization
            set_latest: Whether to set this as the latest version

        Returns:
            Path to the saved model directory

        Raises:
            ValueError: If model is not trained
        """
        if not model.is_trained:
            raise ValueError("Cannot save untrained model")

        # Create version directory
        version_dir = self.base_path / name / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)

        model_path = version_dir / self.MODEL_FILENAME
        metadata_path = version_dir / self.METADATA_FILENAME

        # Save model
        model.save(model_path)

        # Calculate file hash
        file_hash = self._calculate_hash(model_path)

        # Create and save metadata
        metadata = ModelMetadata.from_model(
            model=model,
            name=name,
            version=version,
            description=description,
            tags=tags,
        )
        metadata.file_hash = file_hash

        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Update latest symlink
        if set_latest:
            self._update_latest(name, version)

        logger.info(f"Model saved: {name} v{version} at {version_dir}")

        return version_dir

    def load(
        self,
        name: str,
        version: Optional[str] = None,
        verify_hash: bool = True,
    ) -> Tuple[BaseReDoSModel, ModelMetadata]:
        """Load a saved model.

        Args:
            name: Model name
            version: Specific version to load (None for latest)
            verify_hash: Whether to verify file hash

        Returns:
            Tuple of (loaded model, metadata)

        Raises:
            FileNotFoundError: If model not found
            ValueError: If hash verification fails
        """
        # Resolve version
        if version is None:
            version = self._get_latest_version(name)
            if version is None:
                raise FileNotFoundError(f"No versions found for model: {name}")

        version_dir = self.base_path / name / f"v{version}"
        model_path = version_dir / self.MODEL_FILENAME
        metadata_path = version_dir / self.METADATA_FILENAME

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {name} v{version}")

        # Load metadata
        with open(metadata_path) as f:
            metadata = ModelMetadata.from_dict(json.load(f))

        # Verify hash
        if verify_hash and metadata.file_hash:
            current_hash = self._calculate_hash(model_path)
            if current_hash != metadata.file_hash:
                raise ValueError(
                    f"Model file hash mismatch. File may be corrupted: {model_path}"
                )

        # Import model factory here to avoid circular import
        from truthound.validators.security.redos.ml.models import create_model

        # Create and load model
        config = ModelConfig.from_dict(metadata.config) if metadata.config else None
        model = create_model(metadata.model_type, config)
        model.load(model_path)

        logger.info(f"Model loaded: {name} v{version}")

        return model, metadata

    def list_models(self) -> List[str]:
        """List all available model names.

        Returns:
            List of model names
        """
        models = []
        for item in self.base_path.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                models.append(item.name)
        return sorted(models)

    def list_versions(self, name: str) -> List[str]:
        """List all versions of a model.

        Args:
            name: Model name

        Returns:
            List of version strings (sorted)
        """
        model_dir = self.base_path / name
        if not model_dir.exists():
            return []

        versions = []
        for item in model_dir.iterdir():
            if item.is_dir() and item.name.startswith("v"):
                versions.append(item.name[1:])  # Remove 'v' prefix

        return sorted(versions, key=self._parse_version)

    def get_metadata(
        self, name: str, version: Optional[str] = None
    ) -> ModelMetadata:
        """Get metadata for a model.

        Args:
            name: Model name
            version: Specific version (None for latest)

        Returns:
            ModelMetadata
        """
        if version is None:
            version = self._get_latest_version(name)

        metadata_path = self.base_path / name / f"v{version}" / self.METADATA_FILENAME

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {name} v{version}")

        with open(metadata_path) as f:
            return ModelMetadata.from_dict(json.load(f))

    def delete(self, name: str, version: Optional[str] = None) -> None:
        """Delete a model or version.

        Args:
            name: Model name
            version: Specific version to delete (None to delete all)
        """
        if version is None:
            # Delete entire model
            model_dir = self.base_path / name
            if model_dir.exists():
                shutil.rmtree(model_dir)
                logger.info(f"Deleted model: {name}")
        else:
            # Delete specific version
            version_dir = self.base_path / name / f"v{version}"
            if version_dir.exists():
                shutil.rmtree(version_dir)
                logger.info(f"Deleted version: {name} v{version}")

                # Update latest if necessary
                remaining_versions = self.list_versions(name)
                if remaining_versions:
                    self._update_latest(name, remaining_versions[-1])

    def copy(
        self,
        name: str,
        new_name: str,
        version: Optional[str] = None,
    ) -> None:
        """Copy a model to a new name.

        Args:
            name: Source model name
            new_name: Destination model name
            version: Specific version to copy (None for all)
        """
        if version is None:
            # Copy all versions
            versions = self.list_versions(name)
        else:
            versions = [version]

        for ver in versions:
            src_dir = self.base_path / name / f"v{ver}"
            dst_dir = self.base_path / new_name / f"v{ver}"

            if src_dir.exists():
                shutil.copytree(src_dir, dst_dir)

                # Update metadata
                metadata_path = dst_dir / self.METADATA_FILENAME
                with open(metadata_path) as f:
                    metadata = json.load(f)
                metadata["name"] = new_name
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

        # Set latest
        if versions:
            self._update_latest(new_name, versions[-1])

        logger.info(f"Copied {name} to {new_name}")

    def export(
        self,
        name: str,
        output_path: str | Path,
        version: Optional[str] = None,
        include_metadata: bool = True,
    ) -> Path:
        """Export a model to a single file.

        Args:
            name: Model name
            output_path: Path for the exported file
            version: Specific version (None for latest)
            include_metadata: Whether to include metadata

        Returns:
            Path to exported file
        """
        if version is None:
            version = self._get_latest_version(name)

        version_dir = self.base_path / name / f"v{version}"
        output_path = Path(output_path)

        # Create archive
        shutil.make_archive(
            str(output_path.with_suffix("")),
            "zip",
            version_dir,
        )

        exported_path = output_path.with_suffix(".zip")
        logger.info(f"Exported {name} v{version} to {exported_path}")

        return exported_path

    def import_model(
        self,
        archive_path: str | Path,
        name: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Import a model from an archive.

        Args:
            archive_path: Path to the archive file
            name: Optional new name (uses original if None)

        Returns:
            Tuple of (model name, version)
        """
        archive_path = Path(archive_path)

        # Extract to temp directory
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.unpack_archive(archive_path, temp_dir)

            # Read metadata
            metadata_path = Path(temp_dir) / self.METADATA_FILENAME
            with open(metadata_path) as f:
                metadata = ModelMetadata.from_dict(json.load(f))

            model_name = name or metadata.name
            version = metadata.version

            # Copy to storage
            version_dir = self.base_path / model_name / f"v{version}"
            version_dir.mkdir(parents=True, exist_ok=True)

            for item in Path(temp_dir).iterdir():
                shutil.copy2(item, version_dir)

            # Update metadata name if changed
            if name:
                new_metadata_path = version_dir / self.METADATA_FILENAME
                with open(new_metadata_path) as f:
                    meta_dict = json.load(f)
                meta_dict["name"] = name
                with open(new_metadata_path, "w") as f:
                    json.dump(meta_dict, f, indent=2)

            self._update_latest(model_name, version)

        logger.info(f"Imported {model_name} v{version}")

        return model_name, version

    def _get_latest_version(self, name: str) -> Optional[str]:
        """Get the latest version of a model."""
        versions = self.list_versions(name)
        return versions[-1] if versions else None

    def _update_latest(self, name: str, version: str) -> None:
        """Update the 'latest' marker for a model."""
        model_dir = self.base_path / name
        latest_file = model_dir / "latest.txt"

        with open(latest_file, "w") as f:
            f.write(version)

    def _calculate_hash(self, path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def _parse_version(version: str) -> Tuple[int, ...]:
        """Parse version string for sorting."""
        parts = version.split(".")
        result = []
        for part in parts:
            try:
                result.append(int(part))
            except ValueError:
                result.append(0)
        return tuple(result)


# Convenience functions for single-file operations


def save_model(
    model: BaseReDoSModel,
    path: str | Path,
    include_metadata: bool = True,
) -> None:
    """Save a model to a single file.

    Simple convenience function for quick model saving without
    the full storage infrastructure.

    Args:
        model: Model to save
        path: Path to save to
        include_metadata: Whether to include metadata
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "model_data": model._save_model_data(),
        "config": model.config.to_dict(),
        "trained": model.is_trained,
        "feature_names": model.feature_names,
        "metrics": model.metrics.to_dict() if model.metrics else None,
        "name": model.name,
        "version": model.version,
    }

    if include_metadata:
        data["metadata"] = {
            "saved_at": datetime.now().isoformat(),
            "storage_format_version": STORAGE_FORMAT_VERSION,
        }

    with open(path, "wb") as f:
        pickle.dump(data, f)

    logger.info(f"Model saved to: {path}")


def load_model(path: str | Path) -> BaseReDoSModel:
    """Load a model from a single file.

    Args:
        path: Path to the saved model

    Returns:
        Loaded model
    """
    from truthound.validators.security.redos.ml.models import create_model

    path = Path(path)

    with open(path, "rb") as f:
        data = pickle.load(f)

    config = ModelConfig.from_dict(data["config"])
    model = create_model(data["name"], config)
    model._load_model_data(data.get("model_data", data.get("model", {})))
    model._trained = data["trained"]
    model._feature_names = data["feature_names"]

    logger.info(f"Model loaded from: {path}")

    return model
