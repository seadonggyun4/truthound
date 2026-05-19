"""Private dataset artifact contract bootstrap.

The models in this module are internal scaffolding for dataset repository
artifacts. They deliberately avoid product-specific Depot business state and
should not be exported from the root ``truthound`` public API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

from truthound._datasets._serialization import (
    DatasetArtifactContractError,
    DatasetArtifactVersionError,
    parse_datetime,
    require_field,
    require_mapping,
    require_non_empty_str,
    utc_now,
)
from truthound._datasets.redaction import assert_dataset_artifact_safe

TRUTHOUND_DATASET_ARTIFACT_SCHEMA_VERSION = "0.1"
TRUTHOUND_DATASET_FINGERPRINT_POLICY_VERSION = "0.1"
TRUTHOUND_DATASET_SAMPLING_POLICY_VERSION = "0.1"


@dataclass(frozen=True)
class DatasetArtifactEnvelope:
    """Versioned container for future dataset repository artifact payloads."""

    artifact_type: str
    payload: Mapping[str, Any]
    artifact_schema_version: str = TRUTHOUND_DATASET_ARTIFACT_SCHEMA_VERSION
    fingerprint_policy_version: str = TRUTHOUND_DATASET_FINGERPRINT_POLICY_VERSION
    sampling_policy_version: str = TRUTHOUND_DATASET_SAMPLING_POLICY_VERSION
    created_at: datetime = field(default_factory=utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "payload",
            require_mapping(self.payload, field_name="payload"),
        )
        object.__setattr__(
            self,
            "metadata",
            require_mapping(self.metadata, field_name="metadata"),
        )
        object.__setattr__(
            self,
            "artifact_type",
            require_non_empty_str(self.artifact_type, field_name="artifact_type"),
        )
        object.__setattr__(
            self,
            "created_at",
            parse_datetime(self.created_at, field_name="created_at"),
        )
        self.validate()

    def validate(self) -> None:
        """Validate version, shape, and redaction safety."""

        if self.artifact_schema_version != TRUTHOUND_DATASET_ARTIFACT_SCHEMA_VERSION:
            raise DatasetArtifactVersionError(
                f"Unsupported dataset artifact schema version: {self.artifact_schema_version!r}"
            )
        if self.fingerprint_policy_version != TRUTHOUND_DATASET_FINGERPRINT_POLICY_VERSION:
            raise DatasetArtifactVersionError(
                "Unsupported dataset fingerprint policy version: "
                f"{self.fingerprint_policy_version!r}"
            )
        if self.sampling_policy_version != TRUTHOUND_DATASET_SAMPLING_POLICY_VERSION:
            raise DatasetArtifactVersionError(
                f"Unsupported dataset sampling policy version: {self.sampling_policy_version!r}"
            )
        assert_dataset_artifact_safe(
            {
                "artifact_type": self.artifact_type,
                "payload": self.payload,
                "metadata": self.metadata,
            },
            label=f"dataset artifact {self.artifact_type}",
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable envelope payload."""

        self.validate()
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "artifact_type": self.artifact_type,
            "payload": dict(self.payload),
            "fingerprint_policy_version": self.fingerprint_policy_version,
            "sampling_policy_version": self.sampling_policy_version,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetArtifactEnvelope:
        """Restore a dataset artifact envelope from a mapping."""

        payload = require_mapping(data, field_name="envelope")
        return cls(
            artifact_schema_version=str(require_field(payload, "artifact_schema_version")),
            artifact_type=str(require_field(payload, "artifact_type")),
            payload=require_mapping(
                require_field(payload, "payload"),
                field_name="payload",
            ),
            fingerprint_policy_version=str(require_field(payload, "fingerprint_policy_version")),
            sampling_policy_version=str(require_field(payload, "sampling_policy_version")),
            created_at=parse_datetime(
                require_field(payload, "created_at"),
                field_name="created_at",
            ),
            metadata=require_mapping(
                payload.get("metadata", {}),
                field_name="metadata",
            ),
        )


__all__ = [
    "TRUTHOUND_DATASET_ARTIFACT_SCHEMA_VERSION",
    "TRUTHOUND_DATASET_FINGERPRINT_POLICY_VERSION",
    "TRUTHOUND_DATASET_SAMPLING_POLICY_VERSION",
    "DatasetArtifactContractError",
    "DatasetArtifactEnvelope",
    "DatasetArtifactVersionError",
]
