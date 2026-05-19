"""Private dataset repository primitives for future Truthound Depot layers.

This package is intentionally private. It provides internal bootstrap contracts
without exposing a public ``truthound.datasets`` or ``truthound.depot`` surface.
"""

from __future__ import annotations

from truthound._datasets.bundles import (
    DatasetDiffBundle,
    DatasetEvidenceInputPayload,
    DatasetSnapshotBundle,
    QualityGateBundle,
    restore_dataset_artifact,
)
from truthound._datasets.contracts import (
    TRUTHOUND_DATASET_ARTIFACT_SCHEMA_VERSION,
    TRUTHOUND_DATASET_FINGERPRINT_POLICY_VERSION,
    TRUTHOUND_DATASET_SAMPLING_POLICY_VERSION,
    DatasetArtifactContractError,
    DatasetArtifactEnvelope,
    DatasetArtifactVersionError,
)
from truthound._datasets.diffing import diff_datasets, diff_fingerprints
from truthound._datasets.fingerprinting import (
    DatasetFingerprintOptions,
    fingerprint_dataset,
    normalize_dataset_input,
)
from truthound._datasets.gates import (
    QualityGateContext,
    QualityGateDisposition,
    QualityGatePolicy,
    evaluate_quality_gate,
)
from truthound._datasets.primitives import (
    DatasetArtifactType,
    DatasetAssetManifest,
    DatasetAssetType,
    DatasetDiff,
    DatasetDiffCategory,
    DatasetFingerprint,
    DatasetSnapshotManifest,
    QualityGateResult,
    QualityGateStatus,
    QualityGateType,
)
from truthound._datasets.redaction import assert_dataset_artifact_safe

__all__ = [
    "TRUTHOUND_DATASET_ARTIFACT_SCHEMA_VERSION",
    "TRUTHOUND_DATASET_FINGERPRINT_POLICY_VERSION",
    "TRUTHOUND_DATASET_SAMPLING_POLICY_VERSION",
    "DatasetArtifactContractError",
    "DatasetArtifactEnvelope",
    "DatasetArtifactType",
    "DatasetArtifactVersionError",
    "DatasetAssetManifest",
    "DatasetAssetType",
    "DatasetDiff",
    "DatasetDiffBundle",
    "DatasetDiffCategory",
    "DatasetEvidenceInputPayload",
    "DatasetFingerprintOptions",
    "DatasetFingerprint",
    "DatasetSnapshotBundle",
    "DatasetSnapshotManifest",
    "QualityGateContext",
    "QualityGateDisposition",
    "QualityGatePolicy",
    "QualityGateBundle",
    "QualityGateResult",
    "QualityGateStatus",
    "QualityGateType",
    "assert_dataset_artifact_safe",
    "diff_datasets",
    "diff_fingerprints",
    "evaluate_quality_gate",
    "fingerprint_dataset",
    "normalize_dataset_input",
    "restore_dataset_artifact",
]
