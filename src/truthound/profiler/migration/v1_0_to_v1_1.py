"""V1.0 to V1.1 Schema Migration.

Implements the migration logic for upgrading profile data from
schema version 1.0 to 1.1, and downgrading back.

Changes in v1.1:
- Added 'metadata' section with profiling info
- Renamed 'null_count' to 'missing_count' in columns
- Added 'data_quality_score' to each column
- Added 'semantic_type' field for ML-inferred types
- Restructured 'patterns_found' to include confidence
- Added 'recommendations' section
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from truthound.profiler.migration.base import (
    Migration,
    MigrationDirection,
    MigrationError,
    MigrationResult,
    SchemaVersion,
    migration_registry,
)


logger = logging.getLogger(__name__)


class V1_0_to_V1_1_Migration(Migration):
    """Migration from schema v1.0 to v1.1.

    This migration handles the following changes:

    Additions:
    - metadata: Object containing profiling metadata
    - columns[].data_quality_score: Per-column quality score
    - columns[].semantic_type: ML-inferred semantic type
    - recommendations: List of improvement recommendations

    Renames:
    - columns[].null_count -> columns[].missing_count

    Restructures:
    - patterns_found: Now includes confidence scores

    Example v1.0:
        {
            "schema_version": "1.0",
            "name": "dataset",
            "columns": [
                {"name": "id", "null_count": 0, "type": "integer"}
            ],
            "patterns_found": ["email", "phone"]
        }

    Example v1.1:
        {
            "schema_version": "1.1",
            "metadata": {
                "profiler_version": "0.2.0",
                "created_at": "2024-01-01T00:00:00",
                "migrated_from": "1.0"
            },
            "name": "dataset",
            "columns": [
                {
                    "name": "id",
                    "missing_count": 0,
                    "type": "integer",
                    "semantic_type": null,
                    "data_quality_score": 1.0
                }
            ],
            "patterns_found": [
                {"pattern": "email", "confidence": 1.0},
                {"pattern": "phone", "confidence": 1.0}
            ],
            "recommendations": []
        }
    """

    from_version = SchemaVersion(1, 0)
    to_version = SchemaVersion(1, 1)
    description = "Add metadata, semantic types, quality scores, and recommendations"
    reversible = True

    # Default values for new fields
    DEFAULT_QUALITY_SCORE = 1.0
    DEFAULT_CONFIDENCE = 1.0

    def upgrade(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Upgrade from v1.0 to v1.1.

        Args:
            data: Profile data in v1.0 format

        Returns:
            Profile data in v1.1 format
        """
        result = data.copy()

        # 1. Add metadata section
        result["metadata"] = self._create_metadata(data)

        # 2. Migrate columns
        if "columns" in result:
            result["columns"] = [
                self._upgrade_column(col) for col in result["columns"]
            ]

        # 3. Restructure patterns_found
        if "patterns_found" in result:
            result["patterns_found"] = self._upgrade_patterns(result["patterns_found"])

        # 4. Add recommendations section if missing
        if "recommendations" not in result:
            result["recommendations"] = []

        # 5. Update schema version
        result["schema_version"] = str(self.to_version)

        logger.debug(f"Upgraded profile from v1.0 to v1.1")
        return result

    def downgrade(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Downgrade from v1.1 to v1.0.

        Args:
            data: Profile data in v1.1 format

        Returns:
            Profile data in v1.0 format
        """
        result = data.copy()

        # 1. Remove metadata section
        result.pop("metadata", None)

        # 2. Migrate columns back
        if "columns" in result:
            result["columns"] = [
                self._downgrade_column(col) for col in result["columns"]
            ]

        # 3. Revert patterns_found structure
        if "patterns_found" in result:
            result["patterns_found"] = self._downgrade_patterns(result["patterns_found"])

        # 4. Remove recommendations
        result.pop("recommendations", None)

        # 5. Update schema version
        result["schema_version"] = str(self.from_version)

        logger.debug(f"Downgraded profile from v1.1 to v1.0")
        return result

    def validate_upgrade(self, data: Dict[str, Any]) -> List[str]:
        """Validate data after upgrade to v1.1.

        Args:
            data: Upgraded data

        Returns:
            List of validation errors
        """
        errors = []

        # Check metadata
        if "metadata" not in data:
            errors.append("Missing 'metadata' section")
        else:
            metadata = data["metadata"]
            if "profiler_version" not in metadata:
                errors.append("Missing 'profiler_version' in metadata")
            if "created_at" not in metadata:
                errors.append("Missing 'created_at' in metadata")

        # Check columns
        if "columns" in data:
            for i, col in enumerate(data["columns"]):
                if "missing_count" not in col:
                    errors.append(f"Column {i}: missing 'missing_count'")
                if "null_count" in col:
                    errors.append(f"Column {i}: 'null_count' should be renamed to 'missing_count'")

        # Check patterns format
        if "patterns_found" in data:
            for i, pattern in enumerate(data["patterns_found"]):
                if isinstance(pattern, str):
                    errors.append(f"Pattern {i}: should be object with 'pattern' and 'confidence'")

        return errors

    def validate_downgrade(self, data: Dict[str, Any]) -> List[str]:
        """Validate data after downgrade to v1.0.

        Args:
            data: Downgraded data

        Returns:
            List of validation errors
        """
        errors = []

        # Check no v1.1 fields remain
        if "metadata" in data:
            errors.append("'metadata' should not exist in v1.0")

        if "recommendations" in data:
            errors.append("'recommendations' should not exist in v1.0")

        # Check columns
        if "columns" in data:
            for i, col in enumerate(data["columns"]):
                if "missing_count" in col:
                    errors.append(f"Column {i}: 'missing_count' should be 'null_count' in v1.0")
                if "semantic_type" in col:
                    errors.append(f"Column {i}: 'semantic_type' should not exist in v1.0")
                if "data_quality_score" in col:
                    errors.append(f"Column {i}: 'data_quality_score' should not exist in v1.0")

        return errors

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _create_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata section for v1.1.

        Args:
            data: Original data

        Returns:
            Metadata dictionary
        """
        return {
            "profiler_version": "0.2.0",
            "created_at": datetime.now().isoformat(),
            "migrated_from": str(self.from_version),
            "original_name": data.get("name", "unknown"),
        }

    def _upgrade_column(self, column: Dict[str, Any]) -> Dict[str, Any]:
        """Upgrade a column definition from v1.0 to v1.1.

        Args:
            column: Column in v1.0 format

        Returns:
            Column in v1.1 format
        """
        result = column.copy()

        # Rename null_count to missing_count
        if "null_count" in result:
            result["missing_count"] = result.pop("null_count")

        # Add data_quality_score if missing
        if "data_quality_score" not in result:
            # Calculate from existing data if possible
            result["data_quality_score"] = self._calculate_column_quality(result)

        # Add semantic_type if missing
        if "semantic_type" not in result:
            result["semantic_type"] = self._infer_semantic_type(result)

        return result

    def _downgrade_column(self, column: Dict[str, Any]) -> Dict[str, Any]:
        """Downgrade a column definition from v1.1 to v1.0.

        Args:
            column: Column in v1.1 format

        Returns:
            Column in v1.0 format
        """
        result = column.copy()

        # Rename missing_count back to null_count
        if "missing_count" in result:
            result["null_count"] = result.pop("missing_count")

        # Remove v1.1 fields
        result.pop("data_quality_score", None)
        result.pop("semantic_type", None)

        return result

    def _upgrade_patterns(
        self,
        patterns: List[Any],
    ) -> List[Dict[str, Any]]:
        """Upgrade patterns_found from v1.0 to v1.1 format.

        Args:
            patterns: List of patterns (strings in v1.0)

        Returns:
            List of pattern objects with confidence
        """
        result = []

        for pattern in patterns:
            if isinstance(pattern, str):
                # Convert string to object
                result.append({
                    "pattern": pattern,
                    "confidence": self.DEFAULT_CONFIDENCE,
                    "migrated": True,
                })
            elif isinstance(pattern, dict):
                # Already in new format
                if "confidence" not in pattern:
                    pattern["confidence"] = self.DEFAULT_CONFIDENCE
                result.append(pattern)
            else:
                # Unknown format, try to convert
                result.append({
                    "pattern": str(pattern),
                    "confidence": self.DEFAULT_CONFIDENCE,
                    "migrated": True,
                })

        return result

    def _downgrade_patterns(
        self,
        patterns: List[Any],
    ) -> List[str]:
        """Downgrade patterns_found from v1.1 to v1.0 format.

        Args:
            patterns: List of pattern objects

        Returns:
            List of pattern strings
        """
        result = []

        for pattern in patterns:
            if isinstance(pattern, dict):
                result.append(pattern.get("pattern", str(pattern)))
            else:
                result.append(str(pattern))

        return result

    def _calculate_column_quality(self, column: Dict[str, Any]) -> float:
        """Calculate quality score for a column.

        Args:
            column: Column data

        Returns:
            Quality score (0.0 to 1.0)
        """
        scores = []

        # Completeness (from null/missing ratio)
        missing_count = column.get("missing_count", column.get("null_count", 0))
        row_count = column.get("row_count", 0)

        if row_count > 0:
            completeness = 1.0 - (missing_count / row_count)
            scores.append(completeness)
        else:
            scores.append(self.DEFAULT_QUALITY_SCORE)

        # Uniqueness (if available)
        if "unique_ratio" in column:
            # Higher uniqueness is generally better for IDs, not for categories
            scores.append(min(1.0, column["unique_ratio"] * 1.2))

        # Return average
        return sum(scores) / len(scores) if scores else self.DEFAULT_QUALITY_SCORE

    def _infer_semantic_type(self, column: Dict[str, Any]) -> Optional[str]:
        """Infer semantic type from column metadata.

        This provides a simple inference based on column name and type.
        More sophisticated inference would use the ML module.

        Args:
            column: Column data

        Returns:
            Inferred semantic type or None
        """
        name = column.get("name", "").lower()
        dtype = column.get("type", column.get("physical_type", "")).lower()

        # Name-based inference
        if "email" in name:
            return "email"
        if "phone" in name or "tel" in name:
            return "phone"
        if "url" in name or "link" in name or "website" in name:
            return "url"
        if "uuid" in name or "guid" in name:
            return "uuid"
        if "id" in name or name.endswith("_id"):
            return "identifier"
        if "date" in name or "time" in name:
            return "datetime"
        if "price" in name or "cost" in name or "amount" in name:
            return "currency"
        if "percent" in name or "ratio" in name or "rate" in name:
            return "percentage"
        if "status" in name or "type" in name or "category" in name:
            return "categorical"

        # Type-based inference
        if "int" in dtype:
            return "integer"
        if "float" in dtype or "double" in dtype:
            return "float"
        if "bool" in dtype:
            return "boolean"
        if "date" in dtype:
            return "datetime"

        return None


# =============================================================================
# Additional Migrations (Future)
# =============================================================================


class V1_1_to_V1_2_Migration(Migration):
    """Placeholder for future v1.1 to v1.2 migration.

    This demonstrates the pattern for adding new migrations.
    """

    from_version = SchemaVersion(1, 1)
    to_version = SchemaVersion(1, 2)
    description = "Future migration placeholder"
    reversible = True

    def upgrade(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Upgrade from v1.1 to v1.2."""
        result = data.copy()
        result["schema_version"] = str(self.to_version)
        # Future changes would go here
        return result

    def downgrade(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Downgrade from v1.2 to v1.1."""
        result = data.copy()
        result["schema_version"] = str(self.from_version)
        # Reverse future changes here
        return result

    def validate_upgrade(self, data: Dict[str, Any]) -> List[str]:
        """Validate upgrade to v1.2."""
        return []

    def validate_downgrade(self, data: Dict[str, Any]) -> List[str]:
        """Validate downgrade to v1.1."""
        return []


# =============================================================================
# Register Migrations
# =============================================================================


def register_migrations() -> None:
    """Register all migrations with the global registry."""
    migration_registry.register(V1_0_to_V1_1_Migration())
    migration_registry.register(V1_1_to_V1_2_Migration())


# Auto-register on import
register_migrations()
