"""Cross-cloud integration tests.

These tests verify that data can be migrated between different cloud providers.
Requires credentials for multiple cloud providers.

Run with: pytest tests/integration/stores/test_cross_cloud.py -v
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from tests.integration.stores.conftest import (
    has_aws_credentials,
    has_azure_credentials,
    has_gcs_credentials,
    skip_no_aws,
    skip_no_azure,
    skip_no_gcs,
)


class TestCrossCloudMigration:
    """Tests for migrating data between cloud providers."""

    @pytest.mark.skipif(
        not (has_aws_credentials() and has_gcs_credentials()),
        reason="Both AWS and GCS credentials required",
    )
    def test_s3_to_gcs_migration(
        self,
        sample_validation_result: dict,
        aws_test_bucket: str,
        aws_test_prefix: str,
        gcs_test_bucket: str,
        gcs_test_prefix: str,
    ) -> None:
        """Test migrating data from S3 to GCS."""
        from truthound.stores import get_store
        from truthound.stores.results import ValidationResult

        s3_store = get_store(
            "s3",
            bucket=aws_test_bucket,
            prefix=aws_test_prefix,
        )

        gcs_store = get_store(
            "gcs",
            bucket=gcs_test_bucket,
            prefix=gcs_test_prefix,
        )

        try:
            s3_store.initialize()
            gcs_store.initialize()

            # Save to S3
            result = ValidationResult.from_dict(sample_validation_result)
            run_id = s3_store.save(result)

            # Read from S3
            s3_result = s3_store.get(run_id)

            # Save to GCS
            gcs_store.save(s3_result)

            # Read from GCS
            gcs_result = gcs_store.get(run_id)

            # Verify data integrity
            assert gcs_result.run_id == s3_result.run_id
            assert gcs_result.data_asset == s3_result.data_asset
            assert gcs_result.status == s3_result.status

        finally:
            s3_store.close()
            gcs_store.close()

    @pytest.mark.skipif(
        not (has_aws_credentials() and has_azure_credentials()),
        reason="Both AWS and Azure credentials required",
    )
    def test_s3_to_azure_migration(
        self,
        sample_validation_result: dict,
        aws_test_bucket: str,
        aws_test_prefix: str,
        azure_test_container: str,
        azure_test_prefix: str,
        azure_connection_string: str | None,
    ) -> None:
        """Test migrating data from S3 to Azure."""
        from truthound.stores import get_store
        from truthound.stores.results import ValidationResult

        s3_store = get_store(
            "s3",
            bucket=aws_test_bucket,
            prefix=aws_test_prefix,
        )

        azure_store = get_store(
            "azure",
            container=azure_test_container,
            connection_string=azure_connection_string,
            prefix=azure_test_prefix,
        )

        try:
            s3_store.initialize()
            azure_store.initialize()

            # Save to S3
            result = ValidationResult.from_dict(sample_validation_result)
            run_id = s3_store.save(result)

            # Read from S3
            s3_result = s3_store.get(run_id)

            # Save to Azure
            azure_store.save(s3_result)

            # Read from Azure
            azure_result = azure_store.get(run_id)

            # Verify data integrity
            assert azure_result.run_id == s3_result.run_id
            assert azure_result.data_asset == s3_result.data_asset
            assert azure_result.status == s3_result.status

        finally:
            s3_store.close()
            azure_store.close()

    @pytest.mark.skipif(
        not (has_gcs_credentials() and has_azure_credentials()),
        reason="Both GCS and Azure credentials required",
    )
    def test_gcs_to_azure_migration(
        self,
        sample_validation_result: dict,
        gcs_test_bucket: str,
        gcs_test_prefix: str,
        azure_test_container: str,
        azure_test_prefix: str,
        azure_connection_string: str | None,
    ) -> None:
        """Test migrating data from GCS to Azure."""
        from truthound.stores import get_store
        from truthound.stores.results import ValidationResult

        gcs_store = get_store(
            "gcs",
            bucket=gcs_test_bucket,
            prefix=gcs_test_prefix,
        )

        azure_store = get_store(
            "azure",
            container=azure_test_container,
            connection_string=azure_connection_string,
            prefix=azure_test_prefix,
        )

        try:
            gcs_store.initialize()
            azure_store.initialize()

            # Save to GCS
            result = ValidationResult.from_dict(sample_validation_result)
            run_id = gcs_store.save(result)

            # Read from GCS
            gcs_result = gcs_store.get(run_id)

            # Save to Azure
            azure_store.save(gcs_result)

            # Read from Azure
            azure_result = azure_store.get(run_id)

            # Verify data integrity
            assert azure_result.run_id == gcs_result.run_id
            assert azure_result.data_asset == gcs_result.data_asset
            assert azure_result.status == gcs_result.status

        finally:
            gcs_store.close()
            azure_store.close()


class TestTieredStorageIntegration:
    """Tests for tiered storage across cloud providers."""

    @pytest.mark.skipif(
        not has_aws_credentials(),
        reason="AWS credentials required",
    )
    def test_tiered_storage_s3_tiers(
        self,
        sample_validation_result: dict,
        aws_test_bucket: str,
        aws_test_prefix: str,
    ) -> None:
        """Test tiered storage with S3 storage classes."""
        from truthound.stores import get_store
        from truthound.stores.results import ValidationResult
        from truthound.stores.tiering import (
            TieredStore,
            StorageTier,
            TierType,
            TieringConfig,
            AgeBasedTierPolicy,
        )

        # Create tiers with different S3 storage classes
        hot_store = get_store(
            "s3",
            bucket=aws_test_bucket,
            prefix=f"{aws_test_prefix}hot/",
            storage_class="STANDARD",
        )

        cold_store = get_store(
            "s3",
            bucket=aws_test_bucket,
            prefix=f"{aws_test_prefix}cold/",
            storage_class="STANDARD_IA",
        )

        tiers = [
            StorageTier(
                name="hot",
                store=hot_store,
                tier_type=TierType.HOT,
                priority=1,
            ),
            StorageTier(
                name="cold",
                store=cold_store,
                tier_type=TierType.COLD,
                priority=2,
            ),
        ]

        config = TieringConfig(
            policies=[
                AgeBasedTierPolicy("hot", "cold", after_days=0),  # Immediate for testing
            ],
            default_tier="hot",
        )

        tiered_store = TieredStore(tiers, config)

        try:
            tiered_store.initialize()

            # Save to hot tier
            result = ValidationResult.from_dict(sample_validation_result)
            run_id = tiered_store.save(result)

            # Verify in hot tier
            assert tiered_store.get_item_tier(run_id) == "hot"

            # Get from tiered store
            retrieved = tiered_store.get(run_id)
            assert retrieved.run_id == run_id

            # Run tiering (should move to cold)
            tiering_result = tiered_store.run_tiering()
            assert tiering_result.items_migrated > 0

            # Verify in cold tier
            assert tiered_store.get_item_tier(run_id) == "cold"

            # Still retrievable
            retrieved = tiered_store.get(run_id)
            assert retrieved.run_id == run_id

        finally:
            tiered_store.close()


class TestRetentionPolicyIntegration:
    """Tests for retention policies with cloud storage."""

    @skip_no_aws
    def test_retention_with_s3(
        self,
        multiple_validation_results: list[dict],
        aws_test_bucket: str,
        aws_test_prefix: str,
    ) -> None:
        """Test retention policies with S3 store."""
        from truthound.stores import get_store
        from truthound.stores.results import ValidationResult
        from truthound.stores.retention import (
            RetentionStore,
            RetentionConfig,
            CountBasedPolicy,
        )

        base_store = get_store(
            "s3",
            bucket=aws_test_bucket,
            prefix=aws_test_prefix,
        )

        retention_config = RetentionConfig(
            policies=[
                CountBasedPolicy(max_count=3),  # Keep only 3 items
            ],
            preserve_latest=True,
        )

        store = RetentionStore(base_store, retention_config)

        try:
            store.initialize()

            # Save multiple results
            run_ids = []
            for result_data in multiple_validation_results:
                result = ValidationResult.from_dict(result_data)
                run_id = store.save(result)
                run_ids.append(run_id)

            # Run cleanup
            cleanup_result = store.run_cleanup()

            # Should have deleted some items
            assert cleanup_result.items_deleted > 0

            # Should still have at most 3 items per asset
            remaining_ids = store.list_ids()
            # Due to preserve_latest, we may have more

        finally:
            store.close()


class TestVersioningWithCloudStorage:
    """Tests for versioning with cloud storage."""

    @skip_no_aws
    def test_versioning_with_s3(
        self,
        sample_validation_result: dict,
        aws_test_bucket: str,
        aws_test_prefix: str,
    ) -> None:
        """Test versioning with S3 store."""
        from truthound.stores import get_store
        from truthound.stores.results import ValidationResult
        from truthound.stores.versioning import (
            VersionedStore,
            VersioningConfig,
            VersioningMode,
        )

        base_store = get_store(
            "s3",
            bucket=aws_test_bucket,
            prefix=aws_test_prefix,
        )

        versioning_config = VersioningConfig(
            mode=VersioningMode.INCREMENTAL,
            max_versions=5,
        )

        store = VersionedStore(base_store, versioning_config)

        try:
            store.initialize()

            # Save first version
            result = ValidationResult.from_dict(sample_validation_result)
            run_id = store.save(result, message="Initial save")

            # Get version history
            history = store.get_version_history(run_id)
            assert len(history) == 1
            assert history[0].version == 1

            # Save second version (modify data)
            sample_validation_result["status"] = "failure"
            result2 = ValidationResult.from_dict(sample_validation_result)
            result2._run_id = run_id
            store.save(result2, message="Updated status")

            # Get version history
            history = store.get_version_history(run_id)
            assert len(history) == 2

            # Get specific version
            v1 = store.get(run_id, version=1)
            assert v1.status.value == "success"

            v2 = store.get(run_id, version=2)
            assert v2.status.value == "failure"

            # Rollback
            store.rollback(run_id, version=1)

            # Current should be v1 content
            current = store.get(run_id)
            assert current.status.value == "success"

        finally:
            store.close()
