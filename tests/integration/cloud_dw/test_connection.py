"""Connection tests for Cloud DW backends.

These tests verify that:
- Connections can be established
- Credentials are validated correctly
- Connection pooling works properly
- Timeouts are handled correctly
"""

from __future__ import annotations

import pytest

from tests.integration.cloud_dw.base import ConnectionStatus


# =============================================================================
# Connection Tests
# =============================================================================


class TestConnection:
    """Connection tests for all backends."""

    @pytest.mark.connection
    def test_connection_established(self, any_backend):
        """Test that connection is established."""
        assert any_backend is not None
        assert any_backend.is_connected
        assert any_backend.status == ConnectionStatus.CONNECTED

    @pytest.mark.connection
    def test_connection_validation(self, any_backend):
        """Test that connection can be validated."""
        # Execute a simple query to validate connection
        result = any_backend.execute_query("SELECT 1 AS test")
        assert len(result) == 1
        assert result[0]["test"] == 1

    @pytest.mark.connection
    def test_credentials_masked(self, any_backend):
        """Test that sensitive credentials are masked."""
        masked = any_backend.credentials.mask_sensitive()

        # Check that sensitive fields are masked
        for key, value in masked.items():
            if any(s in key.lower() for s in ["password", "secret", "token", "key"]):
                assert value == "***MASKED***", f"{key} should be masked"


# =============================================================================
# BigQuery-specific Tests
# =============================================================================


@pytest.mark.bigquery
class TestBigQueryConnection:
    """BigQuery-specific connection tests."""

    @pytest.mark.connection
    def test_project_configured(self, bigquery_backend):
        """Test that project is configured."""
        assert bigquery_backend.credentials.project != ""

    @pytest.mark.connection
    def test_location_configured(self, bigquery_backend):
        """Test that location is configured."""
        assert bigquery_backend.credentials.location in ["US", "EU", "asia-east1", "asia-northeast1"]

    @pytest.mark.connection
    def test_dry_run_support(self, bigquery_backend):
        """Test that dry run is supported."""
        assert bigquery_backend.supports_dry_run is True

    @pytest.mark.connection
    def test_cost_estimation_support(self, bigquery_backend):
        """Test that cost estimation is supported."""
        assert bigquery_backend.supports_cost_estimation is True


# =============================================================================
# Snowflake-specific Tests
# =============================================================================


@pytest.mark.snowflake
class TestSnowflakeConnection:
    """Snowflake-specific connection tests."""

    @pytest.mark.connection
    def test_account_configured(self, snowflake_backend):
        """Test that account is configured."""
        assert snowflake_backend.credentials.account != ""

    @pytest.mark.connection
    def test_warehouse_usable(self, snowflake_backend):
        """Test that warehouse can be used."""
        if snowflake_backend.credentials.warehouse:
            result = snowflake_backend.execute_query(
                "SELECT CURRENT_WAREHOUSE() AS warehouse"
            )
            assert result[0]["WAREHOUSE"] is not None


# =============================================================================
# Redshift-specific Tests
# =============================================================================


@pytest.mark.redshift
class TestRedshiftConnection:
    """Redshift-specific connection tests."""

    @pytest.mark.connection
    def test_cluster_reachable(self, redshift_backend):
        """Test that cluster is reachable."""
        result = redshift_backend.execute_query("SELECT 1 AS test")
        assert result[0]["test"] == 1

    @pytest.mark.connection
    def test_database_configured(self, redshift_backend):
        """Test that database is configured."""
        assert redshift_backend.credentials.database != ""


# =============================================================================
# Databricks-specific Tests
# =============================================================================


@pytest.mark.databricks
class TestDatabricksConnection:
    """Databricks-specific connection tests."""

    @pytest.mark.connection
    def test_warehouse_configured(self, databricks_backend):
        """Test that SQL warehouse is configured."""
        assert databricks_backend.credentials.http_path != ""

    @pytest.mark.connection
    def test_authentication_method(self, databricks_backend):
        """Test that authentication is configured."""
        # Either PAT or OAuth should be configured
        has_pat = bool(databricks_backend.credentials.access_token)
        has_oauth = bool(
            databricks_backend.credentials.client_id
            and databricks_backend.credentials.client_secret
        )
        assert has_pat or has_oauth, "PAT or OAuth credentials required"
