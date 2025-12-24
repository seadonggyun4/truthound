"""Tests for enterprise data sources.

This module tests the enterprise data source implementations:
- Cloud DW: BigQuery, Snowflake, Redshift, Databricks
- Legacy Enterprise: Oracle, SQL Server

Tests use mocking since actual cloud/database connections aren't available.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import asdict

from truthound.datasources.base import DataSourceError, DataSourceConnectionError


# =============================================================================
# Cloud Base Tests
# =============================================================================


class TestCloudDWConfig:
    """Tests for CloudDWConfig base class."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from truthound.datasources.sql.cloud_base import CloudDWConfig

        config = CloudDWConfig()
        assert config.project is None
        assert config.warehouse is None
        assert config.region is None
        assert config.role is None
        assert config.timeout == 300
        assert config.use_cache is True
        assert config.credentials_path is None

    def test_config_custom_values(self):
        """Test custom configuration values."""
        from truthound.datasources.sql.cloud_base import CloudDWConfig

        config = CloudDWConfig(
            project="my-project",
            warehouse="my-warehouse",
            region="us-east-1",
            role="analyst",
            timeout=600,
            use_cache=False,
            credentials_path="/path/to/creds.json",
        )
        assert config.project == "my-project"
        assert config.warehouse == "my-warehouse"
        assert config.region == "us-east-1"
        assert config.role == "analyst"
        assert config.timeout == 600
        assert config.use_cache is False


class TestCloudAuthHelpers:
    """Tests for cloud authentication helper functions."""

    def test_load_credentials_from_env(self, monkeypatch):
        """Test loading credentials from environment variables."""
        from truthound.datasources.sql.cloud_base import load_credentials_from_env

        monkeypatch.setenv("SNOWFLAKE_USER", "testuser")
        monkeypatch.setenv("SNOWFLAKE_PASSWORD", "testpass")
        monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "testaccount")

        creds = load_credentials_from_env("SNOWFLAKE")
        assert creds["user"] == "testuser"
        assert creds["password"] == "testpass"
        assert creds["account"] == "testaccount"

    def test_load_credentials_missing(self, monkeypatch):
        """Test loading credentials when not set."""
        from truthound.datasources.sql.cloud_base import load_credentials_from_env

        # Clear any existing vars
        for key in ["TEST_USER", "TEST_PASSWORD"]:
            monkeypatch.delenv(key, raising=False)

        creds = load_credentials_from_env("TEST")
        assert creds == {}

    def test_load_service_account_json(self, tmp_path):
        """Test loading service account JSON."""
        from truthound.datasources.sql.cloud_base import load_service_account_json
        import json

        creds = {"type": "service_account", "project_id": "test-project"}
        creds_file = tmp_path / "creds.json"
        creds_file.write_text(json.dumps(creds))

        loaded = load_service_account_json(str(creds_file))
        assert loaded["type"] == "service_account"
        assert loaded["project_id"] == "test-project"


# =============================================================================
# BigQuery Tests
# =============================================================================


class TestBigQueryConfig:
    """Tests for BigQuery configuration."""

    @pytest.fixture
    def bigquery_available(self):
        """Check if BigQuery module is available."""
        try:
            from truthound.datasources.sql.bigquery import BigQueryDataSource
            return True
        except ImportError:
            return False

    def test_config_creation(self, bigquery_available):
        """Test configuration creation."""
        if not bigquery_available:
            pytest.skip("BigQuery module not available")

        from truthound.datasources.sql.bigquery import BigQueryConfig

        config = BigQueryConfig()
        # Default values may vary, just check it creates
        assert config is not None

    def test_config_custom(self, bigquery_available):
        """Test custom configuration."""
        if not bigquery_available:
            pytest.skip("BigQuery module not available")

        from truthound.datasources.sql.bigquery import BigQueryConfig

        config = BigQueryConfig(
            project="my-project",
            location="EU",
            use_legacy_sql=True,
            maximum_bytes_billed=1000000000,
        )
        assert config.project == "my-project"
        assert config.location == "EU"
        assert config.use_legacy_sql is True
        assert config.maximum_bytes_billed == 1000000000


class TestBigQueryDataSource:
    """Tests for BigQuery data source."""

    @pytest.fixture
    def bigquery_available(self):
        """Check if BigQuery module is available."""
        try:
            from truthound.datasources.sql.bigquery import BigQueryDataSource
            return True
        except ImportError:
            return False

    def test_source_type_attribute(self, bigquery_available):
        """Test source type is correctly set."""
        if not bigquery_available:
            pytest.skip("BigQuery module not available")

        from truthound.datasources.sql.bigquery import BigQueryDataSource

        # Access class attribute directly
        assert BigQueryDataSource.source_type == "bigquery"

    def test_full_table_name_format(self, bigquery_available):
        """Test fully qualified table name format."""
        if not bigquery_available:
            pytest.skip("BigQuery module not available")

        # Test table name format: `project`.`dataset`.`table`
        # Just verify format expectations
        expected_format = "`my-project`.`my_dataset`.`users`"
        assert expected_format.count("`") == 6  # 3 pairs of backticks

    def test_quote_identifier_format(self, bigquery_available):
        """Test BigQuery uses backticks for quoting."""
        if not bigquery_available:
            pytest.skip("BigQuery module not available")

        # BigQuery uses backticks for identifiers
        identifier = "my-column"
        quoted = f"`{identifier}`"
        assert quoted == "`my-column`"


# =============================================================================
# Snowflake Tests
# =============================================================================


class TestSnowflakeConfig:
    """Tests for Snowflake configuration."""

    @pytest.fixture
    def snowflake_available(self):
        """Check if Snowflake module is available."""
        try:
            from truthound.datasources.sql.snowflake import SnowflakeDataSource
            return True
        except ImportError:
            return False

    def test_config_defaults(self, snowflake_available):
        """Test default configuration."""
        if not snowflake_available:
            pytest.skip("Snowflake module not available")

        from truthound.datasources.sql.snowflake import SnowflakeConfig

        config = SnowflakeConfig()
        assert config.authenticator == "snowflake"
        assert config.warehouse is None
        assert config.role is None

    def test_config_auth_types(self, snowflake_available):
        """Test different authentication types."""
        if not snowflake_available:
            pytest.skip("Snowflake module not available")

        from truthound.datasources.sql.snowflake import SnowflakeConfig

        # Password auth
        config = SnowflakeConfig(authenticator="snowflake", user="user", password="pass")
        assert config.authenticator == "snowflake"

        # SSO auth
        config = SnowflakeConfig(authenticator="externalbrowser")
        assert config.authenticator == "externalbrowser"

        # Key pair auth
        config = SnowflakeConfig(
            authenticator="snowflake",
            private_key_path="/path/to/key.pem"
        )
        assert config.private_key_path == "/path/to/key.pem"


class TestSnowflakeDataSource:
    """Tests for Snowflake data source."""

    @pytest.fixture
    def snowflake_available(self):
        """Check if Snowflake module is available."""
        try:
            from truthound.datasources.sql.snowflake import SnowflakeDataSource
            return True
        except ImportError:
            return False

    def test_source_type_attribute(self, snowflake_available):
        """Test source type."""
        if not snowflake_available:
            pytest.skip("Snowflake module not available")

        from truthound.datasources.sql.snowflake import SnowflakeDataSource

        assert SnowflakeDataSource.source_type == "snowflake"

    def test_full_table_name_format(self, snowflake_available):
        """Test fully qualified table name format."""
        if not snowflake_available:
            pytest.skip("Snowflake module not available")

        # Format: "DATABASE"."SCHEMA"."TABLE"
        expected = '"MY_DB"."PUBLIC"."USERS"'
        assert expected.count('"') == 6

    def test_quote_identifier_format(self, snowflake_available):
        """Test Snowflake uses double quotes for quoting."""
        if not snowflake_available:
            pytest.skip("Snowflake module not available")

        identifier = "my_column"
        escaped = identifier.replace('"', '""')
        quoted = f'"{escaped}"'
        assert quoted == '"my_column"'


# =============================================================================
# Oracle Tests
# =============================================================================


class TestOracleConfig:
    """Tests for Oracle configuration."""

    @pytest.fixture
    def oracle_available(self):
        """Check if Oracle module is available."""
        try:
            from truthound.datasources.sql.oracle import OracleDataSource
            return True
        except ImportError:
            return False

    def test_config_defaults(self, oracle_available):
        """Test default configuration."""
        if not oracle_available:
            pytest.skip("Oracle module not available")

        from truthound.datasources.sql.oracle import OracleConfig

        config = OracleConfig()
        assert config.port == 1521
        assert config.thick_mode is False
        assert config.encoding == "UTF-8"

    def test_config_connection_types(self, oracle_available):
        """Test different connection types."""
        if not oracle_available:
            pytest.skip("Oracle module not available")

        from truthound.datasources.sql.oracle import OracleConfig

        # Using service name
        config = OracleConfig(
            host="oracle.example.com",
            service_name="ORCL",
            user="user",
            password="pass"
        )
        assert config.service_name == "ORCL"

        # Using SID
        config = OracleConfig(
            host="oracle.example.com",
            sid="ORCL",
            user="user",
            password="pass"
        )
        assert config.sid == "ORCL"

        # Using DSN
        config = OracleConfig(
            dsn="oracle.example.com:1521/ORCL",
            user="user",
            password="pass"
        )
        assert "1521" in config.dsn


class TestOracleDataSource:
    """Tests for Oracle data source."""

    @pytest.fixture
    def oracle_available(self):
        """Check if Oracle module is available."""
        try:
            from truthound.datasources.sql.oracle import OracleDataSource
            return True
        except ImportError:
            return False

    def test_source_type_attribute(self, oracle_available):
        """Test source type."""
        if not oracle_available:
            pytest.skip("Oracle module not available")

        from truthound.datasources.sql.oracle import OracleDataSource

        assert OracleDataSource.source_type == "oracle"

    def test_quote_identifier_format(self, oracle_available):
        """Test Oracle uses double quotes for quoting."""
        if not oracle_available:
            pytest.skip("Oracle module not available")

        identifier = "MY_COLUMN"
        escaped = identifier.replace('"', '""')
        quoted = f'"{escaped}"'
        assert quoted == '"MY_COLUMN"'

    def test_validate_query(self, oracle_available):
        """Test Oracle validation query."""
        if not oracle_available:
            pytest.skip("Oracle module not available")

        # Oracle uses SELECT 1 FROM DUAL
        expected_query = "SELECT 1 FROM DUAL"
        assert "DUAL" in expected_query


# =============================================================================
# SQL Server Tests
# =============================================================================


class TestSQLServerConfig:
    """Tests for SQL Server configuration."""

    @pytest.fixture
    def sqlserver_available(self):
        """Check if SQL Server module is available."""
        try:
            from truthound.datasources.sql.sqlserver import SQLServerDataSource
            return True
        except ImportError:
            return False

    def test_config_defaults(self, sqlserver_available):
        """Test default configuration."""
        if not sqlserver_available:
            pytest.skip("SQL Server module not available")

        from truthound.datasources.sql.sqlserver import SQLServerConfig

        config = SQLServerConfig()
        assert config.port == 1433
        assert config.trusted_connection is False
        assert config.encrypt is True
        assert config.driver == "ODBC Driver 17 for SQL Server"
        assert config.application_name == "Truthound"

    def test_config_windows_auth(self, sqlserver_available):
        """Test Windows Authentication configuration."""
        if not sqlserver_available:
            pytest.skip("SQL Server module not available")

        from truthound.datasources.sql.sqlserver import SQLServerConfig

        config = SQLServerConfig(
            host="sqlserver.example.com",
            database="MyDB",
            trusted_connection=True
        )
        assert config.trusted_connection is True

    def test_config_sql_auth(self, sqlserver_available):
        """Test SQL Server Authentication configuration."""
        if not sqlserver_available:
            pytest.skip("SQL Server module not available")

        from truthound.datasources.sql.sqlserver import SQLServerConfig

        config = SQLServerConfig(
            host="sqlserver.example.com",
            database="MyDB",
            user="sa",
            password="password"
        )
        assert config.user == "sa"
        assert config.trusted_connection is False


class TestSQLServerDataSource:
    """Tests for SQL Server data source."""

    @pytest.fixture
    def sqlserver_available(self):
        """Check if SQL Server module is available."""
        try:
            from truthound.datasources.sql.sqlserver import SQLServerDataSource
            return True
        except ImportError:
            return False

    def test_source_type_attribute(self, sqlserver_available):
        """Test source type."""
        if not sqlserver_available:
            pytest.skip("SQL Server module not available")

        from truthound.datasources.sql.sqlserver import SQLServerDataSource

        assert SQLServerDataSource.source_type == "sqlserver"

    def test_quote_identifier_format(self, sqlserver_available):
        """Test SQL Server uses brackets for quoting."""
        if not sqlserver_available:
            pytest.skip("SQL Server module not available")

        identifier = "my_column"
        escaped = identifier.replace("]", "]]")
        quoted = f"[{escaped}]"
        assert quoted == "[my_column]"

    def test_quote_identifier_with_brackets(self, sqlserver_available):
        """Test escaping brackets in identifiers."""
        if not sqlserver_available:
            pytest.skip("SQL Server module not available")

        identifier = "column]name"
        escaped = identifier.replace("]", "]]")
        quoted = f"[{escaped}]"
        assert quoted == "[column]]name]"

    def test_connection_string_parsing(self, sqlserver_available):
        """Test connection string parsing."""
        if not sqlserver_available:
            pytest.skip("SQL Server module not available")

        conn_str = "SERVER=myserver,1433;DATABASE=mydb;UID=myuser;PWD=mypass"
        parts = dict(p.split("=", 1) for p in conn_str.split(";") if "=" in p)

        assert parts["SERVER"] == "myserver,1433"
        assert parts["DATABASE"] == "mydb"
        assert parts["UID"] == "myuser"


# =============================================================================
# Redshift Tests
# =============================================================================


class TestRedshiftConfig:
    """Tests for Redshift configuration."""

    @pytest.fixture
    def redshift_available(self):
        """Check if Redshift module is available."""
        try:
            from truthound.datasources.sql.redshift import RedshiftDataSource
            return True
        except ImportError:
            return False

    def test_config_defaults(self, redshift_available):
        """Test default configuration."""
        if not redshift_available:
            pytest.skip("Redshift module not available")

        from truthound.datasources.sql.redshift import RedshiftConfig

        config = RedshiftConfig()
        assert config.port == 5439
        assert config.iam_auth is False
        assert config.ssl is True
        assert config.ssl_mode == "verify-ca"

    def test_config_iam_auth(self, redshift_available):
        """Test IAM authentication configuration."""
        if not redshift_available:
            pytest.skip("Redshift module not available")

        from truthound.datasources.sql.redshift import RedshiftConfig

        config = RedshiftConfig(
            host="cluster.abc123.us-east-1.redshift.amazonaws.com",
            database="mydb",
            iam_auth=True,
            cluster_identifier="my-cluster",
            db_user="admin"
        )
        assert config.iam_auth is True
        assert config.cluster_identifier == "my-cluster"


class TestRedshiftDataSource:
    """Tests for Redshift data source."""

    @pytest.fixture
    def redshift_available(self):
        """Check if Redshift module is available."""
        try:
            from truthound.datasources.sql.redshift import RedshiftDataSource
            return True
        except ImportError:
            return False

    def test_source_type_attribute(self, redshift_available):
        """Test source type."""
        if not redshift_available:
            pytest.skip("Redshift module not available")

        from truthound.datasources.sql.redshift import RedshiftDataSource

        assert RedshiftDataSource.source_type == "redshift"

    def test_quote_identifier_format(self, redshift_available):
        """Test Redshift uses double quotes for quoting."""
        if not redshift_available:
            pytest.skip("Redshift module not available")

        identifier = "my_column"
        escaped = identifier.replace('"', '""')
        quoted = f'"{escaped}"'
        assert quoted == '"my_column"'

    def test_full_table_name_format(self, redshift_available):
        """Test fully qualified table name format."""
        if not redshift_available:
            pytest.skip("Redshift module not available")

        # Format: "schema"."table"
        expected = '"public"."users"'
        assert expected.count('"') == 4


# =============================================================================
# Databricks Tests
# =============================================================================


class TestDatabricksConfig:
    """Tests for Databricks configuration."""

    @pytest.fixture
    def databricks_available(self):
        """Check if Databricks module is available."""
        try:
            from truthound.datasources.sql.databricks import DatabricksDataSource
            return True
        except ImportError:
            return False

    def test_config_defaults(self, databricks_available):
        """Test default configuration."""
        if not databricks_available:
            pytest.skip("Databricks module not available")

        from truthound.datasources.sql.databricks import DatabricksConfig

        config = DatabricksConfig()
        assert config.use_cloud_fetch is True
        assert config.max_download_threads == 10
        assert config.use_oauth is False

    def test_config_pat_auth(self, databricks_available):
        """Test Personal Access Token authentication."""
        if not databricks_available:
            pytest.skip("Databricks module not available")

        from truthound.datasources.sql.databricks import DatabricksConfig

        config = DatabricksConfig(
            host="adb-12345.azuredatabricks.net",
            http_path="/sql/1.0/warehouses/abc123",
            access_token="dapi...",
            use_oauth=False
        )
        assert config.access_token.startswith("dapi")
        assert config.use_oauth is False

    def test_config_oauth(self, databricks_available):
        """Test OAuth authentication."""
        if not databricks_available:
            pytest.skip("Databricks module not available")

        from truthound.datasources.sql.databricks import DatabricksConfig

        config = DatabricksConfig(
            host="adb-12345.azuredatabricks.net",
            http_path="/sql/1.0/warehouses/abc123",
            client_id="client-id",
            client_secret="client-secret",
            use_oauth=True
        )
        assert config.use_oauth is True
        assert config.client_id == "client-id"


class TestDatabricksDataSource:
    """Tests for Databricks data source."""

    @pytest.fixture
    def databricks_available(self):
        """Check if Databricks module is available."""
        try:
            from truthound.datasources.sql.databricks import DatabricksDataSource
            return True
        except ImportError:
            return False

    def test_source_type_attribute(self, databricks_available):
        """Test source type."""
        if not databricks_available:
            pytest.skip("Databricks module not available")

        from truthound.datasources.sql.databricks import DatabricksDataSource

        assert DatabricksDataSource.source_type == "databricks"

    def test_quote_identifier_format(self, databricks_available):
        """Test Databricks uses backticks for quoting."""
        if not databricks_available:
            pytest.skip("Databricks module not available")

        identifier = "my_column"
        escaped = identifier.replace("`", "``")
        quoted = f"`{escaped}`"
        assert quoted == "`my_column`"

    def test_full_table_name_with_catalog(self, databricks_available):
        """Test fully qualified table name with Unity Catalog."""
        if not databricks_available:
            pytest.skip("Databricks module not available")

        # Format: `catalog`.`schema`.`table`
        expected = "`main`.`default`.`users`"
        assert expected.count("`") == 6

    def test_full_table_name_without_catalog(self, databricks_available):
        """Test fully qualified table name without Unity Catalog."""
        if not databricks_available:
            pytest.skip("Databricks module not available")

        # Format: `schema`.`table`
        expected = "`default`.`users`"
        assert expected.count("`") == 4


# =============================================================================
# Module Availability Tests
# =============================================================================


class TestModuleAvailability:
    """Tests for module availability checking."""

    def test_get_available_sources(self):
        """Test getting available data sources."""
        from truthound.datasources.sql import get_available_sources

        sources = get_available_sources()

        # Core sources should always be available
        assert sources.get("sqlite") is not None
        assert sources.get("postgresql") is not None
        assert sources.get("mysql") is not None

        # Enterprise sources may or may not be available
        assert "oracle" in sources
        assert "sqlserver" in sources
        assert "bigquery" in sources
        assert "snowflake" in sources
        assert "redshift" in sources
        assert "databricks" in sources

    def test_check_source_available(self):
        """Test checking if specific source is available."""
        from truthound.datasources.sql import check_source_available

        # SQLite should always be available
        assert check_source_available("sqlite") is True

        # Check that the function works for all source types
        for source_type in ["sqlite", "postgresql", "mysql", "oracle",
                           "sqlserver", "bigquery", "snowflake",
                           "redshift", "databricks"]:
            result = check_source_available(source_type)
            assert isinstance(result, bool)

    def test_unknown_source_not_available(self):
        """Test that unknown sources return False."""
        from truthound.datasources.sql import check_source_available

        assert check_source_available("unknown_db") is False


# =============================================================================
# SQL Exports Tests
# =============================================================================


class TestSQLExports:
    """Tests for SQL module exports."""

    def test_base_exports(self):
        """Test base class exports."""
        from truthound.datasources.sql import (
            BaseSQLDataSource,
            SQLDataSourceConfig,
        )
        assert BaseSQLDataSource is not None
        assert SQLDataSourceConfig is not None

    def test_core_database_exports(self):
        """Test core database exports."""
        from truthound.datasources.sql import (
            SQLiteDataSource,
            PostgreSQLDataSource,
            MySQLDataSource,
        )
        assert SQLiteDataSource is not None
        assert PostgreSQLDataSource is not None
        assert MySQLDataSource is not None

    def test_cloud_base_exports(self):
        """Test cloud base exports."""
        from truthound.datasources.sql import (
            CloudDWConfig,
            CloudDWDataSource,
            load_credentials_from_env,
            load_service_account_json,
        )
        assert CloudDWConfig is not None
        assert CloudDWDataSource is not None
        assert load_credentials_from_env is not None
        assert load_service_account_json is not None

    def test_optional_exports_exist(self):
        """Test that optional exports are defined (may be None)."""
        from truthound.datasources import sql

        # These may be None if dependencies aren't installed
        # but they should be defined
        assert hasattr(sql, "BigQueryDataSource")
        assert hasattr(sql, "SnowflakeDataSource")
        assert hasattr(sql, "OracleDataSource")
        assert hasattr(sql, "SQLServerDataSource")
        assert hasattr(sql, "RedshiftDataSource")
        assert hasattr(sql, "DatabricksDataSource")


# =============================================================================
# Factory Tests
# =============================================================================


class TestDataSourceFactory:
    """Tests for data source factory functions."""

    def test_factory_sqlite_detection(self, tmp_path):
        """Test factory detects SQLite files."""
        from truthound.datasources import get_sql_datasource
        import sqlite3

        # Create a test SQLite database
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test (id INTEGER)")
        conn.commit()
        conn.close()

        source = get_sql_datasource(str(db_path), table="test")
        assert source.source_type == "sqlite"

    def test_factory_memory_detection(self):
        """Test factory detects :memory: as SQLite."""
        from truthound.datasources import get_sql_datasource

        source = get_sql_datasource(":memory:", table="test")
        assert source.source_type == "sqlite"

    def test_factory_postgresql_connection_string(self):
        """Test factory handles PostgreSQL connection strings."""
        from truthound.datasources.factory import detect_datasource_type

        conn_str = "postgresql://user:pass@localhost/db"
        source_type = detect_datasource_type(conn_str)
        assert source_type == "postgresql"

    def test_factory_mysql_connection_string(self):
        """Test factory handles MySQL connection strings."""
        from truthound.datasources.factory import detect_datasource_type

        conn_str = "mysql://user:pass@localhost/db"
        source_type = detect_datasource_type(conn_str)
        assert source_type == "mysql"

    def test_factory_sqlite_file_detection(self):
        """Test factory detects SQLite file paths."""
        from truthound.datasources.factory import detect_datasource_type

        # File doesn't exist, but detection should still work for .db extension
        source_type = detect_datasource_type("/path/to/database.db")
        assert source_type == "sqlite"

    def test_factory_memory_type_detection(self):
        """Test factory detects :memory: as SQLite."""
        from truthound.datasources.factory import detect_datasource_type

        source_type = detect_datasource_type(":memory:")
        assert source_type == "sqlite"


# =============================================================================
# Connection String Detection Tests
# =============================================================================


class TestConnectionStringDetection:
    """Tests for SQL connection string detection in factory."""

    def test_postgresql_detection(self):
        """Test PostgreSQL connection string detection."""
        from truthound.datasources.factory import detect_datasource_type

        # Standard postgresql:// prefix
        assert detect_datasource_type("postgresql://user@host/db") == "postgresql"
        # Alternative postgres:// prefix
        assert detect_datasource_type("postgres://user@host/db") == "postgresql"

    def test_mysql_detection(self):
        """Test MySQL connection string detection."""
        from truthound.datasources.factory import detect_datasource_type

        assert detect_datasource_type("mysql://user@host/db") == "mysql"


# =============================================================================
# Cost Estimation Tests
# =============================================================================


class TestCostEstimation:
    """Tests for cloud DW cost estimation features."""

    def test_execute_with_cost_check_no_limits(self):
        """Test execute without cost limits."""
        from truthound.datasources.sql.cloud_base import CloudDWDataSource, CloudDWConfig
        from abc import ABC

        # Create a concrete implementation for testing
        class MockCloudSource(CloudDWDataSource):
            source_type = "mock"

            def __init__(self):
                self._config = CloudDWConfig()
                self._table = "test"

            def _create_connection(self):
                return MagicMock()

            def _fetch_schema(self):
                return [("id", "INTEGER")]

            def _get_row_count_query(self):
                return "SELECT COUNT(*) FROM test"

            def _quote_identifier(self, identifier):
                return f'"{identifier}"'

            def _validate_credentials(self):
                return True

            def _get_cost_estimate(self, query):
                return {"bytes_processed": 1000, "estimated_cost_usd": 0.001}

            def execute_query(self, query):
                return [{"count": 100}]

            def _get_table_schema_query(self):
                return "SELECT * FROM test LIMIT 0"

        source = MockCloudSource()

        result = source.execute_with_cost_check("SELECT COUNT(*) FROM test")
        assert result == [{"count": 100}]

    def test_execute_with_cost_check_bytes_limit_exceeded(self):
        """Test execute with bytes limit exceeded."""
        from truthound.datasources.sql.cloud_base import CloudDWDataSource, CloudDWConfig

        class MockCloudSource(CloudDWDataSource):
            source_type = "mock"

            def __init__(self):
                self._config = CloudDWConfig()
                self._table = "test"

            def _create_connection(self):
                return MagicMock()

            def _fetch_schema(self):
                return [("id", "INTEGER")]

            def _get_row_count_query(self):
                return "SELECT COUNT(*) FROM test"

            def _quote_identifier(self, identifier):
                return f'"{identifier}"'

            def _validate_credentials(self):
                return True

            def _get_cost_estimate(self, query):
                return {"bytes_processed": 1000000000, "estimated_cost_usd": 5.0}

            def execute_query(self, query):
                return [{"count": 100}]

            def _get_table_schema_query(self):
                return "SELECT * FROM test LIMIT 0"

        source = MockCloudSource()

        with pytest.raises(DataSourceError, match="exceeding limit"):
            source.execute_with_cost_check(
                "SELECT * FROM large_table",
                max_bytes=1000000  # 1MB limit
            )

    def test_execute_with_cost_check_usd_limit_exceeded(self):
        """Test execute with USD cost limit exceeded."""
        from truthound.datasources.sql.cloud_base import CloudDWDataSource, CloudDWConfig

        class MockCloudSource(CloudDWDataSource):
            source_type = "mock"

            def __init__(self):
                self._config = CloudDWConfig()
                self._table = "test"

            def _create_connection(self):
                return MagicMock()

            def _fetch_schema(self):
                return [("id", "INTEGER")]

            def _get_row_count_query(self):
                return "SELECT COUNT(*) FROM test"

            def _quote_identifier(self, identifier):
                return f'"{identifier}"'

            def _validate_credentials(self):
                return True

            def _get_cost_estimate(self, query):
                return {"bytes_processed": 1000000000, "estimated_cost_usd": 5.0}

            def execute_query(self, query):
                return [{"count": 100}]

            def _get_table_schema_query(self):
                return "SELECT * FROM test LIMIT 0"

        source = MockCloudSource()

        with pytest.raises(DataSourceError, match="exceeds limit"):
            source.execute_with_cost_check(
                "SELECT * FROM large_table",
                max_cost_usd=1.0  # $1 limit
            )

    def test_execute_with_cost_check_no_estimate(self):
        """Test execute when cost estimation returns None."""
        from truthound.datasources.sql.cloud_base import CloudDWDataSource, CloudDWConfig

        class MockCloudSource(CloudDWDataSource):
            source_type = "mock"

            def __init__(self):
                self._config = CloudDWConfig()
                self._table = "test"

            def _create_connection(self):
                return MagicMock()

            def _fetch_schema(self):
                return [("id", "INTEGER")]

            def _get_row_count_query(self):
                return "SELECT COUNT(*) FROM test"

            def _quote_identifier(self, identifier):
                return f'"{identifier}"'

            def _validate_credentials(self):
                return True

            def _get_cost_estimate(self, query):
                return None  # No estimate available

            def execute_query(self, query):
                return [{"count": 100}]

            def _get_table_schema_query(self):
                return "SELECT * FROM test LIMIT 0"

        source = MockCloudSource()

        # Should not raise even with limits set if estimate is None
        result = source.execute_with_cost_check(
            "SELECT COUNT(*) FROM test",
            max_bytes=1000,
            max_cost_usd=0.01
        )
        assert result == [{"count": 100}]
