"""Backend registry for cloud DW test backends.

This module provides a registry pattern for managing and discovering
available cloud DW test backends at runtime.

Features:
    - Dynamic backend registration
    - Lazy loading of backend implementations
    - Environment-based credential discovery
    - Availability checking without importing dependencies

Usage:
    >>> from tests.integration.cloud_dw.backends import get_backend, get_available_backends
    >>>
    >>> # Check what's available
    >>> available = get_available_backends()
    >>> print(available)  # ['bigquery', 'snowflake', ...]
    >>>
    >>> # Get a backend instance
    >>> backend = get_backend("bigquery")
    >>> with backend:
    ...     result = backend.execute_query("SELECT 1")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Type

if TYPE_CHECKING:
    from tests.integration.cloud_dw.base import (
        CloudDWTestBackend,
        BaseCredentials,
        IntegrationTestConfig,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Backend Registration
# =============================================================================


@dataclass
class BackendInfo:
    """Information about a registered backend.

    Attributes:
        name: Backend name (e.g., "bigquery").
        backend_class: The backend class or a factory function.
        credentials_class: The credentials class.
        required_env_vars: Environment variables required for this backend.
        optional_env_vars: Optional environment variables.
        required_packages: Python packages required for this backend.
        description: Human-readable description.
    """

    name: str
    backend_class: Type["CloudDWTestBackend"] | Callable[..., "CloudDWTestBackend"]
    credentials_class: Type["BaseCredentials"]
    required_env_vars: list[str] = field(default_factory=list)
    optional_env_vars: list[str] = field(default_factory=list)
    required_packages: list[str] = field(default_factory=list)
    description: str = ""

    def check_packages_available(self) -> tuple[bool, list[str]]:
        """Check if required packages are installed.

        Returns:
            Tuple of (all_available, missing_packages).
        """
        missing = []
        for package in self.required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing.append(package)
        return len(missing) == 0, missing

    def check_env_vars_available(self) -> tuple[bool, list[str]]:
        """Check if required environment variables are set.

        Returns:
            Tuple of (all_available, missing_vars).
        """
        missing = []
        for var in self.required_env_vars:
            if not os.getenv(var):
                missing.append(var)
        return len(missing) == 0, missing

    def is_available(self) -> bool:
        """Check if this backend is available for use."""
        packages_ok, _ = self.check_packages_available()
        env_ok, _ = self.check_env_vars_available()
        return packages_ok and env_ok

    def get_availability_reason(self) -> str:
        """Get a human-readable reason for availability status."""
        packages_ok, missing_packages = self.check_packages_available()
        env_ok, missing_vars = self.check_env_vars_available()

        reasons = []
        if not packages_ok:
            reasons.append(f"Missing packages: {', '.join(missing_packages)}")
        if not env_ok:
            reasons.append(f"Missing env vars: {', '.join(missing_vars)}")

        return "; ".join(reasons) if reasons else "Available"


class BackendRegistry:
    """Registry for cloud DW test backends.

    This class maintains a registry of available backends and provides
    methods for discovery and instantiation.

    Example:
        >>> registry = BackendRegistry()
        >>> registry.register(
        ...     name="bigquery",
        ...     backend_class=BigQueryTestBackend,
        ...     credentials_class=BigQueryCredentials,
        ...     required_env_vars=["GOOGLE_APPLICATION_CREDENTIALS"],
        ...     required_packages=["google-cloud-bigquery"],
        ... )
        >>> backend = registry.get("bigquery")
    """

    _instance: "BackendRegistry | None" = None

    def __init__(self) -> None:
        self._backends: dict[str, BackendInfo] = {}

    @classmethod
    def get_instance(cls) -> "BackendRegistry":
        """Get the singleton registry instance."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_default_backends()
        return cls._instance

    def register(
        self,
        name: str,
        backend_class: Type["CloudDWTestBackend"] | Callable[..., "CloudDWTestBackend"],
        credentials_class: Type["BaseCredentials"],
        required_env_vars: list[str] | None = None,
        optional_env_vars: list[str] | None = None,
        required_packages: list[str] | None = None,
        description: str = "",
    ) -> None:
        """Register a backend.

        Args:
            name: Unique name for the backend.
            backend_class: Backend class or factory function.
            credentials_class: Credentials class.
            required_env_vars: Required environment variables.
            optional_env_vars: Optional environment variables.
            required_packages: Required Python packages.
            description: Human-readable description.
        """
        self._backends[name] = BackendInfo(
            name=name,
            backend_class=backend_class,
            credentials_class=credentials_class,
            required_env_vars=required_env_vars or [],
            optional_env_vars=optional_env_vars or [],
            required_packages=required_packages or [],
            description=description,
        )
        logger.debug(f"Registered backend: {name}")

    def unregister(self, name: str) -> bool:
        """Unregister a backend.

        Args:
            name: Backend name to unregister.

        Returns:
            True if backend was unregistered, False if not found.
        """
        if name in self._backends:
            del self._backends[name]
            return True
        return False

    def get_info(self, name: str) -> BackendInfo | None:
        """Get information about a backend.

        Args:
            name: Backend name.

        Returns:
            BackendInfo or None if not found.
        """
        return self._backends.get(name)

    def get(
        self,
        name: str,
        config: "IntegrationTestConfig | None" = None,
        **credential_kwargs: Any,
    ) -> "CloudDWTestBackend":
        """Get a backend instance.

        Args:
            name: Backend name.
            config: Test configuration.
            **credential_kwargs: Override credential values.

        Returns:
            Configured backend instance.

        Raises:
            KeyError: If backend not found.
            RuntimeError: If backend is not available.
        """
        info = self._backends.get(name)
        if info is None:
            available = ", ".join(self._backends.keys())
            raise KeyError(
                f"Backend '{name}' not found. Available: {available}"
            )

        # Check availability
        if not info.is_available():
            reason = info.get_availability_reason()
            raise RuntimeError(
                f"Backend '{name}' is not available: {reason}"
            )

        # Create credentials from environment + overrides
        credentials = self._create_credentials(info, credential_kwargs)

        # Create backend instance
        return info.backend_class(credentials=credentials, config=config)

    def _create_credentials(
        self,
        info: BackendInfo,
        overrides: dict[str, Any],
    ) -> "BaseCredentials":
        """Create credentials instance from environment and overrides.

        Args:
            info: Backend info.
            overrides: Override values.

        Returns:
            Credentials instance.
        """
        # Gather all env vars
        env_values = {}
        for var in info.required_env_vars + info.optional_env_vars:
            value = os.getenv(var)
            if value:
                # Convert env var name to credential field name
                # e.g., BIGQUERY_PROJECT -> project
                field_name = var.lower()
                for prefix in ["bigquery_", "snowflake_", "redshift_", "databricks_"]:
                    if field_name.startswith(prefix):
                        field_name = field_name[len(prefix):]
                        break
                env_values[field_name] = value

        # Merge with overrides (overrides take precedence)
        final_values = {**env_values, **overrides}

        return info.credentials_class(**final_values)

    def list_all(self) -> list[str]:
        """List all registered backend names."""
        return list(self._backends.keys())

    def list_available(self) -> list[str]:
        """List available (properly configured) backend names."""
        return [
            name for name, info in self._backends.items()
            if info.is_available()
        ]

    def get_availability_report(self) -> dict[str, dict[str, Any]]:
        """Get detailed availability report for all backends.

        Returns:
            Dictionary mapping backend names to their availability status.
        """
        report = {}
        for name, info in self._backends.items():
            packages_ok, missing_packages = info.check_packages_available()
            env_ok, missing_vars = info.check_env_vars_available()

            report[name] = {
                "available": info.is_available(),
                "packages_installed": packages_ok,
                "missing_packages": missing_packages,
                "env_vars_set": env_ok,
                "missing_env_vars": missing_vars,
                "description": info.description,
            }
        return report

    def _register_default_backends(self) -> None:
        """Register the default set of backends."""
        # Import here to avoid circular imports
        from tests.integration.cloud_dw.backends.bigquery import (
            BigQueryTestBackend,
            BigQueryCredentials,
        )
        from tests.integration.cloud_dw.backends.snowflake import (
            SnowflakeTestBackend,
            SnowflakeCredentials,
        )
        from tests.integration.cloud_dw.backends.redshift import (
            RedshiftTestBackend,
            RedshiftCredentials,
        )
        from tests.integration.cloud_dw.backends.databricks import (
            DatabricksTestBackend,
            DatabricksCredentials,
        )

        # BigQuery
        self.register(
            name="bigquery",
            backend_class=BigQueryTestBackend,
            credentials_class=BigQueryCredentials,
            required_env_vars=["BIGQUERY_PROJECT"],
            optional_env_vars=[
                "GOOGLE_APPLICATION_CREDENTIALS",
                "BIGQUERY_LOCATION",
                "BIGQUERY_DATASET",
            ],
            required_packages=["google.cloud.bigquery"],
            description="Google BigQuery data warehouse",
        )

        # Snowflake
        self.register(
            name="snowflake",
            backend_class=SnowflakeTestBackend,
            credentials_class=SnowflakeCredentials,
            required_env_vars=[
                "SNOWFLAKE_ACCOUNT",
                "SNOWFLAKE_USER",
            ],
            optional_env_vars=[
                "SNOWFLAKE_PASSWORD",
                "SNOWFLAKE_WAREHOUSE",
                "SNOWFLAKE_DATABASE",
                "SNOWFLAKE_SCHEMA",
                "SNOWFLAKE_ROLE",
                "SNOWFLAKE_PRIVATE_KEY_PATH",
            ],
            required_packages=["snowflake.connector"],
            description="Snowflake data warehouse",
        )

        # Redshift
        self.register(
            name="redshift",
            backend_class=RedshiftTestBackend,
            credentials_class=RedshiftCredentials,
            required_env_vars=[
                "REDSHIFT_HOST",
                "REDSHIFT_DATABASE",
            ],
            optional_env_vars=[
                "REDSHIFT_USER",
                "REDSHIFT_PASSWORD",
                "REDSHIFT_PORT",
                "REDSHIFT_IAM_ROLE",
                "REDSHIFT_CLUSTER_IDENTIFIER",
                "AWS_REGION",
            ],
            required_packages=["redshift_connector"],
            description="AWS Redshift data warehouse",
        )

        # Databricks
        self.register(
            name="databricks",
            backend_class=DatabricksTestBackend,
            credentials_class=DatabricksCredentials,
            required_env_vars=[
                "DATABRICKS_HOST",
                "DATABRICKS_HTTP_PATH",
            ],
            optional_env_vars=[
                "DATABRICKS_TOKEN",
                "DATABRICKS_CATALOG",
                "DATABRICKS_SCHEMA",
                "DATABRICKS_CLIENT_ID",
                "DATABRICKS_CLIENT_SECRET",
            ],
            required_packages=["databricks.sql"],
            description="Databricks SQL warehouse",
        )


# =============================================================================
# Module-level convenience functions
# =============================================================================


def get_backend(
    name: str,
    config: "IntegrationTestConfig | None" = None,
    **credential_kwargs: Any,
) -> "CloudDWTestBackend":
    """Get a backend instance from the global registry.

    Args:
        name: Backend name.
        config: Test configuration.
        **credential_kwargs: Override credential values.

    Returns:
        Configured backend instance.
    """
    return BackendRegistry.get_instance().get(name, config, **credential_kwargs)


def register_backend(
    name: str,
    backend_class: Type["CloudDWTestBackend"],
    credentials_class: Type["BaseCredentials"],
    **kwargs: Any,
) -> None:
    """Register a backend in the global registry.

    Args:
        name: Unique name for the backend.
        backend_class: Backend class.
        credentials_class: Credentials class.
        **kwargs: Additional registration options.
    """
    BackendRegistry.get_instance().register(
        name=name,
        backend_class=backend_class,
        credentials_class=credentials_class,
        **kwargs,
    )


def get_available_backends() -> list[str]:
    """Get list of available backends from the global registry.

    Returns:
        List of available backend names.
    """
    return BackendRegistry.get_instance().list_available()


def get_all_backends() -> list[str]:
    """Get list of all registered backends from the global registry.

    Returns:
        List of all backend names.
    """
    return BackendRegistry.get_instance().list_all()


def get_backend_availability_report() -> dict[str, dict[str, Any]]:
    """Get availability report for all backends.

    Returns:
        Dictionary with availability status for each backend.
    """
    return BackendRegistry.get_instance().get_availability_report()
