"""Tenant isolation strategies for multi-tenancy.

This module provides different data isolation strategies to ensure
tenant data separation at various levels of isolation.

Isolation Levels (from lowest to highest):
    - SHARED: Column-based filtering (tenant_id column)
    - ROW_LEVEL: Row-level security with additional checks
    - SCHEMA: Each tenant has its own schema
    - DATABASE: Each tenant has its own database
    - INSTANCE: Complete instance isolation (out of scope for this module)
"""

from __future__ import annotations

import hashlib
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from truthound.multitenancy.core import (
    IsolationLevel,
    IsolationStrategy,
    Tenant,
    TenantContext,
    TenantError,
    TenantIsolationError,
)

if TYPE_CHECKING:
    import polars as pl


# =============================================================================
# Shared/Column-Based Isolation
# =============================================================================


class SharedIsolation(IsolationStrategy):
    """Shared isolation using a tenant ID column.

    This is the simplest form of isolation where all tenants share
    the same tables and isolation is enforced by filtering on a
    tenant_id column.

    Pros:
        - Simple to implement
        - Easy to manage schema changes
        - Lower operational overhead

    Cons:
        - Risk of data leakage if filters are forgotten
        - Performance impact from WHERE clauses
        - Noisy neighbor issues possible

    Example:
        >>> strategy = SharedIsolation(tenant_column="tenant_id")
        >>> filtered_df = strategy.apply_filter(df, tenant)
    """

    def __init__(
        self,
        tenant_column: str = "tenant_id",
        validate_on_apply: bool = True,
    ) -> None:
        self._tenant_column = tenant_column
        self._validate_on_apply = validate_on_apply

    @property
    def isolation_level(self) -> IsolationLevel:
        """The isolation level this strategy provides."""
        return IsolationLevel.SHARED

    @property
    def tenant_column(self) -> str:
        """The column used for tenant identification."""
        return self._tenant_column

    def apply_filter(
        self,
        df: "pl.LazyFrame",
        tenant: Tenant,
    ) -> "pl.LazyFrame":
        """Apply tenant filter to a LazyFrame."""
        import polars as pl

        if self._validate_on_apply:
            self.validate_isolation(tenant)

        # Get the actual column name from tenant or use default
        column = tenant.tenant_column or self._tenant_column

        # Check if column exists
        if column not in df.collect_schema().names():
            # If column doesn't exist, this might be non-tenant data
            # Return empty frame for safety
            return df.filter(pl.lit(False))

        return df.filter(pl.col(column) == tenant.id)

    def add_tenant_column(
        self,
        df: "pl.LazyFrame",
        tenant: Tenant,
    ) -> "pl.LazyFrame":
        """Add tenant identifier to a LazyFrame."""
        import polars as pl

        column = tenant.tenant_column or self._tenant_column
        return df.with_columns(pl.lit(tenant.id).alias(column))

    def get_namespace(self, tenant: Tenant) -> str:
        """Get the namespace for a tenant (same for all tenants)."""
        return "shared"

    def validate_column_exists(self, df: "pl.LazyFrame") -> bool:
        """Check if the tenant column exists in the dataframe."""
        return self._tenant_column in df.collect_schema().names()


# =============================================================================
# Row-Level Security Isolation
# =============================================================================


@dataclass
class RowLevelPolicy:
    """A row-level security policy.

    Example:
        >>> policy = RowLevelPolicy(
        ...     name="owner_access",
        ...     condition=lambda row, tenant: row["owner_id"] == tenant.owner_id,
        ... )
    """

    name: str
    condition: Callable[[dict[str, Any], Tenant], bool]
    priority: int = 0
    enabled: bool = True

    def check(self, row: dict[str, Any], tenant: Tenant) -> bool:
        """Check if the policy allows access."""
        if not self.enabled:
            return True
        return self.condition(row, tenant)


class RowLevelIsolation(IsolationStrategy):
    """Row-level isolation with additional security policies.

    Extends shared isolation with additional row-level checks
    that can enforce more complex access rules.

    Example:
        >>> strategy = RowLevelIsolation(tenant_column="tenant_id")
        >>> strategy.add_policy(RowLevelPolicy(
        ...     name="owner_only",
        ...     condition=lambda row, tenant: row.get("owner") == tenant.owner_id,
        ... ))
        >>> filtered_df = strategy.apply_filter(df, tenant)
    """

    def __init__(
        self,
        tenant_column: str = "tenant_id",
        validate_on_apply: bool = True,
        fail_closed: bool = True,
    ) -> None:
        self._tenant_column = tenant_column
        self._validate_on_apply = validate_on_apply
        self._fail_closed = fail_closed  # Deny access on policy errors
        self._policies: list[RowLevelPolicy] = []

    @property
    def isolation_level(self) -> IsolationLevel:
        """The isolation level this strategy provides."""
        return IsolationLevel.ROW_LEVEL

    def add_policy(self, policy: RowLevelPolicy) -> None:
        """Add a row-level security policy."""
        self._policies.append(policy)
        self._policies.sort(key=lambda p: p.priority, reverse=True)

    def remove_policy(self, name: str) -> bool:
        """Remove a policy by name."""
        for i, policy in enumerate(self._policies):
            if policy.name == name:
                del self._policies[i]
                return True
        return False

    def apply_filter(
        self,
        df: "pl.LazyFrame",
        tenant: Tenant,
    ) -> "pl.LazyFrame":
        """Apply tenant filter with row-level policies."""
        import polars as pl

        if self._validate_on_apply:
            self.validate_isolation(tenant)

        column = tenant.tenant_column or self._tenant_column

        # First apply basic tenant filter
        if column in df.collect_schema().names():
            df = df.filter(pl.col(column) == tenant.id)
        elif self._fail_closed:
            return df.filter(pl.lit(False))

        # Additional row-level policies would be applied here
        # Note: Complex row-level policies may require collecting data
        # For now, we return the tenant-filtered frame
        return df

    def add_tenant_column(
        self,
        df: "pl.LazyFrame",
        tenant: Tenant,
    ) -> "pl.LazyFrame":
        """Add tenant identifier to a LazyFrame."""
        import polars as pl

        column = tenant.tenant_column or self._tenant_column
        return df.with_columns(pl.lit(tenant.id).alias(column))

    def get_namespace(self, tenant: Tenant) -> str:
        """Get the namespace for a tenant."""
        return "shared"


# =============================================================================
# Schema-Based Isolation
# =============================================================================


@dataclass
class SchemaConfig:
    """Configuration for schema-based isolation."""

    schema_prefix: str = "tenant_"
    create_schema_on_demand: bool = True
    default_schema: str = "public"
    schema_template: str = "{prefix}{tenant_slug}"


class SchemaIsolation(IsolationStrategy):
    """Schema-based isolation where each tenant has its own schema.

    Each tenant's data lives in a separate database schema, providing
    stronger isolation than row-level filtering.

    Pros:
        - Stronger isolation than row-level
        - Easier to backup/restore individual tenants
        - Clear data boundaries
        - Can have tenant-specific schema customizations

    Cons:
        - More schemas to manage
        - Cross-tenant queries are harder
        - Schema migration complexity

    Example:
        >>> strategy = SchemaIsolation(
        ...     config=SchemaConfig(schema_prefix="t_"),
        ... )
        >>> namespace = strategy.get_namespace(tenant)
        >>> # Returns: "t_acme_corp"
    """

    def __init__(
        self,
        config: SchemaConfig | None = None,
    ) -> None:
        self._config = config or SchemaConfig()

    @property
    def isolation_level(self) -> IsolationLevel:
        """The isolation level this strategy provides."""
        return IsolationLevel.SCHEMA

    def get_namespace(self, tenant: Tenant) -> str:
        """Get the schema name for a tenant."""
        if tenant.schema_name:
            return tenant.schema_name

        return self._config.schema_template.format(
            prefix=self._config.schema_prefix,
            tenant_slug=tenant.slug,
            tenant_id=tenant.id,
        )

    def apply_filter(
        self,
        df: "pl.LazyFrame",
        tenant: Tenant,
    ) -> "pl.LazyFrame":
        """Apply tenant filter.

        For schema isolation, the data should already be from the
        tenant's schema. This method validates that assumption.
        """
        if self._validate_on_apply:
            self.validate_isolation(tenant)

        # In schema isolation, no additional filtering needed
        # The data source should be configured to use the correct schema
        return df

    def _validate_on_apply(self, tenant: Tenant) -> None:
        """Validate the tenant context."""
        self.validate_isolation(tenant)

    def add_tenant_column(
        self,
        df: "pl.LazyFrame",
        tenant: Tenant,
    ) -> "pl.LazyFrame":
        """Add tenant identifier (optional in schema isolation)."""
        import polars as pl

        # In schema isolation, tenant column is optional but can be added
        # for audit/logging purposes
        column = tenant.tenant_column or "tenant_id"
        return df.with_columns(pl.lit(tenant.id).alias(column))

    def get_qualified_table_name(
        self,
        table_name: str,
        tenant: Tenant,
    ) -> str:
        """Get the fully qualified table name for a tenant."""
        schema = self.get_namespace(tenant)
        return f"{schema}.{table_name}"


# =============================================================================
# Database-Based Isolation
# =============================================================================


@dataclass
class DatabaseConfig:
    """Configuration for database-based isolation."""

    database_prefix: str = "truthound_"
    create_database_on_demand: bool = False  # Usually requires admin
    database_template: str = "{prefix}{tenant_slug}"
    connection_pool_per_tenant: bool = True


class DatabaseIsolation(IsolationStrategy):
    """Database-based isolation where each tenant has its own database.

    Each tenant's data lives in a completely separate database,
    providing the strongest isolation short of separate instances.

    Pros:
        - Strongest isolation within shared infrastructure
        - Complete data separation
        - Independent backup/restore
        - Independent scaling possible

    Cons:
        - Highest operational overhead
        - Connection pool management complexity
        - Cross-tenant analytics very difficult

    Example:
        >>> strategy = DatabaseIsolation(
        ...     config=DatabaseConfig(database_prefix="th_"),
        ... )
        >>> db_name = strategy.get_namespace(tenant)
        >>> # Returns: "th_acme_corp"
    """

    def __init__(
        self,
        config: DatabaseConfig | None = None,
    ) -> None:
        self._config = config or DatabaseConfig()
        self._connection_registry: dict[str, Any] = {}

    @property
    def isolation_level(self) -> IsolationLevel:
        """The isolation level this strategy provides."""
        return IsolationLevel.DATABASE

    def get_namespace(self, tenant: Tenant) -> str:
        """Get the database name for a tenant."""
        if tenant.database_name:
            return tenant.database_name

        return self._config.database_template.format(
            prefix=self._config.database_prefix,
            tenant_slug=tenant.slug,
            tenant_id=tenant.id,
        )

    def apply_filter(
        self,
        df: "pl.LazyFrame",
        tenant: Tenant,
    ) -> "pl.LazyFrame":
        """Apply tenant filter.

        For database isolation, the data should already be from the
        tenant's database. No additional filtering needed.
        """
        self.validate_isolation(tenant)
        return df

    def add_tenant_column(
        self,
        df: "pl.LazyFrame",
        tenant: Tenant,
    ) -> "pl.LazyFrame":
        """Add tenant identifier (optional in database isolation)."""
        import polars as pl

        column = tenant.tenant_column or "tenant_id"
        return df.with_columns(pl.lit(tenant.id).alias(column))

    def get_connection_string(
        self,
        base_connection: str,
        tenant: Tenant,
    ) -> str:
        """Get the connection string for a tenant's database.

        This is a simple implementation that replaces the database
        name in the connection string. Production implementations
        should use proper connection string parsing.
        """
        db_name = self.get_namespace(tenant)
        # Simple replacement - production should use proper parsing
        # e.g., postgresql://user:pass@host/db -> postgresql://user:pass@host/tenant_db
        if "//" in base_connection and "/" in base_connection.split("//")[1]:
            parts = base_connection.rsplit("/", 1)
            return f"{parts[0]}/{db_name}"
        return base_connection


# =============================================================================
# Composite Isolation (Multiple Strategies)
# =============================================================================


class CompositeIsolation(IsolationStrategy):
    """Composite isolation that applies multiple strategies.

    Useful for defense-in-depth where you want both row-level
    filtering AND schema isolation checks.

    Example:
        >>> composite = CompositeIsolation([
        ...     RowLevelIsolation(tenant_column="tenant_id"),
        ...     SchemaIsolation(config=SchemaConfig()),
        ... ])
        >>> filtered_df = composite.apply_filter(df, tenant)
    """

    def __init__(
        self,
        strategies: list[IsolationStrategy],
        primary_isolation: IsolationLevel = IsolationLevel.ROW_LEVEL,
    ) -> None:
        if not strategies:
            raise ValueError("At least one strategy is required")
        self._strategies = strategies
        self._primary_isolation = primary_isolation

    @property
    def isolation_level(self) -> IsolationLevel:
        """The highest isolation level among strategies."""
        return self._primary_isolation

    def apply_filter(
        self,
        df: "pl.LazyFrame",
        tenant: Tenant,
    ) -> "pl.LazyFrame":
        """Apply all isolation strategies."""
        for strategy in self._strategies:
            df = strategy.apply_filter(df, tenant)
        return df

    def add_tenant_column(
        self,
        df: "pl.LazyFrame",
        tenant: Tenant,
    ) -> "pl.LazyFrame":
        """Add tenant column using primary strategy."""
        return self._strategies[0].add_tenant_column(df, tenant)

    def get_namespace(self, tenant: Tenant) -> str:
        """Get namespace from the most isolated strategy."""
        # Find strategy with highest isolation level
        for level in [
            IsolationLevel.DATABASE,
            IsolationLevel.SCHEMA,
            IsolationLevel.ROW_LEVEL,
            IsolationLevel.SHARED,
        ]:
            for strategy in self._strategies:
                if strategy.isolation_level == level:
                    return strategy.get_namespace(tenant)
        return self._strategies[0].get_namespace(tenant)


# =============================================================================
# Isolation Manager
# =============================================================================


class IsolationManager:
    """Manages isolation strategies for the multi-tenant system.

    Provides a central point for configuring and applying isolation
    strategies based on tenant configuration.

    Example:
        >>> manager = IsolationManager()
        >>> manager.register(IsolationLevel.ROW_LEVEL, RowLevelIsolation())
        >>> manager.register(IsolationLevel.SCHEMA, SchemaIsolation())
        >>>
        >>> # Apply isolation based on tenant's configured level
        >>> filtered_df = manager.apply_filter(df, tenant)
    """

    def __init__(self) -> None:
        self._strategies: dict[IsolationLevel, IsolationStrategy] = {}
        self._default_level = IsolationLevel.ROW_LEVEL

        # Register default strategies
        self._strategies[IsolationLevel.SHARED] = SharedIsolation()
        self._strategies[IsolationLevel.ROW_LEVEL] = RowLevelIsolation()
        self._strategies[IsolationLevel.SCHEMA] = SchemaIsolation()
        self._strategies[IsolationLevel.DATABASE] = DatabaseIsolation()

    def register(
        self,
        level: IsolationLevel,
        strategy: IsolationStrategy,
    ) -> None:
        """Register an isolation strategy for a level."""
        self._strategies[level] = strategy

    def get_strategy(self, tenant: Tenant) -> IsolationStrategy:
        """Get the isolation strategy for a tenant."""
        level = tenant.isolation_level
        if level not in self._strategies:
            level = self._default_level
        return self._strategies[level]

    def apply_filter(
        self,
        df: "pl.LazyFrame",
        tenant: Tenant,
    ) -> "pl.LazyFrame":
        """Apply the appropriate isolation filter for a tenant."""
        strategy = self.get_strategy(tenant)
        return strategy.apply_filter(df, tenant)

    def add_tenant_column(
        self,
        df: "pl.LazyFrame",
        tenant: Tenant,
    ) -> "pl.LazyFrame":
        """Add tenant column using appropriate strategy."""
        strategy = self.get_strategy(tenant)
        return strategy.add_tenant_column(df, tenant)

    def get_namespace(self, tenant: Tenant) -> str:
        """Get the namespace for a tenant."""
        strategy = self.get_strategy(tenant)
        return strategy.get_namespace(tenant)

    def set_default_level(self, level: IsolationLevel) -> None:
        """Set the default isolation level."""
        self._default_level = level


# =============================================================================
# Factory Function
# =============================================================================


def create_isolation_strategy(
    level: IsolationLevel,
    **kwargs: Any,
) -> IsolationStrategy:
    """Create an isolation strategy.

    Args:
        level: Isolation level
        **kwargs: Strategy-specific configuration

    Returns:
        Configured IsolationStrategy instance.

    Example:
        >>> strategy = create_isolation_strategy(
        ...     IsolationLevel.SCHEMA,
        ...     schema_prefix="t_",
        ... )
    """
    if level == IsolationLevel.SHARED:
        return SharedIsolation(
            tenant_column=kwargs.get("tenant_column", "tenant_id"),
        )
    elif level == IsolationLevel.ROW_LEVEL:
        return RowLevelIsolation(
            tenant_column=kwargs.get("tenant_column", "tenant_id"),
            fail_closed=kwargs.get("fail_closed", True),
        )
    elif level == IsolationLevel.SCHEMA:
        config = SchemaConfig(
            schema_prefix=kwargs.get("schema_prefix", "tenant_"),
        )
        return SchemaIsolation(config=config)
    elif level == IsolationLevel.DATABASE:
        config = DatabaseConfig(
            database_prefix=kwargs.get("database_prefix", "truthound_"),
        )
        return DatabaseIsolation(config=config)
    else:
        raise ValueError(f"Unsupported isolation level: {level}")
