"""Integration with Truthound core functionality.

This module provides integration points between the multi-tenancy
system and Truthound's core validation, checkpoint, and reporting features.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Iterator, TypeVar

from truthound.multitenancy.core import (
    IsolationLevel,
    ResourceType,
    Tenant,
    TenantContext,
    TenantError,
    TenantQuota,
)
from truthound.multitenancy.manager import TenantManager, get_tenant_manager
from truthound.multitenancy.quota import QuotaEnforcer

if TYPE_CHECKING:
    import polars as pl
    from truthound.report import Report


F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Tenant-Aware Validation
# =============================================================================


@dataclass
class TenantValidationConfig:
    """Configuration for tenant-aware validation."""

    # Quota tracking
    track_validations: bool = True
    track_rows: bool = True
    track_api_calls: bool = True

    # Isolation
    apply_isolation: bool = True
    add_tenant_column: bool = False

    # Limits
    enforce_row_limit: bool = True
    enforce_daily_limit: bool = True

    # Results
    store_results: bool = True
    results_retention_days: int = 30


class TenantAwareValidation:
    """Mixin for tenant-aware validation operations.

    Provides methods for running validations with tenant context,
    quota enforcement, and data isolation.

    Example:
        >>> validation = TenantAwareValidation(manager)
        >>> with validation.tenant_scope(tenant):
        ...     result = th.check(data)
    """

    def __init__(
        self,
        manager: TenantManager | None = None,
        config: TenantValidationConfig | None = None,
    ) -> None:
        self._manager = manager or get_tenant_manager()
        self._config = config or TenantValidationConfig()

    @contextmanager
    def tenant_scope(self, tenant: Tenant | str) -> Iterator[Tenant]:
        """Context manager for tenant-scoped validation.

        Args:
            tenant: Tenant object or ID

        Yields:
            Tenant object.

        Example:
            >>> with validation.tenant_scope("tenant_123") as tenant:
            ...     result = th.check(data)
        """
        if isinstance(tenant, str):
            tenant = self._manager.require(tenant)

        with TenantContext.set_current(tenant):
            yield tenant

    def prepare_data(
        self,
        df: "pl.LazyFrame",
        tenant: Tenant | None = None,
    ) -> "pl.LazyFrame":
        """Prepare data for validation with tenant isolation.

        Args:
            df: Input LazyFrame
            tenant: Tenant (uses current if not provided)

        Returns:
            Isolated LazyFrame.
        """
        if tenant is None:
            tenant = TenantContext.get_current_tenant()

        if tenant and self._config.apply_isolation:
            df = self._manager.apply_isolation(df, tenant)

        return df

    def check_quotas(
        self,
        row_count: int,
        tenant: Tenant | None = None,
    ) -> None:
        """Check quotas before validation.

        Args:
            row_count: Number of rows to validate
            tenant: Tenant (uses current if not provided)

        Raises:
            TenantQuotaExceededError: If quotas exceeded.
        """
        if tenant is None:
            tenant = TenantContext.get_current_tenant()

        if not tenant:
            return

        enforcer = self._manager.quota_enforcer

        if self._config.enforce_daily_limit:
            enforcer.require(tenant, ResourceType.VALIDATIONS)

        if self._config.enforce_row_limit:
            enforcer.require(tenant, ResourceType.ROWS_PROCESSED, row_count)

    def track_usage(
        self,
        row_count: int,
        tenant: Tenant | None = None,
    ) -> None:
        """Track validation usage.

        Args:
            row_count: Number of rows validated
            tenant: Tenant (uses current if not provided)
        """
        if tenant is None:
            tenant = TenantContext.get_current_tenant()

        if not tenant:
            return

        if self._config.track_validations:
            self._manager.track_usage(ResourceType.VALIDATIONS, 1, tenant)

        if self._config.track_rows:
            self._manager.track_usage(ResourceType.ROWS_PROCESSED, row_count, tenant)

    def annotate_report(
        self,
        report: "Report",
        tenant: Tenant | None = None,
    ) -> "Report":
        """Annotate a validation report with tenant info.

        Args:
            report: Validation report
            tenant: Tenant (uses current if not provided)

        Returns:
            Annotated report.
        """
        if tenant is None:
            tenant = TenantContext.get_current_tenant()

        if tenant:
            # Add tenant metadata to report
            if hasattr(report, "metadata"):
                report.metadata["tenant_id"] = tenant.id
                report.metadata["tenant_name"] = tenant.name
                report.metadata["tenant_tier"] = tenant.tier.value

        return report


# =============================================================================
# Tenant-Aware Checkpoint
# =============================================================================


@dataclass
class TenantCheckpointConfig:
    """Configuration for tenant-aware checkpoints."""

    # Namespace
    namespace_by_tenant: bool = True
    namespace_template: str = "{tenant_id}:{checkpoint_name}"

    # Quota
    track_checkpoint_runs: bool = True

    # Results storage
    isolate_results: bool = True
    results_path_template: str = "{base_path}/{tenant_id}/{checkpoint_name}"


class TenantAwareCheckpoint:
    """Mixin for tenant-aware checkpoint operations.

    Example:
        >>> checkpoint = TenantAwareCheckpoint(manager)
        >>> checkpoint_name = checkpoint.get_checkpoint_name("daily_check", tenant)
        >>> # Returns: "tenant_123:daily_check"
    """

    def __init__(
        self,
        manager: TenantManager | None = None,
        config: TenantCheckpointConfig | None = None,
    ) -> None:
        self._manager = manager or get_tenant_manager()
        self._config = config or TenantCheckpointConfig()

    def get_checkpoint_name(
        self,
        name: str,
        tenant: Tenant | None = None,
    ) -> str:
        """Get tenant-scoped checkpoint name.

        Args:
            name: Base checkpoint name
            tenant: Tenant (uses current if not provided)

        Returns:
            Tenant-scoped checkpoint name.
        """
        if tenant is None:
            tenant = TenantContext.get_current_tenant()

        if not tenant or not self._config.namespace_by_tenant:
            return name

        return self._config.namespace_template.format(
            tenant_id=tenant.id,
            tenant_slug=tenant.slug,
            checkpoint_name=name,
        )

    def get_results_path(
        self,
        base_path: str,
        checkpoint_name: str,
        tenant: Tenant | None = None,
    ) -> str:
        """Get tenant-specific results path.

        Args:
            base_path: Base path for results
            checkpoint_name: Checkpoint name
            tenant: Tenant (uses current if not provided)

        Returns:
            Tenant-specific path.
        """
        if tenant is None:
            tenant = TenantContext.get_current_tenant()

        if not tenant or not self._config.isolate_results:
            return f"{base_path}/{checkpoint_name}"

        return self._config.results_path_template.format(
            base_path=base_path,
            tenant_id=tenant.id,
            tenant_slug=tenant.slug,
            checkpoint_name=checkpoint_name,
        )

    def track_checkpoint_run(
        self,
        tenant: Tenant | None = None,
    ) -> None:
        """Track checkpoint execution.

        Args:
            tenant: Tenant (uses current if not provided)
        """
        if not self._config.track_checkpoint_runs:
            return

        if tenant is None:
            tenant = TenantContext.get_current_tenant()

        if tenant:
            self._manager.track_usage(ResourceType.CHECKPOINTS, 1, tenant)


# =============================================================================
# Tenant-Aware Storage
# =============================================================================


class TenantAwareStore:
    """Wrapper for stores that adds tenant isolation.

    Example:
        >>> from truthound.stores import get_store
        >>> base_store = get_store("filesystem")
        >>> tenant_store = TenantAwareStore(base_store, manager)
        >>> with manager.context(tenant):
        ...     tenant_store.save(result)  # Saved to tenant namespace
    """

    def __init__(
        self,
        store: Any,
        manager: TenantManager | None = None,
        namespace_by_tenant: bool = True,
    ) -> None:
        self._store = store
        self._manager = manager or get_tenant_manager()
        self._namespace_by_tenant = namespace_by_tenant

    def _get_tenant_namespace(self) -> str:
        """Get namespace for current tenant."""
        tenant = TenantContext.get_current_tenant()
        if tenant and self._namespace_by_tenant:
            return f"tenants/{tenant.id}"
        return ""

    def save(self, result: Any, *args: Any, **kwargs: Any) -> Any:
        """Save with tenant namespace."""
        namespace = self._get_tenant_namespace()
        if namespace and hasattr(self._store, "save"):
            # Prepend namespace to path if supported
            if "path" in kwargs:
                kwargs["path"] = f"{namespace}/{kwargs['path']}"
        return self._store.save(result, *args, **kwargs)

    def get(self, *args: Any, **kwargs: Any) -> Any:
        """Get with tenant namespace."""
        namespace = self._get_tenant_namespace()
        if namespace and hasattr(self._store, "get"):
            if "path" in kwargs:
                kwargs["path"] = f"{namespace}/{kwargs['path']}"
        return self._store.get(*args, **kwargs)

    def list(self, *args: Any, **kwargs: Any) -> Any:
        """List with tenant namespace."""
        namespace = self._get_tenant_namespace()
        if namespace and hasattr(self._store, "list"):
            if "path" in kwargs:
                kwargs["path"] = f"{namespace}/{kwargs.get('path', '')}"
            elif args:
                args = (f"{namespace}/{args[0]}",) + args[1:]
        return self._store.list(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate to wrapped store."""
        return getattr(self._store, name)


# =============================================================================
# Decorators for Tenant-Aware Functions
# =============================================================================


def tenant_validation(
    track_usage: bool = True,
    check_quotas: bool = True,
    apply_isolation: bool = True,
) -> Callable[[F], F]:
    """Decorator for tenant-aware validation functions.

    Example:
        >>> @tenant_validation()
        ... def run_validation(df):
        ...     return th.check(df)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tenant = TenantContext.get_current_tenant()
            if not tenant:
                return func(*args, **kwargs)

            manager = get_tenant_manager()

            # Check quotas
            if check_quotas:
                manager.quota_enforcer.require(tenant, ResourceType.VALIDATIONS)

            # Apply isolation to DataFrame arguments
            if apply_isolation:
                new_args = []
                for arg in args:
                    if hasattr(arg, "lazy"):
                        arg = manager.apply_isolation(arg.lazy(), tenant)
                    elif hasattr(arg, "collect_schema"):
                        arg = manager.apply_isolation(arg, tenant)
                    new_args.append(arg)
                args = tuple(new_args)

            try:
                result = func(*args, **kwargs)

                # Track usage on success
                if track_usage:
                    manager.track_usage(ResourceType.VALIDATIONS, 1, tenant)

                return result
            except Exception:
                raise

        return wrapper  # type: ignore

    return decorator


def tenant_checkpoint(
    track_usage: bool = True,
    check_quotas: bool = True,
) -> Callable[[F], F]:
    """Decorator for tenant-aware checkpoint functions.

    Example:
        >>> @tenant_checkpoint()
        ... def run_checkpoint():
        ...     return checkpoint.run()
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tenant = TenantContext.get_current_tenant()
            if not tenant:
                return func(*args, **kwargs)

            manager = get_tenant_manager()

            # Check quotas
            if check_quotas:
                manager.quota_enforcer.require(tenant, ResourceType.CHECKPOINTS)

            try:
                result = func(*args, **kwargs)

                # Track usage on success
                if track_usage:
                    manager.track_usage(ResourceType.CHECKPOINTS, 1, tenant)

                return result
            except Exception:
                raise

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# Truthound API Integration
# =============================================================================


def tenant_check(
    data: Any,
    tenant: Tenant | str | None = None,
    **kwargs: Any,
) -> "Report":
    """Tenant-aware version of th.check().

    Args:
        data: Data to validate
        tenant: Tenant (uses current if not provided)
        **kwargs: Arguments passed to th.check()

    Returns:
        Validation report.

    Example:
        >>> from truthound.multitenancy.integration import tenant_check
        >>> report = tenant_check(df, tenant="acme_corp")
    """
    import truthound as th

    manager = get_tenant_manager()

    if tenant is not None:
        if isinstance(tenant, str):
            tenant = manager.require(tenant)
        ctx = TenantContext.set_current(tenant)
        ctx.__enter__()
    else:
        tenant = TenantContext.get_current_tenant()
        ctx = None

    try:
        # Apply isolation
        if tenant and hasattr(data, "lazy"):
            data = manager.apply_isolation(data.lazy(), tenant)
        elif tenant and hasattr(data, "collect_schema"):
            data = manager.apply_isolation(data, tenant)

        # Check quota
        if tenant:
            manager.quota_enforcer.require(tenant, ResourceType.VALIDATIONS)

        # Run validation
        report = th.check(data, **kwargs)

        # Track usage
        if tenant:
            manager.track_usage(ResourceType.VALIDATIONS, 1, tenant)

        return report

    finally:
        if ctx:
            ctx.__exit__(None, None, None)


def tenant_compare(
    baseline: Any,
    current: Any,
    tenant: Tenant | str | None = None,
    **kwargs: Any,
) -> Any:
    """Tenant-aware version of th.compare().

    Args:
        baseline: Baseline data
        current: Current data to compare
        tenant: Tenant (uses current if not provided)
        **kwargs: Arguments passed to th.compare()

    Returns:
        Comparison result.
    """
    import truthound as th

    manager = get_tenant_manager()

    if tenant is not None:
        if isinstance(tenant, str):
            tenant = manager.require(tenant)
        ctx = TenantContext.set_current(tenant)
        ctx.__enter__()
    else:
        tenant = TenantContext.get_current_tenant()
        ctx = None

    try:
        # Apply isolation to both datasets
        if tenant:
            if hasattr(baseline, "lazy"):
                baseline = manager.apply_isolation(baseline.lazy(), tenant)
            elif hasattr(baseline, "collect_schema"):
                baseline = manager.apply_isolation(baseline, tenant)

            if hasattr(current, "lazy"):
                current = manager.apply_isolation(current.lazy(), tenant)
            elif hasattr(current, "collect_schema"):
                current = manager.apply_isolation(current, tenant)

        # Check quota
        if tenant:
            manager.quota_enforcer.require(tenant, ResourceType.VALIDATIONS)

        # Run comparison
        result = th.compare(baseline, current, **kwargs)

        # Track usage
        if tenant:
            manager.track_usage(ResourceType.VALIDATIONS, 1, tenant)

        return result

    finally:
        if ctx:
            ctx.__exit__(None, None, None)
