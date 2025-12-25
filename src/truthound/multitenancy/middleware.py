"""Middleware and decorators for multi-tenancy.

This module provides middleware for web frameworks and decorators
for tenant-aware function execution.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar

from truthound.multitenancy.core import (
    Tenant,
    TenantContext,
    TenantError,
    TenantNotFoundError,
    TenantAccessDeniedError,
    TenantSuspendedError,
    TenantStore,
    TenantResolver,
    ResourceType,
)
from truthound.multitenancy.isolation import IsolationManager
from truthound.multitenancy.quota import QuotaEnforcer, QuotaTracker


F = TypeVar("F", bound=Callable[..., Any])
AsyncF = TypeVar("AsyncF", bound=Callable[..., Awaitable[Any]])


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TenantMiddlewareConfig:
    """Configuration for tenant middleware."""

    # Resolution
    require_tenant: bool = True
    allow_anonymous: bool = False
    anonymous_tenant_id: str = "anonymous"

    # Validation
    check_status: bool = True
    check_quota: bool = True

    # Logging
    log_tenant_access: bool = True
    log_resolution_time: bool = False

    # Error handling
    raise_on_error: bool = True
    error_response_code: int = 403

    # Headers
    tenant_id_header: str = "X-Tenant-ID"
    tenant_name_header: str = "X-Tenant-Name"
    add_headers_to_response: bool = True


# =============================================================================
# Tenant Middleware Base
# =============================================================================


class TenantMiddleware:
    """Base middleware for tenant resolution and context management.

    This middleware handles:
    1. Resolving tenant from request context
    2. Validating tenant status
    3. Setting up tenant context for the request
    4. Enforcing quotas (optional)
    5. Adding tenant info to response headers

    Example:
        >>> middleware = TenantMiddleware(
        ...     store=tenant_store,
        ...     resolver=header_resolver,
        ... )
    """

    def __init__(
        self,
        store: TenantStore,
        resolver: TenantResolver,
        config: TenantMiddlewareConfig | None = None,
        quota_enforcer: QuotaEnforcer | None = None,
    ) -> None:
        self._store = store
        self._resolver = resolver
        self._config = config or TenantMiddlewareConfig()
        self._quota_enforcer = quota_enforcer

    def resolve_tenant(self, context: dict[str, Any]) -> Tenant | None:
        """Resolve tenant from request context."""
        start_time = time.time()

        try:
            tenant_id = self._resolver.resolve(context)
            if not tenant_id:
                if self._config.allow_anonymous:
                    return self._get_anonymous_tenant()
                if self._config.require_tenant:
                    raise TenantNotFoundError("No tenant ID in request")
                return None

            tenant = self._store.get(tenant_id)
            if not tenant:
                # Try by slug
                tenant = self._store.get_by_slug(tenant_id)

            if not tenant:
                raise TenantNotFoundError(
                    f"Tenant not found: {tenant_id}",
                    tenant_id=tenant_id,
                )

            return tenant

        finally:
            if self._config.log_resolution_time:
                elapsed = (time.time() - start_time) * 1000
                # Logging would go here
                pass

    def validate_tenant(self, tenant: Tenant) -> None:
        """Validate tenant status."""
        if not self._config.check_status:
            return

        if tenant.is_suspended:
            raise TenantSuspendedError(
                f"Tenant {tenant.id} is suspended",
                tenant_id=tenant.id,
            )

        if not tenant.is_active:
            raise TenantAccessDeniedError(
                f"Tenant {tenant.id} is not active (status: {tenant.status.value})",
                tenant_id=tenant.id,
            )

    def check_quota(
        self,
        tenant: Tenant,
        resource_type: ResourceType = ResourceType.API_CALLS,
    ) -> None:
        """Check quota before processing request."""
        if not self._config.check_quota or not self._quota_enforcer:
            return

        self._quota_enforcer.require(tenant, resource_type)

    def track_usage(
        self,
        tenant: Tenant,
        resource_type: ResourceType = ResourceType.API_CALLS,
    ) -> None:
        """Track resource usage after processing request."""
        if self._quota_enforcer:
            self._quota_enforcer.track(tenant, resource_type)

    def _get_anonymous_tenant(self) -> Tenant:
        """Get or create anonymous tenant."""
        tenant = self._store.get(self._config.anonymous_tenant_id)
        if not tenant:
            from truthound.multitenancy.core import TenantStatus, TenantTier
            tenant = Tenant(
                id=self._config.anonymous_tenant_id,
                name="Anonymous",
                status=TenantStatus.ACTIVE,
                tier=TenantTier.FREE,
            )
        return tenant

    def get_response_headers(self, tenant: Tenant) -> dict[str, str]:
        """Get headers to add to response."""
        if not self._config.add_headers_to_response:
            return {}

        return {
            self._config.tenant_id_header: tenant.id,
            self._config.tenant_name_header: tenant.name,
        }


# =============================================================================
# ASGI Middleware
# =============================================================================


class ASGITenantMiddleware:
    """ASGI middleware for tenant context management.

    Compatible with FastAPI, Starlette, and other ASGI frameworks.

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> app.add_middleware(
        ...     ASGITenantMiddleware,
        ...     store=tenant_store,
        ...     resolver=header_resolver,
        ... )
    """

    def __init__(
        self,
        app: Any,
        store: TenantStore,
        resolver: TenantResolver,
        config: TenantMiddlewareConfig | None = None,
        quota_enforcer: QuotaEnforcer | None = None,
    ) -> None:
        self.app = app
        self._middleware = TenantMiddleware(
            store=store,
            resolver=resolver,
            config=config,
            quota_enforcer=quota_enforcer,
        )
        self._config = config or TenantMiddlewareConfig()

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Awaitable[Any]],
        send: Callable[..., Awaitable[Any]],
    ) -> None:
        """Process ASGI request."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Build context from ASGI scope
        context = self._build_context(scope)

        try:
            # Resolve and validate tenant
            tenant = self._middleware.resolve_tenant(context)

            if tenant:
                # Validate tenant status
                self._middleware.validate_tenant(tenant)

                # Check quota
                self._middleware.check_quota(tenant)

                # Set up tenant context and process request
                with TenantContext.set_current(tenant):
                    # Wrap send to add headers
                    async def send_with_headers(message: dict[str, Any]) -> None:
                        if message["type"] == "http.response.start":
                            headers = dict(message.get("headers", []))
                            for key, value in self._middleware.get_response_headers(
                                tenant
                            ).items():
                                headers[key.lower().encode()] = value.encode()
                            message["headers"] = list(headers.items())
                        await send(message)

                    await self.app(scope, receive, send_with_headers)

                    # Track usage after successful request
                    self._middleware.track_usage(tenant)
            else:
                await self.app(scope, receive, send)

        except TenantError as e:
            if self._config.raise_on_error:
                await self._send_error_response(
                    send,
                    self._config.error_response_code,
                    str(e),
                )
            else:
                await self.app(scope, receive, send)

    def _build_context(self, scope: dict[str, Any]) -> dict[str, Any]:
        """Build context dict from ASGI scope."""
        headers = {}
        for key, value in scope.get("headers", []):
            headers[key.decode()] = value.decode()

        return {
            "headers": headers,
            "path": scope.get("path", ""),
            "method": scope.get("method", ""),
            "query_string": scope.get("query_string", b"").decode(),
            "host": headers.get("host", ""),
        }

    async def _send_error_response(
        self,
        send: Callable[..., Awaitable[Any]],
        status_code: int,
        message: str,
    ) -> None:
        """Send error response."""
        import json

        body = json.dumps({"error": message}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
            }
        )


# =============================================================================
# WSGI Middleware
# =============================================================================


class WSGITenantMiddleware:
    """WSGI middleware for tenant context management.

    Compatible with Flask, Django, and other WSGI frameworks.

    Example:
        >>> from flask import Flask
        >>> app = Flask(__name__)
        >>> app.wsgi_app = WSGITenantMiddleware(
        ...     app.wsgi_app,
        ...     store=tenant_store,
        ...     resolver=header_resolver,
        ... )
    """

    def __init__(
        self,
        app: Any,
        store: TenantStore,
        resolver: TenantResolver,
        config: TenantMiddlewareConfig | None = None,
        quota_enforcer: QuotaEnforcer | None = None,
    ) -> None:
        self.app = app
        self._middleware = TenantMiddleware(
            store=store,
            resolver=resolver,
            config=config,
            quota_enforcer=quota_enforcer,
        )
        self._config = config or TenantMiddlewareConfig()

    def __call__(
        self,
        environ: dict[str, Any],
        start_response: Callable[..., Any],
    ) -> Any:
        """Process WSGI request."""
        # Build context from environ
        context = self._build_context(environ)

        try:
            # Resolve and validate tenant
            tenant = self._middleware.resolve_tenant(context)

            if tenant:
                # Validate tenant status
                self._middleware.validate_tenant(tenant)

                # Check quota
                self._middleware.check_quota(tenant)

                # Set up tenant context
                with TenantContext.set_current(tenant):
                    # Wrap start_response to add headers
                    def start_response_with_headers(
                        status: str,
                        headers: list[tuple[str, str]],
                        exc_info: Any = None,
                    ) -> Any:
                        for key, value in self._middleware.get_response_headers(
                            tenant
                        ).items():
                            headers.append((key, value))
                        return start_response(status, headers, exc_info)

                    result = self.app(environ, start_response_with_headers)

                    # Track usage
                    self._middleware.track_usage(tenant)

                    return result
            else:
                return self.app(environ, start_response)

        except TenantError as e:
            if self._config.raise_on_error:
                return self._error_response(
                    start_response,
                    self._config.error_response_code,
                    str(e),
                )
            else:
                return self.app(environ, start_response)

    def _build_context(self, environ: dict[str, Any]) -> dict[str, Any]:
        """Build context dict from WSGI environ."""
        headers = {}
        for key, value in environ.items():
            if key.startswith("HTTP_"):
                header_name = key[5:].replace("_", "-").title()
                headers[header_name] = value

        return {
            "headers": headers,
            "path": environ.get("PATH_INFO", ""),
            "method": environ.get("REQUEST_METHOD", ""),
            "query_string": environ.get("QUERY_STRING", ""),
            "host": environ.get("HTTP_HOST", ""),
        }

    def _error_response(
        self,
        start_response: Callable[..., Any],
        status_code: int,
        message: str,
    ) -> list[bytes]:
        """Generate error response."""
        import json

        body = json.dumps({"error": message})
        status = f"{status_code} {'Forbidden' if status_code == 403 else 'Error'}"
        headers = [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
        ]
        start_response(status, headers)
        return [body.encode()]


# =============================================================================
# Decorators
# =============================================================================


def tenant_required(
    store: TenantStore | None = None,
    resolver: TenantResolver | None = None,
    resource_type: ResourceType | None = None,
    quota_enforcer: QuotaEnforcer | None = None,
) -> Callable[[F], F]:
    """Decorator that requires a valid tenant context.

    Example:
        >>> @tenant_required()
        ... def process_data(data):
        ...     tenant = TenantContext.require_current_tenant()
        ...     return validate(data, tenant)

        >>> @tenant_required(resource_type=ResourceType.VALIDATIONS)
        ... def run_validation(data):
        ...     return validate(data)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tenant = TenantContext.get_current_tenant()

            if not tenant:
                raise TenantError("No tenant context set")

            if not tenant.is_active:
                raise TenantAccessDeniedError(
                    f"Tenant {tenant.id} is not active",
                    tenant_id=tenant.id,
                )

            # Check quota if specified
            if resource_type and quota_enforcer:
                quota_enforcer.require(tenant, resource_type)

            try:
                result = func(*args, **kwargs)

                # Track usage on success
                if resource_type and quota_enforcer:
                    quota_enforcer.track(tenant, resource_type)

                return result

            except Exception:
                raise

        return wrapper  # type: ignore

    return decorator


def tenant_required_async(
    store: TenantStore | None = None,
    resolver: TenantResolver | None = None,
    resource_type: ResourceType | None = None,
    quota_enforcer: QuotaEnforcer | None = None,
) -> Callable[[AsyncF], AsyncF]:
    """Async decorator that requires a valid tenant context.

    Example:
        >>> @tenant_required_async()
        ... async def process_data(data):
        ...     tenant = TenantContext.require_current_tenant()
        ...     return await validate(data, tenant)
    """

    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tenant = TenantContext.get_current_tenant()

            if not tenant:
                raise TenantError("No tenant context set")

            if not tenant.is_active:
                raise TenantAccessDeniedError(
                    f"Tenant {tenant.id} is not active",
                    tenant_id=tenant.id,
                )

            # Check quota if specified
            if resource_type and quota_enforcer:
                quota_enforcer.require(tenant, resource_type)

            try:
                result = await func(*args, **kwargs)

                # Track usage on success
                if resource_type and quota_enforcer:
                    quota_enforcer.track(tenant, resource_type)

                return result

            except Exception:
                raise

        return wrapper  # type: ignore

    return decorator


def with_tenant(tenant: Tenant) -> Callable[[F], F]:
    """Decorator that sets tenant context for a function.

    Example:
        >>> @with_tenant(my_tenant)
        ... def process_data(data):
        ...     return validate(data)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with TenantContext.set_current(tenant):
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def with_tenant_async(tenant: Tenant) -> Callable[[AsyncF], AsyncF]:
    """Async decorator that sets tenant context for a function.

    Example:
        >>> @with_tenant_async(my_tenant)
        ... async def process_data(data):
        ...     return await validate(data)
    """

    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with TenantContext.set_current(tenant):
                return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def tenant_isolated(
    isolation_manager: IsolationManager | None = None,
) -> Callable[[F], F]:
    """Decorator that applies tenant isolation to DataFrame operations.

    Example:
        >>> @tenant_isolated()
        ... def query_data(df):
        ...     # df will be automatically filtered by tenant
        ...     return df.collect()
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tenant = TenantContext.get_current_tenant()
            if not tenant:
                return func(*args, **kwargs)

            # Apply isolation to any DataFrame arguments
            if isolation_manager:
                new_args = []
                for arg in args:
                    if hasattr(arg, "lazy"):  # Polars DataFrame
                        arg = isolation_manager.apply_filter(arg.lazy(), tenant)
                    elif hasattr(arg, "collect_schema"):  # LazyFrame
                        arg = isolation_manager.apply_filter(arg, tenant)
                    new_args.append(arg)
                args = tuple(new_args)

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# Flask Integration
# =============================================================================


def create_flask_tenant_middleware(
    store: TenantStore,
    resolver: TenantResolver,
    config: TenantMiddlewareConfig | None = None,
    quota_enforcer: QuotaEnforcer | None = None,
) -> Callable[[Any], Any]:
    """Create Flask before_request handler for tenant context.

    Example:
        >>> from flask import Flask, g
        >>> app = Flask(__name__)
        >>> handler = create_flask_tenant_middleware(store, resolver)
        >>> app.before_request(handler)
    """
    middleware = TenantMiddleware(
        store=store,
        resolver=resolver,
        config=config,
        quota_enforcer=quota_enforcer,
    )
    _config = config or TenantMiddlewareConfig()

    def before_request() -> Any:
        from flask import request, g, abort

        context = {
            "headers": dict(request.headers),
            "path": request.path,
            "method": request.method,
            "host": request.host,
        }

        try:
            tenant = middleware.resolve_tenant(context)
            if tenant:
                middleware.validate_tenant(tenant)
                middleware.check_quota(tenant)
                g.tenant = tenant
                g.tenant_context = TenantContext.set_current(tenant).__enter__()
            else:
                g.tenant = None
                g.tenant_context = None
        except TenantError as e:
            if _config.raise_on_error:
                abort(_config.error_response_code, str(e))

    return before_request


def create_flask_tenant_teardown(
    quota_enforcer: QuotaEnforcer | None = None,
) -> Callable[[Any], None]:
    """Create Flask teardown handler for tenant context cleanup.

    Example:
        >>> teardown = create_flask_tenant_teardown(quota_enforcer)
        >>> app.teardown_request(teardown)
    """

    def teardown_request(exception: Any = None) -> None:
        from flask import g

        if hasattr(g, "tenant_context") and g.tenant_context:
            g.tenant_context.__exit__(None, None, None)

        if exception is None and quota_enforcer and hasattr(g, "tenant") and g.tenant:
            quota_enforcer.track(g.tenant, ResourceType.API_CALLS)

    return teardown_request


# =============================================================================
# FastAPI Integration
# =============================================================================


def create_fastapi_tenant_dependency(
    store: TenantStore,
    resolver: TenantResolver,
    config: TenantMiddlewareConfig | None = None,
    quota_enforcer: QuotaEnforcer | None = None,
) -> Callable[..., Tenant]:
    """Create FastAPI dependency for tenant injection.

    Example:
        >>> from fastapi import FastAPI, Depends
        >>> app = FastAPI()
        >>> get_tenant = create_fastapi_tenant_dependency(store, resolver)
        >>>
        >>> @app.get("/data")
        >>> async def get_data(tenant: Tenant = Depends(get_tenant)):
        ...     return {"tenant": tenant.name}
    """
    middleware = TenantMiddleware(
        store=store,
        resolver=resolver,
        config=config,
        quota_enforcer=quota_enforcer,
    )

    async def get_tenant(request: Any) -> Tenant:
        from fastapi import HTTPException

        context = {
            "headers": dict(request.headers),
            "path": request.url.path,
            "method": request.method,
            "host": request.headers.get("host", ""),
        }

        try:
            tenant = middleware.resolve_tenant(context)
            if not tenant:
                raise HTTPException(status_code=403, detail="Tenant required")

            middleware.validate_tenant(tenant)
            middleware.check_quota(tenant)

            return tenant

        except TenantError as e:
            raise HTTPException(status_code=403, detail=str(e))

    return get_tenant
