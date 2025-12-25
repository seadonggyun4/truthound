"""Middleware and decorators for RBAC.

This module provides middleware for web frameworks and decorators
for permission-protected function execution.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar

from truthound.rbac.core import (
    AccessContext,
    AccessDecision,
    Permission,
    PermissionAction,
    PermissionDeniedError,
    Principal,
    PrincipalStore,
    ResourceType,
    RoleStore,
    SecurityContext,
)
from truthound.rbac.policy import PolicyEngine


F = TypeVar("F", bound=Callable[..., Any])
AsyncF = TypeVar("AsyncF", bound=Callable[..., Awaitable[Any]])


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RBACMiddlewareConfig:
    """Configuration for RBAC middleware."""

    # Principal resolution
    principal_header: str = "X-Principal-ID"
    principal_from_jwt: bool = True
    jwt_claim: str = "sub"

    # Default behavior
    deny_anonymous: bool = True
    allow_public_paths: list[str] = field(
        default_factory=lambda: ["/health", "/ready", "/metrics"]
    )

    # Error handling
    raise_on_denied: bool = True
    denied_status_code: int = 403

    # Logging
    log_access_decisions: bool = True

    # Audit
    audit_decisions: bool = True


# =============================================================================
# RBAC Middleware Base
# =============================================================================


class RBACMiddleware:
    """Base middleware for RBAC enforcement.

    Example:
        >>> middleware = RBACMiddleware(
        ...     engine=policy_engine,
        ...     principal_store=principal_store,
        ... )
    """

    def __init__(
        self,
        engine: PolicyEngine,
        principal_store: PrincipalStore,
        config: RBACMiddlewareConfig | None = None,
    ) -> None:
        self._engine = engine
        self._principal_store = principal_store
        self._config = config or RBACMiddlewareConfig()

    def resolve_principal(self, context: dict[str, Any]) -> Principal | None:
        """Resolve principal from request context."""
        headers = context.get("headers", {})

        # Try header-based resolution
        principal_id = headers.get(self._config.principal_header)
        if principal_id:
            return self._principal_store.get(principal_id)

        # Try JWT-based resolution
        if self._config.principal_from_jwt:
            auth_header = headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                principal_id = self._extract_from_jwt(auth_header[7:])
                if principal_id:
                    return self._principal_store.get(principal_id)

        return None

    def _extract_from_jwt(self, token: str) -> str | None:
        """Extract principal ID from JWT token."""
        import base64
        import json

        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            payload = parts[1]
            # Add padding
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += "=" * padding

            decoded = base64.urlsafe_b64decode(payload)
            claims = json.loads(decoded)
            return claims.get(self._config.jwt_claim)
        except Exception:
            return None

    def is_public_path(self, path: str) -> bool:
        """Check if the path is public (no auth required)."""
        for public_path in self._config.allow_public_paths:
            if path.startswith(public_path):
                return True
        return False

    def check_access(
        self,
        principal: Principal | None,
        resource: str,
        action: str,
        resource_attributes: dict[str, Any] | None = None,
    ) -> AccessDecision:
        """Check access for the principal."""
        if principal is None:
            if self._config.deny_anonymous:
                return AccessDecision.deny("Anonymous access denied")
            principal = Principal.anonymous()

        return self._engine.check(
            principal=principal,
            resource=resource,
            action=action,
            resource_attributes=resource_attributes,
        )


# =============================================================================
# ASGI Middleware
# =============================================================================


class ASGIRBACMiddleware:
    """ASGI middleware for RBAC enforcement.

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> app.add_middleware(
        ...     ASGIRBACMiddleware,
        ...     engine=policy_engine,
        ...     principal_store=principal_store,
        ... )
    """

    def __init__(
        self,
        app: Any,
        engine: PolicyEngine,
        principal_store: PrincipalStore,
        config: RBACMiddlewareConfig | None = None,
        resource_mapper: Callable[[str, str], tuple[str, str]] | None = None,
    ) -> None:
        self.app = app
        self._middleware = RBACMiddleware(
            engine=engine,
            principal_store=principal_store,
            config=config,
        )
        self._config = config or RBACMiddlewareConfig()
        self._resource_mapper = resource_mapper or self._default_resource_mapper

    def _default_resource_mapper(self, path: str, method: str) -> tuple[str, str]:
        """Default resource/action mapper from HTTP path and method."""
        # Map HTTP method to action
        method_to_action = {
            "GET": "read",
            "HEAD": "read",
            "POST": "create",
            "PUT": "update",
            "PATCH": "update",
            "DELETE": "delete",
        }
        action = method_to_action.get(method.upper(), "read")

        # Use path as resource
        resource = path.strip("/").replace("/", ":")
        if not resource:
            resource = "root"

        return resource, action

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

        path = scope.get("path", "")
        method = scope.get("method", "GET")

        # Check public paths
        if self._middleware.is_public_path(path):
            await self.app(scope, receive, send)
            return

        # Build context
        headers = {}
        for key, value in scope.get("headers", []):
            headers[key.decode()] = value.decode()

        context = {"headers": headers, "path": path, "method": method}

        # Resolve principal
        principal = self._middleware.resolve_principal(context)

        # Map to resource/action
        resource, action = self._resource_mapper(path, method)

        # Check access
        decision = self._middleware.check_access(principal, resource, action)

        if not decision.allowed:
            await self._send_error_response(
                send,
                self._config.denied_status_code,
                decision.reason,
            )
            return

        # Set security context and continue
        if principal:
            with SecurityContext.set_principal(principal):
                await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)

    async def _send_error_response(
        self,
        send: Callable[..., Awaitable[Any]],
        status_code: int,
        message: str,
    ) -> None:
        """Send error response."""
        import json

        body = json.dumps({"error": "Permission denied", "message": message}).encode()
        await send({
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body,
        })


# =============================================================================
# WSGI Middleware
# =============================================================================


class WSGIRBACMiddleware:
    """WSGI middleware for RBAC enforcement.

    Example:
        >>> from flask import Flask
        >>> app = Flask(__name__)
        >>> app.wsgi_app = WSGIRBACMiddleware(
        ...     app.wsgi_app,
        ...     engine=policy_engine,
        ...     principal_store=principal_store,
        ... )
    """

    def __init__(
        self,
        app: Any,
        engine: PolicyEngine,
        principal_store: PrincipalStore,
        config: RBACMiddlewareConfig | None = None,
        resource_mapper: Callable[[str, str], tuple[str, str]] | None = None,
    ) -> None:
        self.app = app
        self._middleware = RBACMiddleware(
            engine=engine,
            principal_store=principal_store,
            config=config,
        )
        self._config = config or RBACMiddlewareConfig()
        self._resource_mapper = resource_mapper or self._default_resource_mapper

    def _default_resource_mapper(self, path: str, method: str) -> tuple[str, str]:
        """Default resource/action mapper."""
        method_to_action = {
            "GET": "read",
            "HEAD": "read",
            "POST": "create",
            "PUT": "update",
            "PATCH": "update",
            "DELETE": "delete",
        }
        action = method_to_action.get(method.upper(), "read")
        resource = path.strip("/").replace("/", ":") or "root"
        return resource, action

    def __call__(
        self,
        environ: dict[str, Any],
        start_response: Callable[..., Any],
    ) -> Any:
        """Process WSGI request."""
        path = environ.get("PATH_INFO", "")
        method = environ.get("REQUEST_METHOD", "GET")

        # Check public paths
        if self._middleware.is_public_path(path):
            return self.app(environ, start_response)

        # Build context
        headers = {}
        for key, value in environ.items():
            if key.startswith("HTTP_"):
                header_name = key[5:].replace("_", "-").title()
                headers[header_name] = value

        context = {"headers": headers, "path": path, "method": method}

        # Resolve principal
        principal = self._middleware.resolve_principal(context)

        # Map to resource/action
        resource, action = self._resource_mapper(path, method)

        # Check access
        decision = self._middleware.check_access(principal, resource, action)

        if not decision.allowed:
            return self._error_response(
                start_response,
                self._config.denied_status_code,
                decision.reason,
            )

        # Set security context and continue
        if principal:
            with SecurityContext.set_principal(principal):
                return self.app(environ, start_response)
        else:
            return self.app(environ, start_response)

    def _error_response(
        self,
        start_response: Callable[..., Any],
        status_code: int,
        message: str,
    ) -> list[bytes]:
        """Generate error response."""
        import json

        body = json.dumps({"error": "Permission denied", "message": message})
        status = f"{status_code} Forbidden"
        headers = [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
        ]
        start_response(status, headers)
        return [body.encode()]


# =============================================================================
# Decorators
# =============================================================================


def require_permission(
    resource: str | ResourceType,
    action: str | PermissionAction,
    resource_attributes: dict[str, Any] | None = None,
    engine: PolicyEngine | None = None,
) -> Callable[[F], F]:
    """Decorator that requires a specific permission.

    Example:
        >>> @require_permission("dataset", "read")
        ... def get_dataset(dataset_id: str):
        ...     return load_dataset(dataset_id)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            principal = SecurityContext.get_current_principal()
            if principal is None:
                raise PermissionDeniedError("No authenticated principal")

            # Use provided engine or get from somewhere
            policy_engine = engine
            if policy_engine is None:
                policy_engine = _get_default_engine()

            # Normalize resource and action
            res = resource.value if isinstance(resource, ResourceType) else resource
            act = action.value if isinstance(action, PermissionAction) else action

            # Check permission
            decision = policy_engine.check(
                principal=principal,
                resource=res,
                action=act,
                resource_attributes=resource_attributes,
            )

            if not decision.allowed:
                raise PermissionDeniedError(
                    decision.reason,
                    principal_id=principal.id,
                    resource=res,
                    action=act,
                )

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def require_permission_async(
    resource: str | ResourceType,
    action: str | PermissionAction,
    resource_attributes: dict[str, Any] | None = None,
    engine: PolicyEngine | None = None,
) -> Callable[[AsyncF], AsyncF]:
    """Async decorator that requires a specific permission.

    Example:
        >>> @require_permission_async("dataset", "read")
        ... async def get_dataset(dataset_id: str):
        ...     return await load_dataset(dataset_id)
    """

    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            principal = SecurityContext.get_current_principal()
            if principal is None:
                raise PermissionDeniedError("No authenticated principal")

            policy_engine = engine
            if policy_engine is None:
                policy_engine = _get_default_engine()

            res = resource.value if isinstance(resource, ResourceType) else resource
            act = action.value if isinstance(action, PermissionAction) else action

            decision = policy_engine.check(
                principal=principal,
                resource=res,
                action=act,
                resource_attributes=resource_attributes,
            )

            if not decision.allowed:
                raise PermissionDeniedError(
                    decision.reason,
                    principal_id=principal.id,
                    resource=res,
                    action=act,
                )

            return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def require_role(
    role: str | set[str],
    require_all: bool = False,
) -> Callable[[F], F]:
    """Decorator that requires specific role(s).

    Example:
        >>> @require_role("admin")
        ... def admin_function():
        ...     pass

        >>> @require_role({"editor", "admin"}, require_all=False)
        ... def editor_or_admin():
        ...     pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            principal = SecurityContext.get_current_principal()
            if principal is None:
                raise PermissionDeniedError("No authenticated principal")

            required_roles = {role} if isinstance(role, str) else role

            if require_all:
                # Must have all roles
                if not required_roles.issubset(principal.roles):
                    missing = required_roles - principal.roles
                    raise PermissionDeniedError(
                        f"Missing required roles: {missing}",
                        principal_id=principal.id,
                    )
            else:
                # Must have at least one role
                if not required_roles.intersection(principal.roles):
                    raise PermissionDeniedError(
                        f"Requires one of roles: {required_roles}",
                        principal_id=principal.id,
                    )

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def require_role_async(
    role: str | set[str],
    require_all: bool = False,
) -> Callable[[AsyncF], AsyncF]:
    """Async decorator that requires specific role(s)."""

    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            principal = SecurityContext.get_current_principal()
            if principal is None:
                raise PermissionDeniedError("No authenticated principal")

            required_roles = {role} if isinstance(role, str) else role

            if require_all:
                if not required_roles.issubset(principal.roles):
                    missing = required_roles - principal.roles
                    raise PermissionDeniedError(
                        f"Missing required roles: {missing}",
                        principal_id=principal.id,
                    )
            else:
                if not required_roles.intersection(principal.roles):
                    raise PermissionDeniedError(
                        f"Requires one of roles: {required_roles}",
                        principal_id=principal.id,
                    )

            return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def with_principal(principal: Principal) -> Callable[[F], F]:
    """Decorator that sets principal context for a function.

    Example:
        >>> @with_principal(admin_principal)
        ... def admin_operation():
        ...     pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with SecurityContext.set_principal(principal):
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def with_principal_async(principal: Principal) -> Callable[[AsyncF], AsyncF]:
    """Async decorator that sets principal context."""

    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with SecurityContext.set_principal(principal):
                return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# Flask Integration
# =============================================================================


def create_flask_rbac_middleware(
    engine: PolicyEngine,
    principal_store: PrincipalStore,
    config: RBACMiddlewareConfig | None = None,
    resource_mapper: Callable[[str, str], tuple[str, str]] | None = None,
) -> Callable[[Any], Any]:
    """Create Flask before_request handler for RBAC.

    Example:
        >>> from flask import Flask
        >>> app = Flask(__name__)
        >>> handler = create_flask_rbac_middleware(engine, principal_store)
        >>> app.before_request(handler)
    """
    middleware = RBACMiddleware(
        engine=engine,
        principal_store=principal_store,
        config=config,
    )
    _config = config or RBACMiddlewareConfig()

    def _default_mapper(path: str, method: str) -> tuple[str, str]:
        method_to_action = {
            "GET": "read", "POST": "create", "PUT": "update",
            "PATCH": "update", "DELETE": "delete",
        }
        action = method_to_action.get(method.upper(), "read")
        resource = path.strip("/").replace("/", ":") or "root"
        return resource, action

    mapper = resource_mapper or _default_mapper

    def before_request() -> Any:
        from flask import request, g, abort

        path = request.path
        method = request.method

        # Check public paths
        if middleware.is_public_path(path):
            return None

        context = {
            "headers": dict(request.headers),
            "path": path,
            "method": method,
        }

        # Resolve principal
        principal = middleware.resolve_principal(context)

        # Map to resource/action
        resource, action = mapper(path, method)

        # Check access
        decision = middleware.check_access(principal, resource, action)

        if not decision.allowed:
            if _config.raise_on_denied:
                abort(_config.denied_status_code, decision.reason)
            return None

        # Store principal in g
        g.principal = principal
        if principal:
            g.security_context = SecurityContext.set_principal(principal).__enter__()
        else:
            g.security_context = None

        return None

    return before_request


def create_flask_rbac_teardown(
) -> Callable[[Any], None]:
    """Create Flask teardown handler for RBAC cleanup."""

    def teardown_request(exception: Any = None) -> None:
        from flask import g

        if hasattr(g, "security_context") and g.security_context:
            g.security_context.__exit__(None, None, None)

    return teardown_request


# =============================================================================
# FastAPI Integration
# =============================================================================


def create_fastapi_rbac_dependency(
    engine: PolicyEngine,
    principal_store: PrincipalStore,
    config: RBACMiddlewareConfig | None = None,
) -> Callable[..., Principal]:
    """Create FastAPI dependency for principal injection.

    Example:
        >>> from fastapi import FastAPI, Depends
        >>> app = FastAPI()
        >>> get_principal = create_fastapi_rbac_dependency(engine, principal_store)
        >>>
        >>> @app.get("/data")
        >>> async def get_data(principal: Principal = Depends(get_principal)):
        ...     return {"user": principal.name}
    """
    middleware = RBACMiddleware(
        engine=engine,
        principal_store=principal_store,
        config=config,
    )
    _config = config or RBACMiddlewareConfig()

    async def get_principal(request: Any) -> Principal:
        from fastapi import HTTPException

        context = {
            "headers": dict(request.headers),
            "path": request.url.path,
            "method": request.method,
        }

        principal = middleware.resolve_principal(context)

        if principal is None:
            if _config.deny_anonymous:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required",
                )
            principal = Principal.anonymous()

        return principal

    return get_principal


def create_fastapi_permission_dependency(
    engine: PolicyEngine,
    principal_store: PrincipalStore,
    resource: str,
    action: str,
    config: RBACMiddlewareConfig | None = None,
) -> Callable[..., Principal]:
    """Create FastAPI dependency that requires a specific permission.

    Example:
        >>> require_read = create_fastapi_permission_dependency(
        ...     engine, principal_store, "dataset", "read"
        ... )
        >>>
        >>> @app.get("/datasets")
        >>> async def list_datasets(principal: Principal = Depends(require_read)):
        ...     return get_datasets()
    """
    middleware = RBACMiddleware(
        engine=engine,
        principal_store=principal_store,
        config=config,
    )
    _config = config or RBACMiddlewareConfig()

    async def check_permission(request: Any) -> Principal:
        from fastapi import HTTPException

        context = {
            "headers": dict(request.headers),
            "path": request.url.path,
            "method": request.method,
        }

        principal = middleware.resolve_principal(context)

        if principal is None:
            if _config.deny_anonymous:
                raise HTTPException(status_code=401, detail="Authentication required")
            principal = Principal.anonymous()

        decision = engine.check(principal, resource, action)

        if not decision.allowed:
            raise HTTPException(
                status_code=_config.denied_status_code,
                detail=decision.reason,
            )

        return principal

    return check_permission


# =============================================================================
# Default Engine Management
# =============================================================================


_default_engine: PolicyEngine | None = None


def set_default_engine(engine: PolicyEngine) -> None:
    """Set the default policy engine."""
    global _default_engine
    _default_engine = engine


def _get_default_engine() -> PolicyEngine:
    """Get the default policy engine."""
    if _default_engine is None:
        raise RuntimeError("No default policy engine configured")
    return _default_engine
