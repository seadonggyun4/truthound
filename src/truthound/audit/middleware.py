"""Middleware and integration utilities for audit logging.

This module provides middleware implementations for common frameworks
and utilities for integrating audit logging into applications.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

from truthound.audit.core import (
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    AuditCategory,
    AuditOutcome,
    AuditActor,
    AuditResource,
    AuditContext,
)
from truthound.audit.logger import (
    AuditLogger,
    get_audit_logger,
    _AuditContext,
)


Request = TypeVar("Request")
Response = TypeVar("Response")


# =============================================================================
# Base Middleware
# =============================================================================


class AuditMiddleware(ABC, Generic[Request, Response]):
    """Abstract base class for audit logging middleware."""

    def __init__(
        self,
        logger: AuditLogger | None = None,
        *,
        log_requests: bool = True,
        log_responses: bool = True,
        skip_paths: list[str] | None = None,
        include_body: bool = False,
        include_headers: bool = False,
    ) -> None:
        """Initialize middleware.

        Args:
            logger: Audit logger to use.
            log_requests: Log incoming requests.
            log_responses: Log responses.
            skip_paths: Paths to skip auditing.
            include_body: Include request/response body.
            include_headers: Include headers in log.
        """
        self._logger = logger or get_audit_logger()
        self._log_requests = log_requests
        self._log_responses = log_responses
        self._skip_paths = set(skip_paths or [])
        self._include_body = include_body
        self._include_headers = include_headers

    @abstractmethod
    def get_path(self, request: Request) -> str:
        """Get request path."""
        pass

    @abstractmethod
    def get_method(self, request: Request) -> str:
        """Get request method."""
        pass

    @abstractmethod
    def get_actor(self, request: Request) -> AuditActor:
        """Extract actor from request."""
        pass

    @abstractmethod
    def get_request_id(self, request: Request) -> str:
        """Get or generate request ID."""
        pass

    def should_skip(self, request: Request) -> bool:
        """Check if request should skip auditing."""
        return self.get_path(request) in self._skip_paths


# =============================================================================
# ASGI Middleware
# =============================================================================


class ASGIAuditMiddleware:
    """ASGI middleware for audit logging.

    Compatible with Starlette, FastAPI, and other ASGI frameworks.

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> app.add_middleware(ASGIAuditMiddleware, logger=my_logger)
    """

    def __init__(
        self,
        app: Any,
        logger: AuditLogger | None = None,
        *,
        skip_paths: list[str] | None = None,
        include_body: bool = False,
        user_extractor: Callable[[dict], AuditActor | None] | None = None,
    ) -> None:
        """Initialize ASGI middleware.

        Args:
            app: ASGI application.
            logger: Audit logger.
            skip_paths: Paths to skip.
            include_body: Include request body.
            user_extractor: Function to extract user from scope.
        """
        self.app = app
        self._logger = logger or get_audit_logger()
        self._skip_paths = set(skip_paths or [])
        self._include_body = include_body
        self._user_extractor = user_extractor

    async def __call__(
        self,
        scope: dict,
        receive: Callable,
        send: Callable,
    ) -> None:
        """ASGI interface."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in self._skip_paths:
            await self.app(scope, receive, send)
            return

        # Generate request ID
        request_id = str(uuid.uuid4())

        # Extract actor
        actor = None
        if self._user_extractor:
            actor = self._user_extractor(scope)
        if actor is None:
            client = scope.get("client", ("", 0))
            actor = AuditActor.anonymous(ip_address=client[0] if client else "")

        # Set context
        _AuditContext.set_actor(actor)
        _AuditContext.set_request_id(request_id)

        # Track response
        status_code = 200
        start_time = time.time()

        async def send_wrapper(message: dict) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 200)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
            outcome = AuditOutcome.SUCCESS if status_code < 400 else AuditOutcome.FAILURE
        except Exception as e:
            outcome = AuditOutcome.FAILURE
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000

            # Log the request
            method = scope.get("method", "GET")
            self._logger.log(
                event_type=AuditEventType.READ,
                action=f"{method} {path}",
                actor=actor,
                resource=AuditResource(
                    id=path,
                    type="endpoint",
                    path=path,
                ),
                outcome=outcome,
                category=AuditCategory.DATA_ACCESS,
                data={
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                    "request_id": request_id,
                },
            )

            # Clear context
            _AuditContext.clear()


# =============================================================================
# WSGI Middleware
# =============================================================================


class WSGIAuditMiddleware:
    """WSGI middleware for audit logging.

    Compatible with Flask, Django, and other WSGI frameworks.

    Example:
        >>> from flask import Flask
        >>> app = Flask(__name__)
        >>> app.wsgi_app = WSGIAuditMiddleware(app.wsgi_app)
    """

    def __init__(
        self,
        app: Any,
        logger: AuditLogger | None = None,
        *,
        skip_paths: list[str] | None = None,
        user_extractor: Callable[[dict], AuditActor | None] | None = None,
    ) -> None:
        """Initialize WSGI middleware.

        Args:
            app: WSGI application.
            logger: Audit logger.
            skip_paths: Paths to skip.
            user_extractor: Function to extract user from environ.
        """
        self.app = app
        self._logger = logger or get_audit_logger()
        self._skip_paths = set(skip_paths or [])
        self._user_extractor = user_extractor

    def __call__(
        self,
        environ: dict,
        start_response: Callable,
    ) -> Any:
        """WSGI interface."""
        path = environ.get("PATH_INFO", "")
        if path in self._skip_paths:
            return self.app(environ, start_response)

        # Generate request ID
        request_id = str(uuid.uuid4())

        # Extract actor
        actor = None
        if self._user_extractor:
            actor = self._user_extractor(environ)
        if actor is None:
            # Try X-Forwarded-For first
            forwarded = environ.get("HTTP_X_FORWARDED_FOR")
            if forwarded:
                ip = forwarded.split(",")[0].strip()
            else:
                ip = environ.get("REMOTE_ADDR", "")
            actor = AuditActor.anonymous(ip_address=ip)

        # Set context
        _AuditContext.set_actor(actor)
        _AuditContext.set_request_id(request_id)

        status_code = [200]  # Use list to allow modification in closure
        start_time = time.time()

        def start_response_wrapper(
            status: str,
            response_headers: list,
            exc_info: Any = None,
        ) -> Callable:
            # Extract status code from status string
            status_code[0] = int(status.split(" ")[0])
            return start_response(status, response_headers, exc_info)

        try:
            result = self.app(environ, start_response_wrapper)
            outcome = AuditOutcome.SUCCESS if status_code[0] < 400 else AuditOutcome.FAILURE
        except Exception:
            outcome = AuditOutcome.FAILURE
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000

            method = environ.get("REQUEST_METHOD", "GET")
            self._logger.log(
                event_type=AuditEventType.READ,
                action=f"{method} {path}",
                actor=actor,
                resource=AuditResource(
                    id=path,
                    type="endpoint",
                    path=path,
                ),
                outcome=outcome,
                category=AuditCategory.DATA_ACCESS,
                data={
                    "method": method,
                    "path": path,
                    "status_code": status_code[0],
                    "duration_ms": duration_ms,
                    "request_id": request_id,
                },
            )

            _AuditContext.clear()

        return result


# =============================================================================
# Database Audit Integration
# =============================================================================


@dataclass
class DatabaseAuditConfig:
    """Configuration for database audit logging."""

    log_queries: bool = False  # Log SELECT queries
    log_modifications: bool = True  # Log INSERT/UPDATE/DELETE
    include_data: bool = False  # Include row data
    mask_sensitive: bool = True  # Mask sensitive columns
    sensitive_columns: list[str] | None = None


class DatabaseAuditHook:
    """Hook for auditing database operations.

    Can be integrated with ORMs like SQLAlchemy.

    Example:
        >>> hook = DatabaseAuditHook(logger)
        >>> # Integrate with SQLAlchemy events
        >>> event.listen(session, "after_insert", hook.on_insert)
        >>> event.listen(session, "after_update", hook.on_update)
        >>> event.listen(session, "after_delete", hook.on_delete)
    """

    def __init__(
        self,
        logger: AuditLogger | None = None,
        config: DatabaseAuditConfig | None = None,
    ) -> None:
        self._logger = logger or get_audit_logger()
        self._config = config or DatabaseAuditConfig()

    def on_insert(
        self,
        table: str,
        row_id: Any,
        data: dict[str, Any] | None = None,
    ) -> AuditEvent | None:
        """Log an insert operation."""
        if not self._config.log_modifications:
            return None

        return self._logger.log(
            event_type=AuditEventType.CREATE,
            action="insert",
            resource=AuditResource(
                id=f"{table}:{row_id}",
                type="database_row",
                name=table,
            ),
            new_value=self._mask_data(data) if data else None,
            category=AuditCategory.DATA_MODIFICATION,
        )

    def on_update(
        self,
        table: str,
        row_id: Any,
        old_data: dict[str, Any] | None = None,
        new_data: dict[str, Any] | None = None,
    ) -> AuditEvent | None:
        """Log an update operation."""
        if not self._config.log_modifications:
            return None

        return self._logger.log(
            event_type=AuditEventType.UPDATE,
            action="update",
            resource=AuditResource(
                id=f"{table}:{row_id}",
                type="database_row",
                name=table,
            ),
            old_value=self._mask_data(old_data) if old_data else None,
            new_value=self._mask_data(new_data) if new_data else None,
            category=AuditCategory.DATA_MODIFICATION,
        )

    def on_delete(
        self,
        table: str,
        row_id: Any,
        data: dict[str, Any] | None = None,
    ) -> AuditEvent | None:
        """Log a delete operation."""
        if not self._config.log_modifications:
            return None

        return self._logger.log(
            event_type=AuditEventType.DELETE,
            action="delete",
            resource=AuditResource(
                id=f"{table}:{row_id}",
                type="database_row",
                name=table,
            ),
            old_value=self._mask_data(data) if data else None,
            category=AuditCategory.DATA_MODIFICATION,
        )

    def on_query(
        self,
        query: str,
        params: tuple | None = None,
        rows_returned: int = 0,
    ) -> AuditEvent | None:
        """Log a query operation."""
        if not self._config.log_queries:
            return None

        return self._logger.log(
            event_type=AuditEventType.READ,
            action="query",
            category=AuditCategory.DATA_ACCESS,
            data={
                "query": query[:500],  # Truncate long queries
                "rows_returned": rows_returned,
            },
        )

    def _mask_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Mask sensitive columns."""
        if not self._config.mask_sensitive or not data:
            return data

        sensitive = set(self._config.sensitive_columns or [
            "password", "secret", "token", "api_key",
        ])

        result = {}
        for key, value in data.items():
            if key.lower() in sensitive:
                result[key] = "***MASKED***"
            else:
                result[key] = value
        return result


# =============================================================================
# Checkpoint Integration
# =============================================================================


class CheckpointAuditHook:
    """Hook for auditing checkpoint operations.

    Example:
        >>> hook = CheckpointAuditHook(logger)
        >>> # Use in checkpoint
        >>> checkpoint.add_hook(hook)
    """

    def __init__(self, logger: AuditLogger | None = None) -> None:
        self._logger = logger or get_audit_logger()

    def on_checkpoint_start(
        self,
        checkpoint_name: str,
        config: dict[str, Any] | None = None,
    ) -> AuditEvent | None:
        """Log checkpoint start."""
        return self._logger.log(
            event_type=AuditEventType.VALIDATION_START,
            action="checkpoint_start",
            resource=AuditResource(
                id=f"checkpoint:{checkpoint_name}",
                type="checkpoint",
                name=checkpoint_name,
            ),
            category=AuditCategory.VALIDATION,
            data=config or {},
        )

    def on_checkpoint_complete(
        self,
        checkpoint_name: str,
        success: bool,
        duration_ms: float,
        results: dict[str, Any] | None = None,
    ) -> AuditEvent | None:
        """Log checkpoint completion."""
        return self._logger.log(
            event_type=AuditEventType.VALIDATION_COMPLETE if success else AuditEventType.VALIDATION_FAILED,
            action="checkpoint_complete",
            resource=AuditResource(
                id=f"checkpoint:{checkpoint_name}",
                type="checkpoint",
                name=checkpoint_name,
            ),
            outcome=AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE,
            category=AuditCategory.VALIDATION,
            data={
                "duration_ms": duration_ms,
                **(results or {}),
            },
        )

    def on_validation_error(
        self,
        checkpoint_name: str,
        error: str,
        details: dict[str, Any] | None = None,
    ) -> AuditEvent | None:
        """Log validation error."""
        return self._logger.log(
            event_type=AuditEventType.VALIDATION_FAILED,
            action="validation_error",
            resource=AuditResource(
                id=f"checkpoint:{checkpoint_name}",
                type="checkpoint",
                name=checkpoint_name,
            ),
            outcome=AuditOutcome.FAILURE,
            severity=AuditSeverity.ERROR,
            message=error,
            category=AuditCategory.VALIDATION,
            data=details or {},
        )


# =============================================================================
# Export Functions
# =============================================================================


def create_asgi_middleware(
    app: Any,
    logger: AuditLogger | None = None,
    **kwargs: Any,
) -> ASGIAuditMiddleware:
    """Create ASGI audit middleware."""
    return ASGIAuditMiddleware(app, logger, **kwargs)


def create_wsgi_middleware(
    app: Any,
    logger: AuditLogger | None = None,
    **kwargs: Any,
) -> WSGIAuditMiddleware:
    """Create WSGI audit middleware."""
    return WSGIAuditMiddleware(app, logger, **kwargs)
