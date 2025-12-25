"""Middleware and integration utilities for rate limiting.

This module provides middleware implementations for common frameworks
and utilities for integrating rate limiting into applications.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generic, TypeVar

from truthound.ratelimit.core import (
    RateLimitConfig,
    RateLimitResult,
    RateLimitExceeded,
    RateLimitAction,
    KeyExtractor,
    current_time,
)
from truthound.ratelimit.limiter import RateLimiter, get_limiter


# Type variables
Request = TypeVar("Request")
Response = TypeVar("Response")


# =============================================================================
# Base Middleware
# =============================================================================


class RateLimitMiddleware(ABC, Generic[Request, Response]):
    """Abstract base class for rate limiting middleware.

    Implement this for your specific framework.
    """

    def __init__(
        self,
        limiter: RateLimiter | None = None,
        *,
        key_extractor: KeyExtractor | Callable[[Request], str] | None = None,
        on_exceeded: Callable[[Request, RateLimitResult], Response] | None = None,
        add_headers: bool = True,
        skip_paths: list[str] | None = None,
    ) -> None:
        """Initialize middleware.

        Args:
            limiter: Rate limiter to use.
            key_extractor: Function to extract rate limit key from request.
            on_exceeded: Custom handler for exceeded limits.
            add_headers: Add rate limit headers to response.
            skip_paths: Paths to skip rate limiting.
        """
        self._limiter = limiter or get_limiter()
        self._key_extractor = key_extractor
        self._on_exceeded = on_exceeded
        self._add_headers = add_headers
        self._skip_paths = set(skip_paths or [])

    @abstractmethod
    def extract_key(self, request: Request) -> str:
        """Extract rate limit key from request.

        Override this or provide key_extractor in constructor.
        """
        pass

    @abstractmethod
    def get_path(self, request: Request) -> str:
        """Get request path for skip check."""
        pass

    @abstractmethod
    def create_exceeded_response(
        self,
        request: Request,
        result: RateLimitResult,
    ) -> Response:
        """Create response for rate limit exceeded."""
        pass

    @abstractmethod
    def add_rate_limit_headers(
        self,
        response: Response,
        result: RateLimitResult,
    ) -> Response:
        """Add rate limit headers to response."""
        pass

    def should_skip(self, request: Request) -> bool:
        """Check if request should skip rate limiting."""
        path = self.get_path(request)
        return path in self._skip_paths

    def process_request(self, request: Request) -> tuple[bool, RateLimitResult | None]:
        """Process request through rate limiter.

        Returns:
            Tuple of (allowed, result).
        """
        if self.should_skip(request):
            return True, None

        # Extract key
        if self._key_extractor:
            if isinstance(self._key_extractor, KeyExtractor):
                key = self._key_extractor.extract(request)
            else:
                key = self._key_extractor(request)
        else:
            key = self.extract_key(request)

        # Acquire
        result = self._limiter.acquire(key)

        return result.allowed, result


# =============================================================================
# ASGI Middleware
# =============================================================================


class ASGIRateLimitMiddleware:
    """ASGI middleware for rate limiting.

    Compatible with Starlette, FastAPI, and other ASGI frameworks.

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> app.add_middleware(
        ...     ASGIRateLimitMiddleware,
        ...     limiter=my_limiter,
        ...     key_func=lambda scope: scope.get("client", ("",))[0],
        ... )
    """

    def __init__(
        self,
        app: Any,
        limiter: RateLimiter | None = None,
        *,
        key_func: Callable[[dict], str] | None = None,
        skip_paths: list[str] | None = None,
        status_code: int = 429,
    ) -> None:
        """Initialize ASGI middleware.

        Args:
            app: ASGI application.
            limiter: Rate limiter.
            key_func: Function to extract key from ASGI scope.
            skip_paths: Paths to skip.
            status_code: Status code for rate limit response.
        """
        self.app = app
        self._limiter = limiter or get_limiter()
        self._key_func = key_func or self._default_key_func
        self._skip_paths = set(skip_paths or [])
        self._status_code = status_code

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

        key = self._key_func(scope)
        result = self._limiter.acquire(key)

        if not result.allowed:
            await self._send_rate_limit_response(send, result)
            return

        # Wrap send to add headers
        async def send_with_headers(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                for name, value in result.to_headers().items():
                    headers.append((name.lower().encode(), value.encode()))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)

    async def _send_rate_limit_response(
        self,
        send: Callable,
        result: RateLimitResult,
    ) -> None:
        """Send rate limit exceeded response."""
        headers = [
            (b"content-type", b"application/json"),
        ]
        for name, value in result.to_headers().items():
            headers.append((name.lower().encode(), value.encode()))

        await send({
            "type": "http.response.start",
            "status": self._status_code,
            "headers": headers,
        })

        body = (
            f'{{"error": "Rate limit exceeded", '
            f'"retry_after": {result.retry_after}}}'
        ).encode()

        await send({
            "type": "http.response.body",
            "body": body,
        })

    def _default_key_func(self, scope: dict) -> str:
        """Default key extraction from ASGI scope."""
        client = scope.get("client", ("unknown",))
        return f"ip:{client[0]}"


# =============================================================================
# WSGI Middleware
# =============================================================================


class WSGIRateLimitMiddleware:
    """WSGI middleware for rate limiting.

    Compatible with Flask, Django, and other WSGI frameworks.

    Example:
        >>> from flask import Flask
        >>> app = Flask(__name__)
        >>> app.wsgi_app = WSGIRateLimitMiddleware(
        ...     app.wsgi_app,
        ...     limiter=my_limiter,
        ... )
    """

    def __init__(
        self,
        app: Any,
        limiter: RateLimiter | None = None,
        *,
        key_func: Callable[[dict], str] | None = None,
        skip_paths: list[str] | None = None,
        status_code: int = 429,
    ) -> None:
        """Initialize WSGI middleware.

        Args:
            app: WSGI application.
            limiter: Rate limiter.
            key_func: Function to extract key from environ.
            skip_paths: Paths to skip.
            status_code: Status code for rate limit response.
        """
        self.app = app
        self._limiter = limiter or get_limiter()
        self._key_func = key_func or self._default_key_func
        self._skip_paths = set(skip_paths or [])
        self._status_code = status_code

    def __call__(
        self,
        environ: dict,
        start_response: Callable,
    ) -> Any:
        """WSGI interface."""
        path = environ.get("PATH_INFO", "")
        if path in self._skip_paths:
            return self.app(environ, start_response)

        key = self._key_func(environ)
        result = self._limiter.acquire(key)

        if not result.allowed:
            return self._rate_limit_response(start_response, result)

        # Wrap start_response to add headers
        def start_response_with_headers(
            status: str,
            response_headers: list,
            exc_info: Any = None,
        ) -> Callable:
            for name, value in result.to_headers().items():
                response_headers.append((name, value))
            return start_response(status, response_headers, exc_info)

        return self.app(environ, start_response_with_headers)

    def _rate_limit_response(
        self,
        start_response: Callable,
        result: RateLimitResult,
    ) -> list[bytes]:
        """Generate rate limit exceeded response."""
        headers = [("Content-Type", "application/json")]
        for name, value in result.to_headers().items():
            headers.append((name, value))

        status = f"{self._status_code} Too Many Requests"
        start_response(status, headers)

        body = (
            f'{{"error": "Rate limit exceeded", '
            f'"retry_after": {result.retry_after}}}'
        ).encode()

        return [body]

    def _default_key_func(self, environ: dict) -> str:
        """Default key extraction from WSGI environ."""
        # Try X-Forwarded-For first (for proxied requests)
        forwarded = environ.get("HTTP_X_FORWARDED_FOR")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = environ.get("REMOTE_ADDR", "unknown")
        return f"ip:{ip}"


# =============================================================================
# Flask Integration
# =============================================================================


def create_flask_limiter(
    limiter: RateLimiter | None = None,
    *,
    key_func: Callable[[], str] | None = None,
    default_limits: list[str] | None = None,
) -> Any:
    """Create Flask rate limiting integration.

    Returns a decorator factory for Flask routes.

    Example:
        >>> from flask import Flask, request
        >>> app = Flask(__name__)
        >>> flask_limit = create_flask_limiter(
        ...     key_func=lambda: request.remote_addr,
        ... )
        >>>
        >>> @app.route("/api")
        >>> @flask_limit("10/minute")
        ... def api():
        ...     return "OK"
    """
    _limiter = limiter or get_limiter()

    def decorator_factory(limit_string: str) -> Callable:
        """Create decorator from limit string like '10/minute'."""
        config = _parse_limit_string(limit_string)
        route_limiter = RateLimiter(config=config)

        def decorator(func: Callable) -> Callable:
            import functools

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Get key
                if key_func:
                    key = key_func()
                else:
                    # Flask request context
                    try:
                        from flask import request
                        key = request.remote_addr or "unknown"
                    except ImportError:
                        key = "unknown"

                result = route_limiter.acquire(key)

                if not result.allowed:
                    # Flask abort
                    try:
                        from flask import abort, make_response, jsonify
                        response = make_response(
                            jsonify(error="Rate limit exceeded"),
                            429,
                        )
                        for name, value in result.to_headers().items():
                            response.headers[name] = value
                        return response
                    except ImportError:
                        raise RateLimitExceeded(result=result)

                return func(*args, **kwargs)

            return wrapper

        return decorator

    return decorator_factory


def _parse_limit_string(limit_string: str) -> RateLimitConfig:
    """Parse limit string like '10/minute' or '100/hour'."""
    parts = limit_string.lower().split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid limit string: {limit_string}")

    count = int(parts[0])
    period = parts[1]

    period_seconds = {
        "second": 1,
        "minute": 60,
        "hour": 3600,
        "day": 86400,
    }.get(period)

    if period_seconds is None:
        raise ValueError(f"Unknown period: {period}")

    return RateLimitConfig(
        requests_per_second=count / period_seconds,
        window_size_seconds=float(period_seconds),
    )


# =============================================================================
# Retry Handler
# =============================================================================


class RetryHandler:
    """Handler for retrying rate-limited requests.

    Implements exponential backoff with jitter.

    Example:
        >>> handler = RetryHandler(max_retries=3)
        >>> result = await handler.execute(my_function, "arg1")
    """

    def __init__(
        self,
        limiter: RateLimiter | None = None,
        *,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: float = 0.1,
    ) -> None:
        """Initialize retry handler.

        Args:
            limiter: Rate limiter to check.
            max_retries: Maximum retry attempts.
            base_delay: Base delay in seconds.
            max_delay: Maximum delay.
            exponential_base: Exponential backoff base.
            jitter: Jitter factor (0-1).
        """
        self._limiter = limiter or get_limiter()
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._exponential_base = exponential_base
        self._jitter = jitter

    def execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        key: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute function with retry on rate limit.

        Args:
            func: Function to execute.
            *args: Function arguments.
            key: Rate limit key.
            **kwargs: Function keyword arguments.

        Returns:
            Function result.

        Raises:
            RateLimitExceeded: If all retries exhausted.
        """
        import random

        for attempt in range(self._max_retries + 1):
            # Check rate limit
            result = self._limiter.acquire(key or func.__name__)

            if result.allowed:
                return func(*args, **kwargs)

            if attempt >= self._max_retries:
                raise RateLimitExceeded(
                    f"Rate limit exceeded after {self._max_retries} retries",
                    result=result,
                )

            # Calculate delay
            delay = min(
                self._base_delay * (self._exponential_base ** attempt),
                self._max_delay,
            )

            # Use retry_after if provided
            if result.retry_after > 0:
                delay = min(delay, result.retry_after)

            # Add jitter
            jitter_range = delay * self._jitter
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)

            time.sleep(delay)

        raise RateLimitExceeded("Rate limit exceeded")

    async def execute_async(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        key: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute async function with retry.

        Args:
            func: Async function to execute.
            *args: Function arguments.
            key: Rate limit key.
            **kwargs: Function keyword arguments.

        Returns:
            Function result.
        """
        import asyncio
        import random

        for attempt in range(self._max_retries + 1):
            result = self._limiter.acquire(key or func.__name__)

            if result.allowed:
                return await func(*args, **kwargs)

            if attempt >= self._max_retries:
                raise RateLimitExceeded(
                    f"Rate limit exceeded after {self._max_retries} retries",
                    result=result,
                )

            delay = min(
                self._base_delay * (self._exponential_base ** attempt),
                self._max_delay,
            )

            if result.retry_after > 0:
                delay = min(delay, result.retry_after)

            jitter_range = delay * self._jitter
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)

            await asyncio.sleep(delay)

        raise RateLimitExceeded("Rate limit exceeded")


# =============================================================================
# Metrics Integration
# =============================================================================


@dataclass
class RateLimitMetrics:
    """Metrics for rate limiting.

    Tracks rate limit decisions and timing.
    """

    total_requests: int = 0
    allowed_requests: int = 0
    rejected_requests: int = 0
    total_wait_time: float = 0.0
    avg_utilization: float = 0.0
    _utilization_samples: list[float] = field(default_factory=list)

    def record(self, result: RateLimitResult, wait_time: float = 0.0) -> None:
        """Record a rate limit decision.

        Args:
            result: Rate limit result.
            wait_time: Time spent waiting (if any).
        """
        self.total_requests += 1

        if result.allowed:
            self.allowed_requests += 1
        else:
            self.rejected_requests += 1

        self.total_wait_time += wait_time
        self._utilization_samples.append(result.utilization)

        # Keep only last 1000 samples for avg calculation
        if len(self._utilization_samples) > 1000:
            self._utilization_samples = self._utilization_samples[-1000:]

        if self._utilization_samples:
            self.avg_utilization = sum(self._utilization_samples) / len(self._utilization_samples)

    @property
    def rejection_rate(self) -> float:
        """Get rejection rate (0-1)."""
        if self.total_requests == 0:
            return 0.0
        return self.rejected_requests / self.total_requests

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "allowed_requests": self.allowed_requests,
            "rejected_requests": self.rejected_requests,
            "rejection_rate": self.rejection_rate,
            "total_wait_time": self.total_wait_time,
            "avg_utilization": self.avg_utilization,
        }


class MetricsMiddleware:
    """Middleware that collects rate limit metrics.

    Wraps a rate limiter to collect statistics.

    Example:
        >>> metrics = RateLimitMetrics()
        >>> middleware = MetricsMiddleware(limiter, metrics)
        >>> result = middleware.acquire("key")
        >>> print(metrics.rejection_rate)
    """

    def __init__(
        self,
        limiter: RateLimiter,
        metrics: RateLimitMetrics | None = None,
    ) -> None:
        self._limiter = limiter
        self._metrics = metrics or RateLimitMetrics()

    @property
    def metrics(self) -> RateLimitMetrics:
        """Get metrics."""
        return self._metrics

    def acquire(
        self,
        key: str,
        *,
        tokens: int = 1,
        wait: bool = False,
        timeout: float | None = None,
    ) -> RateLimitResult:
        """Acquire with metrics tracking."""
        start = current_time()
        result = self._limiter.acquire(
            key,
            tokens=tokens,
            wait=wait,
            timeout=timeout,
        )
        wait_time = current_time() - start

        self._metrics.record(result, wait_time)
        return result

    def peek(self, key: str) -> RateLimitResult:
        """Peek without metrics."""
        return self._limiter.peek(key)

    def reset(self, key: str) -> None:
        """Reset limiter."""
        self._limiter.reset(key)
