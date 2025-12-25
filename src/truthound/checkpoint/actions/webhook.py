"""Generic webhook action.

This action sends HTTP requests to external webhooks when checkpoint
validations complete, enabling integration with any HTTP-compatible system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from truthound.checkpoint.actions.base import (
    ActionConfig,
    ActionResult,
    ActionStatus,
    BaseAction,
    NotifyCondition,
)

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import CheckpointResult


@dataclass
class WebhookConfig(ActionConfig):
    """Configuration for webhook action.

    Attributes:
        url: Webhook URL to call.
        method: HTTP method (GET, POST, PUT, PATCH).
        headers: Additional HTTP headers to send.
        auth_type: Authentication type ("none", "basic", "bearer", "api_key").
        auth_credentials: Authentication credentials (varies by auth_type).
        payload_template: Custom JSON payload template (supports placeholders).
        include_full_result: Include full validation result in payload.
        ssl_verify: Verify SSL certificates.
        success_codes: HTTP status codes to consider successful.
    """

    url: str = ""
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    auth_type: str = "none"
    auth_credentials: dict[str, str] = field(default_factory=dict)
    payload_template: dict[str, Any] | None = None
    include_full_result: bool = True
    ssl_verify: bool = True
    success_codes: list[int] = field(default_factory=lambda: [200, 201, 202, 204])
    notify_on: NotifyCondition | str = NotifyCondition.ALWAYS


class WebhookAction(BaseAction[WebhookConfig]):
    """Action to call external webhooks.

    Sends HTTP requests to any webhook endpoint with configurable
    authentication, headers, and payload formats.

    Example:
        >>> action = WebhookAction(
        ...     url="https://api.example.com/data-quality/events",
        ...     method="POST",
        ...     auth_type="bearer",
        ...     auth_credentials={"token": "secret-token"},
        ...     headers={"X-Custom-Header": "value"},
        ... )
        >>> result = action.execute(checkpoint_result)
    """

    action_type = "webhook"

    @classmethod
    def _default_config(cls) -> WebhookConfig:
        return WebhookConfig()

    def _execute(self, checkpoint_result: "CheckpointResult") -> ActionResult:
        """Send webhook request."""
        import urllib.request
        import urllib.error
        import ssl

        config = self._config

        if not config.url:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="No URL configured",
                error="url is required",
            )

        # Build payload
        payload = self._build_payload(checkpoint_result)

        # Build headers
        headers = self._build_headers()

        # Create request
        data = json.dumps(payload, default=str).encode("utf-8") if payload else None

        request = urllib.request.Request(
            config.url,
            data=data,
            headers=headers,
            method=config.method.upper(),
        )

        # SSL context
        ssl_context = None
        if not config.ssl_verify:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        # Send request
        try:
            with urllib.request.urlopen(
                request,
                timeout=config.timeout_seconds,
                context=ssl_context,
            ) as response:
                status_code = response.status
                response_body = response.read().decode("utf-8")

            if status_code in config.success_codes:
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.SUCCESS,
                    message=f"Webhook called successfully (HTTP {status_code})",
                    details={
                        "url": config.url,
                        "method": config.method,
                        "status_code": status_code,
                        "response": response_body[:500] if response_body else None,
                    },
                )
            else:
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.FAILURE,
                    message=f"Webhook returned unexpected status: {status_code}",
                    details={
                        "url": config.url,
                        "status_code": status_code,
                        "response": response_body[:500] if response_body else None,
                    },
                )

        except urllib.error.HTTPError as e:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message=f"Webhook HTTP error: {e.code}",
                error=str(e),
                details={
                    "url": config.url,
                    "status_code": e.code,
                    "reason": e.reason,
                },
            )
        except urllib.error.URLError as e:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="Webhook request failed",
                error=str(e),
                details={"url": config.url},
            )

    def _build_payload(self, checkpoint_result: "CheckpointResult") -> dict[str, Any]:
        """Build webhook payload."""
        config = self._config

        if config.payload_template:
            # Use custom template with placeholder substitution
            payload = self._substitute_placeholders(
                config.payload_template,
                checkpoint_result,
            )
        else:
            # Default payload
            validation = checkpoint_result.validation_result
            stats = validation.statistics if validation else None

            payload: dict[str, Any] = {
                "event": "validation_completed",
                "checkpoint": checkpoint_result.checkpoint_name,
                "run_id": checkpoint_result.run_id,
                "status": checkpoint_result.status.value,
                "run_time": checkpoint_result.run_time.isoformat(),
                "data_asset": checkpoint_result.data_asset,
                "summary": {
                    "total_issues": stats.total_issues if stats else 0,
                    "critical_issues": stats.critical_issues if stats else 0,
                    "high_issues": stats.high_issues if stats else 0,
                    "medium_issues": stats.medium_issues if stats else 0,
                    "low_issues": stats.low_issues if stats else 0,
                    "pass_rate": stats.pass_rate if stats else 1.0,
                },
            }

            if config.include_full_result:
                payload["full_result"] = checkpoint_result.to_dict()

        return payload

    def _substitute_placeholders(
        self,
        template: dict[str, Any],
        checkpoint_result: "CheckpointResult",
    ) -> dict[str, Any]:
        """Substitute placeholders in template with actual values."""
        import copy

        result = copy.deepcopy(template)
        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None

        placeholders = {
            "${checkpoint}": checkpoint_result.checkpoint_name,
            "${run_id}": checkpoint_result.run_id,
            "${status}": checkpoint_result.status.value,
            "${run_time}": checkpoint_result.run_time.isoformat(),
            "${data_asset}": checkpoint_result.data_asset,
            "${total_issues}": stats.total_issues if stats else 0,
            "${critical_issues}": stats.critical_issues if stats else 0,
            "${high_issues}": stats.high_issues if stats else 0,
            "${medium_issues}": stats.medium_issues if stats else 0,
            "${low_issues}": stats.low_issues if stats else 0,
            "${pass_rate}": stats.pass_rate if stats else 1.0,
        }

        def substitute(obj: Any) -> Any:
            if isinstance(obj, str):
                for key, value in placeholders.items():
                    obj = obj.replace(key, str(value))
                return obj
            elif isinstance(obj, dict):
                return {k: substitute(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute(item) for item in obj]
            else:
                return obj

        return substitute(result)

    def _build_headers(self) -> dict[str, str]:
        """Build request headers with authentication."""
        config = self._config

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Truthound/1.0",
            **config.headers,
        }

        # Add authentication
        if config.auth_type == "basic":
            import base64
            credentials = f"{config.auth_credentials.get('username', '')}:{config.auth_credentials.get('password', '')}"
            encoded = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"

        elif config.auth_type == "bearer":
            token = config.auth_credentials.get("token", "")
            headers["Authorization"] = f"Bearer {token}"

        elif config.auth_type == "api_key":
            header_name = config.auth_credentials.get("header", "X-API-Key")
            api_key = config.auth_credentials.get("key", "")
            headers[header_name] = api_key

        return headers

    def validate_config(self) -> list[str]:
        """Validate configuration."""
        errors = []

        if not self._config.url:
            errors.append("url is required")

        if self._config.method.upper() not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
            errors.append(f"Invalid HTTP method: {self._config.method}")

        if self._config.auth_type not in ("none", "basic", "bearer", "api_key"):
            errors.append(f"Invalid auth_type: {self._config.auth_type}")

        return errors
