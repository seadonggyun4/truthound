"""Async action implementations.

This module provides async-native implementations of common actions
for optimal performance in async checkpoint pipelines.

Each action uses aiohttp (if available) or asyncio-compatible HTTP
for non-blocking network operations.
"""

from __future__ import annotations

import asyncio
import json
import ssl
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from truthound.checkpoint.async_base import AsyncBaseAction
from truthound.checkpoint.actions.base import (
    ActionConfig,
    ActionResult,
    ActionStatus,
    NotifyCondition,
)

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import CheckpointResult


# Try to import aiohttp, fall back to stdlib if not available
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


# =============================================================================
# Async Webhook Action
# =============================================================================


@dataclass
class AsyncWebhookConfig(ActionConfig):
    """Configuration for async webhook action.

    Attributes:
        url: Webhook URL.
        method: HTTP method.
        headers: Additional headers.
        auth_type: Authentication type.
        auth_credentials: Auth credentials.
        payload_template: Custom payload template.
        include_full_result: Include full result in payload.
        ssl_verify: Verify SSL certificates.
        success_codes: HTTP codes considered successful.
        connection_timeout: Connection timeout seconds.
        read_timeout: Read timeout seconds.
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
    connection_timeout: float = 10.0
    read_timeout: float = 30.0
    notify_on: NotifyCondition | str = NotifyCondition.ALWAYS


class AsyncWebhookAction(AsyncBaseAction[AsyncWebhookConfig]):
    """Async webhook action using aiohttp or asyncio.

    Provides non-blocking HTTP requests for webhook integration.

    Example:
        >>> action = AsyncWebhookAction(
        ...     url="https://api.example.com/webhook",
        ...     method="POST",
        ...     auth_type="bearer",
        ...     auth_credentials={"token": "secret"},
        ... )
        >>> result = await action.execute_async(checkpoint_result)
    """

    action_type = "async_webhook"

    @classmethod
    def _default_config(cls) -> AsyncWebhookConfig:
        return AsyncWebhookConfig()

    async def _execute_async(
        self, checkpoint_result: "CheckpointResult"
    ) -> ActionResult:
        """Execute webhook request asynchronously."""
        config = self._config

        if not config.url:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="No URL configured",
                error="url is required",
            )

        payload = self._build_payload(checkpoint_result)
        headers = self._build_headers()

        if AIOHTTP_AVAILABLE:
            return await self._execute_with_aiohttp(payload, headers)
        else:
            return await self._execute_with_stdlib(payload, headers)

    async def _execute_with_aiohttp(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> ActionResult:
        """Execute using aiohttp."""
        config = self._config

        ssl_context = None
        if not config.ssl_verify:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        timeout = aiohttp.ClientTimeout(
            connect=config.connection_timeout,
            total=config.read_timeout,
        )

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(
                    method=config.method.upper(),
                    url=config.url,
                    json=payload,
                    headers=headers,
                    ssl=ssl_context,
                ) as response:
                    status_code = response.status
                    response_body = await response.text()

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
                    message=f"Webhook returned status {status_code}",
                    details={
                        "url": config.url,
                        "status_code": status_code,
                        "response": response_body[:500] if response_body else None,
                    },
                )

        except asyncio.TimeoutError:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="Webhook request timed out",
                error=f"Timeout after {config.read_timeout}s",
            )
        except Exception as e:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="Webhook request failed",
                error=str(e),
            )

    async def _execute_with_stdlib(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> ActionResult:
        """Execute using stdlib (run in executor)."""
        import urllib.request
        import urllib.error

        config = self._config

        ssl_context = None
        if not config.ssl_verify:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        data = json.dumps(payload, default=str).encode("utf-8")
        request = urllib.request.Request(
            config.url,
            data=data,
            headers=headers,
            method=config.method.upper(),
        )

        loop = asyncio.get_running_loop()

        def do_request() -> tuple[int, str]:
            with urllib.request.urlopen(
                request,
                timeout=config.read_timeout,
                context=ssl_context,
            ) as response:
                return response.status, response.read().decode("utf-8")

        try:
            status_code, response_body = await loop.run_in_executor(
                None, do_request
            )

            if status_code in config.success_codes:
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.SUCCESS,
                    message=f"Webhook called successfully (HTTP {status_code})",
                    details={
                        "url": config.url,
                        "status_code": status_code,
                    },
                )
            else:
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.FAILURE,
                    message=f"Webhook returned status {status_code}",
                    details={"url": config.url, "status_code": status_code},
                )

        except Exception as e:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="Webhook request failed",
                error=str(e),
            )

    def _build_payload(
        self, checkpoint_result: "CheckpointResult"
    ) -> dict[str, Any]:
        """Build webhook payload."""
        config = self._config
        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None

        if config.payload_template:
            return self._substitute_placeholders(
                config.payload_template, checkpoint_result
            )

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
        """Substitute template placeholders."""
        import copy

        result = copy.deepcopy(template)
        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None

        placeholders = {
            "${checkpoint}": checkpoint_result.checkpoint_name,
            "${run_id}": checkpoint_result.run_id,
            "${status}": checkpoint_result.status.value,
            "${total_issues}": stats.total_issues if stats else 0,
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
            return obj

        return substitute(result)

    def _build_headers(self) -> dict[str, str]:
        """Build request headers."""
        config = self._config

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Truthound-Async/1.0",
            **config.headers,
        }

        if config.auth_type == "basic":
            import base64
            creds = f"{config.auth_credentials.get('username', '')}:{config.auth_credentials.get('password', '')}"
            headers["Authorization"] = f"Basic {base64.b64encode(creds.encode()).decode()}"
        elif config.auth_type == "bearer":
            headers["Authorization"] = f"Bearer {config.auth_credentials.get('token', '')}"
        elif config.auth_type == "api_key":
            header_name = config.auth_credentials.get("header", "X-API-Key")
            headers[header_name] = config.auth_credentials.get("key", "")

        return headers

    def validate_config(self) -> list[str]:
        """Validate configuration."""
        errors = []
        if not self._config.url:
            errors.append("url is required")
        if self._config.method.upper() not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
            errors.append(f"Invalid method: {self._config.method}")
        return errors


# =============================================================================
# Async Slack Notification
# =============================================================================


@dataclass
class AsyncSlackConfig(ActionConfig):
    """Configuration for async Slack notification."""

    webhook_url: str = ""
    channel: str | None = None
    username: str = "Truthound"
    icon_emoji: str = ":mag:"
    include_details: bool = True
    mention_on_failure: list[str] = field(default_factory=list)
    custom_message: str | None = None
    notify_on: NotifyCondition | str = NotifyCondition.FAILURE


class AsyncSlackNotification(AsyncBaseAction[AsyncSlackConfig]):
    """Async Slack notification using non-blocking HTTP.

    Example:
        >>> action = AsyncSlackNotification(
        ...     webhook_url="https://hooks.slack.com/...",
        ...     notify_on="failure",
        ...     channel="#data-quality",
        ... )
    """

    action_type = "async_slack"

    @classmethod
    def _default_config(cls) -> AsyncSlackConfig:
        return AsyncSlackConfig()

    async def _execute_async(
        self, checkpoint_result: "CheckpointResult"
    ) -> ActionResult:
        """Send Slack notification asynchronously."""
        config = self._config

        if not config.webhook_url:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="No webhook URL configured",
                error="webhook_url is required",
            )

        payload = self._build_payload(checkpoint_result)

        if AIOHTTP_AVAILABLE:
            return await self._send_with_aiohttp(payload)
        else:
            return await self._send_with_stdlib(payload)

    async def _send_with_aiohttp(
        self, payload: dict[str, Any]
    ) -> ActionResult:
        """Send using aiohttp."""
        config = self._config

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    response_text = await response.text()

            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.SUCCESS,
                message="Slack notification sent",
                details={"response": response_text},
            )

        except Exception as e:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="Failed to send Slack notification",
                error=str(e),
            )

    async def _send_with_stdlib(
        self, payload: dict[str, Any]
    ) -> ActionResult:
        """Send using stdlib in executor."""
        import urllib.request

        config = self._config
        loop = asyncio.get_running_loop()

        def do_send() -> str:
            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                config.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8")

        try:
            response_text = await loop.run_in_executor(None, do_send)
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.SUCCESS,
                message="Slack notification sent",
                details={"response": response_text},
            )
        except Exception as e:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="Failed to send Slack notification",
                error=str(e),
            )

    def _build_payload(
        self, checkpoint_result: "CheckpointResult"
    ) -> dict[str, Any]:
        """Build Slack message payload."""
        config = self._config
        status = checkpoint_result.status.value
        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None

        color_map = {
            "success": "#28a745",
            "failure": "#dc3545",
            "error": "#dc3545",
            "warning": "#ffc107",
        }
        color = color_map.get(status, "#6c757d")

        emoji_map = {
            "success": ":white_check_mark:",
            "failure": ":x:",
            "error": ":exclamation:",
            "warning": ":warning:",
        }
        status_emoji = emoji_map.get(status, ":question:")

        mentions = ""
        if status in ("failure", "error") and config.mention_on_failure:
            mentions = " ".join(f"<@{uid}>" for uid in config.mention_on_failure) + " "

        if config.custom_message:
            text = config.custom_message.format(
                checkpoint=checkpoint_result.checkpoint_name,
                status=status.upper(),
                run_id=checkpoint_result.run_id,
                total_issues=stats.total_issues if stats else 0,
            )
        else:
            text = f"{mentions}{status_emoji} *Checkpoint '{checkpoint_result.checkpoint_name}'* - *{status.upper()}*"

        attachment: dict[str, Any] = {
            "color": color,
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": text}},
            ],
        }

        if config.include_details and stats:
            attachment["blocks"].append({
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Total Issues:*\n{stats.total_issues}"},
                    {"type": "mrkdwn", "text": f"*Pass Rate:*\n{stats.pass_rate * 100:.1f}%"},
                ],
            })

        payload: dict[str, Any] = {
            "username": config.username,
            "icon_emoji": config.icon_emoji,
            "attachments": [attachment],
        }

        if config.channel:
            payload["channel"] = config.channel

        return payload

    def validate_config(self) -> list[str]:
        """Validate configuration."""
        errors = []
        if not self._config.webhook_url:
            errors.append("webhook_url is required")
        return errors


# =============================================================================
# Async Store Result Action
# =============================================================================


@dataclass
class AsyncStoreConfig(ActionConfig):
    """Configuration for async store result action."""

    store_path: str = ""
    store_type: str = "file"  # file, s3, gcs
    format: str = "json"
    partition_by: str = "date"  # date, checkpoint, status
    compress: bool = False
    notify_on: NotifyCondition | str = NotifyCondition.ALWAYS


class AsyncStoreValidationResult(AsyncBaseAction[AsyncStoreConfig]):
    """Async action to store validation results.

    Supports local filesystem with async I/O.
    S3/GCS support available when boto3/google-cloud-storage installed.
    """

    action_type = "async_store_result"

    @classmethod
    def _default_config(cls) -> AsyncStoreConfig:
        return AsyncStoreConfig()

    async def _execute_async(
        self, checkpoint_result: "CheckpointResult"
    ) -> ActionResult:
        """Store result asynchronously."""
        config = self._config

        if not config.store_path:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="No store path configured",
                error="store_path is required",
            )

        if config.store_type == "file":
            return await self._store_to_file(checkpoint_result)
        elif config.store_type == "s3":
            return await self._store_to_s3(checkpoint_result)
        else:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message=f"Unsupported store type: {config.store_type}",
            )

    async def _store_to_file(
        self, checkpoint_result: "CheckpointResult"
    ) -> ActionResult:
        """Store to local filesystem."""
        config = self._config
        loop = asyncio.get_running_loop()

        def do_store() -> str:
            base_path = Path(config.store_path)

            # Partition
            if config.partition_by == "date":
                partition = checkpoint_result.run_time.strftime("%Y/%m/%d")
            elif config.partition_by == "status":
                partition = checkpoint_result.status.value
            else:
                partition = checkpoint_result.checkpoint_name

            output_dir = base_path / partition
            output_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{checkpoint_result.run_id}.{config.format}"
            output_path = output_dir / filename

            data = checkpoint_result.to_dict()
            content = json.dumps(data, indent=2, default=str)

            if config.compress:
                import gzip
                with gzip.open(f"{output_path}.gz", "wt") as f:
                    f.write(content)
                return str(output_path) + ".gz"
            else:
                output_path.write_text(content)
                return str(output_path)

        try:
            output_path = await loop.run_in_executor(None, do_store)
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.SUCCESS,
                message=f"Result stored to {output_path}",
                details={"path": output_path},
            )
        except Exception as e:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="Failed to store result",
                error=str(e),
            )

    async def _store_to_s3(
        self, checkpoint_result: "CheckpointResult"
    ) -> ActionResult:
        """Store to S3 (requires aioboto3 or runs boto3 in executor)."""
        try:
            import boto3
        except ImportError:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="boto3 not installed",
                error="pip install boto3",
            )

        config = self._config
        loop = asyncio.get_running_loop()

        def do_upload() -> str:
            # Parse s3://bucket/key
            path = config.store_path.replace("s3://", "")
            parts = path.split("/", 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""

            key = f"{prefix}/{checkpoint_result.run_id}.json"

            s3 = boto3.client("s3")
            data = json.dumps(checkpoint_result.to_dict(), default=str)
            s3.put_object(Bucket=bucket, Key=key, Body=data.encode())

            return f"s3://{bucket}/{key}"

        try:
            s3_path = await loop.run_in_executor(None, do_upload)
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.SUCCESS,
                message=f"Result stored to {s3_path}",
                details={"path": s3_path},
            )
        except Exception as e:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="Failed to store to S3",
                error=str(e),
            )


# =============================================================================
# Async Custom Action
# =============================================================================


@dataclass
class AsyncCustomConfig(ActionConfig):
    """Configuration for async custom action."""

    callback: Any = None  # Async callable
    notify_on: NotifyCondition | str = NotifyCondition.ALWAYS


class AsyncCustomAction(AsyncBaseAction[AsyncCustomConfig]):
    """Async custom action with user-defined logic.

    Example:
        >>> async def my_handler(result):
        ...     await some_async_operation(result)
        ...     return {"processed": True}
        >>>
        >>> action = AsyncCustomAction(callback=my_handler)
    """

    action_type = "async_custom"

    @classmethod
    def _default_config(cls) -> AsyncCustomConfig:
        return AsyncCustomConfig()

    async def _execute_async(
        self, checkpoint_result: "CheckpointResult"
    ) -> ActionResult:
        """Execute custom callback."""
        config = self._config

        if not config.callback:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="No callback configured",
            )

        try:
            if asyncio.iscoroutinefunction(config.callback):
                result = await config.callback(checkpoint_result)
            else:
                # Run sync callback in executor
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, config.callback, checkpoint_result
                )

            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.SUCCESS,
                message="Callback executed",
                details={"result": result if isinstance(result, (dict, list, str)) else str(result)},
            )

        except Exception as e:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="Callback failed",
                error=str(e),
            )
