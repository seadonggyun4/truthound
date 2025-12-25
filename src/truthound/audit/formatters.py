"""Formatters and serializers for audit events.

This module provides various output formats for audit events:
- JSON: Standard JSON format
- JSONL: JSON Lines format for streaming
- CEF: Common Event Format (security industry standard)
- LEEF: Log Event Extended Format (IBM QRadar)
- Syslog: RFC 5424 syslog format
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from truthound.audit.core import (
    AuditEvent,
    AuditFormatter,
    AuditSeverity,
)


# =============================================================================
# JSON Formatter
# =============================================================================


class JSONFormatter(AuditFormatter):
    """JSON formatter for audit events.

    Example:
        >>> formatter = JSONFormatter(pretty=True)
        >>> json_str = formatter.format(event)
    """

    def __init__(
        self,
        pretty: bool = False,
        include_null: bool = False,
        date_format: str = "iso",
    ) -> None:
        """Initialize JSON formatter.

        Args:
            pretty: Pretty print with indentation.
            include_null: Include null/None values.
            date_format: Date format ("iso", "unix", "both").
        """
        self._pretty = pretty
        self._include_null = include_null
        self._date_format = date_format

    def format(self, event: AuditEvent) -> str:
        """Format event to JSON string."""
        data = event.to_dict()

        if not self._include_null:
            data = self._remove_nulls(data)

        if self._date_format == "unix":
            data["timestamp"] = event.timestamp_unix
        elif self._date_format == "both":
            data["timestamp_iso"] = event.timestamp_iso
            data["timestamp_unix"] = event.timestamp_unix

        indent = 2 if self._pretty else None
        return json.dumps(data, indent=indent, default=str)

    def parse(self, data: str) -> AuditEvent:
        """Parse JSON string to event."""
        parsed = json.loads(data)
        return AuditEvent.from_dict(parsed)

    def _remove_nulls(self, data: Any) -> Any:
        """Recursively remove null values."""
        if isinstance(data, dict):
            return {
                k: self._remove_nulls(v)
                for k, v in data.items()
                if v is not None and v != "" and v != []
            }
        elif isinstance(data, list):
            return [self._remove_nulls(v) for v in data if v is not None]
        return data


# =============================================================================
# CEF Formatter
# =============================================================================


class CEFFormatter(AuditFormatter):
    """Common Event Format (CEF) formatter.

    CEF is a standard log format used by security tools like Splunk,
    ArcSight, and others.

    Format:
        CEF:Version|Device Vendor|Device Product|Device Version|Signature ID|Name|Severity|Extension

    Example:
        >>> formatter = CEFFormatter(
        ...     vendor="Truthound",
        ...     product="DataQuality",
        ...     version="1.0",
        ... )
        >>> cef_str = formatter.format(event)
    """

    SEVERITY_MAP = {
        AuditSeverity.DEBUG: 1,
        AuditSeverity.INFO: 3,
        AuditSeverity.WARNING: 6,
        AuditSeverity.ERROR: 8,
        AuditSeverity.CRITICAL: 10,
    }

    def __init__(
        self,
        vendor: str = "Truthound",
        product: str = "AuditLog",
        version: str = "1.0",
    ) -> None:
        """Initialize CEF formatter.

        Args:
            vendor: Device vendor name.
            product: Device product name.
            version: Device version.
        """
        self._vendor = vendor
        self._product = product
        self._version = version

    def format(self, event: AuditEvent) -> str:
        """Format event to CEF string."""
        # CEF header fields
        cef_version = 0
        signature_id = event.event_type.value
        name = event.action or event.event_type.value
        severity = self.SEVERITY_MAP.get(event.severity, 5)

        # Build header
        header = f"CEF:{cef_version}|{self._vendor}|{self._product}|{self._version}|{signature_id}|{name}|{severity}"

        # Build extension
        extensions = []

        # Standard CEF fields
        extensions.append(f"rt={int(event.timestamp_unix * 1000)}")
        extensions.append(f"outcome={event.outcome.value}")

        if event.actor:
            if event.actor.id:
                extensions.append(f"suser={self._escape(event.actor.id)}")
            if event.actor.ip_address:
                extensions.append(f"src={event.actor.ip_address}")
            if event.actor.name:
                extensions.append(f"sname={self._escape(event.actor.name)}")

        if event.resource:
            if event.resource.id:
                extensions.append(f"duid={self._escape(event.resource.id)}")
            if event.resource.type:
                extensions.append(f"deviceCustomString1={self._escape(event.resource.type)}")
            if event.resource.name:
                extensions.append(f"fname={self._escape(event.resource.name)}")

        if event.message:
            extensions.append(f"msg={self._escape(event.message)}")

        if event.reason:
            extensions.append(f"reason={self._escape(event.reason)}")

        if event.context.request_id:
            extensions.append(f"cn1={self._escape(event.context.request_id)}")

        extension_str = " ".join(extensions)
        return f"{header}|{extension_str}"

    def parse(self, data: str) -> AuditEvent:
        """Parse CEF string to event."""
        # Basic CEF parsing
        parts = data.split("|")
        if len(parts) < 8:
            raise ValueError(f"Invalid CEF format: {data}")

        # Parse extensions
        extensions = self._parse_extensions(parts[7] if len(parts) > 7 else "")

        from truthound.audit.core import (
            AuditActor,
            AuditResource,
            AuditContext,
            AuditEventType,
            AuditOutcome,
        )

        # Map severity back
        cef_severity = int(parts[6]) if parts[6].isdigit() else 5
        severity = AuditSeverity.INFO
        for sev, val in self.SEVERITY_MAP.items():
            if val == cef_severity:
                severity = sev
                break

        # Create event
        event_type = AuditEventType.CUSTOM
        try:
            event_type = AuditEventType(parts[4])
        except ValueError:
            pass

        outcome = AuditOutcome.UNKNOWN
        if "outcome" in extensions:
            try:
                outcome = AuditOutcome(extensions["outcome"])
            except ValueError:
                pass

        actor = None
        if "suser" in extensions or "src" in extensions:
            actor = AuditActor(
                id=extensions.get("suser", ""),
                ip_address=extensions.get("src", ""),
                name=extensions.get("sname", ""),
            )

        resource = None
        if "duid" in extensions:
            resource = AuditResource(
                id=extensions.get("duid", ""),
                type=extensions.get("deviceCustomString1", ""),
                name=extensions.get("fname", ""),
            )

        context = AuditContext(
            request_id=extensions.get("cn1", ""),
        )

        # Parse timestamp
        timestamp = datetime.now(timezone.utc)
        if "rt" in extensions:
            timestamp = datetime.fromtimestamp(
                int(extensions["rt"]) / 1000,
                tz=timezone.utc,
            )

        return AuditEvent(
            event_type=event_type,
            severity=severity,
            action=parts[5],
            outcome=outcome,
            message=extensions.get("msg", ""),
            reason=extensions.get("reason", ""),
            actor=actor,
            resource=resource,
            context=context,
            timestamp=timestamp,
        )

    def _escape(self, value: str) -> str:
        """Escape CEF special characters."""
        return (
            value.replace("\\", "\\\\")
            .replace("=", "\\=")
            .replace("|", "\\|")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
        )

    def _parse_extensions(self, ext_str: str) -> dict[str, str]:
        """Parse CEF extension string."""
        extensions = {}
        current_key = ""
        current_value = ""
        in_value = False

        i = 0
        while i < len(ext_str):
            char = ext_str[i]

            if not in_value:
                if char == "=":
                    in_value = True
                else:
                    current_key += char
            else:
                if char == " " and i + 1 < len(ext_str):
                    # Check if next is a new key
                    rest = ext_str[i + 1 :]
                    if "=" in rest:
                        next_eq = rest.index("=")
                        potential_key = rest[:next_eq]
                        if potential_key.replace("_", "").isalnum():
                            extensions[current_key.strip()] = current_value.strip()
                            current_key = ""
                            current_value = ""
                            in_value = False
                            i += 1
                            continue
                current_value += char

            i += 1

        if current_key and in_value:
            extensions[current_key.strip()] = current_value.strip()

        return extensions


# =============================================================================
# LEEF Formatter
# =============================================================================


class LEEFFormatter(AuditFormatter):
    """Log Event Extended Format (LEEF) formatter.

    LEEF is used by IBM QRadar and other SIEM systems.

    Format:
        LEEF:Version|Vendor|Product|Version|EventID|Key=Value\tKey=Value...

    Example:
        >>> formatter = LEEFFormatter()
        >>> leef_str = formatter.format(event)
    """

    def __init__(
        self,
        vendor: str = "Truthound",
        product: str = "AuditLog",
        version: str = "1.0",
    ) -> None:
        """Initialize LEEF formatter."""
        self._vendor = vendor
        self._product = product
        self._version = version

    def format(self, event: AuditEvent) -> str:
        """Format event to LEEF string."""
        leef_version = "2.0"
        event_id = event.event_type.value

        header = f"LEEF:{leef_version}|{self._vendor}|{self._product}|{self._version}|{event_id}"

        # Build attributes
        attrs = []
        attrs.append(f"devTime={event.timestamp_iso}")
        attrs.append(f"cat={event.category.value}")
        attrs.append(f"sev={self._map_severity(event.severity)}")
        attrs.append(f"outcome={event.outcome.value}")

        if event.actor:
            if event.actor.id:
                attrs.append(f"usrName={event.actor.id}")
            if event.actor.ip_address:
                attrs.append(f"src={event.actor.ip_address}")

        if event.resource:
            if event.resource.id:
                attrs.append(f"resource={event.resource.id}")
            if event.resource.type:
                attrs.append(f"resourceType={event.resource.type}")

        if event.action:
            attrs.append(f"action={event.action}")

        if event.message:
            attrs.append(f"msg={self._escape(event.message)}")

        return header + "|" + "\t".join(attrs)

    def parse(self, data: str) -> AuditEvent:
        """Parse LEEF string to event."""
        parts = data.split("|")
        if len(parts) < 6:
            raise ValueError(f"Invalid LEEF format: {data}")

        # Parse attributes
        attrs = {}
        if len(parts) > 5:
            for pair in parts[5].split("\t"):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    attrs[key] = value

        from truthound.audit.core import (
            AuditActor,
            AuditResource,
            AuditEventType,
            AuditOutcome,
            AuditCategory,
        )

        event_type = AuditEventType.CUSTOM
        try:
            event_type = AuditEventType(parts[4])
        except ValueError:
            pass

        outcome = AuditOutcome.UNKNOWN
        if "outcome" in attrs:
            try:
                outcome = AuditOutcome(attrs["outcome"])
            except ValueError:
                pass

        category = AuditCategory.CUSTOM
        if "cat" in attrs:
            try:
                category = AuditCategory(attrs["cat"])
            except ValueError:
                pass

        actor = None
        if "usrName" in attrs or "src" in attrs:
            actor = AuditActor(
                id=attrs.get("usrName", ""),
                ip_address=attrs.get("src", ""),
            )

        resource = None
        if "resource" in attrs:
            resource = AuditResource(
                id=attrs.get("resource", ""),
                type=attrs.get("resourceType", ""),
            )

        timestamp = datetime.now(timezone.utc)
        if "devTime" in attrs:
            try:
                timestamp = datetime.fromisoformat(
                    attrs["devTime"].replace("Z", "+00:00")
                )
            except ValueError:
                pass

        return AuditEvent(
            event_type=event_type,
            category=category,
            action=attrs.get("action", ""),
            outcome=outcome,
            message=attrs.get("msg", ""),
            actor=actor,
            resource=resource,
            timestamp=timestamp,
        )

    def _map_severity(self, severity: AuditSeverity) -> int:
        """Map severity to LEEF numeric value."""
        mapping = {
            AuditSeverity.DEBUG: 1,
            AuditSeverity.INFO: 3,
            AuditSeverity.WARNING: 6,
            AuditSeverity.ERROR: 8,
            AuditSeverity.CRITICAL: 10,
        }
        return mapping.get(severity, 5)

    def _escape(self, value: str) -> str:
        """Escape special characters."""
        return value.replace("\t", " ").replace("\n", " ").replace("\r", " ")


# =============================================================================
# Syslog Formatter
# =============================================================================


class SyslogFormatter(AuditFormatter):
    """RFC 5424 Syslog formatter.

    Example:
        >>> formatter = SyslogFormatter(app_name="myapp")
        >>> syslog_str = formatter.format(event)
    """

    SEVERITY_MAP = {
        AuditSeverity.DEBUG: 7,    # Debug
        AuditSeverity.INFO: 6,     # Informational
        AuditSeverity.WARNING: 4,  # Warning
        AuditSeverity.ERROR: 3,    # Error
        AuditSeverity.CRITICAL: 2, # Critical
    }

    def __init__(
        self,
        facility: int = 16,  # local0
        app_name: str = "truthound",
        hostname: str = "-",
    ) -> None:
        """Initialize Syslog formatter.

        Args:
            facility: Syslog facility (0-23).
            app_name: Application name.
            hostname: Hostname.
        """
        self._facility = facility
        self._app_name = app_name
        self._hostname = hostname

    def format(self, event: AuditEvent) -> str:
        """Format event to Syslog string."""
        # Calculate priority
        severity = self.SEVERITY_MAP.get(event.severity, 6)
        priority = self._facility * 8 + severity

        # RFC 5424 format
        version = 1
        timestamp = event.timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        procid = "-"
        msgid = event.event_type.value

        # Structured data
        sd = self._build_structured_data(event)

        # Message
        msg = event.message or event.action or ""

        return f"<{priority}>{version} {timestamp} {self._hostname} {self._app_name} {procid} {msgid} {sd} {msg}"

    def parse(self, data: str) -> AuditEvent:
        """Parse Syslog string to event."""
        # Basic parsing - extract priority and message
        import re

        match = re.match(r"<(\d+)>(\d+) (\S+) (\S+) (\S+) (\S+) (\S+) (\[.*?\])?\s*(.*)", data)
        if not match:
            raise ValueError(f"Invalid syslog format: {data}")

        priority = int(match.group(1))
        timestamp_str = match.group(3)
        msgid = match.group(7)
        message = match.group(9) or ""

        # Calculate severity from priority
        severity_num = priority % 8
        severity = AuditSeverity.INFO
        for sev, num in self.SEVERITY_MAP.items():
            if num == severity_num:
                severity = sev
                break

        # Parse timestamp
        try:
            timestamp = datetime.fromisoformat(
                timestamp_str.replace("Z", "+00:00")
            )
        except ValueError:
            timestamp = datetime.now(timezone.utc)

        event_type = AuditEventType.CUSTOM
        try:
            event_type = AuditEventType(msgid)
        except ValueError:
            pass

        return AuditEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            timestamp=timestamp,
        )

    def _build_structured_data(self, event: AuditEvent) -> str:
        """Build RFC 5424 structured data."""
        parts = []

        # Audit metadata
        sd_id = "audit@32473"
        params = []
        params.append(f'id="{event.id}"')
        params.append(f'outcome="{event.outcome.value}"')
        params.append(f'category="{event.category.value}"')

        if event.actor:
            params.append(f'actor="{event.actor.id}"')
        if event.resource:
            params.append(f'resource="{event.resource.id}"')

        parts.append(f"[{sd_id} {' '.join(params)}]")

        return "".join(parts) if parts else "-"


# =============================================================================
# Human-Readable Formatter
# =============================================================================


class HumanFormatter(AuditFormatter):
    """Human-readable formatter for console output.

    Example:
        >>> formatter = HumanFormatter(color=True)
        >>> print(formatter.format(event))
    """

    COLORS = {
        AuditSeverity.DEBUG: "\033[90m",    # Gray
        AuditSeverity.INFO: "\033[37m",     # White
        AuditSeverity.WARNING: "\033[33m",  # Yellow
        AuditSeverity.ERROR: "\033[31m",    # Red
        AuditSeverity.CRITICAL: "\033[91m", # Bright Red
    }
    RESET = "\033[0m"

    def __init__(
        self,
        color: bool = False,
        include_data: bool = False,
        timestamp_format: str = "%Y-%m-%d %H:%M:%S",
    ) -> None:
        """Initialize human formatter.

        Args:
            color: Use ANSI colors.
            include_data: Include event data.
            timestamp_format: Timestamp format string.
        """
        self._color = color
        self._include_data = include_data
        self._timestamp_format = timestamp_format

    def format(self, event: AuditEvent) -> str:
        """Format event to human-readable string."""
        parts = []

        # Timestamp
        ts = event.timestamp.strftime(self._timestamp_format)
        parts.append(f"[{ts}]")

        # Severity with color
        severity = event.severity.value.upper()
        if self._color:
            color = self.COLORS.get(event.severity, "")
            parts.append(f"{color}{severity:8}{self.RESET}")
        else:
            parts.append(f"{severity:8}")

        # Event type and action
        parts.append(f"{event.event_type.value}")
        if event.action:
            parts.append(f"action={event.action}")

        # Outcome
        outcome_symbol = "✓" if event.outcome.value == "success" else "✗"
        parts.append(f"[{outcome_symbol} {event.outcome.value}]")

        # Actor
        if event.actor:
            parts.append(f"actor={event.actor.id}")

        # Resource
        if event.resource:
            parts.append(f"resource={event.resource.id}")

        # Message
        if event.message:
            parts.append(f'"{event.message}"')

        # Data
        if self._include_data and event.data:
            parts.append(f"data={json.dumps(event.data)}")

        return " ".join(parts)

    def parse(self, data: str) -> AuditEvent:
        """Parse is not supported for human format."""
        raise NotImplementedError("Human format cannot be parsed")


# =============================================================================
# Formatter Factory
# =============================================================================


def create_formatter(
    format_type: str = "json",
    **kwargs: Any,
) -> AuditFormatter:
    """Create formatter from type string.

    Args:
        format_type: Formatter type ("json", "cef", "leef", "syslog", "human").
        **kwargs: Formatter-specific options.

    Returns:
        Formatter instance.
    """
    formatters = {
        "json": JSONFormatter,
        "cef": CEFFormatter,
        "leef": LEEFFormatter,
        "syslog": SyslogFormatter,
        "human": HumanFormatter,
    }

    if format_type not in formatters:
        raise ValueError(f"Unknown formatter: {format_type}")

    return formatters[format_type](**kwargs)
