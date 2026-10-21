"""Explicit local capabilities, independent of transport and validation engines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class MCPError(Exception):
    """A stable public code; private exception details never cross the boundary."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class MCPPolicy:
    allowed_directories: tuple[Path, ...]
    max_file_bytes: int = 100 * 1024 * 1024
    max_items: int = 100
    max_response_bytes: int = 65536
    timeout_seconds: int = 30
    profile: str = "summary"
    allow_sensitive_results: bool = False
    workspace_directories: tuple[Path, ...] = ()
    control_directory: Path | None = None
    max_handles: int = 128
    max_session_bytes: int = 1024**3
    handle_ttl_seconds: int = 900

    def __post_init__(self) -> None:
        roots = tuple(Path(p).expanduser().resolve(strict=True) for p in self.allowed_directories)
        if not roots or any(not p.is_dir() for p in roots):
            raise ValueError("At least one existing allowed directory is required")
        if self.profile not in {"summary", "core"}:
            raise ValueError("Invalid MCP profile")
        if type(self.allow_sensitive_results) is not bool:
            raise ValueError("Invalid disclosure policy")
        if self.allow_sensitive_results and self.profile != "core":
            raise ValueError("Sensitive results require the core profile")
        for name, low, high in (
            ("max_file_bytes", 1, 1024**3),
            ("max_items", 1, 1000),
            ("max_response_bytes", 1024, 1024**2),
            ("timeout_seconds", 1, 300),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid {name}")
        object.__setattr__(self, "allowed_directories", roots)
        workspaces = tuple(
            Path(p).expanduser().resolve(strict=True) for p in self.workspace_directories
        )
        if any(not p.is_dir() for p in workspaces):
            raise ValueError("Workspace roots must exist")
        object.__setattr__(self, "workspace_directories", workspaces)
        for name, lower, upper in (
            ("max_handles", 1, 1024),
            ("max_session_bytes", 1024, 8 * 1024**3),
            ("handle_ttl_seconds", 1, 3600),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"Invalid {name}")
        if self.control_directory is not None:
            import os

            control = Path(self.control_directory).expanduser().resolve(strict=True)
            if not control.is_dir() or any(
                control.is_relative_to(root) or root.is_relative_to(control)
                for root in roots + workspaces
            ):
                raise ValueError("Control directory must be separate from data/workspace roots")
            if os.name != "nt" and (
                control.stat().st_uid != os.getuid() or control.stat().st_mode & 0o077
            ):
                raise ValueError("Control directory must be private and owned by the operator")
            object.__setattr__(self, "control_directory", control)

    def authorize_origin(self, origin: Path) -> None:
        """Recheck a captured asset's directory grant without reopening its source."""
        if not any(origin.is_relative_to(root) for root in self.allowed_directories):
            raise MCPError("access_denied")

    def resolve_workspace(self, value: str) -> Path:
        if (
            not self.workspace_directories
            or not isinstance(value, str)
            or not value
            or len(value) > 4096
        ):
            raise MCPError("access_denied")
        if any(char in value for char in ("\x00", "://", "*", "?", "[", "]")):
            raise MCPError("invalid_path")
        supplied = Path(value)
        if ".." in supplied.parts:
            raise MCPError("invalid_path")
        path = supplied if supplied.is_absolute() else self.workspace_directories[0] / supplied
        try:
            resolved = path.resolve(strict=True)
            root = next((r for r in self.workspace_directories if resolved.is_relative_to(r)), None)
            if root is None or not resolved.is_dir():
                raise MCPError("access_denied")
            cursor = path
            while True:
                if cursor.is_symlink():
                    raise MCPError("access_denied")
                if cursor.samefile(root):
                    return resolved
                if cursor == cursor.parent:
                    raise MCPError("access_denied")
                cursor = cursor.parent
        except (OSError, ValueError, RuntimeError):
            raise MCPError("access_denied") from None

    def resolve(self, value: str, *, suffixes=None) -> Path:
        if not isinstance(value, str) or not value or len(value) > 4096:
            raise MCPError("invalid_path")
        if any(c in value for c in ("\x00", "://", "*", "?", "[", "]")):
            raise MCPError("invalid_path")
        supplied = Path(value)
        if ".." in supplied.parts:
            raise MCPError("invalid_path")
        path = supplied if supplied.is_absolute() else self.allowed_directories[0] / supplied
        try:
            resolved = path.resolve(strict=True)
            if not any(resolved.is_relative_to(root) for root in self.allowed_directories):
                raise MCPError("access_denied")
            # Refuse symlinks below the configured root, including in directories.
            root = next(r for r in self.allowed_directories if resolved.is_relative_to(r))
            cursor = path
            while True:
                if cursor.is_symlink():
                    raise MCPError("access_denied")
                if cursor.samefile(root):
                    break
                if cursor == cursor.parent:
                    raise MCPError("access_denied")
                cursor = cursor.parent
            if resolved.suffix.lower() not in (
                suffixes or {".csv", ".parquet", ".pq", ".json", ".ndjson", ".jsonl"}
            ):
                raise MCPError("unsupported_format")
            if not resolved.is_file():
                raise MCPError("invalid_path")
            if resolved.stat().st_size > self.max_file_bytes:
                raise MCPError("file_too_large")
            return resolved
        except MCPError:
            raise
        except (OSError, ValueError, RuntimeError):
            raise MCPError("access_denied") from None
