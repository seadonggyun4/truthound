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

    def __post_init__(self) -> None:
        roots = tuple(Path(p).expanduser().resolve(strict=True) for p in self.allowed_directories)
        if not roots or any(not p.is_dir() for p in roots):
            raise ValueError("At least one existing allowed directory is required")
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

    def resolve(self, value: str) -> Path:
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
            if resolved.suffix.lower() not in {".csv", ".parquet", ".json", ".ndjson", ".jsonl"}:
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
