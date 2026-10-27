"""Owned, expiring immutable snapshots; never a broker for arbitrary Python objects."""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from truthound.mcp.policy import MCPError


@dataclass(frozen=True)
class Resource:
    id: str
    kind: str
    path: Path
    size: int
    sha256: str
    media_type: str
    expires: float
    origins: tuple[Path, ...] = ()
    parents: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)


class SessionResources:
    """The service serializes access; ownership never comes from request arguments."""

    def __init__(
        self, policy, *, clock=time.monotonic, before_release=None, authorize_metadata=None
    ):
        self.policy = policy
        self.clock = clock
        self.before_release = before_release
        self.authorize_metadata = authorize_metadata
        self._temporary = tempfile.TemporaryDirectory(prefix="truthound-mcp-session-")
        self.root = Path(self._temporary.name)
        self._entries: dict[str, Resource] = {}
        self._pending: dict[str, Path] = {}
        self._closed = False

    def _check_open(self):
        if self._closed:
            raise MCPError("session_closed")

    @property
    def available_bytes(self):
        self.reap()
        return self.policy().max_session_bytes - sum(entry.size for entry in self._entries.values())

    def bind_model_inputs(self, identity, parents):
        """Resident model state must retain and reauthorize all training inputs."""
        return self.bind_resident_inputs(identity, parents, kind="model")

    def validate_resident_inputs(self, identity, parents, *, kind):
        """Reject self/descendant dependencies before approval or graph mutation."""
        self.get(identity, kind)
        for parent in parents:
            item = self.get(parent)
            if item.kind not in ({"asset", "artifact", "graph"} if kind == "monitor" else {"asset", "artifact"}):
                raise MCPError("invalid_handle")
            pending, visited = [item.id], set()
            while pending:
                current = pending.pop()
                if current == identity:
                    raise MCPError("invalid_arguments")
                if current in visited:
                    continue
                visited.add(current)
                pending.extend(self._entries[current].parents)

    def bind_resident_inputs(self, identity, parents, *, kind):
        """Keep native resident state dependent on all observed input grants."""
        from dataclasses import replace

        self.validate_resident_inputs(identity, parents, kind=kind)
        entry = self.get(identity, kind)
        bound = [self.get(parent) for parent in parents]
        if any(item.kind not in ({"asset", "artifact", "graph"} if kind == "monitor" else {"asset", "artifact"}) for item in bound):
            raise MCPError("invalid_handle")
        if bound:
            self._entries[identity] = replace(
                entry,
                parents=tuple(dict.fromkeys(entry.parents + tuple(parents))),
                origins=tuple(
                    set(entry.origins + tuple(origin for item in bound for origin in item.origins))
                ),
                expires=min(entry.expires, *(item.expires for item in bound)),
            )
        return self._entries[identity]

    def reserve(self, kind: str) -> tuple[str, Path]:
        self._check_open()
        self.reap()
        if len(self._entries) + len(self._pending) >= self.policy().max_handles:
            raise MCPError("resource_exhausted")
        if kind not in {
            "asset",
            "artifact",
            "schema",
            "context",
            "graph",
            "model",
            "source",
            "suite",
            "history",
            "monitor",
        }:
            raise MCPError("invalid_resource_kind")
        identity = kind + "_" + secrets.token_urlsafe(24)
        path = self.root / identity
        self._pending[identity] = path
        return identity, path

    def discard(self, identity: str):
        path = self._pending.pop(identity, None)
        if path is not None:
            path.unlink(missing_ok=True)

    def commit(self, identity, *, kind, media_type, origins=(), parents=(), metadata=None):
        self._check_open()
        path = self._pending.get(identity)
        if path is None or not identity.startswith(kind + "_"):
            raise MCPError("invalid_handle")
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            with os.fdopen(os.open(path, flags), "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode):
                    raise MCPError("invalid_resource")
                total = sum(entry.size for entry in self._entries.values())
                if info.st_size > self.policy().max_session_bytes - total:
                    raise MCPError("resource_exhausted")
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            expiry = self.clock() + self.policy().handle_ttl_seconds
            for origin in origins:
                self.policy().authorize_origin(origin)
            for parent in parents:
                expiry = min(expiry, self.get(parent).expires)
            entry = Resource(
                identity,
                kind,
                path,
                info.st_size,
                digest,
                media_type,
                expiry,
                tuple(origins),
                tuple(parents),
                dict(metadata or {}),
            )
            self._entries[identity] = entry
            del self._pending[identity]
            return entry
        except BaseException:
            self.discard(identity)
            raise

    def get(self, identity: str, kind: str | None = None) -> Resource:
        self._check_open()
        entry = self._entries.get(identity)
        if entry is None or (kind and entry.kind != kind):
            raise MCPError("invalid_handle")
        if self.clock() >= entry.expires:
            self.release(identity)
            raise MCPError("handle_expired")
        for origin in entry.origins:
            self.policy().authorize_origin(origin)
        if any(
            key in entry.metadata
            for key in (
                "durable_policy",
                "source_binding",
                "monitor_binding",
                "checkpoint_binding",
                "native_binding",
            )
        ):
            if self.authorize_metadata is None:
                raise MCPError("access_denied")
            self.authorize_metadata(entry.metadata)
        if "workspace_root" in entry.metadata:
            self.policy().resolve_workspace(entry.metadata["workspace_root"])
        if "binding_digest" in entry.metadata:
            from truthound.mcp.bindings import load

            if str(self.policy().control_directory) != entry.metadata.get("binding_control_root"):
                raise MCPError("access_denied")
            try:
                current_digest = load(self.policy())[1]
            except (MCPError, OSError, ValueError):
                raise MCPError("access_denied") from None
            if current_digest != entry.metadata["binding_digest"]:
                raise MCPError("access_denied")
        for parent in entry.parents:
            self.get(parent)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            with os.fdopen(os.open(entry.path, flags), "rb") as stream:
                info = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_size != entry.size
                    or hashlib.file_digest(stream, "sha256").hexdigest() != entry.sha256
                ):
                    raise MCPError("resource_changed")
        except OSError:
            raise MCPError("resource_changed") from None
        return entry

    def release_owned(self, identity: str):
        """Revoke an owned session input together with its native source.

        Cleanup must work after a grant is revoked, without authorizing a new
        read of the denied resource. Metadata is created only by the adapter.
        """
        self._check_open()
        entry = self._entries.get(identity)
        if entry is None:
            raise MCPError("invalid_handle")
        self.release(entry.metadata.get("owned_input", identity))

    def release(self, identity: str):
        self._check_open()
        if identity not in self._entries:
            raise MCPError("invalid_handle")
        for child in [item.id for item in self._entries.values() if identity in item.parents]:
            if child in self._entries:
                self.release(child)
        entry = self._entries[identity]
        if self.before_release is not None:
            self.before_release(identity)
        entry.path.unlink(missing_ok=True)
        del self._entries[identity]

    def reap(self):
        for identity in [
            entry.id for entry in self._entries.values() if self.clock() >= entry.expires
        ]:
            if identity in self._entries:
                self.release(identity)

    def close(self):
        if not self._closed:
            self._closed = True
            self._entries.clear()
            self._pending.clear()
            self._temporary.cleanup()


class BoundedWriter:
    """File-like sink for Core writers; partial output never becomes a handle."""

    def __init__(self, path: Path, maximum: int):
        self.stream = path.open("xb")
        self.maximum = maximum
        self.written = 0
        self.exhausted = False

    def write(self, data):
        if len(data) > self.maximum - self.written:
            self.exhausted = True
            raise MCPError("resource_exhausted")
        count = self.stream.write(data)
        self.written += count
        return count

    def flush(self):
        self.stream.flush()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()
        # Native serializers can wrap Python sink exceptions. Preserve the
        # quota verdict, including writers that catch a failed write internally.
        if self.exhausted:
            raise MCPError("resource_exhausted") from None
