"""Committed local native-Parquet checkpoints with explicit integrity receipts.

Native engines own serialization. This module publishes only completed snapshots;
failed staging directories remain available for operator inspection. Checkpoints
are operator-owned data, never automatic package or MCP response artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

_MANIFEST = "checkpoint.json"
_MAX_MANIFEST = 16 * 1024 * 1024


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Native checkpoint cannot contain symbolic links")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {
                "size": path.stat().st_size,
                "sha256": _digest(path),
            }
        elif not path.is_dir():
            raise ValueError("Native checkpoint contains a non-regular entry")
    if not result or not any(name.endswith(".parquet") for name in result):
        raise ValueError("Native checkpoint contains no Parquet data")
    return result


def _sync_directory(path):
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish(directory, backend, writer):
    """Write native Parquet then publish its checked inventory under a unique ID."""
    if not isinstance(directory, str) or not directory or "://" in directory:
        raise ValueError("A local checkpoint_dir is required for this engine")
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = uuid.uuid4().hex
    pending = root / (".pending-" + identity)
    pending.mkdir()
    data = pending / "data"
    # Native writers create data/. An exception leaves an uncommitted snapshot.
    writer(data)
    files = _inventory(data)
    for relative in files:
        # Windows' CRT commit requires a writable descriptor. These are the
        # files just created by our native writer in the unpublished snapshot.
        with (data / relative).open("r+b") as stream:
            os.fsync(stream.fileno())
    document = json.dumps(
        {"version": 1, "backend": backend, "files": files}, sort_keys=True
    ).encode()
    if len(document) > _MAX_MANIFEST:
        raise ValueError("Native checkpoint inventory exceeds its manifest bound")
    with (pending / _MANIFEST).open("xb") as stream:
        stream.write(document)
        stream.flush()
        os.fsync(stream.fileno())
    for child in sorted((path for path in data.rglob("*") if path.is_dir()), reverse=True):
        _sync_directory(child)
    _sync_directory(data)
    _sync_directory(pending)
    committed = root / identity
    os.rename(pending, committed)
    _sync_directory(root)
    return str(committed)


def validate(location, backend):
    """Check a completed checkpoint before a native engine opens its Parquet files."""
    root = Path(location).expanduser()
    if root.is_symlink() or root.name.startswith(".pending-"):
        raise ValueError("Native checkpoint is not a committed snapshot")
    manifest = root / _MANIFEST
    data = root / "data"
    if manifest.is_symlink() or data.is_symlink() or not manifest.is_file() or not data.is_dir():
        raise ValueError("Native checkpoint is incomplete")
    with manifest.open("rb") as stream:
        content = stream.read(_MAX_MANIFEST + 1)
    if len(content) > _MAX_MANIFEST:
        raise ValueError("Native checkpoint manifest exceeds its bound")
    try:
        document = json.loads(content)
    except (ValueError, UnicodeError) as error:
        raise ValueError("Native checkpoint manifest is invalid") from error
    if (
        not isinstance(document, dict)
        or set(document) != {"version", "backend", "files"}
        or document["version"] != 1
        or document["backend"] != backend
        or document["files"] != _inventory(data)
    ):
        raise ValueError("Native checkpoint integrity or backend differs")
    return str(data.resolve())
