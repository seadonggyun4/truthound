"""Shared filesystem containment and private atomic publication for stores."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path


def component(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or any(token in value for token in ("/", "\\", "\x00"))
    ):
        raise ValueError("A native stream identifier must be one path component")
    return value


def contained_path(root: Path, value: str | Path) -> Path:
    root = root.absolute()
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if path.is_symlink() or path.resolve().parent != root.resolve():
        raise ValueError("Native streaming file must remain inside its selected directory")
    return path


def publish(path: Path, content: bytes, *, replace: bool = True) -> None:
    """Flush a private sibling before publishing; retain previous bytes on failure.

    Chunk publication is create-only. An unconfirmed existing chunk is evidence,
    never an invitation to overwrite it or infer that an earlier write succeeded.
    """
    path = path.absolute()
    path = contained_path(path.parent, path)
    descriptor, temporary = tempfile.mkstemp(prefix=".stream-", suffix=".tmp", dir=path.parent)
    scratch = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(scratch, path)
        else:
            os.link(scratch, path)
            scratch.unlink()
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        scratch.unlink(missing_ok=True)


def read_json(content):
    """Reject ambiguous keys and nonfinite numbers in persisted records/indexes."""

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate persisted JSON key")
            result[key] = value
        return result

    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Nonfinite persisted JSON number")
        return number

    return json.loads(content, object_pairs_hook=unique, parse_float=finite, parse_constant=finite)


def validate_values(value, active=None, depth=0):
    """Reject nonfinite numeric values and cyclic persisted container graphs."""
    if depth > 64:
        raise ValueError("Persisted metadata exceeds nesting limit")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite persisted number")
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        active = set() if active is None else active
        identity = id(value)
        if identity in active:
            raise ValueError("Cyclic persisted value")
        active.add(identity)
        try:
            items = (
                (part for pair in value.items() for part in pair)
                if isinstance(value, dict)
                else value
            )
            for item in items:
                validate_values(item, active, depth + 1)
        finally:
            active.remove(identity)


def read_yaml(content):
    """Safe YAML with unambiguous mapping keys and finite container values."""
    import yaml

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        loader.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError("Duplicate persisted YAML key")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    result = yaml.load(content, Loader=UniqueLoader)
    validate_values(result)
    return result
