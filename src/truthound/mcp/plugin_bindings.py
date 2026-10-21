"""Pinned installed Core PluginManager capabilities, not an import-string endpoint."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from importlib.machinery import PathFinder
from pathlib import Path

from pydantic import Field, JsonValue

from truthound.mcp.contracts import Name, WireModel
from truthound.mcp.policy import MCPError


class PluginBinding(WireModel):
    distribution: Name
    version: Name
    entry_point: Name
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capabilities: dict[Name, Name]
    settings: dict[str, JsonValue] = Field(default_factory=dict)


def fingerprint(distribution):
    """Inspect installed distribution files without importing its Python code."""
    try:
        package = importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        raise MCPError("missing_dependency") from None
    if not package.files:
        raise MCPError("invalid_plugin_binding")
    direct = package.read_text("direct_url.json")
    if direct and json.loads(direct).get("dir_info", {}).get("editable"):
        raise MCPError("editable_plugin_not_supported")
    root = Path(package.locate_file("")).resolve(strict=True)
    digest, owned = hashlib.sha256(), set()
    for relative in sorted(package.files, key=str):
        if "__pycache__" in relative.parts or str(relative).endswith((".pyc", ".pyo")):
            continue
        path = Path(package.locate_file(relative))
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            # Installed executable launchers live outside site-packages; they
            # are not imported by this binding and never become callable inputs.
            continue
        if path.is_symlink() or not resolved.is_file():
            raise MCPError("invalid_plugin_binding")
        label = resolved.relative_to(root).as_posix()
        owned.add(label)
        digest.update(label.encode() + b"\x00")
        with resolved.open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    return package, digest.hexdigest(), owned


def _verify_import_path(module, root, owned):
    """Resolve without importing parent packages; reject shadowed entrypoints.

    Metadata ownership alone does not prove Python would load those files.
    Inspect every parent before entry.load() can execute its initializer.
    """
    search = sys.path
    parts = module.split(".")
    for index in range(len(parts)):
        name = ".".join(parts[: index + 1])
        resolved = PathFinder.find_spec(name, search)
        if resolved is None:
            raise MCPError("plugin_entrypoint_not_owned")
        if resolved.origin is None:
            locations = list(resolved.submodule_search_locations or ())
            expected = root.joinpath(*parts[: index + 1]).resolve()
            if not locations or any(Path(path).resolve() != expected for path in locations):
                raise MCPError("plugin_entrypoint_not_owned")
        else:
            path = Path(resolved.origin).resolve()
            if not path.is_relative_to(root) or path.relative_to(root).as_posix() not in owned:
                raise MCPError("plugin_entrypoint_not_owned")
            cached = sys.modules.get(name)
            if cached is not None and (
                not getattr(cached, "__file__", None) or Path(cached.__file__).resolve() != path
            ):
                raise MCPError("plugin_already_imported")
        search = resolved.submodule_search_locations


@contextmanager
def approved_capabilities(specifications):
    """Load only exact approved entrypoints, then use Core lifecycle/ports.

    This is trusted code execution with the host user's privileges. It is not
    an OS sandbox, a package installer, or protection from hostile local users.
    """
    if not specifications:
        yield {}
        return
    from truthound.plugins import Plugin, PluginConfig, PluginManager
    from truthound.plugins.manager import PluginManagerConfig

    # Force source imports into a fresh owned bytecode cache, rather than
    # trusting pre-existing unpinned __pycache__ contents for plugin modules.
    previous = sys.pycache_prefix
    with (
        tempfile.TemporaryDirectory(prefix="truthound-mcp-plugin-cache-") as cache,
        ExitStack() as stack,
    ):
        sys.pycache_prefix = cache
        try:
            result = {}
            loaded = {}
            for raw in specifications.values():
                bound = PluginBinding.model_validate(raw)
                package, actual, owned = fingerprint(bound.distribution)
                if package.version != bound.version or actual != bound.sha256:
                    raise MCPError("plugin_identity_mismatch")
                matches = [
                    entry
                    for entry in package.entry_points
                    if entry.group == "truthound.plugins" and entry.name == bound.entry_point
                ]
                if len(matches) != 1:
                    raise MCPError("invalid_plugin_binding")
                entry = matches[0]
                path = entry.module.replace(".", "/")
                if path + ".py" not in owned and path + "/__init__.py" not in owned:
                    raise MCPError("plugin_entrypoint_not_owned")
                root = Path(package.locate_file("")).resolve(strict=True)
                parts = path.split("/")
                for size in range(1, len(parts)):
                    initializer = "/".join(parts[:size]) + "/__init__.py"
                    if (root / initializer).exists() and initializer not in owned:
                        raise MCPError("plugin_entrypoint_not_owned")
                identity = (bound.distribution, bound.version, bound.sha256, entry.value)
                if entry.module in sys.modules and loaded.get(entry.module) != identity:
                    raise MCPError("plugin_already_imported")
                _verify_import_path(entry.module, root, owned)
                cls = entry.load()
                loaded[entry.module] = identity
                if not isinstance(cls, type) or not issubclass(cls, Plugin):
                    raise MCPError("invalid_plugin_binding")
                manager = PluginManager(
                    PluginManagerConfig(scan_entrypoints=False, auto_load=False)
                )
                stack.callback(manager.shutdown)
                manager.load_from_class(
                    cls, config=PluginConfig(settings=bound.settings), activate=True
                )
                for exposed, native in bound.capabilities.items():
                    capability = manager.get_capability(native)
                    if exposed in result or native not in manager.list_capabilities():
                        raise MCPError("invalid_plugin_binding")
                    result[exposed] = capability
            yield result
        finally:
            # ExitStack cleanup must still execute with the same cache policy.
            stack.close()
            sys.pycache_prefix = previous
