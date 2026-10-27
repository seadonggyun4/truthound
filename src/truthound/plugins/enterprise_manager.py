"""Enterprise plugin manager built on the unified 2.0 PluginManager.

This module preserves the enterprise-facing API while delegating the actual
plugin lifecycle to the core ``PluginManager``. Security, hot reload, and
version metadata are treated as optional capabilities instead of creating a
separate lifecycle runtime.
"""

from __future__ import annotations

import inspect
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from truthound.plugins.base import Plugin, PluginConfig, PluginLoadError
from truthound.plugins.manager import PluginManager, PluginManagerConfig
from truthound.plugins.security.protocols import SecurityPolicy, SignatureInfo
from truthound.plugins.security.signing.service import SigningServiceImpl
from truthound.plugins.security.signing.trust_store import TrustStoreImpl
from truthound.plugins.security.signing.verifier import create_verification_chain


@dataclass
class EnterprisePluginManagerConfig(PluginManagerConfig):
    default_security_policy: SecurityPolicy = field(default_factory=SecurityPolicy.standard)
    require_signature: bool = False
    trust_store_path: Path | None = None
    enable_hot_reload: bool = False
    watch_for_changes: bool = False
    allow_missing_optional: bool = True
    verification_keys: dict[str, bytes] = field(default_factory=dict, repr=False)
    verification_secrets: dict[str, bytes] = field(default_factory=dict, repr=False)
    security_metadata: dict[str, Any] = field(default_factory=dict)


class EnterprisePluginManager(PluginManager):
    """Capability-driven enterprise facade over the unified PluginManager."""

    def __init__(self, config: EnterprisePluginManagerConfig | None = None) -> None:
        super().__init__(config or EnterprisePluginManagerConfig())
        self._enterprise_config = self._config
        self._signed_plugins: dict[str, tuple[type[Plugin], Path, SignatureInfo]] = {}
        self._signature_service = SigningServiceImpl(
            verification_keys=self._config.verification_keys,
            verification_secrets=self._config.verification_secrets,
        )
        self.signature_trust_store = TrustStoreImpl(self._config.trust_store_path, auto_save=False)

        self.register_capability("versioning", {"host_version": self._config.host_version})
        self.register_capability(
            "security_policy",
            {"policy": self._config.default_security_policy},
        )
        if self._config.trust_store_path is not None:
            self.register_capability("trust_store", {"path": str(self._config.trust_store_path)})
        self._hot_reload = None
        self._reload_factories = {}
        if self._config.enable_hot_reload:
            from truthound.plugins.lifecycle.hot_reload import HotReloadManager

            self._hot_reload = HotReloadManager(self, plugin_loader=self._reload_class)
            self.register_capability("hot_reload", self._hot_reload)
        if self._config.require_signature:
            self.register_capability("signature_verification", {"required": True})

    def _reload_class(self, name):
        if name not in self._reload_factories:
            raise ValueError("No operator reload factory configured")
        return self._reload_factories[name](name)

    async def configure_hot_reload(self, name, loader, *, path=None):
        """Select an operator factory and optionally start configured watching."""
        if self._hot_reload is None:
            raise ValueError("Hot reload is disabled")
        if not callable(loader):
            raise TypeError("Reload factory must be callable")
        if self._config.watch_for_changes and path is None:
            raise ValueError("Automatic watching requires an explicit source path")
        if self._config.watch_for_changes:
            await self._hot_reload.watch(name, path)
        self._reload_factories[name] = loader

    async def reload(self, name, strategy=None):
        if self._hot_reload is None:
            raise ValueError("Hot reload is disabled")
        return await self._hot_reload.reload(name, strategy)

    def _unload_all_plugins(self):
        if self._hot_reload is not None and self._hot_reload._watches:
            raise RuntimeError("Await unload_all to join active plugin watchers")
        super()._unload_all_plugins()

    async def load(
        self,
        name: str,
        config: PluginConfig | None = None,
        *,
        activate: bool | None = None,
        verify_signature: bool = False,
    ) -> Plugin:
        with self._lock:
            if verify_signature:
                self._verify_plugin_signature(name, required=True)
            return self.load_plugin(name, config=config, activate=activate)

    def register_signed_plugin(
        self,
        name: str,
        plugin_class: type[Plugin],
        plugin_path: Path,
        signature: SignatureInfo,
    ) -> None:
        """Bind an already imported operator-selected class to signed source.

        Discovery/import is not sandboxed. This gate authenticates the selected
        source before constructor/setup/activation, not arbitrary Python imports.
        """
        path = Path(plugin_path).resolve(strict=True)
        source = inspect.getsourcefile(plugin_class)
        if source is None:
            raise ValueError("Signed plugins must have an inspectable source file")
        source_path = Path(source).resolve(strict=True)
        if source_path != path and not (path.is_dir() and source_path.is_relative_to(path)):
            raise ValueError("Plugin class is outside its signed source")
        with self._lock:
            if self.is_plugin_loaded(name):
                raise ValueError("Unload the plugin before replacing its signature binding")
            self._signed_plugins[name] = (plugin_class, path, deepcopy(signature))
            self._discovered_classes[name] = plugin_class

    def _verify_plugin_signature(self, name: str, *, required: bool = False) -> None:
        binding = self._signed_plugins.get(name)
        if binding is None:
            if required or self._config.require_signature:
                raise PluginLoadError("Plugin requires a signed source binding", plugin_name=name)
            return
        plugin_class, path, signature = binding
        if self._discovered_classes.get(name) is not plugin_class:
            raise PluginLoadError("Signed plugin class binding changed", plugin_name=name)
        result = create_verification_chain(
            self.signature_trust_store,
            self._signature_service,
            require_trusted_signer=True,
        ).verify(path, signature, {})
        if not result.is_valid:
            raise PluginLoadError("Plugin signature verification failed", plugin_name=name)

    def load_plugin(
        self, name: str, config: PluginConfig | None = None, activate: bool | None = None
    ) -> Plugin:
        with self._lock:
            self._verify_plugin_signature(name)
            return super().load_plugin(name, config, activate)

    def activate_plugin(self, name: str) -> None:
        with self._lock:
            self._verify_plugin_signature(name)
            super().activate_plugin(name)

    def load_from_class(
        self,
        plugin_cls: type[Plugin],
        config: PluginConfig | None = None,
        activate: bool | None = None,
    ) -> Plugin:
        with self._lock:
            for name, (bound, _, _) in self._signed_plugins.items():
                if bound is plugin_cls:
                    return self.load_plugin(name, config, activate)
            if self._config.require_signature:
                raise PluginLoadError("Plugin requires a signed source binding")
            return super().load_from_class(plugin_cls, config, activate)

    async def load_all(self, activate: bool | None = None) -> list[Plugin]:
        return self._load_all_plugins(activate)

    async def unload(self, name: str) -> None:
        self.unload_plugin(name)

    async def unload_all(self) -> None:
        errors = []
        if self._hot_reload is not None:
            try:
                await self._hot_reload.aclose()
            except Exception as error:
                errors.append(error)
        try:
            super().unload_all()
        except Exception as error:
            errors.append(error)
        if errors:
            raise ExceptionGroup("Enterprise plugin cleanup failed", errors)

    async def activate(self, name: str) -> None:
        self.activate_plugin(name)

    async def deactivate(self, name: str) -> None:
        self.deactivate_plugin(name)

    async def __aenter__(self) -> EnterprisePluginManager:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.unload_all()


def create_enterprise_manager(
    config: EnterprisePluginManagerConfig | None = None,
    **kwargs: Any,
) -> EnterprisePluginManager:
    if config is None:
        config = EnterprisePluginManagerConfig(**kwargs)
    return EnterprisePluginManager(config)
