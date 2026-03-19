"""Enterprise plugin manager built on the unified 2.0 PluginManager.

This module preserves the enterprise-facing API while delegating the actual
plugin lifecycle to the core ``PluginManager``. Security, hot reload, and
version metadata are treated as optional capabilities instead of creating a
separate lifecycle runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from truthound.plugins.base import Plugin, PluginConfig
from truthound.plugins.manager import PluginManager, PluginManagerConfig
from truthound.plugins.security.protocols import SecurityPolicy


@dataclass
class EnterprisePluginManagerConfig(PluginManagerConfig):
    default_security_policy: SecurityPolicy = field(
        default_factory=SecurityPolicy.standard
    )
    require_signature: bool = False
    trust_store_path: Path | None = None
    enable_hot_reload: bool = False
    watch_for_changes: bool = False
    allow_missing_optional: bool = True
    security_metadata: dict[str, Any] = field(default_factory=dict)


class EnterprisePluginManager(PluginManager):
    """Capability-driven enterprise facade over the unified PluginManager."""

    def __init__(self, config: EnterprisePluginManagerConfig | None = None) -> None:
        super().__init__(config or EnterprisePluginManagerConfig())
        self._enterprise_config = self._config
        self.register_capability('versioning', {'host_version': self._config.host_version})
        self.register_capability(
            'security_policy',
            {'policy': self._config.default_security_policy},
        )
        if self._config.trust_store_path is not None:
            self.register_capability('trust_store', {'path': str(self._config.trust_store_path)})
        if self._config.enable_hot_reload:
            self.register_capability('hot_reload', {'watch_for_changes': self._config.watch_for_changes})
        if self._config.require_signature:
            self.register_capability('signature_verification', {'required': True})

    async def load(
        self,
        name: str,
        config: PluginConfig | None = None,
        *,
        activate: bool | None = None,
        verify_signature: bool = False,
    ) -> Plugin:
        if verify_signature:
            self.register_capability('signature_verification', {'required': True})
        return self.load_plugin(name, config=config, activate=activate)

    async def load_all(self, activate: bool | None = None) -> list[Plugin]:
        return super().load_all(activate=activate)

    async def unload(self, name: str) -> None:
        self.unload_plugin(name)

    async def unload_all(self) -> None:
        super().unload_all()

    async def activate(self, name: str) -> None:
        self.activate_plugin(name)

    async def deactivate(self, name: str) -> None:
        self.deactivate_plugin(name)

    async def __aenter__(self) -> 'EnterprisePluginManager':
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
