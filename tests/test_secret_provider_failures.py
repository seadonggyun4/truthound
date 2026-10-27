"""Native secret lookup errors must preserve absence semantics and redact logs."""

import logging

import pytest

from truthound.secrets.base import BaseSecretProvider
from truthound.secrets.manager import SecretManager, SecretManagerConfig

pytestmark = pytest.mark.fault


def test_non_strict_missing_provider_returns_none_without_wrapping_a_null_secret():
    manager = SecretManager(SecretManagerConfig(strict_mode=False))
    assert manager.get("absent", provider="missing") is None


def test_provider_failure_does_not_log_native_exception_material(caplog):
    class FailedProvider(BaseSecretProvider):
        @property
        def name(self):
            return "local"

        def _fetch(self, key, version=None, field=None):
            raise RuntimeError("synthetic-sensitive-provider-message")

    manager = SecretManager(SecretManagerConfig(strict_mode=False))
    manager.add_provider(FailedProvider())
    with caplog.at_level(logging.WARNING):
        assert manager.get("operator-reference") is None
    assert caplog.records
    assert "synthetic-sensitive-provider-message" not in caplog.text
