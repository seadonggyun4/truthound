from __future__ import annotations

import secrets

import pytest


@pytest.mark.contract
def test_shadowed_secrets_package_reexports_stdlib_api():
    assert hasattr(secrets, "randbits")
    assert callable(secrets.randbits)
