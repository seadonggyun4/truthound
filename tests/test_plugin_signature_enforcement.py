"""Native signature rejection must depend on authenticated bytes."""

from dataclasses import replace

import pytest

from truthound.plugins.security.signing.service import SignatureAlgorithm, SigningServiceImpl
from truthound.plugins.security.signing.verifier import create_verification_chain


def test_forged_signature_is_rejected_by_service_and_chain(tmp_path):
    path = tmp_path / "plugin.py"
    path.write_text("value = 1\n")
    service = SigningServiceImpl()
    signed = service.sign(path, b"")
    forged = replace(signed, signature=b"forged")
    assert not service.verify(path, forged).is_valid
    assert not create_verification_chain().verify(path, forged, {}).is_valid


def test_required_trust_is_not_just_a_warning(tmp_path):
    path = tmp_path / "plugin.py"
    path.write_text("value = 1\n")
    signed = SigningServiceImpl().sign(path, b"")
    assert (
        not create_verification_chain(require_trusted_signer=True).verify(path, signed, {}).is_valid
    )


def test_signing_metadata_cannot_replace_content_digest(tmp_path):
    path = tmp_path / "plugin.py"
    path.write_text("value = 1\n")
    service = SigningServiceImpl()
    with pytest.raises(ValueError):
        service.sign(path, b"", metadata={"plugin_hash": "forged"})


@pytest.mark.parametrize("algorithm", list(SignatureAlgorithm))
def test_native_algorithm_checks_signature_and_signed_identity(tmp_path, algorithm):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519, rsa

    from truthound.plugins.security.protocols import TrustLevel
    from truthound.plugins.security.signing.trust_store import TrustStoreImpl

    path = tmp_path / "plugin.py"
    path.write_text("value = 1\n")
    secret = b"test-only-signing-secret"
    keys = {}
    secrets = {}
    if algorithm in {SignatureAlgorithm.RSA_SHA256, SignatureAlgorithm.ED25519}:
        key = (
            rsa.generate_private_key(public_exponent=65537, key_size=2048)
            if algorithm == SignatureAlgorithm.RSA_SHA256
            else ed25519.Ed25519PrivateKey.generate()
        )
        secret = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        keys["operator"] = key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    elif algorithm in {SignatureAlgorithm.HMAC_SHA256, SignatureAlgorithm.HMAC_SHA512}:
        secrets["operator"] = secret
    service = SigningServiceImpl(
        algorithm, signer_id="operator", verification_keys=keys, verification_secrets=secrets
    )
    signed = service.sign(path, secret)
    assert service.verify(path, signed).is_valid
    assert not service.verify(path, replace(signed, signature=b"invalid")).is_valid
    assert not service.verify(path, replace(signed, signer_id="other")).is_valid
    store = TrustStoreImpl(auto_save=False)
    store.set_signer_trust("operator", TrustLevel.TRUSTED)
    chain = create_verification_chain(store, service, require_trusted_signer=True)
    authenticated = algorithm not in {SignatureAlgorithm.SHA256, SignatureAlgorithm.SHA512}
    assert chain.verify(path, signed, {}).is_valid == authenticated
    if authenticated:
        assert not SigningServiceImpl().verify(path, signed).is_valid
    store.set_signer_trust("operator", TrustLevel.REVOKED)
    assert not chain.verify(path, signed, {}).is_valid


@pytest.mark.asyncio
async def test_enterprise_signature_gate_covers_lifecycle_and_changed_source(tmp_path):
    import importlib.util
    import sys

    from truthound.plugins.base import PluginLoadError
    from truthound.plugins.enterprise_manager import (
        EnterprisePluginManager,
        EnterprisePluginManagerConfig,
    )
    from truthound.plugins.security.protocols import TrustLevel

    path = tmp_path / "signed_plugin.py"
    path.write_text("""from truthound.plugins.base import Plugin, PluginInfo, PluginType
calls = []
class SignedPlugin(Plugin):
    @property
    def info(self):
        return PluginInfo(name="signed", version="1.0.0", plugin_type=PluginType.CUSTOM)
    def setup(self):
        calls.append("setup")
    def teardown(self):
        calls.append("teardown")
    def register(self, manager):
        calls.append("register")
""")
    spec = importlib.util.spec_from_file_location("truthound_test_signed_plugin", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    manager = EnterprisePluginManager(
        EnterprisePluginManagerConfig(
            require_signature=True,
            auto_activate=False,
            verification_secrets={"operator": b"test-secret"},
        )
    )
    try:
        with pytest.raises(PluginLoadError):
            manager.load_from_class(module.SignedPlugin)
        manager._discovered_classes["signed"] = module.SignedPlugin
        with pytest.raises(PluginLoadError):
            await manager.load("signed")
        with pytest.raises(ExceptionGroup):
            await manager.load_all()
        assert module.calls == []
        signed = SigningServiceImpl(SignatureAlgorithm.HMAC_SHA256, "operator").sign(
            path, b"test-secret"
        )
        manager.register_signed_plugin("signed", module.SignedPlugin, path, signed)
        with pytest.raises(PluginLoadError):
            manager.load_plugin("signed")
        manager.signature_trust_store.set_signer_trust("operator", TrustLevel.TRUSTED)
        await manager.load("signed", verify_signature=True)
        assert module.calls == ["setup"]
        path.write_text(path.read_text() + "\nchanged = True\n")
        with pytest.raises(PluginLoadError):
            await manager.activate("signed")
        assert module.calls == ["setup"]
        await manager.unload("signed")
        assert module.calls == ["setup", "teardown"]
    finally:
        await manager.unload_all()
        sys.modules.pop(spec.name, None)


def test_trust_store_reopens_certificates_and_revocations(tmp_path):
    from truthound.plugins.security.protocols import TrustLevel
    from truthound.plugins.security.signing.trust_store import TrustStoreImpl

    path = tmp_path / "trust.json"
    store = TrustStoreImpl(path)
    cert_id = store.add_trusted_certificate(b"operator-certificate")
    reopened = TrustStoreImpl(path)
    assert reopened.is_trusted(b"operator-certificate") == (True, TrustLevel.TRUSTED)
    reopened.revoke_certificate(cert_id, "test revocation")
    assert TrustStoreImpl(path).is_trusted(b"operator-certificate") == (False, TrustLevel.REVOKED)


def test_corrupt_trust_store_fails_without_partial_trust(tmp_path):
    from truthound.plugins.security.signing.trust_store import TrustStoreImpl

    path = tmp_path / "trust.json"
    path.write_text('{"signer_trust": {"first": "trusted", "broken": "invalid"}}')
    with pytest.raises(ValueError):
        TrustStoreImpl(path)


def test_trust_publication_failure_preserves_memory_and_disk(tmp_path, monkeypatch):
    from truthound.plugins.security.protocols import TrustLevel
    from truthound.plugins.security.signing import trust_store

    path = tmp_path / "trust.json"
    store = trust_store.TrustStoreImpl(path)
    store.set_signer_trust("operator", TrustLevel.TRUSTED)
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("publication failed")

    monkeypatch.setattr(trust_store, "publish", fail)
    with pytest.raises(OSError):
        store.set_signer_trust("operator", TrustLevel.REVOKED)
    assert store.get_trust_level("operator") == TrustLevel.TRUSTED
    assert path.read_bytes() == before


def test_enterprise_sync_context_runs_cleanup():
    from truthound.plugins.base import Plugin, PluginInfo, PluginType
    from truthound.plugins.enterprise_manager import EnterprisePluginManager

    calls = []

    class Sample(Plugin):
        @property
        def info(self):
            return PluginInfo(name="sample", version="1.0.0", plugin_type=PluginType.CUSTOM)

        def setup(self):
            calls.append("setup")

        def teardown(self):
            calls.append("teardown")

    with EnterprisePluginManager() as manager:
        manager.load_from_class(Sample, activate=False)
    assert calls == ["setup", "teardown"]
    assert not manager.is_plugin_loaded("sample")


def test_bulk_cleanup_reports_failures_after_attempting_every_plugin():
    from truthound.plugins.base import Plugin, PluginInfo, PluginType
    from truthound.plugins.manager import PluginManager

    calls = []

    class First(Plugin):
        @property
        def info(self):
            return PluginInfo(
                name=type(self).__name__.lower(), version="1.0.0", plugin_type=PluginType.CUSTOM
            )

        def setup(self):
            pass

        def teardown(self):
            calls.append(type(self).__name__)
            raise OSError("cleanup failed")

    class Second(First):
        pass

    manager = PluginManager()
    manager.load_from_class(First, activate=False)
    manager.load_from_class(Second, activate=False)
    with pytest.raises(ExceptionGroup) as failure:
        manager.unload_all()
    assert set(calls) == {"First", "Second"}
    assert len(failure.value.exceptions) == 2
