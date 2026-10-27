"""Provider contracts use real SDK JSON/media requests on an offline transport."""

from urllib.parse import parse_qs, urlsplit

import pytest

from tests.native_cloud_transports import GoogleObjectHTTP, google_client
from tests.test_store_filesystem_contract import record
from truthound.stores.backends import gcs
from truthound.stores.base import StoreQuery, StoreReadError, StoreWriteError

pytestmark = pytest.mark.contract


@pytest.fixture
def google_objects(monkeypatch):
    transport = GoogleObjectHTTP()
    client = google_client(transport)
    monkeypatch.setattr(gcs.storage, "Client", lambda **kwargs: client)
    yield transport
    client.close()


def test_gcs_native_gzip_roundtrip_uses_stored_bytes_without_double_decoding(google_objects):
    store = gcs.GCSStore("synthetic")
    store.save(record())
    assert store.get("run").run_id == "run"
    store.close()
    assert google_objects.closed is True


def test_gcs_corrupt_or_denied_index_cannot_be_replaced_as_an_empty_store(google_objects):
    key = "truthound/default/_index.json"
    google_objects.objects[key] = (
        b"synthetic corrupted index",
        GoogleObjectHTTP.metadata(key, b"synthetic corrupted index", 1),
    )
    before = dict(google_objects.objects)
    with pytest.raises(StoreReadError):
        gcs.GCSStore("synthetic").list_ids()
    assert google_objects.objects == before


def test_gcs_native_publication_carries_generation_preconditions(google_objects):
    store = gcs.GCSStore("synthetic", compression=False)
    store.save(record())
    writes = [
        parse_qs(urlsplit(url).query)
        for method, url, _ in google_objects.requests
        if method == "POST"
    ]
    assert len(writes) == 2
    assert all(request.get("ifGenerationMatch") == ["0"] for request in writes)


@pytest.mark.parametrize("codec,format", [(False, "json"), (True, "yaml"), ("zstd", "json")])
def test_gcs_native_format_configuration_survives_reopen_and_explicit_rebuild(
    google_objects, codec, format
):
    store = gcs.GCSStore(
        "synthetic",
        compression=codec,
        serialization_format=format,
        metadata={"configured": "default"},
    )
    store.save(record())
    index = store.config.get_full_prefix() + "_index.json"
    google_objects.seed(index, b"synthetic corrupted index")
    fresh = gcs.GCSStore(config=store.config)
    with pytest.raises(StoreReadError):
        fresh.list_ids()
    assert fresh.rebuild_index() == 1
    assert fresh.get("run").metadata == {"configured": "default"}
    assert fresh.list_ids(StoreQuery(limit=0)) == []
    assert fresh.delete("run") is True
    assert fresh.exists("run") is False


def test_gcs_stale_owner_and_denied_index_do_not_publish_records(google_objects):
    first, stale = gcs.GCSStore("synthetic"), gcs.GCSStore("synthetic")
    first.initialize()
    stale.initialize()
    first.save(record())
    before = dict(google_objects.objects)
    with pytest.raises(StoreWriteError):
        stale.save(record("other"))
    assert google_objects.objects == before
    google_objects.denied.add("truthound/default/_index.json")
    with pytest.raises(StoreReadError):
        gcs.GCSStore("synthetic").list_ids()
    assert google_objects.objects == before


def test_gcs_record_generation_is_pinned_between_metadata_and_media_requests(google_objects):
    store = gcs.GCSStore("synthetic", compression=False)
    store.save(record())
    google_objects.before_media = lambda key: google_objects.seed(key, b"synthetic replaced bytes")
    with pytest.raises(StoreReadError):
        store.get("run")
    after = dict(google_objects.objects)
    with pytest.raises(StoreWriteError):
        store.save(record("after-failed-read"))
    assert google_objects.objects == after


def test_missing_native_generation_receipt_is_not_committed_and_fences_retries(google_objects):
    store = gcs.GCSStore("synthetic", compression=False)
    store.initialize()
    google_objects.omit_upload_generation = True
    with pytest.raises(StoreWriteError):
        store.save(record())
    assert "truthound/default/_index.json" not in google_objects.objects
    before = dict(google_objects.objects)
    with pytest.raises(StoreWriteError):
        store.save(record("other"))
    assert google_objects.objects == before
    google_objects.omit_upload_generation = False
    fresh = gcs.GCSStore("synthetic", compression=False)
    with pytest.raises(StoreReadError):
        fresh.list_ids()
    assert fresh.rebuild_index() == 1
    assert fresh.get("run").run_id == "run"


@pytest.fixture
def azure_objects(monkeypatch):
    from tests.native_cloud_transports import AzureObjectHTTP
    from truthound.stores.backends import azure_blob

    transport = AzureObjectHTTP()
    native_class = azure_blob.BlobServiceClient
    clients = []

    def create(**kwargs):
        client = native_class(transport=transport, retry_total=0, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(azure_blob, "BlobServiceClient", create)
    yield transport
    for client in clients:
        client.close()


def azure_store(**kwargs):
    from truthound.stores.backends.azure_blob import AzureBlobStore

    return AzureBlobStore(
        "synthetic",
        account_url="https://synthetic.blob.core.windows.net",
        account_key="c3ludGhldGljLWF6dXJlLWtleS1vbmx5",
        **kwargs,
    )


@pytest.mark.parametrize("compression", [True, False])
def test_native_azure_pipeline_roundtrip_closes_actual_client(azure_objects, compression):
    store = azure_store(compression=compression)
    store.save(record())
    assert store.get("run").run_id == "run"
    store.close()
    assert azure_objects.closed is True


def test_native_azure_writes_include_conditional_etag_headers(azure_objects):
    store = azure_store(compression=False)
    store.save(record())
    writes = [headers for method, _, headers in azure_objects.requests if method == "PUT"]
    assert len(writes) == 2
    assert all(headers.get("If-None-Match") == "*" for headers in writes)


def test_native_azure_metadata_is_applied_and_corrupt_index_is_a_typed_error(azure_objects):
    store = azure_store(compression=False, metadata={"owner": "configured"})
    store.save(record())
    key = "truthound/default/results/run.json"
    assert azure_objects.objects[key]["metadata"].get("owner") == "configured"
    azure_objects.objects["truthound/default/_index.json"]["data"] = b"corrupt synthetic index"
    with pytest.raises(StoreReadError):
        azure_store(compression=False).list_ids()


def test_native_azure_corrupt_index_failure_remains_a_read_error(azure_objects):
    store = azure_store(compression=False)
    store.save(record())
    azure_objects.objects["truthound/default/_index.json"]["data"] = b"corrupt synthetic index"
    with pytest.raises(StoreReadError):
        azure_store(compression=False).list_ids()


@pytest.mark.parametrize("codec,format", [(False, "json"), (True, "yaml"), ("zstd", "json")])
def test_native_azure_formats_reopen_rebuild_and_conditional_delete(azure_objects, codec, format):
    store = azure_store(
        compression=codec, serialization_format=format, record_metadata={"default": "record"}
    )
    store.save(record())
    azure_objects.objects["truthound/default/_index.json"]["data"] = b"corrupt synthetic index"
    fresh = azure_store(compression=codec, serialization_format=format)
    with pytest.raises(StoreReadError):
        fresh.list_ids()
    assert fresh.rebuild_index() == 1
    assert fresh.get("run").metadata == {"default": "record"}
    assert fresh.delete("run") is True
    assert fresh.exists("run") is False


def test_native_azure_stale_owner_and_metadata_media_replacement_cannot_commit(azure_objects):
    first, stale = azure_store(compression=False), azure_store(compression=False)
    first.initialize()
    stale.initialize()
    first.save(record())
    before = dict(azure_objects.objects)
    with pytest.raises(StoreWriteError):
        stale.save(record("other"))
    assert azure_objects.objects == before

    def replaced(key):
        azure_objects.objects[key]["etag"] = '"replacement"'
        azure_objects.objects[key]["data"] = b"synthetic replacement"

    azure_objects.before_download = replaced
    with pytest.raises(StoreReadError):
        first.get("run")
    after = dict(azure_objects.objects)
    with pytest.raises(StoreWriteError):
        first.save(record("after-failed-read"))
    assert azure_objects.objects == after


def test_native_azure_receipt_loss_fences_retries_and_requires_inventory_rebuild(azure_objects):
    store = azure_store(compression=False)
    store.initialize()
    azure_objects.omit_upload_etag = True
    with pytest.raises(StoreWriteError):
        store.save(record())
    assert "truthound/default/_index.json" not in azure_objects.objects
    before = dict(azure_objects.objects)
    with pytest.raises(StoreWriteError):
        store.save(record("other"))
    assert azure_objects.objects == before
    azure_objects.omit_upload_etag = False
    fresh = azure_store(compression=False)
    with pytest.raises(StoreReadError):
        fresh.list_ids()
    assert fresh.rebuild_index() == 1
    assert fresh.get("run").run_id == "run"


def test_native_azure_configured_tier_and_explicit_tier_failure_are_observable(azure_objects):
    store = azure_store(access_tier="Cool")
    store.save(record())
    assert store.get_access_tier("run")["tier"] == "Cool"
    assert store.set_access_tier("run", "Hot") is True
    assert store.get_access_tier("run")["tier"] == "Hot"
    azure_objects.reject_tier = True
    with pytest.raises(StoreWriteError):
        store.set_access_tier("run", "Cool")
    before = dict(azure_objects.objects)
    with pytest.raises(StoreWriteError):
        store.save(record("other"))
    assert azure_objects.objects == before
    fresh = azure_store(access_tier="Cool")
    with pytest.raises(StoreWriteError):
        fresh.save(record("new-configured-tier"))
    assert "truthound/default/results/new-configured-tier.json.gz" not in azure_objects.objects


def test_native_azure_failed_client_cleanup_fences_work_and_preserves_retry_handle(
    azure_objects, monkeypatch
):
    from truthound.stores.base import StoreConnectionError

    store = azure_store()
    store.save(record())
    client = store._client
    original = azure_objects.close

    def failed():
        original()
        raise OSError("synthetic native close lost receipt")

    monkeypatch.setattr(azure_objects, "close", failed)
    with pytest.raises(BaseExceptionGroup):
        store.close()
    assert store._client is client
    before = len(azure_objects.requests)
    with pytest.raises(StoreConnectionError):
        store.list_ids()
    assert len(azure_objects.requests) == before
    monkeypatch.setattr(azure_objects, "close", original)
    store.close()
    assert store._client is None


def test_native_azure_owned_default_credential_signs_and_retries_failed_cleanup(
    azure_objects, monkeypatch
):
    import time

    import azure.identity
    from azure.core.credentials import AccessToken

    from truthound.stores.backends.azure_blob import AzureBlobStore
    from truthound.stores.base import StoreConnectionError

    class SyntheticCredential:
        def __init__(self):
            self.scopes = []
            self.closed = 0
            self.fail_close = True

        def get_token(self, *scopes, **kwargs):
            self.scopes.append(scopes)
            return AccessToken("synthetic-offline-token", int(time.time()) + 3600)

        def close(self):
            self.closed += 1
            if self.fail_close:
                raise OSError("synthetic credential cleanup failure")

    credential = SyntheticCredential()
    monkeypatch.setattr(azure.identity, "DefaultAzureCredential", lambda: credential)
    store = AzureBlobStore("synthetic", account_url="https://synthetic.blob.core.windows.net")
    store.save(record())
    assert credential.scopes == [("https://storage.azure.com/.default",)]
    assert any(
        headers.get("Authorization") == "Bearer synthetic-offline-token"
        for _, _, headers in azure_objects.requests
    )
    with pytest.raises(BaseExceptionGroup):
        store.close()
    assert store._credential is credential
    assert store._client is None
    before = len(azure_objects.requests)
    with pytest.raises(StoreConnectionError):
        store.get("run")
    assert len(azure_objects.requests) == before
    credential.fail_close = False
    store.close()
    assert credential.closed == 2
    assert store._credential is None


def test_native_azure_explicit_anonymous_never_constructs_default_credential(
    azure_objects, monkeypatch
):
    import azure.identity

    from truthound.stores.backends.azure_blob import AzureBlobStore

    def forbidden():
        raise AssertionError("Anonymous configuration must bypass credential discovery")

    monkeypatch.setattr(azure.identity, "DefaultAzureCredential", forbidden)
    store = AzureBlobStore(
        "synthetic", account_url="https://synthetic.blob.core.windows.net", anonymous=True
    )
    store.save(record())
    assert all("Authorization" not in headers for _, _, headers in azure_objects.requests)
    store.close()


def test_native_gcs_failed_close_preserves_handle_and_blocks_new_io(google_objects, monkeypatch):
    from truthound.stores.base import StoreConnectionError

    store = gcs.GCSStore("synthetic")
    store.save(record())
    client = store._client
    original = client.close

    def fail():
        raise OSError("synthetic close uncertainty")

    monkeypatch.setattr(client, "close", fail)
    with pytest.raises(OSError):
        store.close()
    assert store._client is client
    before = len(google_objects.requests)
    with pytest.raises(StoreConnectionError):
        store.get("run")
    assert len(google_objects.requests) == before
    monkeypatch.setattr(client, "close", original)
    store.close()
    assert store._client is None
