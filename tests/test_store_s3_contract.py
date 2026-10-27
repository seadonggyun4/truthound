"""Native ordinary S3 workflows against an owned synthetic object protocol.

The botocore model is checked separately; these tests do not claim live cloud
availability and never resolve credentials or call a remote endpoint.
"""

import json
from io import BytesIO

import pytest
from botocore.exceptions import ClientError

from tests.synthetic_s3 import MemoryS3
from tests.test_store_filesystem_contract import record
from truthound.stores.backends import s3
from truthound.stores.base import StoreQuery, StoreReadError, StoreWriteError

pytestmark = pytest.mark.contract


class ObjectProtocol(MemoryS3):
    def __init__(self):
        super().__init__()
        self.bodies = []
        self.puts = []

    def get_object(self, **arguments):
        try:
            response = super().get_object(**arguments)
        except KeyError:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject") from None
        response["ContentLength"] = len(self.objects[arguments["Key"]])
        self.bodies.append(response["Body"])
        return response

    def head_object(self, **arguments):
        try:
            return super().head_object(**arguments)
        except KeyError:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject") from None

    def put_object(self, **arguments):
        self.puts.append(arguments)
        return super().put_object(**arguments)


@pytest.fixture
def objects(monkeypatch):
    client = ObjectProtocol()
    monkeypatch.setattr(s3.boto3, "client", lambda *args, **kwargs: client)
    return client


def store(**options):
    return s3.S3Store("synthetic", compression=False, **options)


def test_selected_metadata_tags_and_encoding_reach_the_native_object(objects):
    native = store(tags={"slash/name": "one/two & 한"}, metadata={"owner": "configured"})
    native.save(record())
    restored = native.get("run")
    assert restored.metadata == {"owner": "configured"}
    request = next(row for row in objects.puts if "/results/" in row["Key"])
    assert request["Tagging"] == "slash%2Fname=one%2Ftwo+%26+%ED%95%9C"
    assert all(body.closed for body in objects.bodies)


def test_stale_owner_cannot_overwrite_another_owners_index_or_publish_an_orphan(objects):
    first, stale = store(), store()
    first.initialize()
    stale.initialize()
    first.save(record("first"))
    before = dict(objects.objects)
    with pytest.raises(StoreWriteError):
        stale.save(record("second"))
    assert objects.objects == before
    assert store().list_ids() == ["first"]


def test_unconfirmed_index_write_fences_mutation_and_requires_explicit_rebuild(
    objects, monkeypatch
):
    native = store()
    native.initialize()
    original = objects.put_object

    def failed_index(**arguments):
        if arguments["Key"].endswith("/_index.json"):
            raise OSError("synthetic lost index receipt")
        return original(**arguments)

    monkeypatch.setattr(objects, "put_object", failed_index)
    with pytest.raises(StoreWriteError):
        native.save(record())
    monkeypatch.setattr(objects, "put_object", original)
    before = dict(objects.objects)
    with pytest.raises(StoreWriteError):
        native.save(record("second"))
    assert objects.objects == before
    fresh = store()
    with pytest.raises(StoreReadError):
        fresh.list_ids()
    assert fresh.rebuild_index() == 1
    assert fresh.get("run").run_id == "run"


def test_corrupt_index_and_record_identity_are_errors_not_empty_queries(objects):
    native = store()
    native.save(record())
    index = native.config.get_full_prefix() + "_index.json"
    objects.objects[index] = b"synthetic-corrupt-index"
    fresh = store()
    with pytest.raises(StoreReadError):
        fresh.query(StoreQuery())
    assert objects.objects[index] == b"synthetic-corrupt-index"
    assert fresh.rebuild_index() == 1
    objects.objects[fresh._get_key("run")] = json.dumps(record("different").to_dict()).encode()
    with pytest.raises(StoreReadError):
        fresh.get("run")


def test_native_query_sort_limit_zero_and_missing_record_fail_visibly(objects):
    native = store()
    native.save(record("z", "alpha"))
    native.save(record("a", "omega"))
    assert native.list_ids(StoreQuery(limit=0)) == []
    assert native.list_ids(StoreQuery(order_by="data_asset", ascending=True)) == ["z", "a"]
    del objects.objects[native._get_key("z")]
    assert native.exists("z") is False
    with pytest.raises(StoreReadError):
        native.query(StoreQuery())


def test_access_denied_is_not_absence_and_client_close_releases_owned_resources(
    objects, monkeypatch
):
    native = store()
    native.initialize()

    def denied(**kwargs):
        raise ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "synthetic-private-provider-detail"}},
            "HeadObject",
        )

    monkeypatch.setattr(objects, "head_object", denied)
    with pytest.raises(StoreReadError) as failed:
        native.exists("absent")
    assert "synthetic-private-provider-detail" not in str(failed.value)
    native.close()
    assert objects.closed is True


@pytest.mark.parametrize("identity", ["../escape", "a/b", "a\\b", "", "C:\\escape"])
def test_unsafe_record_ids_fail_before_native_publication(objects, identity):
    native = store()
    with pytest.raises(ValueError):
        native.save(record(identity))
    assert objects.objects == {}


def test_real_botocore_conditional_receipt_and_closed_response_contract(monkeypatch):
    import boto3
    from botocore.response import StreamingBody
    from botocore.stub import ANY, Stubber

    client = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="synthetic",
        aws_secret_access_key="synthetic",
    )
    monkeypatch.setattr(s3.boto3, "client", lambda *args, **kwargs: client)
    body = BytesIO(b"{}")
    with Stubber(client) as stub:
        stub.add_response("head_bucket", {}, {"Bucket": "synthetic"})
        stub.add_response(
            "get_object",
            {"Body": StreamingBody(body, 2), "ETag": '"initial"'},
            {"Bucket": "synthetic", "Key": "truthound/default/_index.json"},
        )
        stub.add_response(
            "list_objects_v2",
            {"Contents": []},
            {"Bucket": "synthetic", "Prefix": "truthound/default/results/"},
        )
        stub.add_response(
            "head_object",
            {"ETag": '"initial"'},
            {"Bucket": "synthetic", "Key": "truthound/default/_index.json", "IfMatch": '"initial"'},
        )
        stub.add_response(
            "put_object",
            {"ETag": '"record"'},
            {
                "Bucket": "synthetic",
                "Key": "truthound/default/results/run.json",
                "Body": ANY,
                "ContentType": "application/json",
                "StorageClass": "STANDARD",
                "IfNoneMatch": "*",
            },
        )
        stub.add_response(
            "put_object",
            {"ETag": '"next"'},
            {
                "Bucket": "synthetic",
                "Key": "truthound/default/_index.json",
                "Body": ANY,
                "ContentType": "application/json",
                "IfMatch": '"initial"',
            },
        )
        native = store()
        assert native.save(record()) == "run"
        assert body.closed is True
        stub.assert_no_pending_responses()
        native.close()


def test_lost_delete_receipt_preserves_index_and_fences_following_writes(objects, monkeypatch):
    native = store()
    native.save(record())
    original = objects.delete_object

    def lost_receipt(**arguments):
        original(**arguments)
        return {}

    monkeypatch.setattr(objects, "delete_object", lost_receipt)
    with pytest.raises(StoreWriteError):
        native.delete("run")
    before = dict(objects.objects)
    with pytest.raises(StoreWriteError):
        native.save(record("second"))
    assert objects.objects == before
    assert native.rebuild_index() == 0
    assert native.list_ids() == []


def test_captured_s3_config_and_native_write_guard_prevent_revoked_publication(objects):
    config = s3.S3Config(bucket="synthetic", use_compression=False, tags={"owner": "captured"})
    native = s3.S3Store(config=config)
    native.initialize()
    config.tags["owner"] = "changed"
    config.metadata["later"] = True
    native.save(record())
    assert native.get("run").metadata == {}
    assert (
        next(row for row in objects.puts if "/results/" in row["Key"])["Tagging"]
        == "owner=captured"
    )
    before = dict(objects.objects)

    def revoked():
        raise PermissionError("synthetic revoked authority")

    native.set_io_guard(revoked)
    with pytest.raises(StoreWriteError):
        native.save(record("second"))
    assert objects.objects == before


def test_s3_failed_close_preserves_handle_and_blocks_new_io(objects, monkeypatch):
    from truthound.stores.base import StoreConnectionError

    native = store()
    native.save(record())
    original = objects.close

    def fail():
        raise OSError("synthetic close uncertainty")

    monkeypatch.setattr(objects, "close", fail)
    with pytest.raises(OSError):
        native.close()
    assert native._client is objects
    before = len(objects.puts)
    with pytest.raises(StoreConnectionError):
        native.save(record("after-close-failure"))
    assert len(objects.puts) == before
    monkeypatch.setattr(objects, "close", original)
    native.close()
    assert native._client is None
