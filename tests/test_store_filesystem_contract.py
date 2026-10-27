"""Ordinary filesystem stores preserve records and fail visibly on corruption."""

import json
from datetime import datetime

import pytest

from truthound.stores.backends.filesystem import FileSystemExpectationStore, FileSystemStore
from truthound.stores.base import StoreQuery, StoreReadError, StoreWriteError
from truthound.stores.expectations import ExpectationSuite
from truthound.stores.results import ResultStatus, ValidationResult

pytestmark = pytest.mark.contract


def record(identity="run", asset="asset"):
    return ValidationResult(
        run_id=identity,
        run_time=datetime(2026, 1, 1),
        data_asset=asset,
        status=ResultStatus.SUCCESS,
    )


def test_corrupt_index_is_not_empty_and_explicit_rebuild_preserves_original_records(tmp_path):
    original = FileSystemStore(base_path=str(tmp_path))
    original.save(record())
    index = original.config.get_full_path() / "_index.json"
    index.write_bytes(b"corrupt synthetic index")
    restored = FileSystemStore(base_path=str(tmp_path))
    with pytest.raises(StoreReadError):
        restored.list_ids()
    assert index.read_bytes() == b"corrupt synthetic index"
    assert restored.rebuild_index() == 1
    assert restored.get("run").run_id == "run"
    assert json.loads(index.read_bytes())["run"]["data_asset"] == "asset"


def test_record_replacement_failure_keeps_previous_bytes(tmp_path, monkeypatch):
    import os

    store = FileSystemStore(base_path=str(tmp_path))
    store.save(record())
    path = store._get_file_path("run")
    old = path.read_bytes()
    native = os.replace

    def rejected(source, destination):
        if str(destination) == str(path):
            raise OSError("synthetic publish failed")
        return native(source, destination)

    monkeypatch.setattr(os, "replace", rejected)
    with pytest.raises(StoreWriteError):
        store.save(record(asset="replacement"))
    assert path.read_bytes() == old
    assert store.get("run").data_asset == "asset"


def test_unconfirmed_index_publish_fences_mutations_until_explicit_rebuild(tmp_path, monkeypatch):
    store = FileSystemStore(base_path=str(tmp_path))
    store.initialize()
    publish_index = store._save_index

    def failed_index():
        raise OSError("synthetic index publish failed")

    monkeypatch.setattr(store, "_save_index", failed_index)
    with pytest.raises(StoreWriteError):
        store.save(record())
    monkeypatch.setattr(store, "_save_index", publish_index)
    with pytest.raises(StoreWriteError, match="reconcil"):
        store.save(record("second"))
    fresh = FileSystemStore(base_path=str(tmp_path))
    with pytest.raises(StoreReadError, match="reconcil"):
        fresh.list_ids()
    assert fresh.rebuild_index() == 1
    assert fresh.list_ids() == ["run"]


def test_query_and_rebuild_do_not_suppress_corrupt_records(tmp_path):
    store = FileSystemStore(base_path=str(tmp_path))
    store.save(record())
    path = store._get_file_path("run")
    index = store.config.get_full_path() / "_index.json"
    before = index.read_bytes()
    path.write_bytes(b"invalid synthetic record")
    with pytest.raises(StoreReadError):
        store.query(StoreQuery())
    with pytest.raises(StoreReadError):
        store.rebuild_index()
    assert index.read_bytes() == before
    with pytest.raises(StoreWriteError, match="reconcil"):
        store.save(record("new"))


def test_query_honors_selected_order_zero_limit_and_offset(tmp_path):
    store = FileSystemStore(base_path=str(tmp_path))
    store.save(record("a", "z"))
    store.save(record("b", "a"))
    assert store.query(StoreQuery(limit=0)) == []
    assert store.list_ids(StoreQuery(order_by="data_asset", ascending=True, offset=1, limit=1)) == [
        "a"
    ]


@pytest.mark.parametrize("identity", ["../escape", "_index", "nested/record", "back\\record"])
def test_invalid_record_identity_cannot_escape_or_overwrite_index(tmp_path, identity):
    store = FileSystemStore(base_path=str(tmp_path))
    with pytest.raises((ValueError, StoreWriteError)):
        store.save(record(identity))


def test_selected_yaml_gzip_and_default_metadata_are_actually_serialized(tmp_path):
    import gzip

    import yaml

    store = FileSystemStore(
        base_path=str(tmp_path),
        serialization_format="yaml",
        file_extension=".yaml",
        compression=True,
        metadata={"default": "selected"},
    )
    store.save(record())
    content = gzip.decompress(store._get_file_path("run").read_bytes())
    assert not content.lstrip().startswith(b"{")
    assert yaml.safe_load(content)["metadata"] == {"default": "selected"}
    assert store.get("run").metadata == {"default": "selected"}


def test_selected_zstd_is_not_treated_as_boolean_gzip(tmp_path):
    from truthound.stores.compression.providers import get_compressor

    store = FileSystemStore(base_path=str(tmp_path), compression="zstd")
    store.save(record())
    raw = get_compressor("zstd").decompress(store._get_file_path("run").read_bytes())
    assert json.loads(raw)["run_id"] == "run"


def test_expectation_names_are_distinct_and_compression_config_is_applied(tmp_path):
    import gzip

    store = FileSystemExpectationStore(base_path=str(tmp_path), use_compression=True)
    store.save(ExpectationSuite.create("a/b", "first"))
    store.save(ExpectationSuite.create("a_b", "second"))
    assert store.get("a/b").data_asset == "first"
    assert store.get("a_b").data_asset == "second"
    assert store.list_ids() == ["a/b", "a_b"]
    assert json.loads(gzip.decompress(store._get_file_path("a_b").read_bytes()))["name"] == "a_b"


def test_encoded_suite_names_include_reserved_looking_names_and_query_pagination(tmp_path):
    store = FileSystemExpectationStore(base_path=str(tmp_path))
    for name in ["_index", ".stream-example", "normal"]:
        store.save(ExpectationSuite.create(name, "asset"))
    assert store.list_ids() == [".stream-example", "_index", "normal"]
    assert store.list_ids(StoreQuery(offset=1, limit=1)) == [".stream-example"]
    assert store.list_ids(StoreQuery(order_by="name", ascending=True, offset=1, limit=1)) == [
        "_index"
    ]
    assert store.query(StoreQuery(limit=0)) == []


def test_explicit_native_config_is_captured_and_used_by_both_store_types(tmp_path):
    from truthound.stores.backends.filesystem import FileSystemConfig

    config = FileSystemConfig(
        base_path=str(tmp_path),
        namespace="selected",
        prefix="results",
        use_compression=True,
        metadata={"configured": "value"},
    )
    store = FileSystemStore(config=config)
    config.namespace = "unrelated"
    store.save(record())
    assert (tmp_path / "selected" / "results" / "run.json.gz").exists()
    assert store.get("run").metadata["configured"] == "value"
    suites = FileSystemExpectationStore(
        config=FileSystemConfig(base_path=str(tmp_path), prefix="suites", use_compression=True)
    )
    suites.save(ExpectationSuite.create("suite", "asset"))
    assert suites.get("suite").name == "suite"


def test_legacy_suite_update_and_delete_use_the_verified_existing_identity(tmp_path):
    store = FileSystemExpectationStore(base_path=str(tmp_path))
    store.save(ExpectationSuite.create("a/b", "first"))
    encoded = store._get_file_path("a/b")
    legacy = store._get_file_path("a_b")
    encoded.rename(legacy)
    assert store.get("a/b").data_asset == "first"
    store.save(ExpectationSuite.create("a/b", "updated"))
    assert store.list_ids() == ["a/b"]
    assert store.get("a/b").data_asset == "updated"
    with pytest.raises(StoreWriteError):
        store.save(ExpectationSuite.create("a_b", "collision"))
    assert store.delete("a/b") is True
    assert store.list_ids() == []


def test_expectation_query_orders_before_pagination_and_rejects_unknown_order(tmp_path):
    store = FileSystemExpectationStore(base_path=str(tmp_path))
    store.save(ExpectationSuite.create("a", "z"))
    store.save(ExpectationSuite.create("b", "a"))
    assert store.list_ids(StoreQuery(order_by="data_asset", ascending=True, limit=1)) == ["b"]
    assert store.list_ids(StoreQuery(order_by="data_asset", ascending=False, offset=1)) == ["b"]
    with pytest.raises(ValueError):
        store.list_ids(StoreQuery(order_by="unknown"))


def test_duplicate_record_json_keys_cannot_be_reconciled_as_valid_data(tmp_path):
    store = FileSystemStore(base_path=str(tmp_path))
    store.save(record())
    path = store._get_file_path("run")
    payload = path.read_text()
    path.write_text(
        payload.replace('"data_asset": "asset"', '"data_asset": "other", "data_asset": "asset"')
    )
    with pytest.raises(StoreReadError):
        store.rebuild_index()
    assert '"data_asset": "other"' in path.read_text()


def test_yaml_rejects_duplicate_keys_and_nonfinite_metadata_before_publication(tmp_path):
    store = FileSystemStore(base_path=str(tmp_path), serialization_format="yaml")
    invalid = record()
    invalid.metadata["value"] = float("nan")
    with pytest.raises(StoreWriteError):
        store.save(invalid)
    assert not store._get_file_path("run").exists()
    store.save(record())
    path = store._get_file_path("run")
    payload = path.read_text()
    path.write_text(payload.replace("data_asset: asset", "data_asset: other\ndata_asset: asset"))
    with pytest.raises(StoreReadError):
        store.rebuild_index()
