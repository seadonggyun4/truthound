"""Concurrent stores apply captured settings and reject corrupt native records."""

import gzip

import pytest

from tests.test_store_filesystem_contract import record
from truthound.stores.backends.concurrent_filesystem import (
    ConcurrencyConfig,
    ConcurrentFileSystemExpectationStore,
    ConcurrentFileSystemStore,
)
from truthound.stores.base import StoreQuery, StoreReadError
from truthound.stores.expectations import ExpectationSuite

pytestmark = pytest.mark.contract


def test_expectations_honor_selected_codec_and_lock_configuration(tmp_path):
    config = ConcurrencyConfig(lock_strategy="none", enable_wal=False)
    store = ConcurrentFileSystemExpectationStore(
        base_path=str(tmp_path),
        concurrency=config,
        use_compression=True,
        serialization_format="yaml",
        file_extension=".yaml",
        metadata={"origin": "store"},
    )
    config.lock_strategy = "invalid-after-construction"
    store.save(ExpectationSuite(name="suite", data_asset="asset", meta={"selected": True}))
    assert type(store._lock_manager._strategy).__name__ == "NoOpLockStrategy"
    path = store._get_file_path("suite")
    assert path.suffix == ".gz"
    assert b"name: suite" in gzip.decompress(path.read_bytes())
    restored = ConcurrentFileSystemExpectationStore(
        base_path=str(tmp_path),
        use_compression=True,
        serialization_format="yaml",
        file_extension=".yaml",
    )
    assert restored.get("suite").meta == {"origin": "store", "selected": True}
    store.close()
    restored.close()


def test_concurrent_query_zero_limit_and_corrupt_record_are_not_success(tmp_path):
    store = ConcurrentFileSystemStore(base_path=str(tmp_path))
    store.save(record())
    assert store.list_ids(StoreQuery(limit=0)) == []
    store._get_file_path("run").write_text('{"run_id":"run","run_id":"other"}')
    with pytest.raises(StoreReadError):
        store.query(StoreQuery())
    store.close()


def test_nonfinite_update_preserves_previous_concurrent_record(tmp_path):
    store = ConcurrentFileSystemStore(base_path=str(tmp_path))
    item = record()
    store.save(item)
    before = store._get_file_path("run").read_bytes()
    item.metadata["invalid"] = float("nan")
    with pytest.raises(ValueError):
        store.save(item)
    assert store._get_file_path("run").read_bytes() == before
    store.close()


def test_concurrent_paths_and_create_dirs_are_enforced(tmp_path):
    store = ConcurrentFileSystemStore(base_path=str(tmp_path / "missing"), create_dirs=False)
    with pytest.raises(StoreReadError):
        store.initialize()
    assert not (tmp_path / "missing").exists()
    with pytest.raises(ValueError):
        ConcurrentFileSystemStore(base_path=str(tmp_path), namespace="../escape")
    with pytest.raises(TypeError):
        ConcurrentFileSystemStore(base_path=str(tmp_path), unknown_setting=True)


def test_concurrent_rebuild_preserves_index_when_a_record_is_corrupt(tmp_path):
    store = ConcurrentFileSystemStore(base_path=str(tmp_path))
    store.save(record("_kept"))
    before = (store.config.get_full_path() / "_index.json").read_bytes()
    store._get_file_path("broken").write_text('{"run_id":"broken","run_id":"ambiguous"}')
    with pytest.raises(StoreReadError):
        store.rebuild_index()
    assert (store.config.get_full_path() / "_index.json").read_bytes() == before
    store._get_file_path("broken").unlink()
    assert store.rebuild_index() == 1
    assert store.get("_kept").id == "_kept"
    store.close()


def test_concurrent_index_rejects_duplicate_keys_without_rewriting(tmp_path):
    from truthound.stores.concurrency.index import ConcurrentIndex

    path = tmp_path / "_index.json"
    path.write_text('{"entries":{},"entries":{},"version":0}')
    before = path.read_bytes()
    with pytest.raises(ValueError):
        ConcurrentIndex(tmp_path).initialize()
    assert path.read_bytes() == before


def test_concurrent_suite_names_do_not_collide_and_pagination_is_applied_once(tmp_path):
    store = ConcurrentFileSystemExpectationStore(base_path=str(tmp_path))
    for name in ("a/b", "a_b", "a%b"):
        store.save(ExpectationSuite(name=name, data_asset="asset"))
    assert len(set(store._get_file_path(name) for name in ("a/b", "a_b", "a%b"))) == 3
    query = StoreQuery(order_by="name", ascending=True, offset=1, limit=1)
    assert store.list_ids(query) == ["a/b"]
    assert [suite.name for suite in store.query(query)] == ["a/b"]
    assert store.query(StoreQuery(limit=0)) == []
    changed = store.config
    changed.base_path = str(tmp_path / "wrong")
    assert store.get("a/b").name == "a/b"
    assert not (tmp_path / "wrong").exists()
    store.close()


def test_invalid_later_index_operation_cannot_publish_a_recoverable_prefix(tmp_path):
    from truthound.stores.concurrency.index import ConcurrentIndex

    index = ConcurrentIndex(tmp_path)
    index.add("existing", {})
    before = (tmp_path / "_index.json").read_bytes()
    txn = index.begin_transaction()
    txn.add("must-not-appear", {"value": 1})
    txn.add("invalid", {"value": float("nan")})
    with pytest.raises(ValueError):
        txn.commit()
    assert (tmp_path / "_index.json").read_bytes() == before
    assert not (tmp_path / "_index.json.wal").exists()
    assert ConcurrentIndex(tmp_path).snapshot().list_ids() == ["existing"]


@pytest.mark.parametrize("selected", ["index", "wal"])
def test_concurrent_index_refuses_symbolic_link_state_paths(tmp_path, selected):
    from truthound.stores.concurrency.index import ConcurrentIndex

    root = tmp_path / "selected"
    index = ConcurrentIndex(root)
    index.add("existing", {})
    outside = tmp_path / "outside.json"
    if selected == "index":
        target = root / "_index.json"
        outside.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(outside)
        before = outside.read_bytes()
    else:
        target = root / "_index.json.wal"
        target.symlink_to(outside)
        before = None
    with pytest.raises(ValueError):
        ConcurrentIndex(root).add("must-not-write", {})
    assert target.is_symlink()
    assert (outside.read_bytes() if outside.exists() else None) == before


def test_concurrent_relative_directory_rebuild_keeps_native_records(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = ConcurrentFileSystemStore(base_path="relative-store")
    store.save(record())
    assert store.rebuild_index() == 1
    assert store.get("run").id == "run"
    store.close()
