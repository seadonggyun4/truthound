"""Memory stores retain independent snapshots and honor bounded updates/query settings."""

from datetime import datetime

import pytest

from truthound.stores.backends.memory import MemoryConfig, MemoryExpectationStore, MemoryStore
from truthound.stores.base import StoreQuery
from truthound.stores.expectations import ExpectationSuite
from truthound.stores.results import ResultStatus, ValidationResult

pytestmark = pytest.mark.contract


def record(identity, asset="asset"):
    return ValidationResult(
        identity, datetime(2026, 1, 1), asset, ResultStatus.SUCCESS, tags={"selected": "yes"}
    )


def test_update_at_capacity_cannot_evict_an_unrelated_record():
    store = MemoryStore(max_items=2)
    store.save(record("a"))
    store.save(record("b"))
    store.save(record("b", "replacement"))
    assert set(store.list_ids()) == {"a", "b"}
    assert store.get("b").data_asset == "replacement"


def test_deep_copy_includes_filter_metadata_not_only_record_body():
    store = MemoryStore(deep_copy=True)
    original = record("a")
    store.save(original)
    original.tags["selected"] = "changed"
    assert store.list_ids(StoreQuery(tags={"selected": "yes"})) == ["a"]
    assert store.get("a").tags == {"selected": "yes"}


def test_selected_order_and_zero_limit_match_actual_record_selection():
    store = MemoryStore()
    store.save(record("a", "z"))
    store.save(record("b", "a"))
    assert store.query(StoreQuery(limit=0)) == []
    assert store.list_ids(StoreQuery(order_by="data_asset", ascending=True, offset=1, limit=1)) == [
        "a"
    ]


def test_explicit_memory_config_and_suite_capacity_are_applied():
    store = MemoryStore(config=MemoryConfig(max_items=1, metadata={"configured": "value"}))
    store.save(record("a"))
    store.save(record("b"))
    assert store.list_ids() == ["b"]
    assert store.get("b").metadata["configured"] == "value"
    suites = MemoryExpectationStore(max_items=1)
    suites.save(ExpectationSuite.create("a", "asset"))
    suites.save(ExpectationSuite.create("b", "asset"))
    assert suites.list_ids() == ["b"]
    assert suites.query(StoreQuery(limit=0)) == []


@pytest.mark.parametrize(
    "settings",
    [{"max_items": -1}, {"max_items": True}, {"deep_copy": 1}, {"unrecognized_option": 1}],
)
def test_invalid_memory_configuration_is_not_silently_ignored(settings):
    with pytest.raises((ValueError, TypeError)):
        MemoryStore(**settings)


def test_native_copy_callback_revocation_prevents_memory_commit():
    allowed = [True]

    class RevokeDuringCopy:
        def __deepcopy__(self, memo):
            allowed[0] = False
            return self

    def guard():
        if not allowed[0]:
            raise PermissionError("revoked")

    store = MemoryStore()
    store.set_io_guard(guard)
    item = record("a")
    item.metadata["callback"] = RevokeDuringCopy()
    with pytest.raises(PermissionError, match="revoked"):
        store.save(item)
    allowed[0] = True
    assert not store.exists("a")
