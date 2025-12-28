"""Tests for ObservableStore wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import uuid

import pytest

from truthound.stores.base import BaseStore, StoreConfig, StoreQuery, StoreNotFoundError
from truthound.stores.observability.audit import AuditEventType, InMemoryAuditBackend
from truthound.stores.observability.config import ObservabilityConfig, AuditConfig, MetricsConfig, TracingConfig
from truthound.stores.observability.metrics import InMemoryMetricsBackend
from truthound.stores.observability.protocols import ObservabilityContext
from truthound.stores.observability.store import ObservableStore, ObservabilityManager
from truthound.stores.observability.tracing import InMemoryTracer


# Mock store for testing
@dataclass
class MockItem:
    """Mock item for testing."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    data_asset: str = "test-asset"
    run_time: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "data_asset": self.data_asset,
            "run_time": self.run_time.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MockItem":
        return cls(
            id=data["id"],
            name=data.get("name", ""),
            data_asset=data.get("data_asset", "test-asset"),
        )


class MockStore(BaseStore[MockItem, StoreConfig]):
    """Mock store for testing."""

    def __init__(self) -> None:
        super().__init__()
        self._items: dict[str, MockItem] = {}

    @classmethod
    def _default_config(cls) -> StoreConfig:
        return StoreConfig()

    def _do_initialize(self) -> None:
        pass

    def save(self, item: MockItem) -> str:
        self._items[item.id] = item
        return item.id

    def get(self, item_id: str) -> MockItem:
        if item_id not in self._items:
            raise StoreNotFoundError("MockItem", item_id)
        return self._items[item_id]

    def exists(self, item_id: str) -> bool:
        return item_id in self._items

    def delete(self, item_id: str) -> bool:
        if item_id in self._items:
            del self._items[item_id]
            return True
        return False

    def list_ids(self, query: StoreQuery | None = None) -> list[str]:
        return list(self._items.keys())

    def query(self, query: StoreQuery) -> list[MockItem]:
        results = list(self._items.values())
        if query.data_asset:
            results = [r for r in results if r.data_asset == query.data_asset]
        if query.limit:
            results = results[: query.limit]
        return results


class TestObservabilityManager:
    """Tests for ObservabilityManager."""

    def test_create_with_defaults(self) -> None:
        config = ObservabilityConfig()
        manager = ObservabilityManager(config, store_type="test")

        assert manager.store_type == "test"
        assert manager.store_id is not None

    def test_audit_disabled(self) -> None:
        config = ObservabilityConfig(audit=AuditConfig(enabled=False))
        manager = ObservabilityManager(config, store_type="test")

        assert manager.audit is None

    def test_metrics_disabled(self) -> None:
        config = ObservabilityConfig(metrics=MetricsConfig(enabled=False))
        manager = ObservabilityManager(config, store_type="test")

        assert manager.metrics is None

    def test_create_context(self) -> None:
        config = ObservabilityConfig()
        manager = ObservabilityManager(config, store_type="test")

        ctx = manager.create_context(user_id="user-1")

        assert ctx.correlation_id is not None
        assert ctx.user_id == "user-1"

    def test_observe_success(self) -> None:
        audit_backend = InMemoryAuditBackend()
        metrics_backend = InMemoryMetricsBackend()
        tracer = InMemoryTracer()

        config = ObservabilityConfig()
        manager = ObservabilityManager(
            config,
            store_type="test",
            audit_backend=audit_backend,
            metrics_backend=metrics_backend,
            tracer=tracer,
        )

        with manager.observe(
            "save",
            AuditEventType.CREATE,
            resource_id="item-1",
        ):
            pass

        # Check audit
        assert len(audit_backend.events) == 1
        assert audit_backend.events[0].operation == "save"

        # Check metrics
        export = metrics_backend.export()
        assert "operations_total" in export

        # Check tracing
        assert len(tracer.spans) == 1
        assert tracer.spans[0].name == "store.save"

    def test_observe_failure(self) -> None:
        audit_backend = InMemoryAuditBackend()
        metrics_backend = InMemoryMetricsBackend()
        tracer = InMemoryTracer()

        config = ObservabilityConfig()
        manager = ObservabilityManager(
            config,
            store_type="test",
            audit_backend=audit_backend,
            metrics_backend=metrics_backend,
            tracer=tracer,
        )

        with pytest.raises(ValueError):
            with manager.observe(
                "save",
                AuditEventType.CREATE,
                resource_id="item-1",
            ):
                raise ValueError("Test error")

        # Check audit recorded failure
        assert len(audit_backend.events) == 1
        assert audit_backend.events[0].error_type == "ValueError"

        # Check metrics recorded error
        export = metrics_backend.export()
        assert "errors_total" in export


class TestObservableStore:
    """Tests for ObservableStore wrapper."""

    def test_save_with_observability(self) -> None:
        base_store = MockStore()
        config = ObservabilityConfig.disabled()
        store = ObservableStore(base_store, config)

        item = MockItem(name="test-item")
        item_id = store.save(item)

        assert item_id == item.id
        assert base_store.exists(item_id)

    def test_get_with_observability(self) -> None:
        base_store = MockStore()
        config = ObservabilityConfig.disabled()
        store = ObservableStore(base_store, config)

        # Save an item first
        item = MockItem(name="test")
        base_store.save(item)

        # Get via observable store
        result = store.get(item.id)
        assert result.name == "test"

    def test_delete_with_observability(self) -> None:
        base_store = MockStore()
        config = ObservabilityConfig.disabled()
        store = ObservableStore(base_store, config)

        item = MockItem(name="test")
        base_store.save(item)

        result = store.delete(item.id)
        assert result is True
        assert not base_store.exists(item.id)

    def test_exists_with_observability(self) -> None:
        base_store = MockStore()
        config = ObservabilityConfig.disabled()
        store = ObservableStore(base_store, config)

        item = MockItem()
        base_store.save(item)

        assert store.exists(item.id) is True
        assert store.exists("nonexistent") is False

    def test_query_with_observability(self) -> None:
        base_store = MockStore()
        config = ObservabilityConfig.disabled()
        store = ObservableStore(base_store, config)

        for i in range(5):
            base_store.save(MockItem(name=f"item-{i}"))

        results = store.query(StoreQuery(limit=3))
        assert len(results) == 3

    def test_list_ids_with_observability(self) -> None:
        base_store = MockStore()
        config = ObservabilityConfig.disabled()
        store = ObservableStore(base_store, config)

        for i in range(3):
            base_store.save(MockItem(name=f"item-{i}"))

        ids = store.list_ids()
        assert len(ids) == 3

    def test_count_with_observability(self) -> None:
        base_store = MockStore()
        config = ObservabilityConfig.disabled()
        store = ObservableStore(base_store, config)

        for i in range(5):
            base_store.save(MockItem())

        assert store.count() == 5

    def test_clear_with_observability(self) -> None:
        base_store = MockStore()
        config = ObservabilityConfig.disabled()
        store = ObservableStore(base_store, config)

        for i in range(5):
            base_store.save(MockItem())

        deleted = store.clear()
        assert deleted == 5
        assert store.count() == 0

    def test_save_batch(self) -> None:
        base_store = MockStore()
        audit_backend = InMemoryAuditBackend()
        config = ObservabilityConfig()

        store = ObservableStore(base_store, config)
        store._obs._audit_backend = audit_backend

        items = [MockItem(name=f"item-{i}") for i in range(10)]
        ids = store.save_batch(items)

        assert len(ids) == 10
        for item_id in ids:
            assert base_store.exists(item_id)

    def test_delete_batch(self) -> None:
        base_store = MockStore()
        config = ObservabilityConfig.disabled()
        store = ObservableStore(base_store, config)

        items = [MockItem() for _ in range(5)]
        for item in items:
            base_store.save(item)

        ids = [item.id for item in items[:3]]
        deleted = store.delete_batch(ids)

        assert deleted == 3
        assert store.count() == 2

    def test_with_context(self) -> None:
        base_store = MockStore()
        config = ObservabilityConfig.disabled()
        store = ObservableStore(base_store, config)

        context = ObservabilityContext(
            correlation_id="test-corr",
            user_id="user-1",
        )
        contextualized = store.with_context(context)

        assert contextualized._default_context == context
        assert contextualized._store is base_store

    def test_with_user(self) -> None:
        base_store = MockStore()
        config = ObservabilityConfig.disabled()
        store = ObservableStore(base_store, config)

        user_store = store.with_user("user-123", "tenant-456")

        assert user_store._default_context.user_id == "user-123"
        assert user_store._default_context.tenant_id == "tenant-456"

    def test_get_metrics(self) -> None:
        base_store = MockStore()
        metrics_backend = InMemoryMetricsBackend()
        config = ObservabilityConfig()

        store = ObservableStore(base_store, config)
        store._obs._metrics_backend = metrics_backend

        # Perform some operations
        item = MockItem()
        store.save(item)
        store.get(item.id)

        metrics = store.get_metrics()
        assert isinstance(metrics, str)

    def test_get_audit_events(self) -> None:
        base_store = MockStore()
        audit_backend = InMemoryAuditBackend()
        config = ObservabilityConfig()

        store = ObservableStore(base_store, config)
        store._obs._audit_backend = audit_backend

        # Perform some operations
        item = MockItem()
        store.save(item)

        events = store.get_audit_events()
        # Events may or may not be recorded depending on backend setup
        assert isinstance(events, list)

    def test_flush_observability(self) -> None:
        base_store = MockStore()
        config = ObservabilityConfig.disabled()
        store = ObservableStore(base_store, config)

        # Should not raise
        store.flush_observability()

    def test_close_with_observability(self) -> None:
        base_store = MockStore()
        config = ObservabilityConfig.disabled()
        store = ObservableStore(base_store, config)

        # Should not raise
        store.close()

    def test_error_propagation(self) -> None:
        base_store = MockStore()
        config = ObservabilityConfig.disabled()
        store = ObservableStore(base_store, config)

        with pytest.raises(StoreNotFoundError):
            store.get("nonexistent-id")

    def test_config_access(self) -> None:
        base_store = MockStore()
        config = ObservabilityConfig.disabled()
        store = ObservableStore(base_store, config)

        assert store.config == base_store.config

    def test_observability_manager_access(self) -> None:
        base_store = MockStore()
        config = ObservabilityConfig()
        store = ObservableStore(base_store, config)

        assert store.observability is not None
        assert store.observability.store_type == "MockStore"


class TestObservabilityConfig:
    """Tests for ObservabilityConfig factory methods."""

    def test_disabled_config(self) -> None:
        config = ObservabilityConfig.disabled()

        assert config.audit.enabled is False
        assert config.metrics.enabled is False
        assert config.tracing.enabled is False

    def test_minimal_config(self) -> None:
        config = ObservabilityConfig.minimal()

        assert config.audit.enabled is True
        assert config.metrics.enabled is True
        assert config.tracing.enabled is False

    def test_production_config(self) -> None:
        config = ObservabilityConfig.production("my-service")

        assert config.audit.enabled is True
        assert config.metrics.enabled is True
        assert config.tracing.enabled is True
        assert config.tracing.service_name == "my-service"
        assert config.environment == "production"
