"""Real observable stores initialize and retain truthful metrics through failed teardown."""

from datetime import datetime

import pytest

from truthound.stores.backends.filesystem import FileSystemStore
from truthound.stores.base import StoreQuery
from truthound.stores.observability.config import AuditConfig, ObservabilityConfig, TracingConfig
from truthound.stores.observability.store import ObservableStore
from truthound.stores.results import ResultStatus, ValidationResult

pytestmark = pytest.mark.fault


def test_native_observable_initialization_and_one_counter_per_actual_operation(tmp_path):
    store = ObservableStore(
        FileSystemStore(base_path=tmp_path),
        ObservabilityConfig(
            audit=AuditConfig(backend="memory"), tracing=TracingConfig(enabled=False)
        ),
    )
    try:
        store.initialize()
        store.save(
            ValidationResult("local-run", datetime(2026, 1, 1), "local-asset", ResultStatus.SUCCESS)
        )
        assert len(store.query(StoreQuery())) == 1
        metrics = store.observability.metrics
        for operation in ("save", "query"):
            assert (
                metrics.backend.get_counter(
                    "operations_total",
                    {
                        "store_type": metrics.store_type,
                        "store_id": metrics.store_id,
                        "operation": operation,
                    },
                )
                == 1
            )
        assert store.config is store._store.config
    finally:
        store.close()


def test_native_failed_backend_close_still_shuts_down_observability(tmp_path, monkeypatch):
    store = ObservableStore(
        FileSystemStore(base_path=tmp_path),
        ObservabilityConfig(
            audit=AuditConfig(backend="memory"), tracing=TracingConfig(enabled=False)
        ),
    )
    closed = []

    def fail():
        raise RuntimeError("synthetic-private-backend-close")

    monkeypatch.setattr(store._store, "close", fail)
    monkeypatch.setattr(store.observability, "shutdown", lambda: closed.append("shutdown"))
    with pytest.raises(RuntimeError):
        store.close()
    assert closed == ["shutdown"]


def test_native_audit_close_failure_still_closes_tracer(tmp_path, monkeypatch):
    store = ObservableStore(
        FileSystemStore(base_path=tmp_path),
        ObservabilityConfig(
            audit=AuditConfig(backend="memory"), tracing=TracingConfig(enabled=False)
        ),
    )
    closed = []

    def fail():
        raise RuntimeError("synthetic-private-audit-close")

    monkeypatch.setattr(store.observability.audit, "close", fail)
    monkeypatch.setattr(store.observability.tracer, "shutdown", lambda: closed.append("tracer"))
    with pytest.raises(RuntimeError):
        store.observability.shutdown()
    assert closed == ["tracer"]
