from __future__ import annotations

import json
import threading

import pytest

from truthound.observability.logging import (
    JSONFormatter,
    LogContext,
    LogHandler,
    LogLevel,
    StructuredLogger,
    log_context,
)


pytestmark = pytest.mark.fault


class _CollectingHandler(LogHandler):
    def __init__(self) -> None:
        super().__init__(formatter=JSONFormatter())
        self.records: list[dict[str, object]] = []

    def emit(self, record) -> None:  # type: ignore[override]
        self.records.append(json.loads(self.formatter.format(record)))


class _ExplodingHandler(LogHandler):
    def emit(self, record) -> None:  # type: ignore[override]
        raise RuntimeError("sink offline")


@pytest.mark.contract
def test_structured_logger_emits_contextual_json_records():
    handler = _CollectingHandler()
    logger = StructuredLogger("truthound.test", level=LogLevel.INFO, handlers=[handler])

    with log_context(run_id="run-1"):
        logger.bind(component="checkpoint").info("validation started", batch_id="b-1")

    assert len(handler.records) == 1
    assert handler.records[0]["message"] == "validation started"
    assert handler.records[0]["run_id"] == "run-1"
    assert handler.records[0]["component"] == "checkpoint"
    assert handler.records[0]["batch_id"] == "b-1"


def test_structured_logger_isolates_handler_failures():
    healthy = _CollectingHandler()
    logger = StructuredLogger(
        "truthound.test",
        level=LogLevel.INFO,
        handlers=[_ExplodingHandler(), healthy],
    )

    logger.error("validation failed", checkpoint="daily-orders")

    assert len(healthy.records) == 1
    assert healthy.records[0]["message"] == "validation failed"
    assert healthy.records[0]["checkpoint"] == "daily-orders"


def test_log_context_does_not_leak_across_threads():
    LogContext.clear()
    observed: dict[str, dict[str, object]] = {}

    with log_context(run_id="main-thread"):
        def worker() -> None:
            observed["context"] = LogContext.get_current()

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

    assert observed["context"] == {}
    assert LogContext.get_current() == {}
