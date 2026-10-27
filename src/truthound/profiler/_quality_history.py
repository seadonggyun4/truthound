"""Atomic, bounded quality-history snapshots and cooperating writer locks."""

import json
from contextlib import contextmanager
from datetime import UTC, datetime


@contextmanager
def writer_lock(path):
    from truthound.stores.concurrency.locks import (
        LockMode,
        NoOpLockStrategy,
        get_default_lock_strategy,
    )

    if path is None:
        yield
        return
    strategy = get_default_lock_strategy()
    if isinstance(strategy, NoOpLockStrategy):
        raise RuntimeError("Quality history requires a native file lock")
    handle = strategy.acquire(path.with_name(path.name + ".lock"), LockMode.EXCLUSIVE, timeout=5)
    try:
        yield
    finally:
        strategy.release(handle)


def load(path):
    from truthound.profiler.quality import (
        ConfusionMatrix,
        QualityLevel,
        QualityMetrics,
        QualityTrendPoint,
    )

    if not path.exists():
        return {}
    if path.is_symlink() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Invalid quality history snapshot")
    try:
        def invalid_constant(value):
            raise ValueError("Nonfinite quality history value")

        data = json.loads(path.read_text(), parse_constant=invalid_constant)
        if not isinstance(data, dict) or len(data) > 4096:
            raise ValueError
        result = {}
        for name, points in data.items():
            if not isinstance(name, str) or not isinstance(points, list) or len(points) > 100:
                raise ValueError
            restored = []
            for point in points:
                values = dict(point["metrics"])
                values["quality_level"] = QualityLevel(values["quality_level"])
                values["evaluated_at"] = datetime.fromisoformat(values["evaluated_at"])
                for key in ("precision_ci", "recall_ci", "f1_ci"):
                    values[key] = tuple(values[key])
                if values.get("confusion_matrix") is not None:
                    counts = values["confusion_matrix"]
                    values["confusion_matrix"] = ConfusionMatrix(**{
                        key: counts[key] for key in ConfusionMatrix.__dataclass_fields__
                    })
                timestamp = datetime.fromisoformat(point["timestamp"])
                if timestamp.tzinfo is not None:
                    timestamp = timestamp.astimezone(UTC).replace(tzinfo=None)
                restored.append(QualityTrendPoint(
                    timestamp, QualityMetrics(**values),
                    point["data_size"], point["notes"],
                ))
            result[name] = restored
        return result
    except (ValueError, TypeError, KeyError, AttributeError):
        raise ValueError("Invalid quality history snapshot") from None


def save(path, trends):
    from truthound.stores._filesystem_io import publish

    data = {
        name: [{"timestamp": point.timestamp.isoformat(), "metrics": point.metrics.to_dict(),
                "data_size": point.data_size, "notes": point.notes} for point in points]
        for name, points in trends.items()
    }
    content = json.dumps(data, allow_nan=False).encode()
    if len(data) > 4096 or len(content) > 16 * 1024 * 1024:
        raise ValueError("Quality history capacity exceeded")
    publish(path, content)
