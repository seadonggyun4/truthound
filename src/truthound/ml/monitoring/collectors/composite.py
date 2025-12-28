"""Composite metric collector.

Combines multiple collectors into a single interface.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from truthound.ml.monitoring.protocols import (
    IMetricCollector,
    ModelMetrics,
    PredictionRecord,
)


class CompositeCollector(IMetricCollector):
    """Combines multiple collectors.

    Merges metrics from all collectors into a single ModelMetrics.

    Example:
        >>> collector = CompositeCollector([
        ...     PerformanceCollector(),
        ...     DriftCollector(),
        ...     QualityCollector(),
        ... ])
        >>> metrics = collector.collect(model_id, predictions)
    """

    def __init__(self, collectors: list[IMetricCollector]):
        """Initialize composite collector.

        Args:
            collectors: List of collectors to combine
        """
        self._collectors = collectors

    @property
    def name(self) -> str:
        return "composite"

    @property
    def collectors(self) -> list[IMetricCollector]:
        """Get list of collectors."""
        return list(self._collectors)

    def add_collector(self, collector: IMetricCollector) -> None:
        """Add a collector.

        Args:
            collector: Collector to add
        """
        self._collectors.append(collector)

    def remove_collector(self, name: str) -> bool:
        """Remove a collector by name.

        Args:
            name: Collector name

        Returns:
            True if removed
        """
        for i, c in enumerate(self._collectors):
            if c.name == name:
                self._collectors.pop(i)
                return True
        return False

    def collect(
        self,
        model_id: str,
        predictions: list[PredictionRecord],
    ) -> ModelMetrics:
        """Collect metrics from all collectors.

        Args:
            model_id: Model identifier
            predictions: Prediction records

        Returns:
            Merged metrics from all collectors
        """
        if not self._collectors:
            return ModelMetrics(
                model_id=model_id,
                timestamp=datetime.now(timezone.utc),
            )

        # Collect from first collector
        result = self._collectors[0].collect(model_id, predictions)

        # Merge remaining collectors
        for collector in self._collectors[1:]:
            metrics = collector.collect(model_id, predictions)
            result = result.merge(metrics)

        return result

    def reset(self) -> None:
        """Reset all collectors."""
        for collector in self._collectors:
            collector.reset()
