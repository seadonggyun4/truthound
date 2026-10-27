"""Quality metric collector.

Collects model quality metrics (accuracy, precision, recall, F1).
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from truthound.ml.monitoring.protocols import (
    IMetricCollector,
    ModelMetrics,
    PredictionRecord,
)


@dataclass
class QualityConfig:
    """Configuration for quality collector.

    Attributes:
        window_size: Number of predictions to track
        classification_threshold: Threshold for binary classification
        positive_label: Positive class label for binary classification
    """

    window_size: int = 1000
    classification_threshold: float = 0.5
    positive_label: Any = 1
    task_type: str = "auto"

    def __post_init__(self):
        if type(self.window_size) is not int or self.window_size <= 0:
            raise ValueError("Quality window size must be a positive integer")
        if (
            isinstance(self.classification_threshold, bool)
            or not isinstance(self.classification_threshold, (int, float))
            or not math.isfinite(self.classification_threshold)
            or not 0 <= self.classification_threshold <= 1
        ):
            raise ValueError("Classification threshold must be finite and between zero and one")
        if self.task_type not in {"auto", "binary", "multiclass", "regression"}:
            raise ValueError("Unsupported quality task type")


class QualityCollector(IMetricCollector):
    """Collects model quality metrics.

    Supports:
    - Binary classification metrics
    - Multi-class classification metrics
    - Regression metrics (if configured)

    Example:
        >>> collector = QualityCollector()
        >>> # Record predictions with actual values
        >>> metrics = collector.collect(model_id, predictions_with_actuals)
        >>> print(f"Accuracy: {metrics.accuracy}")
    """

    def __init__(self, config: QualityConfig | None = None):
        self._config = config or QualityConfig()
        # Confusion matrix components per model
        self._true_positives: dict[str, int] = defaultdict(int)
        self._true_negatives: dict[str, int] = defaultdict(int)
        self._false_positives: dict[str, int] = defaultdict(int)
        self._false_negatives: dict[str, int] = defaultdict(int)
        # For multi-class
        self._class_counts: dict[str, dict[Any, dict[str, int]]] = {}
        # For regression
        self._squared_errors: dict[str, list[float]] = {}
        self._absolute_errors: dict[str, list[float]] = {}
        self._windows: dict[str, list[PredictionRecord]] = {}

    @property
    def name(self) -> str:
        return "quality"

    def collect(
        self,
        model_id: str,
        predictions: list[PredictionRecord],
    ) -> ModelMetrics:
        """Collect quality metrics from predictions.

        Args:
            model_id: Model identifier
            predictions: Prediction records (must have actual values)

        Returns:
            Quality metrics
        """
        # Filter predictions with actual values
        with_actuals = [p for p in predictions if p.actual is not None]
        window = self._windows.setdefault(model_id, [])
        window.extend(copy.deepcopy(with_actuals))
        del window[: -self._config.window_size]
        with_actuals = window

        if not with_actuals:
            # Return empty metrics if no actuals
            return ModelMetrics(
                model_id=model_id,
                timestamp=datetime.now(UTC),
            )

        # Determine task type and compute metrics
        first_pred = with_actuals[0].prediction
        first_actual = with_actuals[0].actual

        is_classification = self._config.task_type != "regression" and (
            self._config.task_type != "auto" or self._is_classification(first_pred, first_actual)
        )

        if is_classification:
            return self._collect_classification_metrics(model_id, with_actuals)
        else:
            return self._collect_regression_metrics(model_id, with_actuals)

    def _is_classification(self, prediction: Any, actual: Any) -> bool:
        """Determine if this is a classification task."""
        # Check if values are numeric and continuous
        try:
            pred_val = float(prediction)
            actual_val = float(actual)
            # If both are small integers, treat as classification
            if pred_val == int(pred_val) and actual_val == int(actual_val):
                return True
            # If prediction is probability-like, treat as classification
            return 0 <= pred_val <= 1
        except (TypeError, ValueError):
            # Non-numeric, treat as classification
            return True

    def _collect_classification_metrics(
        self,
        model_id: str,
        predictions: list[PredictionRecord],
    ) -> ModelMetrics:
        """Collect classification metrics."""
        # Determine if binary or multi-class
        classes = set()
        for p in predictions:
            classes.add(p.prediction)
            classes.add(p.actual)

        actual_classes = {record.actual for record in predictions}
        binary_candidates = (
            len(actual_classes) <= 2
            and (self._config.positive_label in classes or actual_classes <= {0, 1})
            and all(
                record.prediction in actual_classes
                or record.prediction == self._config.positive_label
                or isinstance(record.prediction, float)
                and 0 <= record.prediction <= 1
                for record in predictions
            )
        )

        if self._config.task_type == "binary" or (
            self._config.task_type != "multiclass" and binary_candidates
        ):
            return self._collect_binary_metrics(model_id, predictions)
        else:
            return self._collect_multiclass_metrics(model_id, predictions)

    def _collect_binary_metrics(
        self,
        model_id: str,
        predictions: list[PredictionRecord],
    ) -> ModelMetrics:
        """Collect binary classification metrics."""
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for pred in predictions:
            # Handle probability outputs
            pred_val = pred.prediction
            if isinstance(pred_val, float) and 0 <= pred_val <= 1:
                pred_positive = pred_val >= self._config.classification_threshold
            else:
                pred_positive = pred_val == self._config.positive_label

            actual_val = pred.actual

            # Convert to positive/negative
            actual_positive = actual_val == self._config.positive_label

            if actual_positive and pred_positive:
                tp += 1
            elif not actual_positive and not pred_positive:
                tn += 1
            elif not actual_positive and pred_positive:
                fp += 1
            else:
                fn += 1

        # Update running counts
        self._true_positives[model_id] = tp
        self._true_negatives[model_id] = tn
        self._false_positives[model_id] = fp
        self._false_negatives[model_id] = fn

        # Compute metrics
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else None

        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None

        f1 = None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)

        custom_metrics: dict[str, float] = {
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "total_predictions": total,
        }

        # Specificity
        if (tn + fp) > 0:
            custom_metrics["specificity"] = tn / (tn + fp)

        # Balanced accuracy
        if recall is not None and "specificity" in custom_metrics:
            custom_metrics["balanced_accuracy"] = (recall + custom_metrics["specificity"]) / 2

        return ModelMetrics(
            model_id=model_id,
            timestamp=datetime.now(UTC),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            custom_metrics=custom_metrics,
        )

    def _collect_multiclass_metrics(
        self,
        model_id: str,
        predictions: list[PredictionRecord],
    ) -> ModelMetrics:
        """Collect multi-class classification metrics."""
        # Build confusion matrix
        self._class_counts[model_id] = defaultdict(lambda: defaultdict(int))

        correct = 0
        total = 0

        for pred in predictions:
            self._class_counts[model_id][pred.actual][pred.prediction] += 1
            if pred.prediction == pred.actual:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else None

        # Compute macro-averaged precision/recall/F1
        class_metrics = self._compute_per_class_metrics(model_id)

        precisions = [m["precision"] for m in class_metrics.values() if m["precision"] is not None]
        recalls = [m["recall"] for m in class_metrics.values() if m["recall"] is not None]
        f1s = [m["f1"] for m in class_metrics.values() if m["f1"] is not None]

        macro_precision = sum(precisions) / len(precisions) if precisions else None
        macro_recall = sum(recalls) / len(recalls) if recalls else None
        macro_f1 = sum(f1s) / len(f1s) if f1s else None

        custom_metrics: dict[str, float] = {
            "total_predictions": total,
            "num_classes": len(class_metrics),
        }

        # Add per-class metrics
        for cls, metrics in class_metrics.items():
            cls_str = str(cls)
            if metrics["precision"] is not None:
                custom_metrics[f"precision_{cls_str}"] = metrics["precision"]
            if metrics["recall"] is not None:
                custom_metrics[f"recall_{cls_str}"] = metrics["recall"]
            if metrics["f1"] is not None:
                custom_metrics[f"f1_{cls_str}"] = metrics["f1"]

        return ModelMetrics(
            model_id=model_id,
            timestamp=datetime.now(UTC),
            accuracy=accuracy,
            precision=macro_precision,
            recall=macro_recall,
            f1_score=macro_f1,
            custom_metrics=custom_metrics,
        )

    def _compute_per_class_metrics(self, model_id: str) -> dict[Any, dict[str, float | None]]:
        """Compute per-class precision/recall/F1."""
        confusion = self._class_counts.get(model_id, {})
        all_classes = set()

        for actual, preds in confusion.items():
            all_classes.add(actual)
            all_classes.update(preds.keys())

        metrics: dict[Any, dict[str, float | None]] = {}

        for cls in all_classes:
            tp = confusion.get(cls, {}).get(cls, 0)
            fp = sum(confusion.get(other, {}).get(cls, 0) for other in all_classes if other != cls)
            fn = sum(confusion.get(cls, {}).get(other, 0) for other in all_classes if other != cls)

            precision = tp / (tp + fp) if (tp + fp) > 0 else None
            recall = tp / (tp + fn) if (tp + fn) > 0 else None
            f1 = None
            if precision is not None and recall is not None and (precision + recall) > 0:
                f1 = 2 * precision * recall / (precision + recall)

            metrics[cls] = {"precision": precision, "recall": recall, "f1": f1}

        return metrics

    def _collect_regression_metrics(
        self,
        model_id: str,
        predictions: list[PredictionRecord],
    ) -> ModelMetrics:
        """Collect regression metrics."""
        if model_id not in self._squared_errors:
            self._squared_errors[model_id] = []
            self._absolute_errors[model_id] = []

        squared_errors = []
        absolute_errors = []

        for pred in predictions:
            try:
                pred_val = float(pred.prediction)
                actual_val = float(pred.actual)
                error = pred_val - actual_val
                squared_errors.append(error**2)
                absolute_errors.append(abs(error))
            except (TypeError, ValueError):
                raise ValueError("Native regression labels must be finite numbers") from None
            if not math.isfinite(pred_val) or not math.isfinite(actual_val):
                raise ValueError("Native regression labels must be finite numbers")

        self._squared_errors[model_id] = squared_errors
        self._absolute_errors[model_id] = absolute_errors

        # Keep window size
        while len(self._squared_errors[model_id]) > self._config.window_size:
            self._squared_errors[model_id].pop(0)
            self._absolute_errors[model_id].pop(0)

        # Compute metrics
        all_squared = self._squared_errors[model_id]
        all_absolute = self._absolute_errors[model_id]

        mse = sum(all_squared) / len(all_squared) if all_squared else None
        mae = sum(all_absolute) / len(all_absolute) if all_absolute else None
        rmse = mse**0.5 if mse is not None else None

        custom_metrics: dict[str, float] = {
            "total_predictions": len(all_squared),
        }

        if mse is not None:
            custom_metrics["mse"] = mse
        if mae is not None:
            custom_metrics["mae"] = mae
        if rmse is not None:
            custom_metrics["rmse"] = rmse

        return ModelMetrics(
            model_id=model_id,
            timestamp=datetime.now(UTC),
            custom_metrics=custom_metrics,
        )

    def reset(self) -> None:
        """Reset collector state."""
        self._true_positives.clear()
        self._true_negatives.clear()
        self._false_positives.clear()
        self._false_negatives.clear()
        self._class_counts.clear()
        self._squared_errors.clear()
        self._absolute_errors.clear()
        self._windows.clear()

    def get_confusion_matrix(self, model_id: str) -> dict[Any, dict[Any, int]]:
        """Get confusion matrix for model.

        Args:
            model_id: Model identifier

        Returns:
            Confusion matrix as nested dict
        """
        return dict(self._class_counts.get(model_id, {}))
