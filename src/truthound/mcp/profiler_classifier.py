"""Factory-owned native type classifiers for the existing inference boundary."""

import math
from copy import deepcopy
from dataclasses import dataclass

from truthound.mcp.policy import MCPError


@dataclass
class NativeClassifierBinding:
    model: object
    confidence_threshold: float = 0.0
    sample_size: int = 1000

    def __post_init__(self):
        from truthound.profiler.ml.base import BaseModel
        from truthound.profiler.ml.feature_extraction import FeatureExtractor
        from truthound.profiler.ml.pretrained import PretrainedModel

        if not math.isfinite(self.confidence_threshold) or not 0 <= self.confidence_threshold <= 1:
            raise ValueError("Invalid classifier confidence threshold")
        if type(self.sample_size) is not int or self.sample_size < 1:
            raise ValueError("sample_size must be a positive integer")
        if isinstance(self.model, BaseModel):
            self.model = PretrainedModel(
                self.model.name, self.model.version, self.model, [], list(self.model._feature_names)
            )
        if not isinstance(self.model, PretrainedModel) or not self.model.model.is_trained:
            raise TypeError("Expected a trained native classifier or pretrained model")
        self.model = deepcopy(self.model)
        self.extractor = FeatureExtractor(sample_size=self.sample_size)

    def infer(self, column, context=None):
        from truthound.profiler.base import DataType
        from truthound.profiler.ml_inference import InferenceResult

        features = self.extractor.extract(column, deepcopy(context) if context else {})
        result = self.model.predict(features)
        if not math.isfinite(result.confidence) or not 0 <= result.confidence <= 1:
            raise ValueError("Invalid native classifier confidence")
        return InferenceResult(
            column_name=column.name,
            inferred_type=result.predicted_type
            if result.confidence >= self.confidence_threshold
            else DataType.UNKNOWN,
            confidence=result.confidence,
            alternatives=result.top_alternatives,
            features_used=features.names,
            model_version=result.model_version,
            inference_time_ms=result.inference_time_ms,
        )

    def infer_table(self, frame):
        return {name: self.infer(frame[name], {"column_name": name}) for name in frame.columns}

    def clear_cache(self):
        """No implicit cache; native model state belongs to the factory."""

    def provide_feedback(self, *args):
        raise MCPError("unsupported_operation")
