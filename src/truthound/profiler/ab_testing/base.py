"""Base types and data structures for A/B testing.

Provides core abstractions for experiment configuration and results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# Enums
# =============================================================================


class ExperimentStatus(str, Enum):
    """Status of an experiment."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MetricType(str, Enum):
    """Types of metrics to track."""

    # Validation metrics
    VIOLATION_COUNT = "violation_count"
    VIOLATION_RATE = "violation_rate"
    FALSE_POSITIVE_RATE = "false_positive_rate"
    FALSE_NEGATIVE_RATE = "false_negative_rate"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"

    # Performance metrics
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"

    # Quality metrics
    DATA_QUALITY_SCORE = "data_quality_score"
    COMPLETENESS = "completeness"
    VALIDITY = "validity"

    # Custom
    CUSTOM = "custom"


class VariantType(str, Enum):
    """Type of variant in experiment."""

    CONTROL = "control"
    TREATMENT = "treatment"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ThresholdVariant:
    """A threshold configuration variant for testing.

    Represents either control or treatment in an A/B test.
    """

    name: str
    variant_type: VariantType
    thresholds: Dict[str, Any]
    description: str = ""
    weight: float = 0.5  # Traffic allocation weight
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "variant_type": self.variant_type.value,
            "thresholds": self.thresholds,
            "description": self.description,
            "weight": self.weight,
            "metadata": self.metadata,
        }


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment."""

    name: str
    description: str = ""
    # Variants
    control: Optional[ThresholdVariant] = None
    treatments: List[ThresholdVariant] = field(default_factory=list)
    # Metrics to track
    primary_metric: MetricType = MetricType.VIOLATION_RATE
    secondary_metrics: List[MetricType] = field(default_factory=list)
    # Statistical settings
    confidence_level: float = 0.95  # 95% confidence
    minimum_effect_size: float = 0.05  # 5% relative improvement
    minimum_sample_size: int = 100
    maximum_sample_size: int = 100000
    # Experiment settings
    allocation_strategy: str = "equal"  # equal, weighted, epsilon_greedy
    early_stopping: bool = True
    early_stopping_threshold: float = 0.01  # Stop if p-value < threshold
    # Guardrail metrics
    guardrail_metrics: List[tuple[MetricType, str, float]] = field(default_factory=list)
    # Time settings
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_duration_hours: int = 168  # 1 week default
    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "control": self.control.to_dict() if self.control else None,
            "treatments": [t.to_dict() for t in self.treatments],
            "primary_metric": self.primary_metric.value,
            "secondary_metrics": [m.value for m in self.secondary_metrics],
            "confidence_level": self.confidence_level,
            "minimum_effect_size": self.minimum_effect_size,
            "minimum_sample_size": self.minimum_sample_size,
            "early_stopping": self.early_stopping,
            "tags": self.tags,
            "metadata": self.metadata,
        }


# =============================================================================
# Results
# =============================================================================


@dataclass
class MetricResult:
    """Result for a single metric."""

    metric_type: MetricType
    variant_name: str
    value: float
    sample_size: int
    std_dev: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_type": self.metric_type.value,
            "variant_name": self.variant_name,
            "value": self.value,
            "sample_size": self.sample_size,
            "std_dev": self.std_dev,
            "confidence_interval": self.confidence_interval,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class StatisticalAnalysis:
    """Statistical analysis results."""

    metric_type: MetricType
    control_mean: float
    treatment_mean: float
    absolute_difference: float
    relative_difference: float
    p_value: float
    is_significant: bool
    confidence_level: float
    confidence_interval: tuple[float, float]
    effect_size: float  # Cohen's d
    power: float
    sample_size_control: int
    sample_size_treatment: int
    winner: Optional[str] = None
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_type": self.metric_type.value,
            "control_mean": self.control_mean,
            "treatment_mean": self.treatment_mean,
            "absolute_difference": self.absolute_difference,
            "relative_difference": self.relative_difference,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "confidence_level": self.confidence_level,
            "confidence_interval": self.confidence_interval,
            "effect_size": self.effect_size,
            "power": self.power,
            "sample_size_control": self.sample_size_control,
            "sample_size_treatment": self.sample_size_treatment,
            "winner": self.winner,
            "recommendation": self.recommendation,
        }


@dataclass
class ExperimentResult:
    """Complete result of an A/B experiment."""

    experiment_id: str
    experiment_name: str
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime]

    # Metrics per variant
    control_metrics: Dict[MetricType, MetricResult] = field(default_factory=dict)
    treatment_metrics: Dict[str, Dict[MetricType, MetricResult]] = field(default_factory=dict)

    # Analysis
    primary_analysis: Optional[StatisticalAnalysis] = None
    secondary_analyses: List[StatisticalAnalysis] = field(default_factory=list)

    # Summary
    winner: Optional[str] = None
    recommendation: str = ""
    guardrail_violations: List[str] = field(default_factory=list)

    # Metadata
    total_samples: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "control_metrics": {
                k.value: v.to_dict() for k, v in self.control_metrics.items()
            },
            "treatment_metrics": {
                name: {k.value: v.to_dict() for k, v in metrics.items()}
                for name, metrics in self.treatment_metrics.items()
            },
            "primary_analysis": self.primary_analysis.to_dict() if self.primary_analysis else None,
            "secondary_analyses": [a.to_dict() for a in self.secondary_analyses],
            "winner": self.winner,
            "recommendation": self.recommendation,
            "guardrail_violations": self.guardrail_violations,
            "total_samples": self.total_samples,
            "metadata": self.metadata,
        }
