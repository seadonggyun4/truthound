"""Transformers for the Data Docs report pipeline.

Transformers modify the ReportContext data during the pipeline's transform phase.
They are applied in order, with each receiving the output of the previous.

Available Transformers:
- I18nTransformer: Internationalization and localization
- FilterTransformer: Section and data filtering
- EnricherTransformer: Metadata and data enrichment
- AggregatorTransformer: Data aggregation and summarization
"""

from truthound.datadocs.transformers.base import (
    Transformer,
    BaseTransformer,
    ChainedTransformer,
    ConditionalTransformer,
    TransformResult,
)
from truthound.datadocs.transformers.i18n import (
    I18nTransformer,
    TranslationResolver,
)
from truthound.datadocs.transformers.filters import (
    FilterTransformer,
    SectionFilter,
    ColumnFilter,
    AlertFilter,
)
from truthound.datadocs.transformers.enrichers import (
    EnricherTransformer,
    MetadataEnricher,
    QualityScoreEnricher,
    RecommendationEnricher,
)

__all__ = [
    # Base
    "Transformer",
    "BaseTransformer",
    "ChainedTransformer",
    "ConditionalTransformer",
    "TransformResult",
    # I18n
    "I18nTransformer",
    "TranslationResolver",
    # Filters
    "FilterTransformer",
    "SectionFilter",
    "ColumnFilter",
    "AlertFilter",
    # Enrichers
    "EnricherTransformer",
    "MetadataEnricher",
    "QualityScoreEnricher",
    "RecommendationEnricher",
]
