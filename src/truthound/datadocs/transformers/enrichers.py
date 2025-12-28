"""Enricher transformers for the report pipeline.

These transformers add computed metadata, quality scores,
and recommendations to the report data.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any, Callable, List

from truthound.datadocs.engine.context import ReportContext, ReportData
from truthound.datadocs.transformers.base import BaseTransformer


class EnricherTransformer(BaseTransformer):
    """General-purpose enricher transformer.

    Adds computed fields and metadata to the report data.

    Example:
        enricher = EnricherTransformer(
            add_timestamp=True,
            add_quality_score=True,
            add_recommendations=True,
        )
    """

    def __init__(
        self,
        add_timestamp: bool = True,
        add_quality_score: bool = True,
        add_recommendations: bool = True,
        add_summary: bool = True,
        custom_enrichers: list[Callable[[ReportData], ReportData]] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the enricher.

        Args:
            add_timestamp: Add generation timestamp.
            add_quality_score: Calculate and add quality score.
            add_recommendations: Generate recommendations.
            add_summary: Add data summary.
            custom_enrichers: List of custom enricher functions.
            name: Transformer name.
        """
        super().__init__(name=name or "EnricherTransformer")
        self._add_timestamp = add_timestamp
        self._add_quality_score = add_quality_score
        self._add_recommendations = add_recommendations
        self._add_summary = add_summary
        self._custom_enrichers = custom_enrichers or []

    def _do_transform(self, ctx: ReportContext) -> ReportContext:
        data = ctx.data

        if self._add_timestamp:
            data = data.with_metadata(
                generated_at=datetime.now().isoformat(),
                generated_by="Truthound",
            )

        if self._add_quality_score:
            quality_score = self._calculate_quality_score(data)
            data = data.with_metadata(quality_score=quality_score)

        if self._add_recommendations:
            data = self._add_recommendations_to_data(data)

        if self._add_summary:
            data = self._add_summary_to_data(data)

        for enricher in self._custom_enrichers:
            data = enricher(data)

        return ctx.with_data(data)

    def _calculate_quality_score(self, data: ReportData) -> dict[str, float]:
        """Calculate quality scores.

        Args:
            data: Report data.

        Returns:
            Dictionary of quality scores.
        """
        columns = data.raw.get("columns", [])
        if not columns:
            return {"overall": 100.0, "completeness": 100.0, "uniqueness": 100.0}

        # Completeness score (based on null ratios)
        null_ratios = [c.get("null_ratio", 0) for c in columns]
        avg_null_ratio = sum(null_ratios) / len(null_ratios) if null_ratios else 0
        completeness = (1 - avg_null_ratio) * 100

        # Uniqueness score
        unique_ratios = [c.get("unique_ratio", 0) for c in columns]
        avg_unique = sum(unique_ratios) / len(unique_ratios) if unique_ratios else 0
        uniqueness = min(avg_unique * 100, 100)

        # Validity score (based on type inference confidence)
        validity = 100.0
        for col in columns:
            if col.get("inferred_type") == "unknown":
                validity -= 5
            if col.get("is_constant"):
                validity -= 2
        validity = max(validity, 0)

        # Overall score
        overall = (completeness * 0.4 + uniqueness * 0.3 + validity * 0.3)

        return {
            "overall": round(overall, 1),
            "completeness": round(completeness, 1),
            "uniqueness": round(uniqueness, 1),
            "validity": round(validity, 1),
        }

    def _add_recommendations_to_data(self, data: ReportData) -> ReportData:
        """Add recommendations based on data analysis.

        Args:
            data: Report data.

        Returns:
            Data with recommendations added.
        """
        recommendations = list(data.recommendations)
        columns = data.raw.get("columns", [])

        for col in columns:
            col_name = col.get("name", "")

            # High null ratio
            null_ratio = col.get("null_ratio", 0)
            if null_ratio > 0.5:
                recommendations.append(
                    f"Column '{col_name}' has {null_ratio:.1%} missing values. "
                    "Consider imputation or removal."
                )

            # Constant column
            if col.get("is_constant"):
                recommendations.append(
                    f"Column '{col_name}' is constant. "
                    "Consider removing if not informative."
                )

            # Suggested validators
            validators = col.get("suggested_validators", [])
            for v in validators[:2]:
                recommendations.append(
                    f"Add {v} validator for column '{col_name}'."
                )

        # Duplicate rows
        dup_ratio = data.raw.get("duplicate_row_ratio", 0)
        if dup_ratio > 0.05:
            recommendations.append(
                f"{dup_ratio:.1%} of rows are duplicates. "
                "Consider implementing deduplication."
            )

        return replace(data, recommendations=recommendations[:20])

    def _add_summary_to_data(self, data: ReportData) -> ReportData:
        """Add summary statistics to metadata.

        Args:
            data: Report data.

        Returns:
            Data with summary added.
        """
        columns = data.raw.get("columns", [])

        summary = {
            "row_count": data.raw.get("row_count", 0),
            "column_count": len(columns),
            "total_null_cells": sum(c.get("null_count", 0) for c in columns),
            "constant_columns": sum(1 for c in columns if c.get("is_constant")),
            "numeric_columns": sum(
                1 for c in columns
                if c.get("inferred_type") in ("integer", "float", "decimal")
            ),
            "text_columns": sum(
                1 for c in columns
                if c.get("inferred_type") in ("string", "text")
            ),
            "date_columns": sum(
                1 for c in columns
                if c.get("inferred_type") in ("date", "datetime", "timestamp")
            ),
        }

        return data.with_metadata(summary=summary)


class MetadataEnricher(BaseTransformer):
    """Add custom metadata to the report.

    Example:
        enricher = MetadataEnricher(
            title="Monthly Sales Report",
            author="Data Team",
            department="Analytics",
            custom_fields={"environment": "production"},
        )
    """

    def __init__(
        self,
        title: str | None = None,
        subtitle: str | None = None,
        author: str | None = None,
        description: str | None = None,
        version: str | None = None,
        custom_fields: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize metadata enricher.

        Args:
            title: Report title.
            subtitle: Report subtitle.
            author: Report author.
            description: Report description.
            version: Report version.
            custom_fields: Additional custom fields.
            name: Transformer name.
        """
        super().__init__(name=name or "MetadataEnricher")
        self._title = title
        self._subtitle = subtitle
        self._author = author
        self._description = description
        self._version = version
        self._custom_fields = custom_fields or {}

    def _do_transform(self, ctx: ReportContext) -> ReportContext:
        data = ctx.data

        metadata_updates = {}

        if self._title:
            metadata_updates["title"] = self._title
        if self._subtitle:
            metadata_updates["subtitle"] = self._subtitle
        if self._author:
            metadata_updates["author"] = self._author
        if self._description:
            metadata_updates["description"] = self._description
        if self._version:
            metadata_updates["version"] = self._version

        metadata_updates.update(self._custom_fields)

        if metadata_updates:
            data = data.with_metadata(**metadata_updates)

        return ctx.with_data(data)


class QualityScoreEnricher(BaseTransformer):
    """Dedicated quality score calculator.

    Provides detailed quality scoring with configurable weights.

    Example:
        enricher = QualityScoreEnricher(
            weights={
                "completeness": 0.4,
                "uniqueness": 0.3,
                "validity": 0.2,
                "consistency": 0.1,
            },
        )
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        thresholds: dict[str, float] | None = None,
        include_column_scores: bool = True,
        name: str | None = None,
    ) -> None:
        """Initialize quality score enricher.

        Args:
            weights: Dimension weights (must sum to 1.0).
            thresholds: Alert thresholds for each dimension.
            include_column_scores: Include per-column scores.
            name: Transformer name.
        """
        super().__init__(name=name or "QualityScoreEnricher")
        self._weights = weights or {
            "completeness": 0.4,
            "uniqueness": 0.3,
            "validity": 0.2,
            "consistency": 0.1,
        }
        self._thresholds = thresholds or {
            "critical": 50.0,
            "warning": 70.0,
            "good": 85.0,
        }
        self._include_column_scores = include_column_scores

    def _do_transform(self, ctx: ReportContext) -> ReportContext:
        data = ctx.data
        columns = data.raw.get("columns", [])

        if not columns:
            return ctx

        # Calculate dimension scores
        dimensions = {
            "completeness": self._calculate_completeness(columns),
            "uniqueness": self._calculate_uniqueness(columns),
            "validity": self._calculate_validity(columns),
            "consistency": self._calculate_consistency(data.raw),
        }

        # Calculate overall score
        overall = sum(
            dimensions.get(dim, 0) * weight
            for dim, weight in self._weights.items()
        )

        # Determine grade
        if overall >= self._thresholds["good"]:
            grade = "A"
        elif overall >= self._thresholds["warning"]:
            grade = "B"
        elif overall >= self._thresholds["critical"]:
            grade = "C"
        else:
            grade = "D"

        quality_score = {
            "overall": round(overall, 1),
            "dimensions": {k: round(v, 1) for k, v in dimensions.items()},
            "grade": grade,
            "thresholds": self._thresholds,
        }

        # Add column-level scores if requested
        if self._include_column_scores:
            column_scores = self._calculate_column_scores(columns)
            quality_score["columns"] = column_scores

        return ctx.with_data(data.with_metadata(quality_score=quality_score))

    def _calculate_completeness(self, columns: List[dict]) -> float:
        """Calculate completeness score."""
        null_ratios = [c.get("null_ratio", 0) for c in columns]
        if not null_ratios:
            return 100.0
        avg_null = sum(null_ratios) / len(null_ratios)
        return (1 - avg_null) * 100

    def _calculate_uniqueness(self, columns: List[dict]) -> float:
        """Calculate uniqueness score."""
        unique_ratios = [c.get("unique_ratio", 0) for c in columns]
        if not unique_ratios:
            return 100.0
        avg_unique = sum(unique_ratios) / len(unique_ratios)
        return min(avg_unique * 100, 100)

    def _calculate_validity(self, columns: List[dict]) -> float:
        """Calculate validity score."""
        score = 100.0
        for col in columns:
            if col.get("inferred_type") == "unknown":
                score -= 5
            if col.get("is_constant"):
                score -= 2
        return max(score, 0)

    def _calculate_consistency(self, raw_data: dict) -> float:
        """Calculate consistency score."""
        score = 100.0

        # Penalize duplicate rows
        dup_ratio = raw_data.get("duplicate_row_ratio", 0)
        if dup_ratio > 0.1:
            score -= 20
        elif dup_ratio > 0.05:
            score -= 10
        elif dup_ratio > 0.01:
            score -= 5

        return max(score, 0)

    def _calculate_column_scores(self, columns: List[dict]) -> List[dict]:
        """Calculate per-column quality scores."""
        column_scores = []

        for col in columns:
            col_name = col.get("name", "")
            null_ratio = col.get("null_ratio", 0)
            unique_ratio = col.get("unique_ratio", 0)

            completeness = (1 - null_ratio) * 100
            uniqueness = min(unique_ratio * 100, 100)
            validity = 100.0 if col.get("inferred_type") != "unknown" else 50.0

            if col.get("is_constant"):
                validity -= 20

            overall = (completeness * 0.4 + uniqueness * 0.3 + validity * 0.3)

            column_scores.append({
                "name": col_name,
                "overall": round(overall, 1),
                "completeness": round(completeness, 1),
                "uniqueness": round(uniqueness, 1),
                "validity": round(validity, 1),
            })

        return column_scores


class RecommendationEnricher(BaseTransformer):
    """Generate recommendations based on data quality issues.

    Example:
        enricher = RecommendationEnricher(
            max_recommendations=10,
            include_validators=True,
            include_improvements=True,
        )
    """

    def __init__(
        self,
        max_recommendations: int = 10,
        include_validators: bool = True,
        include_improvements: bool = True,
        null_threshold: float = 0.3,
        duplicate_threshold: float = 0.05,
        custom_rules: list[Callable[[dict], str | None]] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize recommendation enricher.

        Args:
            max_recommendations: Maximum recommendations to generate.
            include_validators: Include validator suggestions.
            include_improvements: Include improvement suggestions.
            null_threshold: Threshold for null ratio alerts.
            duplicate_threshold: Threshold for duplicate alerts.
            custom_rules: Custom recommendation rules.
            name: Transformer name.
        """
        super().__init__(name=name or "RecommendationEnricher")
        self._max_recommendations = max_recommendations
        self._include_validators = include_validators
        self._include_improvements = include_improvements
        self._null_threshold = null_threshold
        self._duplicate_threshold = duplicate_threshold
        self._custom_rules = custom_rules or []

    def _do_transform(self, ctx: ReportContext) -> ReportContext:
        data = ctx.data
        recommendations = list(data.recommendations)
        columns = data.raw.get("columns", [])

        if self._include_improvements:
            recommendations.extend(self._generate_improvements(data.raw, columns))

        if self._include_validators:
            recommendations.extend(self._generate_validator_suggestions(columns))

        # Apply custom rules
        for rule in self._custom_rules:
            result = rule(data.raw)
            if result:
                recommendations.append(result)

        # Deduplicate and limit
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recs.append(rec)

        return ctx.with_data(replace(
            data,
            recommendations=unique_recs[:self._max_recommendations]
        ))

    def _generate_improvements(
        self,
        raw_data: dict,
        columns: List[dict],
    ) -> List[str]:
        """Generate improvement recommendations."""
        recs = []

        # High null columns
        high_null_cols = [
            c.get("name", "")
            for c in columns
            if c.get("null_ratio", 0) > self._null_threshold
        ]
        if high_null_cols:
            recs.append(
                f"Review data collection for columns with high missing values: "
                f"{', '.join(high_null_cols[:3])}"
            )

        # Duplicate rows
        dup_ratio = raw_data.get("duplicate_row_ratio", 0)
        if dup_ratio > self._duplicate_threshold:
            recs.append(
                "Implement duplicate row detection in your data pipeline."
            )

        # Constant columns
        constant_cols = [c.get("name", "") for c in columns if c.get("is_constant")]
        if constant_cols:
            recs.append(
                f"Consider removing constant columns: {', '.join(constant_cols[:3])}"
            )

        return recs

    def _generate_validator_suggestions(self, columns: List[dict]) -> List[str]:
        """Generate validator suggestions."""
        recs = []

        for col in columns:
            validators = col.get("suggested_validators", [])
            col_name = col.get("name", "")

            for v in validators[:2]:
                recs.append(f"Add {v} validator for column '{col_name}'.")

        return recs
