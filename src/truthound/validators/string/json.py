"""JSON parseable validators."""

import concurrent.futures
import json
import os
from typing import Any

import polars as pl

from truthound.validators.base import (
    ValidationIssue,
    Validator,
    StringValidatorMixin,
    SampledEarlyTerminationMixin,
)
from truthound.validators.registry import register_validator

# Batch processing configuration
_BATCH_CHUNK_SIZE = int(os.environ.get("TRUTHOUND_JSON_CHUNK_SIZE", "10000"))
_MAX_WORKERS = int(os.environ.get("TRUTHOUND_JSON_MAX_WORKERS", "4"))


@register_validator
class JsonParseableValidator(Validator, StringValidatorMixin, SampledEarlyTerminationMixin):
    """Validates that string values are valid JSON."""

    name = "json_parseable"
    category = "string"

    def __init__(
        self,
        strict: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.strict = strict

    @staticmethod
    def _is_valid_json(val: str | None) -> bool:
        """Check if value is valid JSON."""
        if val is None:
            return True  # Nulls are not considered invalid
        try:
            json.loads(val)
            return True
        except (json.JSONDecodeError, TypeError):
            return False

    @staticmethod
    def _validate_json_batch(values: list[str | None]) -> tuple[int, list[str]]:
        """Validate a batch of JSON values.

        Args:
            values: List of string values to validate.

        Returns:
            Tuple of (invalid_count, sample_invalid_values).
        """
        invalid_count = 0
        samples: list[str] = []
        max_samples = 5  # Collect limited samples per batch

        for v in values:
            if v is None:
                continue
            try:
                json.loads(v)
            except (json.JSONDecodeError, TypeError):
                invalid_count += 1
                if len(samples) < max_samples:
                    samples.append(v[:50] + "..." if len(v) > 50 else v)

        return invalid_count, samples

    def _count_invalid_parallel(
        self,
        col_data: list[str | None],
    ) -> tuple[int, list[str]]:
        """Count invalid JSON values using parallel batch processing.

        Args:
            col_data: Column data as list of strings.

        Returns:
            Tuple of (total_invalid_count, sample_invalid_values).
        """
        # For small datasets, process directly without parallelization overhead
        if len(col_data) < _BATCH_CHUNK_SIZE * 2:
            return self._validate_json_batch(col_data)

        # Split into chunks for parallel processing
        chunks = [
            col_data[i : i + _BATCH_CHUNK_SIZE]
            for i in range(0, len(col_data), _BATCH_CHUNK_SIZE)
        ]

        total_invalid = 0
        all_samples: list[str] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            results = list(executor.map(self._validate_json_batch, chunks))

        for invalid_count, samples in results:
            total_invalid += invalid_count
            # Collect samples up to limit
            remaining = self.config.sample_size - len(all_samples)
            if remaining > 0:
                all_samples.extend(samples[:remaining])

        return total_invalid, all_samples

    def _build_invalid_expr(self, col: str) -> pl.Expr:
        """Build expression that returns True for invalid JSON values."""
        is_valid_expr = pl.col(col).map_elements(
            self._is_valid_json,
            return_dtype=pl.Boolean,
        )
        return pl.col(col).is_not_null() & ~is_valid_expr

    def _get_invalid_samples(self, lf: pl.LazyFrame, col: str) -> list[str]:
        """Get sample invalid values for error reporting."""
        samples_df = lf.with_columns(
            pl.col(col).map_elements(
                self._is_valid_json,
                return_dtype=pl.Boolean,
            ).alias("_is_valid")
        ).filter(
            pl.col(col).is_not_null() & ~pl.col("_is_valid")
        ).select(pl.col(col)).head(self.config.sample_size).collect(engine="streaming")

        return [
            (v[:50] + "..." if len(v) > 50 else v)
            for v in samples_df[col].to_list()
            if isinstance(v, str)
        ]

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_string_columns(lf)

        if not columns:
            return issues

        # Check for early termination opportunity
        early_results = self._check_early_termination(
            lf,
            columns=columns,
            build_invalid_expr=self._build_invalid_expr,
        )

        # Separate columns into early-terminatable and full-validation needed
        early_term_cols: list[str] = []
        full_validate_cols: list[str] = []

        for col in columns:
            if early_results[col].should_terminate:
                early_term_cols.append(col)
            else:
                full_validate_cols.append(col)

        # Process early termination columns
        for col in early_term_cols:
            result = early_results[col]
            samples = self._get_invalid_samples(lf.head(self.early_termination_sample_size), col)

            issues.append(
                self._build_early_termination_issue(
                    col=col,
                    result=result,
                    issue_type="invalid_json",
                    details="Values are not valid JSON",
                    sample_values=samples,
                )
            )

        # Full validation for remaining columns
        if not full_validate_cols:
            return issues

        # Collect data once for batch processing with streaming
        df = lf.select(full_validate_cols).collect(engine="streaming")

        for col in full_validate_cols:
            col_data = df[col].to_list()
            non_null_count = sum(1 for v in col_data if v is not None)

            if non_null_count == 0:
                continue

            # Use parallel batch processing for large datasets
            invalid_count, samples = self._count_invalid_parallel(col_data)

            if invalid_count == 0:
                continue

            ratio = invalid_count / non_null_count
            issues.append(
                ValidationIssue(
                    column=col,
                    issue_type="invalid_json",
                    count=invalid_count,
                    severity=self._calculate_severity(ratio),
                    details="Values are not valid JSON",
                    sample_values=samples[: self.config.sample_size],
                )
            )

        return issues
