"""Native River concept detectors; persist data-only reference windows."""

from __future__ import annotations

import json
import math

import polars as pl

from truthound.ml.base import DriftResult


def validate(config):
    if config.method not in {"correlation", "ddm", "adwin", "page_hinkley"}:
        raise ValueError("Unknown concept drift method")
    for name in (
        "reference_window",
        "detection_window",
        "min_samples_per_window",
        "min_window",
        "adwin_clock",
        "adwin_max_buckets",
        "adwin_min_window_length",
    ):
        value = getattr(config, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in (
        "warning_threshold",
        "drift_threshold",
        "adwin_delta",
        "page_hinkley_delta",
        "page_hinkley_threshold",
        "page_hinkley_alpha",
    ):
        value = getattr(config, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if config.warning_threshold >= config.drift_threshold:
        raise ValueError("warning_threshold must be below drift_threshold")
    if not 0 < config.adwin_delta < 1 or not 0 < config.page_hinkley_alpha <= 1:
        raise ValueError("Invalid streaming detector probability/fading parameter")
    if config.page_hinkley_mode not in {"up", "down", "both"}:
        raise ValueError("Invalid Page-Hinkley mode")
    if config.method == "correlation":
        if config.stream_column is not None or config.prediction_column is not None:
            raise ValueError("Stream selectors require a streaming method")
    else:
        if (config.stream_column is None) == (config.prediction_column is None):
            raise ValueError("Select stream_column or prediction_column")
        if config.prediction_column is not None and config.target_column is None:
            raise ValueError("prediction_column requires target_column")
        if config.feature_columns is not None:
            raise ValueError("Streaming methods do not select feature correlations")


def detector(config):
    validate(config)
    try:
        from river import drift
    except ImportError as error:
        raise ImportError("Streaming concept drift requires truthound[drift]") from error
    if config.method == "ddm":
        return drift.binary.DDM(
            warm_start=config.min_window,
            warning_threshold=config.warning_threshold,
            drift_threshold=config.drift_threshold,
        )
    if config.method == "adwin":
        return drift.ADWIN(
            delta=config.adwin_delta,
            clock=config.adwin_clock,
            max_buckets=config.adwin_max_buckets,
            min_window_length=config.adwin_min_window_length,
            grace_period=config.min_window,
        )
    if config.method == "page_hinkley":
        return drift.PageHinkley(
            min_instances=config.min_window,
            delta=config.page_hinkley_delta,
            threshold=config.page_hinkley_threshold,
            alpha=config.page_hinkley_alpha,
            mode=config.page_hinkley_mode,
        )
    raise ValueError("Correlation does not use a streaming detector")


def window(config, data, *, reference):
    size = config.reference_window if reference else config.detection_window
    frame = data.tail(size).collect()
    if frame.height < config.min_samples_per_window:
        raise ValueError("Insufficient samples in selected concept window")
    selected = config.stream_column or config.prediction_column
    columns = [selected] if config.stream_column else [selected, config.target_column]
    if any(column not in frame.columns for column in columns):
        raise ValueError("Selected concept stream columns are missing")
    if any(frame[column].null_count() for column in columns):
        raise ValueError("Concept streams must not contain null observations")
    if config.stream_column:
        series = frame[selected]
        if not (series.dtype.is_numeric() or series.dtype == pl.Boolean):
            raise ValueError("Concept stream must be numeric")
        values = series.cast(pl.Float64).to_list()
    else:
        for column in columns:
            series = frame[column]
            if series.dtype.is_float() and not series.is_finite().all():
                raise ValueError("Prediction and target must be finite")
        values = (
            frame.select((pl.col(selected) != pl.col(config.target_column)).cast(pl.Float64))
            .to_series()
            .to_list()
        )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Concept stream must be finite")
    if config.method == "ddm" and any(value not in (0.0, 1.0) for value in values):
        raise ValueError("DDM requires a binary error stream")
    return frame, values


def detect(config, reference, current, columns):
    validate(config)
    selected = config.stream_column or config.prediction_column
    if columns is not None and columns != [selected]:
        raise ValueError("Streaming detection uses only its configured input column")
    _, baseline = window(config, reference, reference=True)
    _, observed = window(config, current, reference=False)
    native = detector(config)
    for value in baseline:
        native.update(value)
    count = warnings = 0
    first = None
    for index, value in enumerate(observed):
        native.update(value)
        warnings += int(getattr(native, "warning_detected", False))
        if native.drift_detected:
            count += 1
            if first is None:
                first = index
    score = float(count > 0)
    return DriftResult(
        is_drifted=count > 0,
        drift_score=score,
        column_scores=((selected, score),),
        drift_type=config.method + "_change" if count else "none",
        confidence=0.0,
        details=json.dumps(
            {
                "method": config.method,
                "reference_samples": len(baseline),
                "current_samples": len(observed),
                "drift_count": count,
                "first_drift_index": first,
                "warning_samples": warnings,
            },
            sort_keys=True,
        ),
    )
