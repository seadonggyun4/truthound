"""Approved native custom profiler compositions, retained results and fixed persistence slots."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from truthound.mcp.approved_sources import SourceBindingOpen
from truthound.mcp.bound_services import BoundServiceSession
from truthound.mcp.checkpoint_services import ServiceProcess
from truthound.mcp.contracts import Name
from truthound.mcp.policy import MCPError
from truthound.mcp.session_contracts import Handle


@dataclass
class NativeSamplingBinding:
    """Factory-selected native strategy and private sampling configuration."""

    strategy: Any
    config: Any

    def __post_init__(self):
        from dataclasses import replace

        from truthound.profiler.sampling import SamplingConfig, SamplingStrategy
        if not isinstance(self.strategy, SamplingStrategy) or not isinstance(self.config, SamplingConfig):
            raise TypeError("Expected a native sampling strategy and configuration")
        self.config = replace(self.config)

    def sample(self, frame):
        result = self.strategy.sample(frame, self.config)
        if self.config.max_memory_mb > 0:
            snapshot = result.data.collect()
            if snapshot.estimated_size() > self.config.max_memory_mb * 1024 * 1024:
                raise MemoryError("Sample exceeds the configured output memory budget")
            result.data = snapshot.lazy()
        return result


@dataclass
class ProfilerRuntime:
    """Factory owns configuration, custom code, executor registry and physical paths."""

    profilers: dict[str, Any] = field(default_factory=dict)
    generators: dict[str, Any] = field(default_factory=dict)
    executors: dict[str, Any] = field(default_factory=dict)
    profile_slots: dict[str, Path] = field(default_factory=dict)
    suite_slots: dict[str, Path] = field(default_factory=dict)
    incrementals: dict[str, Any] = field(default_factory=dict)
    streaming: dict[str, Any] = field(default_factory=dict)
    chunk_factories: dict[str, Any] = field(default_factory=dict)
    stream_progress: dict[str, Any] = field(default_factory=dict)
    caches: dict[str, Any] = field(default_factory=dict)
    cache_keys: dict[str, Any] = field(default_factory=dict)
    cache_patterns: dict[str, str] = field(default_factory=dict)
    visualizers: dict[str, Any] = field(default_factory=dict)
    visualization_slots: dict[str, Path] = field(default_factory=dict)
    inferrers: dict[str, Any] = field(default_factory=dict)
    inference_contexts: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    samplers: dict[str, Any] = field(default_factory=dict)
    quality_scorers: dict[str, Any] = field(default_factory=dict)
    quality_trends: dict[str, Any] = field(default_factory=dict)
    quality_rules: dict[str, Any] = field(default_factory=dict)
    schema_analyzers: dict[str, Any] = field(default_factory=dict)


class ProfilerCommand(SourceBindingOpen):
    service_id: Handle
    command: Literal[
        "visualization_export", "visualization_save",
        "inference_infer", "inference_table", "inference_feedback", "inference_cache_clear",
        "sampling_sample", "sample_get", "sample_release", "sample_profile", "sample_sketch_query",
        "quality_score", "quality_compare", "quality_cache_clear",
        "quality_record", "quality_history", "quality_trend",
        "schema_compare",
        "cache_get", "cache_set", "cache_get_or_compute", "cache_invalidate", "cache_invalidate_pattern", "cache_clear", "cache_stats",
        "inspect", "incremental_profile", "stream_profile", "profile", "profile_get", "profile_export", "profile_import", "profile_save",
        "profile_restore", "profile_release", "generate_suite", "suite_get",
        "suite_save", "suite_export", "suite_restore", "suite_release", "suite_execute", "slot_delete",
    ]
    export_format: Literal["html", "pdf"] = "html"
    visualizer_id: Name | None = None
    visualization_slot_id: Name | None = None
    inferrer_id: Name | None = None
    inference_context_id: Name | None = None
    column: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    true_type: Literal[
        "integer", "float", "string", "boolean", "datetime", "date", "time", "duration",
        "email", "url", "phone", "uuid", "ip_address", "json", "categorical", "identifier",
        "currency", "percentage", "korean_rrn", "korean_phone", "korean_business_number", "unknown",
    ] | None = None
    sampler_id: Name | None = None
    sketch_value: Annotated[str, Field(max_length=4096)] | int | Annotated[float, Field(allow_inf_nan=False)] | bool | None = None
    sample_id: Name | None = None
    offset: Annotated[int, Field(ge=0)] = 0
    columns: Annotated[list[str], Field(min_length=1, max_length=100)] | None = None
    trend_id: Name | None = None
    window_days: Annotated[int, Field(ge=1, le=3650)] = 30
    scorer_id: Name | None = None
    rule_ids: Annotated[list[Name], Field(min_length=1, max_length=100)] | None = None
    label_column: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    use_cache: bool = True
    profiler_id: Name | None = None
    cache_id: Name | None = None
    cache_key_id: Name | None = None
    pattern_id: Name | None = None
    ttl_seconds: Annotated[float, Field(ge=0, le=31_536_000, allow_inf_nan=False)] | None = None
    incremental_id: Name | None = None
    streaming_id: Name | None = None
    stream_id: Name | None = None
    progress_id: Name | None = None
    previous_profile_id: Name | None = None
    force_refresh: bool = False
    generator_id: Name | None = None
    executor_id: Name | None = None
    asset_id: Handle | None = None
    baseline_id: Handle | None = None
    current_id: Handle | None = None
    analyzer_id: Name | None = None
    artifact_id: Handle | None = None
    profile_id: Name | None = None
    suite_id: Name | None = None
    slot_id: Name | None = None
    slot_kind: Literal["profile", "suite"] | None = None
    name: Annotated[str, Field(max_length=512)] = ""
    strictness: Literal["loose", "medium", "strict"] = "medium"
    include_categories: Annotated[list[Name], Field(min_length=1, max_length=32)] | None = None
    exclude_categories: Annotated[list[Name], Field(min_length=1, max_length=32)] | None = None
    min_confidence: Literal["low", "medium", "high"] | None = None
    parallel: bool = False
    fail_fast: bool = False
    dry_run: bool = False
    timeout_seconds: Annotated[float, Field(gt=0, le=3600, allow_inf_nan=False)] | None = None
    max_workers: Annotated[int, Field(ge=1, le=32)] | None = None
    limit: Annotated[int, Field(ge=1, le=1000)] = 100

    @model_validator(mode="after")
    def shape(self):
        if "export_format" in self.model_fields_set and self.command not in {"visualization_export", "visualization_save"}:
            raise ValueError("Export format does not match command")
        required = {
            "visualizer_id": {"visualization_export", "visualization_save"},
            "visualization_slot_id": {"visualization_save"},
            "inferrer_id": {"inference_infer", "inference_table", "inference_feedback", "inference_cache_clear"},
            "column": {"inference_infer", "inference_feedback", "sample_sketch_query"},
            "true_type": {"inference_feedback"},
            "sampler_id": {"sampling_sample"},
            "sample_id": {"sample_get", "sample_release", "sample_profile", "sample_sketch_query"},
            "scorer_id": {"quality_record", "quality_score", "quality_compare", "quality_cache_clear"},
            "rule_ids": {"quality_record", "quality_history", "quality_trend", "quality_score", "quality_compare"},
            "trend_id": {"quality_record", "quality_history", "quality_trend"},
            "baseline_id": {"schema_compare"},
            "current_id": {"schema_compare"},
            "analyzer_id": {"schema_compare"},
            "profiler_id": {"sample_profile", "profile", "cache_get_or_compute"},
            "cache_id": {"cache_get", "cache_set", "cache_get_or_compute", "cache_invalidate", "cache_invalidate_pattern", "cache_clear", "cache_stats"},
            "cache_key_id": {"cache_get", "cache_set", "cache_get_or_compute", "cache_invalidate"},
            "pattern_id": {"cache_invalidate_pattern"},
            "generator_id": {"generate_suite"},
            "executor_id": {"suite_execute"},
            "incremental_id": {"incremental_profile"},
            "streaming_id": {"stream_profile"},
            "stream_id": {"stream_profile"},
            "asset_id": {"inference_infer", "inference_table", "inference_feedback", "sampling_sample", "quality_record", "quality_score", "quality_compare", "profile", "incremental_profile", "suite_execute", "cache_get_or_compute"},
            "artifact_id": {"profile_import"},
            "profile_id": {"visualization_export", "visualization_save", "profile_get", "profile_export", "profile_save", "profile_release", "generate_suite", "cache_set"},
            "suite_id": {"suite_get", "suite_export", "suite_save", "suite_release", "suite_execute"},
            "slot_id": {"profile_save", "profile_restore", "suite_save", "suite_restore", "slot_delete"},
            "slot_kind": {"slot_delete"},
        }
        for key, commands in required.items():
            if (self.command in commands) != (getattr(self, key) is not None):
                raise ValueError("Field does not match command")
        if ("sketch_value" in self.model_fields_set) != (self.command == "sample_sketch_query"):
            raise ValueError("Sketch query requires an explicit scalar value")
        if self.command in {"quality_score", "quality_record", "quality_history", "quality_trend"} and len(self.rule_ids) != 1:
            raise ValueError("quality_score requires one rule")
        if self.rule_ids is not None and len(set(self.rule_ids)) != len(self.rule_ids):
            raise ValueError("Duplicate quality rules")
        for key in ("label_column", "use_cache"):
            if key in self.model_fields_set and self.command not in {"quality_record", "quality_score", "quality_compare"}:
                raise ValueError("Field does not match command")
        for key in ("name", "strictness", "include_categories", "exclude_categories", "min_confidence"):
            if key in self.model_fields_set and self.command != "generate_suite":
                raise ValueError("Field does not match command")
        for key in ("parallel", "fail_fast", "dry_run", "timeout_seconds", "max_workers"):
            if key in self.model_fields_set and self.command != "suite_execute":
                raise ValueError("Field does not match command")
        if "ttl_seconds" in self.model_fields_set and self.command not in {"cache_set", "cache_get_or_compute"}:
            raise ValueError("Field does not match command")
        for key in ("previous_profile_id", "force_refresh"):
            if key in self.model_fields_set and self.command != "incremental_profile":
                raise ValueError("Field does not match command")
        if self.progress_id is not None and self.command != "stream_profile":
            raise ValueError("Field does not match command")
        if self.inference_context_id is not None and self.command not in {"inference_infer", "inference_feedback"}:
            raise ValueError("Field does not match command")
        for key in ("offset", "columns"):
            if key in self.model_fields_set and self.command != "sample_get":
                raise ValueError("Field does not match command")
        if "window_days" in self.model_fields_set and self.command != "quality_trend":
            raise ValueError("Field does not match command")
        if "limit" in self.model_fields_set and self.command not in {"inspect", "quality_history", "sample_get"}:
            raise ValueError("Field does not match command")
        return self


class ProfilerProcess(ServiceProcess):
    worker_module = "truthound.mcp.profiler_worker"

    def validate_reply(self, result):
        if result in (
            {"ok": True},
            {"ok": True, "native_failed": "native_profile_incomplete"},
            {"ok": True, "native_failed": "native_suite_incomplete"},
            {"ok": True, "native_failed": "native_cache_incomplete"},
        ):
            return result
        raise MCPError("worker_error")


class ProfilerServices(BoundServiceSession):
    group = "profiler_services"
    process_type = ProfilerProcess
    limit_commands = {"inspect", "quality_history", "sample_get"}
    effects = (
        "installed_extension_execute", "operator_selected_native_profiler_and_custom_components",
        "native_profile_and_rule_generation", "operator_selected_generated_suite_validation",
        "operator_configured_profile_suite_slot_read_write_delete", "sensitive_data_artifact",
        "operator_selected_native_incremental_and_chunked_profile",
        "operator_selected_chunk_source_read_close",
        "operator_selected_native_profile_cache_read_write_clear",
        "operator_selected_native_schema_analyzer_and_bound_rules",
        "native_quality_scoring_on_pinned_input_and_bound_rules",
        "operator_selected_quality_history_write_read",
        "native_sampling_on_pinned_input_and_retained_sample_profile",
        "native_type_inference_and_selected_model_feedback_persistence",
        "native_profile_visualization_and_operator_selected_report_slot_write",
    )

    def input_resources(self, arguments):
        if arguments.command == "schema_compare":
            return {
                "baseline_path": self.session.resources.get(arguments.baseline_id, "asset"),
                "current_path": self.session.resources.get(arguments.current_id, "asset"),
            }
        if arguments.command in {"inference_infer", "inference_table", "inference_feedback", "sampling_sample", "quality_record", "quality_score", "quality_compare", "profile", "incremental_profile", "suite_execute", "cache_get_or_compute"}:
            return {"asset_path": self.session.resources.get(arguments.asset_id, "asset")}
        if arguments.command == "profile_import":
            entry = self.session.resources.get(arguments.artifact_id, "artifact")
            if entry.media_type != "application/vnd.truthound.advanced-profile+json":
                raise MCPError("invalid_handle")
            return {"profile_path": entry}
        return {}

    def artifact_media_type(self, arguments):
        if arguments.command == "visualization_export":
            return "application/pdf" if arguments.export_format == "pdf" else "text/html"
        if arguments.command == "profile_export":
            return "application/vnd.truthound.advanced-profile+json"
        if arguments.command == "suite_export":
            return "application/vnd.truthound.generated-suite+json"
        return super().artifact_media_type(arguments)

    def reply_error(self, arguments, reply):
        return reply.get("native_failed")
