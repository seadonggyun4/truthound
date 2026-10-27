"""Execute native profiler objects and fixed slots in one approved owner process."""

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from truthound.mcp.broker_worker import NativeServiceResult
from truthound.mcp.checkpoint_worker import binding_guard
from truthound.mcp.encoding import encode
from truthound.mcp.governance_worker import selected
from truthound.mcp.policy import MCPError
from truthound.mcp.profiler_services import ProfilerRuntime


class ProfilerOwner:
    def __init__(self, runtime, request):
        from truthound.mcp.profiler_classifier import NativeClassifierBinding
        from truthound.mcp.profiler_services import NativeSamplingBinding
        from truthound.profiler.base import Profiler
        from truthound.profiler.caching import ProfileCache
        from truthound.profiler.distributed import DistributedProfiler
        from truthound.profiler.evolution.compatibility import SchemaCompatibilityAnalyzer
        from truthound.profiler.generators.suite_generator import ValidationSuiteGenerator
        from truthound.profiler.incremental import IncrementalProfiler
        from truthound.profiler.integration.executor import SuiteExecutor
        from truthound.profiler.ml_inference import MLTypeInferrer
        from truthound.profiler.quality import (
            QualityTrendAnalyzer,
            RuleQualityScorer,
            ValidationRule,
        )
        from truthound.profiler.sampling import Sampler
        from truthound.profiler.streaming import StreamingProfiler
        from truthound.profiler.visualization import HTMLReportGenerator

        if request["native_group"] != "profiler_services" or not isinstance(runtime, ProfilerRuntime):
            raise MCPError("invalid_binding")
        if (
            not (runtime.profilers or runtime.incrementals or runtime.streaming or runtime.schema_analyzers or runtime.quality_scorers or runtime.quality_trends or runtime.samplers or runtime.inferrers or runtime.visualizers)
            or any(not isinstance(value, HTMLReportGenerator) for value in runtime.visualizers.values())
            or any(not isinstance(value, Path) or not value.is_absolute() for value in runtime.visualization_slots.values())
            or any(not isinstance(value, (MLTypeInferrer, NativeClassifierBinding)) for value in runtime.inferrers.values())
            or any(not isinstance(value, dict) for value in runtime.inference_contexts.values())
            or any(not isinstance(value, (Sampler, NativeSamplingBinding)) for value in runtime.samplers.values())
            or any(not isinstance(value, RuleQualityScorer) for value in runtime.quality_scorers.values())
            or any(not isinstance(value, QualityTrendAnalyzer) for value in runtime.quality_trends.values())
            or any(not isinstance(value, ValidationRule) for value in runtime.quality_rules.values())
            or any(not isinstance(value, SchemaCompatibilityAnalyzer) for value in runtime.schema_analyzers.values())
            or any(not isinstance(value, (Profiler, DistributedProfiler)) for value in runtime.profilers.values())
            or any(not isinstance(value, ValidationSuiteGenerator) for value in runtime.generators.values())
            or any(not isinstance(value, SuiteExecutor) for value in runtime.executors.values())
            or any(not isinstance(value, ProfileCache) for value in runtime.caches.values())
            or any(not callable(getattr(value, "to_string", None)) for value in runtime.cache_keys.values())
            or any(not isinstance(value, str) for value in runtime.cache_patterns.values())
            or any(not isinstance(value, IncrementalProfiler) for value in runtime.incrementals.values())
            or any(not isinstance(value, StreamingProfiler) for value in runtime.streaming.values())
            or any(not callable(value) for value in (*runtime.chunk_factories.values(), *runtime.stream_progress.values()))
            or any(not isinstance(value, Path) or not value.is_absolute() for value in (*runtime.profile_slots.values(), *runtime.suite_slots.values()))
        ):
            raise MCPError("invalid_binding")
        self.inference_contexts = deepcopy(runtime.inference_contexts)
        self.runtime = runtime
        self.guard = binding_guard(request)
        for value in (*runtime.profilers.values(), *runtime.generators.values(), *runtime.executors.values(), *runtime.incrementals.values(), *runtime.streaming.values(), *runtime.caches.values()):
            value.set_execution_guard(self.guard)
        for value in runtime.schema_analyzers.values():
            value.set_execution_guard(self.guard)
        self.samples = {}
        self.sample_sketches = {}
        self.profiles = {}
        self.suites = {}
        self.sizes = {}

    def retain(self, value, kind, budget):
        rows = self.profiles if kind == "profile" else self.suites
        if len(rows) >= 16:
            raise MCPError("resource_exhausted")
        size = len(json.dumps(encode(value.to_dict()), allow_nan=False).encode())
        if size + sum(self.sizes.values()) > budget:
            raise MCPError("resource_exhausted")
        identity = kind + "_" + uuid4().hex
        rows[identity], self.sizes[identity] = value, size
        return identity

    def profile_view(self, identity, value):
        payload = {"profile_id": identity, "profile": value.to_dict()}
        return payload if value.complete else NativeServiceResult(payload, "native_profile_incomplete")

    def slot(self, kind, identity):
        return selected(self.runtime.profile_slots if kind == "profile" else self.runtime.suite_slots, identity)

    def save(self, value, kind, target):
        from truthound.profiler.generators.suite_generator import save_suite
        from truthound.profiler.table_profiler import save_profile

        self.guard()
        # Serialize through Core before atomically replacing the operator-selected slot.
        descriptor, name = tempfile.mkstemp(prefix=".truthound-profile-", dir=target.parent)
        os.close(descriptor)
        temporary = Path(name)
        try:
            if kind == "profile":
                save_profile(value, temporary)
            else:
                save_suite(value, temporary, format="json")
            self.guard()
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def restore(self, path, kind, budget):
        from truthound.profiler.generators.suite_generator import load_suite
        from truthound.profiler.table_profiler import load_profile

        self.guard()
        if path.stat().st_size > budget:
            raise MCPError("resource_exhausted")
        value = load_profile(path) if kind == "profile" else load_suite(path)
        self.guard()
        return value

    async def execute(self, args, budget, *, inputs=None):
        import polars as pl

        from truthound.profiler.base import Strictness, TableProfile
        from truthound.profiler.errors import IncompleteProfileError
        from truthound.profiler.generators.base import RuleConfidence
        from truthound.profiler.integration.protocols import ExecutionContext

        command = args["command"]
        if command in {"visualization_export", "visualization_save"}:
            from truthound.mcp.profiler_visualization import execute

            return execute(self, args, budget)
        if command in {"inference_infer", "inference_table", "inference_feedback", "inference_cache_clear"}:
            from truthound.mcp.profiler_inference import execute

            return execute(self, args, inputs)
        if command in {"sampling_sample", "sample_get", "sample_release", "sample_profile", "sample_sketch_query"}:
            from truthound.mcp.profiler_sampling import execute

            return execute(self, args, inputs, budget)
        if command in {"quality_score", "quality_compare", "quality_cache_clear", "quality_record", "quality_history", "quality_trend"}:
            from truthound.mcp.profiler_quality import execute

            return execute(self, args, inputs)
        if command == "schema_compare":
            if not inputs or set(inputs) != {"baseline_path", "current_path"}:
                raise MCPError("invalid_handle")
            self.guard()
            analyzer = selected(self.runtime.schema_analyzers, args["analyzer_id"])
            baseline = pl.scan_ipc(inputs["baseline_path"], memory_map=False).collect_schema()
            current = pl.scan_ipc(inputs["current_path"], memory_map=False).collect_schema()
            result = analyzer.analyze(baseline, current)
            self.guard()
            return {"is_compatible": result.is_compatible, "level": result.level.value,
                    "issues": [issue.to_dict() for issue in result.issues],
                    "rules_checked": result.rules_checked}
        if command == "inspect":
            limit = args.get("limit", 100)
            return {
                "visualizers": sorted(self.runtime.visualizers)[:limit],
                "visualization_slots": sorted(self.runtime.visualization_slots)[:limit],
                "inferrers": sorted(self.runtime.inferrers)[:limit],
                "inference_contexts": sorted(self.inference_contexts)[:limit],
                "samplers": sorted(self.runtime.samplers)[:limit],
                "samples": list(self.samples)[:limit],
                "quality_scorers": sorted(self.runtime.quality_scorers)[:limit],
                "quality_trends": sorted(self.runtime.quality_trends)[:limit],
                "quality_rules": sorted(self.runtime.quality_rules)[:limit],
                "profilers": sorted(self.runtime.profilers)[:limit],
                "schema_analyzers": sorted(self.runtime.schema_analyzers)[:limit],
                "generators": sorted(self.runtime.generators)[:limit],
                "executors": sorted(self.runtime.executors)[:limit],
                "incrementals": sorted(self.runtime.incrementals)[:limit],
                "streaming": sorted(self.runtime.streaming)[:limit],
                "streams": sorted(self.runtime.chunk_factories)[:limit],
                "caches": sorted(self.runtime.caches)[:limit],
                "cache_keys": sorted(self.runtime.cache_keys)[:limit],
                "cache_patterns": sorted(self.runtime.cache_patterns)[:limit],
                "stream_progress": sorted(self.runtime.stream_progress)[:limit],
                "profile_slots": sorted(self.runtime.profile_slots)[:limit],
                "suite_slots": sorted(self.runtime.suite_slots)[:limit],
                "profiles": list(self.profiles)[:limit], "suites": list(self.suites)[:limit],
            }
        if command in {"profile", "incremental_profile", "suite_execute", "cache_get_or_compute"}:
            if not inputs or set(inputs) != {"asset_path"}:
                raise MCPError("invalid_handle")
            frame = pl.scan_ipc(inputs["asset_path"], memory_map=False)
        if command in {"cache_get", "cache_get_or_compute", "profile", "incremental_profile", "stream_profile", "profile_import", "profile_restore"} and len(self.profiles) >= 16:
            raise MCPError("resource_exhausted")
        if command in {"generate_suite", "suite_restore"} and len(self.suites) >= 16:
            raise MCPError("resource_exhausted")
        if command.startswith("cache_"):
            from datetime import timedelta

            cache = selected(self.runtime.caches, args["cache_id"])
            key = selected(self.runtime.cache_keys, args["cache_key_id"]) if args.get("cache_key_id") else None
            ttl = timedelta(seconds=args["ttl_seconds"]) if args.get("ttl_seconds") is not None else None
            self.guard()
            if command == "cache_stats":
                # No backend paths, connection metadata, cache keys or credential diagnostics.
                native = cache.get_stats()
                allowed = {"type", "size", "max_size", "hits", "misses", "hit_ratio", "file_count", "total_size_mb", "enabled", "connected", "errors", "key_count", "default_ttl_seconds"}
                payload = {"stats": {key: value for key, value in native.items() if key in allowed}}
                return NativeServiceResult(payload, "native_cache_incomplete") if native.get("key_count") == -1 else payload
            if command == "cache_clear":
                return {"cleared": cache.clear()}
            if command == "cache_invalidate_pattern":
                return {"invalidated": cache.invalidate_by_pattern(selected(self.runtime.cache_patterns, args["pattern_id"]))}
            if command == "cache_invalidate":
                return {"invalidated": cache.invalidate(key)}
            if command == "cache_set":
                value = selected(self.profiles, args["profile_id"])
                cache.set(key, value, ttl=ttl)
                return {"stored": cache.enabled, "complete": value.complete}
            if command == "cache_get_or_compute":
                native = selected(self.runtime.profilers, args["profiler_id"])
                value = cache.get_or_compute(key, lambda: native.profile(frame), ttl=ttl)
            else:
                value = cache.get(key)
            self.guard()
            if value is None:
                return {"found": False}
            identity = self.retain(value, "profile", budget)
            result = self.profile_view(identity, value)
            payload = result.payload if isinstance(result, NativeServiceResult) else result
            payload["found"] = True
            return result
        if command == "incremental_profile":
            native = selected(self.runtime.incrementals, args["incremental_id"])
            previous = selected(self.profiles, args["previous_profile_id"]) if args.get("previous_profile_id") else None
            self.guard()
            value = native.profile(frame, previous=previous, force_refresh=args.get("force_refresh", False))
            self.guard()
            identity = self.retain(value, "profile", budget)
            result = self.profile_view(identity, value)
            payload = result.payload if isinstance(result, NativeServiceResult) else result
            payload["incremental"] = {"profiled_columns": sorted(native.last_profiled_columns),
                                      "skipped_columns": sorted(native.last_skipped_columns),
                                      "change_reasons": {key: reason.value for key, reason in native.last_change_reasons.items()}}
            return result
        if command == "stream_profile":
            native = selected(self.runtime.streaming, args["streaming_id"])
            factory = selected(self.runtime.chunk_factories, args["stream_id"])
            callback = selected(self.runtime.stream_progress, args["progress_id"]) if args.get("progress_id") else None
            self.guard()
            async with factory() as chunks:
                self.guard()
                value = native.profile_chunks(chunks, progress_callback=callback)
            self.guard()
            identity = self.retain(value, "profile", budget)
            return self.profile_view(identity, value)
        if command == "profile":
            self.guard()
            value = selected(self.runtime.profilers, args["profiler_id"]).profile(frame)
            self.guard()
            if not isinstance(value, TableProfile):
                raise MCPError("unsupported_value")
            identity = self.retain(value, "profile", budget)
            return self.profile_view(identity, value)
        if command == "profile_import":
            if not inputs or set(inputs) != {"profile_path"}:
                raise MCPError("invalid_handle")
            from truthound.mcp.tables import decode
            from truthound.profiler.schema import ProfileSerializer

            path = Path(inputs["profile_path"])
            if path.stat().st_size > budget:
                raise MCPError("resource_exhausted")
            data = decode(json.loads(path.read_text(encoding="utf-8")))
            if not isinstance(data, dict) or not {"row_count", "column_count", "columns"} <= data.keys():
                raise MCPError("invalid_handle")
            value = ProfileSerializer().deserialize(data)
            identity = self.retain(value, "profile", budget)
            return self.profile_view(identity, value)
        if command in {"profile_restore", "suite_restore"}:
            kind = command.split("_", 1)[0]
            value = self.restore(self.slot(kind, args["slot_id"]), kind, budget)
            identity = self.retain(value, kind, budget)
            return self.profile_view(identity, value) if kind == "profile" else {"suite_id": identity, "suite": value.to_dict()}
        if command == "slot_delete":
            self.guard()
            path = self.slot(args["slot_kind"], args["slot_id"])
            existed = path.exists()
            path.unlink(missing_ok=True)
            return {"deleted": existed, "slot_id": args["slot_id"]}
        if command.startswith("profile_") or command == "generate_suite":
            identity = args["profile_id"]
            value = selected(self.profiles, identity)
            if command == "profile_export":
                return value.to_dict() if value.complete else NativeServiceResult(value.to_dict(), "native_profile_incomplete")
            if command == "profile_get":
                return self.profile_view(identity, value)
            if command == "profile_release":
                del self.profiles[identity], self.sizes[identity]
                return {"released": True}
            if command == "profile_save":
                self.save(value, "profile", self.slot("profile", args["slot_id"]))
                return {"saved": True, "slot_id": args["slot_id"], "complete": value.complete}
            if command == "generate_suite":
                if not value.complete:
                    return self.profile_view(identity, value)
                self.guard()
                try:
                    suite = selected(self.runtime.generators, args["generator_id"]).generate_from_profile(
                        value, strictness=Strictness(args.get("strictness", "medium")),
                        include_categories=args.get("include_categories"),
                        exclude_categories=args.get("exclude_categories"),
                        min_confidence=RuleConfidence(args["min_confidence"]) if args.get("min_confidence") else None,
                        name=args.get("name") or None,
                    )
                except IncompleteProfileError as error:
                    return self.profile_view(identity, error.profile)
                self.guard()
                suite_id = self.retain(suite, "suite", budget)
                return {"suite_id": suite_id, "suite": suite.to_dict()}
        if command.startswith("suite_"):
            identity = args["suite_id"]
            suite = selected(self.suites, identity)
            if command == "suite_export":
                return suite.to_dict()
            if command == "suite_get":
                return {"suite_id": identity, "suite": suite.to_dict()}
            if command == "suite_release":
                del self.suites[identity], self.sizes[identity]
                return {"released": True}
            if command == "suite_save":
                self.save(suite, "suite", self.slot("suite", args["slot_id"]))
                return {"saved": True, "slot_id": args["slot_id"]}
            if command == "suite_execute":
                self.guard()
                executor = selected(self.runtime.executors, args["executor_id"])
                result = await executor.execute_async(
                    suite, frame, ExecutionContext(
                        parallel=args.get("parallel", False), fail_fast=args.get("fail_fast", False),
                        dry_run=args.get("dry_run", False),
                        timeout_seconds=args.get("timeout_seconds", executor._timeout_seconds),
                        max_workers=args.get("max_workers", executor._max_workers),
                    ),
                )
                self.guard()
                payload = {"execution": result.to_dict(), "report": result.report,
                           "error_types": [type(error).__name__ for error in result.errors]}
                return NativeServiceResult(payload, "native_suite_incomplete") if result.errors or result.skipped_rules else payload
        raise MCPError("invalid_arguments")

    async def close(self):
        self.profiles.clear()
        self.suites.clear()
        self.sizes.clear()
        # The approved context factory owns custom component/executor cleanup.


if __name__ == "__main__":
    from truthound.mcp.broker_worker import main

    main(ProfilerOwner)
