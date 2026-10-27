"""Native audit/telemetry dispatch. Endpoints, storage and exporter instances stay local."""

import asyncio

from truthound.mcp.administration import AuditRuntime, TelemetryRuntime
from truthound.mcp.policy import MCPError


class AdministrationOwner:
    def __init__(self, runtime, request):
        from truthound.audit.core import AuditStorage
        from truthound.observability.metrics import Metric, MetricsExporter, MetricsRegistry
        from truthound.observability.tracing.provider import TracerProvider

        group = request["native_group"]
        if group == "audit_services":
            if not isinstance(runtime, AuditRuntime) or not isinstance(
                runtime.storage, AuditStorage
            ):
                raise MCPError("invalid_binding")
        elif group == "telemetry_services":
            if (
                not isinstance(runtime, TelemetryRuntime)
                or not isinstance(runtime.registry, MetricsRegistry)
                or not isinstance(runtime.provider, TracerProvider)
                or any(not isinstance(item, MetricsExporter) for item in runtime.exporters.values())
                or any(not isinstance(item, Metric) for item in runtime.metrics.values())
            ):
                raise MCPError("invalid_binding")
        else:
            raise MCPError("invalid_binding")
        self.runtime = runtime
        self.stopped = False

    def execute_native(self, args):
        if isinstance(self.runtime, AuditRuntime):
            return self.audit(args)
        return self.telemetry(args)

    def audit(self, args):
        from datetime import datetime

        from truthound.audit.core import AuditActor, AuditEvent, AuditEventType, AuditOutcome

        command, storage = args["command"], self.runtime.storage
        filters = {
            key: args[key]
            for key in ("start_time", "end_time", "event_types")
            if args.get(key) is not None
        }
        for key in ("start_time", "end_time"):
            if key in filters:
                filters[key] = datetime.fromisoformat(filters[key])
        if "event_types" in filters:
            filters["event_types"] = [AuditEventType(value) for value in filters["event_types"]]
        if command == "query":
            filters.update(
                {key: args[key] for key in ("actor_id", "resource_id") if args.get(key) is not None}
            )
            if args.get("outcome"):
                filters["outcome"] = AuditOutcome(args["outcome"])
            return {
                "events": [
                    event.to_dict()
                    for event in storage.query(
                        **filters, limit=args.get("limit", 100), offset=args.get("offset", 0)
                    )
                ]
            }
        if command == "count":
            return {"count": storage.count(**filters)}
        if command == "read":
            event = storage.read(args["event_id"])
            return {"event": event.to_dict() if event else None}
        if command == "record":
            event = AuditEvent(
                event_type=AuditEventType.CUSTOM,
                action=args["action"],
                actor=AuditActor(
                    id="mcp-approved-operator", type="service", name="MCP approved operator"
                ),
                outcome=AuditOutcome(args.get("outcome") or "unknown"),
                message=args.get("message", ""),
                data=args.get("data", {}),
            )
            event.checksum = event.compute_checksum()
            storage.write(event)
            return {"event": event.to_dict()}
        if command == "delete_before":
            return {"removed": storage.delete_before(datetime.fromisoformat(args["before"]))}
        if command == "flush":
            storage.flush()
            return {"flushed": True}
        raise MCPError("invalid_arguments")

    def telemetry(self, args):
        from truthound.observability.metrics import Counter, Gauge, Histogram, Summary

        runtime, command = self.runtime, args["command"]
        if self.stopped and command != "shutdown":
            raise MCPError("invalid_handle")
        if command == "collect":
            return {"metrics": runtime.registry.collect_all()}
        if command == "export":
            exporter = runtime.exporters.get(args["exporter_id"])
            if exporter is None:
                raise MCPError("invalid_binding")
            return {"output": exporter.export(runtime.registry)}
        if command == "metric":
            metric = runtime.metrics.get(args["metric_id"])
            update = args["update"]
            allowed = (
                update == "inc"
                and isinstance(metric, (Counter, Gauge))
                or update == "set"
                and isinstance(metric, Gauge)
                or update == "observe"
                and isinstance(metric, (Histogram, Summary))
            )
            if not allowed:
                raise MCPError("invalid_arguments")
            # Selection is closed above; no wire method names reach getattr.
            method = {
                "inc": lambda: metric.inc(args["value"], **args.get("labels", {})),
                "set": lambda: metric.set(args["value"], **args.get("labels", {})),
                "observe": lambda: metric.observe(args["value"], **args.get("labels", {})),
            }[update]
            method()
            return {"metrics": runtime.registry.collect_all()}
        if command == "span":
            tracer = runtime.provider.get_tracer("truthound.mcp.approved")
            with tracer.start_as_current_span(
                args["span_name"], attributes=args.get("attributes", {})
            ) as span:
                context = span.context
            return {"trace_id": context.trace_id, "span_id": context.span_id}
        if command == "flush":
            return {"flushed": runtime.provider.force_flush(args.get("timeout_ms", 1000))}
        if command == "shutdown":
            success = runtime.provider.shutdown()
            self.stopped = True
            return {"shutdown": success}
        raise MCPError("invalid_arguments")

    async def execute(self, args, budget):
        return await asyncio.to_thread(self.execute_native, args)

    async def close(self):
        if isinstance(self.runtime, AuditRuntime):
            self.runtime.storage.close()
        else:
            self.runtime.provider.shutdown()


if __name__ == "__main__":
    from truthound.mcp.broker_worker import main

    main(AdministrationOwner)
