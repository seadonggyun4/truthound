"""OpenLineage integration.

Provides OpenLineage standard event emission for
cross-platform lineage compatibility.

OpenLineage is an open framework for data lineage collection
and analysis: https://openlineage.io
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import uuid
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

if TYPE_CHECKING:
    from truthound.lineage.base import LineageGraph


logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """OpenLineage event types."""

    START = "START"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    ABORT = "ABORT"
    FAIL = "FAIL"
    OTHER = "OTHER"


@dataclass
class OpenLineageConfig:
    """Configuration for OpenLineage emitter.

    Attributes:
        endpoint: OpenLineage API endpoint
        api_key: API key for authentication
        namespace: Default namespace
        producer: Producer identifier
        timeout_seconds: Request timeout
    """

    endpoint: str = "http://localhost:5000/api/v1/lineage"
    api_key: str | None = field(default=None, repr=False)
    namespace: str = "truthound"
    producer: str = "truthound"
    timeout_seconds: float = 30
    max_event_bytes: int = 1024 * 1024

    def __post_init__(self):
        if not isinstance(self.endpoint, str):
            raise ValueError("Endpoint must be a URL string")
        endpoint = urlsplit(self.endpoint)
        if (
            endpoint.scheme not in {"http", "https"}
            or not endpoint.hostname
            or endpoint.username
            or endpoint.password
            or endpoint.fragment
        ):
            raise ValueError("OpenLineage requires an explicit HTTP(S) endpoint without userinfo")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("OpenLineage timeout must be finite and positive")
        if type(self.max_event_bytes) is not int or self.max_event_bytes < 256:
            raise ValueError("OpenLineage event budget must be at least 256 bytes")
        if any(
            not isinstance(value, str) or not value for value in (self.namespace, self.producer)
        ):
            raise ValueError("OpenLineage namespace and producer must be nonempty strings")
        if self.api_key is not None and (
            not isinstance(self.api_key, str) or any(c in self.api_key for c in ("\r", "\n", "\0"))
        ):
            raise ValueError("Invalid OpenLineage authorization value")


@dataclass
class DatasetFacets:
    """Dataset facets for OpenLineage.

    Attributes:
        schema_fields: List of schema fields
        data_source: Data source information
        lifecycle_state: Dataset lifecycle state
        ownership: Dataset ownership
        quality_metrics: Data quality metrics
    """

    schema_fields: list[dict[str, Any]] = field(default_factory=list)
    data_source: dict[str, str] | None = None
    lifecycle_state: str | None = None
    ownership: dict[str, Any] | None = None
    quality_metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenLineage facets format."""
        facets = {}

        if self.schema_fields:
            facets["schema"] = {
                "_producer": "https://github.com/truthound/truthound",
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/SchemaDatasetFacet.json",
                "fields": self.schema_fields,
            }

        if self.data_source:
            facets["dataSource"] = {
                "_producer": "https://github.com/truthound/truthound",
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DatasourceDatasetFacet.json",
                **self.data_source,
            }

        if self.lifecycle_state:
            facets["lifecycleStateChange"] = {
                "_producer": "https://github.com/truthound/truthound",
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/LifecycleStateChangeDatasetFacet.json",
                "lifecycleStateChange": self.lifecycle_state,
            }

        if self.ownership:
            facets["ownership"] = {
                "_producer": "https://github.com/truthound/truthound",
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/OwnershipDatasetFacet.json",
                **self.ownership,
            }

        if self.quality_metrics:
            facets["dataQualityMetrics"] = {
                "_producer": "https://github.com/truthound/truthound",
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DataQualityMetricsInputDatasetFacet.json",
                **self.quality_metrics,
            }

        return facets


@dataclass
class RunEvent:
    """OpenLineage run event.

    Attributes:
        run_id: Unique run identifier
        job_name: Job name
        job_namespace: Job namespace
        event_type: Event type
        inputs: Input datasets
        outputs: Output datasets
        parent: Parent run (optional)
        facets: Run facets
    """

    run_id: str
    job_name: str
    job_namespace: str
    event_type: EventType = EventType.START
    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    parent: dict[str, Any] | None = None
    facets: dict[str, Any] = field(default_factory=dict)


class OpenLineageEmissionError(RuntimeError):
    """A delivery attempt was not acknowledged; do not replay it automatically."""

    def __init__(self, run_id, event_type):
        super().__init__("OpenLineage delivery outcome is unknown")
        self.run_id = run_id
        self.event_type = event_type
        self.outcome_unknown = True


class OpenLineageEmitter:
    """OpenLineage event emitter.

    Emits lineage events in OpenLineage standard format to
    compatible lineage platforms.

    Example:
        >>> emitter = OpenLineageEmitter(OpenLineageConfig(
        ...     endpoint="http://localhost:5000/api/v1/lineage",
        ... ))
        >>>
        >>> # Start a job
        >>> run = emitter.start_run("my-job")
        >>>
        >>> # Complete the job
        >>> emitter.emit_complete(run, outputs=[output_dataset])

    OpenLineage Spec: https://openlineage.io/spec
    """

    def __init__(self, config: OpenLineageConfig | None = None):
        self._config = deepcopy(config) if config is not None else OpenLineageConfig()
        self._config.__post_init__()
        self._active_runs = {}
        self._receipts = {}
        self._uncertain = {}
        self._lock = threading.RLock()
        self._pending_lock = threading.Lock()
        self._pending = set()
        self._async_loop = None
        self._closed = False

    @staticmethod
    def _require_sync():
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError("Use the awaited OpenLineage async method inside an event loop")

    def _require_open(self):
        if self._closed:
            raise RuntimeError("OpenLineage emitter is closed")

    @property
    def active_runs(self):
        with self._lock:
            return deepcopy(self._active_runs)

    @property
    def unknown_runs(self):
        with self._lock:
            return {
                key: {"event_type": value.event_type.value, "job_name": value.job_name}
                for key, value in self._uncertain.items()
            }

    def get_run(self, run_id):
        with self._lock:
            return self._receipts[run_id]

    def start_run(self, job_name, inputs=None, parent_run_id=None, facets=None):
        self._require_sync()
        with self._lock:
            self._require_open()
            if not isinstance(job_name, str) or not job_name:
                raise ValueError("Job name must be nonempty")
            parent = None
            if parent_run_id is not None:
                if parent_run_id in self._uncertain or parent_run_id not in self._active_runs:
                    raise ValueError("Parent run must be active and acknowledged by this emitter")
                owned = self._active_runs[parent_run_id]
                parent = {
                    "run": {"runId": owned.run_id},
                    "job": {"namespace": owned.job_namespace, "name": owned.job_name},
                }
            run = RunEvent(
                str(uuid.uuid4()),
                job_name,
                self._config.namespace,
                inputs=deepcopy(inputs) if inputs is not None else [],
                parent=parent,
                facets=deepcopy(facets) if facets is not None else {},
            )
            self._receipts[run.run_id] = run
            try:
                self._publish(run, deepcopy(run))
            except BaseException:
                if run.run_id not in self._uncertain:
                    self._receipts.pop(run.run_id, None)
                raise
            return run

    def _transition(self, run, event_type, outputs=None, facets=None):
        self._require_sync()
        with self._lock:
            self._require_open()
            if run.run_id in self._uncertain:
                raise RuntimeError("Resolve the unknown delivery before another event")
            if self._receipts.get(run.run_id) is not run or run.run_id not in self._active_runs:
                raise ValueError("Run is not active and owned by this emitter")
            candidate = deepcopy(self._active_runs[run.run_id])
            candidate.event_type = event_type
            if outputs is not None:
                candidate.outputs = deepcopy(outputs)
            if facets is not None:
                candidate.facets.update(deepcopy(facets))
            self._publish(run, candidate)

    def emit_running(self, run, facets=None):
        self._transition(run, EventType.RUNNING, facets=facets)

    def emit_complete(self, run, outputs=None, facets=None):
        self._transition(run, EventType.COMPLETE, outputs=outputs, facets=facets)

    def emit_fail(self, run, error, facets=None):
        values = deepcopy(facets) if facets is not None else {}
        values["errorMessage"] = {
            "_producer": self._producer(),
            "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/ErrorMessageRunFacet.json",
            "message": str(error),
            "programmingLanguage": "python",
        }
        self._transition(run, EventType.FAIL, facets=values)

    def emit_abort(self, run, reason=None):
        facets = {"abortInfo": {"reason": reason}} if reason is not None else None
        self._transition(run, EventType.ABORT, facets=facets)

    def _commit(self, receipt, candidate):
        receipt.__dict__.update(deepcopy(candidate.__dict__))
        if candidate.event_type in {EventType.COMPLETE, EventType.FAIL, EventType.ABORT}:
            self._active_runs.pop(candidate.run_id, None)
            self._receipts.pop(candidate.run_id, None)
        else:
            self._active_runs[candidate.run_id] = deepcopy(candidate)
        self._uncertain.pop(candidate.run_id, None)

    def _encode_event(self, candidate):
        from truthound.stores._filesystem_io import validate_values

        event = self._build_event(candidate)
        validate_values(event)
        payload = json.dumps(event, allow_nan=False).encode("utf-8")
        if len(payload) > self._config.max_event_bytes:
            raise ValueError("OpenLineage event exceeds configured byte budget")
        return payload

    def _publish(self, receipt, candidate):
        payload = self._encode_event(candidate)
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = "Bearer " + self._config.api_key
        request = Request(self._config.endpoint, data=payload, headers=headers, method="POST")

        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None

        self._uncertain[candidate.run_id] = deepcopy(candidate)
        try:
            opener = build_opener(ProxyHandler({}), NoRedirect())
            with opener.open(request, timeout=self._config.timeout_seconds) as response:
                if not 200 <= response.status < 300:
                    raise OpenLineageEmissionError(candidate.run_id, candidate.event_type.value)
        except HTTPError as error:
            error.close()
            raise OpenLineageEmissionError(candidate.run_id, candidate.event_type.value) from None
        except (URLError, OSError):
            raise OpenLineageEmissionError(candidate.run_id, candidate.event_type.value) from None
        self._commit(receipt, candidate)

    def resolve_unknown(self, run_id, *, delivered):
        """Record an operator's external delivery determination; never resend."""
        self._require_sync()
        if type(delivered) is not bool:
            raise ValueError("delivered must be boolean")
        with self._lock:
            candidate = self._uncertain[run_id]
            if delivered:
                self._commit(self._receipts[run_id], candidate)
            else:
                self._uncertain.pop(run_id)
                if run_id not in self._active_runs:
                    self._receipts.pop(run_id, None)

    def _producer(self):
        selected = self._config.producer
        return (
            selected
            if urlsplit(selected).scheme
            else "https://github.com/truthound/" + quote(selected, safe="")
        )

    def _facets(self, facets, kind="RunFacet"):
        result = deepcopy(facets)
        for name, value in result.items():
            if not isinstance(value, dict):
                value = {"value": value}
                result[name] = value
            value.setdefault("_producer", self._producer())
            value.setdefault(
                "_schemaURL", "https://openlineage.io/spec/1-0-5/OpenLineage.json#/$defs/" + kind
            )
        return result

    def _build_event(self, run):
        facets = self._facets(run.facets)
        if run.parent:
            facets["parent"] = {
                "_producer": self._producer(),
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/ParentRunFacet.json",
                **deepcopy(run.parent),
            }
        return {
            "eventType": run.event_type.value,
            "eventTime": datetime.now(UTC).isoformat(),
            "producer": self._producer(),
            "schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json#/$defs/RunEvent",
            "run": {"runId": run.run_id, "facets": facets},
            "job": {"namespace": run.job_namespace, "name": run.job_name},
            "inputs": deepcopy(run.inputs),
            "outputs": deepcopy(run.outputs),
        }

    async def _async_call(self, method, *args, **kwargs):
        loop = asyncio.get_running_loop()
        with self._pending_lock:
            if self._async_loop is not None and self._async_loop is not loop:
                raise RuntimeError("Async emitter belongs to another event loop")
            self._async_loop = loop
        task = asyncio.create_task(asyncio.to_thread(method, *args, **kwargs))
        with self._pending_lock:
            self._pending.add(task)

        def done(completed):
            with self._pending_lock:
                self._pending.discard(completed)
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(done)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Delivery uncertainty remains recorded and queryable.
            with suppress(Exception):
                await asyncio.shield(task)
            raise

    async def start_run_async(self, *args, **kwargs):
        return await self._async_call(self.start_run, *args, **kwargs)

    async def emit_running_async(self, *args, **kwargs):
        return await self._async_call(self.emit_running, *args, **kwargs)

    async def emit_complete_async(self, *args, **kwargs):
        return await self._async_call(self.emit_complete, *args, **kwargs)

    async def emit_fail_async(self, *args, **kwargs):
        return await self._async_call(self.emit_fail, *args, **kwargs)

    async def emit_abort_async(self, *args, **kwargs):
        return await self._async_call(self.emit_abort, *args, **kwargs)

    async def emit_from_graph_async(self, *args, **kwargs):
        return await self._async_call(self.emit_from_graph, *args, **kwargs)

    async def resolve_unknown_async(self, *args, **kwargs):
        return await self._async_call(self.resolve_unknown, *args, **kwargs)

    def close(self):
        self._require_sync()
        with self._lock:
            self._closed = True

    async def aclose(self):
        await self._async_call(self.close)
        with self._pending_lock:
            pending = tuple(self._pending)
        await asyncio.gather(*pending, return_exceptions=True)

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *args):
        self.close()

    async def __aenter__(self):
        self._require_open()
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    # -------------------------------------------------------------------------
    # Helper methods for building datasets
    # -------------------------------------------------------------------------

    def build_dataset(
        self,
        name: str,
        namespace: str | None = None,
        facets: DatasetFacets | None = None,
    ) -> dict[str, Any]:
        """Build an OpenLineage dataset.

        Args:
            name: Dataset name
            namespace: Dataset namespace
            facets: Dataset facets

        Returns:
            Dataset dict
        """
        dataset: dict[str, Any] = {
            "namespace": namespace or self._config.namespace,
            "name": name,
        }

        if facets:
            dataset["facets"] = self._facets(facets.to_dict(), "DatasetFacet")
            metrics = dataset["facets"].pop("dataQualityMetrics", None)
            if metrics is not None:
                dataset["inputFacets"] = {"dataQualityMetrics": metrics}

        return dataset

    def build_input_dataset(
        self,
        name: str,
        namespace: str | None = None,
        schema: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Build an input dataset with optional schema.

        Args:
            name: Dataset name
            namespace: Dataset namespace
            schema: Schema as list of {name, type} dicts

        Returns:
            Dataset dict
        """
        facets = DatasetFacets()

        if schema:
            facets.schema_fields = [
                {"name": f["name"], "type": f.get("type", "string")} for f in schema
            ]

        return self.build_dataset(name, namespace, facets)

    def build_output_dataset(
        self,
        name: str,
        namespace: str | None = None,
        schema: list[dict[str, str]] | None = None,
        row_count: int | None = None,
    ) -> dict[str, Any]:
        """Build an output dataset with optional metrics.

        Args:
            name: Dataset name
            namespace: Dataset namespace
            schema: Schema as list of {name, type} dicts
            row_count: Number of rows written

        Returns:
            Dataset dict
        """
        facets = DatasetFacets()

        if schema:
            facets.schema_fields = [
                {"name": f["name"], "type": f.get("type", "string")} for f in schema
            ]

        dataset = self.build_dataset(name, namespace, facets)
        if row_count is not None:
            if type(row_count) is not int or row_count < 0:
                raise ValueError("Output row count must be a nonnegative integer")
            dataset["outputFacets"] = {
                "outputStatistics": {
                    "_producer": self._producer(),
                    "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/OutputStatisticsOutputDatasetFacet.json",
                    "rowCount": row_count,
                }
            }
        return dataset

    # -------------------------------------------------------------------------
    # Convert from Truthound lineage
    # -------------------------------------------------------------------------

    def emit_from_graph(
        self,
        graph: LineageGraph,
        job_name: str = "truthound-lineage",
    ) -> list[RunEvent]:
        """Emit OpenLineage events from a Truthound lineage graph.

        Args:
            graph: Lineage graph
            job_name: Job name for all events

        Returns:
            List of emitted run events
        """

        from truthound.lineage.base import LineageGraph

        self._require_sync()
        if not isinstance(job_name, str) or not job_name:
            raise ValueError("Job name must be nonempty")
        # Capture one coherent graph before delivery. Reconstruction validates all
        # edge references instead of silently dropping missing provenance.
        snapshot = LineageGraph.from_dict(deepcopy(graph.to_dict()))
        transformations: dict[str, list[str]] = {}
        for edge in snapshot.edges:
            transformations.setdefault(edge.target, []).append(edge.source)

        planned = []
        with self._lock:
            self._require_open()
            for target_id, source_ids in transformations.items():
                target = snapshot.get_node(target_id)
                sources = [snapshot.get_node(source_id) for source_id in source_ids]
                if any(
                    not isinstance(node.name, str) or not node.name for node in [target, *sources]
                ):
                    raise ValueError("Graph dataset names must be nonempty strings")
                run = RunEvent(
                    str(uuid.uuid4()),
                    f"{job_name}:{target.name}",
                    self._config.namespace,
                    inputs=[self.build_dataset(source.name) for source in sources],
                )
                terminal = deepcopy(run)
                terminal.event_type = EventType.COMPLETE
                terminal.outputs = [self.build_dataset(target.name)]
                self._encode_event(run)
                self._encode_event(terminal)
                planned.append((run, terminal))

            runs = []
            for run, terminal in planned:
                self._receipts[run.run_id] = run
                try:
                    self._publish(run, deepcopy(run))
                    self._publish(run, terminal)
                except BaseException:
                    if run.run_id not in self._uncertain and run.run_id not in self._active_runs:
                        self._receipts.pop(run.run_id, None)
                    raise
                runs.append(run)
            return runs
