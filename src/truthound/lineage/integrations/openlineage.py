"""OpenLineage integration.

Provides OpenLineage standard event emission for
cross-platform lineage compatibility.

OpenLineage is an open framework for data lineage collection
and analysis: https://openlineage.io
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, TYPE_CHECKING
import json
import uuid
import logging

if TYPE_CHECKING:
    from truthound.lineage.base import LineageGraph, LineageNode, LineageEdge


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
    api_key: str | None = None
    namespace: str = "truthound"
    producer: str = "truthound"
    timeout_seconds: int = 30


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
                "_producer": "truthound",
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/SchemaDatasetFacet.json",
                "fields": self.schema_fields,
            }

        if self.data_source:
            facets["dataSource"] = {
                "_producer": "truthound",
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DatasourceDatasetFacet.json",
                **self.data_source,
            }

        if self.lifecycle_state:
            facets["lifecycleStateChange"] = {
                "_producer": "truthound",
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/LifecycleStateChangeDatasetFacet.json",
                "lifecycleStateChange": self.lifecycle_state,
            }

        if self.ownership:
            facets["ownership"] = {
                "_producer": "truthound",
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/OwnershipDatasetFacet.json",
                **self.ownership,
            }

        if self.quality_metrics:
            facets["dataQualityMetrics"] = {
                "_producer": "truthound",
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
        self._config = config or OpenLineageConfig()
        self._active_runs: dict[str, RunEvent] = {}

    def start_run(
        self,
        job_name: str,
        inputs: list[dict[str, Any]] | None = None,
        parent_run_id: str | None = None,
        facets: dict[str, Any] | None = None,
    ) -> RunEvent:
        """Start a new run and emit START event.

        Args:
            job_name: Job name
            inputs: Input datasets
            parent_run_id: Parent run ID for nested jobs
            facets: Additional run facets

        Returns:
            RunEvent for the started run
        """
        run_id = str(uuid.uuid4())

        parent = None
        if parent_run_id:
            parent_run = self._active_runs.get(parent_run_id)
            if parent_run:
                parent = {
                    "run": {"runId": parent_run_id},
                    "job": {
                        "namespace": parent_run.job_namespace,
                        "name": parent_run.job_name,
                    },
                }

        run = RunEvent(
            run_id=run_id,
            job_name=job_name,
            job_namespace=self._config.namespace,
            event_type=EventType.START,
            inputs=inputs or [],
            parent=parent,
            facets=facets or {},
        )

        self._active_runs[run_id] = run
        self._emit(run)

        return run

    def emit_running(
        self,
        run: RunEvent,
        facets: dict[str, Any] | None = None,
    ) -> None:
        """Emit RUNNING event for progress updates.

        Args:
            run: Run event
            facets: Additional facets
        """
        run.event_type = EventType.RUNNING
        if facets:
            run.facets.update(facets)
        self._emit(run)

    def emit_complete(
        self,
        run: RunEvent,
        outputs: list[dict[str, Any]] | None = None,
        facets: dict[str, Any] | None = None,
    ) -> None:
        """Emit COMPLETE event.

        Args:
            run: Run event
            outputs: Output datasets
            facets: Additional facets
        """
        run.event_type = EventType.COMPLETE
        if outputs:
            run.outputs = outputs
        if facets:
            run.facets.update(facets)

        self._emit(run)
        self._active_runs.pop(run.run_id, None)

    def emit_fail(
        self,
        run: RunEvent,
        error: Exception | str,
        facets: dict[str, Any] | None = None,
    ) -> None:
        """Emit FAIL event.

        Args:
            run: Run event
            error: Error that caused failure
            facets: Additional facets
        """
        run.event_type = EventType.FAIL

        # Add error facet
        error_message = str(error)
        run.facets["errorMessage"] = {
            "_producer": self._config.producer,
            "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/ErrorMessageRunFacet.json",
            "message": error_message,
            "programmingLanguage": "python",
        }

        if facets:
            run.facets.update(facets)

        self._emit(run)
        self._active_runs.pop(run.run_id, None)

    def emit_abort(
        self,
        run: RunEvent,
        reason: str | None = None,
    ) -> None:
        """Emit ABORT event.

        Args:
            run: Run event
            reason: Abort reason
        """
        run.event_type = EventType.ABORT

        if reason:
            run.facets["abortInfo"] = {
                "_producer": self._config.producer,
                "reason": reason,
            }

        self._emit(run)
        self._active_runs.pop(run.run_id, None)

    def _emit(self, run: RunEvent) -> None:
        """Emit event to OpenLineage endpoint.

        Args:
            run: Run event to emit
        """
        event = self._build_event(run)

        try:
            import aiohttp
            import asyncio

            async def send():
                headers = {"Content-Type": "application/json"}
                if self._config.api_key:
                    headers["Authorization"] = f"Bearer {self._config.api_key}"

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self._config.endpoint,
                        json=event,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self._config.timeout_seconds),
                    ) as response:
                        if response.status >= 400:
                            logger.warning(f"OpenLineage emit failed: {response.status}")

            # Run async if in async context, otherwise use thread
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(send())
            except RuntimeError:
                # No running event loop
                asyncio.run(send())

        except ImportError:
            # Fallback to synchronous requests
            try:
                import requests
                headers = {"Content-Type": "application/json"}
                if self._config.api_key:
                    headers["Authorization"] = f"Bearer {self._config.api_key}"

                response = requests.post(
                    self._config.endpoint,
                    json=event,
                    headers=headers,
                    timeout=self._config.timeout_seconds,
                )

                if response.status_code >= 400:
                    logger.warning(f"OpenLineage emit failed: {response.status_code}")

            except ImportError:
                logger.warning("Neither aiohttp nor requests available for OpenLineage emit")
            except Exception as e:
                logger.warning(f"OpenLineage emit error: {e}")

        except Exception as e:
            logger.warning(f"OpenLineage emit error: {e}")

    def _build_event(self, run: RunEvent) -> dict[str, Any]:
        """Build OpenLineage event structure.

        Args:
            run: Run event

        Returns:
            OpenLineage event dict
        """
        event = {
            "eventType": run.event_type.value,
            "eventTime": datetime.now(timezone.utc).isoformat(),
            "producer": f"https://github.com/truthound/{self._config.producer}",
            "schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json#/$defs/RunEvent",
            "run": {
                "runId": run.run_id,
                "facets": run.facets,
            },
            "job": {
                "namespace": run.job_namespace,
                "name": run.job_name,
            },
            "inputs": run.inputs,
            "outputs": run.outputs,
        }

        if run.parent:
            event["run"]["facets"]["parent"] = run.parent

        return event

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
            dataset["facets"] = facets.to_dict()

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
                {"name": f["name"], "type": f.get("type", "string")}
                for f in schema
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
                {"name": f["name"], "type": f.get("type", "string")}
                for f in schema
            ]

        if row_count is not None:
            facets.quality_metrics = {
                "rowCount": row_count,
            }

        return self.build_dataset(name, namespace, facets)

    # -------------------------------------------------------------------------
    # Convert from Truthound lineage
    # -------------------------------------------------------------------------

    def emit_from_graph(
        self,
        graph: "LineageGraph",
        job_name: str = "truthound-lineage",
    ) -> list[RunEvent]:
        """Emit OpenLineage events from a Truthound lineage graph.

        Args:
            graph: Lineage graph
            job_name: Job name for all events

        Returns:
            List of emitted run events
        """
        from truthound.lineage.base import NodeType, EdgeType

        runs = []

        # Group edges by target to find "jobs" (transformations)
        transformations: dict[str, list[str]] = {}

        for edge in graph.edges:
            # Handle both source/source_id and target/target_id attribute names
            target = getattr(edge, 'target_id', None) or getattr(edge, 'target', None)
            source = getattr(edge, 'source_id', None) or getattr(edge, 'source', None)
            if target not in transformations:
                transformations[target] = []
            transformations[target].append(source)

        # Emit events for each transformation
        for target_id, source_ids in transformations.items():
            # Use get_node for graph API compatibility
            try:
                target_node = graph.get_node(target_id) if graph.has_node(target_id) else None
            except Exception:
                target_node = None
            if not target_node:
                continue

            # Build inputs
            inputs = []
            for source_id in source_ids:
                try:
                    source_node = graph.get_node(source_id) if graph.has_node(source_id) else None
                except Exception:
                    source_node = None
                if source_node:
                    inputs.append(self.build_dataset(source_node.name))

            # Build output
            output = self.build_dataset(target_node.name)

            # Start and complete run
            run = self.start_run(
                job_name=f"{job_name}:{target_node.name}",
                inputs=inputs,
            )
            self.emit_complete(run, outputs=[output])
            runs.append(run)

        return runs
