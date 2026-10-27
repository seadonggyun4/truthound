"""Lineage integrations with external systems.

Provides integration with industry-standard lineage platforms:
- OpenLineage: Standard lineage format
- Apache Atlas: Metadata governance
- DataHub: Data discovery platform
"""

from truthound.lineage.integrations.openlineage import (
    DatasetFacets,
    OpenLineageConfig,
    OpenLineageEmissionError,
    OpenLineageEmitter,
    RunEvent,
)

__all__ = [
    # OpenLineage
    "OpenLineageEmitter",
    "OpenLineageEmissionError",
    "OpenLineageConfig",
    "RunEvent",
    "DatasetFacets",
]
