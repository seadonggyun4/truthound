"""Lineage integrations with external systems.

Provides integration with industry-standard lineage platforms:
- OpenLineage: Standard lineage format
- Apache Atlas: Metadata governance
- DataHub: Data discovery platform
"""

from truthound.lineage.integrations.openlineage import (
    OpenLineageEmitter,
    OpenLineageConfig,
    RunEvent,
    DatasetFacets,
)

__all__ = [
    # OpenLineage
    "OpenLineageEmitter",
    "OpenLineageConfig",
    "RunEvent",
    "DatasetFacets",
]
