"""Referential integrity validators.

This module provides comprehensive validators for checking referential integrity
across table relationships:

- **Foreign Key Validation**: FK constraint checking with composite key support
- **Cascade Integrity**: Cascade rule validation and chain analysis
- **Orphan Detection**: Finding orphan and dangling records
- **Circular Reference**: Detecting cycles in schema and data hierarchies

Validators:
    ForeignKeyValidator: Basic and composite FK validation
    CompositeForeignKeyValidator: Advanced composite key with partial match detection
    SelfReferentialFKValidator: Self-referencing FK in hierarchies
    CascadeIntegrityValidator: Cascade rule enforcement checking
    CascadeDepthValidator: Cascade chain depth analysis
    OrphanRecordValidator: Orphan record detection
    MultiTableOrphanValidator: Cross-table orphan analysis
    DanglingReferenceValidator: Dangling parent record detection
    CircularReferenceValidator: Schema-level cycle detection
    HierarchyCircularValidator: Data-level hierarchy cycle detection
    HierarchyDepthValidator: Hierarchy depth constraint validation
"""

from truthound.validators.referential.base import (
    ForeignKeyRelation,
    TableNode,
    ReferentialValidator,
    MultiTableValidator,
)

from truthound.validators.referential.foreign_key import (
    ForeignKeyValidator,
    CompositeForeignKeyValidator,
    SelfReferentialFKValidator,
)

from truthound.validators.referential.cascade import (
    CascadeAction,
    CascadeRule,
    CascadeIntegrityValidator,
    CascadeDepthValidator,
)

from truthound.validators.referential.orphan import (
    OrphanRecordValidator,
    MultiTableOrphanValidator,
    DanglingReferenceValidator,
)

from truthound.validators.referential.circular import (
    CircularReferenceValidator,
    HierarchyCircularValidator,
    HierarchyDepthValidator,
)

__all__ = [
    # Base classes
    "ForeignKeyRelation",
    "TableNode",
    "ReferentialValidator",
    "MultiTableValidator",
    # Foreign key validators
    "ForeignKeyValidator",
    "CompositeForeignKeyValidator",
    "SelfReferentialFKValidator",
    # Cascade validators
    "CascadeAction",
    "CascadeRule",
    "CascadeIntegrityValidator",
    "CascadeDepthValidator",
    # Orphan validators
    "OrphanRecordValidator",
    "MultiTableOrphanValidator",
    "DanglingReferenceValidator",
    # Circular reference validators
    "CircularReferenceValidator",
    "HierarchyCircularValidator",
    "HierarchyDepthValidator",
]
