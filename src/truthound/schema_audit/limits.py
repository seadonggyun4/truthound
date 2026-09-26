"""Execution budgets, distinct from the larger immutable model representation limit."""

MAX_AUDIT_OBJECTS = 10000
MAX_CATALOG_TABLES = 500
# Result checks are more numerous than input objects; input array limits stay unchanged.
MAX_AUDIT_CHECKS = 40 * MAX_AUDIT_OBJECTS
